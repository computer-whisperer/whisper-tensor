//! Bytewise A/B test harness: compare `X86JitSpan` output against the
//! `pool_eval` reference for the same NanoGraph and inputs.
//!
//! Pool_eval is the slow-but-correct reference per the dtype contract
//! (see `docs/dtype_contract.md`); a conformant runtime must produce
//! bit-identical bytes.
//!
//! Under the memory-placement design every compiled span addresses
//! atoms through a `buffer_ptrs` array keyed by `buffer_id`. The
//! harness runs the placer against a trivial single-span phase,
//! allocates one `Vec<u8>` per placer buffer, copies the caller's
//! input bytes into the input buffers, dispatches the span with the
//! assembled pointer array, and then reads each output buffer back.

use crate::compiler::attempts::v14::executor::{CompiledSpanFn, PoolEvalSpan};
use crate::compiler::attempts::v14::placer::{AtomPlacementMap, BufferKind, run_placer};
use crate::compiler::attempts::v14::types::{Phase, Span};
use crate::compiler::attempts::v14::x86_jit::X86JitSpan;
use crate::nano_graph::pattern::{AtomId, AtomRange, NanoGraph};
use crate::numeric_dtype::NumericDType;
use crate::pool::SystemPool;

/// Build a trivial placement: one phase, one span containing the
/// whole graph with the declared input tensors and output ranges.
fn build_placement(
    graph: &NanoGraph<'static, SystemPool>,
    outputs: &[AtomRange],
) -> AtomPlacementMap {
    let span_inputs: Vec<AtomRange> = graph
        .input_tensors()
        .iter()
        .map(|it| AtomRange {
            base: it.base_id,
            count: it.count,
            dtype: it.dtype,
        })
        .collect();
    let phases = vec![Phase {
        spans: vec![Span {
            graph: graph.clone(),
            inputs: span_inputs,
            outputs: outputs.to_vec(),
        }],
    }];
    run_placer(graph, &phases, outputs, &std::collections::HashMap::new())
        .expect("placer failed in test harness")
}

/// Populate the literal buffer slice from graph literals. Mirrors
/// `populate_literal_buffer` in the executor — walks every
/// `Literal`/`LiteralSpan` group and writes its bytes at the placer's
/// `literal_sources[base]` offset. **Every** literal group has an
/// entry, including groups whose primary slot is an output buffer —
/// the JIT reads from this source when emitting a copy for
/// output-overlapping literals, so the bytes must live here too.
fn populate_literals(
    graph: &NanoGraph<'static, SystemPool>,
    placement: &AtomPlacementMap,
    buffer_storage: &mut [Vec<u8>],
) {
    use crate::compiler::attempts::v14::placer::LITERAL_BUFFER;
    use crate::nano_graph::ops::ScalarOp;

    let lit_slot = LITERAL_BUFFER.0 as usize;
    if lit_slot >= buffer_storage.len() {
        return;
    }
    let buffer = &mut buffer_storage[lit_slot];
    for group in graph.groups() {
        let src_off = match placement.literal_sources.get(&group.base_id) {
            Some(&o) => o,
            None => continue,
        };
        let elem_bytes = group.output_dtype.bytes_per_element();
        match &group.op {
            ScalarOp::Literal(scalar) => {
                let stored = scalar.cast_to(group.output_dtype);
                let bytes = stored.as_le_bytes();
                for i in 0..group.count {
                    let off = (src_off + i * elem_bytes as u64) as usize;
                    if off + elem_bytes <= buffer.len() {
                        buffer[off..off + bytes.len()].copy_from_slice(bytes);
                    }
                }
            }
            ScalarOp::LiteralSpan(tensor) => {
                for i in 0..group.count {
                    let scalar = tensor.read_element(i as usize);
                    let stored = scalar.cast_to(group.output_dtype);
                    let bytes = stored.as_le_bytes();
                    let off = (src_off + i * elem_bytes as u64) as usize;
                    if off + elem_bytes <= buffer.len() {
                        buffer[off..off + bytes.len()].copy_from_slice(bytes);
                    }
                }
            }
            _ => {}
        }
    }
}

/// A/B harness: compile via `X86JitSpan` AND `PoolEvalSpan`, run on
/// the same inputs, assert byte-identical outputs, and return the
/// (now-agreed) output bytes.
///
/// **Panics** if the two backends produce different bytes.
pub fn ab_test_bytes(
    graph: &NanoGraph<'static, SystemPool>,
    inputs: &[(AtomId, NumericDType, Vec<u8>)],
    outputs: &[AtomRange],
) -> Vec<Vec<u8>> {
    let placement = build_placement(graph, outputs);

    // PoolEvalSpan needs the declared input ranges up front; the
    // graph's `input_tensors()` list is exactly that.
    let pool_input_ranges: Vec<AtomRange> = graph
        .input_tensors()
        .iter()
        .map(|it| AtomRange {
            base: it.base_id,
            count: it.count,
            dtype: it.dtype,
        })
        .collect();

    let pool_span = PoolEvalSpan::new(
        graph.clone(),
        pool_input_ranges,
        outputs.to_vec(),
        &placement,
        &crate::range_map::RangeMap::new(),
    );
    let pool_outs = run_harness(&pool_span, graph, &placement, inputs, outputs);

    let x86_span = X86JitSpan::compile(
        graph,
        outputs,
        &placement,
        &std::collections::HashMap::new(),
        &crate::range_map::RangeMap::new(),
    )
    .expect("x86_jit compile (ab_test_bytes is for spans inside the support envelope)");
    let x86_outs = run_harness(&x86_span, graph, &placement, inputs, outputs);

    for (i, (a, b)) in pool_outs.iter().zip(x86_outs.iter()).enumerate() {
        assert_eq!(
            a, b,
            "output {i}: pool_eval vs x86_jit byte mismatch \
             (pool={a:?}, x86={b:?})"
        );
    }
    pool_outs
}

/// Wrapper that populates the literal buffer from graph literals
/// before dispatching the span.
fn run_harness(
    span: &dyn CompiledSpanFn,
    graph: &NanoGraph<'static, SystemPool>,
    placement: &AtomPlacementMap,
    inputs: &[(AtomId, NumericDType, Vec<u8>)],
    outputs: &[AtomRange],
) -> Vec<Vec<u8>> {
    // We need to populate the literal buffer before building the
    // pointer array. Since `run_with_buffer_ptrs` owns the storage,
    // inline an equivalent here that does the literal population.
    let scratch_id = placement.scratch_buffer_id;
    let ptr_array_len = (scratch_id as usize) + 1;

    let mut buffer_storage: Vec<Vec<u8>> = vec![Vec::new(); ptr_array_len];
    for info in &placement.buffers {
        let id = info.id.0 as usize;
        buffer_storage[id] = vec![0u8; info.size_bytes as usize + 16];
    }
    populate_literals(graph, placement, &mut buffer_storage);

    let mut scratch_buf = vec![0u8; span.scratch_bytes() + 16];

    for (base, _dtype, bytes) in inputs {
        let (buf_id, off) = placement
            .byte_offset_of(*base)
            .unwrap_or_else(|| panic!("input base {} not in placement map", base.0));
        let dst = &mut buffer_storage[buf_id.0 as usize];
        let start = off as usize;
        let end = start + bytes.len();
        assert!(
            end <= dst.len(),
            "input bytes overflow buffer (buf_id={}, off={}, len={}, cap={})",
            buf_id.0,
            start,
            bytes.len(),
            dst.len(),
        );
        dst[start..end].copy_from_slice(bytes);
    }

    let mut buffer_ptrs: Vec<*mut u8> = vec![std::ptr::null_mut(); ptr_array_len];
    for info in &placement.buffers {
        let id = info.id.0 as usize;
        buffer_ptrs[id] = buffer_storage[id].as_mut_ptr();
    }
    buffer_ptrs[scratch_id as usize] = scratch_buf.as_mut_ptr();

    span.execute(&buffer_ptrs, &std::collections::HashMap::new());
    let _ = &scratch_buf;

    outputs
        .iter()
        .map(|r| {
            let (buf_id, off) = placement
                .byte_offset_of(r.base)
                .unwrap_or_else(|| panic!("output base {} not in placement map", r.base.0));
            let len = r.count as usize * r.dtype.bytes_per_element();
            let src = &buffer_storage[buf_id.0 as usize];
            let start = off as usize;
            src[start..start + len].to_vec()
        })
        .collect()
}
