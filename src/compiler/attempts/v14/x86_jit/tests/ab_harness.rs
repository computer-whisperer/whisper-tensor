//! Bytewise A/B test harness: compare `X86JitSpan` output against the
//! `pool_eval` reference for the same NanoGraph and inputs.
//!
//! Pool_eval is the slow-but-correct reference per the dtype contract
//! (see `docs/dtype_contract.md`); a conformant runtime must produce
//! bit-identical bytes. This harness is used by every layer's tests
//! once it has something concrete to compile.
//!
//! P2.B.3: lights up for Identity-only graphs. The `inputs` slice is
//! converted into `StoreSlice`s and `PoolEvalSpan` derives its
//! declared input ranges from the graph's `input_tensors()` so the
//! caller doesn't have to specify them twice.

use crate::compiler::attempts::v14::executor::{
    CompiledSpanFn, PoolEvalSpan, SpanOutput, StoreSlice,
};
use crate::compiler::attempts::v14::x86_jit::X86JitSpan;
use crate::nano_graph::pattern::{AtomId, AtomRange, NanoGraph};
use crate::numeric_dtype::NumericDType;
use crate::pool::SystemPool;

/// A/B harness: compile via `X86JitSpan` AND `PoolEvalSpan`, run on
/// the same `StoreSlice` inputs, assert byte-identical outputs, and
/// return the (now-agreed) output bytes.
///
/// Inputs are `(base_atom, dtype, raw_bytes)` triples. The harness
/// constructs `StoreSlice` views over the raw bytes for both runs.
///
/// **Panics** if the two backends produce different bytes for any
/// output. The panic message identifies which output index differs.
pub fn ab_test_bytes(
    graph: &NanoGraph<'static, SystemPool>,
    inputs: &[(AtomId, NumericDType, Vec<u8>)],
    outputs: &[AtomRange],
) -> Vec<Vec<u8>> {
    let store_slices: Vec<StoreSlice<'_>> = inputs
        .iter()
        .map(|(base, dtype, bytes)| StoreSlice {
            base: *base,
            data: bytes.as_slice(),
            dtype: *dtype,
            count: (bytes.len() / dtype.bytes_per_element()) as u64,
            src_bit_offset: 0,
            src_bit_stride: (dtype.bytes_per_element() as u64) * 8,
        })
        .collect();

    let alloc_out = || -> Vec<Vec<u8>> {
        outputs
            .iter()
            .map(|r| vec![0u8; r.count as usize * r.dtype.bytes_per_element()])
            .collect()
    };

    // PoolEvalSpan needs to know which atom ranges to extract from
    // the StoreSlices and turn into NumericTensors. The graph's
    // declared `input_tensors()` are exactly that — derive them here
    // so the caller only specifies the data once.
    let pool_input_ranges: Vec<AtomRange> = graph
        .input_tensors()
        .iter()
        .map(|it| AtomRange {
            base: it.base_id,
            count: it.count,
            dtype: it.dtype,
        })
        .collect();

    let mut pool_outs = alloc_out();
    {
        let span = PoolEvalSpan::new(graph.clone(), pool_input_ranges, outputs.to_vec());
        let mut span_outs: Vec<SpanOutput<'_>> = outputs
            .iter()
            .zip(pool_outs.iter_mut())
            .map(|(r, buf)| SpanOutput {
                data: buf.as_mut_slice(),
                dtype: r.dtype,
                count: r.count,
            })
            .collect();
        span.execute(&store_slices, &mut span_outs);
    }

    let mut x86_outs = alloc_out();
    {
        let span = X86JitSpan::compile(graph, outputs)
            .expect("x86_jit compile (ab_test_bytes is for spans inside the support envelope)");
        let mut span_outs: Vec<SpanOutput<'_>> = outputs
            .iter()
            .zip(x86_outs.iter_mut())
            .map(|(r, buf)| SpanOutput {
                data: buf.as_mut_slice(),
                dtype: r.dtype,
                count: r.count,
            })
            .collect();
        span.execute(&store_slices, &mut span_outs);
    }

    for (i, (a, b)) in pool_outs.iter().zip(x86_outs.iter()).enumerate() {
        assert_eq!(
            a, b,
            "output {i}: pool_eval vs x86_jit byte mismatch \
             (pool={a:?}, x86={b:?})"
        );
    }
    pool_outs
}
