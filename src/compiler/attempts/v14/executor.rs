#![allow(clippy::all, dead_code, unreachable_patterns, unused)]
//! Executable plan runner.
//!
//! `ExecutablePlan` is a frozen compilation artifact: compiled spans
//! grouped by phase/lane, the global `AtomPlacementMap`, a fixed-worker
//! rayon thread pool for lane dispatch, and a plan-wide literal buffer.
//! Every field is immutable.
//!
//! `execute(&self, input_ptrs, pool)` is the one entry point. On each
//! call it allocates the intermediate buffer, per-lane scratch arenas,
//! and output tensors from the pool; builds the stack `buffer_ptrs`
//! array; dispatches phases over the lane workers; and returns the
//! output `NumericTensor`s directly, in `pinned_output_ranges` order.
//! The plan mutates nothing and `execute` is safely callable from
//! concurrent threads (subject to the pool's own rules).

use std::collections::HashMap;
use std::env;
use std::time::Instant;

use rayon::{ThreadPool, ThreadPoolBuilder};

use crate::nano_graph::pattern::GraphConstantId;
use crate::nano_graph::{AtomId, AtomRange};
use crate::numeric_dtype::NumericDType;
use crate::numeric_tensor::{NumericTensor, TensorLayout};
use crate::pool::Pool;
use crate::tensor_rank::DynRank;

use super::placer::{AtomPlacementMap, BufferInfo, BufferKind};

// ─── Compiled span trait ────────────────────────────────────────────────────

/// A prepared span that can execute given a `buffer_ptrs` array.
///
/// Backend-agnostic: x86_jit and pool_eval fallback both implement
/// this. Under the memory-placement design every span addresses its
/// atoms via `buffer_ptrs[buffer_id]`, so the entire I/O surface is
/// the array the executor hands in on every dispatch.
///
/// # buffer_ptrs contract
///
/// - `buffer_ptrs[buffer_id]` is the base of the buffer with that id.
/// - The intermediate, literal, input, and output slots are filled in
///   by the executor before every phase.
/// - The scratch slot (`placement.scratch_buffer_id`) is patched
///   per-lane inside the dispatch closure, so each worker sees its
///   own lane's scratch arena.
/// - Every pointer is valid for the duration of the `execute` call.
///   The span may read/write within each buffer's declared size and
///   must not stray outside.
pub trait CompiledSpanFn: Send + Sync {
    /// Bytes of lane-private scratch this span needs. The executor
    /// allocates a scratch arena on each lane sized to the max over
    /// all spans running on that lane.
    fn scratch_bytes(&self) -> usize;

    /// Execute the span. See the buffer_ptrs contract above.
    ///
    /// `bindings` supplies runtime values for every `GraphConstantId`
    /// referenced by the span's sym_dims. Empty when the span has no
    /// sym dims. The JIT path ignores this (sym dims are compiled out
    /// of its code) — the pool-eval path threads it into
    /// `pool_eval` so sym extents resolve correctly at runtime.
    fn execute(&self, buffer_ptrs: &[*mut u8], bindings: &HashMap<GraphConstantId, u64>);
}

// ─── Pool-eval fallback span ────────────────────────────────────────────────

/// Pre-resolved byte-range metadata for a span input or output.
///
/// PoolEvalSpan resolves each declared `AtomRange` to its placer-
/// assigned `(buffer_id, byte_offset)` at construction time; at
/// execute time it walks this list to gather from / scatter to
/// `buffer_ptrs` without touching the placement map.
#[derive(Debug, Clone)]
struct PoolEvalRange {
    range: AtomRange,
    buffer_id: u8,
    byte_offset: u64,
    /// Sym dims the range's producer iterates over. Each atom in the
    /// range stores `∏ bindings[gc]` elements at runtime; this list
    /// resolves to that product via `bindings`. Empty for sym-free
    /// ranges (atom = scalar).
    sym_dims: Vec<GraphConstantId>,
    /// Atom-to-atom byte stride in the placer buffer the range points
    /// into. For intermediate-buffer sym groups this is
    /// `max_sym_prod * bpe` (max-bound slot reservation); for sym-free
    /// groups and input/output/literal entries it's just `bpe`. When
    /// this differs from `sym_prod * bpe`, `PoolEvalSpan::execute`
    /// does per-atom strided I/O between the placer buffer and its
    /// internal packed buffer. The contract: live data occupies the
    /// first `sym_prod * bpe` bytes of each max-stride atom slot;
    /// the trailing `(max_sym_prod - sym_prod) * bpe` bytes are slack.
    atom_byte_stride: u64,
}

/// A span that evaluates its NanoGraph via `pool_eval` instead of JIT.
///
/// Used for opaque ops and for anything the JIT can't compile. Pool
/// eval needs owned tensors, so this boundary performs a memcpy at
/// entry (buffer_ptrs → flat tensors) and exit (results → buffer_ptrs).
/// Opaque ops are a tiny fraction of execute time so the cost is
/// acceptable.
pub struct PoolEvalSpan {
    graph: crate::nano_graph::pattern::NanoGraph<'static, crate::pool::SystemPool>,
    inputs: Vec<PoolEvalRange>,
    outputs: Vec<PoolEvalRange>,
}

impl PoolEvalSpan {
    /// `sym_dims_by_range` supplies sym_dims for atom ranges whose base
    /// does not match a producing group in the span's graph — i.e. external
    /// input tensors or cross-span intermediates, whose sym structure lives
    /// in the caller's main-graph map. Lookup is range-based so a span
    /// receiving an AtomRange with base in the middle of a main-graph
    /// group still resolves sym_dims (the partitioner may split a group
    /// into count=1 ranges across spans).
    /// For atoms produced by groups in the span's graph, sym_dims are
    /// read directly from the group.
    pub fn new(
        graph: crate::nano_graph::pattern::NanoGraph<'static, crate::pool::SystemPool>,
        inputs: Vec<AtomRange>,
        outputs: Vec<AtomRange>,
        placement: &AtomPlacementMap,
        sym_dims_by_range: &crate::range_map::RangeMap<Vec<GraphConstantId>>,
    ) -> Self {
        let resolve_sym =
            |base: AtomId,
             g: &crate::nano_graph::pattern::NanoGraph<'static, crate::pool::SystemPool>|
             -> Vec<GraphConstantId> {
                if let Some(gi) = g.find_group_idx(base) {
                    g.groups()[gi].sym_dims.clone()
                } else if let Some((sd, _)) = sym_dims_by_range.get(base.0) {
                    sd.clone()
                } else {
                    Vec::new()
                }
            };
        let resolve = |range: AtomRange| -> PoolEvalRange {
            let (buf, off) = placement.byte_offset_of(range.base).unwrap_or_else(|| {
                panic!(
                    "PoolEvalSpan: atom range base={} not in placement map",
                    range.base.0
                )
            });
            let atom_byte_stride = placement
                .atom_byte_stride_of(range.base)
                .unwrap_or_else(|| range.dtype.bytes_per_element() as u64);
            let sym_dims = resolve_sym(range.base, &graph);
            PoolEvalRange {
                range,
                buffer_id: buf.0,
                byte_offset: off,
                sym_dims,
                atom_byte_stride,
            }
        };
        let resolved_inputs: Vec<PoolEvalRange> = inputs.into_iter().map(resolve).collect();
        let resolved_outputs: Vec<PoolEvalRange> = outputs.into_iter().map(resolve).collect();

        // DIAG WT_DUMP_POOLSPAN_INPUTS: dump caller-side PoolEvalRange
        // vs sub-graph input_tensors alignment at construction time.
        // If graph.input_tensors[ti] range doesn't match the caller's
        // resolved input range for that slot, pool_eval will compute
        // (offset * sym_prod + sym_flat) from graph.input_tensors[ti]
        // while the store is sized by the caller's TAM count → OOB or
        // wrong data.
        if std::env::var("WT_DUMP_POOLSPAN_INPUTS")
            .ok()
            .is_some_and(|v| v != "0")
        {
            let span_id: u64 = {
                let mut h: u64 = 0xcbf29ce484222325;
                for pr in &resolved_inputs {
                    h ^= pr.range.base.0;
                    h = h.wrapping_mul(0x100000001b3);
                }
                h ^= 0x9E3779B97F4A7C15;
                for pr in &resolved_outputs {
                    h ^= pr.range.base.0;
                    h = h.wrapping_mul(0x100000001b3);
                }
                h
            };
            for (i, pr) in resolved_inputs.iter().enumerate() {
                let sdstr = pr
                    .sym_dims
                    .iter()
                    .map(|gc| format!("gc{}", gc.0))
                    .collect::<Vec<_>>()
                    .join(",");
                eprintln!(
                    "[SPANDECL] span={:016x} kind=CALLER_IN i={} base={} count={} stride={} sd=[{}]",
                    span_id,
                    i,
                    pr.range.base.0,
                    pr.range.count,
                    pr.atom_byte_stride,
                    sdstr,
                );
            }
            for (ti, it) in graph.input_tensors().iter().enumerate() {
                eprintln!(
                    "[SPANDECL] span={:016x} kind=GRAPH_IN  ti={} base={} count={} dtype={:?}",
                    span_id,
                    ti,
                    it.base_id.0,
                    it.count,
                    it.dtype,
                );
            }
        }

        Self {
            inputs: resolved_inputs,
            outputs: resolved_outputs,
            graph,
        }
    }
}

// Deterministic FNV-1a hash of a byte slice. Used for the
// `WT_POOLEVAL_TRACE` diagnostic log so the trace line stays short but
// still detects any byte-level divergence between runs.
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn trace_enabled() -> bool {
    env::var("WT_POOLEVAL_TRACE")
        .map(|v| v != "0" && !v.is_empty())
        .unwrap_or(false)
}

impl CompiledSpanFn for PoolEvalSpan {
    fn scratch_bytes(&self) -> usize {
        // pool_eval manages its own tensor allocations internally;
        // the executor-provided scratch slot is unused.
        0
    }

    fn execute(&self, buffer_ptrs: &[*mut u8], bindings: &HashMap<GraphConstantId, u64>) {
        use crate::nano_graph::lower::{DimKind, TensorAtomMapInfo};
        use crate::nano_graph::pool_eval;
        use crate::numeric_tensor::{NumericTensor, NumericTensorView};
        use crate::pool::SystemPool;

        static SYS: SystemPool = SystemPool;

        // WT_POOLEVAL_TRACE=1 → log every input/output range's bytes at
        // gather/scatter boundaries. Stable span_id lets a Trivial vs
        // LaneSplit diff line up by base-atom across runs.
        let trace = trace_enabled();
        let span_id: u64 = {
            let mut h: u64 = 0xcbf29ce484222325;
            for pr in &self.inputs {
                h ^= pr.range.base.0;
                h = h.wrapping_mul(0x100000001b3);
            }
            h ^= 0x9E3779B97F4A7C15;
            for pr in &self.outputs {
                h ^= pr.range.base.0;
                h = h.wrapping_mul(0x100000001b3);
            }
            h
        };

        // Helpers -------------------------------------------------------
        // Resolve sym_prod for a range's sym_dims using the runtime
        // bindings. Sym-free ranges yield 1. Missing binding panics —
        // callers must supply every gc the plan references.
        let sym_prod = |sym_dims: &[GraphConstantId]| -> u64 {
            sym_dims
                .iter()
                .map(|gc| {
                    bindings.get(gc).copied().unwrap_or_else(|| {
                        panic!("pool_eval span: sym gc {:?} missing from bindings", gc)
                    })
                })
                .product()
        };
        // Build a TAMI that describes a range's internal atom-major-
        // sym-innermost layout: a single Known dim spanning the atoms
        // with stride 1 (so atom i has atom_id=i within the store),
        // followed by one Sym entry per sym_dim in insertion order.
        // Pool_eval uses this to decompose the flat view's elements
        // into (atom_idx, sym_point) positions.
        let build_tami = |base: AtomId,
                          count: u64,
                          dtype: crate::numeric_dtype::NumericDType,
                          sym_dims: &[GraphConstantId]|
         -> TensorAtomMapInfo {
            let mut dims: Vec<DimKind> = Vec::with_capacity(1 + sym_dims.len());
            dims.push(DimKind::Known {
                size: count,
                stride: 1,
            });
            for (axis, _gc) in sym_dims.iter().enumerate() {
                dims.push(DimKind::Sym {
                    gc: sym_dims[axis],
                    axis,
                });
            }
            TensorAtomMapInfo {
                base_id: base,
                count,
                dtype,
                dims,
                segments: vec![],
            }
        };

        // Gather inputs: copy each range's bytes out of its assigned
        // buffer into a pool-allocated flat tensor. pool_eval expects
        // owned tensors, not raw pointers — this boundary copy is
        // unavoidable. Size accounts for sym-prod expansion.
        //
        // When `atom_byte_stride > runtime_sym_prod * bpe` — true for
        // intermediate-buffer sym groups, which reserve `max_sym_prod`
        // slots per atom — the copy is per-atom strided: atom `k`'s
        // live data sits at `byte_offset + k * atom_byte_stride` in
        // the placer buffer, and we pack it into `k * sym_prod * bpe`
        // in the local buffer. Sym-free and runtime-packed cases hit
        // the degenerate branch (stride == row_bytes, one memcpy).
        let mut input_tamis: Vec<TensorAtomMapInfo> = Vec::new();
        let mut input_tensors: Vec<NumericTensor<'_, DynRank, SystemPool>> = Vec::new();

        for pr in &self.inputs {
            let range = &pr.range;
            let bpe = range.dtype.bytes_per_element();
            let sp = sym_prod(&pr.sym_dims);
            let total_elems = range.count * sp;
            let needed_bytes = total_elems as usize * bpe;
            let layout = jit_flat_layout(total_elems, range.dtype);
            let mut buf = SYS
                .allocate(layout.buffer_size_bytes().max(needed_bytes))
                .expect("pool_eval span: alloc failed");

            // Null input pointer → zero-fill (see explanation above for
            // declared-but-unused lowering inputs).
            let base_ptr = buffer_ptrs[pr.buffer_id as usize];
            if !base_ptr.is_null() {
                // SAFETY: the executor guarantees each non-null
                // buffer_ptrs slot is valid for the declared buffer
                // size and the placer's byte range sits inside it.
                let src_base = unsafe { base_ptr.add(pr.byte_offset as usize) as *const u8 };
                let row_bytes = sp as usize * bpe;
                if (pr.atom_byte_stride as usize) >= row_bytes {
                    // Strided path. `atom_byte_stride == row_bytes`
                    // (sym-free or sym-prod == max) degenerates to a
                    // single memcpy pattern; `>` is the sym-max-stride
                    // case where each atom's live data sits in the
                    // first `row_bytes` of a larger slot.
                    if pr.atom_byte_stride as usize == row_bytes {
                        let copy_bytes = needed_bytes.min(buf.len());
                        unsafe {
                            std::ptr::copy_nonoverlapping(src_base, buf.as_mut_ptr(), copy_bytes);
                        }
                    } else {
                        let dst_base = buf.as_mut_ptr();
                        for k in 0..range.count as usize {
                            let src = unsafe {
                                src_base.add(k * pr.atom_byte_stride as usize) as *const u8
                            };
                            let dst = unsafe { dst_base.add(k * row_bytes) };
                            let copy = row_bytes.min(buf.len().saturating_sub(k * row_bytes));
                            if copy == 0 {
                                break;
                            }
                            unsafe {
                                std::ptr::copy_nonoverlapping(src, dst, copy);
                            }
                        }
                    }
                } else {
                    // Runtime-packed path: placer entry's
                    // `atom_byte_stride == bpe` but the source actually
                    // stores each atom at `row_bytes` (= sym_prod *
                    // bpe) because the caller / executor wrote it that
                    // way (input/output buffers with sym). Whole range
                    // is a contiguous `count * row_bytes` block.
                    let copy_bytes = needed_bytes.min(buf.len());
                    unsafe {
                        std::ptr::copy_nonoverlapping(src_base, buf.as_mut_ptr(), copy_bytes);
                    }
                }
            }

            if trace {
                let live = needed_bytes.min(buf.len());
                let hash = fnv1a64(&buf[..live]);
                let preview_len = live.min(16);
                let mut hex = String::with_capacity(preview_len * 2);
                for b in &buf[..preview_len] {
                    hex.push_str(&format!("{:02x}", b));
                }
                let sym_dims_str = pr
                    .sym_dims
                    .iter()
                    .map(|gc| format!("gc{}", gc.0))
                    .collect::<Vec<_>>()
                    .join(",");
                eprintln!(
                    "[POOLTRACE] span={:016x} IN  base={} buf={} off={} stride={} sd=[{}] sp={} count={} bpe={} bytes={} first16={} hash={:016x}",
                    span_id,
                    range.base.0,
                    pr.buffer_id,
                    pr.byte_offset,
                    pr.atom_byte_stride,
                    sym_dims_str,
                    sp,
                    range.count,
                    bpe,
                    live,
                    hex,
                    hash,
                );
            }

            input_tamis.push(build_tami(
                range.base,
                range.count,
                range.dtype,
                &pr.sym_dims,
            ));
            input_tensors.push(NumericTensor::from_parts(buf, layout));
        }

        let input_views: Vec<NumericTensorView<'_, DynRank>> =
            input_tensors.iter().map(|t| t.view()).collect();
        let eval_inputs: Vec<(&TensorAtomMapInfo, &NumericTensorView<'_, DynRank>)> =
            input_tamis.iter().zip(input_views.iter()).collect();

        let output_tamis: Vec<TensorAtomMapInfo> = self
            .outputs
            .iter()
            .map(|pr| build_tami(pr.range.base, pr.range.count, pr.range.dtype, &pr.sym_dims))
            .collect();
        let output_tami_refs: Vec<&TensorAtomMapInfo> = output_tamis.iter().collect();

        let results =
            pool_eval::pool_eval(&self.graph, &eval_inputs, &output_tami_refs, bindings, &SYS)
                .expect("pool_eval span: eval failed");

        // Scatter results back into buffer_ptrs. Same strided contract
        // as the input gather: when `atom_byte_stride > row_bytes`
        // (intermediate sym slot), write each atom's `row_bytes` into
        // the head of its max-stride slot and leave the trailing slack
        // untouched. Downstream spans read the same prefix via the
        // same `atom_byte_stride` and get the live data back.
        for (pr, result_tensor) in self.outputs.iter().zip(results.iter()) {
            let src = result_tensor.buffer();
            let bpe = pr.range.dtype.bytes_per_element();
            let sp = sym_prod(&pr.sym_dims);
            let row_bytes = sp as usize * bpe;
            let needed = pr.range.count as usize * row_bytes;
            let dst_base =
                unsafe { buffer_ptrs[pr.buffer_id as usize].add(pr.byte_offset as usize) };
            if trace {
                let live = needed.min(src.len());
                let hash = fnv1a64(&src[..live]);
                let preview_len = live.min(16);
                let mut hex = String::with_capacity(preview_len * 2);
                for b in &src[..preview_len] {
                    hex.push_str(&format!("{:02x}", b));
                }
                let sym_dims_str = pr
                    .sym_dims
                    .iter()
                    .map(|gc| format!("gc{}", gc.0))
                    .collect::<Vec<_>>()
                    .join(",");
                eprintln!(
                    "[POOLTRACE] span={:016x} OUT base={} buf={} off={} stride={} sd=[{}] sp={} count={} bpe={} bytes={} first16={} hash={:016x}",
                    span_id,
                    pr.range.base.0,
                    pr.buffer_id,
                    pr.byte_offset,
                    pr.atom_byte_stride,
                    sym_dims_str,
                    sp,
                    pr.range.count,
                    bpe,
                    live,
                    hex,
                    hash,
                );
            }
            if (pr.atom_byte_stride as usize) >= row_bytes {
                if pr.atom_byte_stride as usize == row_bytes {
                    let copy = src.len().min(needed);
                    unsafe {
                        std::ptr::copy_nonoverlapping(src.as_ptr(), dst_base, copy);
                    }
                } else {
                    let src_base = src.as_ptr();
                    for k in 0..pr.range.count as usize {
                        let src_off = k * row_bytes;
                        if src_off >= src.len() {
                            break;
                        }
                        let copy = row_bytes.min(src.len() - src_off);
                        let dst = unsafe { dst_base.add(k * pr.atom_byte_stride as usize) };
                        let srcp = unsafe { src_base.add(src_off) };
                        unsafe {
                            std::ptr::copy_nonoverlapping(srcp, dst, copy);
                        }
                    }
                }
            } else {
                // Runtime-packed destination (input/output buffer with
                // sym — `atom_byte_stride == bpe`, but writes need to
                // pack at `sym_prod * bpe`). Whole range is contiguous.
                let copy = src.len().min(needed);
                unsafe {
                    std::ptr::copy_nonoverlapping(src.as_ptr(), dst_base, copy);
                }
            }
        }
    }
}

// ─── Executable plan ────────────────────────────────────────────────────────

/// A fully compiled execution plan, ready to run.
///
/// Fully immutable — all working buffers are allocated per execute
/// call from the pool. The plan owns:
///
/// - The compiled spans, grouped by phase and lane.
/// - The placement map (with all buffer sizes and ids).
/// - A fixed-worker rayon thread pool for lane dispatch.
/// - A `Box<[u8]>` for the plan-wide literal buffer, populated once
///   from the main graph at build time and never mutated again.
/// - Per-lane scratch sizes computed at build time.
/// - The `pinned_output_ranges` list the executor returns in order.
pub struct ExecutablePlan {
    phases: Vec<ExecutablePhase>,
    /// Fixed-worker lane pool. With `broadcast`, lane index `i`
    /// executes on worker `i` every phase.
    lane_pool: Option<ThreadPool>,
    /// Placer output. Holds every atom's `(buffer_id, byte_offset)`
    /// plus the per-buffer size metadata the executor needs at
    /// allocation time.
    placement: AtomPlacementMap,
    /// Plan-wide literal buffer. Sized to `placement.literal_buffer_size`
    /// and populated once at plan-build — every `Literal`/`LiteralSpan`
    /// group's source bytes live here (even groups whose atoms overlap
    /// a model output range; the JIT reads from here and emits an
    /// ordinary copy to the output buffer in that case). Never mutated
    /// after plan-build; linked in as a read-only source on every
    /// `execute` call.
    literal_buffer: Box<[u8]>,
    /// Per-lane scratch high-water marks. `scratch_sizes[i]` is the
    /// max `scratch_bytes()` of any span that runs on lane `i`.
    scratch_sizes: Vec<usize>,
    /// Model output atom ranges in declaration order. Each entry
    /// corresponds to one output buffer_id in the placement map.
    /// Carries the producing group's `sym_dims` so `execute` can
    /// allocate the tensor at runtime size `count × sym_prod`.
    output_ranges: Vec<OutputSlot>,
}

/// One pinned model output: atom range + the sym dims its producing
/// group iterates over. The sym dims resolve at execute-time via the
/// `bindings` HashMap to determine the buffer's runtime size.
#[derive(Clone)]
pub struct OutputSlot {
    pub range: AtomRange,
    pub sym_dims: Vec<GraphConstantId>,
}

struct ExecutablePhase {
    lanes: Vec<ExecutableLane>,
}

struct ExecutableLane {
    span: Box<dyn CompiledSpanFn>,
    #[allow(dead_code)]
    inputs: Vec<AtomRange>,
    #[allow(dead_code)]
    outputs: Vec<AtomRange>,
}

/// Builder for constructing an ExecutablePlan from compiled spans.
pub struct ExecutablePlanBuilder {
    phases: Vec<ExecutablePhase>,
    pinned_output_ranges: Vec<OutputSlot>,
}

impl ExecutablePlanBuilder {
    pub fn new() -> Self {
        ExecutablePlanBuilder {
            phases: Vec::new(),
            pinned_output_ranges: Vec::new(),
        }
    }

    /// Register model-output atom ranges with their producing group's
    /// sym_dims. Stored verbatim; the order matches the output
    /// buffer_ids the placer assigned.
    pub fn pin_outputs(
        &mut self,
        ranges: &[AtomRange],
        sym_dims_by_range: &[Vec<GraphConstantId>],
    ) {
        assert_eq!(
            ranges.len(),
            sym_dims_by_range.len(),
            "pin_outputs: ranges and sym_dims_by_range length mismatch"
        );
        for (r, s) in ranges.iter().zip(sym_dims_by_range.iter()) {
            self.pinned_output_ranges.push(OutputSlot {
                range: r.clone(),
                sym_dims: s.clone(),
            });
        }
    }

    /// Add a phase with one compiled span per lane.
    pub fn add_phase(
        &mut self,
        lanes: Vec<(Box<dyn CompiledSpanFn>, Vec<AtomRange>, Vec<AtomRange>)>,
    ) {
        let mut exec_lanes = Vec::with_capacity(lanes.len());
        for (span, inputs, outputs) in lanes {
            exec_lanes.push(ExecutableLane {
                span,
                inputs,
                outputs,
            });
        }
        self.phases.push(ExecutablePhase { lanes: exec_lanes });
    }

    /// Consume the builder and produce a runnable plan. The main
    /// graph is walked once to populate the literal buffer.
    pub fn build(
        self,
        placement: AtomPlacementMap,
        main_graph: &crate::nano_graph::pattern::NanoGraph<'static, crate::pool::SystemPool>,
    ) -> ExecutablePlan {
        let max_lanes = self.phases.iter().map(|p| p.lanes.len()).max().unwrap_or(1);

        // Per-lane scratch sizing. Lane i runs the span at index i of
        // every phase (the broadcast scheduler guarantees this).
        let mut scratch_sizes: Vec<usize> = vec![0; max_lanes];
        for phase in &self.phases {
            for (li, lane) in phase.lanes.iter().enumerate() {
                scratch_sizes[li] = scratch_sizes[li].max(lane.span.scratch_bytes());
            }
        }

        // Allocate and populate the plan-wide literal buffer. Every
        // Literal/LiteralSpan group's source bytes land here — both
        // groups whose primary slot is the literal buffer (read-only
        // consumer path) and groups whose primary slot is an output
        // buffer (the JIT reads from here and emits an ordinary copy
        // to the output buffer).
        let literal_size = placement.literal_buffer_size as usize;
        let mut literal_buffer: Box<[u8]> = vec![0u8; literal_size].into_boxed_slice();
        populate_literal_buffer(main_graph, &placement, &mut literal_buffer);

        let output_ranges = self.pinned_output_ranges.clone();
        let lane_pool = build_lane_thread_pool(max_lanes);

        ExecutablePlan {
            phases: self.phases,
            lane_pool,
            placement,
            literal_buffer,
            scratch_sizes,
            output_ranges,
        }
    }
}

/// Walk the main graph and write every `Literal`/`LiteralSpan`
/// group's source bytes into the plan-wide literal buffer at the
/// offset the placer assigned in `literal_sources`. **Every** literal
/// group has an entry — groups whose primary slot is an output
/// buffer still have their source bytes here, because the JIT emits
/// an ordinary copy from this buffer to the destination when it
/// compiles such groups.
fn populate_literal_buffer(
    graph: &crate::nano_graph::pattern::NanoGraph<'static, crate::pool::SystemPool>,
    placement: &AtomPlacementMap,
    buffer: &mut [u8],
) {
    use crate::nano_graph::ops::ScalarOp;

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

fn build_lane_thread_pool(max_lanes: usize) -> Option<ThreadPool> {
    if max_lanes <= 1 {
        return None;
    }
    let use_dedicated = env::var("WT_EXECUTOR_DEDICATED_LANES")
        .ok()
        .is_none_or(|v| v != "0");
    if !use_dedicated {
        return None;
    }
    Some(
        ThreadPoolBuilder::new()
            .num_threads(max_lanes)
            .thread_name(|idx| format!("wt-lane-{idx}"))
            .build()
            .expect("failed to build dedicated lane thread pool"),
    )
}

impl ExecutablePlan {
    /// Number of phases.
    pub fn num_phases(&self) -> usize {
        self.phases.len()
    }

    /// Placement map the executor was built with. Callers inspect
    /// this to map atom ids back to buffer slots.
    pub fn placement(&self) -> &AtomPlacementMap {
        &self.placement
    }

    /// Execute the plan. Returns output tensors in
    /// `pinned_output_ranges` declaration order.
    ///
    /// `input_ptrs` is a dense array indexed by `buffer_id`; for each
    /// input buffer the caller hands in a non-null `*mut u8` pointing
    /// at the tensor's bytes (any other slot is ignored — the
    /// executor fills intermediate / literal / output / scratch slots
    /// itself). The caller must keep every input pointer alive for
    /// the duration of this call.
    ///
    /// `bindings` supplies runtime values for every `GraphConstantId`
    /// the plan references (sym dims in any span or output tensor).
    /// Empty for sym-free graphs.
    pub fn execute<'p, P: Pool + 'p>(
        &self,
        input_ptrs: &[*mut u8],
        bindings: &HashMap<GraphConstantId, u64>,
        pool: &'p P,
    ) -> Vec<NumericTensor<'p, DynRank, P>> {
        self.execute_inner(input_ptrs, bindings, pool, false)
    }

    /// Execute with per-phase timing diagnostics.
    pub fn execute_timed<'p, P: Pool + 'p>(
        &self,
        input_ptrs: &[*mut u8],
        bindings: &HashMap<GraphConstantId, u64>,
        pool: &'p P,
    ) -> Vec<NumericTensor<'p, DynRank, P>> {
        self.execute_inner(input_ptrs, bindings, pool, true)
    }

    fn execute_inner<'p, P: Pool + 'p>(
        &self,
        input_ptrs: &[*mut u8],
        bindings: &HashMap<GraphConstantId, u64>,
        pool: &'p P,
        timed: bool,
    ) -> Vec<NumericTensor<'p, DynRank, P>> {
        let scratch_id = self.placement.scratch_buffer_id;
        let ptr_array_len = (scratch_id as usize) + 1;

        // Allocate the intermediate buffer from the pool. Pool reuse
        // keeps the backing chunk stable across calls, so cache state
        // can persist within one execute even though the allocation
        // is formally per-call.
        let intermediate_size = self.placement.intermediate_peak_bytes as usize;
        let mut intermediate_buf = pool
            .allocate(intermediate_size.max(1))
            .expect("intermediate buffer allocation failed");

        // Allocate per-lane scratch arenas from the pool.
        let mut scratch_bufs: Vec<_> = self
            .scratch_sizes
            .iter()
            .map(|&n| {
                pool.allocate(n.max(1))
                    .expect("scratch arena allocation failed")
            })
            .collect();

        // Allocate output tensors from the pool. The `buffer_ptrs`
        // slot for each output buffer_id will point at one of these,
        // and they get handed back to the caller at the end — no
        // copy at extraction time.
        //
        // Take pointers *after* the tensor is in the Vec so a move
        // into the Vec can't invalidate the pointer. (Heap buffers
        // are stable across `NumericTensor` moves in practice, but
        // the order here is defensive.)
        let mut output_tensors: Vec<NumericTensor<'p, DynRank, P>> =
            Vec::with_capacity(self.output_ranges.len());
        for slot in &self.output_ranges {
            // Output tensor sizes include runtime sym expansion:
            //   elements = count × ∏ bindings[gc] for gc in sym_dims
            // For sym-free outputs sym_dims is empty → sym_prod = 1
            // and the allocation matches the prior concrete-only path.
            let sym_prod: u64 = slot
                .sym_dims
                .iter()
                .map(|gc| {
                    bindings.get(gc).copied().unwrap_or_else(|| {
                        panic!("execute: output sym gc {:?} missing from bindings", gc)
                    })
                })
                .product();
            let layout = jit_flat_layout(slot.range.count * sym_prod, slot.range.dtype);
            let buf = pool
                .allocate(layout.buffer_size_bytes())
                .expect("output tensor allocation failed");
            output_tensors.push(NumericTensor::from_parts(buf, layout));
        }
        // Indexed construction — we materialize each ptr via an
        // index expression, which refers directly to the Vec's
        // backing storage each iteration. Taking the ptrs via
        // `iter_mut().map(...).collect()` produced divergent values
        // in practice (possibly a Rust aliasing artifact around the
        // temporary `&mut [u8]` slice in the map closure), and the
        // indexed form sidesteps it.
        let mut output_ptrs: Vec<*mut u8> = Vec::with_capacity(output_tensors.len());
        for i in 0..output_tensors.len() {
            let p = output_tensors[i].buffer_mut().as_mut_ptr();
            output_ptrs.push(p);
        }

        // Build the `buffer_ptrs` template. Input/output/intermediate/
        // literal/scratch slots all come from here; scratch is patched
        // per-lane inside the dispatch closure.
        let mut template: Vec<*mut u8> = vec![std::ptr::null_mut(); ptr_array_len];
        let mut next_output = 0usize;
        for info in self.placement.buffers.iter() {
            let slot = info.id.0 as usize;
            if slot >= ptr_array_len {
                panic!(
                    "execute: buffer_id {} exceeds ptr_array_len {}",
                    slot, ptr_array_len
                );
            }
            match info.kind {
                BufferKind::Intermediate => {
                    template[slot] = intermediate_buf.as_mut_ptr();
                }
                BufferKind::Literal => {
                    template[slot] = self.literal_buffer.as_ptr() as *mut u8;
                }
                BufferKind::Input => {
                    // A null input pointer is tolerated: the span's
                    // JIT emits loads only for atoms whose consumers
                    // are alive, so a declared-but-unused input
                    // buffer is never dereferenced. The prologue may
                    // still load the base pointer into a register
                    // (via `buffer_ptrs[buf_id]`), but that's
                    // harmless — nothing dereferences it. Unused
                    // inputs arise in test cases where a ModelOp's
                    // constant attribute gets lifted to an input
                    // tensor that the final lowering never reads.
                    template[slot] = input_ptrs
                        .get(slot)
                        .copied()
                        .unwrap_or(std::ptr::null_mut());
                }
                BufferKind::Output => {
                    let ptr = output_ptrs[next_output];
                    next_output += 1;
                    template[slot] = ptr;
                }
                BufferKind::Scratch => {
                    // Scratch is not a real BufferInfo entry; this
                    // branch only fires if someone sticks one in the
                    // metadata. Safe to ignore — the scratch slot is
                    // patched per-lane below.
                }
            }
        }

        // Dispatch each phase. The scratch slot is patched per lane
        // inside the broadcast closure so every worker sees its own
        // lane's arena.
        let lane_scratch_ptrs: Vec<*mut u8> =
            scratch_bufs.iter_mut().map(|b| b.as_mut_ptr()).collect();

        let mut total_spans = std::time::Duration::ZERO;
        for (pi, phase) in self.phases.iter().enumerate() {
            let t0 = Instant::now();
            Self::dispatch_phase(
                self.lane_pool.as_ref(),
                phase,
                &template,
                &lane_scratch_ptrs,
                scratch_id,
                bindings,
            );
            let dt = t0.elapsed();
            total_spans += dt;

            if timed && (dt.as_millis() > 100 || pi < 3 || pi + 1 == self.phases.len()) {
                let rss_mb = read_rss_mb();
                eprintln!(
                    "  phase {:>3}: spans={:.1}ms (RSS {:.0}MB)",
                    pi,
                    dt.as_secs_f64() * 1e3,
                    rss_mb,
                );
            }
        }

        if timed {
            eprintln!("  TOTALS: spans={:.1}ms", total_spans.as_secs_f64() * 1e3);
        }

        // intermediate_buf and scratch_bufs drop at function exit;
        // the pool reclaims them. Output tensors carry owned
        // allocations and are returned to the caller.
        output_tensors
    }

    /// Dispatch one phase across the lane workers.
    ///
    /// Associated fn (not `&self`) so the broadcast closure captures
    /// only what it needs.
    fn dispatch_phase(
        lane_pool: Option<&ThreadPool>,
        phase: &ExecutablePhase,
        template: &[*mut u8],
        lane_scratch_ptrs: &[*mut u8],
        scratch_buffer_id: u8,
        bindings: &HashMap<GraphConstantId, u64>,
    ) {
        use std::sync::atomic::{AtomicPtr, Ordering};

        let scratch_id = scratch_buffer_id as usize;

        // Rayon's `broadcast` closure has to be `Sync`, but
        // `&[*mut u8]` is `!Sync`. Re-box the pointers into
        // `AtomicPtr<u8>` — which *is* `Sync` by design — and load
        // them back out inside the closure. This is just a transport
        // wrapper, not shared-mutable state; the partitioner
        // guarantees lane workers touch disjoint byte ranges.
        let template_atomic: Vec<AtomicPtr<u8>> =
            template.iter().map(|&p| AtomicPtr::new(p)).collect();
        let scratch_atomic: Vec<AtomicPtr<u8>> = lane_scratch_ptrs
            .iter()
            .map(|&p| AtomicPtr::new(p))
            .collect();

        if let Some(lane_pool) = lane_pool {
            let lane_count = phase.lanes.len();
            let template_atomic = &template_atomic;
            let scratch_atomic = &scratch_atomic;
            lane_pool.broadcast(move |bctx| {
                let lane_idx = bctx.index();
                if lane_idx >= lane_count {
                    return;
                }
                let mut local: Vec<*mut u8> = template_atomic
                    .iter()
                    .map(|a| a.load(Ordering::Relaxed))
                    .collect();
                if scratch_id < local.len() {
                    local[scratch_id] = scratch_atomic[lane_idx].load(Ordering::Relaxed);
                }
                phase.lanes[lane_idx].span.execute(&local, bindings);
            });
        } else {
            // Single-lane fallback.
            for (lane_idx, lane) in phase.lanes.iter().enumerate() {
                let mut local: Vec<*mut u8> = template.to_vec();
                if scratch_id < local.len() {
                    local[scratch_id] = lane_scratch_ptrs[lane_idx];
                }
                lane.span.execute(&local, bindings);
            }
        }
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Create a flat 1D tensor layout with byte-aligned element strides.
///
/// The JIT writes one byte per element for sub-byte dtypes (e.g. BOOL),
/// so the stride must be `bytes_per_element * 8` bits, not `total_bits`.
/// For types ≥ 8 bits this is identical to `row_major`.
pub(crate) fn jit_flat_layout(count: u64, dtype: NumericDType) -> TensorLayout<DynRank> {
    let stride_bits = dtype.bytes_per_element() as u64 * 8;
    TensorLayout::<DynRank>::ElementStrided {
        shape: vec![count],
        dtype,
        strides: vec![stride_bits],
        offset_bits: 0,
    }
}

fn read_rss_mb() -> f64 {
    std::fs::read_to_string("/proc/self/statm")
        .ok()
        .and_then(|s| s.split_whitespace().nth(1)?.parse::<u64>().ok())
        .unwrap_or(0) as f64
        * 4096.0
        / (1024.0 * 1024.0)
}
