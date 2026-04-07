#![allow(clippy::all, dead_code, unreachable_patterns, unused)]
//! Backend-agnostic parallel executor for execution plans.
//!
//! Drives phases with barriers, runs spans in parallel across lanes,
//! and manages the inter-phase value store with liveness-based eviction.
//!
//! The executor is agnostic to how spans compute — JIT, interpreter, and
//! GPU backends all implement the `CompiledSpanFn` trait.

use std::collections::HashMap;
use std::time::Instant;

use rayon::prelude::*;

use crate::nano_graph::{AtomId, AtomRange};
use crate::numeric_dtype::NumericDType;
use crate::numeric_tensor::{NumericTensor, NumericTensorCOW, TensorLayout};
use crate::pool::Pool;
use crate::tensor_rank::DynRank;

// ─── Inter-phase data types ─────────────────────────────────────────────────

/// A zero-copy view into a store entry's data.
///
/// Spans receive these for their declared input ranges. The data pointer
/// is valid for the duration of the phase (store is immutably borrowed).
pub struct StoreSlice<'a> {
    pub base: AtomId,
    pub data: &'a [u8],
    pub dtype: NumericDType,
    pub count: u64,
}

/// A pre-allocated output buffer that a compiled span writes into.
///
/// Borrows the backing `NumericTensor`'s byte buffer. The span fills
/// this with computed values; the executor owns the actual tensors.
pub struct SpanOutput<'a> {
    pub data: &'a mut [u8],
    pub dtype: NumericDType,
    pub count: u64,
}

// ─── Compiled span trait ────────────────────────────────────────────────────

/// A prepared span that can execute given input data.
///
/// Backend-agnostic: JIT, interpreter, and GPU all implement this.
/// The executor calls this once per span per phase, passing zero-copy
/// slices from the store and pre-allocated output buffers.
///
/// Pool-agnostic: the trait operates on raw byte buffers. The executor
/// handles pool-aware allocation and wraps tensors as `SpanOutput`
/// before calling this.
pub trait CompiledSpanFn: Send + Sync {
    /// Execute the span.
    ///
    /// `inputs` contains store slices covering all atoms this span reads
    /// from prior phases (weights, user inputs, prior-phase outputs).
    /// The slices may not exactly match the span's declared input ranges —
    /// they're the overlapping store entries found by the executor.
    ///
    /// `outputs` contains pre-allocated byte buffers, one per declared
    /// output range (same order as the span's output declarations).
    /// The implementation fills these with computed values.
    fn execute(&self, inputs: &[StoreSlice<'_>], outputs: &mut [SpanOutput<'_>]);
}

// ─── Pool-eval fallback span ────────────────────────────────────────────────

/// A span that evaluates its NanoGraph via pool_eval instead of JIT.
///
/// Used for spans containing opaque ops (or any ops the JIT can't compile).
/// The NanoGraph fragment and its declared I/O ranges are carried verbatim
/// from the partitioner.
pub struct PoolEvalSpan {
    graph: crate::nano_graph::pattern::NanoGraph<'static, crate::pool::SystemPool>,
    inputs: Vec<crate::nano_graph::AtomRange>,
    outputs: Vec<crate::nano_graph::AtomRange>,
}

impl PoolEvalSpan {
    pub fn new(
        graph: crate::nano_graph::pattern::NanoGraph<'static, crate::pool::SystemPool>,
        inputs: Vec<crate::nano_graph::AtomRange>,
        outputs: Vec<crate::nano_graph::AtomRange>,
    ) -> Self {
        Self {
            graph,
            inputs,
            outputs,
        }
    }
}

impl CompiledSpanFn for PoolEvalSpan {
    fn execute(&self, inputs: &[StoreSlice<'_>], outputs: &mut [SpanOutput<'_>]) {
        use crate::nano_graph::lower::TensorAtomMapInfo;
        use crate::nano_graph::pool_eval;
        use crate::numeric_tensor::{NumericTensor, NumericTensorView, TensorLayout};
        use crate::pool::SystemPool;

        static SYS: SystemPool = SystemPool;

        // Build input TAMIs and views from the StoreSlices.
        // Each declared input range becomes a flat 1D TAMI.
        let mut input_tamis: Vec<TensorAtomMapInfo> = Vec::new();
        let mut input_tensors: Vec<NumericTensor<'_, DynRank, SystemPool>> = Vec::new();

        for range in &self.inputs {
            let tami = TensorAtomMapInfo {
                base_id: range.base,
                count: range.count,
                dtype: range.dtype,
                sym_dims: vec![],
                known_strides: vec![1],
                known_dims: vec![range.count],
                segments: vec![],
            };

            // Find the matching StoreSlice(s) for this range.
            let bpe = range.dtype.bytes_per_element();
            let needed_bytes = range.count as usize * bpe;
            let layout = jit_flat_layout(range.count, range.dtype);
            let mut buf = SYS
                .allocate(layout.buffer_size_bytes().max(needed_bytes))
                .expect("pool_eval span: alloc failed");

            // Gather data from input slices that overlap this range.
            let range_lo = range.base.0;
            let range_hi = range_lo + range.count;
            for slice in inputs {
                let s_lo = slice.base.0;
                let s_hi = s_lo + slice.count;
                // Compute overlap.
                let overlap_lo = range_lo.max(s_lo);
                let overlap_hi = range_hi.min(s_hi);
                if overlap_lo >= overlap_hi {
                    continue;
                }
                let dst_off = (overlap_lo - range_lo) as usize * bpe;
                let src_off = (overlap_lo - s_lo) as usize * bpe;
                let copy_bytes = (overlap_hi - overlap_lo) as usize * bpe;
                if src_off + copy_bytes <= slice.data.len() && dst_off + copy_bytes <= buf.len() {
                    buf[dst_off..dst_off + copy_bytes]
                        .copy_from_slice(&slice.data[src_off..src_off + copy_bytes]);
                }
            }

            let tensor = NumericTensor::from_parts(buf, layout);
            input_tamis.push(tami);
            input_tensors.push(tensor);
        }

        // Build (TAMI, view) pairs for pool_eval.
        let input_views: Vec<NumericTensorView<'_, DynRank>> =
            input_tensors.iter().map(|t| t.view()).collect();
        let eval_inputs: Vec<(&TensorAtomMapInfo, &NumericTensorView<'_, DynRank>)> =
            input_tamis.iter().zip(input_views.iter()).collect();

        // Build output TAMIs.
        let output_tamis: Vec<TensorAtomMapInfo> = self
            .outputs
            .iter()
            .map(|r| TensorAtomMapInfo {
                base_id: r.base,
                count: r.count,
                dtype: r.dtype,
                sym_dims: vec![],
                known_strides: vec![1],
                known_dims: vec![r.count],
                segments: vec![],
            })
            .collect();
        let output_tami_refs: Vec<&TensorAtomMapInfo> = output_tamis.iter().collect();

        // Run pool_eval.
        let results = pool_eval::pool_eval(&self.graph, &eval_inputs, &output_tami_refs, &SYS)
            .expect("pool_eval span: eval failed");

        // Copy results into the pre-allocated output buffers.
        for (out_buf, result_tensor) in outputs.iter_mut().zip(results.iter()) {
            let src = result_tensor.buffer();
            let copy = src.len().min(out_buf.data.len());
            out_buf.data[..copy].copy_from_slice(&src[..copy]);
        }
    }
}

// ─── Phase store ────────────────────────────────────────────────────────────

/// Inter-phase value store with liveness-based eviction.
///
/// Entries are sorted by base AtomId for O(log N) range-overlap lookup.
/// The executor builds this once with initial inputs (weights + user data),
/// then incrementally inserts span outputs after each phase and evicts
/// entries that no future phase will read.
///
/// Entries hold `NumericTensorCOW` rather than owned tensors so that initial
/// inputs (weights, user data) can flow through as zero-copy borrows from
/// caller-owned source tensors. Span outputs are always wrapped as
/// `Cow::Owned`. The `'a` lifetime parameter is the borrow lifetime for any
/// `Cow::Borrowed` entries; the `'p, P` parameters are the pool lifetime
/// and pool type for `Cow::Owned` entries.
pub struct PhaseStore<'a, 'p, P: Pool + 'p> {
    entries: Vec<StoreEntry<'a, 'p, P>>,
}

struct StoreEntry<'a, 'p, P: Pool + 'p> {
    base: u64, // AtomId.0
    tensor: NumericTensorCOW<'a, 'p, DynRank, P>,
}

impl<'a, 'p, P: Pool + 'p> PhaseStore<'a, 'p, P> {
    /// Create a store from initial inputs.
    pub fn new(inputs: Vec<(AtomId, NumericTensorCOW<'a, 'p, DynRank, P>)>) -> Self {
        let mut entries: Vec<StoreEntry<'a, 'p, P>> = inputs
            .into_iter()
            .map(|(base, tensor)| StoreEntry {
                base: base.0,
                tensor,
            })
            .collect();
        entries.sort_unstable_by_key(|e| e.base);
        PhaseStore { entries }
    }

    /// Number of entries in the store.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Insert an output tensor. Maintains sorted order.
    ///
    /// O(N) per call due to `Vec::insert` shifting. Prefer `insert_batch`
    /// when adding many entries at once (e.g. all outputs of a phase).
    pub fn insert(&mut self, base: AtomId, tensor: NumericTensorCOW<'a, 'p, DynRank, P>) {
        let pos = self.entries.partition_point(|e| e.base < base.0);
        // If an entry at this exact base exists, replace it.
        if pos < self.entries.len() && self.entries[pos].base == base.0 {
            self.entries[pos].tensor = tensor;
        } else {
            self.entries.insert(
                pos,
                StoreEntry {
                    base: base.0,
                    tensor,
                },
            );
        }
    }

    /// Insert many output tensors at once.
    ///
    /// Sorts the new entries by base, then performs a single linear merge
    /// against the existing sorted store. O(N + M log M) instead of the
    /// O(N*M) cost of M repeated `insert` calls (each shifting up to N
    /// `StoreEntry`s of ~96 bytes apiece).
    ///
    /// If multiple new entries share the same base, the **last** one wins,
    /// matching the per-element `insert` semantics. If a new entry's base
    /// matches an existing entry, the new entry replaces it.
    pub fn insert_batch(
        &mut self,
        new_entries: Vec<(AtomId, NumericTensorCOW<'a, 'p, DynRank, P>)>,
    ) {
        if new_entries.is_empty() {
            return;
        }

        // Stable sort so that, when two new entries share a base, the one
        // produced later in the input order ends up later in the sorted run
        // — the dedup pass below then keeps it.
        let mut sorted: Vec<StoreEntry<'a, 'p, P>> = new_entries
            .into_iter()
            .map(|(base, tensor)| StoreEntry {
                base: base.0,
                tensor,
            })
            .collect();
        sorted.sort_by_key(|e| e.base);

        // Collapse adjacent duplicates, keeping the last (latest insert wins).
        let mut deduped: Vec<StoreEntry<'a, 'p, P>> = Vec::with_capacity(sorted.len());
        for entry in sorted {
            if let Some(last) = deduped.last_mut() {
                if last.base == entry.base {
                    *last = entry;
                    continue;
                }
            }
            deduped.push(entry);
        }
        let sorted = deduped;

        // Linear merge of two sorted runs: existing self.entries and `sorted`.
        let existing = std::mem::take(&mut self.entries);
        let mut merged: Vec<StoreEntry<'a, 'p, P>> =
            Vec::with_capacity(existing.len() + sorted.len());
        let mut e_iter = existing.into_iter();
        let mut s_iter = sorted.into_iter();
        let mut e_cur = e_iter.next();
        let mut s_cur = s_iter.next();

        loop {
            match (e_cur.as_ref(), s_cur.as_ref()) {
                (Some(e), Some(s)) => match e.base.cmp(&s.base) {
                    std::cmp::Ordering::Less => {
                        merged.push(e_cur.take().unwrap());
                        e_cur = e_iter.next();
                    }
                    std::cmp::Ordering::Greater => {
                        merged.push(s_cur.take().unwrap());
                        s_cur = s_iter.next();
                    }
                    std::cmp::Ordering::Equal => {
                        // New replaces old.
                        let _ = e_cur.take();
                        merged.push(s_cur.take().unwrap());
                        e_cur = e_iter.next();
                        s_cur = s_iter.next();
                    }
                },
                (Some(_), None) => {
                    merged.push(e_cur.take().unwrap());
                    merged.extend(e_iter);
                    break;
                }
                (None, Some(_)) => {
                    merged.push(s_cur.take().unwrap());
                    merged.extend(s_iter);
                    break;
                }
                (None, None) => break,
            }
        }

        self.entries = merged;
    }

    /// Find all store entries overlapping [base, base+count).
    /// Returns zero-copy slices into the store's buffers.
    pub fn gather(&self, base: AtomId, count: u64) -> Vec<StoreSlice<'_>> {
        let range_lo = base.0;
        let range_hi = range_lo + count;

        // Binary search: find first entry that could overlap.
        // An entry at position `start-1` with base < range_lo could extend
        // past range_lo if it has enough elements.
        let search_start = self.entries.partition_point(|e| e.base < range_lo);
        let start = if search_start > 0 {
            search_start - 1
        } else {
            0
        };

        let mut slices = Vec::new();
        for entry in &self.entries[start..] {
            if entry.base >= range_hi {
                break;
            }
            let entry_count = entry.tensor.numel() as u64;
            let entry_hi = entry.base + entry_count;
            if entry_hi <= range_lo {
                continue;
            }

            // Compute overlap.
            let overlap_lo = entry.base.max(range_lo);
            let overlap_hi = entry_hi.min(range_hi);
            let skip = (overlap_lo - entry.base) as usize;
            let overlap_count = (overlap_hi - overlap_lo) as usize;
            let elem_bytes = entry.tensor.dtype().bytes_per_element();
            let byte_start = skip * elem_bytes;
            let byte_end = byte_start + overlap_count * elem_bytes;
            let buf = entry.tensor.buffer();

            if byte_end <= buf.len() {
                slices.push(StoreSlice {
                    base: AtomId(overlap_lo),
                    data: &buf[byte_start..byte_end],
                    dtype: entry.tensor.dtype(),
                    count: overlap_count as u64,
                });
            }
        }
        slices
    }

    /// Evict store entries that are no longer needed.
    ///
    /// `liveness` maps entry base AtomId to the last phase that reads it.
    /// Entries whose last-consumed phase is before `current_phase` are dropped.
    /// Entries overlapping any `pinned` range are never evicted.
    pub fn evict(
        &mut self,
        current_phase: usize,
        liveness: &HashMap<u64, usize>,
        pinned: &[(u64, u64)],
    ) {
        self.entries.retain(|entry| {
            // Check if this entry overlaps any pinned range.
            if !pinned.is_empty() {
                let entry_end = entry.base + entry.tensor.numel() as u64;
                let start = pinned.partition_point(|&(_, hi)| hi <= entry.base);
                for &(pin_lo, pin_hi) in &pinned[start..] {
                    if pin_lo >= entry_end {
                        break;
                    }
                    // Overlap found — keep.
                    return true;
                }
            }

            match liveness.get(&entry.base) {
                Some(&last_phase) => last_phase >= current_phase,
                // No liveness info → keep (e.g. initial inputs
                // that might be read in any phase).
                None => true,
            }
        });
    }

    /// Extract a tensor from the store by base AtomId.
    pub fn get(&self, base: AtomId) -> Option<&NumericTensorCOW<'a, 'p, DynRank, P>> {
        let pos = self.entries.partition_point(|e| e.base < base.0);
        if pos < self.entries.len() && self.entries[pos].base == base.0 {
            Some(&self.entries[pos].tensor)
        } else {
            None
        }
    }

    /// Iterate all entries (for output extraction).
    pub fn iter(&self) -> impl Iterator<Item = (AtomId, &NumericTensorCOW<'a, 'p, DynRank, P>)> {
        self.entries.iter().map(|e| (AtomId(e.base), &e.tensor))
    }

    /// Total bytes of data in all store entries.
    pub fn data_bytes(&self) -> usize {
        self.entries.iter().map(|e| e.tensor.buffer().len()).sum()
    }
}

// ─── Executable plan ────────────────────────────────────────────────────────

/// A fully compiled execution plan, ready to run.
///
/// Backend-agnostic — the executor doesn't know what produced the spans.
/// Owns the compiled span functions and drives the phase/barrier loop.
pub struct ExecutablePlan {
    phases: Vec<ExecutablePhase>,
    /// Liveness: maps output entry base → last phase that reads it.
    /// Used for store eviction after each phase.
    output_liveness: HashMap<u64, usize>,
    /// Atom ranges for model outputs that must never be evicted.
    /// Sorted by (lo, hi) for binary search.
    pinned_output_atoms: Vec<(u64, u64)>,
}

struct ExecutablePhase {
    lanes: Vec<ExecutableLane>,
}

struct ExecutableLane {
    span: Box<dyn CompiledSpanFn>,
    inputs: Vec<AtomRange>,
    outputs: Vec<AtomRange>,
}

/// Builder for constructing an ExecutablePlan from compiled spans.
pub struct ExecutablePlanBuilder {
    phases: Vec<ExecutablePhase>,
    /// All input ranges declared across all phases (for liveness computation).
    all_input_ranges: Vec<(usize, Vec<AtomRange>)>, // (phase_idx, ranges)
    /// Model-output atom ranges that must survive until after the last phase.
    /// These are pinned to the last phase during liveness computation so they
    /// aren't evicted prematurely.
    pinned_output_ranges: Vec<AtomRange>,
}

impl ExecutablePlanBuilder {
    pub fn new() -> Self {
        ExecutablePlanBuilder {
            phases: Vec::new(),
            all_input_ranges: Vec::new(),
            pinned_output_ranges: Vec::new(),
        }
    }

    /// Register model-output atom ranges that must survive until extraction.
    pub fn pin_outputs(&mut self, ranges: &[AtomRange]) {
        self.pinned_output_ranges.extend_from_slice(ranges);
    }

    /// Add a phase with one compiled span per lane.
    pub fn add_phase(
        &mut self,
        lanes: Vec<(Box<dyn CompiledSpanFn>, Vec<AtomRange>, Vec<AtomRange>)>,
    ) {
        let phase_idx = self.phases.len();
        let mut exec_lanes = Vec::with_capacity(lanes.len());
        let mut phase_inputs = Vec::new();
        for (span, inputs, outputs) in lanes {
            phase_inputs.extend(inputs.iter().cloned());
            exec_lanes.push(ExecutableLane {
                span,
                inputs,
                outputs,
            });
        }
        self.all_input_ranges.push((phase_idx, phase_inputs));
        self.phases.push(ExecutablePhase { lanes: exec_lanes });
    }

    /// Build the final plan, computing output liveness.
    ///
    /// For each output range produced by any phase, determines the last phase
    /// that reads any atom in that range. This handles group splitting correctly:
    /// a single input range [base, base+N) may overlap multiple output ranges
    /// at sub-range offsets.
    pub fn build(self) -> ExecutablePlan {
        // Collect all output ranges (base, end) sorted by base, for overlap queries.
        let mut all_outputs: Vec<(u64, u64)> = Vec::new();
        for phase in &self.phases {
            for lane in &phase.lanes {
                for out in &lane.outputs {
                    all_outputs.push((out.base.0, out.base.0 + out.count));
                }
            }
        }
        all_outputs.sort_unstable_by_key(|&(base, _)| base);
        all_outputs.dedup();

        let mut output_liveness: HashMap<u64, usize> = HashMap::new();

        for (phase_idx, inputs) in &self.all_input_ranges {
            for input in inputs {
                let in_lo = input.base.0;
                let in_hi = in_lo + input.count;

                // Register liveness for the input base itself (covers initial
                // inputs like weights whose base matches exactly).
                let entry = output_liveness.entry(in_lo).or_insert(0);
                *entry = (*entry).max(*phase_idx);

                // Find all output ranges that overlap [in_lo, in_hi) and
                // register liveness for their bases too.
                let start = all_outputs.partition_point(|&(_, end)| end <= in_lo);
                for &(out_base, _) in &all_outputs[start..] {
                    if out_base >= in_hi {
                        break;
                    }
                    let entry = output_liveness.entry(out_base).or_insert(0);
                    *entry = (*entry).max(*phase_idx);
                }
            }
        }

        // Build a set of atom ranges that are model outputs — these must
        // never be evicted regardless of liveness.
        let mut pinned_atoms: Vec<(u64, u64)> = self
            .pinned_output_ranges
            .iter()
            .map(|r| (r.base.0, r.base.0 + r.count))
            .collect();
        pinned_atoms.sort_unstable();

        let tracked = output_liveness.len();
        let total_outputs = all_outputs.len();
        eprintln!(
            "  Liveness: {} output ranges tracked of {} total",
            tracked, total_outputs,
        );

        ExecutablePlan {
            phases: self.phases,
            output_liveness,
            pinned_output_atoms: pinned_atoms,
        }
    }
}

impl ExecutablePlan {
    /// Execute the plan, returning the value store.
    pub fn execute<'a, 'p, P: Pool + 'p>(
        &self,
        initial_inputs: Vec<(AtomId, NumericTensorCOW<'a, 'p, DynRank, P>)>,
        pool: &'p P,
    ) -> PhaseStore<'a, 'p, P> {
        let mut store = PhaseStore::new(initial_inputs);

        for (pi, phase) in self.phases.iter().enumerate() {
            // Parallel: each lane gathers inputs, executes, produces outputs.
            let phase_outputs: Vec<Vec<(AtomId, NumericTensor<'p, DynRank, P>)>> = phase
                .lanes
                .par_iter()
                .map(|lane| execute_lane(lane, &store, pool))
                .collect();

            // Barrier: merge all outputs into store via a single batched
            // sort+linear-merge (much cheaper than per-element insert). Phase
            // outputs are JIT-allocated, owned tensors → wrap as Cow::Owned.
            let flat: Vec<(AtomId, NumericTensorCOW<'a, 'p, DynRank, P>)> = phase_outputs
                .into_iter()
                .flatten()
                .map(|(id, t)| (id, NumericTensorCOW::Owned(t)))
                .collect();
            store.insert_batch(flat);

            // Evict entries no longer needed by future phases.
            store.evict(pi + 1, &self.output_liveness, &self.pinned_output_atoms);
        }

        store
    }

    /// Execute with per-phase timing diagnostics.
    pub fn execute_timed<'a, 'p, P: Pool + 'p>(
        &self,
        initial_inputs: Vec<(AtomId, NumericTensorCOW<'a, 'p, DynRank, P>)>,
        pool: &'p P,
    ) -> PhaseStore<'a, 'p, P> {
        let mut store = PhaseStore::new(initial_inputs);
        let mut total_spans = std::time::Duration::ZERO;
        let mut total_merge = std::time::Duration::ZERO;
        let mut total_evict = std::time::Duration::ZERO;

        for (pi, phase) in self.phases.iter().enumerate() {
            let t0 = Instant::now();
            let phase_outputs: Vec<Vec<(AtomId, NumericTensor<'p, DynRank, P>)>> = phase
                .lanes
                .par_iter()
                .map(|lane| execute_lane(lane, &store, pool))
                .collect();
            let spans_dt = t0.elapsed();
            total_spans += spans_dt;

            let t0 = Instant::now();
            let flat: Vec<(AtomId, NumericTensorCOW<'a, 'p, DynRank, P>)> = phase_outputs
                .into_iter()
                .flatten()
                .map(|(id, t)| (id, NumericTensorCOW::Owned(t)))
                .collect();
            let n_outputs = flat.len();
            store.insert_batch(flat);
            let merge_dt = t0.elapsed();
            total_merge += merge_dt;

            let t0 = Instant::now();
            let store_before = store.len();
            store.evict(pi + 1, &self.output_liveness, &self.pinned_output_atoms);
            let evict_dt = t0.elapsed();
            total_evict += evict_dt;

            if spans_dt.as_millis() > 500 || pi < 3 || pi + 1 == self.phases.len() {
                let rss_mb = read_rss_mb();
                let store_mb = store.data_bytes() as f64 / (1024.0 * 1024.0);
                eprintln!(
                    "  phase {:>3}: spans={:.1}ms merge={:.1}ms evict={:.1}ms ({} out, store {} → {}, {:.0}MB data, RSS {:.0}MB)",
                    pi,
                    spans_dt.as_secs_f64() * 1e3,
                    merge_dt.as_secs_f64() * 1e3,
                    evict_dt.as_secs_f64() * 1e3,
                    n_outputs,
                    store_before,
                    store.len(),
                    store_mb,
                    rss_mb,
                );
            }
        }

        eprintln!(
            "  TOTALS: spans={:.1}ms merge={:.1}ms evict={:.1}ms, final store={}",
            total_spans.as_secs_f64() * 1e3,
            total_merge.as_secs_f64() * 1e3,
            total_evict.as_secs_f64() * 1e3,
            store.len(),
        );

        store
    }

    /// Number of phases.
    pub fn num_phases(&self) -> usize {
        self.phases.len()
    }
}

/// Execute a single lane: gather inputs, run span, return outputs.
fn execute_lane<'a, 'p, P: Pool + 'p>(
    lane: &ExecutableLane,
    store: &PhaseStore<'a, 'p, P>,
    pool: &'p P,
) -> Vec<(AtomId, NumericTensor<'p, DynRank, P>)> {
    // Gather all input slices for this lane's declared input ranges.
    let input_slices: Vec<StoreSlice<'_>> = lane
        .inputs
        .iter()
        .flat_map(|range| store.gather(range.base, range.count))
        .collect();

    // Pre-allocate output tensors with byte-aligned layout.
    // JIT writes one byte per sub-byte element (e.g. BOOL), so use
    // bytes_per_element for stride rather than total_bits.
    let mut output_tensors: Vec<NumericTensor<'p, DynRank, P>> = lane
        .outputs
        .iter()
        .map(|r| {
            let layout = jit_flat_layout(r.count, r.dtype);
            let buf = pool
                .allocate(layout.buffer_size_bytes())
                .expect("failed to allocate output tensor");
            NumericTensor::from_parts(buf, layout)
        })
        .collect();

    // Create mutable byte views for the span.
    {
        let mut span_outputs: Vec<SpanOutput<'_>> = output_tensors
            .iter_mut()
            .zip(lane.outputs.iter())
            .map(|(t, r)| SpanOutput {
                data: t.buffer_mut(),
                dtype: r.dtype,
                count: r.count,
            })
            .collect();

        lane.span.execute(&input_slices, &mut span_outputs);
    }

    // Return (base, tensor) pairs.
    lane.outputs
        .iter()
        .zip(output_tensors)
        .map(|(range, tensor)| (range.base, tensor))
        .collect()
}

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Create a flat 1D tensor layout with byte-aligned element strides.
///
/// The JIT writes one byte per element for sub-byte dtypes (e.g. BOOL), so the
/// stride must be `bytes_per_element * 8` bits, not `total_bits`. For types ≥ 8
/// bits this is identical to `row_major`.
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
