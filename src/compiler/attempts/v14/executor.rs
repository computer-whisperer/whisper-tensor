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

use crate::dtype::DType;
use crate::nano_graph::{AtomId, AtomRange};

// ─── Inter-phase data types ─────────────────────────────────────────────────

/// Contiguous typed byte buffer — the universal inter-phase data format.
///
/// Zero-overhead for JIT backends (already byte buffers).
/// Cheap to wrap for interpreter backends (convert NDArray ↔ raw bytes).
#[derive(Clone)]
pub struct TypedBuffer {
    pub data: Vec<u8>,
    pub dtype: DType,
    pub count: u64,
}

impl TypedBuffer {
    /// Create a zeroed buffer for `count` elements of `dtype`.
    pub fn zeroed(dtype: DType, count: u64) -> Self {
        let elem_bytes = dtype_elem_bytes(dtype);
        TypedBuffer {
            data: vec![0u8; count as usize * elem_bytes],
            dtype,
            count,
        }
    }

    /// Byte size per element.
    pub fn elem_bytes(&self) -> usize {
        dtype_elem_bytes(self.dtype)
    }
}

/// A zero-copy view into a store entry's data.
///
/// Spans receive these for their declared input ranges. The data pointer
/// is valid for the duration of the phase (store is immutably borrowed).
pub struct StoreSlice<'a> {
    pub base: AtomId,
    pub data: &'a [u8],
    pub dtype: DType,
    pub count: u64,
}

// ─── Compiled span trait ────────────────────────────────────────────────────

/// A prepared span that can execute given input data.
///
/// Backend-agnostic: JIT, interpreter, and GPU all implement this.
/// The executor calls this once per span per phase, passing zero-copy
/// slices from the store and pre-allocated output buffers.
pub trait CompiledSpanFn: Send + Sync {
    /// Execute the span.
    ///
    /// `inputs` contains store slices covering all atoms this span reads
    /// from prior phases (weights, user inputs, prior-phase outputs).
    /// The slices may not exactly match the span's declared input ranges —
    /// they're the overlapping store entries found by the executor.
    ///
    /// `outputs` contains pre-allocated TypedBuffers, one per declared
    /// output range (same order as the span's output declarations).
    /// The implementation fills these with computed values.
    fn execute(&self, inputs: &[StoreSlice<'_>], outputs: &mut [TypedBuffer]);
}

// ─── Phase store ────────────────────────────────────────────────────────────

/// Inter-phase value store with liveness-based eviction.
///
/// Entries are sorted by base AtomId for O(log N) range-overlap lookup.
/// The executor builds this once with initial inputs (weights + user data),
/// then incrementally inserts span outputs after each phase and evicts
/// entries that no future phase will read.
pub struct PhaseStore {
    entries: Vec<StoreEntry>,
}

struct StoreEntry {
    base: u64, // AtomId.0
    buffer: TypedBuffer,
}

impl PhaseStore {
    /// Create a store from initial inputs.
    pub fn new(inputs: Vec<(AtomId, TypedBuffer)>) -> Self {
        let mut entries: Vec<StoreEntry> = inputs
            .into_iter()
            .map(|(base, buffer)| StoreEntry {
                base: base.0,
                buffer,
            })
            .collect();
        entries.sort_unstable_by_key(|e| e.base);
        PhaseStore { entries }
    }

    /// Number of entries in the store.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Insert an output buffer. Maintains sorted order.
    pub fn insert(&mut self, base: AtomId, buffer: TypedBuffer) {
        let pos = self.entries.partition_point(|e| e.base < base.0);
        // If an entry at this exact base exists, replace it.
        if pos < self.entries.len() && self.entries[pos].base == base.0 {
            self.entries[pos].buffer = buffer;
        } else {
            self.entries.insert(
                pos,
                StoreEntry {
                    base: base.0,
                    buffer,
                },
            );
        }
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
            let entry_hi = entry.base + entry.buffer.count;
            if entry_hi <= range_lo {
                continue;
            }

            // Compute overlap.
            let overlap_lo = entry.base.max(range_lo);
            let overlap_hi = entry_hi.min(range_hi);
            let skip = (overlap_lo - entry.base) as usize;
            let overlap_count = (overlap_hi - overlap_lo) as usize;
            let elem_bytes = entry.buffer.elem_bytes();
            let byte_start = skip * elem_bytes;
            let byte_end = byte_start + overlap_count * elem_bytes;

            if byte_end <= entry.buffer.data.len() {
                slices.push(StoreSlice {
                    base: AtomId(overlap_lo),
                    data: &entry.buffer.data[byte_start..byte_end],
                    dtype: entry.buffer.dtype,
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
    pub fn evict(&mut self, current_phase: usize, liveness: &HashMap<u64, usize>) {
        self.entries.retain(|entry| {
            match liveness.get(&entry.base) {
                Some(&last_phase) => last_phase >= current_phase,
                // No liveness info → keep (e.g. model outputs, initial inputs
                // that might be read in any phase).
                None => true,
            }
        });
    }

    /// Extract a value from the store by base AtomId.
    pub fn get(&self, base: AtomId) -> Option<&TypedBuffer> {
        let pos = self.entries.partition_point(|e| e.base < base.0);
        if pos < self.entries.len() && self.entries[pos].base == base.0 {
            Some(&self.entries[pos].buffer)
        } else {
            None
        }
    }

    /// Iterate all entries (for output extraction).
    pub fn iter(&self) -> impl Iterator<Item = (AtomId, &TypedBuffer)> {
        self.entries.iter().map(|e| (AtomId(e.base), &e.buffer))
    }

    /// Total bytes of data in all store entries.
    pub fn data_bytes(&self) -> usize {
        self.entries.iter().map(|e| e.buffer.data.len()).sum()
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
}

impl ExecutablePlanBuilder {
    pub fn new() -> Self {
        ExecutablePlanBuilder {
            phases: Vec::new(),
            all_input_ranges: Vec::new(),
        }
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

        let tracked = output_liveness.len();
        let total_outputs = all_outputs.len();
        eprintln!(
            "  Liveness: {} output ranges tracked of {} total",
            tracked, total_outputs
        );

        ExecutablePlan {
            phases: self.phases,
            output_liveness,
        }
    }
}

impl ExecutablePlan {
    /// Execute the plan, returning the value store.
    pub fn execute(&self, initial_inputs: Vec<(AtomId, TypedBuffer)>) -> PhaseStore {
        let mut store = PhaseStore::new(initial_inputs);

        for (pi, phase) in self.phases.iter().enumerate() {
            // Parallel: each lane gathers inputs, executes, produces outputs.
            let phase_outputs: Vec<Vec<(AtomId, TypedBuffer)>> = phase
                .lanes
                .par_iter()
                .map(|lane| execute_lane(lane, &store))
                .collect();

            // Barrier: merge all outputs into store.
            for lane_outputs in phase_outputs {
                for (base, buffer) in lane_outputs {
                    store.insert(base, buffer);
                }
            }

            // Evict entries no longer needed by future phases.
            store.evict(pi + 1, &self.output_liveness);
        }

        store
    }

    /// Execute with per-phase timing diagnostics.
    pub fn execute_timed(&self, initial_inputs: Vec<(AtomId, TypedBuffer)>) -> PhaseStore {
        let mut store = PhaseStore::new(initial_inputs);
        let mut total_spans = std::time::Duration::ZERO;
        let mut total_merge = std::time::Duration::ZERO;
        let mut total_evict = std::time::Duration::ZERO;

        for (pi, phase) in self.phases.iter().enumerate() {
            let t0 = Instant::now();
            let phase_outputs: Vec<Vec<(AtomId, TypedBuffer)>> = phase
                .lanes
                .par_iter()
                .map(|lane| execute_lane(lane, &store))
                .collect();
            let spans_dt = t0.elapsed();
            total_spans += spans_dt;

            let t0 = Instant::now();
            let mut n_outputs = 0usize;
            for lane_outputs in phase_outputs {
                n_outputs += lane_outputs.len();
                for (base, buffer) in lane_outputs {
                    store.insert(base, buffer);
                }
            }
            let merge_dt = t0.elapsed();
            total_merge += merge_dt;

            let t0 = Instant::now();
            let store_before = store.len();
            store.evict(pi + 1, &self.output_liveness);
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
fn execute_lane(lane: &ExecutableLane, store: &PhaseStore) -> Vec<(AtomId, TypedBuffer)> {
    // Gather all input slices for this lane's declared input ranges.
    let input_slices: Vec<StoreSlice<'_>> = lane
        .inputs
        .iter()
        .flat_map(|range| store.gather(range.base, range.count))
        .collect();

    // Pre-allocate output buffers.
    let mut outputs: Vec<TypedBuffer> = lane
        .outputs
        .iter()
        .map(|r| TypedBuffer::zeroed(r.dtype, r.count))
        .collect();

    // Execute the span.
    lane.span.execute(&input_slices, &mut outputs);

    // Return (base, buffer) pairs.
    lane.outputs
        .iter()
        .zip(outputs)
        .map(|(range, buf)| (range.base, buf))
        .collect()
}

// ─── Helpers ────────────────────────────────────────────────────────────────

fn read_rss_mb() -> f64 {
    std::fs::read_to_string("/proc/self/statm")
        .ok()
        .and_then(|s| s.split_whitespace().nth(1)?.parse::<u64>().ok())
        .unwrap_or(0) as f64
        * 4096.0
        / (1024.0 * 1024.0)
}

pub fn dtype_elem_bytes(dtype: DType) -> usize {
    match dtype {
        DType::F64 | DType::I64 | DType::U64 => 8,
        DType::F32 | DType::I32 | DType::U32 => 4,
        DType::BF16 | DType::F16 | DType::I16 | DType::U16 => 2,
        DType::I8 | DType::U8 | DType::BOOL => 1,
        _ => 4, // fallback
    }
}
