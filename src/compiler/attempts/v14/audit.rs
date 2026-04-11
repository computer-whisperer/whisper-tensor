#![allow(clippy::all, dead_code, unused)]
//! Cross-span slab coalescing audit.
//!
//! Walks the partitioned `Vec<Phase>` and counts how often an InputRef's
//! access pattern would force slab coalescing — and whether those slabs
//! would pull atoms from multiple buffer kinds under the memory-placement
//! rework.
//!
//! See `MEMORY_PLACEMENT.md` §"Algorithm sketch" step 2 for context. This
//! is the instrumentation that answers the question: does the global
//! placer actually need cross-buffer coalescing, or is the per-span
//! matmul case the only coalescing that fires in practice?
//!
//! Enable with `WT_AUDIT_SLABS=1` at compile time.

use std::collections::{HashMap, HashSet};

use crate::nano_graph::pattern::{AtomRange, InputRef, NanoGraph};
use crate::pool::SystemPool;

use super::partitioner_m::input_access_segments;
use super::types::Phase;

/// Which kind of buffer a main-graph item would live in under the new
/// memory placement scheme.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BufferKind {
    /// Model input (weight or user input) — lives in a dedicated input
    /// buffer.
    Input,
    /// Model output — lives in a dedicated output buffer.
    Output,
    /// Cross-span intermediate — lives in the shared intermediate buffer.
    Intermediate,
    /// Span-local scratch — lives in per-lane scratch, invisible to the
    /// global placer.
    Scratch,
}

/// Classification of a single coalescing constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoalescingClass {
    /// All members would live in per-lane scratch. The global placer
    /// doesn't see these — they're handled by the per-span layout.
    AllScratch,
    /// All members come from input buffers (e.g., weight ↔ weight).
    AllInput,
    /// All members come from the intermediate buffer (pure cross-span).
    AllIntermediate,
    /// All members are model outputs.
    AllOutput,
    /// Mixed Input + Intermediate: the "cross-buffer" case the design
    /// doc flagged. If this fires, the global placer can't coalesce
    /// these into one contiguous slab without copying weight bytes
    /// into the intermediate buffer at startup.
    MixedInputIntermediate,
    /// Some other combination (e.g., Intermediate + Output, Input +
    /// Output, or three-way). Rare in practice but worth tracking.
    OtherMixed,
}

/// Summary of what the audit found.
#[derive(Debug, Default)]
pub struct SlabAuditReport {
    pub total_input_refs: usize,
    /// Count of InputRefs whose access range touches ≥2 main-graph items.
    pub coalescing_refs: usize,
    pub all_scratch: usize,
    pub all_input: usize,
    pub all_intermediate: usize,
    pub all_output: usize,
    pub mixed_input_intermediate: usize,
    pub other_mixed: usize,
    /// First few examples of all-intermediate constraints (sanity check).
    pub intermediate_examples: Vec<String>,
    /// First few examples of mixed-buffer constraints, for manual inspection.
    pub mixed_examples: Vec<String>,
}

impl SlabAuditReport {
    pub fn print(&self) {
        eprintln!("[placer audit] cross-span slab coalescing");
        eprintln!(
            "  total InputRefs scanned:        {}",
            self.total_input_refs
        );
        eprintln!(
            "  InputRefs with >1 item (coalescing): {}",
            self.coalescing_refs
        );
        if self.coalescing_refs == 0 {
            return;
        }
        let pct = |n: usize| (n as f64 / self.coalescing_refs as f64) * 100.0;
        eprintln!("  classification of coalescing InputRefs:");
        eprintln!(
            "    all-scratch    (span-local, handled today):  {:>6}  ({:>5.1}%)",
            self.all_scratch,
            pct(self.all_scratch),
        );
        eprintln!(
            "    all-input      (weight-only coalescing):     {:>6}  ({:>5.1}%)",
            self.all_input,
            pct(self.all_input),
        );
        eprintln!(
            "    all-intermediate (pure cross-span):          {:>6}  ({:>5.1}%)",
            self.all_intermediate,
            pct(self.all_intermediate),
        );
        eprintln!(
            "    all-output     (model output only):          {:>6}  ({:>5.1}%)",
            self.all_output,
            pct(self.all_output),
        );
        eprintln!(
            "    mixed input+intermediate (cross-buffer):     {:>6}  ({:>5.1}%)  <-- problem case",
            self.mixed_input_intermediate,
            pct(self.mixed_input_intermediate),
        );
        eprintln!(
            "    other mixed:                                 {:>6}  ({:>5.1}%)",
            self.other_mixed,
            pct(self.other_mixed),
        );
        if !self.mixed_examples.is_empty() {
            eprintln!("  examples of mixed-buffer constraints:");
            for ex in self.mixed_examples.iter().take(10) {
                eprintln!("    {ex}");
            }
            if self.mixed_examples.len() > 10 {
                eprintln!("    ... ({} more)", self.mixed_examples.len() - 10);
            }
        }
        if !self.intermediate_examples.is_empty() {
            eprintln!("  sample all-intermediate constraints:");
            for ex in self.intermediate_examples.iter().take(4) {
                eprintln!("    {ex}");
            }
        }
    }
}

/// Reference to a main-graph item — either a model input tensor or a group.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum ItemRef {
    /// Index into `graph.input_tensors()`.
    Input(usize),
    /// Index into `graph.groups()`.
    Group(usize),
}

/// Run the audit over a partitioned model.
pub fn audit_slab_coalescing(
    main_graph: &NanoGraph<'static, SystemPool>,
    phases: &[Phase],
    all_output_atom_ranges: &[AtomRange],
) -> SlabAuditReport {
    let mut report = SlabAuditReport::default();

    let groups = main_graph.groups();
    let input_tensors = main_graph.input_tensors();
    if groups.is_empty() {
        return report;
    }

    // ── Step 1: classify every main-graph group by buffer kind ──────────

    // Priority: Output > Intermediate > Scratch. (Input kind only applies
    // to input tensors, which are handled separately.)
    let mut group_kind: Vec<BufferKind> = vec![BufferKind::Scratch; groups.len()];

    // Mark outputs. A model output pins an atom range and is always its
    // own buffer under the new design.
    for range in all_output_atom_ranges {
        // Walk every group whose atoms overlap the output range. A single
        // range may cover multiple contiguous groups (e.g. Pad lowering).
        let r_lo = range.base.0;
        let r_hi = r_lo + range.count;
        for (gi, g) in groups.iter().enumerate() {
            let g_lo = g.base_id.0;
            let g_hi = g_lo + g.count;
            if g_lo < r_hi && r_lo < g_hi {
                group_kind[gi] = BufferKind::Output;
            }
        }
    }

    // Mark intermediates: any group whose atoms appear in some span's
    // `outputs` list, meaning they flow between spans. Don't overwrite
    // Output.
    for phase in phases {
        for span in &phase.spans {
            for range in &span.outputs {
                let r_lo = range.base.0;
                let r_hi = r_lo + range.count;
                for (gi, g) in groups.iter().enumerate() {
                    if group_kind[gi] != BufferKind::Scratch {
                        continue;
                    }
                    let g_lo = g.base_id.0;
                    let g_hi = g_lo + g.count;
                    if g_lo < r_hi && r_lo < g_hi {
                        group_kind[gi] = BufferKind::Intermediate;
                    }
                }
            }
        }
    }

    // ── Step 2: walk every InputRef in every span ───────────────────────

    // Pre-sort items by start atom-id for binary search.
    // (atom_lo, atom_hi, item_ref)
    let mut sorted_items: Vec<(u64, u64, ItemRef)> = Vec::new();
    for (ii, it) in input_tensors.iter().enumerate() {
        sorted_items.push((it.base_id.0, it.base_id.0 + it.count, ItemRef::Input(ii)));
    }
    for (gi, g) in groups.iter().enumerate() {
        sorted_items.push((g.base_id.0, g.base_id.0 + g.count, ItemRef::Group(gi)));
    }
    sorted_items.sort_unstable_by_key(|&(lo, _, _)| lo);

    let item_kind = |item: ItemRef| -> BufferKind {
        match item {
            ItemRef::Input(_) => BufferKind::Input,
            ItemRef::Group(gi) => group_kind[gi],
        }
    };

    for (pi, phase) in phases.iter().enumerate() {
        for (li, span) in phase.spans.iter().enumerate() {
            for group in span.graph.groups() {
                for (ii, input_ref) in group.inputs.iter().enumerate() {
                    report.total_input_refs += 1;

                    let segments = input_access_segments(input_ref, group.atom_offset, group.count);
                    if segments.is_empty() {
                        continue;
                    }

                    // Find all main-graph items whose atom range overlaps
                    // any segment.
                    let mut items: HashSet<ItemRef> = HashSet::new();
                    for &(seg_lo, seg_hi) in &segments {
                        if seg_lo >= seg_hi {
                            continue;
                        }
                        // Binary search for the first item whose end > seg_lo.
                        let start_idx = sorted_items.partition_point(|&(_, hi, _)| hi <= seg_lo);
                        for &(item_lo, item_hi, item) in &sorted_items[start_idx..] {
                            if item_lo >= seg_hi {
                                break;
                            }
                            if item_hi > seg_lo && item_lo < seg_hi {
                                items.insert(item);
                            }
                        }
                    }

                    if items.len() <= 1 {
                        continue;
                    }

                    report.coalescing_refs += 1;

                    // Classify.
                    let mut kinds: HashSet<BufferKind> = HashSet::new();
                    for &item in &items {
                        kinds.insert(item_kind(item));
                    }

                    let class = classify(&kinds);
                    match class {
                        CoalescingClass::AllScratch => report.all_scratch += 1,
                        CoalescingClass::AllInput => report.all_input += 1,
                        CoalescingClass::AllIntermediate => {
                            report.all_intermediate += 1;
                            if report.intermediate_examples.len() < 8 {
                                report.intermediate_examples.push(format_example(
                                    pi,
                                    li,
                                    group.base_id.0,
                                    ii,
                                    &segments,
                                    &items,
                                    &item_kind,
                                    main_graph,
                                ));
                            }
                        }
                        CoalescingClass::AllOutput => report.all_output += 1,
                        CoalescingClass::MixedInputIntermediate => {
                            report.mixed_input_intermediate += 1;
                            if report.mixed_examples.len() < 16 {
                                report.mixed_examples.push(format_example(
                                    pi,
                                    li,
                                    group.base_id.0,
                                    ii,
                                    &segments,
                                    &items,
                                    &item_kind,
                                    main_graph,
                                ));
                            }
                        }
                        CoalescingClass::OtherMixed => {
                            report.other_mixed += 1;
                            if report.mixed_examples.len() < 16 {
                                report.mixed_examples.push(format_example(
                                    pi,
                                    li,
                                    group.base_id.0,
                                    ii,
                                    &segments,
                                    &items,
                                    &item_kind,
                                    main_graph,
                                ));
                            }
                        }
                    }
                }
            }
        }
    }

    report
}

fn classify(kinds: &HashSet<BufferKind>) -> CoalescingClass {
    let has_input = kinds.contains(&BufferKind::Input);
    let has_intermediate = kinds.contains(&BufferKind::Intermediate);
    let has_output = kinds.contains(&BufferKind::Output);
    let has_scratch = kinds.contains(&BufferKind::Scratch);

    let n = kinds.len();
    if n == 1 {
        return match kinds.iter().next().unwrap() {
            BufferKind::Scratch => CoalescingClass::AllScratch,
            BufferKind::Input => CoalescingClass::AllInput,
            BufferKind::Intermediate => CoalescingClass::AllIntermediate,
            BufferKind::Output => CoalescingClass::AllOutput,
        };
    }

    // Flag the specific cross-buffer case we care most about.
    if has_input && has_intermediate && !has_output {
        // Scratch mixed in is fine — scratch items will be per-span so
        // they don't force global-placer coalescing. But if Input and
        // Intermediate co-occur, the global placer can't put them in
        // one buffer without copying.
        return CoalescingClass::MixedInputIntermediate;
    }

    CoalescingClass::OtherMixed
}

fn format_example(
    phase_idx: usize,
    lane_idx: usize,
    consumer_group_base: u64,
    input_idx: usize,
    segments: &[(u64, u64)],
    items: &HashSet<ItemRef>,
    item_kind: &impl Fn(ItemRef) -> BufferKind,
    main_graph: &NanoGraph<'static, SystemPool>,
) -> String {
    let seg_str = segments
        .iter()
        .map(|&(lo, hi)| format!("[{lo},{hi})"))
        .collect::<Vec<_>>()
        .join(" ∪ ");
    let mut members: Vec<(u64, String)> = items
        .iter()
        .map(|&item| {
            let (lo, name) = match item {
                ItemRef::Input(ii) => {
                    let it = &main_graph.input_tensors()[ii];
                    (it.base_id.0, format!("In#{ii}@{}", it.base_id.0))
                }
                ItemRef::Group(gi) => {
                    let g = &main_graph.groups()[gi];
                    (g.base_id.0, format!("G#{gi}@{}", g.base_id.0))
                }
            };
            let kind = item_kind(item);
            (lo, format!("{name}:{kind:?}"))
        })
        .collect();
    members.sort_by_key(|m| m.0);
    let members_str = members
        .into_iter()
        .map(|(_, s)| s)
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "phase {phase_idx} lane {lane_idx} consumer g@{consumer_group_base} input #{input_idx} access={seg_str} members={{{members_str}}}"
    )
}
