#![allow(clippy::all, dead_code, unused_variables)]

//! Phase-coalescing partitioner (attempt N).
//!
//! Starts from `partitioner_m` and greedily merges adjacent phases when the
//! merged phase still validates and preserves cross-lane independence.
//!
//! The merge pass is intentionally conservative:
//! - If merge synthesis fails for any lane, the pair is left unchanged.
//! - If merged span validation fails, the pair is left unchanged.
//! - If merged phase cross-lane checks fail, the pair is left unchanged.
//!
//! This keeps behavior safe while reducing barrier count where `partitioner_m`
//! was over-conservative.

use std::collections::{HashSet, VecDeque};

use crate::graph::GlobalId;
use crate::nano_graph::pattern::InputTensor;
use crate::nano_graph::{AtomRange, NanoGraph};
use crate::numeric_dtype::NumericDType;

use super::partitioner_m;
use super::types::{Phase, Span};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct InputEntry {
    lo: u64,
    hi: u64,
    dtype: NumericDType,
    tensor_id: GlobalId,
}

/// Plan with `partitioner_m`, then greedily coalesce adjacent phases.
pub fn plan(
    graph: &NanoGraph<'static, crate::pool::SystemPool>,
    num_lanes: usize,
    input_tensors: &[InputTensor],
    output_atom_ids: &[AtomRange],
) -> Vec<Phase> {
    let base = partitioner_m::plan(graph, num_lanes, input_tensors, output_atom_ids);
    coalesce_adjacent_phases(base)
}

fn coalesce_adjacent_phases(phases: Vec<Phase>) -> Vec<Phase> {
    let mut queue: VecDeque<Phase> = phases.into();
    let mut out = Vec::new();

    while let Some(first) = queue.pop_front() {
        if let Some(second) = queue.front()
            && let Some(merged) = try_merge_phases(&first, second)
        {
            // Consume the second phase and re-attempt merging with the next
            // phase by pushing the merged candidate back to the front.
            let _ = queue.pop_front();
            queue.push_front(merged);
            continue;
        }
        out.push(first);
    }

    out
}

fn try_merge_phases(a: &Phase, b: &Phase) -> Option<Phase> {
    if a.spans.len() != b.spans.len() {
        return None;
    }

    let mut spans = Vec::with_capacity(a.spans.len());
    for lane in 0..a.spans.len() {
        let merged = build_merged_span(&a.spans[lane], &b.spans[lane])?;
        spans.push(merged);
    }

    let merged = Phase { spans };
    if !validate_phase_cross_lane(&merged) {
        return None;
    }
    Some(merged)
}

fn build_merged_span(a: &Span, b: &Span) -> Option<Span> {
    let mut graph = NanoGraph::new();
    graph.graph_constants = a.graph.graph_constants.clone();
    graph.set_opaque_ops(a.graph.opaque_ops().to_vec());

    let mut groups: Vec<_> = a
        .graph
        .groups()
        .iter()
        .chain(b.graph.groups().iter())
        .collect();
    groups.sort_by_key(|g| g.base_id.0);
    for g in groups {
        graph.insert_group_at(
            g.base_id,
            g.count,
            g.atom_offset,
            g.output_dtype,
            g.op.clone(),
            g.sym_dims.clone(),
            g.inputs.clone(),
        );
    }

    let coverage = collect_group_coverage(&graph);

    let mut entries = gather_input_entries(a);
    entries.extend(gather_input_entries(b));
    entries.retain(|e| !is_range_fully_covered(e.lo, e.hi, &coverage));
    let entries = normalize_input_entries(entries)?;

    let mut inputs = Vec::with_capacity(entries.len());
    for entry in entries {
        let count = entry.hi.saturating_sub(entry.lo);
        if count == 0 {
            continue;
        }
        graph.insert_input_tensor_at_allow_overlap(
            crate::nano_graph::AtomId(entry.lo),
            entry.tensor_id,
            count,
            entry.dtype,
        );
        inputs.push(AtomRange {
            base: crate::nano_graph::AtomId(entry.lo),
            count,
            dtype: entry.dtype,
        });
    }

    if !graph.validate().is_empty() {
        return None;
    }

    Some(Span {
        graph,
        inputs,
        outputs: dedup_atom_ranges(a.outputs.iter().chain(b.outputs.iter())),
    })
}

fn gather_input_entries(span: &Span) -> Vec<InputEntry> {
    let mut out = Vec::with_capacity(span.inputs.len());
    for range in &span.inputs {
        let lo = range.base.0;
        let hi = lo.saturating_add(range.count);
        if lo >= hi {
            continue;
        }
        let tensor_id = span
            .graph
            .find_input_idx_by_base(range.base)
            .or_else(|| span.graph.find_input_idx(range.base).map(|(idx, _)| idx))
            .map(|idx| span.graph.input_tensors()[idx].tensor_id)
            .unwrap_or(GlobalId(0));
        out.push(InputEntry {
            lo,
            hi,
            dtype: range.dtype,
            tensor_id,
        });
    }
    out
}

fn normalize_input_entries(mut entries: Vec<InputEntry>) -> Option<Vec<InputEntry>> {
    entries.sort_by(|a, b| {
        a.lo.cmp(&b.lo)
            .then(a.hi.cmp(&b.hi))
            .then(a.tensor_id.cmp(&b.tensor_id))
    });

    let mut merged: Vec<InputEntry> = Vec::with_capacity(entries.len());
    for entry in entries {
        if let Some(last) = merged.last_mut() {
            if entry.lo < last.hi {
                if entry.tensor_id == last.tensor_id && entry.dtype == last.dtype {
                    last.hi = last.hi.max(entry.hi);
                    continue;
                }
                // Different metadata can still be represented now that span
                // input ranges support overlap.
            }
            if entry.lo == last.hi && entry.tensor_id == last.tensor_id && entry.dtype == last.dtype
            {
                last.hi = entry.hi;
                continue;
            }
        }
        merged.push(entry);
    }
    Some(merged)
}

fn collect_group_coverage(graph: &NanoGraph<'static, crate::pool::SystemPool>) -> Vec<(u64, u64)> {
    let mut ranges: Vec<(u64, u64)> = graph
        .groups()
        .iter()
        .map(|g| (g.base_id.0, g.base_id.0 + g.count))
        .collect();
    ranges.sort_by_key(|(lo, _)| *lo);
    ranges
}

fn is_range_fully_covered(lo: u64, hi: u64, coverage: &[(u64, u64)]) -> bool {
    if lo >= hi {
        return true;
    }
    let mut cursor = lo;
    for &(seg_lo, seg_hi) in coverage {
        if seg_hi <= cursor {
            continue;
        }
        if seg_lo > cursor {
            return false;
        }
        cursor = cursor.max(seg_hi);
        if cursor >= hi {
            return true;
        }
    }
    false
}

fn dedup_atom_ranges<'a>(ranges: impl Iterator<Item = &'a AtomRange>) -> Vec<AtomRange> {
    let mut seen: HashSet<(u64, u64, NumericDType)> = HashSet::new();
    let mut out = Vec::new();
    for r in ranges {
        let key = (r.base.0, r.count, r.dtype);
        if seen.insert(key) {
            out.push(r.clone());
        }
    }
    out.sort_by_key(|r| (r.base.0, r.count));
    out
}

/// Validate that no span input overlaps atoms produced by a different lane in
/// the same phase.
fn validate_phase_cross_lane(phase: &Phase) -> bool {
    let span_produces: Vec<Vec<(u64, u64)>> = phase
        .spans
        .iter()
        .map(|s| {
            s.graph
                .groups()
                .iter()
                .map(|g| (g.base_id.0, g.base_id.0 + g.count))
                .collect()
        })
        .collect();

    for (lane, span) in phase.spans.iter().enumerate() {
        for input in &span.inputs {
            let inp_lo = input.base.0;
            let inp_hi = inp_lo + input.count;
            for (other_lane, produced) in span_produces.iter().enumerate() {
                if lane == other_lane {
                    continue;
                }
                for &(prod_lo, prod_hi) in produced {
                    if inp_lo < prod_hi && prod_lo < inp_hi {
                        return false;
                    }
                }
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nano_graph::ops::ScalarBinOp;
    use crate::nano_graph::pattern::GroupInput;
    use crate::nano_graph::{InputRef, ScalarOp};
    use crate::numeric_dtype::NumericDType;
    use crate::numeric_scalar::NumericScalar;

    #[test]
    fn test_coalescer_preserves_validity_on_linear_chain() {
        let mut g = NanoGraph::new();
        let lit = g.push_group(
            4096,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(1.0)),
            vec![],
            vec![],
        );
        let mul = g.push_group(
            4096,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![
                GroupInput::scalar(InputRef::affine(lit, 1)),
                GroupInput::scalar(InputRef::affine(lit, 1)),
            ],
        );
        let add = g.push_group(
            4096,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![
                GroupInput::scalar(InputRef::affine(mul, 1)),
                GroupInput::scalar(InputRef::affine(lit, 1)),
            ],
        );
        g.outputs = vec![g.atom_to_range(add)];

        let m = partitioner_m::plan(&g, 4, &[], &g.outputs.clone());
        let n = plan(&g, 4, &[], &g.outputs.clone());

        assert!(n.len() <= m.len());
        for phase in &n {
            assert!(validate_phase_cross_lane(phase));
            for span in &phase.spans {
                assert!(
                    span.graph.validate().is_empty(),
                    "span validation failed: {:?}",
                    span.graph.validate()
                );
            }
        }
    }
}
