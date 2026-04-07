#![allow(clippy::all, dead_code, unreachable_patterns, unused)]
//! Reporting and visualization for execution plans.
//!
//! Generates textual reports and SVG visualizations showing how milli ops
//! map through nano lowering into partitioned phases and spans.

use std::collections::HashMap;
use std::fmt::Write;

use crate::graph::GlobalId;
use crate::nano_graph::{AtomId, NanoGraph, ScalarOp};

use super::types::*;

/// Provenance map: for each nano group index in the main graph, the milli op
/// that produced it (milli op GlobalId + op kind string).
pub type GroupProvenance = Vec<(GlobalId, String)>;

// ─── Textual report ─────────────────────────────────────────────────────────

/// Generate a textual report of an execution plan.
///
/// Shows each phase and lane, listing the milli ops that contributed groups,
/// atom counts, data flow, and work balance.
pub fn text_report(
    plan: &ExecutionPlan,
    provenance: &GroupProvenance,
    main_graph: &NanoGraph<'static, crate::pool::SystemPool>,
) -> String {
    let mut out = String::new();
    let num_lanes = plan.phases.first().map_or(0, |p| p.spans.len());

    // ── Summary ──
    writeln!(out, "═══ Execution Plan Report ═══").unwrap();
    writeln!(
        out,
        "  Phases: {}  Lanes: {}  Main graph groups: {}  Main graph atoms: {}",
        plan.phases.len(),
        num_lanes,
        main_graph.num_groups(),
        main_graph.num_atoms(),
    )
    .unwrap();

    // Milli op census from provenance.
    let mut milli_census: HashMap<&str, (usize, u64)> = HashMap::new();
    for (gi, (_, op_kind)) in provenance.iter().enumerate() {
        let group = &main_graph.groups()[gi];
        let entry = milli_census.entry(op_kind.as_str()).or_default();
        entry.0 += 1;
        entry.1 += group.count;
    }
    let mut census_sorted: Vec<_> = milli_census.into_iter().collect();
    census_sorted.sort_by(|a, b| b.1.1.cmp(&a.1.1));
    writeln!(out, "\n── Milli Op → Nano Groups ──").unwrap();
    for (kind, (groups, atoms)) in &census_sorted {
        writeln!(out, "  {:>6} groups {:>12} atoms  {}", groups, atoms, kind).unwrap();
    }

    // ── Per-phase detail ──
    writeln!(out, "\n── Phases ──").unwrap();

    let mut total_compute_atoms = 0u64;
    let mut total_input_atoms = 0u64;
    let mut total_output_atoms = 0u64;

    for (pi, phase) in plan.phases.iter().enumerate() {
        let phase_groups: usize = phase.spans.iter().map(|s| s.graph.num_groups()).sum();
        let phase_compute_atoms: u64 = phase
            .spans
            .iter()
            .flat_map(|s| s.graph.groups().iter())
            .map(|g| g.count)
            .sum();
        let phase_input_atoms: u64 = phase
            .spans
            .iter()
            .flat_map(|s| s.inputs.iter())
            .map(|r| r.count)
            .sum();
        let phase_output_atoms: u64 = phase
            .spans
            .iter()
            .flat_map(|s| s.outputs.iter())
            .map(|r| r.count)
            .sum();

        total_compute_atoms += phase_compute_atoms;
        total_input_atoms += phase_input_atoms;
        total_output_atoms += phase_output_atoms;

        // Per-lane atom counts for balance display.
        let lane_atoms: Vec<u64> = phase
            .spans
            .iter()
            .map(|s| s.graph.groups().iter().map(|g| g.count).sum())
            .collect();
        // Imbalance: max lane vs perfectly even split across all lanes.
        // 1.0 = balanced; num_lanes = all work on one lane.
        let max_lane = *lane_atoms.iter().max().unwrap_or(&0);
        let num_lanes_total = lane_atoms.len() as u64;
        let imbalance = if phase_compute_atoms > 0 && num_lanes_total > 0 {
            (max_lane as f64 * num_lanes_total as f64) / phase_compute_atoms as f64
        } else {
            0.0
        };

        writeln!(
            out,
            "\n  Phase {:>3}  groups={:<6} compute_atoms={:<12} in={:<10} out={:<10} balance={:.1}x",
            pi, phase_groups, phase_compute_atoms, phase_input_atoms, phase_output_atoms, imbalance,
        )
        .unwrap();

        // Per-lane breakdown: which milli ops are in each lane?
        for (li, span) in phase.spans.iter().enumerate() {
            if span.graph.num_groups() == 0 {
                continue;
            }
            let span_atoms: u64 = span.graph.groups().iter().map(|g| g.count).sum();

            // Map span groups back to milli ops via provenance.
            let mut span_ops: HashMap<&str, (usize, u64)> = HashMap::new();
            for group in span.graph.groups() {
                // Look up this atom in the main graph to find provenance.
                if let Some(gi) = main_graph.find_group_idx(group.base_id) {
                    if let Some((_, op_kind)) = provenance.get(gi) {
                        let entry = span_ops.entry(op_kind.as_str()).or_default();
                        entry.0 += 1;
                        entry.1 += group.count;
                    }
                }
            }
            let mut ops_sorted: Vec<_> = span_ops.into_iter().collect();
            ops_sorted.sort_by(|a, b| b.1.1.cmp(&a.1.1));

            // Nano op type census (ScalarOp).
            let mut nano_ops: HashMap<&str, u64> = HashMap::new();
            for group in span.graph.groups() {
                let name = scalar_op_name(&group.op);
                *nano_ops.entry(name).or_default() += group.count;
            }
            let mut nano_sorted: Vec<_> = nano_ops.into_iter().collect();
            nano_sorted.sort_by(|a, b| b.1.cmp(&a.1));
            let nano_summary: String = nano_sorted
                .iter()
                .take(5)
                .map(|(name, atoms)| format!("{}:{}", name, atoms))
                .collect::<Vec<_>>()
                .join(" ");

            write!(
                out,
                "    lane {}: {} groups, {} atoms, in={} out={}",
                li,
                span.graph.num_groups(),
                span_atoms,
                span.inputs.len(),
                span.outputs.len(),
            )
            .unwrap();
            writeln!(out, "  [{}]", nano_summary).unwrap();

            // Show top milli ops by atom count.
            let top_ops: String = ops_sorted
                .iter()
                .take(4)
                .map(|(kind, (groups, atoms))| format!("{}({}g/{}a)", kind, groups, atoms))
                .collect::<Vec<_>>()
                .join(", ");
            if !top_ops.is_empty() {
                writeln!(out, "           milli: {}", top_ops).unwrap();
            }
        }
    }

    writeln!(out, "\n── Totals ──").unwrap();
    writeln!(
        out,
        "  compute_atoms={} input_atoms={} output_atoms={}",
        total_compute_atoms, total_input_atoms, total_output_atoms,
    )
    .unwrap();

    out
}

// ─── SVG visualization ──────────────────────────────────────────────────────

/// Generate an SVG visualization of the execution plan.
///
/// Layout: phases as rows (top to bottom), lanes as columns.
/// Each span is a rectangle colored by its dominant milli op type.
/// Width proportional to atom count. Arrows show data flow between phases.
pub fn svg_report(
    plan: &ExecutionPlan,
    provenance: &GroupProvenance,
    main_graph: &NanoGraph<'static, crate::pool::SystemPool>,
) -> String {
    let num_lanes = plan.phases.first().map_or(0, |p| p.spans.len());
    let num_phases = plan.phases.len();
    if num_lanes == 0 || num_phases == 0 {
        return String::from("<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>");
    }

    // Assign colors to milli op kinds.
    let op_colors = build_op_color_map(provenance);

    let lane_w = 160.0f64;
    let phase_h = 60.0f64;
    let pad = 20.0;
    let header_h = 30.0;
    let label_w = 80.0; // left margin for phase labels

    let total_w = label_w + num_lanes as f64 * lane_w + pad * 2.0;
    let total_h = header_h + num_phases as f64 * phase_h + pad * 2.0;

    let mut svg = String::new();
    writeln!(
        svg,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {} {}\" \
         font-family=\"monospace\" font-size=\"10\">",
        total_w, total_h,
    )
    .unwrap();

    // Background.
    writeln!(
        svg,
        "<rect width=\"{}\" height=\"{}\" fill=\"#1a1a2e\"/>",
        total_w, total_h,
    )
    .unwrap();

    // Column headers.
    for li in 0..num_lanes {
        let x = label_w + pad + li as f64 * lane_w + lane_w / 2.0;
        writeln!(
            svg,
            "<text x=\"{}\" y=\"{}\" text-anchor=\"middle\" fill=\"#aaa\">Lane {}</text>",
            x,
            header_h - 5.0,
            li,
        )
        .unwrap();
    }

    // Find max atoms per span for scaling.
    let max_span_atoms: u64 = plan
        .phases
        .iter()
        .flat_map(|p| p.spans.iter())
        .map(|s| s.graph.groups().iter().map(|g| g.count).sum::<u64>())
        .max()
        .unwrap_or(1);

    // Draw phases.
    for (pi, phase) in plan.phases.iter().enumerate() {
        let y = header_h + pad + pi as f64 * phase_h;

        // Phase label.
        writeln!(
            svg,
            "<text x=\"{}\" y=\"{}\" fill=\"#888\" font-size=\"9\">P{}</text>",
            pad,
            y + phase_h / 2.0 + 3.0,
            pi,
        )
        .unwrap();

        for (li, span) in phase.spans.iter().enumerate() {
            let x = label_w + pad + li as f64 * lane_w + 2.0;
            let span_atoms: u64 = span.graph.groups().iter().map(|g| g.count).sum();

            if span_atoms == 0 {
                // Empty span — faint outline.
                writeln!(
                    svg,
                    "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" \
                     fill=\"none\" stroke=\"#333\" stroke-width=\"0.5\" rx=\"3\"/>",
                    x,
                    y + 2.0,
                    lane_w - 4.0,
                    phase_h - 4.0,
                )
                .unwrap();
                continue;
            }

            // Dominant milli op for this span.
            let dominant_op = dominant_op_kind(span, main_graph, provenance);
            let color = op_colors
                .get(dominant_op.as_str())
                .copied()
                .unwrap_or("#555");

            // Height scaled by log of atom count for better visual range.
            let log_atoms = (span_atoms as f64).ln();
            let log_max = (max_span_atoms as f64).ln();
            let h = ((log_atoms / log_max) * (phase_h - 8.0))
                .max(10.0)
                .min(phase_h - 4.0);
            let bar_y = y + (phase_h - h) / 2.0;

            // Build tooltip text.
            let mut tip_ops: HashMap<&str, (usize, u64)> = HashMap::new();
            for group in span.graph.groups() {
                if let Some(gi) = main_graph.find_group_idx(group.base_id) {
                    if let Some((_, op_kind)) = provenance.get(gi) {
                        let e = tip_ops.entry(op_kind.as_str()).or_default();
                        e.0 += 1;
                        e.1 += group.count;
                    }
                }
            }
            let mut tip_sorted: Vec<_> = tip_ops.into_iter().collect();
            tip_sorted.sort_by(|a, b| b.1.1.cmp(&a.1.1));
            let tip_text: String = tip_sorted
                .iter()
                .take(6)
                .map(|(k, (g, a))| format!("{}: {}g {}a", k, g, format_atoms(*a)))
                .collect::<Vec<_>>()
                .join("&#10;");
            let tooltip = format!(
                "P{} L{}: {} atoms, {} groups&#10;in:{} out:{}&#10;{}",
                pi,
                li,
                format_atoms(span_atoms),
                span.graph.num_groups(),
                span.inputs.len(),
                span.outputs.len(),
                tip_text,
            );

            writeln!(
                svg,
                "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" \
                 fill=\"{}\" fill-opacity=\"0.7\" stroke=\"{}\" \
                 stroke-width=\"1\" rx=\"3\"><title>{}</title></rect>",
                x,
                bar_y,
                lane_w - 4.0,
                h,
                color,
                color,
                tooltip,
            )
            .unwrap();

            // Labels — only show what fits in the bar height.
            if h >= 14.0 {
                // Atom count always shown.
                writeln!(
                    svg,
                    "<text x=\"{}\" y=\"{}\" fill=\"#ddd\" font-size=\"8\">{}a {}</text>",
                    x + 3.0,
                    bar_y + 10.0,
                    format_atoms(span_atoms),
                    truncate_str(&dominant_op, 14),
                )
                .unwrap();
            }

            if h >= 26.0 {
                // Nano op summary.
                let mut nano_census: HashMap<&str, u64> = HashMap::new();
                for group in span.graph.groups() {
                    *nano_census.entry(scalar_op_name(&group.op)).or_default() += group.count;
                }
                let mut sorted: Vec<_> = nano_census.into_iter().collect();
                sorted.sort_by(|a, b| b.1.cmp(&a.1));
                let label: String = sorted
                    .iter()
                    .take(3)
                    .map(|(n, a)| format!("{}:{}", n, format_atoms(*a)))
                    .collect::<Vec<_>>()
                    .join(" ");
                writeln!(
                    svg,
                    "<text x=\"{}\" y=\"{}\" fill=\"#aaa\" font-size=\"7\">{}</text>",
                    x + 3.0,
                    bar_y + 20.0,
                    label,
                )
                .unwrap();
            }

            if h >= 38.0 {
                // I/O summary.
                writeln!(
                    svg,
                    "<text x=\"{}\" y=\"{}\" fill=\"#777\" font-size=\"7\">in:{} out:{}</text>",
                    x + 3.0,
                    bar_y + 30.0,
                    span.inputs.len(),
                    span.outputs.len(),
                )
                .unwrap();
            }
        }

        // Phase separator line.
        if pi + 1 < num_phases {
            let line_y = y + phase_h;
            writeln!(
                svg,
                "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" \
                 stroke=\"#444\" stroke-width=\"0.5\" stroke-dasharray=\"4,3\"/>",
                label_w,
                line_y,
                total_w - pad,
                line_y,
            )
            .unwrap();
        }
    }

    // Legend.
    let legend_y = total_h - 15.0;
    let mut legend_x = label_w + pad;
    let mut color_entries: Vec<_> = op_colors.iter().collect();
    color_entries.sort_by_key(|(k, _)| *k);
    for (kind, color) in color_entries.iter().take(12) {
        writeln!(
            svg,
            "<rect x=\"{}\" y=\"{}\" width=\"8\" height=\"8\" fill=\"{}\" rx=\"1\"/>",
            legend_x, legend_y, color,
        )
        .unwrap();
        writeln!(
            svg,
            "<text x=\"{}\" y=\"{}\" fill=\"#aaa\" font-size=\"7\">{}</text>",
            legend_x + 10.0,
            legend_y + 7.0,
            truncate_str(kind, 12),
        )
        .unwrap();
        legend_x += 100.0;
    }

    writeln!(svg, "</svg>").unwrap();
    svg
}

// ─── Structured summary ────────────────────────────────────────────────────

/// Structured summary of an execution plan, suitable for serialization
/// and display in the Build Inspector UI.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PlanSummary {
    pub num_phases: u64,
    pub num_lanes: u64,
    pub main_graph_groups: u64,
    pub main_graph_atoms: u64,
    pub total_compute_atoms: u64,
    pub total_input_atoms: u64,
    pub total_output_atoms: u64,
    /// Milli op census: (op_kind, group_count, atom_count), sorted by atoms desc.
    pub milli_op_census: Vec<(String, u64, u64)>,
    pub phases: Vec<PhaseSummary>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PhaseSummary {
    pub num_groups: u64,
    pub compute_atoms: u64,
    pub input_atoms: u64,
    pub output_atoms: u64,
    /// Imbalance ratio: max_lane_atoms / (compute_atoms / num_lanes).
    /// 1.0 = perfectly balanced across all lanes.
    /// num_lanes = all work parked on a single lane.
    pub balance: f64,
    pub lanes: Vec<LaneSummary>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct LaneSummary {
    pub num_groups: u64,
    pub atoms: u64,
    pub num_inputs: u64,
    pub num_outputs: u64,
    /// Top nano ops: (name, atom_count), sorted desc, max 5.
    pub nano_ops: Vec<(String, u64)>,
    /// Top milli ops: (op_kind, group_count, atom_count), sorted desc, max 4.
    pub milli_ops: Vec<(String, u64, u64)>,
}

/// Build a structured summary from partitioned phases + provenance.
///
/// `main_graph` is the original NanoGraph; span NanoGraphs share its atom IDs.
/// Provenance maps main-graph group indices to (milli_op_id, op_kind).
pub fn summarize_plan(
    phases: &[Phase],
    provenance: &GroupProvenance,
    main_graph: &NanoGraph<'static, crate::pool::SystemPool>,
) -> PlanSummary {
    let num_lanes = phases.first().map_or(0, |p| p.spans.len()) as u64;

    // Build base_id → provenance index lookup for mapping span groups
    // back to provenance entries via the main graph.
    let base_to_prov: HashMap<AtomId, usize> = main_graph
        .groups()
        .iter()
        .enumerate()
        .map(|(i, g)| (g.base_id, i))
        .collect();

    // Milli op census from provenance.
    let mut milli_census: HashMap<&str, (u64, u64)> = HashMap::new();
    for (gi, (_, op_kind)) in provenance.iter().enumerate() {
        let atoms = main_graph.groups().get(gi).map_or(0, |g| g.count);
        let e = milli_census.entry(op_kind.as_str()).or_default();
        e.0 += 1;
        e.1 += atoms;
    }
    let mut milli_op_census: Vec<(String, u64, u64)> = milli_census
        .into_iter()
        .map(|(k, (g, a))| (k.to_string(), g, a))
        .collect();
    milli_op_census.sort_by(|a, b| b.2.cmp(&a.2));

    let mut total_compute_atoms = 0u64;
    let mut total_input_atoms = 0u64;
    let mut total_output_atoms = 0u64;
    let mut phase_summaries = Vec::new();

    for phase in phases {
        let mut phase_groups = 0u64;
        let mut phase_compute = 0u64;
        let mut phase_input = 0u64;
        let mut phase_output = 0u64;
        let mut lane_summaries = Vec::new();
        let mut lane_atoms_vec = Vec::new();

        for span in &phase.spans {
            let span_groups = span.graph.num_groups() as u64;
            let span_atoms: u64 = span.graph.groups().iter().map(|g| g.count).sum();
            let span_in: u64 = span.inputs.iter().map(|r| r.count).sum();
            let span_out: u64 = span.outputs.iter().map(|r| r.count).sum();

            phase_groups += span_groups;
            phase_compute += span_atoms;
            phase_input += span_in;
            phase_output += span_out;
            lane_atoms_vec.push(span_atoms);

            // Nano op census for this lane.
            let mut nano_ops: HashMap<&str, u64> = HashMap::new();
            for group in span.graph.groups() {
                *nano_ops.entry(scalar_op_name(&group.op)).or_default() += group.count;
            }
            let mut nano_sorted: Vec<(String, u64)> = nano_ops
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect();
            nano_sorted.sort_by(|a, b| b.1.cmp(&a.1));
            nano_sorted.truncate(5);

            // Milli op census for this lane via provenance lookup.
            let mut span_milli: HashMap<&str, (u64, u64)> = HashMap::new();
            for group in span.graph.groups() {
                if let Some(&gi) = base_to_prov.get(&group.base_id) {
                    if let Some((_, op_kind)) = provenance.get(gi) {
                        let e = span_milli.entry(op_kind.as_str()).or_default();
                        e.0 += 1;
                        e.1 += group.count;
                    }
                }
            }
            let mut milli_sorted: Vec<(String, u64, u64)> = span_milli
                .into_iter()
                .map(|(k, (g, a))| (k.to_string(), g, a))
                .collect();
            milli_sorted.sort_by(|a, b| b.2.cmp(&a.2));
            milli_sorted.truncate(4);

            lane_summaries.push(LaneSummary {
                num_groups: span_groups,
                atoms: span_atoms,
                num_inputs: span.inputs.len() as u64,
                num_outputs: span.outputs.len() as u64,
                nano_ops: nano_sorted,
                milli_ops: milli_sorted,
            });
        }

        total_compute_atoms += phase_compute;
        total_input_atoms += phase_input;
        total_output_atoms += phase_output;

        // Imbalance metric: ratio of busiest lane to a perfectly even split
        // across ALL lanes (including idle ones). 1.0 = perfectly balanced;
        // num_lanes = all work on one lane. This catches single-lane phases
        // that the old max/min-nonzero formula misreported as 1.0x.
        let max_lane = *lane_atoms_vec.iter().max().unwrap_or(&0);
        let num_lanes_total = lane_atoms_vec.len() as u64;
        let balance = if phase_compute > 0 && num_lanes_total > 0 {
            (max_lane as f64 * num_lanes_total as f64) / phase_compute as f64
        } else {
            0.0
        };

        phase_summaries.push(PhaseSummary {
            num_groups: phase_groups,
            compute_atoms: phase_compute,
            input_atoms: phase_input,
            output_atoms: phase_output,
            balance,
            lanes: lane_summaries,
        });
    }

    let main_graph_atoms: u64 = main_graph.groups().iter().map(|g| g.count).sum();
    PlanSummary {
        num_phases: phases.len() as u64,
        num_lanes,
        main_graph_groups: main_graph.num_groups() as u64,
        main_graph_atoms,
        total_compute_atoms,
        total_input_atoms,
        total_output_atoms,
        milli_op_census,
        phases: phase_summaries,
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

fn scalar_op_name(op: &ScalarOp) -> &'static str {
    match op {
        ScalarOp::Literal(_) => "Literal",
        ScalarOp::Identity | ScalarOp::Cast { .. } => "Identity",
        ScalarOp::Binary { .. } => "Binary",
        ScalarOp::Unary { .. } => "Unary",
        ScalarOp::Select => "Select",
        ScalarOp::Reduce { .. } => "Reduce",
        ScalarOp::IndirectLoad { .. } => "IndirLoad",
        ScalarOp::OpaqueOutput { .. } => "Opaque",
        ScalarOp::LiteralSpan(_) => "LiteralSpan",
    }
}

fn dominant_op_kind(
    span: &Span,
    main_graph: &NanoGraph<'static, crate::pool::SystemPool>,
    provenance: &GroupProvenance,
) -> String {
    let mut op_atoms: HashMap<&str, u64> = HashMap::new();
    for group in span.graph.groups() {
        if let Some(gi) = main_graph.find_group_idx(group.base_id) {
            if let Some((_, op_kind)) = provenance.get(gi) {
                *op_atoms.entry(op_kind.as_str()).or_default() += group.count;
            }
        }
    }
    op_atoms
        .into_iter()
        .max_by_key(|(_, atoms)| *atoms)
        .map(|(kind, _)| kind.to_string())
        .unwrap_or_else(|| "empty".to_string())
}

fn format_atoms(n: u64) -> String {
    if n >= 1_000_000_000 {
        format!("{:.1}G", n as f64 / 1e9)
    } else if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.0}K", n as f64 / 1e3)
    } else {
        format!("{}", n)
    }
}

fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}..", &s[..max_len - 2])
    }
}

const OP_PALETTE: &[&str] = &[
    "#e74c3c", // red
    "#3498db", // blue
    "#2ecc71", // green
    "#f39c12", // orange
    "#9b59b6", // purple
    "#1abc9c", // teal
    "#e67e22", // dark orange
    "#2980b9", // dark blue
    "#27ae60", // dark green
    "#c0392b", // dark red
    "#8e44ad", // dark purple
    "#16a085", // dark teal
    "#d35400", // rust
    "#2c3e50", // navy
    "#7f8c8d", // gray
];

fn build_op_color_map(provenance: &GroupProvenance) -> HashMap<String, &'static str> {
    let mut kinds: HashMap<String, usize> = HashMap::new();
    for (_, op_kind) in provenance {
        *kinds.entry(op_kind.clone()).or_default() += 1;
    }
    // Sort by frequency so the most common ops get the most distinct colors.
    let mut sorted: Vec<_> = kinds.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));

    sorted
        .into_iter()
        .enumerate()
        .map(|(i, (kind, _))| (kind, OP_PALETTE[i % OP_PALETTE.len()]))
        .collect()
}
