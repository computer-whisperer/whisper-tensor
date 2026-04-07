//! Build Inspector cache reports.
//!
//! Structured summaries of `SuperGraphCache` contents — lowered nano graphs,
//! JIT-compiled plans, and other cached state. Used by both the server (sent
//! to the Build Inspector UI over the wire) and the CLI (printed as text).

use crate::graph::GlobalId;
use crate::super_graph::cache::SuperGraphCache;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Write;

/// Per-lowered-model report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoweredModelReport {
    pub graph_id: GlobalId,
    pub info_inputs_hash: u64,
    pub num_groups: u64,
    pub total_atoms: u64,
    pub singleton_groups: u64,
    pub symbolic_groups: u64,
    /// Op name → count (String instead of &'static str for serialization).
    pub groups_by_op: HashMap<String, u64>,
    pub num_tensors: u64,
    pub num_inputs: u64,
    pub num_outputs: u64,
    /// Milli-op census: op_kind → (group_count, atom_count). Built from group_provenance.
    pub milli_op_census: HashMap<String, (u64, u64)>,
    /// Ops that could not be lowered.
    pub unsupported: Vec<(GlobalId, String)>,
    /// Human-readable detail for each unsupported op.
    pub unsupported_details: Vec<String>,
}

/// Per-compiled-plan report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledPlanReport {
    pub graph_id: GlobalId,
    pub info_inputs_hash: u64,
    pub num_outputs: u64,
    pub plan_summary: crate::compiler::attempts::v14::report::PlanSummary,
}

/// One cache slot (keyed by the use_cache u64 in SuperGraphRequest, or 0 for
/// CLI usage where there's only one cache).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheReportEntry {
    pub cache_key: u64,
    pub num_rnn_entries: u64,
    pub num_tensor_entries: u64,
    pub num_tensor_pack_entries: u64,
    pub lowered_models: Vec<LoweredModelReport>,
    pub compiled_plans: Vec<CompiledPlanReport>,
}

/// Full cache report.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheReport {
    pub entries: Vec<CacheReportEntry>,
}

impl SuperGraphCache {
    /// Build a structured report entry for this cache.
    pub fn to_report_entry(&self, cache_key: u64) -> CacheReportEntry {
        let mut lowered_models = Vec::new();
        for (&graph_id, cached) in &self.lowered_model_cache {
            let stats = cached.graph.stats();
            let groups = cached.graph.groups();
            // Milli-op census from provenance: op_kind → (group_count, atom_count).
            let mut milli_op_census: HashMap<String, (u64, u64)> = HashMap::new();
            for (group_idx, (_milli_id, op_kind)) in cached.group_provenance.iter().enumerate() {
                let atom_count = groups.get(group_idx).map(|g| g.count).unwrap_or(0);
                let entry = milli_op_census.entry(op_kind.clone()).or_insert((0, 0));
                entry.0 += 1;
                entry.1 += atom_count;
            }
            lowered_models.push(LoweredModelReport {
                graph_id,
                info_inputs_hash: cached.info_inputs_hash,
                num_groups: stats.num_groups,
                total_atoms: stats.total_atoms,
                singleton_groups: stats.singleton_groups,
                symbolic_groups: stats.symbolic_groups,
                groups_by_op: stats
                    .groups_by_op
                    .into_iter()
                    .map(|(k, v)| (k.to_string(), v))
                    .collect(),
                num_tensors: cached.tensor_map.len() as u64,
                num_inputs: cached.input_map.len() as u64,
                num_outputs: cached.output_map.len() as u64,
                milli_op_census,
                unsupported: cached.unsupported.clone(),
                unsupported_details: cached.unsupported_details.clone(),
            });
        }
        lowered_models.sort_by_key(|m| m.graph_id);

        let mut compiled_plans: Vec<CompiledPlanReport> = Vec::new();
        #[cfg(feature = "cranelift")]
        for (&graph_id, cached) in &self.compiled_plan_cache {
            compiled_plans.push(CompiledPlanReport {
                graph_id,
                info_inputs_hash: cached.info_inputs_hash,
                num_outputs: cached.output_ranges.len() as u64,
                plan_summary: cached.plan_summary.clone(),
            });
        }
        compiled_plans.sort_by_key(|p| p.graph_id);

        CacheReportEntry {
            cache_key,
            num_rnn_entries: self.rnn_cache.len() as u64,
            num_tensor_entries: self.tensor_cache.len() as u64,
            num_tensor_pack_entries: self.tensor_pack_cache.len() as u64,
            lowered_models,
            compiled_plans,
        }
    }
}

/// Format a cache report as text, mirroring the v14 text_report style.
pub fn format_cache_report_text(report: &CacheReport) -> String {
    let mut out = String::new();
    if report.entries.is_empty() {
        writeln!(out, "═══ Cache Report ═══").unwrap();
        writeln!(out, "  (cache is empty)").unwrap();
        return out;
    }

    for entry in &report.entries {
        writeln!(out, "═══ Cache slot {} ═══", entry.cache_key).unwrap();
        writeln!(
            out,
            "  RNN entries: {}  Tensor entries: {}  Tensor pack entries: {}",
            entry.num_rnn_entries, entry.num_tensor_entries, entry.num_tensor_pack_entries,
        )
        .unwrap();

        if entry.lowered_models.is_empty() && entry.compiled_plans.is_empty() {
            writeln!(out, "  (no lowered models or compiled plans cached)").unwrap();
            writeln!(
                out,
                "  Run with --eval-mode=lowered or --eval-mode=compiled to populate."
            )
            .unwrap();
            continue;
        }

        for model in &entry.lowered_models {
            writeln!(out).unwrap();
            writeln!(
                out,
                "── Lowered Model {} (hash {:016x}) ──",
                model.graph_id, model.info_inputs_hash,
            )
            .unwrap();
            writeln!(
                out,
                "  Groups: {:<10} Atoms: {:<12} Singletons: {:<8} Symbolic: {}",
                model.num_groups, model.total_atoms, model.singleton_groups, model.symbolic_groups,
            )
            .unwrap();
            writeln!(
                out,
                "  Tensors: {:<8} Inputs: {:<8} Outputs: {}",
                model.num_tensors, model.num_inputs, model.num_outputs,
            )
            .unwrap();

            if !model.groups_by_op.is_empty() {
                writeln!(out, "\n  Nano op breakdown:").unwrap();
                let mut ops: Vec<_> = model.groups_by_op.iter().collect();
                ops.sort_by(|a, b| b.1.cmp(a.1));
                for (op, count) in ops {
                    writeln!(out, "    {:>8} {}", count, op).unwrap();
                }
            }

            if !model.milli_op_census.is_empty() {
                writeln!(out, "\n  Milli op census:").unwrap();
                let mut ops: Vec<_> = model.milli_op_census.iter().collect();
                ops.sort_by(|a, b| b.1.1.cmp(&a.1.1));
                for (op, (groups, atoms)) in ops {
                    writeln!(out, "    {:>6} groups {:>12} atoms  {}", groups, atoms, op).unwrap();
                }
            }

            if !model.unsupported.is_empty() {
                writeln!(out, "\n  Unsupported ops ({}):", model.unsupported.len()).unwrap();
                for detail in &model.unsupported_details {
                    writeln!(out, "    {detail}").unwrap();
                }
            }
        }

        for plan in &entry.compiled_plans {
            writeln!(out).unwrap();
            writeln!(
                out,
                "── Compiled Plan {} (hash {:016x}) ──",
                plan.graph_id, plan.info_inputs_hash,
            )
            .unwrap();
            let s = &plan.plan_summary;
            writeln!(
                out,
                "  Phases: {}  Lanes: {}  Main graph groups: {}  Main graph atoms: {}",
                s.num_phases, s.num_lanes, s.main_graph_groups, s.main_graph_atoms,
            )
            .unwrap();
            writeln!(
                out,
                "  Compute atoms: {:<12} Input atoms: {:<10} Output atoms: {:<10} Outputs: {}",
                s.total_compute_atoms, s.total_input_atoms, s.total_output_atoms, plan.num_outputs,
            )
            .unwrap();

            if !s.milli_op_census.is_empty() {
                writeln!(out, "\n  Milli op census:").unwrap();
                for (kind, groups, atoms) in &s.milli_op_census {
                    writeln!(
                        out,
                        "    {:>6} groups {:>12} atoms  {}",
                        groups, atoms, kind
                    )
                    .unwrap();
                }
            }

            if !s.phases.is_empty() {
                writeln!(out, "\n  Phases:").unwrap();
                for (pi, phase) in s.phases.iter().enumerate() {
                    writeln!(
                        out,
                        "    Phase {:>3}  groups={:<6} compute={:<12} in={:<10} out={:<10} balance={:.1}x",
                        pi,
                        phase.num_groups,
                        phase.compute_atoms,
                        phase.input_atoms,
                        phase.output_atoms,
                        phase.balance,
                    )
                    .unwrap();
                    for (li, lane) in phase.lanes.iter().enumerate() {
                        if lane.num_groups == 0 {
                            continue;
                        }
                        let nano: String = lane
                            .nano_ops
                            .iter()
                            .map(|(n, a)| format!("{n}:{a}"))
                            .collect::<Vec<_>>()
                            .join(" ");
                        writeln!(
                            out,
                            "      lane {}: {} groups, {} atoms, in={} out={}  [{}]",
                            li,
                            lane.num_groups,
                            lane.atoms,
                            lane.num_inputs,
                            lane.num_outputs,
                            nano,
                        )
                        .unwrap();
                        if !lane.milli_ops.is_empty() {
                            let milli: String = lane
                                .milli_ops
                                .iter()
                                .map(|(k, g, a)| format!("{k}({g}g/{a}a)"))
                                .collect::<Vec<_>>()
                                .join(", ");
                            writeln!(out, "             milli: {milli}").unwrap();
                        }
                    }
                }
            }
        }
        writeln!(out).unwrap();
    }
    out
}
