#![allow(clippy::all, dead_code, unreachable_patterns)]
//! v13-claude: Cache-aware kernel partitioning for NanoGraph.
//!
//! See `src/compiler/problem_shape.md` for the full design rationale.

pub mod cost;
pub mod creative3;
pub mod evaluate;
pub mod execute;
#[cfg(feature = "cranelift")]
pub mod nano_codegen;
#[cfg(feature = "cranelift")]
pub mod nano_codegen_v2;
pub mod nano_execute;
pub mod nano_part_creative;
pub mod nano_plan_spans_a;
pub mod nano_plan_spans_b;
pub mod nano_plan_spans_c;
pub mod nano_plan_spans_d;
pub mod nano_plan_v2c;
pub mod nano_plan_v3a;
pub mod nano_plan_v3b;
pub mod nano_plan_v3c;
pub mod nano_plan_v3d;
pub mod nano_plan_v4a;
pub mod nano_plan_v4b;
pub mod partition;
pub mod simple_dag;
pub mod test_graphs;
pub mod nano_plan_v4c;
