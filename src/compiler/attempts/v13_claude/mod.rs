#![allow(clippy::all, dead_code, unreachable_patterns)]
//! v13-claude: Cache-aware kernel partitioning for NanoGraph.
//!
//! The compiler's central problem is partitioning a scalar DAG into kernels
//! that minimize total memory traffic. Cache is a cost, not a hard constraint —
//! a kernel that exceeds cache pays a streaming penalty, but fragmenting into
//! many tiny kernels can be worse.
//!
//! See `src/compiler/problem_shape.md` for the full design rationale.
//!
//! This module provides:
//! - Common data structures for partition results (`partition`)
//! - Kernel cost estimation from addressing mode analysis (`cost`)
//! - Evaluation metrics for comparing partition quality (`evaluate`)
//! - Test graph builders for benchmarking approaches (`test_graphs`)
//!
//! Partitioning algorithm implementations live in submodules, each implementing
//! the `Partitioner` trait.

pub mod cost;
pub mod creative3;
pub mod evaluate;
pub mod execute;
#[cfg(feature = "cranelift")]
pub mod nano_codegen;
#[cfg(feature = "cranelift")]
pub mod nano_codegen_v2;
pub mod nano_execute;
pub mod nano_part_a;
pub mod nano_part_b;
pub mod nano_part_c;
pub mod nano_part_bisect;
pub mod nano_part_creative;
pub mod nano_part_hybrid;
pub mod nano_part_live;
pub mod nano_part_merge;
pub mod nano_part_topo;
pub mod nano_partition;
pub mod partition;
pub mod nano_plan_creative;
pub mod nano_plan_critical;
pub mod nano_plan_iterative;
pub mod nano_plan_v2a;
pub mod nano_plan_v2b;
pub mod nano_plan_v2c;
pub mod simple_dag;
pub mod test_graphs;
