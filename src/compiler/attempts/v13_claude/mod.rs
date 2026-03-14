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
pub mod evaluate;
pub mod partition;
pub mod creative3;
pub mod simple_dag;
pub mod test_graphs;
