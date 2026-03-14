#![allow(clippy::all, dead_code, unreachable_patterns)]
//! v13-claude: Cache-aware kernel partitioning for NanoGraph.
//!
//! The compiler's central problem is partitioning a scalar DAG into kernels
//! that minimize cross-kernel data movement, subject to each kernel's working
//! set fitting in a target cache budget.
//!
//! See `src/compiler/problem_shape.md` for the full design rationale.
//!
//! This module provides:
//! - Common data structures for partition results (`partition`)
//! - Evaluation metrics for comparing partition quality (`evaluate`)
//! - Test graph builders for benchmarking approaches (`test_graphs`)
//!
//! Partitioning algorithm implementations live in submodules, each implementing
//! the `Partitioner` trait.

pub mod evaluate;
pub mod partition;
pub mod test_graphs;
