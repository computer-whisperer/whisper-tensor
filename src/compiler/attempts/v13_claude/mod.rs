#![allow(clippy::all, dead_code, unreachable_patterns)]
//! v13-claude: Cache-aware kernel partitioning for NanoGraph.
//!
//! See `src/compiler/problem_shape.md` for the full design rationale.

#[cfg(feature = "cranelift")]
pub mod nano_codegen;
#[cfg(feature = "cranelift")]
pub mod nano_codegen_v2;
pub mod plan;
pub mod test_graphs;
