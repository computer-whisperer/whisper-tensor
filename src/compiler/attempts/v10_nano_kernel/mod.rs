#![allow(clippy::all, dead_code, unreachable_patterns)]
//! v10 nano-kernel compiler.
//!
//! Consumes a NanoGraph (the main codebase's compressed scalar DAG) and compiles
//! it to native code via Cranelift. The key innovation over previous attempts:
//!
//! - **Input is NanoGraph**, not flat nano-op streams or expression trees.
//!   Reductions, addressing modes, and dtype semantics are explicit in the IR.
//! - **Cross-group fusion**: AtomGroup boundaries are compression artifacts.
//!   The planner looks across group boundaries to fuse compatible groups into
//!   single kernels, eliminating intermediate materialization.
//! - **Reduction inlining**: Pointwise chains feeding a ReduceSum/ReduceMax
//!   are absorbed into the reduction's inner loop — no intermediate buffer
//!   for the products in a matmul, for example.

pub mod plan;

#[cfg(feature = "cranelift")]
pub mod codegen;

#[cfg(feature = "cranelift")]
pub mod pipeline;
