//! Attempt 2: Direct loop construction with fusion and matmul support.
//!
//! Instead of unrolling to per-element nano ops and re-detecting loops,
//! this attempt builds loop structures directly from the milli graph.
//! Consecutive elementwise ops with compatible shapes are fused into a
//! single kernel so intermediates stay in registers. Matmul is a
//! first-class kernel with its own triple-loop codegen.

pub mod kernel;
pub mod planner;

#[cfg(feature = "cranelift")]
pub mod codegen;
