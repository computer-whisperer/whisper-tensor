//! Attempt 1: Scalar unrolling + crystal re-vectorization.
//!
//! Nano ops fully unroll milli ops to per-element scalar operations.
//! The crystal pass detects repeating index patterns and compresses
//! them back into loops for codegen.

pub mod crystal;
pub mod nano_op;

#[cfg(feature = "cranelift")]
pub mod codegen;
