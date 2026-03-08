//! Attempt 4: Pool growth from unordered nano-op sets.
//!
//! The goal is to recover performant loop crystals from a whitewashed
//! nano-op pool where execution order hints are gone.

pub mod growth;

#[cfg(feature = "cranelift")]
pub mod codegen;

pub use crate::compiler::attempts::v1_scalar_crystal::crystal;
pub use crate::compiler::attempts::v1_scalar_crystal::nano_op;
