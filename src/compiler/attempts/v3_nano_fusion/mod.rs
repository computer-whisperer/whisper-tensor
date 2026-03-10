//! Attempt 3: Nano-op-first loop fusion.
//!
//! Keeps v1's nano-op expansion + crystallization pipeline, then fuses
//! adjacent crystal loops by forwarding per-iteration values directly
//! instead of reloading intermediates from memory.

pub mod fusion;

#[cfg(feature = "cranelift")]
pub mod codegen;

pub use crate::compiler::attempts::v1_scalar_crystal::crystal;
pub use crate::compiler::attempts::v1_scalar_crystal::nano_op;
