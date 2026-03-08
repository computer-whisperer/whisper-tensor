//! Compiler implementation attempts.
//!
//! Each submodule is a self-contained compiler pipeline. The parent
//! `compiler` module re-exports the current best attempt so the rest
//! of the system doesn't need to know which one is active.

pub mod v1_scalar_crystal;
pub mod v2_fusion;
