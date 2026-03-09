//! Compiler implementation attempts.
//!
//! Each submodule is a self-contained compiler pipeline. The parent
//! `compiler` module re-exports the current best attempt so the rest
//! of the system doesn't need to know which one is active.

pub mod v1_scalar_crystal;
pub mod v2_fusion;
pub mod v3_nano_fusion;
pub mod v4_pool_growth;
pub mod v5_typed_synthesis;
pub mod v6_schedule_synthesis;
pub mod v7_parallel_crystal;
pub mod v8_generic_kernel;
