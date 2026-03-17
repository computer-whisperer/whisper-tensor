//! Compiler implementation attempts.
//!
//! Each submodule is a self-contained compiler pipeline. The parent
//! `compiler` module re-exports the current best attempt so the rest
//! of the system doesn't need to know which one is active.

pub mod v13_claude;
pub mod v14;
