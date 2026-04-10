#![allow(clippy::all, dead_code, unreachable_patterns)]

pub mod executor;
#[cfg(feature = "x86_compile")]
pub mod layout;
pub mod partitioner_m;
pub mod partitioner_n;
pub mod plan;
pub mod report;
pub mod types;
#[cfg(feature = "x86_compile")]
pub mod x86_jit;

// Note: `x86_jit` is now a directory module containing the
// coverage-first rewrite (codec / ops / orch). The old single-file
// `x86_jit.rs` was deleted during phase 0; see `X86_JIT_PLAN.md`.
