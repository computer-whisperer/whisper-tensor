#![allow(clippy::all, dead_code, unreachable_patterns)]

#[cfg(feature = "cranelift")]
pub mod codegen;
pub mod executor;
pub mod partitioner_m;
pub mod plan;
pub mod report;
pub mod types;
