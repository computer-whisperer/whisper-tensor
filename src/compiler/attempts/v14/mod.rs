#![allow(clippy::all, dead_code, unreachable_patterns)]

#[cfg(feature = "cranelift")]
pub mod codegen;
pub mod execute;
pub mod partitioner_b; // gen-1 winner, kept as reference (does NOT split groups)
pub mod plan;
pub mod report;
pub mod types;
