#![allow(clippy::all, dead_code, unreachable_patterns)]

#[cfg(feature = "cranelift")]
pub mod codegen;
pub mod execute;
pub mod partitioner_b; // gen-1 winner, kept as reference (does NOT split groups)
pub mod partitioner_i;
pub mod partitioner_j;
pub mod partitioner_l;
pub mod partitioner_m;
pub mod plan;
pub mod report;
pub mod types;
