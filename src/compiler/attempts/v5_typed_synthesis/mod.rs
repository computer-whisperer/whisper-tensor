//! Attempt 5: Typed kernel synthesis from recovered crystals.
//!
//! v5 builds on v4 pool recovery and adds dtype-aware schedule synthesis
//! with a software bf16->fp32 matmul path (no BLAS dependency).

pub mod synth;
