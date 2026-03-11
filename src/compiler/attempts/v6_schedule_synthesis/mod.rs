//! Attempt 6: generic schedule synthesis from whitewashed nano-op pools.
//!
//! v6 intentionally avoids explicit matmul recognizers. It rebuilds grouped
//! expression families, then classifies schedule intent using generic pattern
//! and dependence-style signals.

#[cfg(feature = "cranelift")]
pub mod codegen;
pub mod synthesis;
