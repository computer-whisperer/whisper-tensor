//! Test scaffolding for the new x86_jit.
//!
//! Tests are organized to mirror the layer structure:
//!
//! - [`ab_harness`] — the bytewise A/B harness comparing x86_jit
//!   output against `pool_eval`. Used by every higher-level test
//!   module to assert byte-equality (the dtype contract demands
//!   bit-identical results from any conformant runtime).
//! - Codec tests, op tests, orchestration tests, and end-to-end model
//!   tests land in their own modules as the corresponding phases ship.

pub mod ab_harness;
pub mod jit_harness;

#[cfg(test)]
mod ab_harness_smoke;
#[cfg(test)]
mod codec_bit_io;
#[cfg(test)]
mod codec_format;
