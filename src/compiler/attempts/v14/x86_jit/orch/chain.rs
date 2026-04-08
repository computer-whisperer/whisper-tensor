//! Emit a fused chain of `AtomGroup`s.
//!
//! When two or more groups have producer/consumer relationships and
//! identical iteration shape, the chain emitter forwards compute-repr
//! values between them in registers. Intermediate dtype narrowing is
//! applied via [`super::super::codec::precision::narrow_to`] between
//! ops where the intermediate dtype is narrower than the compute repr.
//!
//! The "no-op compression" property: a chain of aligned F32 ops with
//! F32 compute reduces to bare loads/stores/ALU because every
//! `narrow_to(F32)` and every `format::convert(F32 ↔ F32)` is empty.
//!
//! Phase 0 stub. Fusion lands in phase 6 — the unfused base path comes
//! first.
