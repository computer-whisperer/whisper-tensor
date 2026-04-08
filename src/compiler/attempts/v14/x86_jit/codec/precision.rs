//! In-register precision narrowing.
//!
//! `narrow_to(value, target_dtype)` emits the assembly that takes a
//! value in the compute representation and rounds it to the target
//! dtype's precision **without leaving the register**. The result
//! stays in the same compute repr — narrowing is purely a precision
//! reduction, not a format conversion.
//!
//! This is the third codec primitive, distinct from format conversion.
//! It enforces the dtype contract for fused intermediates: a chain of
//! BF16 ops in F32 compute must round per step (per the §4.7 reduce
//! rule generalized to any narrowing-cast intermediate), and that
//! per-step rounding is done by emitting `narrow_to(BF16)` between
//! ops, never touching memory.
//!
//! Phase 0 stub. Implementation lands in phase 2; the BF16 8-instruction
//! RTNE recipe is documented in `x86_jit_codec.md` §3.5.
