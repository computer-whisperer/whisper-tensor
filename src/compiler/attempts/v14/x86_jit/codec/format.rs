//! Format conversion: raw dtype bits ↔ compute representation.
//!
//! Given a `NumericDType` and a 64-bit register holding either raw
//! storage bits or a compute-repr value, emit the assembly that
//! converts between the two. The compute representation is fixed per
//! dtype family:
//!
//! - Floats with `m_bits ≤ 23`: F32 in an XMM register.
//! - Floats with `m_bits > 23`: F64 in an XMM register.
//! - Signed/unsigned ints: 64-bit GP register, sign- or zero-extended.
//! - Bool: low bit of a 64-bit GP register.
//!
//! Conversion is specialized at JIT-build time on the FloatType /
//! IntType properties — no runtime dispatch, no extern "C" calls. The
//! emitted sequence handles the full `(e_bits, m_bits, has_inf,
//! has_nan)` parameter space, not just named constants.
//!
//! Phase 0 stub. Implementation lands in phase 2.
