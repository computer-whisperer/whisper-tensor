//! Integer arithmetic on compute slots.
//!
//! Wrapping Add/Sub/Mul/Div, signed and unsigned comparisons, bitwise
//! And/Or/Xor/Not, shift left, signed and unsigned shift right (the
//! shift right kind selected at JIT-build time per dtype signedness),
//! signed and unsigned Min/Max, modulo (Mod and Euclidean IMod), and
//! signed/unsigned Pow.
//!
//! Per dtype contract §5.2, division by zero returns 0 (not UB), and
//! `MIN / -1` wraps to `MIN`. The emission handles both edge cases
//! explicitly because x86 `idiv` would trap on the `MIN / -1` overflow.
//!
//! Phase 0 stub. Implementation lands in phase 3.
