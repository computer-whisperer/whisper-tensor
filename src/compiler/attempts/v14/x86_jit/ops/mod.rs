//! Op layer: per-`ScalarOp` emission against the compute slot ABI.
//!
//! Each op is a function that takes compute-repr operands in slots A
//! and B (or just A for unary) and writes the result to slot C. The op
//! layer knows the dtype contract per op — wrapping vs saturating
//! integer arithmetic, IEEE float semantics, NaN propagation rules,
//! etc. — but knows nothing about loops, memory layout, or fusion.
//!
//! Two sub-modules:
//!
//! - [`float`] — Add, Sub, Mul, Div, Min, Max, comparisons, and the
//!   transcendentals (Sqrt inline; the rest via libm trampolines until
//!   bit-equivalent inline approximations land).
//! - [`int`] — wrapping integer arithmetic, comparisons, bitwise,
//!   shifts, modulo, pow.
//!
//! Phase 0 stubs. Op coverage lands in phases 3 and 4.

pub mod float;
pub mod int;
