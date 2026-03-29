//! Scalar numeric operations on raw bits + dtype.
//!
//! Each op is a set of pure functions grouped by operation, with explicit
//! naming for the type category and overflow semantics:
//!
//! - `float_add(a, b, &FloatType)` — IEEE float addition
//! - `signed_add_wrapping(a, b, &IntType)` — wrapping signed integer addition
//! - `signed_add_saturating(a, b, &IntType)` — saturating signed integer addition
//! - `unsigned_add_wrapping(a, b, &IntType)` — wrapping unsigned integer addition
//! - `unsigned_add_saturating(a, b, &IntType)` — saturating unsigned integer addition
//!
//! Float ops have one semantic (IEEE). Integer ops have explicit wrapping vs
//! saturating variants. The caller (e.g. nano-op eval) picks the right variant
//! for the operation's semantics.
//!
//! Float-only ops (sin, cos, etc.) only have a `float_*` variant.

// Arithmetic
pub mod abs;
pub mod add;
pub mod div;
pub mod max;
pub mod min;
pub mod modulo;
pub mod mul;
pub mod neg;
pub mod pow;
pub mod sub;

// Float-only unary
pub mod ceil;
pub mod erf;
pub mod exp;
pub mod floor;
pub mod ln;
pub mod reciprocal;
pub mod round;
pub mod sqrt;

// Trigonometric and hyperbolic
pub mod sin;
pub mod tanh;
pub mod trig;

// Sign
pub mod sign;

// Comparison
pub mod cmp;

// Logical
pub mod logical;

// Bitwise
pub mod bitwise;

// Predicates
pub mod is_inf;
pub mod is_nan;

use std::fmt;

use crate::numeric_dtype::NumericDType;

/// Error from a scalar operation dispatch.
#[derive(Debug, Clone)]
pub enum OpError {
    /// The operation does not support this dtype.
    UnsupportedDType {
        op: &'static str,
        dtype: NumericDType,
    },
    /// Binary operation received mismatched dtypes.
    DTypeMismatch {
        op: &'static str,
        a: NumericDType,
        b: NumericDType,
    },
}

impl fmt::Display for OpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OpError::UnsupportedDType { op, dtype } => {
                write!(f, "op '{op}' does not support dtype {dtype}")
            }
            OpError::DTypeMismatch { op, a, b } => {
                write!(f, "op '{op}' requires matching dtypes, got {a} and {b}")
            }
        }
    }
}

impl std::error::Error for OpError {}
