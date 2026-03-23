//! Scalar numeric operations on raw bits + dtype.
//!
//! Each op is a pure function: `(u64 raw bits, NumericDType) → Result<u64, OpError>`.
//! Operations decode bits according to the dtype, perform the computation in
//! the appropriate native type (f64 for floats, i128 for signed ints, u128 for
//! unsigned ints), and encode the result back.
//!
//! No f64 roundtrip for integer operations — each category uses its native width.

pub mod add;
pub mod mul;
pub mod neg;
pub mod sin;

use std::fmt;

use crate::numeric_dtype::NumericDType;

/// Error from a scalar operation.
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

/// Check that two dtypes match for a binary op.
fn check_dtype_match(op: &'static str, a: NumericDType, b: NumericDType) -> Result<(), OpError> {
    if a != b {
        Err(OpError::DTypeMismatch { op, a, b })
    } else {
        Ok(())
    }
}
