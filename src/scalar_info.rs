use crate::numeric_dtype::NumericDType;
use crate::numeric_dtype::NumericPrimitive;
use crate::numeric_scalar::NumericScalar;
use crate::symbolic_scalar::{SymbolicScalar, SymbolicScalarTyped};
use num_traits::AsPrimitive;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ScalarInfoTyped<T>
where
    T: Clone + Copy + PartialEq + NumericPrimitive,
{
    Numeric(T),
    Symbolic(SymbolicScalarTyped<T>),
}

impl<T> ScalarInfoTyped<T>
where
    T: Clone + Copy + PartialEq + NumericPrimitive,
{
    pub(crate) fn cast<T2>(&self) -> ScalarInfoTyped<T2>
    where
        T2: Clone + Copy + PartialEq + NumericPrimitive + 'static,
        T: AsPrimitive<T2>,
    {
        match self {
            ScalarInfoTyped::Numeric(x) => ScalarInfoTyped::<T2>::Numeric((*x).as_()),
            ScalarInfoTyped::Symbolic(scalar) => ScalarInfoTyped::<T2>::Symbolic(scalar.cast()),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn try_eq(&self, other: &Self) -> Option<bool> {
        if let (Self::Numeric(a), Self::Numeric(b)) = (self, other) {
            Some(a == b)
        } else if let (Self::Symbolic(a), Self::Symbolic(b)) = (self, other) {
            a.try_eq(b)
        } else {
            None
        }
    }

    #[allow(dead_code)]
    pub(crate) fn add_offset(&self, offset: i64) -> Self
    where
        T: 'static + std::ops::Add<Output = T>,
        i64: AsPrimitive<T>,
    {
        match self {
            Self::Numeric(a) => Self::Numeric(*a + offset.as_()),
            Self::Symbolic(scalar) => Self::Symbolic(scalar.add_offset(offset)),
        }
    }

    pub(crate) fn is_numeric(&self) -> bool {
        matches!(self, ScalarInfoTyped::Numeric(_))
    }

    #[allow(dead_code)]
    pub(crate) fn is_symbolic(&self) -> bool {
        matches!(self, ScalarInfoTyped::Symbolic(_))
    }

    pub fn as_numeric(&self) -> Option<&T> {
        if let Self::Numeric(x) = self {
            Some(x)
        } else {
            None
        }
    }

    #[allow(dead_code)]
    pub(crate) fn as_symbolic(&self) -> Option<&SymbolicScalarTyped<T>> {
        if let Self::Symbolic(x) = self {
            Some(x)
        } else {
            None
        }
    }
}

/// Type-erased scalar info: either a concrete numeric value or a symbolic placeholder.
///
/// The `Numeric` variant holds a new [`NumericScalar`] (`[u8; 8]` + `NumericDType`).
/// The `Symbolic` variant holds a [`SymbolicScalar`] with `NumericDType` metadata.
#[derive(Clone, Debug)]
pub enum ScalarInfo {
    Numeric(NumericScalar),
    Symbolic(SymbolicScalar),
}

impl ScalarInfo {
    pub(crate) fn dtype(&self) -> NumericDType {
        match self {
            ScalarInfo::Numeric(x) => x.dtype(),
            ScalarInfo::Symbolic(x) => x.dtype(),
        }
    }

    /// Cast to a typed scalar info. For Numeric, casts the value to `T`'s dtype
    /// then extracts the typed value. For Symbolic, preserves the symbolic
    /// identity with a type cast.
    ///
    /// Primarily used for shape-dim extraction (T = u64, i64, u32).
    #[allow(dead_code)]
    pub(crate) fn cast<T: NumericPrimitive>(&self) -> ScalarInfoTyped<T> {
        match self {
            ScalarInfo::Numeric(x) => {
                let casted = x.cast_to(T::NUMERIC_DTYPE);
                ScalarInfoTyped::Numeric(T::from_scalar(&casted))
            }
            ScalarInfo::Symbolic(scalar) => ScalarInfoTyped::<T>::Symbolic(scalar.cast()),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn is_numeric(&self) -> bool {
        matches!(self, ScalarInfo::Numeric(_))
    }

    #[allow(dead_code)]
    pub(crate) fn is_symbolic(&self) -> bool {
        matches!(self, ScalarInfo::Symbolic(_))
    }
}
