use num_traits::AsPrimitive;
use serde::{Deserialize, Serialize};
use crate::dtype::{DType, DTypeOfPrimitive};
use crate::numeric_scalar::{NumericScalar, NumericScalarType};
use crate::symbolic_scalar::{SymbolicScalar, SymbolicScalarTyped};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ScalarInfoTyped<T>
where
    T: Clone + Copy + PartialEq + NumericScalarType,
{
    Numeric(T),
    Symbolic(SymbolicScalarTyped<T>)
}

impl<T> ScalarInfoTyped<T>
where
    T: Clone + Copy + PartialEq + NumericScalarType,
{
    pub(crate) fn promote(&self) -> ScalarInfo {
        match self {
            ScalarInfoTyped::Numeric(x) => ScalarInfo::Numeric(NumericScalar::from(*x)),
            ScalarInfoTyped::Symbolic(scalar) => ScalarInfo::Symbolic(scalar.to_dyn_type())
        }
    }
    
    pub(crate) fn cast<T2>(&self) -> ScalarInfoTyped<T2>
    where
        T2: Clone + Copy + PartialEq + NumericScalarType + 'static,
        T: AsPrimitive<T2>
    {
        match self {
            ScalarInfoTyped::Numeric(x) => ScalarInfoTyped::<T2>::Numeric((*x).as_()),
            ScalarInfoTyped::Symbolic(scalar) => ScalarInfoTyped::<T2>::Symbolic(scalar.cast())
        }
    }
    
    pub(crate) fn try_eq(&self, other: &Self) -> Option<bool> {
        if let (Self::Numeric(a), Self::Numeric(b)) = (self, other) {
            Some(a == b)
        } else if let (Self::Symbolic(a), Self::Symbolic(b)) = (self, other) {
            a.try_eq(&b)
        } else {
            None
        }
    }
    
    pub(crate) fn add_offset(&self, offset: i64) -> Self 
    where
        T: 'static + std::ops::Add<Output = T>,
        i64: AsPrimitive<T>
    {
        match self {
            Self::Numeric(a) => Self::Numeric(*a + offset.as_()),
            Self::Symbolic(scalar) => Self::Symbolic(scalar.add_offset(offset))
        }
    }
    
    pub(crate) fn to_dyn_type(&self) -> ScalarInfo {
        match self {
            Self::Numeric(x) => ScalarInfo::Numeric(NumericScalar::from(*x)),
            Self::Symbolic(x) => ScalarInfo::Symbolic(x.to_dyn_type())
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

#[derive(Clone, Debug)]
pub enum ScalarInfo {
    Numeric(NumericScalar),
    Symbolic(SymbolicScalar)
}

impl ScalarInfo {
    pub(crate) fn dtype(&self) -> DType {
        match self {
            ScalarInfo::Numeric(x) => x.dtype(),
            ScalarInfo::Symbolic(x) => x.dtype()
        }
    }
    
    pub(crate) fn cast<T: DTypeOfPrimitive>(&self) -> ScalarInfoTyped<T>
    where T: 
       NumericScalarType + PartialEq + Copy + Clone
    {
        match self {
            ScalarInfo::Numeric(x) => ScalarInfoTyped::Numeric(T::cast_from_numeric_scalar(x)),
            ScalarInfo::Symbolic(scalar) => ScalarInfoTyped::<T>::Symbolic(scalar.cast())
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
