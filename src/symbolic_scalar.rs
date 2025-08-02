use crate::dtype::DType;
use crate::numeric_scalar::NumericScalarType;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SymbolicScalarTyped<T>
where
    T: Clone + Copy + NumericScalarType,
{
    _phantom_type: PhantomData<T>,
    offset: i64,
    symbol_idx: usize,
}

impl<T> SymbolicScalarTyped<T>
where
    T: Clone + Copy + NumericScalarType,
{
    pub(crate) fn new(resolver: &mut SymbolicResolver) -> Self {
        Self {
            _phantom_type: PhantomData,
            offset: 0,
            symbol_idx: resolver.new_id(),
        }
    }

    pub fn dtype() -> DType {
        T::DTYPE
    }

    pub(crate) fn to_dyn_type(&self) -> SymbolicScalar {
        SymbolicScalar {
            offset: self.offset,
            dtype: T::DTYPE,
            symbol_idx: self.symbol_idx,
        }
    }

    pub(crate) fn cast<T2>(&self) -> SymbolicScalarTyped<T2>
    where
        T2: Clone + Copy + NumericScalarType,
    {
        SymbolicScalarTyped {
            _phantom_type: PhantomData::<T2>,
            offset: self.offset,
            symbol_idx: self.symbol_idx,
        }
    }

    pub(crate) fn try_eq(&self, other: &Self) -> Option<bool> {
        if self.symbol_idx == other.symbol_idx {
            Some(self.offset == other.offset)
        } else {
            None
        }
    }

    pub(crate) fn add_offset(&self, offset: i64) -> Self {
        Self {
            offset: self.offset + offset,
            symbol_idx: self.symbol_idx,
            _phantom_type: PhantomData,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SymbolicScalar {
    offset: i64,
    dtype: DType,
    symbol_idx: usize,
}

impl SymbolicScalar {
    pub(crate) fn new(dtype: DType, resolver: &mut SymbolicResolver) -> Self {
        SymbolicScalar {
            offset: 0,
            dtype,
            symbol_idx: resolver.new_id(),
        }
    }

    pub(crate) fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn try_eq(&self, other: &Self) -> Option<bool> {
        if self.symbol_idx == other.symbol_idx {
            Some(self.offset == other.offset)
        } else {
            None
        }
    }

    pub(crate) fn cast<T>(&self) -> SymbolicScalarTyped<T>
    where
        T: Copy + Clone + NumericScalarType,
    {
        SymbolicScalarTyped {
            _phantom_type: PhantomData,
            offset: self.offset,
            symbol_idx: self.symbol_idx,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SymbolicResolver {
    next_symbolic_id: usize,
}

impl SymbolicResolver {
    pub fn new() -> Self {
        SymbolicResolver {
            next_symbolic_id: 0,
        }
    }
    pub(crate) fn update_last_assigned(&mut self, scalar: SymbolicScalar) {
        if scalar.symbol_idx >= self.next_symbolic_id {
            self.next_symbolic_id = scalar.symbol_idx + 1;
        }
    }
    fn new_id(&mut self) -> usize {
        let id = self.next_symbolic_id;
        self.next_symbolic_id += 1;
        id
    }
}
