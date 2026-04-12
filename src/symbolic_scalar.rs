use crate::numeric_dtype::NumericDType;
use crate::numeric_dtype::NumericPrimitive;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SymbolicScalarTyped<T>
where
    T: Clone + Copy + NumericPrimitive,
{
    _phantom_type: PhantomData<T>,
    offset: i64,
    symbol_id: u64,
}

impl<T> SymbolicScalarTyped<T>
where
    T: Clone + Copy + NumericPrimitive,
{
    pub(crate) fn new(rng: &mut impl Rng) -> Self {
        Self {
            _phantom_type: PhantomData,
            offset: 0,
            symbol_id: rng.next_u64(),
        }
    }

    pub(crate) fn cast<T2>(&self) -> SymbolicScalarTyped<T2>
    where
        T2: Clone + Copy + NumericPrimitive,
    {
        SymbolicScalarTyped {
            _phantom_type: PhantomData::<T2>,
            offset: self.offset,
            symbol_id: self.symbol_id,
        }
    }

    #[allow(dead_code)]
    pub(crate) fn symbol_id(&self) -> u64 {
        self.symbol_id
    }

    #[allow(dead_code)]
    pub(crate) fn try_eq(&self, other: &Self) -> Option<bool> {
        if self.symbol_id == other.symbol_id {
            Some(self.offset == other.offset)
        } else {
            None
        }
    }

    #[allow(dead_code)]
    pub(crate) fn add_offset(&self, offset: i64) -> Self {
        Self {
            offset: self.offset + offset,
            symbol_id: self.symbol_id,
            _phantom_type: PhantomData,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SymbolicScalar {
    offset: i64,
    dtype: NumericDType,
    symbol_id: u64,
}

impl SymbolicScalar {
    pub(crate) fn new(dtype: NumericDType, rng: &mut impl Rng) -> Self {
        SymbolicScalar {
            offset: 0,
            dtype,
            symbol_id: rng.next_u64(),
        }
    }

    pub(crate) fn dtype(&self) -> NumericDType {
        self.dtype
    }

    pub fn try_eq(&self, other: &Self) -> Option<bool> {
        if self.symbol_id == other.symbol_id {
            Some(self.offset == other.offset)
        } else {
            None
        }
    }

    #[allow(dead_code)]
    pub(crate) fn cast<T>(&self) -> SymbolicScalarTyped<T>
    where
        T: Copy + Clone + NumericPrimitive,
    {
        SymbolicScalarTyped {
            _phantom_type: PhantomData,
            offset: self.offset,
            symbol_id: self.symbol_id,
        }
    }
}
