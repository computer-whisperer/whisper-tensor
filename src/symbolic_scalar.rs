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

    /// Construct with an explicit symbol id. Used by the model-execution
    /// symbolic-input-dim config where deterministic ids (derived from the
    /// user-provided group name) are required to get stable cache keys —
    /// random ids from `new` would invalidate the lowered/compiled caches
    /// on every call even for identical configs.
    pub(crate) fn from_symbol_id(symbol_id: u64) -> Self {
        Self {
            _phantom_type: PhantomData,
            offset: 0,
            symbol_id,
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

    /// Construct an untyped SymbolicScalar with the given dtype that shares
    /// identity (symbol_id, offset) with a typed symbolic scalar. Used to
    /// thread sym identity across the typed/untyped boundary — e.g. a
    /// shape-dim `SymbolicScalarTyped<u64>` can become a
    /// `SymbolicScalar { dtype: I64, .. }` for a Shape op's per-element
    /// output, which downstream ops can cast back to `<u64>` for a
    /// reconstructed shape.
    pub(crate) fn from_typed<T>(typed: &SymbolicScalarTyped<T>, dtype: NumericDType) -> Self
    where
        T: Copy + Clone + NumericPrimitive,
    {
        SymbolicScalar {
            offset: typed.offset,
            dtype,
            symbol_id: typed.symbol_id,
        }
    }
}
