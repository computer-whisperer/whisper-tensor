//! Scalar operations for the nano graph.

use crate::dtype::DType;

/// Binary scalar operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalarBinOp {
    Add,
    Sub,
    Mul,
    Div,
    Max,
    Min,
    Mod,
}

/// Unary scalar operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalarUnaryOp {
    Neg,
    Abs,
    Exp,
    Ln,
    Sqrt,
    Reciprocal,
    Tanh,
    Floor,
    Ceil,
}

/// A scalar operation — the computation performed by a pattern block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalarOp {
    /// Binary operation on two inputs.
    Binary(ScalarBinOp),
    /// Unary operation on one input.
    Unary(ScalarUnaryOp),
    /// Identity pass-through. One input, output = input.
    /// Used for index-remapping ops (Slice, strided views) where the
    /// computation is trivial but the affine map changes.
    Identity,
    /// Produce a constant value. No inputs.
    Literal(LiteralBits),
    /// Reduce a dimension by summation. One input; the block iterates over the
    /// reduction dimension and accumulates.
    ReduceSum,
    /// Reduce a dimension by max.
    ReduceMax,
}

/// A floating-point literal stored as raw bits so it can be Eq + Hash.
#[derive(Clone, Copy)]
pub struct LiteralBits {
    pub bits: u64,
    pub dtype: DType,
}

impl LiteralBits {
    pub fn f64(value: f64) -> Self {
        Self {
            bits: value.to_bits(),
            dtype: DType::F64,
        }
    }

    pub fn f32(value: f32) -> Self {
        Self {
            bits: value.to_bits() as u64,
            dtype: DType::F32,
        }
    }

    pub fn as_f64(&self) -> f64 {
        match self.dtype {
            DType::F64 => f64::from_bits(self.bits),
            DType::F32 => f32::from_bits(self.bits as u32) as f64,
            _ => panic!("LiteralBits::as_f64 on non-float dtype {:?}", self.dtype),
        }
    }
}

impl std::fmt::Debug for LiteralBits {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Literal({}, {:?})", self.as_f64(), self.dtype)
    }
}

impl PartialEq for LiteralBits {
    fn eq(&self, other: &Self) -> bool {
        self.bits == other.bits && self.dtype == other.dtype
    }
}

impl Eq for LiteralBits {}

impl std::hash::Hash for LiteralBits {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.bits.hash(state);
        self.dtype.hash(state);
    }
}
