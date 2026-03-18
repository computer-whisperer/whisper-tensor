//! Scalar operations for the nano graph.

use crate::dtype::DType;
use crate::numeric_scalar::NumericScalar;

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
    Pow,
    // Comparison ops — return 1.0 for true, 0.0 for false (ONNX convention).
    Equal,
    Greater,
    GreaterOrEqual,
    Less,
    LessOrEqual,
    // Logical ops — treat nonzero as true, return 1.0/0.0.
    And,
    Or,
    Xor,
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

/// Reduction accumulator kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceKind {
    Sum,
    Max,
}

/// A scalar operation performed by each atom in a group.
///
/// The output dtype lives on the `AtomGroup`, not here. Variants that
/// perform arithmetic carry a `compute_dtype` specifying the precision
/// inputs are cast to before the operation executes.
#[derive(Debug, Clone)]
pub enum ScalarOp {
    /// Produce a constant value. No inputs. The NumericScalar carries
    /// the exact typed value (BF16, F32, etc.).
    Literal(NumericScalar),
    /// Identity pass-through. One input, output = cast(input, output_dtype).
    /// Used for index-remapping ops (Slice, strided views) and dtype casts.
    /// The cast target is the group's `output_dtype`.
    Identity,
    /// Binary operation on two inputs.
    Binary {
        op: ScalarBinOp,
        compute_dtype: DType,
    },
    /// Unary operation on one input.
    Unary {
        op: ScalarUnaryOp,
        compute_dtype: DType,
    },
    /// Ternary select: condition ? x : y. Three inputs: [condition, x, y].
    Select,
    /// Reduce by accumulation over a known number of steps.
    /// One input; the group iterates k=0..reduce_count,
    /// reading input at offset k*reduce_stride from the resolved base.
    Reduce {
        kind: ReduceKind,
        reduce_count: u64,
        reduce_stride: i64,
        compute_dtype: DType,
    },
    // Note: if you add new ScalarOp variants, update is_reduce() and all
    // match sites in eval.rs, nano_codegen.rs, pattern.rs stats/validate.
    /// Indirect load: given a runtime-computed index (one input), read a value
    /// from a known table of atoms at `table_base + index`. Used for Gather
    /// (embedding lookups). No computation, just a runtime-dependent load.
    IndirectLoad {
        table_base: super::pattern::AtomId,
    },
}

impl ScalarOp {
    /// Returns the compute dtype for this op (None for ops with no arithmetic).
    pub fn compute_dtype(&self) -> Option<DType> {
        match self {
            ScalarOp::Literal(_)
            | ScalarOp::Identity
            | ScalarOp::Select
            | ScalarOp::IndirectLoad { .. } => None,
            ScalarOp::Binary { compute_dtype, .. }
            | ScalarOp::Unary { compute_dtype, .. }
            | ScalarOp::Reduce { compute_dtype, .. } => Some(*compute_dtype),
        }
    }

    /// Returns true if this is a reduce operation.
    pub fn is_reduce(&self) -> bool {
        matches!(self, ScalarOp::Reduce { .. })
    }
}
