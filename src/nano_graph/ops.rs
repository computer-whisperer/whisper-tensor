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

/// A scalar operation with its precision semantics.
///
/// Each variant carries the dtype configuration for how it executes:
/// - `compute_dtype`: precision inputs are cast to and the operation executes at.
/// - `output_dtype`: precision the result is stored at (cast after computation).
///
/// This allows each op to independently describe its numeric behavior,
/// and ops like ReduceSum can later grow op-specific knobs (e.g. accumulation
/// ordering) without affecting other variants.
#[derive(Debug, Clone)]
pub enum ScalarOp {
    /// Produce a constant value. No inputs. The NumericScalar carries
    /// the exact typed value (BF16, F32, etc.).
    Literal(NumericScalar),
    /// Identity pass-through. One input, output = cast(input).
    /// Used for index-remapping ops (Slice, strided views) and dtype casts.
    Identity {
        compute_dtype: DType,
        output_dtype: DType,
    },
    /// Binary operation on two inputs.
    Binary {
        op: ScalarBinOp,
        compute_dtype: DType,
        output_dtype: DType,
    },
    /// Unary operation on one input.
    Unary {
        op: ScalarUnaryOp,
        compute_dtype: DType,
        output_dtype: DType,
    },
    /// Ternary select: condition ? x : y. Three inputs: [condition, x, y].
    Select {
        compute_dtype: DType,
        output_dtype: DType,
    },
    /// Reduce by summation over a known number of steps.
    /// One input (Affine or similar); the group iterates k=0..reduce_count,
    /// reading input at offset k*reduce_stride from the resolved base.
    ReduceSum {
        reduce_count: u64,
        reduce_stride: i64,
        compute_dtype: DType,
        output_dtype: DType,
    },
    /// Reduce by max over a known number of steps.
    ReduceMax {
        reduce_count: u64,
        reduce_stride: i64,
        compute_dtype: DType,
        output_dtype: DType,
    },
    // Note: if you add new ScalarOp variants, update compute_dtype(), output_dtype(),
    // is_reduce(), and all match sites in eval.rs, nano_execute.rs, nano_codegen.rs,
    // nano_part_*.rs, pattern.rs stats/validate.
    /// Indirect load: given a runtime-computed index (one input), read a value
    /// from a known table of atoms at `table_base + index`. Used for Gather
    /// (embedding lookups). No computation, just a runtime-dependent load.
    IndirectLoad {
        table_base: super::pattern::AtomId,
        output_dtype: DType,
    },
}

impl ScalarOp {
    /// Returns the compute dtype for this op (None for Literal and IndirectLoad, which have no computation).
    pub fn compute_dtype(&self) -> Option<DType> {
        match self {
            ScalarOp::Literal(_) | ScalarOp::IndirectLoad { .. } => None,
            ScalarOp::Identity { compute_dtype, .. }
            | ScalarOp::Binary { compute_dtype, .. }
            | ScalarOp::Unary { compute_dtype, .. }
            | ScalarOp::Select { compute_dtype, .. }
            | ScalarOp::ReduceSum { compute_dtype, .. }
            | ScalarOp::ReduceMax { compute_dtype, .. } => Some(*compute_dtype),
        }
    }

    /// Returns the output dtype for this op.
    pub fn output_dtype(&self) -> DType {
        match self {
            ScalarOp::Literal(s) => s.dtype(),
            ScalarOp::Identity { output_dtype, .. }
            | ScalarOp::Binary { output_dtype, .. }
            | ScalarOp::Unary { output_dtype, .. }
            | ScalarOp::Select { output_dtype, .. }
            | ScalarOp::ReduceSum { output_dtype, .. }
            | ScalarOp::ReduceMax { output_dtype, .. }
            | ScalarOp::IndirectLoad { output_dtype, .. } => *output_dtype,
        }
    }

    /// Returns true if this is a reduce operation.
    pub fn is_reduce(&self) -> bool {
        matches!(
            self,
            ScalarOp::ReduceSum { .. } | ScalarOp::ReduceMax { .. }
        )
    }
}
