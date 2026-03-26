//! Scalar operations for the nano graph.

use std::sync::Arc;

use crate::numeric_dtype::NumericDType;
use crate::numeric_scalar::NumericScalar;
use crate::numeric_tensor::{NumericTensor, NumericTensorView};
use crate::pool::Pool;
use crate::tensor_rank::DynRank;

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
    Round,
    Sign,
    Not,
    IsNan,
    Erf,
    Sin,
    Cos,
    IsInf {
        detect_positive: bool,
        detect_negative: bool,
    },
    BitwiseNot,
    Log1p,
    Tan,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
    Asinh,
    Acosh,
    Atanh,
}

/// Reduction accumulator kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceKind {
    Sum,
    Max,
}

// ---------------------------------------------------------------------------
// Opaque milli-op support
// ---------------------------------------------------------------------------

/// An opaque milli-level operation that can't be decomposed into scalar nano-ops.
///
/// Lives in `NanoGraph::opaque_ops`. AtomGroups with `ScalarOp::OpaqueOutput`
/// reference these by index. The evaluator assembles input tensors from atom
/// buffers, calls the eval function, and scatters results back.
#[derive(Clone)]
pub struct OpaqueOp {
    /// The evaluation function.
    pub eval_fn: Arc<dyn OpaqueEval>,
    /// Input tensor mappings — where to read input data from the atom space.
    pub inputs: Vec<OpaqueTensorMapping>,
    /// Output tensor mappings — where to write results in the atom space.
    /// Each output corresponds to one AtomGroup with ScalarOp::OpaqueOutput.
    pub outputs: Vec<OpaqueTensorMapping>,
    /// Human-readable name for debugging.
    pub name: String,
}

/// Mapping between an atom range and a tensor shape.
#[derive(Clone, Debug)]
pub struct OpaqueTensorMapping {
    pub base: super::pattern::AtomId,
    pub count: u64,
    pub shape: Vec<u64>,
    pub dtype: NumericDType,
}

/// Trait for opaque op evaluation. Operates on new pool-backed types only.
///
/// Returns SystemPool-backed tensors for dyn-compatibility. The evaluator
/// copies elements into its own pool as needed.
pub trait OpaqueEval: Send + Sync {
    /// Evaluate the op given input tensor views.
    /// Returns one output tensor per output mapping.
    fn eval(
        &self,
        inputs: &[NumericTensorView<'_, DynRank>],
    ) -> Result<Vec<NumericTensor<'static, DynRank, crate::pool::SystemPool>>, crate::nano_graph::pool_eval::PoolEvalError>;
}

impl std::fmt::Debug for OpaqueOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpaqueOp")
            .field("name", &self.name)
            .field("inputs", &self.inputs)
            .field("outputs", &self.outputs)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Scalar operations
// ---------------------------------------------------------------------------

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
        compute_dtype: NumericDType,
    },
    /// Unary operation on one input.
    Unary {
        op: ScalarUnaryOp,
        compute_dtype: NumericDType,
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
        compute_dtype: NumericDType,
    },
    // Note: if you add new ScalarOp variants, update is_reduce() and all
    // match sites in eval.rs, nano_codegen.rs, pattern.rs stats/validate.
    /// Indirect load: given a runtime-computed index (one input), read a value
    /// from a known table of atoms at `table_base + index`. Used for Gather
    /// (embedding lookups). No computation, just a runtime-dependent load.
    IndirectLoad { table_base: super::pattern::AtomId },
    /// Output of an opaque milli-op. The evaluator looks up the opaque op
    /// by index in `NanoGraph::opaque_ops`, calls it (once, caching results
    /// across all output groups), and reads this group's portion of the output.
    OpaqueOutput {
        /// Index into `NanoGraph::opaque_ops`.
        opaque_idx: usize,
        /// Which output of the opaque op this group corresponds to.
        output_idx: usize,
    },
}

impl ScalarOp {
    /// Returns the compute dtype for this op (None for ops with no arithmetic).
    pub fn compute_dtype(&self) -> Option<NumericDType> {
        match self {
            ScalarOp::Literal(_)
            | ScalarOp::Identity
            | ScalarOp::Select
            | ScalarOp::IndirectLoad { .. }
            | ScalarOp::OpaqueOutput { .. } => None,
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
