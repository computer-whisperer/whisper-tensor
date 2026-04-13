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
    /// C remainder (truncated division) — result sign matches dividend.
    /// ONNX `Mod` with `fmod=1`, or float mod.
    Mod,
    /// Mathematical modulo — result sign matches divisor.
    /// ONNX `Mod` with `fmod=0` (default for integers).
    IMod,
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
    // Bitwise ops — operate on raw integer bits.
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    BitShiftLeft,
    BitShiftRight,
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
    Min,
    Prod,
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
    /// Full dim layout including symbolic dims. Pool_eval resolves
    /// Sym dims via gc_values to build the concrete shape at runtime.
    pub dims: Vec<crate::nano_graph::lower::DimKind>,
    pub dtype: NumericDType,
}

impl OpaqueTensorMapping {
    /// Concrete known-dim shape (for backward compat / non-sym paths).
    pub fn known_shape(&self) -> Vec<u64> {
        self.dims
            .iter()
            .filter_map(|d| match d {
                crate::nano_graph::lower::DimKind::Known { size, .. } => Some(*size),
                _ => None,
            })
            .collect()
    }

    /// Sym dims from the dim layout.
    pub fn sym_dims(&self) -> Vec<super::pattern::GraphConstantId> {
        self.dims
            .iter()
            .filter_map(|d| match d {
                crate::nano_graph::lower::DimKind::Sym { gc, .. } => Some(*gc),
                _ => None,
            })
            .collect()
    }

    /// Build full concrete shape by resolving sym dims from gc_values.
    pub fn full_shape(&self, gc_values: &[u64]) -> Vec<u64> {
        self.dims
            .iter()
            .map(|dk| match dk {
                crate::nano_graph::lower::DimKind::Known { size, .. } => *size,
                crate::nano_graph::lower::DimKind::Sym { gc, .. } => gc_values[gc.0 as usize],
            })
            .collect()
    }
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
    ) -> Result<
        Vec<NumericTensor<'static, DynRank, crate::pool::SystemPool>>,
        crate::nano_graph::pool_eval::PoolEvalError,
    >;
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
///
/// Generic over `'p` and `P: Pool` to allow `LiteralSpan` to hold a
/// pool-backed tensor.
pub enum ScalarOp<'p, P: Pool + 'p = crate::pool::SystemPool> {
    /// Produce a constant value. No inputs. The NumericScalar carries
    /// the exact typed value (BF16, F32, etc.). All atoms in the group
    /// produce this same value (broadcast).
    Literal(NumericScalar),
    /// Produce constant values from a 1D pool-backed tensor. No inputs.
    /// Atom `i` reads element `i` from the tensor. The group's `count`
    /// must equal `tensor.numel()`.
    LiteralSpan(NumericTensor<'p, DynRank, P>),
    /// Identity pass-through. One input, output = cast(input, output_dtype).
    /// Used for index-remapping ops (Slice, strided views) and non-cast
    /// dtype reinterpretations.
    Identity,
    /// Explicit dtype cast. One input, output = cast(input, output_dtype).
    /// If `saturating` is true, overflow clamps to ±max_finite instead of ±inf.
    /// ONNX Cast defaults to saturating for float8 targets.
    Cast { saturating: bool },
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
    /// Reduce over a symbolic dimension axis.  One input; the group's
    /// `sym_dims[axis]` GraphConstant gives the runtime reduction extent.
    /// The output group has one fewer sym_dim than the input (the axis
    /// at `axis` is collapsed).
    SymReduce {
        kind: ReduceKind,
        /// Index into the **input** group's `sym_dims` that is reduced.
        axis: usize,
        compute_dtype: NumericDType,
    },
    // Note: if you add new ScalarOp variants, update is_reduce() and all
    // match sites in eval.rs, nano_codegen.rs, pattern.rs stats/validate.
    /// Indirect load: given a runtime-computed index (one input), read a value
    /// from a known table of atoms at `table_base + index`. Used for Gather
    /// (embedding lookups). No computation, just a runtime-dependent load.
    ///
    /// `index_range` is the upper bound on the runtime index: the op will
    /// only ever read atoms in `[table_base, table_base + index_range)`.
    /// Populated at construction time by lowering (which knows the table's
    /// size), used by the memory placer to know the exact atom range the
    /// op may access without walking graph state.
    IndirectLoad {
        table_base: super::pattern::AtomId,
        index_range: u64,
    },
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

// Manual impls since derive can't handle the pool generic cleanly.

impl<P: Pool> std::fmt::Debug for ScalarOp<'_, P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScalarOp::Literal(s) => f.debug_tuple("Literal").field(s).finish(),
            ScalarOp::LiteralSpan(t) => f
                .debug_tuple("LiteralSpan")
                .field(&format_args!("[{}; {}]", t.dtype(), t.numel()))
                .finish(),
            ScalarOp::Identity => write!(f, "Identity"),
            ScalarOp::Cast { saturating } => f
                .debug_struct("Cast")
                .field("saturating", saturating)
                .finish(),
            ScalarOp::Binary { op, compute_dtype } => f
                .debug_struct("Binary")
                .field("op", op)
                .field("compute_dtype", compute_dtype)
                .finish(),
            ScalarOp::Unary { op, compute_dtype } => f
                .debug_struct("Unary")
                .field("op", op)
                .field("compute_dtype", compute_dtype)
                .finish(),
            ScalarOp::Select => write!(f, "Select"),
            ScalarOp::Reduce {
                kind,
                reduce_count,
                reduce_stride,
                compute_dtype,
            } => f
                .debug_struct("Reduce")
                .field("kind", kind)
                .field("reduce_count", reduce_count)
                .field("reduce_stride", reduce_stride)
                .field("compute_dtype", compute_dtype)
                .finish(),
            ScalarOp::SymReduce {
                kind,
                axis,
                compute_dtype,
            } => f
                .debug_struct("SymReduce")
                .field("kind", kind)
                .field("axis", axis)
                .field("compute_dtype", compute_dtype)
                .finish(),
            ScalarOp::IndirectLoad {
                table_base,
                index_range,
            } => f
                .debug_struct("IndirectLoad")
                .field("table_base", table_base)
                .field("index_range", index_range)
                .finish(),
            ScalarOp::OpaqueOutput {
                opaque_idx,
                output_idx,
            } => f
                .debug_struct("OpaqueOutput")
                .field("opaque_idx", opaque_idx)
                .field("output_idx", output_idx)
                .finish(),
        }
    }
}

impl<'p, P: Pool + 'p> Clone for ScalarOp<'p, P>
where
    P::Buffer<'p>: Clone,
{
    fn clone(&self) -> Self {
        match self {
            ScalarOp::Literal(s) => ScalarOp::Literal(*s),
            ScalarOp::LiteralSpan(t) => ScalarOp::LiteralSpan(t.clone()),
            ScalarOp::Identity => ScalarOp::Identity,
            ScalarOp::Cast { saturating } => ScalarOp::Cast {
                saturating: *saturating,
            },
            ScalarOp::Binary { op, compute_dtype } => ScalarOp::Binary {
                op: *op,
                compute_dtype: *compute_dtype,
            },
            ScalarOp::Unary { op, compute_dtype } => ScalarOp::Unary {
                op: *op,
                compute_dtype: *compute_dtype,
            },
            ScalarOp::Select => ScalarOp::Select,
            ScalarOp::Reduce {
                kind,
                reduce_count,
                reduce_stride,
                compute_dtype,
            } => ScalarOp::Reduce {
                kind: *kind,
                reduce_count: *reduce_count,
                reduce_stride: *reduce_stride,
                compute_dtype: *compute_dtype,
            },
            ScalarOp::SymReduce {
                kind,
                axis,
                compute_dtype,
            } => ScalarOp::SymReduce {
                kind: *kind,
                axis: *axis,
                compute_dtype: *compute_dtype,
            },
            ScalarOp::IndirectLoad {
                table_base,
                index_range,
            } => ScalarOp::IndirectLoad {
                table_base: *table_base,
                index_range: *index_range,
            },
            ScalarOp::OpaqueOutput {
                opaque_idx,
                output_idx,
            } => ScalarOp::OpaqueOutput {
                opaque_idx: *opaque_idx,
                output_idx: *output_idx,
            },
        }
    }
}

impl<'p, P: Pool + 'p> ScalarOp<'p, P> {
    /// Returns the compute dtype for this op (None for ops with no arithmetic).
    pub fn compute_dtype(&self) -> Option<NumericDType> {
        match self {
            ScalarOp::Literal(_)
            | ScalarOp::LiteralSpan(_)
            | ScalarOp::Identity
            | ScalarOp::Cast { .. }
            | ScalarOp::Select
            | ScalarOp::IndirectLoad { .. }
            | ScalarOp::OpaqueOutput { .. } => None,
            ScalarOp::Binary { compute_dtype, .. }
            | ScalarOp::Unary { compute_dtype, .. }
            | ScalarOp::Reduce { compute_dtype, .. }
            | ScalarOp::SymReduce { compute_dtype, .. } => Some(*compute_dtype),
        }
    }

    /// Returns true if this is a reduce operation.
    pub fn is_reduce(&self) -> bool {
        matches!(self, ScalarOp::Reduce { .. } | ScalarOp::SymReduce { .. })
    }
}
