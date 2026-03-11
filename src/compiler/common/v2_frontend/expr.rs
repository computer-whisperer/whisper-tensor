//! Core expression types for the v2 frontend.

use crate::graph::GlobalId;

/// Scalar binary operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalarBinOp {
    Add,
    Sub,
    Mul,
    Div,
    Max,
    Min,
}

/// Scalar unary operations.
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
    Erf,
    /// Round F32 value to BF16 precision in-register.
    /// Inserted by the inlining pass when fusing across BF16 intermediate boundaries.
    RoundBf16,
}

/// A pure scalar ALU expression tree.
///
/// Leaves are either `Element` (a reference to a specific element of a tensor
/// buffer) or `Literal` (a compile-time constant). Internal nodes are binary
/// and unary scalar operations.
///
/// No Load/Store — this is a declarative description of computation, not a
/// schedule. The downstream pass decides when and where to materialize loads.
#[derive(Debug, Clone)]
pub enum ScalarExpr {
    /// Reference to element `flat_index` of tensor `tensor`.
    Element { tensor: GlobalId, flat_index: usize },
    /// Compile-time constant (stored as f64 bits for exact representation).
    Literal { value: f64 },
    /// Binary scalar operation.
    Binary {
        op: ScalarBinOp,
        a: Box<ScalarExpr>,
        b: Box<ScalarExpr>,
    },
    /// Unary scalar operation.
    Unary {
        op: ScalarUnaryOp,
        input: Box<ScalarExpr>,
    },
}

/// An output binding: one output tensor element = one scalar expression.
#[derive(Debug, Clone)]
pub struct OutputBinding {
    /// Which output tensor this element belongs to.
    pub output_tensor: GlobalId,
    /// Flat (row-major) index into the output tensor.
    pub flat_index: usize,
    /// The expression tree computing this element's value.
    pub expr: ScalarExpr,
}
