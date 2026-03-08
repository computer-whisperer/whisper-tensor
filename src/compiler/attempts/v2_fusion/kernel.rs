//! Kernel IR for the v2 fusion compiler.
//!
//! Unlike v1 which unrolls to per-element nano ops and re-detects loops,
//! v2 builds loop structures directly from the milli graph. Consecutive
//! elementwise ops with compatible shapes are fused into a single kernel
//! so intermediates stay in registers.

use crate::graph::GlobalId;

/// Scalar binary operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarBinOp {
    Add,
    Sub,
    Mul,
    Div,
    Max,
    Min,
}

/// Scalar unary operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

/// One operation in a kernel body. References other body ops by index.
#[derive(Debug, Clone)]
pub enum BodyOp {
    /// Load from tensor at index = sum(loop_var[d] * strides[d]).
    Load {
        tensor: GlobalId,
        strides: Vec<isize>,
    },
    /// Store to tensor at index = sum(loop_var[d] * strides[d]).
    Store {
        tensor: GlobalId,
        strides: Vec<isize>,
        value_ref: usize,
    },
    /// Binary scalar op.
    BinOp {
        op: ScalarBinOp,
        a_ref: usize,
        b_ref: usize,
    },
    /// Unary scalar op.
    UnaryOp { op: ScalarUnaryOp, input_ref: usize },
    /// Literal constant.
    Literal { value: f64 },
}

/// A compiled kernel — one unit of work.
#[derive(Debug, Clone)]
pub enum KernelOp {
    /// Fused elementwise loop nest.
    Elementwise(ElementwiseKernel),
    /// Matrix multiplication.
    Gemm(GemmKernel),
}

/// Fused elementwise kernel with multi-dimensional iteration.
///
/// When all tensors have matching shapes (no broadcasting), dims is
/// collapsed to a single dimension for a flat loop.
#[derive(Debug, Clone)]
pub struct ElementwiseKernel {
    /// Iteration space dimensions (outermost first).
    pub dims: Vec<usize>,
    /// Body operations executed per iteration.
    pub body: Vec<BodyOp>,
}

/// Matrix multiplication: C[m,n] = sum_k A[m,k] * B[k,n].
///
/// Row-major layout assumed. For batched matmul, an outer loop
/// over batch_size advances A, B, C by their batch strides.
#[derive(Debug, Clone)]
pub struct GemmKernel {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub a: GlobalId,
    pub b: GlobalId,
    pub c: GlobalId,
    /// Number of batch instances (1 for non-batched).
    pub batch_size: usize,
    /// Elements between consecutive A matrices.
    pub a_batch_stride: usize,
    /// Elements between consecutive B matrices.
    pub b_batch_stride: usize,
    /// Elements between consecutive C matrices.
    pub c_batch_stride: usize,
}

/// Summary statistics from planning.
#[derive(Debug)]
pub struct PlanStats {
    pub num_kernels: usize,
    pub num_elementwise: usize,
    pub num_gemm: usize,
    pub total_fused_ops: usize,
    pub total_loop_dims: usize,
}

pub fn stats(kernels: &[KernelOp]) -> PlanStats {
    let mut num_elementwise = 0;
    let mut num_gemm = 0;
    let mut total_fused_ops = 0;
    let mut total_loop_dims = 0;

    for k in kernels {
        match k {
            KernelOp::Elementwise(ek) => {
                num_elementwise += 1;
                total_fused_ops += ek
                    .body
                    .iter()
                    .filter(|op| matches!(op, BodyOp::BinOp { .. } | BodyOp::UnaryOp { .. }))
                    .count();
                total_loop_dims += ek.dims.iter().product::<usize>();
            }
            KernelOp::Gemm(_) => {
                num_gemm += 1;
            }
        }
    }

    PlanStats {
        num_kernels: kernels.len(),
        num_elementwise,
        num_gemm,
        total_fused_ops,
        total_loop_dims,
    }
}
