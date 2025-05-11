use std::ops::{Add, Mul, Not};
use ndarray::{concatenate, s, ArcArray, Array, Array2, ArrayD, ArrayView2, ArrayViewD, ArrayViewMut2, ArrayViewMutD, Axis, Dimension, Ix1, Ix2, IxDyn, LinalgScalar, ScalarOperand, ShapeError, SliceInfoElem, Zip};
use ndarray::linalg::{general_mat_mul, Dot};
use num_traits::{Float, FromPrimitive, Num, NumCast, One, Zero};
use num_traits::real::Real;
use serde::{Deserialize, Serialize};
use crate::ndarray_backend::ops::NDArrayOperationError::UnimplementedOp;
use crate::TrigOp;

#[derive(Debug, thiserror::Error)]
pub enum NDArrayOperationError {
    #[error(transparent)]
    ShapeError(#[from] ShapeError),
    #[error("out of bounds")]
    OutOfBounds,
    #[error("incompatible shape")]
    IncompatibleShape,
    #[error("shape mismatch: {0}")]
    IncompatibleShapes(String),
    #[error("broadcast error: {0}")]
    BroadcastError(String),
    #[error("unsupported operation")]
    UnimplementedOp(String),
    #[error("Internal error")]
    Internal
}

pub(crate) fn reshape<T, D: Dimension>(input: ArcArray<T, D>, shape: D) -> Result<ArcArray<T, D>, ShapeError>
where
    T: Clone,
{
    Ok(input.to_shape(shape)?.to_shared())
}

/// Gather elements from `tensor` along axis `dim` according to `indices`,
/// following the ONNX spec (axis may be negative; out‐of‐bounds raises).
/// Output shape = data.shape[..dim] ++ indices.shape ++ data.shape[dim+1..].
///
/// # Errors
/// - If `dim` ≥ tensor.ndim()
/// - If any index (after handling negatives) is out of bounds
/// - If the final buffer length mismatches the computed shape
pub(crate) fn gather<T: Clone>(
    dim: usize,
    tensor: ArcArray<T, IxDyn>,
    indices: ArcArray<i64, IxDyn>,
) -> Result<ArcArray<T, IxDyn>, NDArrayOperationError> {
    let data_shape = tensor.shape();
    let rank = data_shape.len();
    if dim >= rank {
        return Err(NDArrayOperationError::OutOfBounds);
    }

    // ONNX allows negative axis, but we require dim non‐negative here.
    let axis_len = data_shape[dim];

    // 1) Compute output shape
    let idx_shape = indices.shape();
    let mut out_shape = Vec::with_capacity(rank - 1 + idx_shape.len());
    out_shape.extend(&data_shape[..dim]);
    out_shape.extend(idx_shape);
    out_shape.extend(&data_shape[dim+1..]);
    let out_len = out_shape.iter().product::<usize>();

    // 2) Compute row‐major strides for output
    let ndim = out_shape.len();
    let mut strides = vec![0; ndim];
    {
        let mut stride = 1usize;
        for i in (0..ndim).rev() {
            strides[i] = stride;
            // check overflow just in case
            stride = stride
                .checked_mul(out_shape[i])
                .ok_or_else(||NDArrayOperationError::OutOfBounds)?;
        }
    }

    // 3) Flatten‐index loop
    let mut out_buf = Vec::with_capacity(out_len);
    for flat in 0..out_len {
        // a) compute multi‐index in the output
        let mut rem = flat;
        let mut idx_multi = Vec::with_capacity(ndim);
        for &st in &strides {
            idx_multi.push(rem / st);
            rem %= st;
        }

        // b) extract the slice that indexes into `indices`
        let idx_inds = &idx_multi[dim .. dim + idx_shape.len()];
        let mut ix = indices[IxDyn(idx_inds)];
        // handle negative
        if ix < 0 {
            ix += axis_len as i64;
        }
        if ix < 0 || ix >= axis_len as i64 {
            return Err(NDArrayOperationError::OutOfBounds);
        }
        let ix = ix as usize;

        // c) build the corresponding index into `tensor`
        let mut data_idx = Vec::with_capacity(rank);
        data_idx.extend(&idx_multi[..dim]);
        data_idx.push(ix);
        data_idx.extend(&idx_multi[dim + idx_shape.len()..]);

        // d) fetch & clone
        out_buf.push(tensor[IxDyn(&data_idx)].clone());
    }

    // 4) Reconstruct the ArrayD and wrap in ArcArray
    let out_array = ArrayD::from_shape_vec(IxDyn(&out_shape), out_buf)?;
    Ok(ArcArray::from(out_array))
}

/// Concatenate ONNX‐style along `dim` (must be 0 ≤ dim < rank of each input),
/// for a nonempty slice of dynamic‐shaped `ArcArray<T, IxDyn>`.
///
/// # Errors
/// - if `inputs` is empty
/// - if any input has a different rank or mismatched shape off `dim`
/// - if `dim` is out of bounds
/// - if the underlying `ndarray::concatenate` fails (e.g. overflow)
pub fn concat<T: Clone>(
    dim: usize,
    inputs: &[ArcArray<T, IxDyn>],
) -> Result<ArcArray<T, IxDyn>, NDArrayOperationError> {
    // 1) Must have at least one input
    if inputs.is_empty() {
        return Err(NDArrayOperationError::OutOfBounds);
    }

    // 2) All inputs must share the same rank...
    let rank = inputs[0].ndim();
    if dim >= rank {
        return Err(NDArrayOperationError::OutOfBounds);
    }
    // ...and the same shape on every axis ≠ dim.
    let mut out_shape = inputs[0].shape().to_vec();
    // We'll accumulate the new size along `dim`:
    let mut axis_sum = 0;
    for arr in inputs {
        if arr.ndim() != rank {
            return Err(NDArrayOperationError::IncompatibleShape);
        }
        for (d, &len) in arr.shape().iter().enumerate() {
            if d == dim {
                axis_sum += len;
            } else if out_shape[d] != len {
                return Err(NDArrayOperationError::IncompatibleShape);
            }
        }
    }
    out_shape[dim] = axis_sum;

    // 3) Build a Vec of ArrayViewDs and call ndarray::concatenate
    let views: Vec<ArrayViewD<T>> = inputs.iter().map(|a| a.view()).collect();
    let concatenated: ArrayD<T> = concatenate(Axis(dim), &views)?;

    // 4) Wrap in ArcArray and return
    Ok(ArcArray::from(concatenated.into_dyn()))
}

/// ONNX NonZero: returns a [r,n] int64 tensor of the coordinates of each nonzero element.
/// - r = input.rank
/// - n = number of elements != 0
///
/// # Errors
/// - if the built buffer doesn’t match the computed [r,n]
///
pub fn nonzero<T>(
    tensor: ArcArray<T, IxDyn>,
) -> Result<ArcArray<i64, IxDyn>, NDArrayOperationError>
where
    T: Clone + PartialEq + Default + Copy,
{
    let rank = tensor.ndim();

    // 1) Gather each coordinate component into its own Vec<i64>
    let mut coords_by_dim: Vec<Vec<i64>> = vec![Vec::new(); rank];
    for (idx, &v) in tensor.indexed_iter() {
        if v != T::default() {
            let iv = idx.as_array_view();
            for (d, &loc) in iv.iter().enumerate() {
                coords_by_dim[d].push(loc as i64);
            }
        }
    }

    // 2) Number of nonzero elements
    let n = coords_by_dim.get(0).map(|v| v.len()).unwrap_or(0);

    // 3) Flatten in row-major order for shape [rank, n]
    let mut flat = Vec::with_capacity(rank * n);
    for d in 0..rank {
        flat.extend(&coords_by_dim[d]);
    }

    // 4) Build and wrap
    let shape = IxDyn(&[rank, n]);
    let arr: ArrayD<i64> = ArrayD::from_shape_vec(shape.clone(), flat)?;
    Ok(ArcArray::from(arr))
}

/// ONNX Transpose: permutes the axes of `tensor` according to `perm`.
/// If `perm` is `None`, reverses the axes.
///
/// # Errors
/// - perm.len() != rank(input) → IncompatibleShape
/// - any perm entry out of bounds → OutOfBounds
/// - perm is not a true permutation → IncompatibleShape
pub fn transpose<T>(
    tensor: ArcArray<T, IxDyn>,
    perm: Option<Vec<i64>>,
) -> Result<ArcArray<T, IxDyn>, NDArrayOperationError>
where
    T: Clone,
{
    let shape = tensor.shape();
    let rank = shape.len();

    // 1) Build a Vec<usize> from `perm` or use default reversed axes
    let axes: Vec<usize> = if let Some(p) = perm {
        if p.len() != rank {
            return Err(NDArrayOperationError::IncompatibleShape);
        }
        p.iter()
            .map(|&x| {
                // handle negatives
                let idx = if x < 0 { x + rank as i64 } else { x };
                if idx < 0 || idx >= rank as i64 {
                    return Err(NDArrayOperationError::OutOfBounds);
                }
                Ok(idx as usize)
            })
            .collect::<Result<_,_>>()?
    } else {
        // default: reverse the axes
        (0..rank).rev().collect()
    };

    // 2) Validate it's a true permutation of [0..rank)
    let mut sorted = axes.clone();
    sorted.sort_unstable();
    if sorted != (0..rank).collect::<Vec<_>>() {
        return Err(NDArrayOperationError::IncompatibleShape);
    }

    Ok(tensor.permuted_axes(axes).into_shared())
}

pub(crate) enum ReduceOp {
    Sum,
    Mean,
    Prod
}

impl ReduceOp {
    pub fn apply<T>(&self, tensor: ArcArray<T, IxDyn>,
                    axes: Vec<usize>,
                    keepdims: bool) -> Result<ArcArray<T, IxDyn>, NDArrayOperationError>
    where
        T: Clone + Zero + NumCast + std::ops::Div<Output = T> + Copy + One
    {
        Ok(match self {
            ReduceOp::Sum => reduce_sum(tensor, axes, keepdims)?,
            ReduceOp::Mean => reduce_mean(tensor, axes, keepdims)?,
            ReduceOp::Prod => reduce_prod(tensor, axes, keepdims)?,
        })
    }
}


/// ONNX ReduceMean: compute the mean over `axes` (or all axes if `None`),
/// with `keepdims` controlling whether reduced dims remain as size 1.
///
/// # Parameters
/// - `tensor`: input tensor, rank r
/// - `axes`: Option<&[i64]> list of axes (may be negative). `None` ⇒ all axes.
///
/// # Errors
/// - `axes` out of range → `OutOfBounds`
/// - duplicate axes → `IncompatibleShape`
/// - final buffer mismatch → `ShapeError`
pub fn reduce_mean<T>(
    tensor: ArcArray<T, IxDyn>,
    axes: Vec<usize>,
    keepdims: bool,
) -> Result<ArcArray<T, IxDyn>, NDArrayOperationError>
where
    T: Clone + Zero + NumCast + std::ops::Div<Output = T> + Copy
{
    let input_shape = tensor.shape().to_vec();
    let rank = input_shape.len();

    // 1) Normalize axes list
    let mut ax = axes;

    // 1a) Deduplicate & sort descending (so reductions don’t shift later axes)
    {
        let mut sorted = ax.clone();
        sorted.sort_unstable();
        sorted.dedup();
        if sorted.len() != ax.len() {
            return Err(NDArrayOperationError::IncompatibleShape);
        }
        sorted.reverse();
        ax = sorted;
    }

    // 2) Start with an owned dynamic array
    let mut result: ArrayD<T> = tensor.view().to_owned().into_dyn();

    // 3) For each axis, sum & divide, then optionally re-insert the dim
    for &axis in &ax {
        if axis >= result.ndim() {
            continue;
        }
        
        // a) sum over this axis
        let summed = result.sum_axis(Axis(axis));
        // b) divide by the count along that axis
        let count = input_shape[axis];
        let divisor = T::from(count).unwrap();
        let meaned = summed.mapv(|v| v / divisor);

        // c) keep or drop the reduced dim
        result = if keepdims {
            meaned.insert_axis(Axis(axis)).into_dyn()
        } else {
            meaned.into_dyn()
        };
    }

    // 4) Wrap in ArcArray and return
    Ok(ArcArray::from(result))
}

pub fn reduce_sum<T>(
    tensor: ArcArray<T, IxDyn>,
    axes: Vec<usize>,
    keepdims: bool,
) -> Result<ArcArray<T, IxDyn>, NDArrayOperationError>
where
    T: Clone + Zero + NumCast + std::ops::Div<Output = T> + Copy
{
    
    let mut ax =  axes;
    
    // 1a) Deduplicate & sort descending (so reductions don’t shift later axes)
    {
        let mut sorted = ax.clone();
        sorted.sort_unstable();
        sorted.dedup();
        if sorted.len() != ax.len() {
            return Err(NDArrayOperationError::IncompatibleShape);
        }
        sorted.reverse();
        ax = sorted;
    }

    // 2) Start with an owned dynamic array
    let mut result: ArrayD<T> = tensor.view().to_owned().into_dyn();

    // 3) For each axis, sum & divide, then optionally re-insert the dim
    for &axis in &ax {
        if axis >= result.ndim() {
            continue;
        }
        
        // a) sum over this axis
        let summed = result.sum_axis(Axis(axis));

        // c) keep or drop the reduced dim
        result = if keepdims {
            summed.insert_axis(Axis(axis)).into_dyn()
        } else {
            summed.into_dyn()
        };
    }

    // 4) Wrap in ArcArray and return
    Ok(ArcArray::from(result))
}

pub fn reduce_prod<T>(
    tensor: ArcArray<T, IxDyn>,
    axes: Vec<usize>,
    keepdims: bool,
) -> Result<ArcArray<T, IxDyn>, NDArrayOperationError>
where
    T: Clone + Zero + NumCast + std::ops::Div<Output = T> + Copy + One
{

    // 1) Normalize axes list
    let mut ax = axes;

    // 1a) Deduplicate & sort descending (so reductions don’t shift later axes)
    {
        let mut sorted = ax.clone();
        sorted.sort_unstable();
        sorted.dedup();
        if sorted.len() != ax.len() {
            return Err(NDArrayOperationError::IncompatibleShape);
        }
        sorted.reverse();
        ax = sorted;
    }

    // 2) Start with an owned dynamic array
    let mut result: ArrayD<T> = tensor.view().to_owned().into_dyn();

    // 3) For each axis, sum & divide, then optionally re-insert the dim
    for &axis in &ax {
        if axis >= result.ndim() {
            continue;
        }
        
        // a) sum over this axis
        let summed = result.product_axis(Axis(axis));

        // c) keep or drop the reduced dim
        result = if keepdims {
            summed.insert_axis(Axis(axis)).into_dyn()
        } else {
            summed.into_dyn()
        };
    }

    // 4) Wrap in ArcArray and return
    Ok(ArcArray::from(result))
}


#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NativeNumericTensorBinaryOperation {
    Add,
    Sub,
    Mul,
    Div,
    Modulo
}

impl core::fmt::Display for NativeNumericTensorBinaryOperation {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:?}", self)
    }
}

fn try_multidirectional_broadcasting(a: &[usize], b: &[usize]) -> Result<Vec<usize>, NDArrayOperationError> {
    let mut a = a.to_vec();
    let mut b = b.to_vec();

    // Prepend dimensions to match
    while a.len() < b.len() {
        a.insert(0, 1);
    }
    while b.len() < a.len() {
        b.insert(0, 1);
    }
    let mut output_dims = vec![];
    for i in 0..a.len() {
        output_dims.push(
            if a[i] == b[i] {
                a[i].clone()
            }
            else {
                if a[i] == 1 {
                    b[i].clone()
                }
                else {
                    if b[i] == 1 {
                        a[i].clone()
                    }
                    else {
                        Err(NDArrayOperationError::IncompatibleShape)?
                    }
                }
            }
        );
    }

    Ok(output_dims)
}

impl NativeNumericTensorBinaryOperation {
    pub(crate) fn apply<'a, T: 'a>(&self, a: ArcArray<T, IxDyn>, b: ArcArray<T, IxDyn>) -> Result<ArcArray<T, IxDyn>, NDArrayOperationError> where
        T: Clone + Copy + std::ops::Add<Output = T> + std::ops::Sub<Output = T> + std::ops::Mul<Output = T> + std::ops::Div<Output = T> + std::ops::Rem<Output = T>,
    {
        let a = if let Some(a) = a.broadcast(b.shape()) {
            a
        } else {
            a.view()
        };

        let b = if let Some(b) = b.broadcast(a.shape()) {
            b
        }
        else {
            b.view()
        };

        let o: Array<T, IxDyn> = match self {
            NativeNumericTensorBinaryOperation::Add => (&a + &b).into(),
            NativeNumericTensorBinaryOperation::Sub => (&a - &b).into(),
             NativeNumericTensorBinaryOperation::Mul => (&a * &b).into(),
            NativeNumericTensorBinaryOperation::Div => (&a / &b).into(),
            NativeNumericTensorBinaryOperation::Modulo => (&a % &b).into(),
        };
        Ok(o.to_shared())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NativeNumericTensorUnaryOperation {
    Sigmoid,
    Trig(TrigOp),
    Exp,
    Log,
    Softplus
}

impl core::fmt::Display for NativeNumericTensorUnaryOperation {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl NativeNumericTensorUnaryOperation {
    pub fn apply<T, D: Dimension>(&self, a: ArcArray<T, D>) -> Result<ArcArray<T, D>, NDArrayOperationError>
    where
        T: Clone + num_traits::Float
    {
        Ok(match self {
            NativeNumericTensorUnaryOperation::Exp => a.mapv(|x| x.exp()).to_shared(),
            NativeNumericTensorUnaryOperation::Trig(trigop) => a.mapv(|x| trigop.apply(x)).to_shared(),
            NativeNumericTensorUnaryOperation::Sigmoid => a.mapv(|x| T::one() / ( T::one() + T::exp(-x))).to_shared(),
            NativeNumericTensorUnaryOperation::Softplus => a.mapv(|x| T::ln(T::exp(x) + T::one())).to_shared(),
            NativeNumericTensorUnaryOperation::Log => a.mapv(|x| x.ln()).to_shared(),
            _ => {
                Err(NDArrayOperationError::UnimplementedOp(format!("{self:?}")))?
            } 
        })
    }
}

pub(crate) fn pow<T, TI>(a: ArcArray<T, IxDyn>, b: ArcArray<TI, IxDyn>) -> Result<ArcArray<T, IxDyn>, NDArrayOperationError>
where
    T: num_traits::Pow<TI, Output = T>,
    T: Num + Clone,
    TI: Num + Clone
{
    let a = if let Some(a) = a.broadcast(b.shape()) {
        a
    } else {
        a.view()
    };

    let b = if let Some(b) = b.broadcast(a.shape()) {
        b
    }
    else {
        b.view()
    };

    let v = ndarray::Zip::from(a).and(b).map_collect(|a, b| a.clone().pow(b.clone()));

    Ok(v.to_shared())
}


/// ONNX Gemm: Y = alpha * A'·B' + beta * C
/// - A, B: must be 2-D (M×K) and (K×N) (after optional transposition)
/// - C: optional, broadcastable to (M,N)
/// - trans_a/trans_b: if true, transpose A/B first
pub fn gemm<T>(
    A: ArcArray<T, IxDyn>,
    B: ArcArray<T, IxDyn>,
    C: Option<ArcArray<T, IxDyn>>,
    alpha: T,
    beta: T,
    trans_a: bool,
    trans_b: bool,
) -> Result<ArcArray<T, IxDyn>, NDArrayOperationError>
where
    T: Clone + Zero + One + Mul<Output = T> + Add<Output = T> + LinalgScalar,
{
    // 1) Convert to 2-D views
    let Av = A.view().into_dimensionality::<Ix2>()?;
    let Bv = B.view().into_dimensionality::<Ix2>()?;
    let A2 = if trans_a { Av.reversed_axes() } else { Av };
    let B2 = if trans_b { Bv.reversed_axes() } else { Bv };

    // 2) Check inner dims
    let (m, k1) = A2.dim();
    let (k2, n) = B2.dim();
    if k1 != k2 {
        return Err(NDArrayOperationError::IncompatibleShape);
    }

    // 3) Prepare output buffer (m×n): if C is provided and broadcastable, clone it; else zeros
    let mut Cmat: Array2<T> = match C {
        Some(c_arr) => {
            c_arr.broadcast((m, n))
                .ok_or_else(|| NDArrayOperationError::IncompatibleShape)?
                .to_owned()
        }
        None => Array2::zeros((m, n)),
    };

    // 4) Call the BLAS‐style routine
    general_mat_mul(alpha, &A2, &B2, beta, &mut Cmat);

    // 5) Wrap back into dynamic ArcArray
    Ok(ArcArray::from(Cmat.into_dyn()))
}


/// Split `tensor` along `axis` into parts sized by `split` (or size=1 chunks if `split` is None).
/// Returns a Vec of sub-tensors, preserving shared ownership via ArcArray.
///
/// # Errors
/// - `axis` out of range → OutOfBounds
/// - any `split` value < 0 or sum(split) ≠ dim_size → IncompatibleShape
pub fn split<T>(
    tensor: ArcArray<T, IxDyn>,
    axis: Option<i64>,
    split: Option<&[i64]>,
) -> Result<Vec<ArcArray<T, IxDyn>>, NDArrayOperationError>
where
    T: Clone,
{
    let shape = tensor.shape().to_vec();
    let rank = shape.len();

    // 1) Normalize axis (default = 0, allow negative)
    let ax = axis.unwrap_or(0);
    let ax = if ax < 0 { ax + rank as i64 } else { ax } as usize;
    if ax >= rank {
        return Err(NDArrayOperationError::IncompatibleShape);
    }
    let axis_len = shape[ax];

    // 2) Determine chunk sizes
    let sizes: Vec<usize> = if let Some(sp) = split {
        let mut out = Vec::with_capacity(sp.len());
        for &x in sp {
            if x < 0 {
                return Err(NDArrayOperationError::IncompatibleShape);
            }
            out.push(x as usize);
        }
        if out.iter().sum::<usize>() != axis_len {
            return Err(NDArrayOperationError::IncompatibleShape);
        }
        out
    } else {
        // default → one-element chunks
        vec![1; axis_len]
    };

    // 3) Compute start offsets
    let mut offsets = Vec::with_capacity(sizes.len());
    let mut acc = 0;
    for &sz in &sizes {
        offsets.push(acc);
        acc += sz;
    }

    // 4) Slice out each chunk
    let mut outputs = Vec::with_capacity(sizes.len());
    let parent_view = tensor.view();
    for (&start, &sz) in offsets.iter().zip(&sizes) {
        // build per-axis slice info: full for all but `ax`
        let mut info = shape
            .iter()
            .map(|_| SliceInfoElem::Slice { start: 0, end: None, step: 1 })
            .collect::<Vec<_>>();
        info[ax] = SliceInfoElem::Slice {
            start: start as isize,
            end: Some((start + sz) as isize),
            step: 1,
        };

        // apply the slice and clone into a new owned ArrayD
        let view = parent_view.slice(&*info);
        let sub: ArrayD<T> = view.to_owned().into_dyn();
        outputs.push(ArcArray::from(sub));
    }

    Ok(outputs)
}

/// Helper: broadcast two “batch” shapes with NumPy-style rules.
fn broadcast_shapes(
    a: &[usize],
    b: &[usize],
) -> Result<Vec<usize>, NDArrayOperationError> {
    let na = a.len();
    let nb = b.len();
    let n = na.max(nb);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let da = if i < n - na { 1 } else { a[i - (n - na)] };
        let db = if i < n - nb { 1 } else { b[i - (n - nb)] };
        if da != db && da != 1 && db != 1 {
            return Err(NDArrayOperationError::IncompatibleShape);
        }
        out.push(da.max(db));
    }
    Ok(out)
}

/// Helper: compute row-major strides for a shape.
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let ndim = shape.len();
    let mut strides = vec![0; ndim];
    let mut acc = 1usize;
    for i in (0..ndim).rev() {
        strides[i] = acc;
        acc = acc.checked_mul(shape[i]).expect("stride overflow");
    }
    strides
}


/// ONNX-style batched/broadcasted matrix multiplication.
///
/// Broadcasting semantics are exactly those of `numpy.matmul` (and thus the
/// ONNX spec):
///
/// * 1-D × 1-D → 0-D (scalar)
/// * 1-D × 2-D → 1-D
/// * 2-D × 1-D → 1-D
/// * ≥2-D tensors are treated as stacks of matrices whose _leading_ dimensions
///   broadcast element-wise. The final two axes are multiplied.
///
/// The implementation works for any `ndarray::LinalgScalar` (f32, f64, c64,
/// etc.) and falls back to `matrixmultiply` if BLAS is not enabled.
pub fn matmul<T>(
    a: &ArcArray<T, IxDyn>,
    b: &ArcArray<T, IxDyn>,
) -> Result<ArcArray<T, IxDyn>, NDArrayOperationError>
where
    T: LinalgScalar + One + Zero,
{
    // ---------- 1. Normalise ranks (add singleton axes where required) ----------
    let mut a_view: ArrayViewD<'_, T> = a.view();
    let mut b_view: ArrayViewD<'_, T> = b.view();

    let mut a_rank = a_view.ndim();
    let mut b_rank = b_view.ndim();

    // Promote 1-D operands to matrices so we can treat everything uniformly.
    let mut drop_first_axis_after = false;
    let mut drop_last_axis_after = false;

    if a_rank == 1 {
        // [N]  → [1, N]
        a_view = a_view.insert_axis(Axis(0));
        a_rank = 2;
        drop_first_axis_after = true;
    }
    if b_rank == 1 {
        // [N]  → [N, 1]
        b_view = b_view.insert_axis(Axis(b_rank));
        b_rank = 2;
        drop_last_axis_after = true;
    }

    // Pre-pend 1-sized axes so both tensors have the same rank.
    let max_rank = a_rank.max(b_rank);
    while a_view.ndim() < max_rank {
        a_view = a_view.insert_axis(Axis(0));
    }
    while b_view.ndim() < max_rank {
        b_view = b_view.insert_axis(Axis(0));
    }

    // ---------- 2. Validate broadcastability & gather size info ----------
    let a_shape = a_view.shape();
    let b_shape = b_view.shape();
    let rank = a_shape.len();

    // Everything except the last two axes must broadcast.
    let mut batch_shape = Vec::with_capacity(rank.saturating_sub(2));
    for d in 0..rank.saturating_sub(2) {
        let (ad, bd) = (a_shape[d], b_shape[d]);
        if ad != bd && ad != 1 && bd != 1 {
            return Err(NDArrayOperationError::BroadcastError(String::new()));
        }
        batch_shape.push(ad.max(bd));
    }

    // Matrix dimensions.
    let (m, k_left) = (a_shape[rank - 2], a_shape[rank - 1]);
    let (k_right, p) = (b_shape[rank - 2], b_shape[rank - 1]);
    if k_left != k_right {
        return Err(NDArrayOperationError::IncompatibleShape);
    }

    // ---------- 3. Broadcast & reshape to 3-D (batch, M, K) / (batch, K, P) ----------
    let mut a_bcast_shape = batch_shape.clone();
    a_bcast_shape.extend([m, k_left]);
    let mut b_bcast_shape = batch_shape.clone();
    b_bcast_shape.extend([k_left, p]);

    let a_b = a_view
        .broadcast(a_bcast_shape.clone())
        .ok_or(NDArrayOperationError::IncompatibleShape)?;
    
    let a_b = a_b
        .to_shape((batch_shape.iter().product::<usize>(), m, k_left))
        .map_err(|_| NDArrayOperationError::Internal)?;

    let b_b = b_view
        .broadcast(b_bcast_shape)
        .ok_or(NDArrayOperationError::IncompatibleShape)?;
    
    let b_b = b_b
        .to_shape((batch_shape.iter().product::<usize>(), k_left, p))
        .map_err(|_| NDArrayOperationError::Internal)?;

    // ---------- 4. Batched multiply ----------
    let mut out = Array::<T, _>::zeros((a_b.dim().0, m, p));
    for i in 0..a_b.dim().0 {
        let (a_mat, b_mat): (ArrayView2<'_, T>, ArrayView2<'_, T>) =
            (a_b.slice(s![i, .., ..]), b_b.slice(s![i, .., ..]));
        let mut c: ArrayViewMut2<'_, T> = out.slice_mut(s![i, .., ..]);

        // C ← 1·A·B + 0·C
        general_mat_mul(T::one(), &a_mat, &b_mat, T::zero(), &mut c);
    }

    // ---------- 5. Reshape back to the ONNX-style output shape ----------
    let mut final_shape = batch_shape;
    final_shape.extend([m, p]);
    let mut result = out
        .into_shape(IxDyn(&final_shape))
        .map_err(|_| NDArrayOperationError::Internal)?;

    // Remove the axes we temporarily added for 1-D inputs.
    if drop_last_axis_after {
        // Result shape …, K, P → …, K
        let last_axis = Axis(result.ndim() - 1);
        result = result.index_axis_move(last_axis, 0);
    }
    if drop_first_axis_after {
        // Result shape 1, … → …
        result = result.index_axis_move(Axis(0), 0);
    }

    Ok(result.into_shared())
}

/// ONNX Where: elementwise select between `x` and `y` based on `condition`,
/// with full NumPy‐style broadcasting across all three inputs.
pub fn where_op<T>(
    condition: ArcArray<bool, IxDyn>,
    x: ArcArray<T, IxDyn>,
    y: ArcArray<T, IxDyn>,
) -> Result<ArcArray<T, IxDyn>, NDArrayOperationError>
where
    T: Clone,
{
    // 1) Compute the common broadcast shape
    let s1 = broadcast_shapes(condition.shape(), x.shape())?;
    let out_shape = broadcast_shapes(&s1, y.shape())?;
    let out_len = out_shape.iter().product::<usize>();

    // 2) Broadcast each input to the common shape
    let cond_b = condition.broadcast(out_shape.as_slice())
        .ok_or_else(|| NDArrayOperationError::IncompatibleShape)?;
    let x_b = x.broadcast(out_shape.as_slice())
        .ok_or_else(|| NDArrayOperationError::IncompatibleShape)?;
    let y_b = y.broadcast(out_shape.as_slice())
        .ok_or_else(|| NDArrayOperationError::IncompatibleShape)?;

    // 3) Build the output buffer by iterating in lockstep
    let mut buf: Vec<T> = Vec::with_capacity(out_len);
    for (&c, xv, yv) in cond_b.iter().zip(x_b.iter()).zip(y_b.iter()).map(|((c, x), y)| (c, x, y)) {
        buf.push(if c { xv.clone() } else { yv.clone() });
    }

    // 4) Assemble into an ArrayD and wrap in ArcArray
    let arr = ArrayD::from_shape_vec(IxDyn(&out_shape), buf)?;
    Ok(ArcArray::from(arr))
}


/// ONNX GroupNormalization (op‑set ≥ 18)
///
/// * **X**      : `(N, C, d₁, d₂, …)`
/// * **scale** : `(C)`      (γ)  
/// * **bias**  : `(C)`      (β)\
/// * **num_groups** ∣ **epsilon** are the required attributes
///
/// Output **Y** has the **same shape** as **X**.
///
/// Broadcasting & semantics exactly match the spec – when
/// `num_groups == 1` this degenerates to **LayerNorm**;
/// when `num_groups == C` it is **InstanceNorm**.
pub fn group_normalization<T>(
    x: &ArcArray<T, IxDyn>,
    scale: &ArcArray<T, IxDyn>,
    bias: &ArcArray<T, IxDyn>,
    num_groups: usize,
    epsilon: f64,
) -> Result<ArcArray<T, IxDyn>, NDArrayOperationError>
where
    T: Float + FromPrimitive,
{
    // ---------- 0. Basic shape checks ----------
    let rank = x.ndim();
    if rank < 3 {
        return Err(NDArrayOperationError::IncompatibleShape);
    }
    let n  = x.shape()[0];
    let c  = x.shape()[1];
    if c % num_groups != 0 {
        return Err(NDArrayOperationError::IncompatibleShape);
    }
    if scale.len() != c || bias.len() != c {
        return Err(NDArrayOperationError::IncompatibleShape);
    }
    let group_size     = c / num_groups;
    let spatial_size: usize = x.shape()[2..].iter().product();
    let elems_per_grp  = group_size * spatial_size;

    // ---------- 1. Reshape to (N, G, elems_per_grp) ----------
    // Safe because we only collapse contiguous trailing axes.
    let x3: Array<T, _> = x
        .view()
        .into_shape((n, num_groups, elems_per_grp))
        .map_err(|_| NDArrayOperationError::Internal)?
        .to_owned();     // keep contiguous for fast SIMD

    // ---------- 2. Compute per‑(N,G) mean & variance ----------
    // ndarray’s `mean_axis`/`var_axis` give us vectors of shape (N, G)
    let mean = x3.mean_axis(Axis(2)).expect("axis always exists");          // (N,G)
    let var  = x3.var_axis(Axis(2), T::zero());                             // (N,G)  ddof = 0  :contentReference[oaicite:1]{index=1}

    // Transient buffers for broadcasting to (N,G,elems_per_grp)
    let mean_b = mean.insert_axis(Axis(2));          // (N,G,1) ➜ broadcast
    let var_b  = var .insert_axis(Axis(2));
    let eps: T = T::from_f64(epsilon).unwrap();
    let denom  = var_b.mapv(|v| (v + eps).sqrt());   // std‑dev

    // ---------- 3. Normalize inside each group ----------
    let y3 = (&x3 - &mean_b) / &denom;               // (N,G,E)

    // ---------- 4. Reshape back to (N,C,…) ----------
    let mut y = y3
        .into_shape(IxDyn(&[n, c, spatial_size]))
        .map_err(|_| NDArrayOperationError::Internal)?;

    // ---------- 5. Apply scale (γ) and bias (β) ----------
    // Broadcast γ/β to (1,C,spatial_size) then multiply/add in‑place.
    let scale_a = scale.clone()
        .insert_axis(Axis(0))
        .insert_axis(Axis(2));
    let scale_b = scale_a
        .broadcast(y.raw_dim())
        .ok_or(NDArrayOperationError::Internal)?;
    let bias_a = bias.to_owned()
        .insert_axis(Axis(0))
        .insert_axis(Axis(2));
    let bias_b  = bias_a
        .broadcast(y.raw_dim())
        .ok_or(NDArrayOperationError::Internal)?;

    Zip::from(&mut y)
        .and(&scale_b)
        .and(&bias_b)
        .for_each(|y_elem, &s, &b| *y_elem = *y_elem * s + b);

    // ---------- 6. Final reshape to original >2‑D layout ----------
    let mut final_shape = x.shape().to_vec();
    final_shape[0] = n;
    y = y
        .into_shape(IxDyn(&final_shape))
        .map_err(|_| NDArrayOperationError::Internal)?;

    Ok(y.into_shared())
}


pub fn range<T>(start: T, end: T, step: T) -> ArcArray<T, Ix1>
where T: Num + Copy + PartialOrd + std::ops::AddAssign
{
    let mut out = vec![];
    let mut i = start;
    while i < end {
        out.push(i);
        i += step;
    }
    ArcArray::from_vec(out)
}