use crate::DynRank;
use crate::backends::ndarray_backend::ops::NDArrayOperationError;
use crate::backends::ndarray_backend::{
    NDArrayNumericTensor, NDArrayNumericTensorError, full_generic_matmul,
};
use crate::dtype::DType;
use half::bf16;
use ndarray::linalg::general_mat_mul;
use ndarray::{
    ArcArray, Array, ArrayView2, ArrayViewD, ArrayViewMut2, Axis, IxDyn, LinalgScalar, s,
};
use num_traits::{One, Zero};

impl NDArrayNumericTensor<DynRank> {
    pub fn matmul(
        outer_a: &Self,
        outer_b: &Self,
        accumulate_dtype: Option<DType>,
    ) -> Result<Self, NDArrayNumericTensorError> {
        Ok(match (outer_a, outer_b) {
            (NDArrayNumericTensor::F32(a), NDArrayNumericTensor::F32(b)) => {
                NDArrayNumericTensor::F32(
                    if accumulate_dtype.is_none() || accumulate_dtype == Some(outer_a.dtype()) {
                        generic_matmul_accum_same_type(a, b)?
                    } else {
                        full_generic_matmul::matmul_with_accum_dtype(a, b, accumulate_dtype)?
                    },
                )
            }
            (NDArrayNumericTensor::F64(a), NDArrayNumericTensor::F64(b)) => {
                NDArrayNumericTensor::F64(
                    if accumulate_dtype.is_none() || accumulate_dtype == Some(outer_a.dtype()) {
                        generic_matmul_accum_same_type(a, b)?
                    } else {
                        full_generic_matmul::matmul_with_accum_dtype(a, b, accumulate_dtype)?
                    },
                )
            }
            (NDArrayNumericTensor::BF16(a), NDArrayNumericTensor::BF16(b)) => {
                NDArrayNumericTensor::BF16(match accumulate_dtype {
                    Some(DType::BF16) => generic_matmul_accum_same_type(a, b)?,
                    Some(DType::F32) => matmul_bf16_fp32_accumulate(a, b)?,
                    _ => full_generic_matmul::matmul_with_accum_dtype(a, b, accumulate_dtype)?,
                })
            }
            (NDArrayNumericTensor::F16(a), NDArrayNumericTensor::F16(b)) => {
                NDArrayNumericTensor::F16(
                    if accumulate_dtype.is_none() || accumulate_dtype == Some(outer_a.dtype()) {
                        generic_matmul_accum_same_type(a, b)?
                    } else {
                        full_generic_matmul::matmul_with_accum_dtype(a, b, accumulate_dtype)?
                    },
                )
            }
            (NDArrayNumericTensor::U64(a), NDArrayNumericTensor::U64(b)) => {
                NDArrayNumericTensor::U64(
                    if accumulate_dtype.is_none() || accumulate_dtype == Some(outer_a.dtype()) {
                        generic_matmul_accum_same_type(a, b)?
                    } else {
                        full_generic_matmul::matmul_with_accum_dtype(a, b, accumulate_dtype)?
                    },
                )
            }
            (NDArrayNumericTensor::I64(a), NDArrayNumericTensor::I64(b)) => {
                NDArrayNumericTensor::I64(
                    if accumulate_dtype.is_none() || accumulate_dtype == Some(outer_a.dtype()) {
                        generic_matmul_accum_same_type(a, b)?
                    } else {
                        full_generic_matmul::matmul_with_accum_dtype(a, b, accumulate_dtype)?
                    },
                )
            }
            (NDArrayNumericTensor::U32(a), NDArrayNumericTensor::U32(b)) => {
                NDArrayNumericTensor::U32(
                    if accumulate_dtype.is_none() || accumulate_dtype == Some(outer_a.dtype()) {
                        generic_matmul_accum_same_type(a, b)?
                    } else {
                        full_generic_matmul::matmul_with_accum_dtype(a, b, accumulate_dtype)?
                    },
                )
            }
            (NDArrayNumericTensor::I32(a), NDArrayNumericTensor::I32(b)) => {
                NDArrayNumericTensor::I32(
                    if accumulate_dtype.is_none() || accumulate_dtype == Some(outer_a.dtype()) {
                        generic_matmul_accum_same_type(a, b)?
                    } else {
                        full_generic_matmul::matmul_with_accum_dtype(a, b, accumulate_dtype)?
                    },
                )
            }
            _ => {
                return Err(NDArrayNumericTensorError::UnsupportedOperationForDTypes(
                    "matmul".to_string(),
                    vec![outer_a.dtype(), outer_b.dtype()],
                ));
            }
        })
    }
}

pub fn matmul_bf16_fp32_accumulate(
    a: &ArcArray<bf16, IxDyn>,
    b: &ArcArray<bf16, IxDyn>,
) -> Result<ArcArray<bf16, IxDyn>, NDArrayOperationError> {
    // 0: cast to fp32
    let a = a.mapv(|x| x.to_f32());
    let b = b.mapv(|x| x.to_f32());

    // ---------- 1. Normalise ranks (add singleton axes where required) ----------
    let mut a_view: ArrayViewD<'_, _> = a.view();
    let mut b_view: ArrayViewD<'_, _> = b.view();

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
    let mut out = Array::<_, _>::zeros((a_b.dim().0, m, p));
    for i in 0..a_b.dim().0 {
        let (a_mat, b_mat): (ArrayView2<'_, _>, ArrayView2<'_, _>) =
            (a_b.slice(s![i, .., ..]), b_b.slice(s![i, .., ..]));
        let mut c: ArrayViewMut2<'_, _> = out.slice_mut(s![i, .., ..]);

        // C ← 1·A·B + 0·C
        general_mat_mul(f32::one(), &a_mat, &b_mat, f32::zero(), &mut c);
    }

    // ---------- 5. Reshape back to the ONNX-style output shape ----------
    let mut final_shape = batch_shape;
    final_shape.extend([m, p]);
    let mut result = out
        .into_shape_with_order(IxDyn(&final_shape))
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

    // Cast back to bf16
    let result: Array<bf16, IxDyn> = result.mapv(bf16::from_f32);

    Ok(result.into_shared())
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
pub fn generic_matmul_accum_same_type<T>(
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
        .into_shape_with_order(IxDyn(&final_shape))
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
