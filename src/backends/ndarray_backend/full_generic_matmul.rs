use half::{bf16, f16};
use ndarray::linalg::general_mat_mul;
use ndarray::{Array, Array3, ArrayView2, ArrayView3, ArrayViewMut2, Axis, Ix3, IxDyn, s};
use num_traits::{FromPrimitive, One, ToPrimitive, Zero};

use crate::backends::ndarray_backend::ops::NDArrayOperationError;
use crate::dtype::DType;

/// Cleaned-up ONNX/Numpy-style matmul with optional *accumulation precision* control.
/// - `accum_dtype = None` => accumulate in the input type `T` (backwards-compat)
/// - `Some(DType::F32/F64/BF16/F16)` => upcast A/B to that accumulator, do the math,
///   then cast back to `T`.
///
/// Fast path:
/// - When accumulation is `f32` or `f64`, we use `general_mat_mul` for each batch.
/// - When accumulation is "same as input" and `T` is `f32`/`f64`, we also use `general_mat_mul`.
///
/// Portable path:
/// - For other accumulator types (e.g., `bf16`), we do a simple triple loop in the
///   accumulator dtype to honor *true* accumulate precision.
///
/// PyTorch-style parity:
/// - For half/bfloat16 inputs, pass `Some(DType::F32)` to emulate fp32 accumulation.
#[allow(dead_code)]
pub fn matmul_with_accum_dtype<T>(
    a: &ndarray::ArcArray<T, IxDyn>,
    b: &ndarray::ArcArray<T, IxDyn>,
    accum_dtype: Option<DType>,
) -> Result<ndarray::ArcArray<T, IxDyn>, NDArrayOperationError>
where
    T: ndarray::LinalgScalar + One + Zero + Copy + ToPrimitive + FromPrimitive,
{
    // ---- 1) Normalize ranks (handle 1-D cases & prefix-broadcast) -----------------
    let (mut a_view, mut b_view) = (a.view(), b.view());

    let mut drop_first_axis_after = false;
    let mut drop_last_axis_after = false;

    if a_view.ndim() == 1 {
        a_view = a_view.insert_axis(Axis(0)); // [N] -> [1, N]
        drop_first_axis_after = true;
    }
    if b_view.ndim() == 1 {
        let rank = b_view.ndim();
        b_view = b_view.insert_axis(Axis(rank)); // [N] -> [N, 1]
        drop_last_axis_after = true;
    }

    let max_rank = a_view.ndim().max(b_view.ndim());
    while a_view.ndim() < max_rank {
        a_view = a_view.insert_axis(Axis(0));
    }
    while b_view.ndim() < max_rank {
        b_view = b_view.insert_axis(Axis(0));
    }

    // ---- 2) Validate broadcastability & gather dims --------------------------------
    let a_shape = a_view.shape().to_vec();
    let b_shape = b_view.shape().to_vec();
    let rank = a_shape.len();

    if rank < 2 {
        return Err(NDArrayOperationError::IncompatibleShape);
    }

    let (m, k_left) = (a_shape[rank - 2], a_shape[rank - 1]);
    let (k_right, p) = (b_shape[rank - 2], b_shape[rank - 1]);
    if k_left != k_right {
        return Err(NDArrayOperationError::IncompatibleShape);
    }

    let mut batch_shape = Vec::with_capacity(rank - 2);
    for d in 0..(rank - 2) {
        let (ad, bd) = (a_shape[d], b_shape[d]);
        if ad != bd && ad != 1 && bd != 1 {
            return Err(NDArrayOperationError::BroadcastError(format!(
                "Cannot broadcast dims {} vs {} at axis {}",
                ad, bd, d
            )));
        }
        batch_shape.push(ad.max(bd));
    }
    let batch: usize = batch_shape.iter().product::<usize>().max(1);

    // ---- 3) Broadcast & reshape to 3-D views (batch, M, K) / (batch, K, P) --------
    let mut a_bcast = batch_shape.clone();
    a_bcast.extend([m, k_left]);
    let mut b_bcast = batch_shape.clone();
    b_bcast.extend([k_left, p]);

    let a_b = a_view
        .broadcast(a_bcast)
        .ok_or(NDArrayOperationError::IncompatibleShape)?;
    let a_b = a_b
        .to_shape((batch, m, k_left))
        .map_err(|_| NDArrayOperationError::Internal)?;

    let b_b = b_view
        .broadcast(b_bcast)
        .ok_or(NDArrayOperationError::IncompatibleShape)?;
    let b_b = b_b
        .to_shape((batch, k_left, p))
        .map_err(|_| NDArrayOperationError::Internal)?;

    // ---- 4) Do the batched multiply with chosen accumulation precision -------------
    let result_3d = match accum_dtype {
        None => {
            // Accumulate in T (fast GEMM path for f32/f64/c64/c128, etc.).
            batched_gemm_in_type::<T>(a_b.view(), b_b.view())?
        }
        Some(DType::F32) => {
            // Cast to f32, gemm, cast back to T.
            let a32 = cast3::<T, f32>(a_b.view())?;
            let b32 = cast3::<T, f32>(b_b.view())?;
            let c32 = batched_gemm_in_type::<f32>(a32.view(), b32.view())?;
            cast3_back::<f32, T>(c32.view())?
        }
        Some(DType::F64) => {
            let a64 = cast3::<T, f64>(a_b.view())?;
            let b64 = cast3::<T, f64>(b_b.view())?;
            let c64 = batched_gemm_in_type::<f64>(a64.view(), b64.view())?;
            cast3_back::<f64, T>(c64.view())?
        }
        Some(DType::BF16) => {
            let ab = cast3_via_f32::<T, bf16>(a_b.view())?;
            let bb = cast3_via_f32::<T, bf16>(b_b.view())?;
            let cb = batched_naive::<bf16>(ab.view(), bb.view());
            cast3_via_f32_back::<bf16, T>(cb.view())?
        }
        Some(DType::F16) => {
            let ah = cast3_via_f32::<T, f16>(a_b.view())?;
            let bh = cast3_via_f32::<T, f16>(b_b.view())?;
            let ch = batched_naive::<f16>(ah.view(), bh.view());
            cast3_via_f32_back::<f16, T>(ch.view())?
        }
        // Extend here for other accumulator types if you add them to DType.
        _ => return Err(NDArrayOperationError::Internal),
    };

    // ---- 5) Reshape back & drop temporary axes ------------------------------------
    let mut final_shape = batch_shape;
    final_shape.extend([m, p]);

    let mut out = result_3d
        .into_shape_with_order(IxDyn(&final_shape))
        .map_err(|_| NDArrayOperationError::Internal)?;

    if drop_last_axis_after {
        // …, K, P -> …, K  (take the first column)
        let last_axis = Axis(out.ndim() - 1);
        out = out.index_axis_move(last_axis, 0);
    }
    if drop_first_axis_after {
        // 1, … -> …
        out = out.index_axis_move(Axis(0), 0);
    }

    Ok(out.into_shared())
}

/// Generic, compile-time version: accumulate in `Acc` (portable inner loop).
/// Useful when you want *true* accumulation in `bf16`/`f16` (or any custom scalar)
/// regardless of runtime switches. This skips BLAS and uses a simple triple loop.
#[allow(dead_code)]
pub fn matmul_with_accum_generic<T, Acc>(
    a: &ndarray::ArcArray<T, IxDyn>,
    b: &ndarray::ArcArray<T, IxDyn>,
) -> Result<ndarray::ArcArray<T, IxDyn>, NDArrayOperationError>
where
    T: Copy + ToPrimitive + FromPrimitive,
    Acc: Copy
        + Zero
        + ToPrimitive
        + FromPrimitive
        + core::ops::Mul<Output = Acc>
        + core::ops::Add<Output = Acc>,
{
    // Reuse the runtime function by mapping a compile-time Acc to a small adapter.
    // We’ll just inline the core of matmul_with_accum_dtype but always use the portable path.

    // ---- 1) Normalize & broadcast (same as above) ---------------------------------
    let (mut a_view, mut b_view) = (a.view(), b.view());
    let mut drop_first_axis_after = false;
    let mut drop_last_axis_after = false;

    if a_view.ndim() == 1 {
        a_view = a_view.insert_axis(Axis(0));
        drop_first_axis_after = true;
    }
    if b_view.ndim() == 1 {
        let rank = b_view.ndim();
        b_view = b_view.insert_axis(Axis(rank));
        drop_last_axis_after = true;
    }
    let max_rank = a_view.ndim().max(b_view.ndim());
    while a_view.ndim() < max_rank {
        a_view = a_view.insert_axis(Axis(0));
    }
    while b_view.ndim() < max_rank {
        b_view = b_view.insert_axis(Axis(0));
    }

    let a_shape = a_view.shape().to_vec();
    let b_shape = b_view.shape().to_vec();
    let rank = a_shape.len();
    if rank < 2 {
        return Err(NDArrayOperationError::IncompatibleShape);
    }

    let (m, k_left) = (a_shape[rank - 2], a_shape[rank - 1]);
    let (k_right, p) = (b_shape[rank - 2], b_shape[rank - 1]);
    if k_left != k_right {
        return Err(NDArrayOperationError::IncompatibleShape);
    }

    let mut batch_shape = Vec::with_capacity(rank - 2);
    for d in 0..(rank - 2) {
        let (ad, bd) = (a_shape[d], b_shape[d]);
        if ad != bd && ad != 1 && bd != 1 {
            return Err(NDArrayOperationError::BroadcastError(String::new()));
        }
        batch_shape.push(ad.max(bd));
    }
    let batch: usize = batch_shape.iter().product::<usize>().max(1);

    let mut a_bcast = batch_shape.clone();
    a_bcast.extend([m, k_left]);
    let mut b_bcast = batch_shape.clone();
    b_bcast.extend([k_left, p]);

    let a_b = a_view
        .broadcast(a_bcast)
        .ok_or(NDArrayOperationError::IncompatibleShape)?;
    let a_b = a_b
        .to_shape((batch, m, k_left))
        .map_err(|_| NDArrayOperationError::Internal)?;
    let b_b = b_view
        .broadcast(b_bcast)
        .ok_or(NDArrayOperationError::IncompatibleShape)?;
    let b_b = b_b
        .to_shape((batch, k_left, p))
        .map_err(|_| NDArrayOperationError::Internal)?;

    // ---- 2) Cast to Acc, naive batched multiply, cast back to T -------------------
    let a_acc = cast3::<T, Acc>(a_b.view())?;
    let b_acc = cast3::<T, Acc>(b_b.view())?;
    let c_acc = batched_naive::<Acc>(a_acc.view(), b_acc.view());
    let mut out = cast3_back::<Acc, T>(c_acc.view())?
        .into_shape_with_order(IxDyn(&{
            let mut s = batch_shape.clone();
            s.extend([m, p]);
            s
        }))
        .map_err(|_| NDArrayOperationError::Internal)?;

    if drop_last_axis_after {
        let last_axis = Axis(out.ndim() - 1);
        out = out.index_axis_move(last_axis, 0);
    }
    if drop_first_axis_after {
        out = out.index_axis_move(Axis(0), 0);
    }
    Ok(out.into_shared())
}

// ===== Helpers ===================================================================

fn batched_gemm_in_type<S>(
    a3: ArrayView3<'_, S>,
    b3: ArrayView3<'_, S>,
) -> Result<Array3<S>, NDArrayOperationError>
where
    S: ndarray::LinalgScalar + One + Zero + Copy,
{
    let (batch, m, k) = a3.dim();
    let (_, _k2, p) = b3.dim();
    debug_assert_eq!(k, _k2);

    let mut out = Array::<S, Ix3>::zeros((batch, m, p));
    for i in 0..batch {
        let a_mat: ArrayView2<'_, S> = a3.slice(s![i, .., ..]);
        let b_mat: ArrayView2<'_, S> = b3.slice(s![i, .., ..]);
        let mut c: ArrayViewMut2<'_, S> = out.slice_mut(s![i, .., ..]);
        // C ← 1·A·B + 0·C
        general_mat_mul(S::one(), &a_mat, &b_mat, S::zero(), &mut c);
    }
    Ok(out)
}

#[allow(dead_code)]
fn batched_naive<S>(a3: ArrayView3<'_, S>, b3: ArrayView3<'_, S>) -> Array3<S>
where
    S: Copy + Zero + core::ops::Add<Output = S> + core::ops::Mul<Output = S>,
{
    let (batch, m, k) = a3.dim();
    let (_, _k2, p) = b3.dim();
    debug_assert_eq!(k, _k2);

    let mut out = Array::<S, Ix3>::zeros((batch, m, p));
    for i in 0..batch {
        let a = a3.slice(s![i, .., ..]);
        let b = b3.slice(s![i, .., ..]);
        let mut c = out.slice_mut(s![i, .., ..]);

        for r in 0..m {
            for ccol in 0..p {
                let mut acc = S::zero();
                for kk in 0..k {
                    // acc += a[r, kk] * b[kk, ccol]
                    acc = acc + (a[(r, kk)] * b[(kk, ccol)]);
                }
                c[(r, ccol)] = acc;
            }
        }
    }
    out
}

// T -> U via ToPrimitive/FromPrimitive (real-only; returns Internal on failure)
fn cast3<T, U>(src: ArrayView3<'_, T>) -> Result<Array3<U>, NDArrayOperationError>
where
    T: Copy + ToPrimitive,
    U: Copy + FromPrimitive,
{
    let mut out = Array::<U, Ix3>::uninit(src.dim());
    // Fill manually to return a clean error if casting isn't possible.
    for ((i, j, k), slot) in out.indexed_iter_mut() {
        let v_t = src[(i, j, k)];
        // Try f64 first for range; fall back to f32.
        if let Some(x) = v_t.to_f64().and_then(U::from_f64) {
            unsafe { slot.as_mut_ptr().write(x) }
        } else if let Some(x) = v_t.to_f32().and_then(U::from_f32) {
            unsafe { slot.as_mut_ptr().write(x) }
        } else {
            return Err(NDArrayOperationError::Internal);
        }
    }
    // SAFETY: fully initialized above
    Ok(unsafe { out.assume_init() })
}

// U -> T via ToPrimitive/FromPrimitive (real-only)
fn cast3_back<U, T>(src: ArrayView3<'_, U>) -> Result<Array3<T>, NDArrayOperationError>
where
    U: Copy + ToPrimitive,
    T: Copy + FromPrimitive,
{
    cast3::<U, T>(src)
}

// Cast via f32 round-trip for half/bfloat16 where FromPrimitive/ToPrimitive may be partial.
fn cast3_via_f32<T, U>(src: ArrayView3<'_, T>) -> Result<Array3<U>, NDArrayOperationError>
where
    T: Copy + ToPrimitive,
    U: Copy + 'static,
    // expecting: U = half::bf16 or half::f16
{
    let mut out = Array::<U, Ix3>::uninit(src.dim());
    for ((i, j, k), slot) in out.indexed_iter_mut() {
        let x = src[(i, j, k)]
            .to_f32()
            .ok_or(NDArrayOperationError::Internal)?;
        let y: U = if core::any::TypeId::of::<U>() == core::any::TypeId::of::<bf16>() {
            // SAFETY: type-id checked
            let v = bf16::from_f32(x);
            unsafe { core::mem::transmute_copy::<bf16, U>(&v) }
        } else if core::any::TypeId::of::<U>() == core::any::TypeId::of::<f16>() {
            let v = f16::from_f32(x);
            unsafe { core::mem::transmute_copy::<f16, U>(&v) }
        } else {
            // For other U you might add more branches or require FromPrimitive.
            return Err(NDArrayOperationError::Internal);
        };
        unsafe { slot.as_mut_ptr().write(y) }
    }
    Ok(unsafe { out.assume_init() })
}

fn cast3_via_f32_back<U, T>(src: ArrayView3<'_, U>) -> Result<Array3<T>, NDArrayOperationError>
where
    U: Copy + 'static,
    T: Copy + FromPrimitive,
{
    let mut out = Array::<T, Ix3>::uninit(src.dim());
    for ((i, j, k), slot) in out.indexed_iter_mut() {
        // Convert U -> f32 -> T
        let x_f32 = if core::any::TypeId::of::<U>() == core::any::TypeId::of::<bf16>() {
            // SAFETY: type-id checked
            let v_b: bf16 = unsafe { core::mem::transmute_copy::<U, bf16>(&src[(i, j, k)]) };
            v_b.to_f32()
        } else if core::any::TypeId::of::<U>() == core::any::TypeId::of::<f16>() {
            let v_h: f16 = unsafe { core::mem::transmute_copy::<U, f16>(&src[(i, j, k)]) };
            v_h.to_f32()
        } else {
            return Err(NDArrayOperationError::Internal);
        };
        let y = T::from_f32(x_f32).ok_or(NDArrayOperationError::Internal)?;
        unsafe { slot.as_mut_ptr().write(y) }
    }
    Ok(unsafe { out.assume_init() })
}
