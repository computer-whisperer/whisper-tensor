//! Conversion bridge between new and legacy tensor types.
//!
//! These functions allow incremental migration: new-type tensors can be
//! converted to legacy types for code that hasn't been migrated yet, and
//! legacy outputs can be converted back.

use crate::backends::ndarray_backend::NDArrayNumericTensor;
use crate::dtype::DType;
use crate::migration::numeric_tensor::NumericTensor as LegacyNumericTensor;
use crate::numeric_dtype::NumericDType;
use crate::numeric_scalar::NumericScalar;
use crate::numeric_tensor::{NumericTensor, NumericTensorView};
use crate::pool::SystemPool;
use crate::tensor_rank::DynRank;
use crate::DynRank as LegacyDynRank;

/// Static SystemPool for bridge allocations.
static BRIDGE_POOL: SystemPool = SystemPool;

/// Convert a new-type tensor view to a legacy NumericTensor.
pub fn view_to_legacy(view: &NumericTensorView<'_, DynRank>) -> LegacyNumericTensor<LegacyDynRank> {
    let numel = view.numel();
    let shape: Vec<usize> = view.shape().iter().map(|&d| d as usize).collect();
    let dtype = view.dtype();

    match dtype {
        NumericDType::F32 => {
            let data: Vec<f32> = (0..numel)
                .map(|i| {
                    let s = view.read_element(i);
                    f32::from_le_bytes([s.raw_bits()[0], s.raw_bits()[1], s.raw_bits()[2], s.raw_bits()[3]])
                })
                .collect();
            LegacyNumericTensor::NDArray(NDArrayNumericTensor::F32(
                ndarray::ArcArray::from_shape_vec(ndarray::IxDyn(&shape), data).unwrap(),
            ))
        }
        NumericDType::F64 => {
            let data: Vec<f64> = (0..numel)
                .map(|i| f64::from_le_bytes(*view.read_element(i).raw_bits()))
                .collect();
            LegacyNumericTensor::NDArray(NDArrayNumericTensor::F64(
                ndarray::ArcArray::from_shape_vec(ndarray::IxDyn(&shape), data).unwrap(),
            ))
        }
        NumericDType::BF16 => {
            let data: Vec<half::bf16> = (0..numel)
                .map(|i| {
                    let s = view.read_element(i);
                    half::bf16::from_le_bytes([s.raw_bits()[0], s.raw_bits()[1]])
                })
                .collect();
            LegacyNumericTensor::NDArray(NDArrayNumericTensor::BF16(
                ndarray::ArcArray::from_shape_vec(ndarray::IxDyn(&shape), data).unwrap(),
            ))
        }
        NumericDType::F16 => {
            let data: Vec<half::f16> = (0..numel)
                .map(|i| {
                    let s = view.read_element(i);
                    half::f16::from_le_bytes([s.raw_bits()[0], s.raw_bits()[1]])
                })
                .collect();
            LegacyNumericTensor::NDArray(NDArrayNumericTensor::F16(
                ndarray::ArcArray::from_shape_vec(ndarray::IxDyn(&shape), data).unwrap(),
            ))
        }
        NumericDType::I32 => {
            let data: Vec<i32> = (0..numel)
                .map(|i| {
                    let s = view.read_element(i);
                    i32::from_le_bytes([s.raw_bits()[0], s.raw_bits()[1], s.raw_bits()[2], s.raw_bits()[3]])
                })
                .collect();
            LegacyNumericTensor::NDArray(NDArrayNumericTensor::I32(
                ndarray::ArcArray::from_shape_vec(ndarray::IxDyn(&shape), data).unwrap(),
            ))
        }
        NumericDType::I64 => {
            let data: Vec<i64> = (0..numel)
                .map(|i| i64::from_le_bytes(*view.read_element(i).raw_bits()))
                .collect();
            LegacyNumericTensor::NDArray(NDArrayNumericTensor::I64(
                ndarray::ArcArray::from_shape_vec(ndarray::IxDyn(&shape), data).unwrap(),
            ))
        }
        other => panic!("view_to_legacy: unsupported dtype {other}"),
    }
}

/// Convert a legacy NumericTensor to a new-type tensor (SystemPool-backed).
pub fn legacy_to_new(
    legacy: &LegacyNumericTensor<LegacyDynRank>,
) -> NumericTensor<'static, DynRank, SystemPool> {
    let legacy_dtype = legacy.dtype();
    let dtype = NumericDType::from_legacy(legacy_dtype).unwrap();
    let shape: Vec<u64> = legacy.shape().iter().map(|&d| d as u64).collect();

    let mut t = NumericTensor::zeros(shape, dtype, &BRIDGE_POOL).unwrap();

    let nd = legacy
        .to_ndarray()
        .unwrap()
        .cast(DType::F64)
        .unwrap()
        .flatten()
        .try_to_vec::<f64>()
        .unwrap();

    for (i, &v) in nd.iter().enumerate() {
        let scalar = NumericScalar::from_f64(v).cast_to(dtype);
        t.write_element(i, scalar);
    }
    t
}
