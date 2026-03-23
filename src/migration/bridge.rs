//! Conversion bridge between new and legacy tensor types.
//!
//! These functions allow incremental migration: new-type tensors can be
//! converted to legacy types for code that hasn't been migrated yet, and
//! legacy outputs can be converted back.

use crate::backends::ndarray_backend::NDArrayNumericTensor;
use crate::migration::numeric_tensor::NumericTensor as LegacyNumericTensor;
use crate::numeric_dtype::NumericDType;
use crate::numeric_scalar::NumericScalar;
use crate::numeric_tensor::{NumericTensor, NumericTensorView};
use crate::pool::SystemPool;
use crate::tensor_rank::DynRank;
use crate::DynRank as LegacyDynRank;

use crate::test_set::POOL;

/// Convert a new-type tensor view to a legacy NumericTensor.
pub(crate) fn view_to_legacy(view: &NumericTensorView<'_, DynRank>) -> LegacyNumericTensor<LegacyDynRank> {
    let numel = view.numel();
    let shape: Vec<usize> = view.shape().iter().map(|&d| d as usize).collect();
    let dtype = view.dtype();

    match dtype {
        NumericDType::F64 => {
            let data: Vec<f64> = (0..numel)
                .map(|i| { let s = view.read_element(i); f64::from_le_bytes(*s.raw_bits()) })
                .collect();
            LegacyNumericTensor::NDArray(NDArrayNumericTensor::F64(
                ndarray::ArcArray::from_shape_vec(ndarray::IxDyn(&shape), data).unwrap(),
            ))
        }
        NumericDType::F32 => {
            let data: Vec<f32> = (0..numel)
                .map(|i| {
                    let s = view.read_element(i);
                    let b = s.raw_bits();
                    f32::from_le_bytes([b[0], b[1], b[2], b[3]])
                })
                .collect();
            LegacyNumericTensor::NDArray(NDArrayNumericTensor::F32(
                ndarray::ArcArray::from_shape_vec(ndarray::IxDyn(&shape), data).unwrap(),
            ))
        }
        NumericDType::BF16 => {
            let data: Vec<half::bf16> = (0..numel)
                .map(|i| {
                    let s = view.read_element(i);
                    let b = s.raw_bits();
                    half::bf16::from_le_bytes([b[0], b[1]])
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
                    let b = s.raw_bits();
                    half::f16::from_le_bytes([b[0], b[1]])
                })
                .collect();
            LegacyNumericTensor::NDArray(NDArrayNumericTensor::F16(
                ndarray::ArcArray::from_shape_vec(ndarray::IxDyn(&shape), data).unwrap(),
            ))
        }
        NumericDType::I64 => {
            let data: Vec<i64> = (0..numel)
                .map(|i| { let s = view.read_element(i); i64::from_le_bytes(*s.raw_bits()) })
                .collect();
            LegacyNumericTensor::NDArray(NDArrayNumericTensor::I64(
                ndarray::ArcArray::from_shape_vec(ndarray::IxDyn(&shape), data).unwrap(),
            ))
        }
        NumericDType::I32 => {
            let data: Vec<i32> = (0..numel)
                .map(|i| {
                    let s = view.read_element(i);
                    let b = s.raw_bits();
                    i32::from_le_bytes([b[0], b[1], b[2], b[3]])
                })
                .collect();
            LegacyNumericTensor::NDArray(NDArrayNumericTensor::I32(
                ndarray::ArcArray::from_shape_vec(ndarray::IxDyn(&shape), data).unwrap(),
            ))
        }
        NumericDType::U8 => {
            let data: Vec<u8> = (0..numel)
                .map(|i| { let s = view.read_element(i); s.raw_bits()[0] })
                .collect();
            LegacyNumericTensor::NDArray(NDArrayNumericTensor::U8(
                ndarray::ArcArray::from_shape_vec(ndarray::IxDyn(&shape), data).unwrap(),
            ))
        }
        NumericDType::BOOL => {
            let data: Vec<bool> = (0..numel)
                .map(|i| { let s = view.read_element(i); s.raw_bits()[0] != 0 })
                .collect();
            LegacyNumericTensor::NDArray(NDArrayNumericTensor::BOOL(
                ndarray::ArcArray::from_shape_vec(ndarray::IxDyn(&shape), data).unwrap(),
            ))
        }
        other => panic!("view_to_legacy: unsupported dtype {other}"),
    }
}

/// Convert a legacy NumericTensor to a new-type tensor (SystemPool-backed).
///
/// For float types, reads raw bytes directly (no precision loss).
/// For integer types, reads raw bytes directly (no f64 roundtrip).
pub(crate) fn legacy_to_new(
    legacy: &LegacyNumericTensor<LegacyDynRank>,
) -> NumericTensor<'static, DynRank, SystemPool> {
    let dtype = NumericDType::from_legacy(legacy.dtype()).unwrap();
    let shape: Vec<u64> = legacy.shape().iter().map(|&d| d as u64).collect();
    let nd = legacy.to_ndarray().unwrap();

    let mut t = NumericTensor::zeros(shape, dtype, &POOL).unwrap();

    // Read from legacy NDArray directly via the appropriate typed path.
    // This avoids the f64 roundtrip that would lose precision for I64/U64.
    match dtype {
        NumericDType::F64 => {
            let flat = nd.flatten().try_to_vec::<f64>().unwrap();
            for (i, &v) in flat.iter().enumerate() {
                t.write_element(i, NumericScalar::from_f64(v));
            }
        }
        NumericDType::F32 => {
            let flat = nd.flatten().try_to_vec::<f32>().unwrap();
            for (i, &v) in flat.iter().enumerate() {
                t.write_element(i, NumericScalar::from_f32(v));
            }
        }
        NumericDType::BF16 => {
            let flat = nd.flatten().try_to_vec::<half::bf16>().unwrap();
            for (i, &v) in flat.iter().enumerate() {
                t.write_element(i, NumericScalar::from_bf16(v));
            }
        }
        NumericDType::F16 => {
            let flat = nd.flatten().try_to_vec::<half::f16>().unwrap();
            for (i, &v) in flat.iter().enumerate() {
                t.write_element(i, NumericScalar::from_f16(v));
            }
        }
        NumericDType::I64 => {
            let flat = nd.flatten().try_to_vec::<i64>().unwrap();
            for (i, &v) in flat.iter().enumerate() {
                t.write_element(i, NumericScalar::from_i64(v));
            }
        }
        NumericDType::I32 => {
            let flat = nd.flatten().try_to_vec::<i32>().unwrap();
            for (i, &v) in flat.iter().enumerate() {
                t.write_element(i, NumericScalar::from_i32(v));
            }
        }
        NumericDType::U8 => {
            let flat = nd.flatten().try_to_vec::<u8>().unwrap();
            for (i, &v) in flat.iter().enumerate() {
                t.write_element(i, NumericScalar::from_u8(v));
            }
        }
        NumericDType::BOOL => {
            let flat = nd.flatten().try_to_vec::<bool>().unwrap();
            for (i, &v) in flat.iter().enumerate() {
                t.write_element(i, NumericScalar::from_bool(v));
            }
        }
        other => panic!("legacy_to_new: unsupported dtype {other}"),
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bridge roundtrip: new → legacy → new should preserve bits exactly.
    #[test]
    fn bridge_roundtrip_f32() {
        let original = crate::test_set::tensor_f32(&[1.0, -2.5, 0.0, f32::INFINITY, f32::NAN]);
        let legacy = view_to_legacy(&original.view());
        let back = legacy_to_new(&legacy);
        // Compare element by element (NaN needs special handling)
        for i in 0..original.numel() {
            let a = original.read_element(i);
            let b = back.read_element(i);
            if a.to_f64().is_nan() {
                assert!(b.to_f64().is_nan(), "element {i} NaN mismatch");
            } else {
                assert_eq!(a, b, "element {i} mismatch");
            }
        }
    }

    #[test]
    fn bridge_roundtrip_bf16() {
        let original = crate::test_set::tensor_bf16(&[
            half::bf16::from_f32(1.0),
            half::bf16::from_f32(-3.14),
            half::bf16::ZERO,
        ]);
        let legacy = view_to_legacy(&original.view());
        let back = legacy_to_new(&legacy);
        for i in 0..original.numel() {
            assert_eq!(original.read_element(i), back.read_element(i), "element {i}");
        }
    }

    #[test]
    fn bridge_roundtrip_i32() {
        let original = crate::test_set::tensor_i32(&[0, 1, -1, i32::MAX, i32::MIN]);
        let legacy = view_to_legacy(&original.view());
        let back = legacy_to_new(&legacy);
        for i in 0..original.numel() {
            assert_eq!(original.read_element(i), back.read_element(i), "element {i}");
        }
    }

    #[test]
    fn bridge_roundtrip_i64_large() {
        // Values that would lose precision through f64
        let original = crate::test_set::tensor_i64_shaped(
            vec![4],
            &[i64::MAX, i64::MIN, i64::MAX - 1, 0],
        );
        let legacy = view_to_legacy(&original.view());
        let back = legacy_to_new(&legacy);
        for i in 0..original.numel() {
            assert_eq!(original.read_element(i), back.read_element(i), "element {i}");
        }
    }
}
