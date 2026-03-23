//! Scalar conversion methods — thin delegation to [`NumericDType`] conversion engine.
//!
//! This module provides:
//! - Raw bit access (`read_raw_bits` / `write_raw_bits`) for byte buffers
//! - Conversion methods on [`NumericScalarView`] / [`NumericScalarViewMut`]
//! - Convenience methods on [`NumericScalar`] that delegate to the view path
//!
//! The heavy conversion logic (software float decode/encode, IEEE 754 RTE,
//! intermediate-based casting) lives in [`crate::numeric_dtype::conversions`].

use crate::numeric_dtype::NumericDType;

use super::{NumericScalar, NumericScalarView, NumericScalarViewMut};

// ---------------------------------------------------------------------------
// Raw bit access: read/write u64 from/to byte slices at arbitrary bit offsets
// ---------------------------------------------------------------------------

/// Read up to 64 bits from a byte slice starting at `bit_offset`.
/// Returns the value as a u64 with only the low `total_bits` bits set.
pub fn read_raw_bits(data: &[u8], bit_offset: usize, total_bits: u8) -> u64 {
    if total_bits == 0 {
        return 0;
    }

    let byte_off = bit_offset / 8;
    let bit_shift = (bit_offset % 8) as u32;

    let bytes_needed = ((bit_shift as usize + total_bits as usize) + 7) / 8;
    let mut buf = [0u8; 9];
    let available = data.len().saturating_sub(byte_off);
    let to_copy = bytes_needed.min(available).min(9);
    buf[..to_copy].copy_from_slice(&data[byte_off..byte_off + to_copy]);

    let mut wide: u128 = 0;
    for (i, &b) in buf.iter().enumerate() {
        wide |= (b as u128) << (i * 8);
    }
    let shifted = (wide >> bit_shift) as u64;

    if total_bits >= 64 {
        shifted
    } else {
        shifted & ((1u64 << total_bits) - 1)
    }
}

/// Write up to 64 bits into a byte slice at `bit_offset`.
/// Only modifies the `total_bits` bits starting at `bit_offset`; other bits
/// in affected bytes are preserved.
pub fn write_raw_bits(data: &mut [u8], bit_offset: usize, total_bits: u8, value: u64) {
    if total_bits == 0 {
        return;
    }

    let byte_off = bit_offset / 8;
    let bit_shift = (bit_offset % 8) as u32;

    let masked_value = if total_bits >= 64 {
        value
    } else {
        value & ((1u64 << total_bits) - 1)
    };

    let bytes_needed = ((bit_shift as usize + total_bits as usize) + 7) / 8;
    let available = data.len().saturating_sub(byte_off);
    let to_touch = bytes_needed.min(available).min(9);

    let mut buf = [0u8; 9];
    buf[..to_touch].copy_from_slice(&data[byte_off..byte_off + to_touch]);

    let mut wide: u128 = 0;
    for (i, &b) in buf.iter().enumerate() {
        wide |= (b as u128) << (i * 8);
    }

    let bit_mask: u128 = if total_bits >= 64 {
        (u64::MAX as u128) << bit_shift
    } else {
        ((1u128 << total_bits) - 1) << bit_shift
    };

    wide = (wide & !bit_mask) | ((masked_value as u128) << bit_shift);

    for i in 0..to_touch {
        data[byte_off + i] = (wide >> (i * 8)) as u8;
    }
}

// ---------------------------------------------------------------------------
// NumericScalarView — conversion methods
// ---------------------------------------------------------------------------

impl<'a> NumericScalarView<'a> {
    /// Read the raw bits of this scalar as a u64.
    pub fn read_raw(&self) -> u64 {
        read_raw_bits(self.data, self.bit_offset, self.dtype.total_bits())
    }

    /// Cast this viewed value to a different dtype, returning an owned scalar.
    pub fn cast_to(&self, target: NumericDType) -> NumericScalar {
        let raw = self.read_raw();
        let cast_raw = self.dtype.cast_raw(raw, target);
        raw_to_scalar(cast_raw, target)
    }

    /// Read as an owned NumericScalar with the same dtype.
    pub fn to_owned_scalar(&self) -> NumericScalar {
        raw_to_scalar(self.read_raw(), self.dtype)
    }

    /// Convert to f64 regardless of source dtype.
    pub fn to_f64(&self) -> f64 {
        self.dtype.decode_to_f64(self.read_raw())
    }

    /// Convert to f32 regardless of source dtype.
    pub fn to_f32(&self) -> f32 {
        self.to_f64() as f32
    }

    /// Convert to i64 regardless of source dtype.
    pub fn to_i64(&self) -> i64 {
        let raw = self.read_raw();
        match self.dtype {
            NumericDType::SignedInt(it) => {
                let wide = it.decode_signed(raw);
                it.clamp_signed(wide) as i64
            }
            NumericDType::UnsignedInt(it) => {
                let u = it.decode_unsigned(raw);
                u.min(i64::MAX as u128) as i64
            }
            NumericDType::Float(ft) => {
                use crate::numeric_dtype::IntType;
                IntType::BITS_64.float_to_signed(ft.decode_f64(raw)) as i64
            }
            NumericDType::Bool => if raw != 0 { 1 } else { 0 },
        }
    }

    /// Convert to bool (nonzero = true).
    pub fn is_nonzero(&self) -> bool {
        let raw = self.read_raw();
        match self.dtype {
            NumericDType::Float(ft) => {
                let f = ft.decode_f64(raw);
                f != 0.0 && !f.is_nan()
            }
            _ => raw != 0,
        }
    }
}

// ---------------------------------------------------------------------------
// NumericScalarViewMut — write methods
// ---------------------------------------------------------------------------

impl<'a> NumericScalarViewMut<'a> {
    /// Write the raw bits of a scalar value at this view's offset.
    /// The scalar's dtype must match this view's dtype.
    pub fn write_scalar(&mut self, scalar: &NumericScalar) {
        debug_assert_eq!(self.dtype, scalar.dtype);
        let raw = read_raw_bits(&scalar.bits, 0, scalar.dtype.total_bits());
        write_raw_bits(self.data, self.bit_offset, self.dtype.total_bits(), raw);
    }

    /// Cast a scalar to this view's dtype and write it.
    pub fn write_cast(&mut self, scalar: &NumericScalar) {
        let raw = read_raw_bits(&scalar.bits, 0, scalar.dtype.total_bits());
        let cast_raw = scalar.dtype.cast_raw(raw, self.dtype);
        write_raw_bits(self.data, self.bit_offset, self.dtype.total_bits(), cast_raw);
    }

    /// Write an f64 value, converting to this view's dtype.
    pub fn write_f64(&mut self, value: f64) {
        let raw = self.dtype.encode_from_f64(value);
        write_raw_bits(self.data, self.bit_offset, self.dtype.total_bits(), raw);
    }

    /// Write raw bits directly.
    pub fn write_raw(&mut self, value: u64) {
        write_raw_bits(self.data, self.bit_offset, self.dtype.total_bits(), value);
    }
}

// ---------------------------------------------------------------------------
// NumericScalar — defers to view for all conversions
// ---------------------------------------------------------------------------

impl NumericScalar {
    /// Cast this scalar to a different dtype.
    pub fn cast_to(&self, target: NumericDType) -> NumericScalar {
        self.view().cast_to(target)
    }

    /// Convert to f64 regardless of source dtype.
    pub fn to_f64(&self) -> f64 {
        self.view().to_f64()
    }

    /// Convert to f32 regardless of source dtype.
    pub fn to_f32(&self) -> f32 {
        self.view().to_f32()
    }

    /// Convert to i64 regardless of source dtype.
    pub fn to_i64(&self) -> i64 {
        self.view().to_i64()
    }

    /// Convert to bool (nonzero = true).
    pub fn is_nonzero(&self) -> bool {
        self.view().is_nonzero()
    }
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

/// Pack raw bits into a NumericScalar.
fn raw_to_scalar(raw: u64, dtype: NumericDType) -> NumericScalar {
    let mut bits = [0u8; 8];
    let nbytes = dtype.bytes_per_element();
    bits[..nbytes].copy_from_slice(&raw.to_le_bytes()[..nbytes]);
    NumericScalar { bits, dtype }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric_dtype::FloatType;
    use half::{bf16, f16};
    use float8::{F8E4M3, F8E5M2};

    macro_rules! assert_bits {
        ($actual:expr, $expected:expr, $msg:expr) => {
            assert_eq!(
                $actual.as_le_bytes(), $expected.as_le_bytes(),
                "{}: actual {:?} != expected {:?}", $msg, $actual, $expected
            );
        };
    }

    // ===================================================================
    // Raw bit access tests
    // ===================================================================

    #[test]
    fn read_write_raw_bits_byte_aligned() {
        let data = [0x12u8, 0x34, 0x56, 0x78];
        assert_eq!(read_raw_bits(&data, 0, 32), 0x78563412);
        assert_eq!(read_raw_bits(&data, 0, 16), 0x3412);
        assert_eq!(read_raw_bits(&data, 16, 16), 0x7856);
        assert_eq!(read_raw_bits(&data, 0, 8), 0x12);
        assert_eq!(read_raw_bits(&data, 8, 8), 0x34);
    }

    #[test]
    fn read_raw_bits_sub_byte() {
        let data = [0b1010_0110u8];
        assert_eq!(read_raw_bits(&data, 0, 4), 0b0110);
        assert_eq!(read_raw_bits(&data, 4, 4), 0b1010);
        assert_eq!(read_raw_bits(&data, 0, 1), 0);
        assert_eq!(read_raw_bits(&data, 1, 1), 1);
        assert_eq!(read_raw_bits(&data, 5, 1), 1);
    }

    #[test]
    fn write_raw_bits_sub_byte_preserves_neighbors() {
        let mut data = [0xFFu8];
        write_raw_bits(&mut data, 2, 4, 0b0000);
        assert_eq!(data[0], 0b1100_0011);
    }

    #[test]
    fn write_raw_bits_spanning_bytes() {
        let mut data = [0u8; 4];
        write_raw_bits(&mut data, 4, 16, 0xABCD);
        assert_eq!(data[0], 0xD0);
        assert_eq!(data[1], 0xBC);
        assert_eq!(data[2], 0x0A);
    }

    #[test]
    fn read_write_raw_roundtrip_at_various_offsets() {
        for bit_offset in [0, 1, 3, 4, 7, 8, 12, 16] {
            for total_bits in [1, 4, 8, 16, 32] {
                let mut data = [0u8; 16];
                let value = 0xDEAD_BEEFu64 & ((1u64 << total_bits.min(64)) - 1);
                write_raw_bits(&mut data, bit_offset, total_bits, value);
                let readback = read_raw_bits(&data, bit_offset, total_bits);
                assert_eq!(readback, value,
                    "roundtrip failed: offset={bit_offset}, bits={total_bits}");
            }
        }
    }

    #[test]
    fn read_write_raw_bits_64bit_at_offsets() {
        for bit_offset in [0, 1, 4, 7, 8] {
            let mut data = [0u8; 16];
            let value = 0xDEAD_BEEF_CAFE_BABEu64;
            write_raw_bits(&mut data, bit_offset, 64, value);
            let readback = read_raw_bits(&data, bit_offset, 64);
            assert_eq!(readback, value,
                "64-bit roundtrip at offset {bit_offset}");
        }
    }

    // ===================================================================
    // View conversion tests
    // ===================================================================

    #[test]
    fn view_to_f64_byte_aligned_f32() {
        let mut buf = [0u8; 8];
        buf[..4].copy_from_slice(&3.14f32.to_le_bytes());
        let view = NumericScalarView { data: &buf, bit_offset: 0, dtype: NumericDType::F32 };
        let f = view.to_f64();
        assert!((f - 3.14f32 as f64).abs() < 1e-6);
    }

    #[test]
    fn view_to_f64_at_offset() {
        let mut buf = [0u8; 8];
        buf[..4].copy_from_slice(&1.5f32.to_le_bytes());
        buf[4..8].copy_from_slice(&2.5f32.to_le_bytes());
        let v0 = NumericScalarView { data: &buf, bit_offset: 0, dtype: NumericDType::F32 };
        let v1 = NumericScalarView { data: &buf, bit_offset: 32, dtype: NumericDType::F32 };
        assert!((v0.to_f64() - 1.5).abs() < 1e-6);
        assert!((v1.to_f64() - 2.5).abs() < 1e-6);
    }

    #[test]
    fn view_cast_bf16_to_f32() {
        let bf = bf16::from_f32(3.14);
        let mut buf = [0u8; 2];
        buf.copy_from_slice(&bf.to_le_bytes());
        let view = NumericScalarView { data: &buf, bit_offset: 0, dtype: NumericDType::BF16 };
        let result = view.cast_to(NumericDType::F32);
        let expected = NumericScalar::from_f32(bf.to_f32());
        assert_bits!(result, expected, "view bf16 → f32");
    }

    #[test]
    fn view_mut_write_f64() {
        let mut buf = [0u8; 4];
        let mut view = NumericScalarViewMut { data: &mut buf, bit_offset: 0, dtype: NumericDType::F32 };
        view.write_f64(3.14);
        let written = f32::from_le_bytes(buf);
        assert!((written - 3.14f32 as f32).abs() < 0.01);
    }

    #[test]
    fn view_mut_write_at_bit_offset_bool() {
        let mut buf = [0u8; 1];
        let mut view = NumericScalarViewMut { data: &mut buf, bit_offset: 3, dtype: NumericDType::BOOL };
        let scalar = NumericScalar::from_bool(true);
        view.write_scalar(&scalar);
        assert_eq!(buf[0], 0b0000_1000);
        let rview = NumericScalarView { data: &buf, bit_offset: 3, dtype: NumericDType::BOOL };
        assert!(rview.is_nonzero());
    }

    #[test]
    fn view_mut_write_cast() {
        let mut buf = [0u8; 2];
        let mut view = NumericScalarViewMut { data: &mut buf, bit_offset: 0, dtype: NumericDType::BF16 };
        let src = NumericScalar::from_f64(3.14);
        view.write_cast(&src);
        let rview = NumericScalarView { data: &buf, bit_offset: 0, dtype: NumericDType::BF16 };
        let result = rview.to_owned_scalar();
        let expected = NumericScalar::from_bf16(bf16::from_f64(3.14));
        assert_bits!(result, expected, "write_cast f64→bf16");
    }

    #[test]
    fn view_sub_byte_u4_packed() {
        let mut buf = [0u8; 1];
        {
            let mut v0 = NumericScalarViewMut { data: &mut buf, bit_offset: 0, dtype: NumericDType::U4 };
            v0.write_scalar(&NumericScalar::from_u4(arbitrary_int::u4::new(5)));
        }
        {
            let mut v1 = NumericScalarViewMut { data: &mut buf, bit_offset: 4, dtype: NumericDType::U4 };
            v1.write_scalar(&NumericScalar::from_u4(arbitrary_int::u4::new(12)));
        }
        assert_eq!(buf[0], 0xC5);
        let r0 = NumericScalarView { data: &buf, bit_offset: 0, dtype: NumericDType::U4 };
        let r1 = NumericScalarView { data: &buf, bit_offset: 4, dtype: NumericDType::U4 };
        assert_eq!(r0.to_owned_scalar(), NumericScalar::from_u4(arbitrary_int::u4::new(5)));
        assert_eq!(r1.to_owned_scalar(), NumericScalar::from_u4(arbitrary_int::u4::new(12)));
    }

    // ===================================================================
    // Scalar cast tests (via dtype engine)
    // ===================================================================

    #[test]
    fn f64_to_f32() {
        for &v in &[0.0f64, -0.0, 1.0, -1.0, 0.5, 3.14, f64::INFINITY, f64::NEG_INFINITY, f64::NAN, 1e-45, 1e38, std::f64::consts::PI] {
            let dst = NumericScalar::from_f64(v).cast_to(NumericDType::F32);
            assert_bits!(dst, NumericScalar::from_f32(v as f32), format!("f64({v}) → f32"));
        }
    }

    #[test]
    fn f32_to_bf16() {
        for &v in &[0.0f32, -0.0, 1.0, -1.0, 3.14, 0.5, f32::INFINITY, f32::NAN, f32::MAX, f32::MIN_POSITIVE, 256.0, 256.5, 257.0] {
            let dst = NumericScalar::from_f32(v).cast_to(NumericDType::BF16);
            assert_bits!(dst, NumericScalar::from_bf16(bf16::from_f32(v)), format!("f32({v}) → bf16"));
        }
    }

    #[test]
    fn f32_to_f16() {
        for &v in &[0.0f32, -0.0, 1.0, -1.0, 3.14, f32::INFINITY, f32::NAN, 65504.0, 65536.0, 5.96e-8, 1.5, 2.5] {
            let dst = NumericScalar::from_f32(v).cast_to(NumericDType::F16);
            assert_bits!(dst, NumericScalar::from_f16(f16::from_f32(v)), format!("f32({v}) → f16"));
        }
    }

    #[test]
    fn f32_to_i32() {
        for &(f, expected) in &[(0.0f32, 0i32), (1.0, 1), (-1.0, -1), (3.7, 3), (-3.7, -3), (f32::NAN, 0), (f32::INFINITY, i32::MAX)] {
            let dst = NumericScalar::from_f32(f).cast_to(NumericDType::I32);
            assert_bits!(dst, NumericScalar::from_i32(expected), format!("f32({f}) → i32"));
        }
    }

    #[test]
    fn i32_to_f32() {
        for &v in &[0i32, 1, -1, i32::MAX, i32::MIN, 16777217] {
            let dst = NumericScalar::from_i32(v).cast_to(NumericDType::F32);
            assert_bits!(dst, NumericScalar::from_f32(v as f32), format!("i32({v}) → f32"));
        }
    }

    #[test]
    fn identity_cast() {
        for s in [NumericScalar::from_f32(3.14), NumericScalar::from_i64(i64::MAX), NumericScalar::from_bool(true)] {
            assert_eq!(s.cast_to(s.dtype()), s);
        }
    }

    #[test]
    fn rte_f64_to_f32() {
        let mid_low = 1.0f64 + 0.5 * f32::EPSILON as f64;
        let mid_high = 1.0f64 + 1.5 * f32::EPSILON as f64;
        assert_bits!(NumericScalar::from_f64(mid_low).cast_to(NumericDType::F32), NumericScalar::from_f32(mid_low as f32), "RTE down");
        assert_bits!(NumericScalar::from_f64(mid_high).cast_to(NumericDType::F32), NumericScalar::from_f32(mid_high as f32), "RTE up");
    }

    #[test]
    fn to_f64_from_various() {
        assert_eq!(NumericScalar::from_f32(3.14f32).to_f64(), 3.14f32 as f64);
        assert_eq!(NumericScalar::from_i32(42).to_f64(), 42.0);
        assert_eq!(NumericScalar::from_bool(true).to_f64(), 1.0);
    }

    #[test]
    fn is_nonzero_from_various() {
        assert!(!NumericScalar::from_f32(0.0).is_nonzero());
        assert!(NumericScalar::from_f32(1.0).is_nonzero());
        assert!(!NumericScalar::from_i32(0).is_nonzero());
        assert!(NumericScalar::from_i32(-1).is_nonzero());
    }

    // ===================================================================
    // Exhaustive format tests (delegate to dtype engine, verify same results)
    // ===================================================================

    #[test]
    fn f4e2m1_exhaustive_roundtrip() {
        let ft = FloatType::F4E2M1;
        for bits in 0u8..16 {
            let decoded = ft.decode_f64(bits as u64);
            if decoded.is_nan() { continue; }
            let reencoded = ft.encode_f64(decoded);
            assert_eq!(reencoded, bits as u64,
                "F4E2M1 roundtrip 0b{:04b}: decoded {decoded}, reencoded 0b{:04b}", bits, reencoded);
        }
    }

    #[test]
    fn f8e4m3fn_exhaustive_roundtrip() {
        let ft = FloatType::F8E4M3FN;
        for bits in 0u8..=255 {
            let our_val = ft.decode_f64(bits as u64);
            // Verify against crate
            let crate_val = F8E4M3::from_bits(bits);
            if crate_val.is_nan() {
                assert!(our_val.is_nan(), "F8E4M3FN 0x{:02X}: crate=NaN, us={our_val}", bits);
                continue;
            }
            let crate_f64 = crate_val.to_f64();
            if crate_f64 == 0.0 {
                assert_eq!(our_val.to_bits(), crate_f64.to_bits(), "F8E4M3FN 0x{:02X} zero sign", bits);
            } else {
                assert_eq!(our_val, crate_f64, "F8E4M3FN 0x{:02X}", bits);
            }
            // Roundtrip
            let reencoded = ft.encode_f64(our_val);
            assert_eq!(reencoded, bits as u64, "F8E4M3FN roundtrip 0x{:02X}", bits);
        }
    }

    #[test]
    fn f16_exhaustive_roundtrip() {
        let ft = FloatType::F16;
        for bits in 0u16..=u16::MAX {
            let hf = f16::from_bits(bits);
            if hf.is_nan() { continue; }
            let decoded = ft.decode_f64(bits as u64);
            let reencoded = ft.encode_f64(decoded);
            assert_eq!(reencoded, bits as u64, "F16 roundtrip 0x{:04X}", bits);
        }
    }

    #[test]
    fn negative_float_to_unsigned_is_zero() {
        let src = NumericScalar::from_f64(-1.0);
        for dtype in [NumericDType::U8, NumericDType::U32, NumericDType::U64] {
            let dst = src.cast_to(dtype);
            assert_eq!(dst.bits, [0u8; 8], "negative f64 → {dtype} should be zero");
        }
    }

    #[test]
    fn i64_max_survives_identity() {
        let s = NumericScalar::from_i64(i64::MAX);
        assert_eq!(s.to_i64(), i64::MAX);
    }
}
