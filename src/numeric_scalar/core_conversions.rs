//! Bit-level pack/unpack between Rust native types and [`NumericScalar`].
//!
//! Every function here just moves bits — no type conversion, no math.
//! `from_f32(3.14)` packs the f32's 4 bytes into `bits[0..4]` and sets
//! dtype to F32. `to_f32()` reads `bits[0..4]` as f32.
//!
//! For types that don't match the stored dtype, these functions will
//! produce garbage or panic — the cast operator in `conversions.rs`
//! handles actual type conversion.

use arbitrary_int::{i4, u4};
use float8::{F8E4M3, F8E5M2};
use half::{bf16, f16};

use crate::numeric_dtype::{FloatType, IntType, NumericDType};

use super::NumericScalar;

// ---------------------------------------------------------------------------
// Pack: Rust type → NumericScalar (sets dtype, copies bits)
// ---------------------------------------------------------------------------

impl NumericScalar {
    pub fn from_f64(v: f64) -> Self {
        Self {
            bits: v.to_le_bytes(),
            dtype: NumericDType::F64,
        }
    }

    pub fn from_f32(v: f32) -> Self {
        let mut bits = [0u8; 8];
        bits[..4].copy_from_slice(&v.to_le_bytes());
        Self {
            bits,
            dtype: NumericDType::F32,
        }
    }

    pub fn from_bf16(v: bf16) -> Self {
        let mut bits = [0u8; 8];
        bits[..2].copy_from_slice(&v.to_le_bytes());
        Self {
            bits,
            dtype: NumericDType::BF16,
        }
    }

    pub fn from_f16(v: f16) -> Self {
        let mut bits = [0u8; 8];
        bits[..2].copy_from_slice(&v.to_le_bytes());
        Self {
            bits,
            dtype: NumericDType::F16,
        }
    }

    pub fn from_f8e4m3fn(v: F8E4M3) -> Self {
        let mut bits = [0u8; 8];
        bits[0] = v.to_bits();
        Self {
            bits,
            dtype: NumericDType::F8E4M3FN,
        }
    }

    pub fn from_f8e5m2(v: F8E5M2) -> Self {
        let mut bits = [0u8; 8];
        bits[0] = v.to_bits();
        Self {
            bits,
            dtype: NumericDType::F8E5M2,
        }
    }

    pub fn from_i64(v: i64) -> Self {
        Self {
            bits: v.to_le_bytes(),
            dtype: NumericDType::I64,
        }
    }

    pub fn from_i32(v: i32) -> Self {
        let mut bits = [0u8; 8];
        bits[..4].copy_from_slice(&v.to_le_bytes());
        Self {
            bits,
            dtype: NumericDType::I32,
        }
    }

    pub fn from_i16(v: i16) -> Self {
        let mut bits = [0u8; 8];
        bits[..2].copy_from_slice(&v.to_le_bytes());
        Self {
            bits,
            dtype: NumericDType::I16,
        }
    }

    pub fn from_i8(v: i8) -> Self {
        let mut bits = [0u8; 8];
        bits[0] = v as u8;
        Self {
            bits,
            dtype: NumericDType::I8,
        }
    }

    pub fn from_u64(v: u64) -> Self {
        Self {
            bits: v.to_le_bytes(),
            dtype: NumericDType::U64,
        }
    }

    pub fn from_u32(v: u32) -> Self {
        let mut bits = [0u8; 8];
        bits[..4].copy_from_slice(&v.to_le_bytes());
        Self {
            bits,
            dtype: NumericDType::U32,
        }
    }

    pub fn from_u16(v: u16) -> Self {
        let mut bits = [0u8; 8];
        bits[..2].copy_from_slice(&v.to_le_bytes());
        Self {
            bits,
            dtype: NumericDType::U16,
        }
    }

    pub fn from_u8(v: u8) -> Self {
        let mut bits = [0u8; 8];
        bits[0] = v;
        Self {
            bits,
            dtype: NumericDType::U8,
        }
    }

    pub fn from_bool(v: bool) -> Self {
        let mut bits = [0u8; 8];
        bits[0] = v as u8;
        Self {
            bits,
            dtype: NumericDType::BOOL,
        }
    }

    pub fn from_i4(v: i4) -> Self {
        let mut bits = [0u8; 8];
        bits[0] = (v.value() as u8) & 0x0F;
        Self {
            bits,
            dtype: NumericDType::I4,
        }
    }

    pub fn from_u4(v: u4) -> Self {
        let mut bits = [0u8; 8];
        bits[0] = v.value() & 0x0F;
        Self {
            bits,
            dtype: NumericDType::U4,
        }
    }
}

// ---------------------------------------------------------------------------
// From trait impls: Rust type → NumericScalar
// ---------------------------------------------------------------------------

impl From<f64> for NumericScalar {
    fn from(v: f64) -> Self {
        Self::from_f64(v)
    }
}
impl From<f32> for NumericScalar {
    fn from(v: f32) -> Self {
        Self::from_f32(v)
    }
}
impl From<bf16> for NumericScalar {
    fn from(v: bf16) -> Self {
        Self::from_bf16(v)
    }
}
impl From<f16> for NumericScalar {
    fn from(v: f16) -> Self {
        Self::from_f16(v)
    }
}
impl From<F8E4M3> for NumericScalar {
    fn from(v: F8E4M3) -> Self {
        Self::from_f8e4m3fn(v)
    }
}
impl From<F8E5M2> for NumericScalar {
    fn from(v: F8E5M2) -> Self {
        Self::from_f8e5m2(v)
    }
}
impl From<i64> for NumericScalar {
    fn from(v: i64) -> Self {
        Self::from_i64(v)
    }
}
impl From<i32> for NumericScalar {
    fn from(v: i32) -> Self {
        Self::from_i32(v)
    }
}
impl From<i16> for NumericScalar {
    fn from(v: i16) -> Self {
        Self::from_i16(v)
    }
}
impl From<i8> for NumericScalar {
    fn from(v: i8) -> Self {
        Self::from_i8(v)
    }
}
impl From<u64> for NumericScalar {
    fn from(v: u64) -> Self {
        Self::from_u64(v)
    }
}
impl From<u32> for NumericScalar {
    fn from(v: u32) -> Self {
        Self::from_u32(v)
    }
}
impl From<u16> for NumericScalar {
    fn from(v: u16) -> Self {
        Self::from_u16(v)
    }
}
impl From<u8> for NumericScalar {
    fn from(v: u8) -> Self {
        Self::from_u8(v)
    }
}
impl From<bool> for NumericScalar {
    fn from(v: bool) -> Self {
        Self::from_bool(v)
    }
}
impl From<i4> for NumericScalar {
    fn from(v: i4) -> Self {
        Self::from_i4(v)
    }
}
impl From<u4> for NumericScalar {
    fn from(v: u4) -> Self {
        Self::from_u4(v)
    }
}

// ---------------------------------------------------------------------------
// Display helper
// ---------------------------------------------------------------------------

/// Format a scalar for display, reading bits according to dtype.
pub(crate) fn display_scalar(s: &NumericScalar) -> String {
    let b = &s.bits;
    match s.dtype {
        NumericDType::Float(ft) if ft == FloatType::F64 => {
            format!("{}", f64::from_le_bytes(*b))
        }
        NumericDType::Float(ft) if ft == FloatType::F32 => {
            format!("{}", f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        }
        NumericDType::Float(ft) if ft == FloatType::BF16 => {
            format!("{}", bf16::from_le_bytes([b[0], b[1]]))
        }
        NumericDType::Float(ft) if ft == FloatType::F16 => {
            format!("{}", f16::from_le_bytes([b[0], b[1]]))
        }
        NumericDType::Float(ft) if ft == FloatType::F8E4M3FN => {
            format!("{}", F8E4M3::from_bits(b[0]))
        }
        NumericDType::Float(ft) if ft == FloatType::F8E5M2 => {
            format!("{}", F8E5M2::from_bits(b[0]))
        }
        NumericDType::SignedInt(it) if it == IntType::BITS_64 => {
            format!("{}", i64::from_le_bytes(*b))
        }
        NumericDType::SignedInt(it) if it == IntType::BITS_32 => {
            format!("{}", i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        }
        NumericDType::SignedInt(it) if it == IntType::BITS_16 => {
            format!("{}", i16::from_le_bytes([b[0], b[1]]))
        }
        NumericDType::SignedInt(it) if it == IntType::BITS_8 => {
            format!("{}", b[0] as i8)
        }
        NumericDType::SignedInt(it) if it == IntType::BITS_4 => {
            let nibble = b[0] & 0x0F;
            let signed = if nibble & 0x08 != 0 { (nibble | 0xF0) as i8 } else { nibble as i8 };
            format!("{}", i4::new(signed))
        }
        NumericDType::UnsignedInt(it) if it == IntType::BITS_64 => {
            format!("{}", u64::from_le_bytes(*b))
        }
        NumericDType::UnsignedInt(it) if it == IntType::BITS_32 => {
            format!("{}", u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        }
        NumericDType::UnsignedInt(it) if it == IntType::BITS_16 => {
            format!("{}", u16::from_le_bytes([b[0], b[1]]))
        }
        NumericDType::UnsignedInt(it) if it == IntType::BITS_8 => {
            format!("{}", b[0])
        }
        NumericDType::UnsignedInt(it) if it == IntType::BITS_4 => {
            format!("{}", u4::new(b[0] & 0x0F))
        }
        NumericDType::Bool => {
            format!("{}", b[0] != 0)
        }
        _ => {
            let mut hex = String::from("0x");
            for byte in s.as_le_bytes().iter().rev() {
                hex.push_str(&format!("{byte:02x}"));
            }
            hex
        }
    }
}

// ---------------------------------------------------------------------------
// Temporary: construct a scalar with a specific dtype from an f64 value.
// This is a stopgap until the cast operator (conversions.rs) is built.
// Uses Rust's native type conversions as the bit source.
// ---------------------------------------------------------------------------

impl NumericScalar {
    /// Create a scalar of the given dtype from an f64 value.
    ///
    /// Converts through the appropriate native Rust type to get correct bits.
    /// This is a temporary API — will be replaced by the generic cast operator.
    pub fn from_f64_with_dtype(value: f64, dtype: NumericDType) -> Self {
        match dtype {
            NumericDType::Float(ft) if ft == FloatType::F64 => Self::from_f64(value),
            NumericDType::Float(ft) if ft == FloatType::F32 => Self::from_f32(value as f32),
            NumericDType::Float(ft) if ft == FloatType::BF16 => Self::from_bf16(bf16::from_f64(value)),
            NumericDType::Float(ft) if ft == FloatType::F16 => Self::from_f16(f16::from_f64(value)),
            NumericDType::Float(ft) if ft == FloatType::F8E4M3FN => Self::from_f8e4m3fn(F8E4M3::from(value as f32)),
            NumericDType::Float(ft) if ft == FloatType::F8E5M2 => Self::from_f8e5m2(F8E5M2::from(value as f32)),
            NumericDType::SignedInt(it) if it == IntType::BITS_64 => Self::from_i64(value as i64),
            NumericDType::SignedInt(it) if it == IntType::BITS_32 => Self::from_i32(value as i32),
            NumericDType::SignedInt(it) if it == IntType::BITS_16 => Self::from_i16(value as i16),
            NumericDType::SignedInt(it) if it == IntType::BITS_8 => Self::from_i8(value as i8),
            NumericDType::SignedInt(it) if it == IntType::BITS_4 => Self::from_i4(i4::new(value as i8)),
            NumericDType::UnsignedInt(it) if it == IntType::BITS_64 => Self::from_u64(value as u64),
            NumericDType::UnsignedInt(it) if it == IntType::BITS_32 => Self::from_u32(value as u32),
            NumericDType::UnsignedInt(it) if it == IntType::BITS_16 => Self::from_u16(value as u16),
            NumericDType::UnsignedInt(it) if it == IntType::BITS_8 => Self::from_u8(value as u8),
            NumericDType::UnsignedInt(it) if it == IntType::BITS_4 => Self::from_u4(u4::new(value as u8)),
            NumericDType::Bool => Self::from_bool(value != 0.0),
            _ => panic!("from_f64_with_dtype: unsupported dtype {dtype}"),
        }
    }
}

// ---------------------------------------------------------------------------
// View operations — read/write scalar data at arbitrary byte offsets in buffers
// ---------------------------------------------------------------------------

use super::{NumericScalarView, NumericScalarViewMut};

impl<'a> NumericScalarView<'a> {
    /// Read the bytes at this view's offset into an owned NumericScalar.
    /// Only supports byte-aligned access (bit_offset must be divisible by 8).
    pub fn to_owned(&self) -> NumericScalar {
        let byte_offset = self.bit_offset / 8;
        let nbytes = self.dtype.bytes_per_element();
        let mut bits = [0u8; 8];
        bits[..nbytes].copy_from_slice(&self.data[byte_offset..byte_offset + nbytes]);
        NumericScalar {
            bits,
            dtype: self.dtype,
        }
    }
}

impl<'a> NumericScalarViewMut<'a> {
    /// Write a NumericScalar's bytes at this view's offset.
    /// The scalar must have the same dtype. Only supports byte-aligned access.
    pub fn write_scalar(&mut self, scalar: &NumericScalar) {
        debug_assert_eq!(self.dtype, scalar.dtype);
        let byte_offset = self.bit_offset / 8;
        let nbytes = self.dtype.bytes_per_element();
        self.data[byte_offset..byte_offset + nbytes]
            .copy_from_slice(&scalar.bits[..nbytes]);
    }
}

// ---------------------------------------------------------------------------
// Tests — every roundtrip asserts bit-equality with the native Rust type path
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Assert that the bits stored in the scalar match the native type's bytes exactly.
    macro_rules! assert_bits_eq {
        ($scalar:expr, $expected_bytes:expr) => {
            assert_eq!(
                $scalar.as_le_bytes(),
                $expected_bytes,
                "bit mismatch for {:?}",
                $scalar
            );
        };
    }

    // -- f64 --

    #[test]
    fn f64_roundtrip() {
        for v in [0.0f64, 1.0, -1.0, f64::INFINITY, f64::NEG_INFINITY, f64::MIN, f64::MAX, f64::EPSILON, std::f64::consts::PI] {
            let s = NumericScalar::from_f64(v);
            assert_eq!(s.dtype(), NumericDType::F64);
            assert_bits_eq!(s, &v.to_le_bytes());
        }
    }

    #[test]
    fn f64_nan() {
        let s = NumericScalar::from_f64(f64::NAN);
        assert_bits_eq!(s, &f64::NAN.to_le_bytes());
    }

    #[test]
    fn f64_neg_zero() {
        assert_ne!(
            NumericScalar::from_f64(-0.0f64).raw_bits(),
            NumericScalar::from_f64(0.0f64).raw_bits()
        );
        assert_bits_eq!(NumericScalar::from_f64(-0.0), &(-0.0f64).to_le_bytes());
    }

    // -- f32 --

    #[test]
    fn f32_roundtrip() {
        for v in [0.0f32, 1.0, -1.0, f32::INFINITY, f32::NEG_INFINITY, f32::MIN, f32::MAX, f32::EPSILON, std::f32::consts::PI, 3.14] {
            let s = NumericScalar::from_f32(v);
            assert_eq!(s.dtype(), NumericDType::F32);
            assert_bits_eq!(s, &v.to_le_bytes());
        }
    }

    #[test]
    fn f32_nan() {
        assert_bits_eq!(NumericScalar::from_f32(f32::NAN), &f32::NAN.to_le_bytes());
    }

    // -- bf16 --

    #[test]
    fn bf16_roundtrip() {
        for v in [bf16::ZERO, bf16::ONE, bf16::NEG_ONE, bf16::INFINITY, bf16::NEG_INFINITY, bf16::from_f32(3.14), bf16::MIN_POSITIVE, bf16::MAX] {
            let s = NumericScalar::from_bf16(v);
            assert_eq!(s.dtype(), NumericDType::BF16);
            assert_bits_eq!(s, &v.to_le_bytes());
        }
    }

    #[test]
    fn bf16_nan() {
        assert_bits_eq!(NumericScalar::from_bf16(bf16::NAN), &bf16::NAN.to_le_bytes());
    }

    // -- f16 --

    #[test]
    fn f16_roundtrip() {
        for v in [f16::ZERO, f16::ONE, f16::NEG_ONE, f16::INFINITY, f16::NEG_INFINITY, f16::from_f32(3.14), f16::MIN_POSITIVE, f16::MAX] {
            let s = NumericScalar::from_f16(v);
            assert_eq!(s.dtype(), NumericDType::F16);
            assert_bits_eq!(s, &v.to_le_bytes());
        }
    }

    #[test]
    fn f16_nan() {
        assert_bits_eq!(NumericScalar::from_f16(f16::NAN), &f16::NAN.to_le_bytes());
    }

    // -- f8 types --

    #[test]
    fn f8e4m3fn_roundtrip() {
        for v in [F8E4M3::ZERO, F8E4M3::ONE, F8E4M3::from(3.0f32), F8E4M3::from(-1.5f32)] {
            let s = NumericScalar::from_f8e4m3fn(v);
            assert_eq!(s.dtype(), NumericDType::F8E4M3FN);
            assert_eq!(s.bits[0], v.to_bits());
        }
    }

    #[test]
    fn f8e5m2_roundtrip() {
        for v in [F8E5M2::ZERO, F8E5M2::ONE, F8E5M2::from(2.0f32), F8E5M2::from(-1.0f32)] {
            let s = NumericScalar::from_f8e5m2(v);
            assert_eq!(s.dtype(), NumericDType::F8E5M2);
            assert_eq!(s.bits[0], v.to_bits());
        }
    }

    // -- i64 --

    #[test]
    fn i64_roundtrip() {
        for v in [0i64, 1, -1, i64::MIN, i64::MAX, 42, -9999999999] {
            let s = NumericScalar::from_i64(v);
            assert_eq!(s.dtype(), NumericDType::I64);
            assert_bits_eq!(s, &v.to_le_bytes());
        }
    }

    // -- i32 --

    #[test]
    fn i32_roundtrip() {
        for v in [0i32, 1, -1, i32::MIN, i32::MAX, 42, -9999] {
            let s = NumericScalar::from_i32(v);
            assert_eq!(s.dtype(), NumericDType::I32);
            assert_bits_eq!(s, &v.to_le_bytes());
        }
    }

    // -- i16 --

    #[test]
    fn i16_roundtrip() {
        for v in [0i16, 1, -1, i16::MIN, i16::MAX, 42] {
            let s = NumericScalar::from_i16(v);
            assert_eq!(s.dtype(), NumericDType::I16);
            assert_bits_eq!(s, &v.to_le_bytes());
        }
    }

    // -- i8 --

    #[test]
    fn i8_roundtrip() {
        for v in [0i8, 1, -1, i8::MIN, i8::MAX, 42] {
            let s = NumericScalar::from_i8(v);
            assert_eq!(s.dtype(), NumericDType::I8);
            assert_eq!(s.as_le_bytes(), &[v as u8]);
        }
    }

    // -- u64 --

    #[test]
    fn u64_roundtrip() {
        for v in [0u64, 1, u64::MAX, u64::MAX - 1, 999999999999] {
            let s = NumericScalar::from_u64(v);
            assert_eq!(s.dtype(), NumericDType::U64);
            assert_bits_eq!(s, &v.to_le_bytes());
        }
    }

    // -- u32 --

    #[test]
    fn u32_roundtrip() {
        for v in [0u32, 1, u32::MAX, 42, 999999] {
            let s = NumericScalar::from_u32(v);
            assert_eq!(s.dtype(), NumericDType::U32);
            assert_bits_eq!(s, &v.to_le_bytes());
        }
    }

    // -- u16 --

    #[test]
    fn u16_roundtrip() {
        for v in [0u16, 1, u16::MAX, 42] {
            let s = NumericScalar::from_u16(v);
            assert_eq!(s.dtype(), NumericDType::U16);
            assert_bits_eq!(s, &v.to_le_bytes());
        }
    }

    // -- u8 --

    #[test]
    fn u8_roundtrip() {
        for v in [0u8, 1, u8::MAX, 42] {
            let s = NumericScalar::from_u8(v);
            assert_eq!(s.dtype(), NumericDType::U8);
            assert_eq!(s.as_le_bytes(), &[v]);
        }
    }

    // -- bool --

    #[test]
    fn bool_roundtrip() {
        let s_true = NumericScalar::from_bool(true);
        let s_false = NumericScalar::from_bool(false);
        assert_eq!(s_true.dtype(), NumericDType::BOOL);
        assert_eq!(s_false.dtype(), NumericDType::BOOL);
        assert_eq!(s_true.as_le_bytes(), &[1]);
        assert_eq!(s_false.as_le_bytes(), &[0]);
    }

    // -- i4 --

    #[test]
    fn i4_roundtrip() {
        for raw in -8i8..=7 {
            let v = i4::new(raw);
            let s = NumericScalar::from_i4(v);
            assert_eq!(s.dtype(), NumericDType::I4);
            // Verify the stored nibble matches the expected bit pattern
            let expected_nibble = (raw as u8) & 0x0F;
            assert_eq!(s.bits[0], expected_nibble);
        }
    }

    // -- u4 --

    #[test]
    fn u4_roundtrip() {
        for raw in 0u8..=15 {
            let v = u4::new(raw);
            let s = NumericScalar::from_u4(v);
            assert_eq!(s.dtype(), NumericDType::U4);
            assert_eq!(s.bits[0], raw);
        }
    }

    // -- From trait impls --

    #[test]
    fn from_trait_f32() {
        let s: NumericScalar = 3.14f32.into();
        assert_eq!(s.dtype(), NumericDType::F32);
        assert_bits_eq!(s, &3.14f32.to_le_bytes());
    }

    #[test]
    fn from_trait_i64() {
        let s: NumericScalar = i64::MAX.into();
        assert_eq!(s.dtype(), NumericDType::I64);
        assert_bits_eq!(s, &i64::MAX.to_le_bytes());
    }

    #[test]
    fn from_trait_bool() {
        let s: NumericScalar = true.into();
        assert_eq!(s.dtype(), NumericDType::BOOL);
        assert_eq!(s.as_le_bytes(), &[1]);
    }

    // -- Zero padding: upper bytes are always zero --

    #[test]
    fn upper_bytes_zeroed() {
        let s = NumericScalar::from_f32(f32::MAX);
        assert_eq!(&s.raw_bits()[4..], &[0, 0, 0, 0]);

        let s = NumericScalar::from_u8(0xFF);
        assert_eq!(&s.raw_bits()[1..], &[0, 0, 0, 0, 0, 0, 0]);

        let s = NumericScalar::from_bf16(bf16::MAX);
        assert_eq!(&s.raw_bits()[2..], &[0, 0, 0, 0, 0, 0]);
    }

    // -- Bit equality: same value produces same bits regardless of path --

    #[test]
    fn bit_equality_f64() {
        let a = NumericScalar::from_f64(1.0);
        let b: NumericScalar = 1.0f64.into();
        assert_eq!(a, b);
    }

    #[test]
    fn bit_equality_i32_negative() {
        let v = -42i32;
        let s = NumericScalar::from_i32(v);
        assert_eq!(&s.as_le_bytes()[..4], &v.to_le_bytes());
    }

    // -- PartialEq is bit-level, not value-level --

    #[test]
    fn neg_zero_ne_pos_zero_f64() {
        let pos = NumericScalar::from_f64(0.0);
        let neg = NumericScalar::from_f64(-0.0);
        assert_ne!(pos, neg);
    }

    #[test]
    fn nan_eq_same_nan_f32() {
        let a = NumericScalar::from_f32(f32::NAN);
        let b = NumericScalar::from_f32(f32::NAN);
        assert_eq!(a, b);
    }

    // -- Display --

    #[test]
    fn display_f32() {
        let s = NumericScalar::from_f32(3.14);
        assert_eq!(format!("{s}"), format!("{}", 3.14f32));
    }

    #[test]
    fn display_i32() {
        let s = NumericScalar::from_i32(-42);
        assert_eq!(format!("{s}"), "-42");
    }

    #[test]
    fn display_bool() {
        assert_eq!(format!("{}", NumericScalar::from_bool(true)), "true");
        assert_eq!(format!("{}", NumericScalar::from_bool(false)), "false");
    }
}
