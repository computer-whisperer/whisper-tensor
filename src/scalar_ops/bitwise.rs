//! Bitwise operations for integers.

use crate::numeric_dtype::IntType;

/// Bitwise NOT: flip all bits within type width.
pub fn bitwise_not(raw: u64, it: &IntType) -> u64 {
    let mask = if it.bits >= 64 {
        u64::MAX
    } else {
        (1u64 << it.bits) - 1
    };
    (!raw) & mask
}

/// Bitwise AND.
pub fn bitwise_and(a: u64, b: u64, _it: &IntType) -> u64 {
    a & b
}

/// Bitwise OR.
pub fn bitwise_or(a: u64, b: u64, _it: &IntType) -> u64 {
    a | b
}

/// Bitwise XOR.
pub fn bitwise_xor(a: u64, b: u64, _it: &IntType) -> u64 {
    a ^ b
}

/// Mask a u64 to `it.bits` width (zeros above bit `it.bits-1`).
fn mask_to_width(raw: u64, it: &IntType) -> u64 {
    if it.bits >= 64 {
        raw
    } else {
        raw & ((1u64 << it.bits) - 1)
    }
}

/// Compute the effective shift count: `b mod it.bits`. The shift count
/// is taken modulo the type width per `docs/dtype_contract.md` §5.8.
fn shift_amount(b: u64, it: &IntType) -> u32 {
    (b as u32) % (it.bits as u32)
}

/// Bit shift left. Same for signed and unsigned: shift in zeros from
/// the right, mask to type width. Shift count is `b mod it.bits`.
pub fn shift_left(a: u64, b: u64, it: &IntType) -> u64 {
    let count = shift_amount(b, it);
    let raw = mask_to_width(a, it);
    mask_to_width(raw << count, it)
}

/// Arithmetic shift right for signed integers. Sign-extends from the
/// left so negative values stay negative. Shift count is `b mod it.bits`.
pub fn signed_shift_right(a: u64, b: u64, it: &IntType) -> u64 {
    let count = shift_amount(b, it);
    // Decode to sign-extended i64; arithmetic shift right preserves sign.
    let signed = it.decode_signed(a) as i64;
    let shifted = signed >> count;
    mask_to_width(shifted as u64, it)
}

/// Logical shift right for unsigned integers. Shifts in zeros from the
/// left. Shift count is `b mod it.bits`.
pub fn unsigned_shift_right(a: u64, b: u64, it: &IntType) -> u64 {
    let count = shift_amount(b, it);
    let raw = mask_to_width(a, it);
    mask_to_width(raw >> count, it)
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Bitwise NOT --

    #[test]
    fn bitwise_not_u8() {
        let it = IntType::BITS_8;
        assert_eq!(bitwise_not(0x00, &it), 0xFF);
        assert_eq!(bitwise_not(0xFF, &it), 0x00);
        assert_eq!(bitwise_not(0x0F, &it), 0xF0);
    }

    #[test]
    fn bitwise_not_u32() {
        let it = IntType::BITS_32;
        assert_eq!(bitwise_not(0, &it), 0xFFFF_FFFF);
        assert_eq!(bitwise_not(0xFFFF_FFFF, &it), 0);
    }

    #[test]
    fn bitwise_not_u64() {
        let it = IntType::BITS_64;
        assert_eq!(bitwise_not(0, &it), u64::MAX);
        assert_eq!(bitwise_not(u64::MAX, &it), 0);
    }

    // -- Bitwise AND --

    #[test]
    fn bitwise_and_basic() {
        let it = IntType::BITS_8;
        assert_eq!(bitwise_and(0xFF, 0x0F, &it), 0x0F);
        assert_eq!(bitwise_and(0xAA, 0x55, &it), 0x00);
        assert_eq!(bitwise_and(0xFF, 0xFF, &it), 0xFF);
    }

    // -- Bitwise OR --

    #[test]
    fn bitwise_or_basic() {
        let it = IntType::BITS_8;
        assert_eq!(bitwise_or(0xF0, 0x0F, &it), 0xFF);
        assert_eq!(bitwise_or(0x00, 0x00, &it), 0x00);
    }

    // -- Bitwise XOR --

    #[test]
    fn bitwise_xor_basic() {
        let it = IntType::BITS_8;
        assert_eq!(bitwise_xor(0xFF, 0xFF, &it), 0x00);
        assert_eq!(bitwise_xor(0xAA, 0x55, &it), 0xFF);
        assert_eq!(bitwise_xor(0x00, 0x00, &it), 0x00);
    }

    #[test]
    fn bitwise_xor_self_inverse() {
        let it = IntType::BITS_32;
        let v = 0xDEAD_BEEF_u64;
        let key = 0x1234_5678_u64;
        let encrypted = bitwise_xor(v, key, &it);
        let decrypted = bitwise_xor(encrypted, key, &it);
        assert_eq!(decrypted, v);
    }

    // -- Shift left --

    #[test]
    fn shift_left_basic() {
        let it = IntType::BITS_32;
        assert_eq!(shift_left(1, 0, &it), 1);
        assert_eq!(shift_left(1, 1, &it), 2);
        assert_eq!(shift_left(1, 4, &it), 16);
        assert_eq!(shift_left(0xFF, 8, &it), 0xFF00);
    }

    #[test]
    fn shift_left_overflow_masks_to_width() {
        let it = IntType::BITS_8;
        // 0x40 << 2 = 0x100, masked to 8 bits = 0x00
        assert_eq!(shift_left(0x40, 2, &it), 0x00);
        // 0x40 << 1 = 0x80, fits in 8 bits
        assert_eq!(shift_left(0x40, 1, &it), 0x80);
    }

    #[test]
    fn shift_left_count_modulo_width() {
        let it = IntType::BITS_32;
        // count 32 == 0 mod 32 → no shift
        assert_eq!(shift_left(1, 32, &it), 1);
        // count 33 == 1 mod 32
        assert_eq!(shift_left(1, 33, &it), 2);
    }

    // -- Signed shift right (arithmetic) --

    #[test]
    fn signed_shift_right_positive() {
        let it = IntType::BITS_32;
        // 16 >> 2 = 4
        let a = it.encode_signed(16);
        let r = signed_shift_right(a, 2, &it);
        assert_eq!(it.decode_signed(r), 4);
    }

    #[test]
    fn signed_shift_right_negative_arithmetic() {
        // Critical test: arithmetic shift right preserves sign for
        // negative values. -8 >> 1 = -4 (NOT some large positive number).
        let it = IntType::BITS_32;
        let a = it.encode_signed(-8);
        let r = signed_shift_right(a, 1, &it);
        assert_eq!(it.decode_signed(r), -4);

        // -1 >> 1 = -1 (all ones stays all ones)
        let a = it.encode_signed(-1);
        let r = signed_shift_right(a, 1, &it);
        assert_eq!(it.decode_signed(r), -1);

        // -16 >> 2 = -4
        let a = it.encode_signed(-16);
        let r = signed_shift_right(a, 2, &it);
        assert_eq!(it.decode_signed(r), -4);
    }

    #[test]
    fn signed_shift_right_narrow() {
        // i8: -100 >> 1 = -50 (sign-extended within 8 bits)
        let it = IntType::BITS_8;
        let a = it.encode_signed(-100);
        let r = signed_shift_right(a, 1, &it);
        assert_eq!(it.decode_signed(r), -50);

        // i16: i16::MIN >> 1 = i16::MIN / 2 = -16384
        let it = IntType::BITS_16;
        let a = it.encode_signed(i16::MIN as i128);
        let r = signed_shift_right(a, 1, &it);
        assert_eq!(it.decode_signed(r), -16384);
    }

    #[test]
    fn signed_shift_right_count_modulo_width() {
        let it = IntType::BITS_32;
        // count 32 == 0 mod 32
        let a = it.encode_signed(42);
        let r = signed_shift_right(a, 32, &it);
        assert_eq!(it.decode_signed(r), 42);
    }

    // -- Unsigned shift right (logical) --

    #[test]
    fn unsigned_shift_right_basic() {
        let it = IntType::BITS_32;
        let a = it.encode_unsigned(16);
        let r = unsigned_shift_right(a, 2, &it);
        assert_eq!(it.decode_unsigned(r), 4);
    }

    #[test]
    fn unsigned_shift_right_high_bit() {
        // u8: 0x80 >> 1 = 0x40 (zeros shifted in from left)
        let it = IntType::BITS_8;
        assert_eq!(unsigned_shift_right(0x80, 1, &it), 0x40);
        // u8: 0xFF >> 4 = 0x0F
        assert_eq!(unsigned_shift_right(0xFF, 4, &it), 0x0F);
    }

    #[test]
    fn unsigned_vs_signed_shift_right_diverge_on_high_bit() {
        // Critical correctness test: a high-bit value of an 8-bit type
        // shifts differently as signed vs unsigned.
        let it = IntType::BITS_8;

        // Unsigned: 0x80 (= 128 unsigned) >> 1 = 0x40 (= 64 unsigned)
        assert_eq!(unsigned_shift_right(0x80, 1, &it), 0x40);

        // Signed: 0x80 (= -128 signed) >> 1 = -64 = 0xC0
        assert_eq!(signed_shift_right(0x80, 1, &it), 0xC0);
    }
}
