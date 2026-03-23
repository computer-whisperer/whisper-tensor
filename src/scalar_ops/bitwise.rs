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
}
