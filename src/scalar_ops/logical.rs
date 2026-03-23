//! Logical operations.
//!
//! Treat nonzero as true, zero as false. Return 0 or 1 as Bool raw bits.
//! These work on raw bits regardless of dtype — the caller provides the
//! truthiness decision by passing already-decoded nonzero/zero values.

/// Logical AND: both nonzero → 1, otherwise 0.
pub fn logical_and(a: u64, b: u64) -> u64 {
    ((a != 0) && (b != 0)) as u64
}

/// Logical OR: either nonzero → 1, otherwise 0.
pub fn logical_or(a: u64, b: u64) -> u64 {
    ((a != 0) || (b != 0)) as u64
}

/// Logical XOR: exactly one nonzero → 1, otherwise 0.
pub fn logical_xor(a: u64, b: u64) -> u64 {
    ((a != 0) ^ (b != 0)) as u64
}

/// Logical NOT: zero → 1, nonzero → 0.
pub fn logical_not(a: u64) -> u64 {
    (a == 0) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Logical AND --

    #[test]
    fn logical_and_basic() {
        assert_eq!(logical_and(1, 1), 1);
        assert_eq!(logical_and(1, 0), 0);
        assert_eq!(logical_and(0, 1), 0);
        assert_eq!(logical_and(0, 0), 0);
    }

    #[test]
    fn logical_and_nonzero_values() {
        assert_eq!(logical_and(42, 7), 1);
        assert_eq!(logical_and(42, 0), 0);
    }

    // -- Logical OR --

    #[test]
    fn logical_or_basic() {
        assert_eq!(logical_or(1, 1), 1);
        assert_eq!(logical_or(1, 0), 1);
        assert_eq!(logical_or(0, 1), 1);
        assert_eq!(logical_or(0, 0), 0);
    }

    #[test]
    fn logical_or_nonzero_values() {
        assert_eq!(logical_or(42, 0), 1);
        assert_eq!(logical_or(0, 0), 0);
    }

    // -- Logical XOR --

    #[test]
    fn logical_xor_basic() {
        assert_eq!(logical_xor(1, 1), 0);
        assert_eq!(logical_xor(1, 0), 1);
        assert_eq!(logical_xor(0, 1), 1);
        assert_eq!(logical_xor(0, 0), 0);
    }

    #[test]
    fn logical_xor_nonzero_values() {
        assert_eq!(logical_xor(42, 7), 0);
        assert_eq!(logical_xor(42, 0), 1);
    }

    // -- Logical NOT --

    #[test]
    fn logical_not_basic() {
        assert_eq!(logical_not(0), 1);
        assert_eq!(logical_not(1), 0);
    }

    #[test]
    fn logical_not_nonzero_values() {
        assert_eq!(logical_not(42), 0);
        assert_eq!(logical_not(u64::MAX), 0);
    }
}
