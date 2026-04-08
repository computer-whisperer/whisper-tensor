//! Emit a Reduce group as outer + inner loops.
//!
//! The inner loop accumulates `reduce_count` values into the
//! compute-repr accumulator. Per dtype contract §4.7, the accumulator
//! must be at the *full compute_dtype precision* after every iteration
//! — for narrow compute_dtypes (BF16, F16, etc.) this means an
//! explicit `narrow_to` call from [`super::super::codec::precision`]
//! after each op. **A wider hardware accumulator is non-compliant.**
//!
//! Phase 0 stub. Implementation lands in phase 4.
