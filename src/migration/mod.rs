//! Legacy tensor and scalar types, being replaced by the new pool-based types.
//!
//! These modules contain the old `NumericTensor` (multi-backend dispatch enum),
//! `NumericScalar` (18-variant typed enum), and `NumericTensorTyped` (typed wrapper).
//! They will be removed once all call sites are migrated to the new type system.

pub mod bridge;
pub mod numeric_scalar;
pub mod numeric_tensor;
pub mod numeric_tensor_typed;
