//! Compressed scalar DAG representation for full-model computation graphs.
//!
//! The nano graph dissolves all tensor boundaries, view ops, and indexing
//! into a single global DAG of scalar operations. Known dimensions (weight
//! shapes, hidden dims) are fully expanded into concrete atom wiring. Unknown
//! dimensions (batch, seq_len) are represented as symbolic iteration
//! parameters on atom groups.

pub mod lower;
pub mod ops;
pub mod pattern;
pub mod pool_eval;

pub use lower::{ConcatSegment, DimKind, NanoLoweringContext, ReduceAccessors, TensorAtomMap};
pub use ops::{ReduceKind, ScalarBinOp, ScalarOp, ScalarUnaryOp};
pub use pattern::{
    AtomGroup, AtomId, AtomRange, GroupUseCount, InputRef, NanoGraph, NanoGraphStats, SymDim,
};
