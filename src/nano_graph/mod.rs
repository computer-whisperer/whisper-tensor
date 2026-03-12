//! Compressed scalar DAG representation for full-model computation graphs.
//!
//! The nano graph dissolves all tensor boundaries, view ops, and indexing
//! into a single global DAG of scalar operations. Known dimensions (weight
//! shapes, hidden dims) are fully expanded into concrete atom wiring. Unknown
//! dimensions (batch, seq_len) are represented as symbolic iteration
//! parameters on atom groups.

pub mod eval;
pub mod lower;
mod ops;
pub mod optimize;
mod pattern;

pub use ops::{LiteralBits, ScalarBinOp, ScalarOp, ScalarUnaryOp};
pub use pattern::{AtomGroup, AtomId, InputRef, NanoGraph, NanoGraphStats, SymDim};
