//! Compressed scalar DAG representation for full-model computation graphs.
//!
//! The nano graph dissolves all tensor boundaries, view ops, and indexing
//! into a single global DAG of scalar ALU operations. The DAG is stored in
//! a compressed form where regular patterns (elementwise, matmul, broadcast)
//! are represented as `PatternBlock`s — a template op plus iteration count
//! plus per-field stride descriptors.
//!
//! Dimensions with known sizes at graph-build time (weight shapes, hidden dims)
//! are fully expanded into concrete wiring. Unknown dimensions (batch, seq_len)
//! are represented as symbolic counts on pattern blocks.

pub mod lower;
mod ops;
pub mod optimize;
mod pattern;

pub use ops::{LiteralBits, ScalarBinOp, ScalarOp, ScalarUnaryOp};
pub use pattern::{
    AffineDimMap, BlockId, Dim, InputRef, NanoGraph, NanoGraphStats, PatternBlock, SourceDimMap,
};
