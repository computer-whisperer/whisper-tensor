//! v2 frontend: pure scalar ALU expression trees, no Load/Store.
//!
//! The v1 frontend produced a flat NanoOp stream with explicit Load/Store ops,
//! baking in kernel boundaries (one per milli op). The v2 frontend instead
//! produces `OutputBinding`s — each binding maps one output tensor element to
//! a `ScalarExpr` tree whose leaves are declarative tensor element references
//! (`Element`) and literal constants.
//!
//! This representation:
//! - Eliminates Load/Store from the IR (when/where to load is a scheduling decision)
//! - Makes view ops (Reshape, Transpose, Squeeze, Unsqueeze) dissolve into
//!   index arithmetic rather than being computation ops
//! - Leaves kernel boundary decisions to downstream passes (enabling fusion)
//! - Is the natural input for affine access pattern recovery and codegen

mod expand;
mod expr;

pub use expand::*;
pub use expr::*;
