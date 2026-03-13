#![allow(clippy::all, dead_code, unreachable_patterns)]
//! v11-claude: Atom-level compiler for NanoGraph → native code via Cranelift.
//!
//! Key differences from v10:
//!
//! - **Atom-level reasoning**: Groups are treated as compression artifacts.
//!   All kernel construction resolves InputRefs at the atom level, not by
//!   reasoning about group relationships.
//!
//! - **Post-reduction fusion**: When a single-group reduction has one consumer
//!   reading via stride-1 Affine, the reduction is fused INTO the consumer's
//!   kernel — no intermediate materialization. The kernel computes the reduction
//!   and immediately applies post-reduction ops.
//!
//! - **Sample-based stride discovery**: Instead of v10's sibling-mapping to
//!   discover stride_k across k groups, we sample InputRef resolution at a few
//!   (i, k) points and fit affine functions. Directly uses atom addressing.

pub mod plan;

#[cfg(feature = "cranelift")]
pub mod codegen;

#[cfg(feature = "cranelift")]
pub mod pipeline;
