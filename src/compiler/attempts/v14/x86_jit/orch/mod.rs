//! Orchestration layer: address computation, loop emission, fusion.
//!
//! The only layer that knows about `BufferLayout` and the executor
//! ABI. Calls into [`super::codec`] for memory I/O and into
//! [`super::ops`] for arithmetic.
//!
//! Sub-modules:
//!
//! - [`address`] — `compute_address`: turn an `InputRef` (Broadcast,
//!   Strided, Explicit) into a bit offset for the codec to read.
//! - [`group`] — emit a single `AtomGroup` as a loop or unrolled
//!   sequence. The base case for a non-fused, non-reduce group.
//! - [`reduce`] — emit a Reduce group as outer + inner loops. The
//!   inner loop must enforce per-step quantization (§4.7).
//! - [`chain`] — emit a fused chain of groups, forwarding compute-repr
//!   values between ops in registers and applying `narrow_to` between
//!   ops where the intermediate dtype demands it.
//!
//! Phase 0 stubs. The simple `group` path lands in phase 2 alongside
//! the bit-aware layout; reduce and fusion follow in phases 4 and 6.

pub mod address;
pub mod chain;
pub mod group;
pub mod reduce;
