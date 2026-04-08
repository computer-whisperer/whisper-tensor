//! Codec layer: bit-level I/O, format conversion, and in-register
//! precision narrowing.
//!
//! Three sub-modules, each with one job:
//!
//! - [`bit_io`] — read/write `N` bits at any bit offset, no dtype
//!   awareness. Operates on bytes and bit offsets.
//! - [`format`] — convert between raw dtype bits (in a `u64` slot) and
//!   the chosen compute representation (`f32`, `f64`, or `u64`/`i64`).
//!   Knows the IEEE structure of the dtype.
//! - [`precision`] — `narrow_to`: emit the in-register sequence that
//!   enforces a target dtype's precision on a value already in the
//!   compute representation. Used for fused intermediates that never
//!   touch memory.
//!
//! See `x86_jit_codec.md` for the full design and recipes.

pub mod bit_io;
pub mod format;
pub mod precision;
