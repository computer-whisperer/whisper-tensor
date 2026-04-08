//! Address computation: `InputRef` → bit offset.
//!
//! Translates a NanoGraph `InputRef` (Broadcast, Strided, Explicit)
//! into the bit offset within the compiled span's buffer where the
//! referenced atom's bits live. Used by [`super::group`] to feed
//! [`super::super::codec::bit_io`] reads.
//!
//! Phase 0 stub. Implementation lands in phase 2.
