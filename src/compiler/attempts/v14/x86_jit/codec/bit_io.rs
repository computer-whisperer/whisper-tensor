//! Bit-level I/O: read or write `N` bits at any bit offset within a
//! buffer, regardless of byte alignment.
//!
//! This module emits the assembly sequences that load `elem_bits` bits
//! from a buffer pointer + bit offset into a 64-bit GP register, and
//! the inverse for writes. It is dtype-agnostic — the bits land in the
//! low `elem_bits` of the register; higher bits are zeroed for unsigned
//! reads. Sign extension and float decoding belong to [`super::format`].
//!
//! For phase 0 this file is a stub; the bit-aware load/store sequences
//! land in phase 2 alongside the bit-aware `BufferLayout` rewrite.
