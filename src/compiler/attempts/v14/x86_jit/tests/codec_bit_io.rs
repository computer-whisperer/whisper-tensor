//! Unit tests for [`super::super::codec::bit_io`].
//!
//! Tests cover:
//!   - Round-trip read/write at every bit alignment 0..7 within a byte
//!   - Element widths from 1 to 64 bits, including the spanning case
//!     (`n_bits ≥ 58`) that uses the SHRD/SHLD path
//!   - Read-modify-write preservation: surrounding bits in the buffer
//!     are not disturbed by a partial-byte store
//!
//! Test harness ABI:
//!   - `extern "C" fn(buf: *const u8, bit_off: u64) -> u64` for loads
//!   - `extern "C" fn(buf: *mut u8, bit_off: u64, value: u64)` for stores
//!
//! Register assignments inside the test JIT:
//!   - `rdi` (7) — buf base
//!   - `rsi` (6) — bit offset
//!   - `rdx` (2) — store source value
//!   - `rax` (0) — load destination / return
//!   - `r8` (8), `r9` (9), `r10` (10) — scratch
//!   - `rcx` (1) — codec shift count (clobbered)
//!
//! All registers used here are caller-saved under System V, so the
//! harness needs no prologue / epilogue beyond the trailing `ret`.

use super::super::codec::bit_io::{emit_load_bits, emit_store_bits};
use super::jit_harness::JitFn;

const BASE: u8 = 7; // rdi
const BIT_OFF: u8 = 6; // rsi
const SRC_VAL: u8 = 2; // rdx
const DST: u8 = 0; // rax
const SCRATCH1: u8 = 8; // r8
const SCRATCH2: u8 = 9; // r9
const SCRATCH3: u8 = 10; // r10

/// Build a load-test JIT for a fixed `n_bits`.
fn build_loader(n_bits: u32) -> JitFn {
    JitFn::build(|asm| {
        emit_load_bits(asm, BASE, BIT_OFF, n_bits, DST, SCRATCH1);
    })
}

/// Build a store-test JIT for a fixed `n_bits`.
fn build_storer(n_bits: u32) -> JitFn {
    JitFn::build(|asm| {
        emit_store_bits(
            asm, BASE, BIT_OFF, n_bits, SRC_VAL, SCRATCH1, SCRATCH2, SCRATCH3,
        );
    })
}

/// Reference loader: read `n_bits` from `buf` at bit offset `off`.
fn ref_load(buf: &[u8], off: u64, n_bits: u32) -> u64 {
    let mut acc: u64 = 0;
    for i in 0..n_bits {
        let bit = (buf[((off + i as u64) / 8) as usize] >> ((off + i as u64) % 8)) & 1;
        acc |= (bit as u64) << i;
    }
    acc
}

/// Reference storer: write `n_bits` of `value` to `buf` at bit offset `off`.
fn ref_store(buf: &mut [u8], off: u64, value: u64, n_bits: u32) {
    for i in 0..n_bits {
        let bit = ((value >> i) & 1) as u8;
        let byte_idx = ((off + i as u64) / 8) as usize;
        let bit_idx = ((off + i as u64) % 8) as u8;
        buf[byte_idx] = (buf[byte_idx] & !(1 << bit_idx)) | (bit << bit_idx);
    }
}

/// Allocate a buffer with at least 16 bytes of slack past the highest
/// addressable bit. The codec's general path may read 16 bytes
/// starting at `bit_off / 8`.
fn alloc_buf(min_bits: u64) -> Vec<u8> {
    let needed_bytes = (min_bits / 8 + 16) as usize;
    vec![0u8; needed_bytes]
}

#[test]
fn load_byte_aligned_widths() {
    // Byte-aligned offset, widths 1..=64. Buffer holds a known
    // pattern; assert each width returns the right low n_bits.
    for &n_bits in &[1u32, 4, 7, 8, 13, 16, 24, 31, 32, 33, 56, 57, 58, 63, 64] {
        let mut buf = alloc_buf(64);
        // Spread a recognisable byte pattern.
        for (i, b) in buf.iter_mut().enumerate() {
            *b = ((i * 17) ^ 0x5a) as u8;
        }
        let jit = build_loader(n_bits);
        let f: extern "C" fn(*const u8, u64) -> u64 = unsafe { std::mem::transmute(jit.ptr()) };
        let got = f(buf.as_ptr(), 0);
        let want = ref_load(&buf, 0, n_bits);
        assert_eq!(
            got, want,
            "load n_bits={n_bits}: got 0x{got:016x} want 0x{want:016x}"
        );
    }
}

#[test]
fn load_every_bit_alignment() {
    // For each width, sweep bit_off through 0..16 (covers all 0..7
    // sub-byte alignments AND tests starting in different bytes).
    for &n_bits in &[1u32, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 56, 57, 58, 63, 64] {
        let mut buf = alloc_buf(128);
        for (i, b) in buf.iter_mut().enumerate() {
            *b = ((i * 31) ^ 0xa5) as u8;
        }
        let jit = build_loader(n_bits);
        let f: extern "C" fn(*const u8, u64) -> u64 = unsafe { std::mem::transmute(jit.ptr()) };
        for off in 0u64..16 {
            let got = f(buf.as_ptr(), off);
            let want = ref_load(&buf, off, n_bits);
            assert_eq!(
                got, want,
                "load n_bits={n_bits} off={off}: got 0x{got:016x} want 0x{want:016x}"
            );
        }
    }
}

#[test]
fn store_byte_aligned_widths() {
    for &n_bits in &[1u32, 4, 7, 8, 13, 16, 24, 31, 32, 33, 56, 57, 58, 63, 64] {
        let mut got_buf = alloc_buf(128);
        let mut want_buf = got_buf.clone();
        let value: u64 = if n_bits == 64 {
            0xdeadbeefcafef00du64
        } else {
            (0xdeadbeefcafef00du64) & ((1u64 << n_bits) - 1)
        };
        let jit = build_storer(n_bits);
        let f: extern "C" fn(*mut u8, u64, u64) = unsafe { std::mem::transmute(jit.ptr()) };
        f(got_buf.as_mut_ptr(), 0, value);
        ref_store(&mut want_buf, 0, value, n_bits);
        assert_eq!(got_buf, want_buf, "store n_bits={n_bits}: buffers differ");
    }
}

#[test]
fn store_every_bit_alignment() {
    for &n_bits in &[1u32, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 56, 57, 58, 63, 64] {
        for off in 0u64..16 {
            let mut got_buf = alloc_buf(256);
            // Pre-fill with a pattern so we can detect any preserved
            // bits getting clobbered.
            for (i, b) in got_buf.iter_mut().enumerate() {
                *b = ((i * 13) ^ 0x3c) as u8;
            }
            let mut want_buf = got_buf.clone();

            let value: u64 = if n_bits == 64 {
                0x0123456789abcdefu64
            } else {
                0x0123456789abcdefu64 & ((1u64 << n_bits) - 1)
            };

            let jit = build_storer(n_bits);
            let f: extern "C" fn(*mut u8, u64, u64) = unsafe { std::mem::transmute(jit.ptr()) };
            f(got_buf.as_mut_ptr(), off, value);
            ref_store(&mut want_buf, off, value, n_bits);

            assert_eq!(
                got_buf, want_buf,
                "store n_bits={n_bits} off={off}: buffers differ"
            );
        }
    }
}

#[test]
fn round_trip_load_after_store_every_alignment() {
    // For each (n_bits, bit_off): write a value, read it back,
    // assert the read returns the value. Catches mask/shift bugs
    // that compensate between load and store.
    for &n_bits in &[1u32, 4, 8, 13, 17, 32, 33, 57, 58, 63, 64] {
        for off in 0u64..16 {
            let mut buf = alloc_buf(256);
            for (i, b) in buf.iter_mut().enumerate() {
                *b = ((i * 7) ^ 0xc3) as u8;
            }
            let value: u64 = if n_bits == 64 {
                0xfedcba9876543210u64
            } else {
                0xfedcba9876543210u64 & ((1u64 << n_bits) - 1)
            };

            let store_jit = build_storer(n_bits);
            let load_jit = build_loader(n_bits);
            let store_fn: extern "C" fn(*mut u8, u64, u64) =
                unsafe { std::mem::transmute(store_jit.ptr()) };
            let load_fn: extern "C" fn(*const u8, u64) -> u64 =
                unsafe { std::mem::transmute(load_jit.ptr()) };

            store_fn(buf.as_mut_ptr(), off, value);
            let got = load_fn(buf.as_ptr(), off);
            assert_eq!(
                got, value,
                "round-trip n_bits={n_bits} off={off}: got 0x{got:016x} want 0x{value:016x}"
            );
        }
    }
}

#[test]
fn store_preserves_surrounding_bits() {
    // Pre-fill a buffer with all-ones, store a known small value at
    // an offset, and verify only the targeted bits changed.
    for &n_bits in &[1u32, 3, 5, 7, 12, 23, 41] {
        for off in 0u64..16 {
            let mut got_buf = alloc_buf(256);
            for b in &mut got_buf {
                *b = 0xff;
            }
            let mut want_buf = got_buf.clone();

            let value: u64 = 0; // store all-zero into the target bits

            let jit = build_storer(n_bits);
            let f: extern "C" fn(*mut u8, u64, u64) = unsafe { std::mem::transmute(jit.ptr()) };
            f(got_buf.as_mut_ptr(), off, value);
            ref_store(&mut want_buf, off, value, n_bits);

            assert_eq!(got_buf, want_buf, "preserve test n_bits={n_bits} off={off}");
        }
    }
}
