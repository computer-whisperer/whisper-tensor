//! Bit-level I/O: read or write `N` bits at any bit offset within a
//! buffer, regardless of byte alignment.
//!
//! This module emits the assembly sequences that load `n_bits` bits
//! from `[base + (bit_off / 8)]` at sub-byte position `bit_off & 7`
//! into a 64-bit GP register, and the inverse for writes. It is
//! dtype-agnostic: the bits land in the low `n_bits` of the register,
//! right-justified, zero-extended for reads. Sign extension and float
//! decoding are [`super::format`]'s job.
//!
//! # General-path algorithm
//!
//! Both load and store handle bit offsets `0..7` within a byte and bit
//! widths `1..=64`. The single-load path covers `n_bits ≤ 57` (worst
//! case `bit_in_byte=7, n_bits=57` → 64 bits, fits in one qword). For
//! `n_bits ≥ 58`, the value may span 9 bytes, so the emitted code reads
//! a second qword at offset+8 and uses `shrd`/`shld` to combine.
//!
//! # Buffer slack contract
//!
//! The general-path emit reads up to 16 bytes starting at `base + (
//! bit_off / 8)`. Callers must ensure the buffer has **at least 8 bytes
//! of valid memory** after the highest addressable bit. The
//! [`super::super::layout::BufferLayout`] will be extended in phase 2.B
//! to guarantee this via tail padding for any slot the codec touches.
//!
//! # `rcx` clobbering
//!
//! x86's variable shifts (`shl`, `shr`, `shrd`, `shld`) require the
//! count in `cl`. The codec uses `rcx` exclusively as the shift count
//! register; both `emit_load_bits` and `emit_store_bits` clobber the
//! entire `rcx` (the high 56 bits are scratch — we use `rcx` first to
//! materialize the byte offset, then overwrite `cl` with the shift
//! count, then *recompute* the byte offset before any subsequent
//! address use). Slot B in the Int compute repr lives in `rcx` by
//! convention; orch must arrange that codec calls do not collide
//! with a live slot-B value.
//!
//! # Byte-aligned fast path
//!
//! Phase 6 will add a byte-aligned fast path that bypasses the bit
//! arithmetic when the static slot offset is known to be byte-aligned
//! AND the dtype's `total_bits` is a power of two ≤ 64. The general
//! path is the default and is what this module emits.

use dynasmrt::{DynasmApi, dynasm, x64::Assembler};

/// Emit code that loads `n_bits` bits at the bit offset held in
/// `bit_off_reg` from the buffer pointed to by `base_reg`, into the low
/// `n_bits` of `dst_reg` (zero-extended).
///
/// # Register usage
///
/// - `base_reg`: 64-bit GP holding the buffer base pointer. **Not
///   modified.**
/// - `bit_off_reg`: 64-bit GP holding the bit offset. **Not modified.**
/// - `dst_reg`: 64-bit GP receiving the loaded bits. **Written.**
/// - `scratch_reg`: 64-bit GP used as temporary storage. **Clobbered.**
/// - `rcx`: **fully clobbered** — used as the shift count register
///   and as a temporary byte-offset holder.
///
/// All four register parameters and `rcx` must be pairwise distinct.
///
/// # Constraints
///
/// - `n_bits` must be in `1..=64`.
///
/// # Buffer slack
///
/// The emitted code reads up to 16 bytes starting at
/// `[base_reg + bit_off_reg/8]`. Callers must ensure the buffer has
/// at least 8 bytes of slack after the highest addressable bit.
pub fn emit_load_bits(
    asm: &mut Assembler,
    base_reg: u8,
    bit_off_reg: u8,
    n_bits: u32,
    dst_reg: u8,
    scratch_reg: u8,
) {
    assert!((1..=64).contains(&n_bits), "n_bits must be 1..=64");
    let regs = [base_reg, bit_off_reg, dst_reg, scratch_reg, RCX];
    assert_distinct("emit_load_bits", &regs);

    // rcx = byte_offset; load chunk0 into dst.
    dynasm!(asm
        ; .arch x64
        ; mov rcx, Rq(bit_off_reg)
        ; shr rcx, 3
        ; mov Rq(dst_reg), QWORD [Rq(base_reg) + rcx]
    );

    if n_bits > 57 {
        // Spanning case: load chunk1 into scratch_reg before we
        // clobber rcx with the shift count.
        dynasm!(asm
            ; mov Rq(scratch_reg), QWORD [Rq(base_reg) + rcx + 8]
        );
    }

    // Now overwrite rcx (cl) with the shift count. The high bits of
    // rcx are scratch from this point on; the byte offset is no longer
    // recoverable from rcx.
    dynasm!(asm
        ; mov cl, Rb(bit_off_reg)
        ; and cl, 7
    );

    if n_bits > 57 {
        // shrd dst, scratch, cl: shift dst right by cl, filling the
        // top of dst from scratch's low bits. Result: bits
        // [bit_in_byte..bit_in_byte+64] from {scratch:dst} → dst.
        dynasm!(asm
            ; shrd Rq(dst_reg), Rq(scratch_reg), cl
        );
    } else {
        dynasm!(asm
            ; shr Rq(dst_reg), cl
        );
    }

    // Mask to n_bits if not the full 64.
    if n_bits < 64 {
        let mask: u64 = (1u64 << n_bits) - 1;
        if mask <= u32::MAX as u64 {
            // 32-bit AND: writing the 32-bit reg auto-clears the high
            // half of the 64-bit reg, which is exactly what we want.
            dynasm!(asm
                ; and Rd(dst_reg), DWORD mask as i32
            );
        } else {
            // 33..63 bit mask: load via mov + and (mov r64, imm64).
            dynasm!(asm
                ; mov Rq(scratch_reg), QWORD mask as i64
                ; and Rq(dst_reg), Rq(scratch_reg)
            );
        }
    }
}

/// Emit code that stores the low `n_bits` of `src_reg` at the bit
/// offset held in `bit_off_reg` from the buffer pointed to by
/// `base_reg`. Bits outside the `[bit_off, bit_off + n_bits)` range
/// in the touched bytes are preserved (read-modify-write).
///
/// # Register usage
///
/// - `base_reg`: 64-bit GP holding the buffer base pointer. **Not
///   modified.**
/// - `bit_off_reg`: 64-bit GP holding the bit offset. **Not modified.**
/// - `src_reg`: 64-bit GP holding the bits to write in its low
///   `n_bits` positions. The bits above `n_bits` MUST be zero —
///   callers are responsible for masking. **Not modified.**
/// - `tmp1_reg`, `tmp2_reg`, `tmp3_reg`: 64-bit GP scratch slots.
///   **Clobbered.**
/// - `rcx`: **fully clobbered** — used as the shift count register and
///   as a temporary byte-offset holder.
///
/// All six register parameters and `rcx` must be pairwise distinct.
///
/// # Constraints
///
/// - `n_bits` must be in `1..=64`.
/// - `src_reg`'s high `(64 - n_bits)` bits must be zero. The
///   bit_io layer trusts the format layer to honour this.
///
/// # Buffer slack
///
/// Same contract as [`emit_load_bits`] — up to 16 bytes are read and
/// written starting at `[base_reg + bit_off_reg/8]`.
pub fn emit_store_bits(
    asm: &mut Assembler,
    base_reg: u8,
    bit_off_reg: u8,
    n_bits: u32,
    src_reg: u8,
    tmp1_reg: u8,
    tmp2_reg: u8,
    tmp3_reg: u8,
) {
    assert!((1..=64).contains(&n_bits), "n_bits must be 1..=64");
    let regs = [
        base_reg,
        bit_off_reg,
        src_reg,
        tmp1_reg,
        tmp2_reg,
        tmp3_reg,
        RCX,
    ];
    assert_distinct("emit_store_bits", &regs);

    let needs_chunk1 = n_bits > 57;
    let raw_mask: u64 = if n_bits == 64 {
        u64::MAX
    } else {
        (1u64 << n_bits) - 1
    };

    // -- Phase 1: load chunk0 (and chunk1 if spanning) using rcx as
    // the byte-offset register. After this, we never use rcx as a
    // byte offset again — we recompute it before each store-back.
    dynasm!(asm
        ; .arch x64
        ; mov rcx, Rq(bit_off_reg)
        ; shr rcx, 3
        ; mov Rq(tmp1_reg), QWORD [Rq(base_reg) + rcx]
    );
    if needs_chunk1 {
        dynasm!(asm
            ; mov Rq(tmp2_reg), QWORD [Rq(base_reg) + rcx + 8]
        );
    }

    // -- Phase 2: cl = bit_in_byte. rcx high bits are now scratch.
    dynasm!(asm
        ; mov cl, Rb(bit_off_reg)
        ; and cl, 7
    );

    // -- Phase 3: build the chunk0 mask in tmp3, clear those bits in
    // tmp1 (chunk0), then OR in (src << cl).
    emit_load_imm64(asm, tmp3_reg, raw_mask);
    dynasm!(asm
        ; shl Rq(tmp3_reg), cl
        ; not Rq(tmp3_reg)
        ; and Rq(tmp1_reg), Rq(tmp3_reg)
        ; mov Rq(tmp3_reg), Rq(src_reg)
        ; shl Rq(tmp3_reg), cl
        ; or Rq(tmp1_reg), Rq(tmp3_reg)
    );

    // -- Phase 4: recompute byte_offset in rcx (clobbering cl) and
    // write chunk0 back. We must do this before phase 5, because
    // phase 5 also overwrites rcx.
    if needs_chunk1 {
        // We still need cl for phase 5; save bit_in_byte in tmp3
        // before recomputing rcx.
        dynasm!(asm
            ; movzx Rd(tmp3_reg), cl
        );
    }
    dynasm!(asm
        ; mov rcx, Rq(bit_off_reg)
        ; shr rcx, 3
        ; mov QWORD [Rq(base_reg) + rcx], Rq(tmp1_reg)
    );

    // -- Phase 5: chunk1 write (spanning case only).
    //
    // We need to:
    //   chunk1 := (chunk1 & ~mask_hi) | (src >> (64 - cl))
    // where mask_hi = raw_mask >> (64 - cl), with the special case
    // cl == 0 → mask_hi == 0 (nothing to write).
    //
    // SHLD(dst, src, cl) computes
    //   dst' = (dst << cl) | (src >> (64 - cl))
    // For cl == 0 (the no-spillover case), Intel's SHLD is documented
    // as a no-op: the destination and flags are unchanged. So
    // SHLD(0, raw_mask, cl) → 0 when cl == 0, else raw_mask >> (64-cl).
    if needs_chunk1 {
        // Restore cl from tmp3 (saved above).
        dynasm!(asm
            ; mov cl, Rb(tmp3_reg)
            // tmp3 = raw_mask
            ; mov Rq(tmp3_reg), QWORD raw_mask as i64
            // tmp1 = 0; SHLD(tmp1, tmp3, cl) → tmp1 = mask_hi
            ; xor Rq(tmp1_reg), Rq(tmp1_reg)
            ; shld Rq(tmp1_reg), Rq(tmp3_reg), cl
            // Clear those bits in chunk1
            ; not Rq(tmp1_reg)
            ; and Rq(tmp2_reg), Rq(tmp1_reg)
            // tmp3 = 0; SHLD(tmp3, src, cl) → tmp3 = src >> (64 - cl)
            ; xor Rq(tmp3_reg), Rq(tmp3_reg)
            ; shld Rq(tmp3_reg), Rq(src_reg), cl
            ; or Rq(tmp2_reg), Rq(tmp3_reg)
            // Recompute byte_offset and write chunk1 back.
            ; mov rcx, Rq(bit_off_reg)
            ; shr rcx, 3
            ; mov QWORD [Rq(base_reg) + rcx + 8], Rq(tmp2_reg)
        );
    }
}

// ─── Byte-aligned fast path ─────────────────────────────────────────
//
// When the address layer reports `byte_aligned == true`, the offset
// register holds a **byte** offset and `n_bits` is 8, 16, 32, or 64.
// These functions emit a single `mov` instead of the general bit
// extraction / read-modify-write sequence.

/// Load `n_bits` bits from byte-aligned `[base_reg + byte_off_reg]`
/// into the low `n_bits` of `dst_reg` (zero-extended to 64 bits).
///
/// `n_bits` must be 8, 16, 32, or 64. Does NOT clobber `rcx`.
pub fn emit_load_aligned(
    asm: &mut Assembler,
    base_reg: u8,
    byte_off_reg: u8,
    n_bits: u32,
    dst_reg: u8,
) {
    match n_bits {
        8 => dynasm!(asm; .arch x64
            ; movzx Rd(dst_reg), BYTE [Rq(base_reg) + Rq(byte_off_reg)]
        ),
        16 => dynasm!(asm; .arch x64
            ; movzx Rd(dst_reg), WORD [Rq(base_reg) + Rq(byte_off_reg)]
        ),
        32 => dynasm!(asm; .arch x64
            ; mov Rd(dst_reg), DWORD [Rq(base_reg) + Rq(byte_off_reg)]
        ),
        64 => dynasm!(asm; .arch x64
            ; mov Rq(dst_reg), QWORD [Rq(base_reg) + Rq(byte_off_reg)]
        ),
        _ => panic!("emit_load_aligned: n_bits={n_bits} not in {{8,16,32,64}}"),
    }
}

/// Store the low `n_bits` of `src_reg` to byte-aligned
/// `[base_reg + byte_off_reg]`.
///
/// `n_bits` must be 8, 16, 32, or 64. Does NOT clobber `rcx` or
/// any scratch registers.
pub fn emit_store_aligned(
    asm: &mut Assembler,
    base_reg: u8,
    byte_off_reg: u8,
    n_bits: u32,
    src_reg: u8,
) {
    match n_bits {
        8 => dynasm!(asm; .arch x64
            ; mov BYTE [Rq(base_reg) + Rq(byte_off_reg)], Rb(src_reg)
        ),
        16 => dynasm!(asm; .arch x64
            ; mov WORD [Rq(base_reg) + Rq(byte_off_reg)], Rw(src_reg)
        ),
        32 => dynasm!(asm; .arch x64
            ; mov DWORD [Rq(base_reg) + Rq(byte_off_reg)], Rd(src_reg)
        ),
        64 => dynasm!(asm; .arch x64
            ; mov QWORD [Rq(base_reg) + Rq(byte_off_reg)], Rq(src_reg)
        ),
        _ => panic!("emit_store_aligned: n_bits={n_bits} not in {{8,16,32,64}}"),
    }
}

/// Reserved register code for `rcx` — the x86 shift count register.
const RCX: u8 = 1;

fn assert_distinct(label: &str, regs: &[u8]) {
    for i in 0..regs.len() {
        for j in (i + 1)..regs.len() {
            assert_ne!(
                regs[i], regs[j],
                "{label}: register {} and {} alias",
                regs[i], regs[j]
            );
        }
    }
}

/// Emit `mov reg, imm64`. Optimizes for the common cases where the
/// value fits in a smaller form (xor for zero, mov r32 imm32 for any
/// 32-bit value).
fn emit_load_imm64(asm: &mut Assembler, reg: u8, value: u64) {
    if value == 0 {
        dynasm!(asm
            ; .arch x64
            ; xor Rq(reg), Rq(reg)
        );
    } else if value <= u32::MAX as u64 {
        dynasm!(asm
            ; .arch x64
            ; mov Rd(reg), DWORD value as i32
        );
    } else {
        dynasm!(asm
            ; .arch x64
            ; mov Rq(reg), QWORD value as i64
        );
    }
}
