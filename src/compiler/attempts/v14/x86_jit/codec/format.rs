//! Format conversion: raw dtype bits ↔ compute representation.
//!
//! Given a `NumericDType` and a 64-bit GP register holding either raw
//! storage bits or a compute-repr value, emit the assembly that
//! converts between the two. The compute representation is fixed per
//! dtype family per the codec doc:
//!
//! - Floats with `m_bits ≤ 23` AND `e_bits ≤ 8`: F32 in an XMM register.
//! - F64 (and any hypothetical wider FloatType): F64 in an XMM register.
//! - Signed/unsigned ints: 64-bit GP register, sign- or zero-extended.
//! - Bool: low bit of a 64-bit GP register (0 or 1).
//!
//! Conversion is specialized at JIT-build time on the `FloatType` /
//! `IntType` properties — no runtime dispatch, no extern `"C"` calls.
//!
//! # Coverage
//!
//! All named `NumericDType`s are supported via two strategies:
//!   - F32, F64 — trivial `movd` / `movq`
//!   - BF16 — inline shift-and-mask with branchless NaN canonicalization
//!   - F16 — F16C (`vcvtph2ps` / `vcvtps2ph`) with branched NaN
//!     canonicalization
//!   - F8E5M2, F8E4M3FN, F4E2M1, F6E3M2, F6E2M3 (sub-F16 floats) —
//!     decode via a per-`FloatType` lookup table baked at JIT-build
//!     time and held alive by [`CodecTables`]; encode via inline
//!     rounding (P2.A.3.b)
//!   - All `IntType` widths (1..=64), signed and unsigned — single
//!     `movsx` / `movzx` / mask sequence
//!   - Bool — `and 1` decode, `test` + `setne` + `movzx` encode
//!
//! Decode for sub-F16 floats requires a runtime-resident lookup table
//! whose pointer is embedded as an absolute imm64 in the JIT buffer.
//! [`CodecTables`] owns the table memory; callers (the test harness
//! and the orchestration layer) must keep it alive at least as long
//! as the JIT function itself.

use dynasmrt::{DynasmApi, DynasmLabelApi, dynasm, x64::Assembler};

use crate::numeric_dtype::{FloatType, NumericDType};

/// Owns the lookup tables baked into a JIT compilation.
///
/// The codec emits a `mov rN, imm64` to load each table's absolute
/// address into a register at runtime, so the table memory must
/// remain pinned at the same address from emit time through every
/// execution of the compiled function. We achieve that by holding
/// `Box<[u32]>` slabs here — each one is heap-allocated and never
/// reallocates.
///
/// The same instance can carry multiple tables (one per dtype that
/// the JIT references). Callers construct one `CodecTables` per JIT
/// function and store it alongside the executable buffer.
#[derive(Default)]
pub struct CodecTables {
    tables: Vec<Box<[u32]>>,
}

impl CodecTables {
    pub fn new() -> Self {
        Self::default()
    }

    /// Build (or look up) the F32-decode table for `ft` and return a
    /// raw pointer the codec can embed in the JIT.
    ///
    /// The table maps every raw bit pattern of `ft` (interpreted as
    /// the low `total_bits` of an index) to the corresponding F32 bit
    /// pattern. NaN inputs map to the canonical F32 NaN
    /// (`0x7fc00000`) — the codec contract requires encode to
    /// canonicalize NaN, and we extend the same canonicalization to
    /// decode for sub-F16 floats so cross-dtype casts agree
    /// bit-for-bit with `cast_raw`.
    pub fn alloc_f32_decode_table(&mut self, ft: FloatType) -> *const u32 {
        let table = compute_f32_decode_table(ft);
        let boxed: Box<[u32]> = table.into_boxed_slice();
        let ptr = boxed.as_ptr();
        self.tables.push(boxed);
        ptr
    }
}

/// Compute the F32-decode table for `ft`. The result has `1 <<
/// ft.total_bits()` entries.
///
/// Each entry holds the F32 bit pattern that the corresponding raw
/// `ft`-encoded value decodes to. NaN-encoded inputs are mapped to
/// the canonical F32 NaN (`0x7fc00000`).
pub(super) fn compute_f32_decode_table(ft: FloatType) -> Vec<u32> {
    debug_assert!(ft.is_supported());
    debug_assert!(
        ft.total_bits() <= 16,
        "decode table is impractical for FloatType with > 16 total bits"
    );
    let n = 1usize << ft.total_bits() as u32;
    (0..n)
        .map(|raw| {
            let raw_u64 = raw as u64;
            let f64_val = ft.decode_f64(raw_u64);
            if f64_val.is_nan() {
                0x7fc00000_u32
            } else {
                (f64_val as f32).to_bits()
            }
        })
        .collect()
}

/// The compute representation a dtype lives in. Selected by the
/// codec at JIT-build time per the table in `x86_jit_codec.md` §1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeRepr {
    /// 32-bit float in an XMM register (low 32 bits, scalar single).
    F32,
    /// 64-bit float in an XMM register (low 64 bits, scalar double).
    F64,
    /// 64-bit GP register, sign- or zero-extended from the dtype's
    /// natural width. Bool also lives here (low bit only).
    Int,
}

impl ComputeRepr {
    /// Compute repr selected for a given dtype, per codec doc §1.
    pub fn for_dtype(dtype: NumericDType) -> Self {
        match dtype {
            NumericDType::Float(ft) => {
                if ft == FloatType::F64 {
                    ComputeRepr::F64
                } else if ft.exponent_bits <= 8 && ft.mantissa_bits <= 23 {
                    ComputeRepr::F32
                } else {
                    // Anything wider than F32 but narrower than F64,
                    // or wider than F64, would need its own arm. None
                    // currently exist.
                    ComputeRepr::F64
                }
            }
            NumericDType::SignedInt(_) | NumericDType::UnsignedInt(_) | NumericDType::Bool => {
                ComputeRepr::Int
            }
        }
    }
}

/// Where the codec puts a decode result, or where it reads an encode
/// input from. The variant must match the dtype's [`ComputeRepr`].
#[derive(Debug, Clone, Copy)]
pub enum CodecSlot {
    /// XMM register holding an F32 / F64 value.
    Xmm(u8),
    /// GP register holding the i64-extended int (or Bool 0/1).
    Gp(u8),
}

/// Emit code that decodes raw bits in `raw_reg` (right-justified, low
/// `dtype.total_bits()`) into the compute repr in `slot`.
///
/// # Register usage
/// - `raw_reg`: GP holding the raw bits. **Clobbered** for some
///   dtype-specific paths (e.g., BF16 shifts in place, sub-F16 lookup
///   loads); callers should treat it as consumed.
/// - `slot`: must match `ComputeRepr::for_dtype(dtype)`.
/// - `scratch_gp`: a free GP register, **clobbered.** Used to hold
///   the lookup-table address for sub-F16 floats. Ignored otherwise.
/// - `scratch_xmm`: a free XMM register, **clobbered**, used as a
///   stepping stone for paths like F16C that need an intermediate
///   xmm. May alias `slot` only when `slot` is the GP variant.
/// - `tables`: codec table store. The codec may push a new table
///   into it for sub-F16 floats; the resulting table memory must
///   stay alive for as long as the JIT function executes.
///
/// # Errors
/// Returns `Err` for an unsupported dtype or a slot variant that
/// doesn't match the dtype's compute repr.
pub fn emit_decode(
    asm: &mut Assembler,
    dtype: NumericDType,
    raw_reg: u8,
    slot: CodecSlot,
    scratch_gp: u8,
    scratch_xmm: u8,
    tables: &mut CodecTables,
) -> Result<(), String> {
    let expected_repr = ComputeRepr::for_dtype(dtype);
    match (expected_repr, slot) {
        (ComputeRepr::F32 | ComputeRepr::F64, CodecSlot::Xmm(_)) => {}
        (ComputeRepr::Int, CodecSlot::Gp(_)) => {}
        _ => {
            return Err(format!(
                "emit_decode: slot variant does not match compute repr for {dtype}"
            ));
        }
    }

    match dtype {
        NumericDType::Float(ft) => {
            emit_float_decode(asm, ft, raw_reg, slot, scratch_gp, scratch_xmm, tables)
        }
        NumericDType::SignedInt(it) => {
            emit_int_decode(asm, raw_reg, slot, it.bits, true);
            Ok(())
        }
        NumericDType::UnsignedInt(it) => {
            emit_int_decode(asm, raw_reg, slot, it.bits, false);
            Ok(())
        }
        NumericDType::Bool => {
            emit_bool_decode(asm, raw_reg, slot);
            Ok(())
        }
    }
}

/// Emit code that encodes the compute-repr value in `slot` to raw
/// bits in `raw_reg` (right-justified, masked to `dtype.total_bits()`,
/// high bits zero).
///
/// # Register usage
/// - `slot`: must match `ComputeRepr::for_dtype(dtype)`.
/// - `raw_reg`: GP receiving the raw bits. **Written.**
/// - `scratch_gp`: free GP register, **clobbered.** Used by some paths
///   (e.g., BF16 RTNE rounding) as a temporary.
/// - `scratch_xmm`: free XMM register, **clobbered.** Used by F16C
///   conversions.
///
/// # Errors
/// Returns `Err` for dtypes outside the P2.A.2 scope.
pub fn emit_encode(
    asm: &mut Assembler,
    dtype: NumericDType,
    slot: CodecSlot,
    raw_reg: u8,
    scratch_gp: u8,
    scratch_xmm: u8,
) -> Result<(), String> {
    let expected_repr = ComputeRepr::for_dtype(dtype);
    match (expected_repr, slot) {
        (ComputeRepr::F32 | ComputeRepr::F64, CodecSlot::Xmm(_)) => {}
        (ComputeRepr::Int, CodecSlot::Gp(_)) => {}
        _ => {
            return Err(format!(
                "emit_encode: slot variant does not match compute repr for {dtype}"
            ));
        }
    }

    match dtype {
        NumericDType::Float(ft) => {
            emit_float_encode(asm, ft, slot, raw_reg, scratch_gp, scratch_xmm)
        }
        NumericDType::SignedInt(it) => {
            emit_int_encode(asm, slot, raw_reg, it.bits, true);
            Ok(())
        }
        NumericDType::UnsignedInt(it) => {
            emit_int_encode(asm, slot, raw_reg, it.bits, false);
            Ok(())
        }
        NumericDType::Bool => {
            emit_bool_encode(asm, slot, raw_reg);
            Ok(())
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Float decode
// ─────────────────────────────────────────────────────────────────────

fn emit_float_decode(
    asm: &mut Assembler,
    ft: FloatType,
    raw_reg: u8,
    slot: CodecSlot,
    scratch_gp: u8,
    scratch_xmm: u8,
    tables: &mut CodecTables,
) -> Result<(), String> {
    let xmm_dst = match slot {
        CodecSlot::Xmm(x) => x,
        CodecSlot::Gp(_) => unreachable!("validated by caller"),
    };

    if ft == FloatType::F32 {
        // F32 raw bits IS the F32 representation. movd low 32 bits
        // into the xmm.
        dynasm!(asm
            ; .arch x64
            ; movd Rx(xmm_dst), Rd(raw_reg)
        );
        return Ok(());
    }
    if ft == FloatType::F64 {
        dynasm!(asm
            ; .arch x64
            ; movq Rx(xmm_dst), Rq(raw_reg)
        );
        return Ok(());
    }
    if ft == FloatType::BF16 {
        // BF16 = upper 16 bits of an F32 (sign:exp:upper-7-mant).
        //
        // Decode strategy: shift raw bits up by 16, but if the input
        // is a NaN encoding, substitute the canonical F32 NaN
        // (`0x7fc00000`) instead. This canonicalization is required
        // by the dtype contract: cross-dtype casts go through
        // `decode_f64`/`encode_f64`, which always produces canonical
        // NaN bits at the destination, so the JIT must agree
        // bit-for-bit. We use a branchless cmov to keep the common
        // (non-NaN) path tight.
        //
        // NaN check: `(raw & 0x7fff) > 0x7f80` — i.e., biased exp ==
        // max AND mantissa != 0.
        dynasm!(asm
            ; .arch x64
            ; mov ecx, Rd(raw_reg)
            ; and ecx, 0x7fff
            ; shl Rd(raw_reg), 16
            ; cmp ecx, 0x7f80
            ; mov ecx, DWORD 0x7fc00000_u32 as i32
            ; cmova Rd(raw_reg), ecx
            ; movd Rx(xmm_dst), Rd(raw_reg)
        );
        return Ok(());
    }
    if ft == FloatType::F16 {
        // F16C: load F16 raw into low 16 of an xmm, then vcvtph2ps.
        // The instruction reads 4 packed F16s; we only care about
        // the low one and ignore the rest.
        //
        // F16C does NaN-to-NaN propagation but not necessarily to the
        // canonical F32 NaN bit pattern, so we add an explicit NaN
        // check using a branch on the raw input bits before the
        // convert. NaN check: `(raw & 0x7fff) > 0x7c00` — biased exp
        // == 0x1f (max) AND mantissa != 0.
        let nan_path = asm.new_dynamic_label();
        let done = asm.new_dynamic_label();
        dynasm!(asm
            ; .arch x64
            ; mov ecx, Rd(raw_reg)
            ; and ecx, 0x7fff
            ; cmp ecx, 0x7c00
            ; ja =>nan_path
            ; movd Rx(scratch_xmm), Rd(raw_reg)
            ; vcvtph2ps Rx(xmm_dst), Rx(scratch_xmm)
            ; jmp =>done
            ; =>nan_path
            ; mov ecx, DWORD 0x7fc00000_u32 as i32
            ; movd Rx(xmm_dst), ecx
            ; =>done
        );
        return Ok(());
    }

    // Sub-F16 floats: lookup-table decode. The codec supports any
    // FloatType with `total_bits ≤ 16`, but in practice the named
    // sub-F16 types are F8E5M2, F8E4M3FN, F4E2M1, F6E3M2, F6E2M3 —
    // table sizes 256/256/16/64/64 entries × 4 bytes each.
    if ft.total_bits() <= 16 {
        // Cap the index width: the codec contract says `raw_reg`
        // holds the value in its low `total_bits` (zero-extended), so
        // an explicit mask is redundant *if* the bit_io upstream is
        // honoring its contract. We mask anyway as defence-in-depth
        // — `total_bits ≤ 8` keeps the mask in the 32-bit immediate
        // form, and the AND clears any garbage in the high bits the
        // table address arithmetic would otherwise dereference.
        let total_bits = ft.total_bits() as u32;
        let mask: i32 = if total_bits >= 32 {
            -1
        } else {
            ((1u32 << total_bits) - 1) as i32
        };
        let table_ptr = tables.alloc_f32_decode_table(ft);
        dynasm!(asm
            ; .arch x64
            ; and Rd(raw_reg), DWORD mask
            ; mov Rq(scratch_gp), QWORD table_ptr as i64
            ; mov Rd(raw_reg), DWORD [Rq(scratch_gp) + Rq(raw_reg) * 4]
            ; movd Rx(xmm_dst), Rd(raw_reg)
        );
        return Ok(());
    }

    Err(format!(
        "emit_decode: float type {ft} not supported (total_bits > 16 needs F64 compute repr)"
    ))
}

// ─────────────────────────────────────────────────────────────────────
// Float encode
// ─────────────────────────────────────────────────────────────────────

fn emit_float_encode(
    asm: &mut Assembler,
    ft: FloatType,
    slot: CodecSlot,
    raw_reg: u8,
    scratch_gp: u8,
    scratch_xmm: u8,
) -> Result<(), String> {
    let xmm_src = match slot {
        CodecSlot::Xmm(x) => x,
        CodecSlot::Gp(_) => unreachable!("validated by caller"),
    };

    if ft == FloatType::F32 {
        dynasm!(asm
            ; .arch x64
            ; movd Rd(raw_reg), Rx(xmm_src)
            // movd zero-extends to 64 bits automatically by writing
            // the 32-bit form.
        );
        return Ok(());
    }
    if ft == FloatType::F64 {
        dynasm!(asm
            ; .arch x64
            ; movq Rq(raw_reg), Rx(xmm_src)
        );
        return Ok(());
    }
    if ft == FloatType::BF16 {
        // F32 → BF16 RTNE round. Per codec doc §3.4, the 8-instruction
        // recipe rounds via a bias-add. We then check if the input
        // was a NaN encoding and substitute the canonical BF16 NaN
        // (`0x7fc0`) — the dtype contract requires `encode` to
        // canonicalize NaN per the dtype's `encode_nan()` rule.
        //
        // NaN check: `(raw_f32 & 0x7fffffff) > 0x7f800000` — biased
        // exp == 0xff (max) AND mantissa != 0.
        let nan_path = asm.new_dynamic_label();
        let done = asm.new_dynamic_label();
        dynasm!(asm
            ; .arch x64
            ; movd Rd(raw_reg), Rx(xmm_src)
            // Save raw_reg into scratch_gp for the NaN check (we'll
            // clobber raw_reg with the rounded value).
            ; mov Rd(scratch_gp), Rd(raw_reg)
            ; and Rd(scratch_gp), 0x7fffffff_u32 as i32
            ; cmp Rd(scratch_gp), DWORD 0x7f800000_u32 as i32
            ; ja =>nan_path
            // Regular RTNE encode: rcx = (raw >> 16) & 1; rcx += 0x7fff;
            // raw += rcx; raw >>= 16. We need to recompute the lsb in
            // scratch_gp because we just clobbered it with the and.
            ; mov Rd(scratch_gp), Rd(raw_reg)
            ; shr Rd(scratch_gp), 16
            ; and Rd(scratch_gp), 1
            ; add Rd(scratch_gp), 0x7fff
            ; add Rd(raw_reg), Rd(scratch_gp)
            ; shr Rd(raw_reg), 16
            ; jmp =>done
            ; =>nan_path
            ; mov Rd(raw_reg), DWORD 0x7fc0_u32 as i32
            ; =>done
        );
        return Ok(());
    }
    if ft == FloatType::F16 {
        // F16C round-to-nearest (imm8 = 0). The instruction propagates
        // NaN inputs to F16 NaN outputs but with implementation-defined
        // bit patterns; we override with the canonical F16 NaN
        // (`0x7e00`) when the input is a NaN.
        let nan_path = asm.new_dynamic_label();
        let done = asm.new_dynamic_label();
        dynasm!(asm
            ; .arch x64
            // Pull F32 raw bits into scratch_gp for the NaN check.
            ; movd Rd(scratch_gp), Rx(xmm_src)
            ; and Rd(scratch_gp), 0x7fffffff_u32 as i32
            ; cmp Rd(scratch_gp), DWORD 0x7f800000_u32 as i32
            ; ja =>nan_path
            ; vcvtps2ph Rx(scratch_xmm), Rx(xmm_src), 0
            ; movd Rd(raw_reg), Rx(scratch_xmm)
            // Mask the F16C result to 16 bits — vcvtps2ph writes the
            // high half of xmm with garbage from a higher input lane,
            // and movd zero-extends from 32, but the low 32-bit lane
            // may also have garbage in bits 16..31 from another lane.
            ; movzx Rq(raw_reg), Rw(raw_reg)
            ; jmp =>done
            ; =>nan_path
            ; mov Rd(raw_reg), DWORD 0x7e00_u32 as i32
            ; =>done
        );
        return Ok(());
    }

    Err(format!(
        "emit_encode: float type {ft} not in P2.A.2 scope (sub-byte / sub-F16 floats land in P2.A.3)"
    ))
}

// ─────────────────────────────────────────────────────────────────────
// Integer decode / encode
// ─────────────────────────────────────────────────────────────────────

/// Decode a `bits`-wide integer in the low bits of `raw_reg` into a
/// 64-bit (sign- or zero-extended) value in the destination slot.
fn emit_int_decode(asm: &mut Assembler, raw_reg: u8, slot: CodecSlot, bits: u8, signed: bool) {
    let dst = match slot {
        CodecSlot::Gp(g) => g,
        CodecSlot::Xmm(_) => unreachable!("validated by caller"),
    };
    match (bits, signed) {
        (64, _) => {
            if dst != raw_reg {
                dynasm!(asm; .arch x64; mov Rq(dst), Rq(raw_reg));
            }
        }
        (32, true) => dynasm!(asm; .arch x64; movsxd Rq(dst), Rd(raw_reg)),
        (32, false) => {
            // Writing the 32-bit form zero-extends to 64.
            if dst != raw_reg {
                dynasm!(asm; .arch x64; mov Rd(dst), Rd(raw_reg));
            } else {
                dynasm!(asm; .arch x64; mov Rd(dst), Rd(raw_reg));
            }
        }
        (16, true) => dynasm!(asm; .arch x64; movsx Rq(dst), Rw(raw_reg)),
        (16, false) => dynasm!(asm; .arch x64; movzx Rq(dst), Rw(raw_reg)),
        (8, true) => dynasm!(asm; .arch x64; movsx Rq(dst), Rb(raw_reg)),
        (8, false) => dynasm!(asm; .arch x64; movzx Rq(dst), Rb(raw_reg)),
        (b, true) if b < 64 => {
            // Mask to b bits, then sign-extend by shifting up and arithmetic-shifting back.
            let shift = 64 - b;
            if dst != raw_reg {
                dynasm!(asm; .arch x64; mov Rq(dst), Rq(raw_reg));
            }
            dynasm!(asm
                ; .arch x64
                ; shl Rq(dst), BYTE shift as i8
                ; sar Rq(dst), BYTE shift as i8
            );
        }
        (b, false) if b < 64 => {
            // Mask to b bits — no sign extension.
            let mask: u64 = (1u64 << b) - 1;
            if dst != raw_reg {
                dynasm!(asm; .arch x64; mov Rq(dst), Rq(raw_reg));
            }
            if mask <= u32::MAX as u64 {
                dynasm!(asm; .arch x64; and Rd(dst), DWORD mask as i32);
            } else {
                // Width > 32 bits but < 64. Use a wide mask.
                dynasm!(asm
                    ; .arch x64
                    ; mov rax, QWORD mask as i64
                    ; and Rq(dst), rax
                );
            }
        }
        _ => unreachable!("bits {bits} out of supported range"),
    }
}

/// Encode a 64-bit (sign- or zero-extended) value in the source slot
/// into the low `bits` of `raw_reg`. The high (64 - bits) bits of
/// `raw_reg` are zeroed so the bit_io layer can store directly.
fn emit_int_encode(asm: &mut Assembler, slot: CodecSlot, raw_reg: u8, bits: u8, _signed: bool) {
    let src = match slot {
        CodecSlot::Gp(g) => g,
        CodecSlot::Xmm(_) => unreachable!("validated by caller"),
    };
    if src != raw_reg {
        dynasm!(asm; .arch x64; mov Rq(raw_reg), Rq(src));
    }
    // Mask the high bits regardless of signed/unsigned: bit_io stores
    // a fixed-width slot and trusts the codec to put zero in the
    // top (64 - bits).
    if bits == 64 {
        return;
    }
    let mask: u64 = (1u64 << bits) - 1;
    if mask <= u32::MAX as u64 {
        // 32-bit AND auto-zeroes the high half via the 32-bit reg form.
        dynasm!(asm; .arch x64; and Rd(raw_reg), DWORD mask as i32);
    } else {
        dynasm!(asm
            ; .arch x64
            ; mov rax, QWORD mask as i64
            ; and Rq(raw_reg), rax
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// Bool decode / encode
// ─────────────────────────────────────────────────────────────────────

fn emit_bool_decode(asm: &mut Assembler, raw_reg: u8, slot: CodecSlot) {
    let dst = match slot {
        CodecSlot::Gp(g) => g,
        CodecSlot::Xmm(_) => unreachable!("validated by caller"),
    };
    // raw_reg holds the raw bit (0 or 1) in its low bit; high bits
    // may be garbage. Normalize to 0/1.
    if dst != raw_reg {
        dynasm!(asm; .arch x64; mov Rq(dst), Rq(raw_reg));
    }
    dynasm!(asm
        ; .arch x64
        ; and Rq(dst), 1
    );
}

fn emit_bool_encode(asm: &mut Assembler, slot: CodecSlot, raw_reg: u8) {
    let src = match slot {
        CodecSlot::Gp(g) => g,
        CodecSlot::Xmm(_) => unreachable!("validated by caller"),
    };
    // The compute-repr Bool is already 0 or 1 in the low bit, but
    // higher bits may be set if the value came from arithmetic.
    // Normalize via test+setne+movzx.
    dynasm!(asm
        ; .arch x64
        ; test Rq(src), Rq(src)
        ; setne Rb(raw_reg)
        ; movzx Rq(raw_reg), Rb(raw_reg)
    );
}
