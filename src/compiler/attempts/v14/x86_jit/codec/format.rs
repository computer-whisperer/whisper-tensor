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
/// - `scratch_gp1`, `scratch_gp2`: free GP registers, **clobbered.**
///   `scratch_gp1` alone is sufficient for native and BF16/F16 paths.
///   The sub-F16 inline encode (P2.A.3.b) needs both. Callers that
///   only emit native dtypes can pass any caller-saved GP for
///   `scratch_gp2`.
/// - `scratch_xmm`: free XMM register, **clobbered.** Used by F16C
///   conversions.
///
/// `rcx` and `rdx` are implicitly clobbered: `rcx` is the shift count
/// register for variable-shift sequences, and `rdx` is used as a
/// hardcoded scratch by the sub-F16 inline encode for the round-bias
/// computation. All caller-supplied registers must be pairwise
/// distinct from `rcx`, `rdx`, and from each other.
///
/// # Errors
/// Returns `Err` for an unsupported dtype or a mismatched slot variant.
pub fn emit_encode(
    asm: &mut Assembler,
    dtype: NumericDType,
    slot: CodecSlot,
    raw_reg: u8,
    scratch_gp1: u8,
    scratch_gp2: u8,
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
        NumericDType::Float(ft) => emit_float_encode(
            asm,
            ft,
            slot,
            raw_reg,
            scratch_gp1,
            scratch_gp2,
            scratch_xmm,
        ),
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
    scratch_gp2: u8,
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

    // Sub-F16 floats: inline encode parameterized on FloatType
    // properties. P2.A.3.b — handles every named sub-F16 dtype
    // (F8E5M2, F8E4M3FN, F4E2M1, F6E3M2, F6E2M3) and any custom
    // FloatType with `total_bits ≤ 16` AND `mantissa_bits ≤ 23` AND
    // `exponent_bits ≤ 8`.
    if ft.total_bits() <= 16 && ft.mantissa_bits <= 23 && ft.exponent_bits <= 8 {
        emit_subf16_float_encode(asm, ft, xmm_src, raw_reg, scratch_gp, scratch_gp2);
        return Ok(());
    }

    Err(format!(
        "emit_encode: float type {ft} not supported (compute repr selection mismatch)"
    ))
}

// ─────────────────────────────────────────────────────────────────────
// Sub-F16 float inline encode
// ─────────────────────────────────────────────────────────────────────

/// Inline F32 → arbitrary small-FloatType encode, parameterized on
/// `(e_bits, m_bits, has_inf, has_nan)`. Used for the named sub-F16
/// dtypes (F8E5M2, F8E4M3FN, F4E2M1, F6E3M2, F6E2M3) and any custom
/// FloatType that fits in F32 compute repr.
///
/// Algorithm summary:
///   1. Pull F32 raw bits, save sign, take magnitude.
///   2. Branch on special values: NaN, ±inf, zero / F32 subnormal.
///   3. Normal F32: re-bias exponent. Check overflow / underflow.
///   4. **Target normal range** — RTNE round F32 mantissa from 23 bits
///      down to `m_bits`, handling carry into the exponent.
///   5. **Target subnormal range** — variable-shift the full F32
///      mantissa (with implicit leading 1) right by
///      `(151 - bias - m_bits) - f32_biased_exp` bits, with RTNE; if
///      the rounded mantissa overflows `mant_mask`, promote to the
///      smallest normal.
///   6. Pack `(sign, exp, mant)` into `raw_reg`.
///
/// Register usage: `raw_reg` is the working register and the output;
/// `scr_sign`, `scr_exp` are scratch GPs; `rcx` is implicitly used
/// as the shift count register. The subnormal path additionally
/// clobbers `rdx` as a temporary for the round-bias computation —
/// callers must ensure `rdx` is not aliased with any input.
fn emit_subf16_float_encode(
    asm: &mut Assembler,
    ft: FloatType,
    xmm_src: u8,
    raw_reg: u8,
    scr_sign: u8,
    scr_exp: u8,
) {
    let m_bits = ft.mantissa_bits as u32;
    let e_bits = ft.exponent_bits as u32;
    let bias = ft.bias();
    let max_e = ft.max_biased_exponent();
    let max_usable_e: i32 = if ft.has_infinity {
        (max_e - 1) as i32
    } else {
        max_e as i32
    };
    let mant_mask = (1u32 << m_bits) - 1;
    let exp_offset: i32 = 127 - bias;
    let drop_bits = 23 - m_bits;

    // Canonical NaN bits per `FloatType::encode_nan` — sign is NOT
    // applied to NaN encodings (per the reference impl).
    let nan_bits: u32 = if ft.has_nan {
        if ft.has_infinity {
            (max_e << m_bits) | (1u32 << (m_bits - 1))
        } else {
            // FN: NaN at all-ones mantissa with max exp
            (max_e << m_bits) | mant_mask
        }
    } else {
        0
    };
    // Max-finite (or ±inf) magnitude for the saturate / inf path.
    let max_finite_bits = ft.encode_max_finite(0) as u32;
    let inf_or_max_mag: u32 = if ft.has_infinity {
        max_e << m_bits
    } else {
        max_finite_bits
    };

    // Subnormal-shift constant: shift = (151 - bias - m_bits) - f32_biased_exp.
    // Derived from full_mant * 2^(f32_true_exp - 24 + bias + m_bits) = target_mant.
    let const_subshift: i32 = 151 - bias - m_bits as i32;

    let nan_label = asm.new_dynamic_label();
    let inf_label = asm.new_dynamic_label();
    let zero_label = asm.new_dynamic_label();
    let overflow_label = asm.new_dynamic_label();
    let subnormal_label = asm.new_dynamic_label();
    let subnormal_pack_path = asm.new_dynamic_label();
    let underflow_zero_label = asm.new_dynamic_label();
    let pack_normal_label = asm.new_dynamic_label();
    let pack_with_sign_label = asm.new_dynamic_label();
    let done_label = asm.new_dynamic_label();

    let sign_shift = (m_bits + e_bits) as i8;

    dynasm!(asm
        ; .arch x64
        // Step 1: pull F32 raw bits
        ; movd Rd(raw_reg), Rx(xmm_src)

        // Step 2: save sign in scr_sign
        ; mov Rd(scr_sign), Rd(raw_reg)
        ; shr Rd(scr_sign), 31

        // Step 3: get magnitude
        ; and Rd(raw_reg), 0x7fffffff_u32 as i32

        // Step 4: special-value branches
        ; cmp Rd(raw_reg), DWORD 0x7f800000_u32 as i32
        ; ja =>nan_label
        ; je =>inf_label
        ; test Rd(raw_reg), Rd(raw_reg)
        ; jz =>zero_label
        ; cmp Rd(raw_reg), DWORD 0x800000_u32 as i32
        ; jb =>zero_label

        // Step 5: extract f32 biased exp into scr_exp
        ; mov Rd(scr_exp), Rd(raw_reg)
        ; shr Rd(scr_exp), 23

        // Step 6: keep f32 mantissa in raw_reg
        ; and Rd(raw_reg), DWORD 0x7fffff_u32 as i32

        // Step 7: re-bias to target
        ; sub Rd(scr_exp), exp_offset

        // Step 8: check overflow
        ; cmp Rd(scr_exp), max_usable_e
        ; jg =>overflow_label

        // Step 9: check underflow (target subnormal range)
        ; test Rd(scr_exp), Rd(scr_exp)
        ; jle =>subnormal_label

        // ─── Target normal range: RTNE round mantissa ───
        ; mov ecx, Rd(raw_reg)
        ; shr ecx, drop_bits as i8
        ; and ecx, 1
        ; add ecx, ((1u32 << (drop_bits - 1)) - 1) as i32
        ; add Rd(raw_reg), ecx
        ; shr Rd(raw_reg), drop_bits as i8

        // Carry check: did mantissa overflow into the exponent?
        ; mov ecx, Rd(raw_reg)
        ; shr ecx, m_bits as i8
        ; add Rd(scr_exp), ecx
        ; and Rd(raw_reg), DWORD mant_mask as i32

        // Re-check overflow after carry
        ; cmp Rd(scr_exp), max_usable_e
        ; jg =>overflow_label

        // Pack: raw_reg = mant; scr_exp = exp; scr_sign = sign
        ; =>pack_normal_label
        ; shl Rd(scr_exp), m_bits as i8
        ; or Rd(raw_reg), Rd(scr_exp)
        ; =>pack_with_sign_label
        ; shl Rd(scr_sign), sign_shift
        ; or Rd(raw_reg), Rd(scr_sign)
        ; jmp =>done_label

        // ─── Target subnormal range ───
        ; =>subnormal_label
        // raw_reg holds f32 mantissa (low 23 bits). Add the implicit
        // leading 1: full_mant = (1 << 23) | f32_mant.
        ; or Rd(raw_reg), DWORD 0x800000_u32 as i32

        // shift = const_subshift - f32_biased_exp
        //       = const_subshift - (scr_exp + exp_offset)
        //       = (const_subshift - exp_offset) - scr_exp
        ; mov ecx, (const_subshift - exp_offset)
        ; sub ecx, Rd(scr_exp)

        // If shift > 24, the value rounds to ±0.
        ; cmp ecx, 24
        ; jg =>underflow_zero_label

        // RTNE shift right by `cl` bits.
        // We need: bias_const = (1 << (cl - 1)) - 1, lsb_kept, then
        // raw_reg += (bias_const + lsb_kept), then raw_reg >>= cl.
        //
        // Reuse scr_exp to hold lsb_kept (we no longer need the
        // target_biased_exp; we'll set it to 0 or 1 after the round).
        ; mov Rd(scr_exp), Rd(raw_reg)
        ; shr Rd(scr_exp), cl
        ; and Rd(scr_exp), 1
        // Compute bias_const in rdx (implicitly clobbered scratch).
        ; mov edx, 1
        ; shl edx, cl
        ; shr edx, 1
        ; sub edx, 1
        ; add edx, Rd(scr_exp)
        ; add Rd(raw_reg), edx
        ; shr Rd(raw_reg), cl

        // Promotion check: if raw_reg > mant_mask, the round bumped us
        // into the smallest normal (exp=1, mant=0).
        ; cmp Rd(raw_reg), DWORD mant_mask as i32
        ; jbe =>subnormal_pack_path
        ; xor Rd(raw_reg), Rd(raw_reg)
        ; mov Rd(scr_exp), 1
        ; jmp =>pack_normal_label
        ; =>subnormal_pack_path
        ; xor Rd(scr_exp), Rd(scr_exp)
        ; jmp =>pack_normal_label

        // ─── Underflow → ±0 ───
        ; =>underflow_zero_label
        ; xor Rd(raw_reg), Rd(raw_reg)
        ; jmp =>pack_with_sign_label

        // ─── Overflow → ±inf or ±max_finite ───
        ; =>overflow_label
        ; mov Rd(raw_reg), DWORD inf_or_max_mag as i32
        ; jmp =>pack_with_sign_label

        // ─── Input was ±inf ───
        ; =>inf_label
        ; mov Rd(raw_reg), DWORD inf_or_max_mag as i32
        ; jmp =>pack_with_sign_label

        // ─── Input was ±0 / F32 subnormal ───
        ; =>zero_label
        ; xor Rd(raw_reg), Rd(raw_reg)
        ; jmp =>pack_with_sign_label

        // ─── Input was NaN ───
        // encode_nan() does not set the sign bit, so we skip the sign
        // pack. For !has_nan dtypes, NaN encodes to +0 (also unsigned).
        ; =>nan_label
        ; mov Rd(raw_reg), DWORD nan_bits as i32
        ; jmp =>done_label

        ; =>done_label
    );
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
