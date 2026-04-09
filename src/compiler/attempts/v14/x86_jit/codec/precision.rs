//! In-register precision narrowing.
//!
//! `narrow_to(value, target_dtype)` emits the assembly that takes a
//! value in the compute representation and rounds (or truncates) it
//! to the target dtype's representation, **without leaving the
//! register**. The result stays in the same compute repr — narrowing
//! does not cross between Float and Int compute reprs.
//!
//! This is the third codec primitive, distinct from format conversion.
//! It enforces the dtype contract for fused intermediates: a chain of
//! BF16 ops in F32 compute must round per step (per the §4.7 reduce
//! rule generalized to any narrowing-cast intermediate), and that
//! per-step rounding is done by emitting `narrow_to(BF16)` between
//! ops, never touching memory.
//!
//! # Implementation
//!
//! Per the codec doc §3 contract:
//!
//! > `narrow_to(value, target_dtype)` produces the same bit pattern
//! > that `decode(encode(value, target_dtype), target_dtype)` would,
//! > but stays in the compute repr.
//!
//! The simplest implementation that satisfies this contract is to
//! actually emit `encode → decode` through a temporary GP register —
//! this reuses the existing format primitives, automatically inherits
//! NaN canonicalization for floats, and produces correct results for
//! every named dtype.
//!
//! No-op cases (the load-bearing property from §3.2 of the codec doc):
//!
//! - F32 compute repr, target F32 → zero instructions
//! - F64 compute repr, target F64 → zero instructions
//! - Int compute repr, target I64 / U64 → zero instructions
//!
//! These cover the common case where the dtype matches the compute
//! repr exactly, so a chain of aligned F32 ops compiles with no
//! `narrow_to` overhead.
//!
//! For all other targets we go through the format codec. The cost
//! per `narrow_to` is the sum of the encode and decode instruction
//! counts for the target dtype.

use dynasmrt::x64::Assembler;

use crate::numeric_dtype::NumericDType;

use super::format::{CodecSlot, CodecTables, ComputeRepr, emit_decode, emit_encode};

/// Emit code that narrows the compute-repr value in `slot` to the
/// precision / range of `target_dtype`, leaving the result back in
/// `slot` in the same compute repr.
///
/// # Register usage
///
/// - `slot`: must match `ComputeRepr::for_dtype(target_dtype)`. The
///   value in `slot` is read and overwritten.
/// - `raw_temp`: GP register used as the intermediate raw-bits
///   storage between the inner `encode` and `decode`. **Clobbered.**
///   For Int slots, must be distinct from the slot's GP.
/// - `scratch_gp1`, `scratch_gp2`: free GP registers, **clobbered.**
///   Used by the inner `encode` / `decode` calls.
/// - `scratch_xmm`: free XMM register, **clobbered.** Used by F16C
///   paths.
/// - `tables`: codec table store; the inner decode allocates a
///   sub-F16 lookup table here when narrowing to one of those dtypes.
///
/// `rcx` and `rdx` are implicitly clobbered, matching the contract
/// in [`emit_encode`].
///
/// # Errors
///
/// Returns `Err` for an unsupported dtype, a slot variant that
/// doesn't match the target's compute repr, or (for Int slots) when
/// `raw_temp` aliases the slot register.
pub fn emit_narrow_to(
    asm: &mut Assembler,
    target_dtype: NumericDType,
    slot: CodecSlot,
    raw_temp: u8,
    scratch_gp1: u8,
    scratch_gp2: u8,
    scratch_xmm: u8,
    tables: &mut CodecTables,
) -> Result<(), String> {
    let compute_repr = ComputeRepr::for_dtype(target_dtype);

    // Validate slot variant first.
    match (compute_repr, slot) {
        (ComputeRepr::F32 | ComputeRepr::F64, CodecSlot::Xmm(_)) => {}
        (ComputeRepr::Int, CodecSlot::Gp(_)) => {}
        _ => {
            return Err(format!(
                "emit_narrow_to: slot variant does not match compute repr for {target_dtype}"
            ));
        }
    }

    // No-op cases: the target dtype IS the compute repr's natural
    // form (or close enough that the bit pattern is unchanged).
    if is_compute_repr_native(target_dtype) {
        return Ok(());
    }

    // For Int slots, ensure raw_temp doesn't alias the slot register
    // (otherwise the encode would clobber its own input).
    if let CodecSlot::Gp(g) = slot
        && raw_temp == g
    {
        return Err(format!(
            "emit_narrow_to: raw_temp ({raw_temp}) aliases the int slot GP ({g})"
        ));
    }

    // General path: encode the slot value to raw_temp, then decode
    // raw_temp back into the slot. The format codec already handles
    // NaN canonicalization on the encode side, so the round-trip
    // produces the contract-correct bit pattern.
    emit_encode(
        asm,
        target_dtype,
        slot,
        raw_temp,
        scratch_gp1,
        scratch_gp2,
        scratch_xmm,
    )?;
    emit_decode(
        asm,
        target_dtype,
        raw_temp,
        slot,
        scratch_gp1,
        scratch_xmm,
        tables,
    )?;
    Ok(())
}

/// Returns `true` when narrowing to this dtype is a no-op because
/// the dtype is the natural form of its compute repr. The codec
/// emits zero instructions for these cases.
///
/// Note: `U64` is **not** in the no-op list. The Int compute repr
/// is "i64 sign-extended for signed sources, u64 zero-extended for
/// unsigned sources" — but the codec sees only the bits, not the
/// source's signedness. A signed i64 = -1 sitting in the slot has
/// bits `0xff..ff`; per `cast_raw(I64, U64)` (which uses
/// `clamp_unsigned`, mapping negatives to 0), narrow_to(U64) of -1
/// must produce 0, not `u64::MAX`. So narrow_to(U64) goes through
/// the encode/decode round-trip, where the saturating
/// `emit_int_encode` clamps negatives to 0. The cost is small
/// (4 instructions: xor + test + cmovs + the trailing decode mov).
fn is_compute_repr_native(dtype: NumericDType) -> bool {
    matches!(
        dtype,
        NumericDType::F32 | NumericDType::F64 | NumericDType::I64
    )
}
