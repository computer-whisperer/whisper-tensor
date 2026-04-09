//! Unit tests for [`super::super::orch::address::emit_compute_bit_offset`].
//!
//! Each test builds a real `BufferLayout` (via `compute_layout`) for a
//! tiny synthetic graph, wraps a single `emit_compute_bit_offset` call
//! in a JIT function, and asserts the bit offset register holds the
//! value `InputRef::resolve(i)` would yield (after multiplying by the
//! slot's bit_stride and adding its bit_offset).
//!
//! Test harness ABI:
//!
//! - `Const` cases: `extern "C" fn() -> u64`, returning the bit
//!   offset in `rax`.
//! - `Reg` cases:   `extern "C" fn(u64) -> u64`, taking the
//!   iteration index in `rdi` and returning the bit offset in `rax`.
//!
//! The address layer never touches the buffer, so we don't need a
//! pointer argument or a real buffer for any of these tests.

use crate::compiler::attempts::v14::layout::{BufferLayout, compute_layout};
use crate::compiler::attempts::v14::x86_jit::orch::address::{
    AddressInfo, IterVar, emit_compute_bit_offset,
};
use crate::graph::GlobalId;
use crate::nano_graph::ops::ScalarOp;
use crate::nano_graph::pattern::{AtomId, AtomRange, InputRef, NanoGraph};
use crate::numeric_dtype::NumericDType;
use crate::pool::SystemPool;

use super::jit_harness::JitFn;

const RAX: u8 = 0; // dst_bit_reg for all tests
const RDI: u8 = 7; // iter register for the variable-i tests
const SCRATCH: u8 = 8; // r8 for arithmetic scratch

/// Build a flat 1D NanoGraph: one input tensor of `dtype` followed by
/// an Identity group consuming it via `affine(1)`. Returns the layout
/// plus the input's base AtomId and the identity group's base AtomId.
fn flat_layout(count: u64, dtype: NumericDType) -> (BufferLayout, AtomId, AtomId, AtomRange) {
    let mut g: NanoGraph<'static, SystemPool> = NanoGraph::new();
    let inp = g.add_input_tensor(GlobalId(0), count, dtype);
    let ident = g.push_group(
        count,
        dtype,
        ScalarOp::Identity,
        vec![],
        vec![InputRef::affine(inp, 1)],
    );
    let out = AtomRange {
        base: ident,
        count,
        dtype,
    };
    let layout = compute_layout(&g, std::slice::from_ref(&out));
    (layout, inp, ident, out)
}

/// Run a no-arg JIT (Const cases) and return the value left in rax.
fn run_const(jit: &JitFn) -> u64 {
    let f: extern "C" fn() -> u64 = unsafe { std::mem::transmute(jit.ptr()) };
    f()
}

/// Run a single-arg JIT (Reg cases): pass `i` in rdi, return rax.
fn run_with_i(jit: &JitFn, i: u64) -> u64 {
    let f: extern "C" fn(u64) -> u64 = unsafe { std::mem::transmute(jit.ptr()) };
    f(i)
}

#[test]
fn broadcast_returns_constant_bit_offset() {
    let (layout, inp, _ident, _out) = flat_layout(8, NumericDType::F32);
    // Pick atom 3 inside the input — its bit offset is the slot's
    // bit_offset + 3 * bit_stride.
    let target = AtomId(inp.0 + 3);
    let (slot, _) = layout.find(target).expect("input slot present");
    let want = slot.bit_offset + 3 * slot.bit_stride;

    let info_cell = std::cell::RefCell::new(None::<AddressInfo>);
    let jit = JitFn::build(|asm| {
        let info = emit_compute_bit_offset(
            asm,
            &layout,
            &InputRef::Broadcast(target),
            IterVar::Const(0),
            0,
            RAX,
            SCRATCH,
        )
        .expect("Broadcast address emit");
        *info_cell.borrow_mut() = Some(info);
    });
    let got = run_const(&jit);
    assert_eq!(got, want, "Broadcast bit_offset");

    let info = info_cell.into_inner().unwrap();
    assert_eq!(info.dtype, NumericDType::F32);
    assert_eq!(info.n_bits, 32);
}

#[test]
fn explicit_single_atom_returns_constant_bit_offset() {
    let (layout, inp, _ident, _out) = flat_layout(4, NumericDType::I64);
    let target = AtomId(inp.0 + 2);
    let (slot, _) = layout.find(target).expect("input slot present");
    let want = slot.bit_offset + 2 * slot.bit_stride;

    let jit = JitFn::build(|asm| {
        emit_compute_bit_offset(
            asm,
            &layout,
            &InputRef::Explicit(vec![target]),
            IterVar::Const(0),
            0,
            RAX,
            SCRATCH,
        )
        .expect("Explicit single emit");
    });
    assert_eq!(run_const(&jit), want);
}

#[test]
fn strided_1d_const_iter() {
    // Affine InputRef base=inp, stride=1. For each i, the bit offset
    // is the slot's bit_offset + i * bit_stride.
    let (layout, inp, _ident, _out) = flat_layout(16, NumericDType::F32);
    let (slot, _) = layout.find(inp).expect("input slot present");

    for i in [0u64, 1, 5, 15] {
        let want = slot.bit_offset + i * slot.bit_stride;
        let jit = JitFn::build(|asm| {
            emit_compute_bit_offset(
                asm,
                &layout,
                &InputRef::affine(inp, 1),
                IterVar::Const(i),
                0,
                RAX,
                SCRATCH,
            )
            .expect("Strided 1D const emit");
        });
        assert_eq!(run_const(&jit), want, "i={i}");
    }
}

#[test]
fn strided_1d_reg_iter_stride_one() {
    let (layout, inp, _ident, _out) = flat_layout(16, NumericDType::F32);
    let (slot, _) = layout.find(inp).expect("input slot present");
    let base_bit = slot.bit_offset;
    let bit_stride = slot.bit_stride;

    let jit = JitFn::build(|asm| {
        emit_compute_bit_offset(
            asm,
            &layout,
            &InputRef::affine(inp, 1),
            IterVar::Reg(RDI),
            0,
            RAX,
            SCRATCH,
        )
        .expect("Strided 1D reg emit");
    });
    for i in [0u64, 1, 7, 15] {
        let want = base_bit + i * bit_stride;
        assert_eq!(run_with_i(&jit, i), want, "i={i}");
    }
}

#[test]
fn strided_1d_reg_iter_stride_two() {
    // Stride-2 affine: bit_offset(i) = base + 2 * bit_stride * i.
    // The slot must cover atoms 0..32 so the stride-2 access stays
    // in range.
    let (layout, inp, _ident, _out) = flat_layout(32, NumericDType::I32);
    let (slot, _) = layout.find(inp).expect("input slot present");
    let base_bit = slot.bit_offset;
    let bit_stride = slot.bit_stride;

    let jit = JitFn::build(|asm| {
        emit_compute_bit_offset(
            asm,
            &layout,
            &InputRef::affine(inp, 2),
            IterVar::Reg(RDI),
            0,
            RAX,
            SCRATCH,
        )
        .expect("Strided 1D stride-2 emit");
    });
    for i in [0u64, 1, 5, 10] {
        let want = base_bit + 2 * bit_stride * i;
        assert_eq!(run_with_i(&jit, i), want, "i={i}");
    }
}

#[test]
fn strided_1d_const_with_atom_offset() {
    // atom_offset shifts the implicit start: bit_offset(c) = base +
    // bit_stride * (atom_offset + c). Verifies the path that adds the
    // atom_offset contribution at compile time.
    let (layout, inp, _ident, _out) = flat_layout(16, NumericDType::F32);
    let (slot, _) = layout.find(inp).expect("input slot present");
    let base_bit = slot.bit_offset;
    let bit_stride = slot.bit_stride;

    // For a logical group with atom_offset=4, asking for i=2 means
    // the consumer-side flat index is 2 — but the address layer
    // (mirroring the cranelift backend) treats `atom_offset` as a
    // shift on the slot lookup, not a shift on the iteration counter.
    // The constant path resolves to base_bit + bit_stride * 2 (the
    // atom_offset is already baked into base_bit_signed — see the
    // back-compute branch in emit_strided_1d).
    //
    // Since this layout's base atom IS in the layout (no split), the
    // back-compute branch is not exercised; the result is identical
    // to atom_offset=0. The atom_offset path is exercised by the
    // split-group P2.B.4 tests, but we add this case to lock in the
    // semantics for non-split groups.
    let jit = JitFn::build(|asm| {
        emit_compute_bit_offset(
            asm,
            &layout,
            &InputRef::affine(inp, 1),
            IterVar::Const(2),
            4, // atom_offset — non-zero, but base is in layout
            RAX,
            SCRATCH,
        )
        .expect("Strided 1D const+atom_offset emit");
    });
    assert_eq!(run_const(&jit), base_bit + bit_stride * 2);
}

#[test]
fn strided_1d_reg_iter_with_bool_input() {
    // Sub-byte semantic width but byte-padded slot in phase 1.
    // bit_stride = 8, elem_bits = 1. The address calculation uses
    // bit_stride only — the n_bits comes back via AddressInfo.
    let (layout, inp, _ident, _out) = flat_layout(8, NumericDType::BOOL);
    let (slot, _) = layout.find(inp).expect("bool input slot present");
    let base_bit = slot.bit_offset;
    let bit_stride = slot.bit_stride;
    assert_eq!(slot.elem_bits, 1, "bool elem_bits");
    assert_eq!(bit_stride, 8, "bool byte-padded in phase 1");

    let info_cell = std::cell::RefCell::new(None::<AddressInfo>);
    let jit = JitFn::build(|asm| {
        let info = emit_compute_bit_offset(
            asm,
            &layout,
            &InputRef::affine(inp, 1),
            IterVar::Reg(RDI),
            0,
            RAX,
            SCRATCH,
        )
        .expect("Bool affine reg emit");
        *info_cell.borrow_mut() = Some(info);
    });
    for i in [0u64, 3, 7] {
        let want = base_bit + i * bit_stride;
        assert_eq!(run_with_i(&jit, i), want, "i={i}");
    }

    let info = info_cell.into_inner().unwrap();
    assert_eq!(info.dtype, NumericDType::BOOL);
    assert_eq!(info.n_bits, 1, "BOOL n_bits is the semantic width");
}

#[test]
fn rejects_strided_n_d() {
    let (layout, inp, _ident, _out) = flat_layout(8, NumericDType::F32);
    let nd = InputRef::Strided {
        base: inp,
        dim_strides: vec![0, 1],
        dim_shape: vec![u64::MAX, 4],
    };
    let mut asm = dynasmrt::x64::Assembler::new().unwrap();
    let err = emit_compute_bit_offset(&mut asm, &layout, &nd, IterVar::Const(0), 0, RAX, SCRATCH)
        .expect_err("n-d Strided should reject in P2.B.2");
    assert!(err.contains("Strided n-d"), "{err}");
}

#[test]
fn rejects_explicit_multi() {
    let (layout, inp, _ident, _out) = flat_layout(4, NumericDType::F32);
    let multi = InputRef::Explicit(vec![inp, AtomId(inp.0 + 1)]);
    let mut asm = dynasmrt::x64::Assembler::new().unwrap();
    let err = emit_compute_bit_offset(
        &mut asm,
        &layout,
        &multi,
        IterVar::Const(0),
        0,
        RAX,
        SCRATCH,
    )
    .expect_err("multi-entry Explicit should reject in P2.B.2");
    assert!(err.contains("Explicit"), "{err}");
}

#[test]
fn rejects_register_aliasing_for_reg_iter() {
    let (layout, inp, _ident, _out) = flat_layout(4, NumericDType::F32);
    let mut asm = dynasmrt::x64::Assembler::new().unwrap();
    // dst_bit_reg == iter_reg
    let err = emit_compute_bit_offset(
        &mut asm,
        &layout,
        &InputRef::affine(inp, 1),
        IterVar::Reg(RAX),
        0,
        RAX,
        SCRATCH,
    )
    .expect_err("dst aliasing iter should reject");
    assert!(err.contains("aliases iter_reg"), "{err}");
}
