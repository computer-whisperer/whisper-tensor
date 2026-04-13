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
use crate::compiler::attempts::v14::placer::{AtomPlacementMap, run_placer};
use crate::compiler::attempts::v14::types::{Phase, Span};
use crate::compiler::attempts::v14::x86_jit::orch::address::{
    AddressInfo, AddressTables, IterVar, emit_compute_bit_offset,
};
use crate::graph::GlobalId;
use crate::nano_graph::ops::ScalarOp;
use crate::nano_graph::pattern::{AtomId, AtomRange, GroupInput, InputRef, NanoGraph};
use crate::numeric_dtype::NumericDType;
use crate::pool::SystemPool;

use super::jit_harness::JitFn;

const RAX: u8 = 0; // dst_bit_reg for all tests
const RDI: u8 = 7; // iter register for the variable-i tests
const SCRATCH: u8 = 8; // r8 for arithmetic scratch

/// Adjust a bit offset to the expected output value: byte offset when
/// the slot qualifies for the byte-aligned fast path, bit offset
/// otherwise.
fn expected_offset(slot: &crate::compiler::attempts::v14::layout::SlotInfo, bit_off: u64) -> u64 {
    if slot.is_byte_aligned() && matches!(slot.elem_bits, 8 | 16 | 32 | 64) {
        bit_off / 8
    } else {
        bit_off
    }
}

/// Build a trivial placement (one phase, one span) for a test graph
/// so `compute_layout` has the AtomPlacementMap it needs.
fn test_placement(
    graph: &NanoGraph<'static, SystemPool>,
    outputs: &[AtomRange],
) -> AtomPlacementMap {
    let span_inputs: Vec<AtomRange> = graph
        .input_tensors()
        .iter()
        .map(|it| AtomRange {
            base: it.base_id,
            count: it.count,
            dtype: it.dtype,
        })
        .collect();
    let phases = vec![Phase {
        spans: vec![Span {
            graph: graph.clone(),
            inputs: span_inputs,
            outputs: outputs.to_vec(),
        }],
    }];
    run_placer(graph, &phases, outputs).expect("placer failed in test harness")
}

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
        vec![GroupInput::scalar(InputRef::affine(inp, 1))],
    );
    let out = AtomRange {
        base: ident,
        count,
        dtype,
    };
    let placement = test_placement(&g, std::slice::from_ref(&out));
    let layout = compute_layout(&g, std::slice::from_ref(&out), false, &placement).expect("layout");
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
    let want = expected_offset(slot, slot.bit_offset + 3 * slot.bit_stride);

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
            &mut AddressTables::new(),
        )
        .expect("Broadcast address emit");
        *info_cell.borrow_mut() = Some(info);
    });
    let got = run_const(&jit);
    assert_eq!(got, want, "Broadcast offset");

    let info = info_cell.into_inner().unwrap();
    assert_eq!(info.dtype, NumericDType::F32);
    assert_eq!(info.n_bits, 32);
}

#[test]
fn explicit_single_atom_returns_constant_bit_offset() {
    let (layout, inp, _ident, _out) = flat_layout(4, NumericDType::I64);
    let target = AtomId(inp.0 + 2);
    let (slot, _) = layout.find(target).expect("input slot present");
    let want = expected_offset(slot, slot.bit_offset + 2 * slot.bit_stride);

    let jit = JitFn::build(|asm| {
        emit_compute_bit_offset(
            asm,
            &layout,
            &InputRef::Explicit(vec![target]),
            IterVar::Const(0),
            0,
            RAX,
            SCRATCH,
            &mut AddressTables::new(),
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
        let want = expected_offset(slot, slot.bit_offset + i * slot.bit_stride);
        let jit = JitFn::build(|asm| {
            emit_compute_bit_offset(
                asm,
                &layout,
                &InputRef::affine(inp, 1),
                IterVar::Const(i),
                0,
                RAX,
                SCRATCH,
                &mut AddressTables::new(),
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
            &mut AddressTables::new(),
        )
        .expect("Strided 1D reg emit");
    });
    for i in [0u64, 1, 7, 15] {
        let want = expected_offset(slot, base_bit + i * bit_stride);
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
            &mut AddressTables::new(),
        )
        .expect("Strided 1D stride-2 emit");
    });
    for i in [0u64, 1, 5, 10] {
        let want = expected_offset(slot, base_bit + 2 * bit_stride * i);
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
            &mut AddressTables::new(),
        )
        .expect("Strided 1D const+atom_offset emit");
    });
    assert_eq!(
        run_const(&jit),
        expected_offset(slot, base_bit + bit_stride * 2)
    );
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
            &mut AddressTables::new(),
        )
        .expect("Bool affine reg emit");
        *info_cell.borrow_mut() = Some(info);
    });
    for i in [0u64, 3, 7] {
        let want = expected_offset(slot, base_bit + i * bit_stride);
        assert_eq!(run_with_i(&jit, i), want, "i={i}");
    }

    let info = info_cell.into_inner().unwrap();
    assert_eq!(info.dtype, NumericDType::BOOL);
    assert_eq!(info.n_bits, 1, "BOOL n_bits is the semantic width");
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
        &mut AddressTables::new(),
    )
    .expect_err("dst aliasing iter should reject");
    assert!(err.contains("aliases iter_reg"), "{err}");
}

// ─── N-d Strided tests ─────────────────────────────────────────────

/// N-d tests use r10 as dst (not rax), since the n-d Reg path
/// clobbers rax for the division. A trailing `mov rax, r10` copies
/// the result to the return register.
const ND_DST: u8 = 10; // r10

/// Build a 2D-modular layout: input count atoms, an Identity group
/// whose input uses `InputRef::modular(inp, stride, modulus)`.
fn modular_layout(
    count: u64,
    stride: i64,
    modulus: u64,
    dtype: NumericDType,
) -> (BufferLayout, AtomId, AtomId, AtomRange) {
    let mut g: NanoGraph<'static, SystemPool> = NanoGraph::new();
    let inp = g.add_input_tensor(GlobalId(0), count, dtype);
    let ident = g.push_group(
        count,
        dtype,
        ScalarOp::Identity,
        vec![],
        vec![GroupInput::scalar(InputRef::modular(inp, stride, modulus))],
    );
    let out = AtomRange {
        base: ident,
        count,
        dtype,
    };
    let placement = test_placement(&g, std::slice::from_ref(&out));
    let layout = compute_layout(&g, std::slice::from_ref(&out), false, &placement).expect("layout");
    (layout, inp, ident, out)
}

/// Run a no-arg JIT (Const cases) where dst is ND_DST (r10) and the
/// function copies to rax before returning.
fn run_nd_const(jit: &JitFn) -> u64 {
    let f: extern "C" fn() -> u64 = unsafe { std::mem::transmute(jit.ptr()) };
    f()
}

/// Run a single-arg JIT (Reg cases) where dst is ND_DST (r10) and
/// the function copies to rax before returning.
fn run_nd_with_i(jit: &JitFn, i: u64) -> u64 {
    let f: extern "C" fn(u64) -> u64 = unsafe { std::mem::transmute(jit.ptr()) };
    f(i)
}

#[test]
fn strided_2d_modular_const() {
    // modular(inp, stride=1, modulus=4): resolve(i) = base + (i % 4)
    let (layout, inp, _ident, _out) = modular_layout(16, 1, 4, NumericDType::F32);
    let (slot, _) = layout.find(inp).expect("input slot");

    for i in [0u64, 1, 3, 4, 7, 15] {
        let expected_atom_off = (i % 4) as i64;
        let want = expected_offset(
            slot,
            (slot.bit_offset as i64 + expected_atom_off * slot.bit_stride as i64) as u64,
        );
        let jit = JitFn::build(|asm| {
            emit_compute_bit_offset(
                asm,
                &layout,
                &InputRef::modular(inp, 1, 4),
                IterVar::Const(i),
                0,
                RAX,
                SCRATCH,
                &mut AddressTables::new(),
            )
            .expect("modular const emit");
        });
        assert_eq!(run_const(&jit), want, "modular const i={i}");
    }
}

#[test]
fn strided_2d_modular_reg_power_of_two() {
    // modular(inp, stride=1, modulus=4) — power-of-2 fast path.
    let (layout, inp, _ident, _out) = modular_layout(16, 1, 4, NumericDType::F32);
    let (slot, _) = layout.find(inp).expect("input slot");
    let base_bit = slot.bit_offset;
    let bit_stride = slot.bit_stride;

    use dynasmrt::{DynasmApi, dynasm};
    let jit = JitFn::build(|asm| {
        emit_compute_bit_offset(
            asm,
            &layout,
            &InputRef::modular(inp, 1, 4),
            IterVar::Reg(RDI),
            0,
            ND_DST,
            SCRATCH,
            &mut AddressTables::new(),
        )
        .expect("modular reg emit");
        // Copy result to rax for return.
        dynasm!(asm ; .arch x64 ; mov rax, Rq(ND_DST));
    });
    for i in [0u64, 1, 3, 4, 7, 15] {
        let want = expected_offset(slot, base_bit + (i % 4) * bit_stride);
        assert_eq!(run_nd_with_i(&jit, i), want, "modular reg i={i}");
    }
}

#[test]
fn strided_2d_modular_reg_non_power_of_two() {
    // modular(inp, stride=1, modulus=3) — general div path.
    let (layout, inp, _ident, _out) = modular_layout(12, 1, 3, NumericDType::F32);
    let (slot, _) = layout.find(inp).expect("input slot");
    let base_bit = slot.bit_offset;
    let bit_stride = slot.bit_stride;

    use dynasmrt::{DynasmApi, dynasm};
    let jit = JitFn::build(|asm| {
        emit_compute_bit_offset(
            asm,
            &layout,
            &InputRef::modular(inp, 1, 3),
            IterVar::Reg(RDI),
            0,
            ND_DST,
            SCRATCH,
            &mut AddressTables::new(),
        )
        .expect("modular reg non-pow2 emit");
        dynasm!(asm ; .arch x64 ; mov rax, Rq(ND_DST));
    });
    for i in [0u64, 1, 2, 3, 5, 11] {
        let want = expected_offset(slot, base_bit + (i % 3) * bit_stride);
        assert_eq!(run_nd_with_i(&jit, i), want, "modular reg mod3 i={i}");
    }
}

#[test]
fn strided_2d_broadcast_reg() {
    // strided_broadcast(inp, stride=1, repeat=4): resolve(i) = base + (i / 4)
    let mut g: NanoGraph<'static, SystemPool> = NanoGraph::new();
    let inp = g.add_input_tensor(GlobalId(0), 16, NumericDType::F32);
    let ident = g.push_group(
        16,
        NumericDType::F32,
        ScalarOp::Identity,
        vec![],
        vec![GroupInput::scalar(InputRef::strided_broadcast(inp, 1, 4))],
    );
    let out = AtomRange {
        base: ident,
        count: 16,
        dtype: NumericDType::F32,
    };
    let placement = test_placement(&g, std::slice::from_ref(&out));
    let layout = compute_layout(&g, std::slice::from_ref(&out), false, &placement).expect("layout");
    let (slot, _) = layout.find(inp).expect("input slot");
    let base_bit = slot.bit_offset;
    let bit_stride = slot.bit_stride;

    use dynasmrt::{DynasmApi, dynasm};
    let jit = JitFn::build(|asm| {
        emit_compute_bit_offset(
            asm,
            &layout,
            &InputRef::strided_broadcast(inp, 1, 4),
            IterVar::Reg(RDI),
            0,
            ND_DST,
            SCRATCH,
            &mut AddressTables::new(),
        )
        .expect("strided_broadcast reg emit");
        dynasm!(asm ; .arch x64 ; mov rax, Rq(ND_DST));
    });
    for i in [0u64, 1, 3, 4, 7, 12, 15] {
        let want = expected_offset(slot, base_bit + (i / 4) * bit_stride);
        assert_eq!(run_nd_with_i(&jit, i), want, "strided_broadcast i={i}");
    }
}

// ─── Multi-entry Explicit tests ─────────────────────────────────────

#[test]
fn explicit_multi_const() {
    let (layout, inp, _ident, _out) = flat_layout(8, NumericDType::F32);
    let ids = vec![
        AtomId(inp.0 + 3),
        AtomId(inp.0 + 0),
        AtomId(inp.0 + 7),
        AtomId(inp.0 + 1),
    ];
    let (slot, _) = layout.find(inp).expect("input slot");

    for (c, id) in ids.iter().enumerate() {
        let elem_idx = id.0 - slot.atom_base.0;
        let want = expected_offset(slot, slot.bit_offset + elem_idx * slot.bit_stride);
        let ids_clone = ids.clone();
        let jit = JitFn::build(|asm| {
            emit_compute_bit_offset(
                asm,
                &layout,
                &InputRef::Explicit(ids_clone),
                IterVar::Const(c as u64),
                0,
                RAX,
                SCRATCH,
                &mut AddressTables::new(),
            )
            .expect("explicit multi const emit");
        });
        assert_eq!(run_const(&jit), want, "explicit multi const c={c}");
    }
}

#[test]
fn explicit_multi_reg() {
    let (layout, inp, _ident, _out) = flat_layout(8, NumericDType::F32);
    let ids = vec![
        AtomId(inp.0 + 5),
        AtomId(inp.0 + 2),
        AtomId(inp.0 + 7),
        AtomId(inp.0 + 0),
    ];
    let (slot, _) = layout.find(inp).expect("input slot");

    // Build expected offsets (byte offsets when byte-aligned).
    let expected: Vec<u64> = ids
        .iter()
        .map(|id| {
            let elem_idx = id.0 - slot.atom_base.0;
            expected_offset(slot, slot.bit_offset + elem_idx * slot.bit_stride)
        })
        .collect();

    // The Reg path builds a lookup table, so AddressTables must
    // outlive the JIT call. Use a struct that holds both.
    let mut tables = AddressTables::new();
    use dynasmrt::{DynasmApi, dynasm};
    let jit = JitFn::build(|asm| {
        emit_compute_bit_offset(
            asm,
            &layout,
            &InputRef::Explicit(ids.clone()),
            IterVar::Reg(RDI),
            0,
            RAX,
            SCRATCH,
            &mut tables,
        )
        .expect("explicit multi reg emit");
    });
    for (i, want) in expected.iter().enumerate() {
        assert_eq!(
            run_with_i(&jit, i as u64),
            *want,
            "explicit multi reg i={i}"
        );
    }
}
