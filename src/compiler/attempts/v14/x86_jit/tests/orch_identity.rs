//! A/B tests for Identity-only graphs through the new x86_jit pipeline.
//!
//! Drives `X86JitSpan` end-to-end against `PoolEvalSpan` for graphs
//! whose only op is [`ScalarOp::Identity`]. Verifies that:
//!
//! - The pipeline (compute_layout → emit_group → marshal IO) is
//!   wired correctly.
//! - `orch::address::emit_compute_bit_offset` produces correct bit
//!   offsets for the Broadcast / Strided 1D / Explicit-1 shapes.
//! - The `bit_io` codec round-trips bits without corruption when
//!   driven from inside the loop body.
//!
//! Each test builds a tiny graph, supplies a few raw input bytes,
//! and lets `ab_test_bytes` panic on any byte mismatch.

use super::ab_harness::{ab_test_bytes, ab_test_bytes_sym};
use crate::graph::GlobalId;
use crate::nano_graph::ops::ScalarOp;
use crate::nano_graph::pattern::{AtomRange, GroupInput, InputRef, NanoGraph};
use crate::numeric_dtype::NumericDType;
use crate::pool::SystemPool;

/// Build an `inputs` argument for `ab_test_bytes` from a single
/// f32 slice. The base atom is the input tensor's first AtomId.
fn f32_input_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// Build a graph: one F32 input tensor → one F32 Identity group via
/// affine(1).
fn identity_chain_f32(count: u64) -> (NanoGraph<'static, SystemPool>, AtomRange, AtomRange) {
    let mut g = NanoGraph::new();
    let inp = g.add_input_tensor(GlobalId(0), count, NumericDType::F32);
    let out = g.push_group(
        count,
        NumericDType::F32,
        ScalarOp::Identity,
        vec![],
        vec![GroupInput::scalar(InputRef::affine(inp, 1))],
    );
    let in_range = AtomRange {
        base: inp,
        count,
        dtype: NumericDType::F32,
    };
    let out_range = AtomRange {
        base: out,
        count,
        dtype: NumericDType::F32,
    };
    (g, in_range, out_range)
}

#[test]
fn identity_f32_count_one() {
    let (graph, in_range, out_range) = identity_chain_f32(1);
    let bytes = f32_input_bytes(&[1.5_f32]);
    let outs = ab_test_bytes(
        &graph,
        &[(in_range.base, NumericDType::F32, bytes.clone())],
        &[out_range],
    );
    assert_eq!(outs[0], bytes, "F32 identity count=1 byte equality");
}

#[test]
fn identity_f32_count_eight() {
    let (graph, in_range, out_range) = identity_chain_f32(8);
    let values: Vec<f32> = (0..8).map(|i| i as f32 * 0.25 - 1.0).collect();
    let bytes = f32_input_bytes(&values);
    let outs = ab_test_bytes(
        &graph,
        &[(in_range.base, NumericDType::F32, bytes.clone())],
        &[out_range],
    );
    assert_eq!(outs[0], bytes, "F32 identity count=8 byte equality");
}

#[test]
fn identity_f32_count_one_thousand() {
    // Bigger than the loop's compile-time bounds — exercises a long
    // run of `bit_io` calls in the loop body.
    let (graph, in_range, out_range) = identity_chain_f32(1000);
    let values: Vec<f32> = (0..1000).map(|i| (i as f32).sin()).collect();
    let bytes = f32_input_bytes(&values);
    let outs = ab_test_bytes(
        &graph,
        &[(in_range.base, NumericDType::F32, bytes.clone())],
        &[out_range],
    );
    assert_eq!(outs[0], bytes, "F32 identity count=1000 byte equality");
}

#[test]
fn identity_i64_count_eight() {
    // 64-bit dtype: exercises the `n_bits == 64` path in bit_io
    // (no mask, full-width chunk).
    let mut g = NanoGraph::new();
    let inp = g.add_input_tensor(GlobalId(0), 8, NumericDType::I64);
    let out = g.push_group(
        8,
        NumericDType::I64,
        ScalarOp::Identity,
        vec![],
        vec![GroupInput::scalar(InputRef::affine(inp, 1))],
    );
    let values: [i64; 8] = [
        0,
        -1,
        i64::MAX,
        i64::MIN,
        42,
        -42,
        0x0123_4567_89ab_cdef,
        -0x0123_4567_89ab_cdef,
    ];
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let outs = ab_test_bytes(
        &g,
        &[(inp, NumericDType::I64, bytes.clone())],
        &[AtomRange {
            base: out,
            count: 8,
            dtype: NumericDType::I64,
        }],
    );
    assert_eq!(outs[0], bytes, "I64 identity byte equality");
}

#[test]
fn identity_u8_count_seventeen() {
    // 8-bit dtype + non-power-of-two count.
    let mut g = NanoGraph::new();
    let n: u64 = 17;
    let inp = g.add_input_tensor(GlobalId(0), n, NumericDType::U8);
    let out = g.push_group(
        n,
        NumericDType::U8,
        ScalarOp::Identity,
        vec![],
        vec![GroupInput::scalar(InputRef::affine(inp, 1))],
    );
    let bytes: Vec<u8> = (0..n as u8).collect();
    let outs = ab_test_bytes(
        &g,
        &[(inp, NumericDType::U8, bytes.clone())],
        &[AtomRange {
            base: out,
            count: n,
            dtype: NumericDType::U8,
        }],
    );
    assert_eq!(outs[0], bytes, "U8 identity byte equality");
}

#[test]
fn identity_bf16_count_four() {
    // Sub-32-bit dtype: bit_io reads 16 bits, byte-aligned slot.
    let mut g = NanoGraph::new();
    let inp = g.add_input_tensor(GlobalId(0), 4, NumericDType::BF16);
    let out = g.push_group(
        4,
        NumericDType::BF16,
        ScalarOp::Identity,
        vec![],
        vec![GroupInput::scalar(InputRef::affine(inp, 1))],
    );
    // BF16 raw bit patterns: 1.0, -1.0, +inf, -0.
    let raw: [u16; 4] = [0x3f80, 0xbf80, 0x7f80, 0x8000];
    let bytes: Vec<u8> = raw.iter().flat_map(|b| b.to_le_bytes()).collect();
    let outs = ab_test_bytes(
        &g,
        &[(inp, NumericDType::BF16, bytes.clone())],
        &[AtomRange {
            base: out,
            count: 4,
            dtype: NumericDType::BF16,
        }],
    );
    assert_eq!(outs[0], bytes, "BF16 identity byte equality");
}

#[test]
fn identity_bool_count_five() {
    // Sub-byte dtype: PoolEvalSpan expands pool_eval's bit-packed
    // output into byte-padded layout before scattering to buffer_ptrs,
    // matching the JIT pipeline's one-byte-per-bool convention.
    let mut g = NanoGraph::new();
    let inp = g.add_input_tensor(GlobalId(0), 5, NumericDType::BOOL);
    let out = g.push_group(
        5,
        NumericDType::BOOL,
        ScalarOp::Identity,
        vec![],
        vec![GroupInput::scalar(InputRef::affine(inp, 1))],
    );
    let bytes: Vec<u8> = vec![1, 0, 1, 1, 1];
    let outs = ab_test_bytes(
        &g,
        &[(inp, NumericDType::BOOL, bytes.clone())],
        &[AtomRange {
            base: out,
            count: 5,
            dtype: NumericDType::BOOL,
        }],
    );
    assert_eq!(outs[0], bytes, "BOOL identity byte equality");
}

#[test]
fn identity_f32_broadcast_input() {
    // Broadcast InputRef: every output atom reads the same source
    // atom. Tests the constant-bit-offset path inside the loop body.
    let mut g = NanoGraph::new();
    let inp = g.add_input_tensor(GlobalId(0), 1, NumericDType::F32);
    let out = g.push_group(
        4,
        NumericDType::F32,
        ScalarOp::Identity,
        vec![],
        vec![GroupInput::scalar(InputRef::Broadcast(inp))],
    );
    let bytes = f32_input_bytes(&[2.5]);
    let outs = ab_test_bytes(
        &g,
        &[(inp, NumericDType::F32, bytes.clone())],
        &[AtomRange {
            base: out,
            count: 4,
            dtype: NumericDType::F32,
        }],
    );
    let expected = f32_input_bytes(&[2.5, 2.5, 2.5, 2.5]);
    assert_eq!(outs[0], expected, "F32 broadcast identity");
}

#[test]
fn identity_f32_explicit_single() {
    // Explicit InputRef with one entry: the consumer count must
    // match `ids.len()` (since `InputRef::resolve(i)` indexes
    // `ids[i]`), so this is a count==1 group reading a single
    // explicit source atom. The variable-i loop is bypassed; the
    // body emits one inlined iteration with IterVar::Const.
    let mut g = NanoGraph::new();
    let inp = g.add_input_tensor(GlobalId(0), 1, NumericDType::F32);
    let out = g.push_group(
        1,
        NumericDType::F32,
        ScalarOp::Identity,
        vec![],
        vec![GroupInput::scalar(InputRef::Explicit(vec![inp]))],
    );
    let bytes = f32_input_bytes(&[-7.25]);
    let outs = ab_test_bytes(
        &g,
        &[(inp, NumericDType::F32, bytes.clone())],
        &[AtomRange {
            base: out,
            count: 1,
            dtype: NumericDType::F32,
        }],
    );
    assert_eq!(outs[0], bytes, "F32 explicit single identity");
}

// ─── N-d Strided identity tests ─────────────────────────────────────

#[test]
fn identity_f32_modular_input() {
    // Modular InputRef: output atom i reads source atom (i % 4).
    // 8 output atoms, 4 source atoms → each source read twice.
    let mut g = NanoGraph::new();
    let inp = g.add_input_tensor(GlobalId(0), 4, NumericDType::F32);
    let out = g.push_group(
        8,
        NumericDType::F32,
        ScalarOp::Identity,
        vec![],
        vec![GroupInput::scalar(InputRef::modular(inp, 1, 4))],
    );
    let src = [1.0_f32, 2.0, 3.0, 4.0];
    let bytes = f32_input_bytes(&src);
    let outs = ab_test_bytes(
        &g,
        &[(inp, NumericDType::F32, bytes)],
        &[AtomRange {
            base: out,
            count: 8,
            dtype: NumericDType::F32,
        }],
    );
    let expected = f32_input_bytes(&[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]);
    assert_eq!(outs[0], expected, "F32 modular identity");
}

#[test]
fn identity_f32_modular_non_power_of_two() {
    // Modular with modulus=3 (exercises the general div path).
    let mut g = NanoGraph::new();
    let inp = g.add_input_tensor(GlobalId(0), 3, NumericDType::F32);
    let out = g.push_group(
        9,
        NumericDType::F32,
        ScalarOp::Identity,
        vec![],
        vec![GroupInput::scalar(InputRef::modular(inp, 1, 3))],
    );
    let src = [10.0_f32, 20.0, 30.0];
    let bytes = f32_input_bytes(&src);
    let outs = ab_test_bytes(
        &g,
        &[(inp, NumericDType::F32, bytes)],
        &[AtomRange {
            base: out,
            count: 9,
            dtype: NumericDType::F32,
        }],
    );
    let expected = f32_input_bytes(&[10.0, 20.0, 30.0, 10.0, 20.0, 30.0, 10.0, 20.0, 30.0]);
    assert_eq!(outs[0], expected, "F32 modular mod3 identity");
}

#[test]
fn identity_f32_strided_broadcast_input() {
    // Strided broadcast: output atom i reads source atom (i / 4).
    // 8 output atoms from 2 source atoms, each repeated 4 times.
    let mut g = NanoGraph::new();
    let inp = g.add_input_tensor(GlobalId(0), 2, NumericDType::F32);
    let out = g.push_group(
        8,
        NumericDType::F32,
        ScalarOp::Identity,
        vec![],
        vec![GroupInput::scalar(InputRef::strided_broadcast(inp, 1, 4))],
    );
    let src = [5.0_f32, 9.0];
    let bytes = f32_input_bytes(&src);
    let outs = ab_test_bytes(
        &g,
        &[(inp, NumericDType::F32, bytes)],
        &[AtomRange {
            base: out,
            count: 8,
            dtype: NumericDType::F32,
        }],
    );
    let expected = f32_input_bytes(&[5.0, 5.0, 5.0, 5.0, 9.0, 9.0, 9.0, 9.0]);
    assert_eq!(outs[0], expected, "F32 strided_broadcast identity");
}

// ─── Multi-entry Explicit identity tests ────────────────────────────

#[test]
fn identity_f32_explicit_multi() {
    // Explicit multi-entry: 4-element output where each atom
    // reads a specific (shuffled) source atom.
    let mut g = NanoGraph::new();
    let inp = g.add_input_tensor(GlobalId(0), 4, NumericDType::F32);
    let ids = vec![
        crate::nano_graph::pattern::AtomId(inp.0 + 3),
        crate::nano_graph::pattern::AtomId(inp.0 + 0),
        crate::nano_graph::pattern::AtomId(inp.0 + 2),
        crate::nano_graph::pattern::AtomId(inp.0 + 1),
    ];
    let out = g.push_group(
        4,
        NumericDType::F32,
        ScalarOp::Identity,
        vec![],
        vec![GroupInput::scalar(InputRef::Explicit(ids))],
    );
    let src = [100.0_f32, 200.0, 300.0, 400.0];
    let bytes = f32_input_bytes(&src);
    let outs = ab_test_bytes(
        &g,
        &[(inp, NumericDType::F32, bytes)],
        &[AtomRange {
            base: out,
            count: 4,
            dtype: NumericDType::F32,
        }],
    );
    // ids = [3, 0, 2, 1] → src[3], src[0], src[2], src[1]
    let expected = f32_input_bytes(&[400.0, 100.0, 300.0, 200.0]);
    assert_eq!(outs[0], expected, "F32 explicit multi identity");
}

// ─── Cast tests ─────────────────────────────────────────────────────

#[test]
fn cast_f32_to_bf16() {
    // F32 → BF16: the format codec's RTNE rounding path.
    let mut g = NanoGraph::new();
    let inp = g.add_input_tensor(GlobalId(0), 4, NumericDType::F32);
    let out = g.push_group(
        4,
        NumericDType::BF16,
        ScalarOp::Cast { saturating: false },
        vec![],
        vec![GroupInput::scalar(InputRef::affine(inp, 1))],
    );
    let src = [1.0_f32, -1.0, 0.0, 3.140625]; // values exact in BF16
    let bytes = f32_input_bytes(&src);
    let outs = ab_test_bytes(
        &g,
        &[(inp, NumericDType::F32, bytes)],
        &[AtomRange {
            base: out,
            count: 4,
            dtype: NumericDType::BF16,
        }],
    );
    // BF16 is 2 bytes each → 8 bytes total.
    assert_eq!(outs[0].len(), 8, "BF16 output size");
}

#[test]
fn cast_bf16_to_f32() {
    // BF16 → F32: decode BF16 raw bits, encode as F32.
    let mut g = NanoGraph::new();
    let inp = g.add_input_tensor(GlobalId(0), 4, NumericDType::BF16);
    let out = g.push_group(
        4,
        NumericDType::F32,
        ScalarOp::Cast { saturating: false },
        vec![],
        vec![GroupInput::scalar(InputRef::affine(inp, 1))],
    );
    // BF16 for 1.0, -1.0, +inf, -0.0
    let raw: [u16; 4] = [0x3f80, 0xbf80, 0x7f80, 0x8000];
    let bytes: Vec<u8> = raw.iter().flat_map(|b| b.to_le_bytes()).collect();
    let outs = ab_test_bytes(
        &g,
        &[(inp, NumericDType::BF16, bytes)],
        &[AtomRange {
            base: out,
            count: 4,
            dtype: NumericDType::F32,
        }],
    );
    let expected = f32_input_bytes(&[1.0, -1.0, f32::INFINITY, -0.0]);
    assert_eq!(outs[0], expected, "BF16→F32 cast");
}

#[test]
fn cast_f32_to_f16_count_eight() {
    // F32 → F16: exercises the F16C encode path.
    let mut g = NanoGraph::new();
    let inp = g.add_input_tensor(GlobalId(0), 8, NumericDType::F32);
    let out = g.push_group(
        8,
        NumericDType::F16,
        ScalarOp::Cast { saturating: false },
        vec![],
        vec![GroupInput::scalar(InputRef::affine(inp, 1))],
    );
    let src: Vec<f32> = vec![
        0.0,
        1.0,
        -1.0,
        0.5,
        65504.0,
        -65504.0,
        f32::INFINITY,
        f32::NAN,
    ];
    let bytes = f32_input_bytes(&src);
    let outs = ab_test_bytes(
        &g,
        &[(inp, NumericDType::F32, bytes)],
        &[AtomRange {
            base: out,
            count: 8,
            dtype: NumericDType::F16,
        }],
    );
    assert_eq!(outs[0].len(), 16, "F16 output size = 8 * 2 bytes");
}

#[test]
fn cast_i32_to_i8_saturating() {
    // I32 → I8: saturating narrowing (values outside [-128, 127] clamp).
    let mut g = NanoGraph::new();
    let inp = g.add_input_tensor(GlobalId(0), 6, NumericDType::I32);
    let out = g.push_group(
        6,
        NumericDType::I8,
        ScalarOp::Cast { saturating: false },
        vec![],
        vec![GroupInput::scalar(InputRef::affine(inp, 1))],
    );
    let values: [i32; 6] = [0, 127, -128, 200, -200, 42];
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let outs = ab_test_bytes(
        &g,
        &[(inp, NumericDType::I32, bytes)],
        &[AtomRange {
            base: out,
            count: 6,
            dtype: NumericDType::I8,
        }],
    );
    assert_eq!(outs[0].len(), 6, "I8 output = 6 bytes");
}

#[test]
fn cast_u8_to_i32_widening() {
    // U8 → I32: widening (zero-extend U8, sign-extend to I32).
    let mut g = NanoGraph::new();
    let inp = g.add_input_tensor(GlobalId(0), 4, NumericDType::U8);
    let out = g.push_group(
        4,
        NumericDType::I32,
        ScalarOp::Cast { saturating: false },
        vec![],
        vec![GroupInput::scalar(InputRef::affine(inp, 1))],
    );
    let bytes: Vec<u8> = vec![0, 127, 128, 255];
    let outs = ab_test_bytes(
        &g,
        &[(inp, NumericDType::U8, bytes)],
        &[AtomRange {
            base: out,
            count: 4,
            dtype: NumericDType::I32,
        }],
    );
    // U8 values 0, 127, 128, 255 → I32 0, 127, 128, 255 (all positive)
    let expected: Vec<u8> = [0i32, 127, 128, 255]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    assert_eq!(outs[0], expected, "U8→I32 widening");
}

// ─── Literal test ───────────────────────────────────────────────────

#[test]
fn literal_f32_group() {
    // A Literal group produces a constant value. The buffer template
    // is pre-populated; the JIT emits no code for it. Another group
    // (Identity) reads the literal output to verify it's correct.
    use crate::numeric_scalar::NumericScalar;
    let mut g = NanoGraph::new();
    let lit = g.push_group(
        1,
        NumericDType::F32,
        ScalarOp::Literal(NumericScalar::from_f32(42.0)),
        vec![],
        vec![],
    );
    // Identity group reads the literal.
    let out = g.push_group(
        1,
        NumericDType::F32,
        ScalarOp::Identity,
        vec![],
        vec![GroupInput::scalar(InputRef::affine(lit, 1))],
    );
    let outs = ab_test_bytes(
        &g,
        &[],
        &[AtomRange {
            base: out,
            count: 1,
            dtype: NumericDType::F32,
        }],
    );
    let expected = f32_input_bytes(&[42.0]);
    assert_eq!(outs[0], expected, "Literal F32 42.0");
}

// ─── Cross-repr Cast tests ──────────────────────────────────────────

#[test]
fn cast_f32_to_i32() {
    // F32 → I32: float-to-int truncation with NaN → 0.
    let mut g = NanoGraph::new();
    let inp = g.add_input_tensor(GlobalId(0), 6, NumericDType::F32);
    let out = g.push_group(
        6,
        NumericDType::I32,
        ScalarOp::Cast { saturating: false },
        vec![],
        vec![GroupInput::scalar(InputRef::affine(inp, 1))],
    );
    let src = [1.5_f32, -2.7, 0.0, 127.9, -128.1, f32::NAN];
    let bytes = f32_input_bytes(&src);
    let outs = ab_test_bytes(
        &g,
        &[(inp, NumericDType::F32, bytes)],
        &[AtomRange {
            base: out,
            count: 6,
            dtype: NumericDType::I32,
        }],
    );
    assert_eq!(outs[0].len(), 24, "I32 output = 6 * 4 bytes");
}

#[test]
fn cast_i32_to_f32() {
    // I32 → F32: int-to-float conversion.
    let mut g = NanoGraph::new();
    let inp = g.add_input_tensor(GlobalId(0), 4, NumericDType::I32);
    let out = g.push_group(
        4,
        NumericDType::F32,
        ScalarOp::Cast { saturating: false },
        vec![],
        vec![GroupInput::scalar(InputRef::affine(inp, 1))],
    );
    let values: [i32; 4] = [0, 1, -1, 42];
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let outs = ab_test_bytes(
        &g,
        &[(inp, NumericDType::I32, bytes)],
        &[AtomRange {
            base: out,
            count: 4,
            dtype: NumericDType::F32,
        }],
    );
    let expected = f32_input_bytes(&[0.0, 1.0, -1.0, 42.0]);
    assert_eq!(outs[0], expected, "I32→F32 cast");
}

// ─── Float Binary op tests ──────────────────────────────────────────

fn binary_f32_test(op: ScalarBinOp, a_vals: &[f32], b_vals: &[f32]) -> Vec<Vec<u8>> {
    assert_eq!(a_vals.len(), b_vals.len());
    let n = a_vals.len() as u64;
    let mut g = NanoGraph::new();
    let inp_a = g.add_input_tensor(GlobalId(0), n, NumericDType::F32);
    let inp_b = g.add_input_tensor(GlobalId(1), n, NumericDType::F32);
    let out = g.push_group(
        n,
        NumericDType::F32,
        ScalarOp::Binary {
            op,
            compute_dtype: NumericDType::F32,
        },
        vec![],
        vec![
            GroupInput::scalar(InputRef::affine(inp_a, 1)),
            GroupInput::scalar(InputRef::affine(inp_b, 1)),
        ],
    );
    let a_bytes = f32_input_bytes(a_vals);
    let b_bytes = f32_input_bytes(b_vals);
    ab_test_bytes(
        &g,
        &[
            (inp_a, NumericDType::F32, a_bytes),
            (inp_b, NumericDType::F32, b_bytes),
        ],
        &[AtomRange {
            base: out,
            count: n,
            dtype: NumericDType::F32,
        }],
    )
}

use crate::nano_graph::ops::ScalarBinOp;

#[test]
fn binary_f32_add() {
    let outs = binary_f32_test(
        ScalarBinOp::Add,
        &[1.0, -1.0, 0.0, f32::INFINITY],
        &[2.0, 3.0, -0.0, 1.0],
    );
    let expected = f32_input_bytes(&[3.0, 2.0, 0.0, f32::INFINITY]);
    assert_eq!(outs[0], expected, "F32 Add");
}

#[test]
fn binary_f32_sub() {
    let outs = binary_f32_test(
        ScalarBinOp::Sub,
        &[5.0, 1.0, 0.0, -1.0],
        &[3.0, 2.0, 0.0, -1.0],
    );
    let expected = f32_input_bytes(&[2.0, -1.0, 0.0, 0.0]);
    assert_eq!(outs[0], expected, "F32 Sub");
}

#[test]
fn binary_f32_mul() {
    let outs = binary_f32_test(
        ScalarBinOp::Mul,
        &[2.0, -3.0, 0.0, f32::INFINITY],
        &[3.0, 4.0, 5.0, 0.0],
    );
    // inf * 0 = NaN — pool_eval agrees.
    assert_eq!(outs[0].len(), 16, "F32 Mul output size");
}

#[test]
fn binary_f32_div() {
    let outs = binary_f32_test(
        ScalarBinOp::Div,
        &[6.0, -6.0, 1.0, 0.0],
        &[3.0, 2.0, 0.0, 0.0],
    );
    // 1.0/0.0 = +inf, 0.0/0.0 = NaN — pool_eval agrees.
    assert_eq!(outs[0].len(), 16, "F32 Div output size");
}

#[test]
fn binary_f32_max_with_nan() {
    // Max with NaN: the non-NaN operand should win (minNum/maxNum).
    let outs = binary_f32_test(
        ScalarBinOp::Max,
        &[1.0, f32::NAN, 3.0, f32::NAN],
        &[2.0, 4.0, f32::NAN, f32::NAN],
    );
    assert_eq!(outs[0].len(), 16, "F32 Max output size");
}

#[test]
fn binary_f32_min_with_nan() {
    let outs = binary_f32_test(
        ScalarBinOp::Min,
        &[1.0, f32::NAN, 3.0, f32::NAN],
        &[2.0, 4.0, f32::NAN, f32::NAN],
    );
    assert_eq!(outs[0].len(), 16, "F32 Min output size");
}

#[test]
fn binary_f32_equal() {
    let outs = binary_f32_test(
        ScalarBinOp::Equal,
        &[1.0, 2.0, f32::NAN, 0.0],
        &[1.0, 3.0, f32::NAN, -0.0],
    );
    // NaN != NaN → 0.0, +0.0 == -0.0 → 1.0
    assert_eq!(outs[0].len(), 16, "F32 Equal output size");
}

#[test]
fn binary_f32_greater() {
    let outs = binary_f32_test(
        ScalarBinOp::Greater,
        &[2.0, 1.0, 1.0, f32::NAN],
        &[1.0, 2.0, 1.0, 1.0],
    );
    assert_eq!(outs[0].len(), 16, "F32 Greater output size");
}

// ─── Float Unary op tests ───────────────────────────────────────────

use crate::nano_graph::ops::ScalarUnaryOp;

fn unary_f32_test(op: ScalarUnaryOp, vals: &[f32]) -> Vec<Vec<u8>> {
    let n = vals.len() as u64;
    let mut g = NanoGraph::new();
    let inp = g.add_input_tensor(GlobalId(0), n, NumericDType::F32);
    let out = g.push_group(
        n,
        NumericDType::F32,
        ScalarOp::Unary {
            op,
            compute_dtype: NumericDType::F32,
        },
        vec![],
        vec![GroupInput::scalar(InputRef::affine(inp, 1))],
    );
    let bytes = f32_input_bytes(vals);
    ab_test_bytes(
        &g,
        &[(inp, NumericDType::F32, bytes)],
        &[AtomRange {
            base: out,
            count: n,
            dtype: NumericDType::F32,
        }],
    )
}

#[test]
fn unary_f32_neg() {
    let outs = unary_f32_test(ScalarUnaryOp::Neg, &[1.0, -1.0, 0.0, f32::INFINITY]);
    let expected = f32_input_bytes(&[-1.0, 1.0, -0.0, f32::NEG_INFINITY]);
    assert_eq!(outs[0], expected, "F32 Neg");
}

#[test]
fn unary_f32_abs() {
    let outs = unary_f32_test(ScalarUnaryOp::Abs, &[-3.0, 3.0, -0.0, f32::NEG_INFINITY]);
    let expected = f32_input_bytes(&[3.0, 3.0, 0.0, f32::INFINITY]);
    assert_eq!(outs[0], expected, "F32 Abs");
}

#[test]
fn unary_f32_sqrt() {
    let outs = unary_f32_test(ScalarUnaryOp::Sqrt, &[4.0, 9.0, 0.0, 1.0]);
    let expected = f32_input_bytes(&[2.0, 3.0, 0.0, 1.0]);
    assert_eq!(outs[0], expected, "F32 Sqrt");
}

#[test]
fn unary_f32_reciprocal() {
    let outs = unary_f32_test(ScalarUnaryOp::Reciprocal, &[2.0, 0.5, -4.0, 1.0]);
    let expected = f32_input_bytes(&[0.5, 2.0, -0.25, 1.0]);
    assert_eq!(outs[0], expected, "F32 Reciprocal");
}

#[test]
fn unary_f32_floor() {
    let outs = unary_f32_test(ScalarUnaryOp::Floor, &[1.7, -1.7, 2.0, -0.5]);
    let expected = f32_input_bytes(&[1.0, -2.0, 2.0, -1.0]);
    assert_eq!(outs[0], expected, "F32 Floor");
}

#[test]
fn unary_f32_exp() {
    // Exp via libm trampoline.
    let outs = unary_f32_test(ScalarUnaryOp::Exp, &[0.0, 1.0, -1.0, 2.0]);
    assert_eq!(outs[0].len(), 16, "F32 Exp output size");
}

#[test]
fn unary_f32_tanh() {
    // Tanh via libm trampoline.
    let outs = unary_f32_test(ScalarUnaryOp::Tanh, &[0.0, 1.0, -1.0, 100.0]);
    assert_eq!(outs[0].len(), 16, "F32 Tanh output size");
}

// ─── Int Binary op tests ────────────────────────────────────────────

fn binary_i32_test(op: ScalarBinOp, a_vals: &[i32], b_vals: &[i32]) -> Vec<Vec<u8>> {
    assert_eq!(a_vals.len(), b_vals.len());
    let n = a_vals.len() as u64;
    let mut g = NanoGraph::new();
    let inp_a = g.add_input_tensor(GlobalId(0), n, NumericDType::I32);
    let inp_b = g.add_input_tensor(GlobalId(1), n, NumericDType::I32);
    let out = g.push_group(
        n,
        NumericDType::I32,
        ScalarOp::Binary {
            op,
            compute_dtype: NumericDType::I32,
        },
        vec![],
        vec![
            GroupInput::scalar(InputRef::affine(inp_a, 1)),
            GroupInput::scalar(InputRef::affine(inp_b, 1)),
        ],
    );
    let a_bytes: Vec<u8> = a_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    let b_bytes: Vec<u8> = b_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    ab_test_bytes(
        &g,
        &[
            (inp_a, NumericDType::I32, a_bytes),
            (inp_b, NumericDType::I32, b_bytes),
        ],
        &[AtomRange {
            base: out,
            count: n,
            dtype: NumericDType::I32,
        }],
    )
}

#[test]
fn binary_i32_add_wrapping() {
    let outs = binary_i32_test(
        ScalarBinOp::Add,
        &[1, -1, i32::MAX, i32::MIN],
        &[2, -2, 1, -1],
    );
    // i32::MAX + 1 wraps to i32::MIN, i32::MIN + (-1) wraps to i32::MAX
    assert_eq!(outs[0].len(), 16, "I32 Add output size");
}

#[test]
fn binary_i32_mul_wrapping() {
    let outs = binary_i32_test(ScalarBinOp::Mul, &[3, -3, 100000, 0], &[7, 7, 100000, 42]);
    assert_eq!(outs[0].len(), 16, "I32 Mul output size");
}

#[test]
fn binary_i32_div() {
    // Includes div-by-zero → 0.
    let outs = binary_i32_test(ScalarBinOp::Div, &[10, -10, 42, i32::MIN], &[3, 3, 0, -1]);
    assert_eq!(outs[0].len(), 16, "I32 Div output size");
}

#[test]
fn binary_i32_bitwise_and() {
    let outs = binary_i32_test(
        ScalarBinOp::BitwiseAnd,
        &[0xFF, 0x0F, -1, 0],
        &[0x0F, 0xFF, 0x55555555, -1],
    );
    let expected: Vec<u8> = [0x0Fi32, 0x0F, 0x55555555, 0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    assert_eq!(outs[0], expected, "I32 BitwiseAnd");
}

#[test]
fn binary_i32_shift_left() {
    let outs = binary_i32_test(ScalarBinOp::BitShiftLeft, &[1, 1, -1, 0xFF], &[0, 8, 16, 4]);
    assert_eq!(outs[0].len(), 16, "I32 ShiftLeft output size");
}

#[test]
fn binary_i32_greater() {
    // Int comparison: result is 0 or 1.
    let mut g = NanoGraph::new();
    let inp_a = g.add_input_tensor(GlobalId(0), 4, NumericDType::I32);
    let inp_b = g.add_input_tensor(GlobalId(1), 4, NumericDType::I32);
    // Comparison with I32 compute outputs Bool.
    let out = g.push_group(
        4,
        NumericDType::I32,
        ScalarOp::Binary {
            op: ScalarBinOp::Greater,
            compute_dtype: NumericDType::I32,
        },
        vec![],
        vec![
            GroupInput::scalar(InputRef::affine(inp_a, 1)),
            GroupInput::scalar(InputRef::affine(inp_b, 1)),
        ],
    );
    let a: Vec<u8> = [5i32, 3, 3, -1]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let b: Vec<u8> = [3i32, 5, 3, 1]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let outs = ab_test_bytes(
        &g,
        &[(inp_a, NumericDType::I32, a), (inp_b, NumericDType::I32, b)],
        &[AtomRange {
            base: out,
            count: 4,
            dtype: NumericDType::I32,
        }],
    );
    assert_eq!(outs[0].len(), 16, "I32 Greater output size");
}

#[test]
fn binary_i32_pow() {
    // Pow: saturating semantics. Includes negative exponent → 0.
    let outs = binary_i32_test(ScalarBinOp::Pow, &[2, 3, 10, 2], &[10, 3, 3, -1]);
    // 2^10 = 1024, 3^3 = 27, 10^3 = 1000, 2^(-1) = 0
    let expected: Vec<u8> = [1024i32, 27, 1000, 0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    assert_eq!(outs[0], expected, "I32 Pow");
}

// ─── Select tests ───────────────────────────────────────────────────

#[test]
fn select_f32_with_float_cond() {
    // cond ? x : y where cond is F32.
    // cond = [1.0, 0.0, -0.0, NaN] → truthy = [T, F, F, T]
    let mut g = NanoGraph::new();
    let cond = g.add_input_tensor(GlobalId(0), 4, NumericDType::F32);
    let x = g.add_input_tensor(GlobalId(1), 4, NumericDType::F32);
    let y = g.add_input_tensor(GlobalId(2), 4, NumericDType::F32);
    let out = g.push_group(
        4,
        NumericDType::F32,
        ScalarOp::Select,
        vec![],
        vec![
            GroupInput::scalar(InputRef::affine(cond, 1)),
            GroupInput::scalar(InputRef::affine(x, 1)),
            GroupInput::scalar(InputRef::affine(y, 1)),
        ],
    );
    let c_bytes = f32_input_bytes(&[1.0, 0.0, -0.0, f32::NAN]);
    let x_bytes = f32_input_bytes(&[10.0, 20.0, 30.0, 40.0]);
    let y_bytes = f32_input_bytes(&[100.0, 200.0, 300.0, 400.0]);
    let outs = ab_test_bytes(
        &g,
        &[
            (cond, NumericDType::F32, c_bytes),
            (x, NumericDType::F32, x_bytes),
            (y, NumericDType::F32, y_bytes),
        ],
        &[AtomRange {
            base: out,
            count: 4,
            dtype: NumericDType::F32,
        }],
    );
    // truthy=[T,F,F,T] → [x0, y1, y2, x3] = [10, 200, 300, 40]
    let expected = f32_input_bytes(&[10.0, 200.0, 300.0, 40.0]);
    assert_eq!(outs[0], expected, "Select F32 with float cond");
}

#[test]
fn select_f32_with_int_cond() {
    // cond is I32: [1, 0, -1, 42] → truthy = [T, F, T, T]
    let mut g = NanoGraph::new();
    let cond = g.add_input_tensor(GlobalId(0), 4, NumericDType::I32);
    let x = g.add_input_tensor(GlobalId(1), 4, NumericDType::F32);
    let y = g.add_input_tensor(GlobalId(2), 4, NumericDType::F32);
    let out = g.push_group(
        4,
        NumericDType::F32,
        ScalarOp::Select,
        vec![],
        vec![
            GroupInput::scalar(InputRef::affine(cond, 1)),
            GroupInput::scalar(InputRef::affine(x, 1)),
            GroupInput::scalar(InputRef::affine(y, 1)),
        ],
    );
    let c_bytes: Vec<u8> = [1i32, 0, -1, 42]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let x_bytes = f32_input_bytes(&[1.0, 2.0, 3.0, 4.0]);
    let y_bytes = f32_input_bytes(&[10.0, 20.0, 30.0, 40.0]);
    let outs = ab_test_bytes(
        &g,
        &[
            (cond, NumericDType::I32, c_bytes),
            (x, NumericDType::F32, x_bytes),
            (y, NumericDType::F32, y_bytes),
        ],
        &[AtomRange {
            base: out,
            count: 4,
            dtype: NumericDType::F32,
        }],
    );
    let expected = f32_input_bytes(&[1.0, 20.0, 3.0, 4.0]);
    assert_eq!(outs[0], expected, "Select F32 with int cond");
}

// ─── Int Unary op tests ─────────────────────────────────────────────

#[test]
fn unary_i32_neg() {
    let mut g = NanoGraph::new();
    let inp = g.add_input_tensor(GlobalId(0), 4, NumericDType::I32);
    let out = g.push_group(
        4,
        NumericDType::I32,
        ScalarOp::Unary {
            op: ScalarUnaryOp::Neg,
            compute_dtype: NumericDType::I32,
        },
        vec![],
        vec![GroupInput::scalar(InputRef::affine(inp, 1))],
    );
    let bytes: Vec<u8> = [1i32, -1, 0, i32::MAX]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let outs = ab_test_bytes(
        &g,
        &[(inp, NumericDType::I32, bytes)],
        &[AtomRange {
            base: out,
            count: 4,
            dtype: NumericDType::I32,
        }],
    );
    assert_eq!(outs[0].len(), 16, "I32 Neg output size");
}

#[test]
fn unary_i32_abs() {
    let mut g = NanoGraph::new();
    let inp = g.add_input_tensor(GlobalId(0), 4, NumericDType::I32);
    let out = g.push_group(
        4,
        NumericDType::I32,
        ScalarOp::Unary {
            op: ScalarUnaryOp::Abs,
            compute_dtype: NumericDType::I32,
        },
        vec![],
        vec![GroupInput::scalar(InputRef::affine(inp, 1))],
    );
    let bytes: Vec<u8> = [5i32, -5, 0, -1]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let outs = ab_test_bytes(
        &g,
        &[(inp, NumericDType::I32, bytes)],
        &[AtomRange {
            base: out,
            count: 4,
            dtype: NumericDType::I32,
        }],
    );
    assert_eq!(outs[0].len(), 16, "I32 Abs output size");
}

#[test]
fn unary_i32_bitwise_not() {
    let mut g = NanoGraph::new();
    let inp = g.add_input_tensor(GlobalId(0), 3, NumericDType::I32);
    let out = g.push_group(
        3,
        NumericDType::I32,
        ScalarOp::Unary {
            op: ScalarUnaryOp::BitwiseNot,
            compute_dtype: NumericDType::I32,
        },
        vec![],
        vec![GroupInput::scalar(InputRef::affine(inp, 1))],
    );
    let bytes: Vec<u8> = [0i32, -1, 0x55555555]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let outs = ab_test_bytes(
        &g,
        &[(inp, NumericDType::I32, bytes)],
        &[AtomRange {
            base: out,
            count: 3,
            dtype: NumericDType::I32,
        }],
    );
    assert_eq!(outs[0].len(), 12, "I32 BitwiseNot output size");
}

// ─── Reduce tests ───────────────────────────────────────────────────

use crate::nano_graph::ops::ReduceKind;

#[test]
fn reduce_sum_f32() {
    // ReduceSum: 4 inputs → 1 output.
    let mut g = NanoGraph::new();
    let inp = g.add_input_tensor(GlobalId(0), 4, NumericDType::F32);
    let out = g.push_group(
        1,
        NumericDType::F32,
        ScalarOp::Reduce {
            kind: ReduceKind::Sum,
            reduce_count: 4,
            reduce_stride: 1,
            compute_dtype: NumericDType::F32,
        },
        vec![],
        vec![GroupInput::scalar(InputRef::affine(inp, 1))],
    );
    let bytes = f32_input_bytes(&[1.0, 2.0, 3.0, 4.0]);
    let outs = ab_test_bytes(
        &g,
        &[(inp, NumericDType::F32, bytes)],
        &[AtomRange {
            base: out,
            count: 1,
            dtype: NumericDType::F32,
        }],
    );
    let expected = f32_input_bytes(&[10.0]);
    assert_eq!(outs[0], expected, "ReduceSum F32 [1,2,3,4] = 10");
}

#[test]
fn reduce_max_f32_with_nan() {
    // ReduceMax: NaN should be ignored per minNum/maxNum semantics.
    let mut g = NanoGraph::new();
    let inp = g.add_input_tensor(GlobalId(0), 4, NumericDType::F32);
    let out = g.push_group(
        1,
        NumericDType::F32,
        ScalarOp::Reduce {
            kind: ReduceKind::Max,
            reduce_count: 4,
            reduce_stride: 1,
            compute_dtype: NumericDType::F32,
        },
        vec![],
        vec![GroupInput::scalar(InputRef::affine(inp, 1))],
    );
    let bytes = f32_input_bytes(&[1.0, f32::NAN, 3.0, 2.0]);
    let outs = ab_test_bytes(
        &g,
        &[(inp, NumericDType::F32, bytes)],
        &[AtomRange {
            base: out,
            count: 1,
            dtype: NumericDType::F32,
        }],
    );
    assert_eq!(outs[0].len(), 4, "ReduceMax F32 output size");
}

#[test]
fn reduce_sum_f32_multiple_outputs() {
    // 2 output atoms, each reducing 3 input atoms with stride=1.
    // Input: [a0, a1, a2, a3, a4, a5]
    // Output[0] = a0+a1+a2, Output[1] = a1+a2+a3
    // Using modular input to map output i → input base i.
    let mut g = NanoGraph::new();
    let inp = g.add_input_tensor(GlobalId(0), 6, NumericDType::F32);
    let out = g.push_group(
        2,
        NumericDType::F32,
        ScalarOp::Reduce {
            kind: ReduceKind::Sum,
            reduce_count: 3,
            reduce_stride: 1,
            compute_dtype: NumericDType::F32,
        },
        vec![],
        vec![GroupInput::scalar(InputRef::affine(inp, 1))],
    );
    let bytes = f32_input_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let outs = ab_test_bytes(
        &g,
        &[(inp, NumericDType::F32, bytes)],
        &[AtomRange {
            base: out,
            count: 2,
            dtype: NumericDType::F32,
        }],
    );
    // out[0] = inp[0]+inp[1]+inp[2] = 6.0
    // out[1] = inp[1]+inp[2]+inp[3] = 9.0
    let expected = f32_input_bytes(&[6.0, 9.0]);
    assert_eq!(outs[0], expected, "ReduceSum F32 multi-output");
}

#[test]
fn reduce_prod_f32() {
    let mut g = NanoGraph::new();
    let inp = g.add_input_tensor(GlobalId(0), 4, NumericDType::F32);
    let out = g.push_group(
        1,
        NumericDType::F32,
        ScalarOp::Reduce {
            kind: ReduceKind::Prod,
            reduce_count: 4,
            reduce_stride: 1,
            compute_dtype: NumericDType::F32,
        },
        vec![],
        vec![GroupInput::scalar(InputRef::affine(inp, 1))],
    );
    let bytes = f32_input_bytes(&[1.0, 2.0, 3.0, 4.0]);
    let outs = ab_test_bytes(
        &g,
        &[(inp, NumericDType::F32, bytes)],
        &[AtomRange {
            base: out,
            count: 1,
            dtype: NumericDType::F32,
        }],
    );
    let expected = f32_input_bytes(&[24.0]);
    assert_eq!(outs[0], expected, "ReduceProd F32 [1,2,3,4] = 24");
}

// ─── IndirectLoad tests ─────────────────────────────────────────────

#[test]
fn indirect_load_f32_gather() {
    // Table of 4 F32 values. Index input selects which one to output.
    use crate::numeric_scalar::NumericScalar;
    let mut g = NanoGraph::new();
    // Table: 4 literals [10.0, 20.0, 30.0, 40.0]
    let t0 = g.push_group(
        1,
        NumericDType::F32,
        ScalarOp::Literal(NumericScalar::from_f32(10.0)),
        vec![],
        vec![],
    );
    let t1 = g.push_group(
        1,
        NumericDType::F32,
        ScalarOp::Literal(NumericScalar::from_f32(20.0)),
        vec![],
        vec![],
    );
    let t2 = g.push_group(
        1,
        NumericDType::F32,
        ScalarOp::Literal(NumericScalar::from_f32(30.0)),
        vec![],
        vec![],
    );
    let _t3 = g.push_group(
        1,
        NumericDType::F32,
        ScalarOp::Literal(NumericScalar::from_f32(40.0)),
        vec![],
        vec![],
    );

    // Index input: [2, 0, 3, 1] as I32.
    let idx = g.add_input_tensor(GlobalId(0), 4, NumericDType::I32);

    // IndirectLoad: table starts at t0, 4 entries (t0..t3).
    let out = g.push_group(
        4,
        NumericDType::F32,
        ScalarOp::IndirectLoad {
            table_base: t0,
            index_range: 4,
        },
        vec![],
        vec![GroupInput::scalar(InputRef::affine(idx, 1))],
    );

    let idx_bytes: Vec<u8> = [2i32, 0, 3, 1]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let outs = ab_test_bytes(
        &g,
        &[(idx, NumericDType::I32, idx_bytes)],
        &[AtomRange {
            base: out,
            count: 4,
            dtype: NumericDType::F32,
        }],
    );
    // indices [2, 0, 3, 1] → table[2]=30, table[0]=10, table[3]=40, table[1]=20
    let expected = f32_input_bytes(&[30.0, 10.0, 40.0, 20.0]);
    assert_eq!(outs[0], expected, "IndirectLoad F32 gather");
}

// ─── Reduce-fold inline regression tests ────────────────────────────

// Regression guards for the `emit_op_compute` slot-contract bug that
// surfaced in RWKV's compiled forward pass (sum-of-cast). The layout
// heuristic folds a pure-scalar producer into the reduce's k-loop
// whenever: producer has a single consumer that is a Reduce with
// `reduce_stride == 1`, the consumer reads the producer via a
// Strided ref with `dim_strides = [reduce_count]`, and the atom
// counts line up. Construct that shape explicitly with a Cast
// producer so every path through `emit_reduce_body`'s inline branch
// (Int repr, Float repr) is exercised. Pre-fix these tests panicked
// inside `ab_test_bytes` with a byte mismatch vs `pool_eval`.

#[test]
fn reduce_sum_inline_cast_int() {
    // Cast i64 → i32 folded into Sum. Int repr: the pre-fix bug
    // overwrote rax (Cast result) with rdx before accumulation,
    // yielding garbage sums.
    let mut g = NanoGraph::new();
    let inp = g.add_input_tensor(GlobalId(0), 6, NumericDType::I64);
    let cast = g.push_group(
        6,
        NumericDType::I32,
        ScalarOp::Cast { saturating: false },
        vec![],
        vec![GroupInput::scalar(InputRef::affine(inp, 1))],
    );
    // reduce_count = 3, output count = 2, consumer stride = reduce_count = 3.
    // The layout inlinability criteria require exactly this shape.
    let out = g.push_group(
        2,
        NumericDType::I32,
        ScalarOp::Reduce {
            kind: ReduceKind::Sum,
            reduce_count: 3,
            reduce_stride: 1,
            compute_dtype: NumericDType::I32,
        },
        vec![],
        vec![GroupInput::scalar(InputRef::affine(cast, 3))],
    );

    let values: [i64; 6] = [10, 20, 30, 40, 50, 60];
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let outs = ab_test_bytes(
        &g,
        &[(inp, NumericDType::I64, bytes)],
        &[AtomRange {
            base: out,
            count: 2,
            dtype: NumericDType::I32,
        }],
    );
    // Expected: [10+20+30, 40+50+60] = [60, 150].
    let expected: Vec<u8> = [60i32, 150].iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(outs[0], expected, "Sum(Cast<i64→i32>) reduce-fold inline");
}

#[test]
fn reduce_sum_inline_cast_float() {
    // Cast f32 → bf16 folded into Sum (bf16 compute → F32 compute repr).
    // Float repr: pre-fix the post-inline move clobbered xmm0 (the
    // Cast result) with xmm2, yielding garbage sums.
    let mut g = NanoGraph::new();
    let inp = g.add_input_tensor(GlobalId(0), 6, NumericDType::F32);
    let cast = g.push_group(
        6,
        NumericDType::BF16,
        ScalarOp::Cast { saturating: false },
        vec![],
        vec![GroupInput::scalar(InputRef::affine(inp, 1))],
    );
    let out = g.push_group(
        2,
        NumericDType::BF16,
        ScalarOp::Reduce {
            kind: ReduceKind::Sum,
            reduce_count: 3,
            reduce_stride: 1,
            compute_dtype: NumericDType::BF16,
        },
        vec![],
        vec![GroupInput::scalar(InputRef::affine(cast, 3))],
    );

    // Values chosen to be bf16-exactly-representable so the
    // byte-equal assertion against pool_eval is deterministic.
    let values: [f32; 6] = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0];
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let outs = ab_test_bytes(
        &g,
        &[(inp, NumericDType::F32, bytes)],
        &[AtomRange {
            base: out,
            count: 2,
            dtype: NumericDType::BF16,
        }],
    );
    // ab_test_bytes already asserts pool_eval == x86_jit; the
    // relaxation here is that bf16 doesn't round-trip every f32, so
    // we don't recompute an expected ourselves — equality between
    // backends is the regression signal.
    assert_eq!(outs[0].len(), 4, "bf16 output = 2 atoms × 2 bytes");
}

// ─── SymReduce tests ────────────────────────────────────────────────

/// SymReduce(Sum, axis=0) over a sym input tensor with 2 atoms × seq_len.
/// At seq_len=3 and data [[1,2,3],[4,5,6]] (atom-major, sym-innermost)
/// the expected output is [6.0, 15.0] — one scalar per atom.
#[test]
fn sym_reduce_sum_f32_1_sym_to_0() {
    let mut g = NanoGraph::new();
    let seq = g.bounded_graph_constant("seq_len", 4);

    let inp = g.add_input_tensor(GlobalId(0), 2, NumericDType::F32);
    let out = g.push_group(
        2,
        NumericDType::F32,
        ScalarOp::SymReduce {
            kind: crate::nano_graph::ops::ReduceKind::Sum,
            axis: 0,
            compute_dtype: NumericDType::F32,
        },
        vec![], // consumer has no sym
        vec![GroupInput {
            input_ref: InputRef::affine(inp, 1),
            sym_dim_map: vec![], // 0 consumer sym axes
        }],
    );

    // Input bytes: atom 0 = [1.0, 2.0, 3.0], atom 1 = [4.0, 5.0, 6.0].
    // Runtime-tight TAMI layout: 2 atoms × 3 sym × 4 bytes = 24 bytes.
    let bytes = f32_input_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let mut bindings = std::collections::HashMap::new();
    bindings.insert(seq, 3u64);

    let outs = ab_test_bytes_sym(
        &g,
        &[(inp, NumericDType::F32, bytes)],
        &[AtomRange {
            base: out,
            count: 2,
            dtype: NumericDType::F32,
        }],
        &[(inp, vec![seq])],
        &bindings,
    );

    let expected = f32_input_bytes(&[6.0, 15.0]);
    assert_eq!(outs[0], expected, "SymReduce Sum F32 per-atom accumulation");
}

#[test]
fn sym_reduce_max_f32_1_sym_to_0() {
    let mut g = NanoGraph::new();
    let seq = g.bounded_graph_constant("seq_len", 4);

    let inp = g.add_input_tensor(GlobalId(0), 2, NumericDType::F32);
    let out = g.push_group(
        2,
        NumericDType::F32,
        ScalarOp::SymReduce {
            kind: crate::nano_graph::ops::ReduceKind::Max,
            axis: 0,
            compute_dtype: NumericDType::F32,
        },
        vec![],
        vec![GroupInput {
            input_ref: InputRef::affine(inp, 1),
            sym_dim_map: vec![],
        }],
    );

    let bytes = f32_input_bytes(&[1.0, 3.0, 2.0, 5.0, 4.0, 6.0]);
    let mut bindings = std::collections::HashMap::new();
    bindings.insert(seq, 3u64);

    let outs = ab_test_bytes_sym(
        &g,
        &[(inp, NumericDType::F32, bytes)],
        &[AtomRange {
            base: out,
            count: 2,
            dtype: NumericDType::F32,
        }],
        &[(inp, vec![seq])],
        &bindings,
    );

    let expected = f32_input_bytes(&[3.0, 6.0]);
    assert_eq!(outs[0], expected, "SymReduce Max F32 per-atom");
}

#[test]
fn sym_reduce_sum_i32_1_sym_to_0() {
    let mut g = NanoGraph::new();
    let seq = g.bounded_graph_constant("seq_len", 4);

    let inp = g.add_input_tensor(GlobalId(0), 2, NumericDType::I32);
    let out = g.push_group(
        2,
        NumericDType::I32,
        ScalarOp::SymReduce {
            kind: crate::nano_graph::ops::ReduceKind::Sum,
            axis: 0,
            compute_dtype: NumericDType::I32,
        },
        vec![],
        vec![GroupInput {
            input_ref: InputRef::affine(inp, 1),
            sym_dim_map: vec![],
        }],
    );

    // Atom 0: [10, 20, 30] → 60. Atom 1: [-1, -2, -3] → -6.
    let bytes: Vec<u8> = [10i32, 20, 30, -1, -2, -3]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let mut bindings = std::collections::HashMap::new();
    bindings.insert(seq, 3u64);

    let outs = ab_test_bytes_sym(
        &g,
        &[(inp, NumericDType::I32, bytes)],
        &[AtomRange {
            base: out,
            count: 2,
            dtype: NumericDType::I32,
        }],
        &[(inp, vec![seq])],
        &bindings,
    );

    let expected: Vec<u8> = [60i32, -6].iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(outs[0], expected, "SymReduce Sum I32 per-atom");
}

/// N=2 producer, axis=0. Consumer keeps the second sym axis.
/// Input 1 atom × [a=2, b=3]: data [1..6] at sym_flat = a*3 + b.
/// Reducing axis=0 (a) gives output[b] = prod[0*3+b] + prod[1*3+b].
/// Expected: [1+4, 2+5, 3+6] = [5, 7, 9].
#[test]
fn sym_reduce_sum_f32_2_sym_to_1_axis0() {
    let mut g = NanoGraph::new();
    let a = g.bounded_graph_constant("a", 4);
    let b = g.bounded_graph_constant("b", 4);

    let inp = g.add_input_tensor(GlobalId(0), 1, NumericDType::F32);
    let out = g.push_group(
        1,
        NumericDType::F32,
        ScalarOp::SymReduce {
            kind: crate::nano_graph::ops::ReduceKind::Sum,
            axis: 0,
            compute_dtype: NumericDType::F32,
        },
        vec![b], // consumer keeps the inner axis (producer's axis 1)
        vec![GroupInput {
            input_ref: InputRef::affine(inp, 1),
            sym_dim_map: vec![crate::nano_graph::pattern::SymDimMap::Identity(1)],
        }],
    );

    let bytes = f32_input_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let mut bindings = std::collections::HashMap::new();
    bindings.insert(a, 2u64);
    bindings.insert(b, 3u64);

    let outs = ab_test_bytes_sym(
        &g,
        &[(inp, NumericDType::F32, bytes)],
        &[AtomRange {
            base: out,
            count: 1,
            dtype: NumericDType::F32,
        }],
        &[(inp, vec![a, b])],
        &bindings,
    );

    let expected = f32_input_bytes(&[5.0, 7.0, 9.0]);
    assert_eq!(outs[0], expected, "SymReduce N=2 axis=0 over a, retains b");
}

/// N=2 producer, axis=1. Consumer keeps the first sym axis.
/// Input 1 atom × [a=2, b=3]: data [1..6]. Reducing axis=1 (b) gives
/// output[a] = sum over b of prod[a*3+b]. Expected: [1+2+3, 4+5+6] =
/// [6, 15].
#[test]
fn sym_reduce_sum_f32_2_sym_to_1_axis_last() {
    let mut g = NanoGraph::new();
    let a = g.bounded_graph_constant("a", 4);
    let b = g.bounded_graph_constant("b", 4);

    let inp = g.add_input_tensor(GlobalId(0), 1, NumericDType::F32);
    let out = g.push_group(
        1,
        NumericDType::F32,
        ScalarOp::SymReduce {
            kind: crate::nano_graph::ops::ReduceKind::Sum,
            axis: 1,
            compute_dtype: NumericDType::F32,
        },
        vec![a],
        vec![GroupInput {
            input_ref: InputRef::affine(inp, 1),
            sym_dim_map: vec![crate::nano_graph::pattern::SymDimMap::Identity(0)],
        }],
    );

    let bytes = f32_input_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let mut bindings = std::collections::HashMap::new();
    bindings.insert(a, 2u64);
    bindings.insert(b, 3u64);

    let outs = ab_test_bytes_sym(
        &g,
        &[(inp, NumericDType::F32, bytes)],
        &[AtomRange {
            base: out,
            count: 1,
            dtype: NumericDType::F32,
        }],
        &[(inp, vec![a, b])],
        &bindings,
    );

    let expected = f32_input_bytes(&[6.0, 15.0]);
    assert_eq!(outs[0], expected, "SymReduce N=2 axis=1 over b, retains a");
}

/// N=3 producer, axis=1 (middle). Consumer keeps first + third.
/// Input 1 atom × [a=2, b=2, c=2]: data [1..8] at sym_flat = a*4+b*2+c.
/// consumer[a*2+c] = sum over b of prod[a*4+b*2+c]. Expected:
/// [1+3, 2+4, 5+7, 6+8] = [4, 6, 12, 14]. Exercises the runtime
/// `div` path for S_outer / S_inner.
#[test]
fn sym_reduce_sum_f32_3_sym_to_2_axis_middle() {
    let mut g = NanoGraph::new();
    let a = g.bounded_graph_constant("a", 4);
    let b = g.bounded_graph_constant("b", 4);
    let c = g.bounded_graph_constant("c", 4);

    let inp = g.add_input_tensor(GlobalId(0), 1, NumericDType::F32);
    let out = g.push_group(
        1,
        NumericDType::F32,
        ScalarOp::SymReduce {
            kind: crate::nano_graph::ops::ReduceKind::Sum,
            axis: 1,
            compute_dtype: NumericDType::F32,
        },
        vec![a, c], // axis=1 (b) removed; keep a, c
        vec![GroupInput {
            input_ref: InputRef::affine(inp, 1),
            sym_dim_map: vec![
                crate::nano_graph::pattern::SymDimMap::Identity(0),
                crate::nano_graph::pattern::SymDimMap::Identity(2),
            ],
        }],
    );

    let bytes = f32_input_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let mut bindings = std::collections::HashMap::new();
    bindings.insert(a, 2u64);
    bindings.insert(b, 2u64);
    bindings.insert(c, 2u64);

    let outs = ab_test_bytes_sym(
        &g,
        &[(inp, NumericDType::F32, bytes)],
        &[AtomRange {
            base: out,
            count: 1,
            dtype: NumericDType::F32,
        }],
        &[(inp, vec![a, b, c])],
        &bindings,
    );

    let expected = f32_input_bytes(&[4.0, 6.0, 12.0, 14.0]);
    assert_eq!(
        outs[0], expected,
        "SymReduce N=3 axis=1 (middle) over b, retains a,c"
    );
}

/// N=2 axis=0 with a larger atom count — exercises the outer atom
/// loop alongside the consumer-sym inner loop.
#[test]
fn sym_reduce_sum_f32_2_sym_atoms_plus_sym() {
    let mut g = NanoGraph::new();
    let a = g.bounded_graph_constant("a", 4);
    let b = g.bounded_graph_constant("b", 4);

    // 2 atoms × [a=2, b=3]. Per-atom flat: a*3 + b. Atom 0 = [1..6],
    // atom 1 = [11..16].
    let inp = g.add_input_tensor(GlobalId(0), 2, NumericDType::F32);
    let out = g.push_group(
        2,
        NumericDType::F32,
        ScalarOp::SymReduce {
            kind: crate::nano_graph::ops::ReduceKind::Sum,
            axis: 0,
            compute_dtype: NumericDType::F32,
        },
        vec![b],
        vec![GroupInput {
            input_ref: InputRef::affine(inp, 1),
            sym_dim_map: vec![crate::nano_graph::pattern::SymDimMap::Identity(1)],
        }],
    );

    let bytes = f32_input_bytes(&[
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // atom 0
        11.0, 12.0, 13.0, 14.0, 15.0, 16.0, // atom 1
    ]);
    let mut bindings = std::collections::HashMap::new();
    bindings.insert(a, 2u64);
    bindings.insert(b, 3u64);

    let outs = ab_test_bytes_sym(
        &g,
        &[(inp, NumericDType::F32, bytes)],
        &[AtomRange {
            base: out,
            count: 2,
            dtype: NumericDType::F32,
        }],
        &[(inp, vec![a, b])],
        &bindings,
    );

    // atom 0 / b=0: 1+4=5, b=1: 2+5=7, b=2: 3+6=9.
    // atom 1 / b=0: 11+14=25, b=1: 12+15=27, b=2: 13+16=29.
    let expected = f32_input_bytes(&[5.0, 7.0, 9.0, 25.0, 27.0, 29.0]);
    assert_eq!(outs[0], expected, "SymReduce atoms × consumer sym");
}

/// Runtime extent = 1 exercises the single-k path (loop body runs
/// exactly once, accumulator ends up holding the sole loaded value).
#[test]
fn sym_reduce_sum_f32_extent_one() {
    let mut g = NanoGraph::new();
    let seq = g.bounded_graph_constant("seq_len", 4);

    let inp = g.add_input_tensor(GlobalId(0), 3, NumericDType::F32);
    let out = g.push_group(
        3,
        NumericDType::F32,
        ScalarOp::SymReduce {
            kind: crate::nano_graph::ops::ReduceKind::Sum,
            axis: 0,
            compute_dtype: NumericDType::F32,
        },
        vec![],
        vec![GroupInput {
            input_ref: InputRef::affine(inp, 1),
            sym_dim_map: vec![],
        }],
    );

    let bytes = f32_input_bytes(&[7.0, 8.0, 9.0]);
    let mut bindings = std::collections::HashMap::new();
    bindings.insert(seq, 1u64);

    let outs = ab_test_bytes_sym(
        &g,
        &[(inp, NumericDType::F32, bytes)],
        &[AtomRange {
            base: out,
            count: 3,
            dtype: NumericDType::F32,
        }],
        &[(inp, vec![seq])],
        &bindings,
    );

    let expected = f32_input_bytes(&[7.0, 8.0, 9.0]);
    assert_eq!(outs[0], expected, "SymReduce extent=1 passes value through");
}

// ─── Non-identity sym_dim_map Binary tests ───────────────────────────
//
// These exercise the general `sym_dim_map` remap in
// `address::apply_sym_offset_pub`: the consumer loop's `sym_i` must be
// decomposed into consumer coords and reassembled into the producer's
// sym flat index when the map is not trivially all-`Identity(j==j)`.

/// Consumer `[B, S]` + producer `[S]` broadcast: map = `[Broadcast,
/// Identity(0)]`. The inner axis is shared, the outer B axis is a
/// broadcast over the producer. Producer sym flat = `jS = sym_i % S`.
#[test]
fn binary_f32_sym_broadcast_outer_only() {
    use crate::nano_graph::pattern::SymDimMap;
    let mut g = NanoGraph::new();
    let big_b = g.bounded_graph_constant("B", 4);
    let big_s = g.bounded_graph_constant("S", 4);

    let inp_a = g.add_input_tensor(GlobalId(0), 1, NumericDType::F32);
    let inp_b = g.add_input_tensor(GlobalId(1), 1, NumericDType::F32);
    let out = g.push_group(
        1,
        NumericDType::F32,
        ScalarOp::Binary {
            op: ScalarBinOp::Add,
            compute_dtype: NumericDType::F32,
        },
        vec![big_b, big_s],
        vec![
            GroupInput {
                input_ref: InputRef::affine(inp_a, 1),
                sym_dim_map: vec![SymDimMap::Identity(0), SymDimMap::Identity(1)],
            },
            GroupInput {
                input_ref: InputRef::affine(inp_b, 1),
                sym_dim_map: vec![SymDimMap::Broadcast, SymDimMap::Identity(0)],
            },
        ],
    );

    // B=2, S=3. A (6 values, row-major jB,jS): [10..60 step 10].
    // B (3 values, one per jS): [1, 2, 3].
    // Expected out[jB*3+jS] = A[jB*3+jS] + B[jS]:
    // row 0 (jB=0): 10+1, 20+2, 30+3 = 11, 22, 33
    // row 1 (jB=1): 40+1, 50+2, 60+3 = 41, 52, 63
    let a_bytes = f32_input_bytes(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
    let b_bytes = f32_input_bytes(&[1.0, 2.0, 3.0]);
    let mut bindings = std::collections::HashMap::new();
    bindings.insert(big_b, 2u64);
    bindings.insert(big_s, 3u64);

    let outs = ab_test_bytes_sym(
        &g,
        &[
            (inp_a, NumericDType::F32, a_bytes),
            (inp_b, NumericDType::F32, b_bytes),
        ],
        &[AtomRange {
            base: out,
            count: 1,
            dtype: NumericDType::F32,
        }],
        &[(inp_a, vec![big_b, big_s]), (inp_b, vec![big_s])],
        &bindings,
    );

    let expected = f32_input_bytes(&[11.0, 22.0, 33.0, 41.0, 52.0, 63.0]);
    assert_eq!(
        outs[0], expected,
        "Binary Add with [Broadcast, Identity(0)] map (inner-axis broadcast)"
    );
}

/// Consumer `[B, S]` + producer `[B]` broadcast: map = `[Identity(0),
/// Broadcast]`. The outer B axis is shared, the inner S axis is a
/// broadcast. Producer sym flat = `jB = sym_i / S`. Exercises the
/// runtime `div` path for outer-coord extraction.
#[test]
fn binary_f32_sym_broadcast_inner_only() {
    use crate::nano_graph::pattern::SymDimMap;
    let mut g = NanoGraph::new();
    let big_b = g.bounded_graph_constant("B", 4);
    let big_s = g.bounded_graph_constant("S", 4);

    let inp_a = g.add_input_tensor(GlobalId(0), 1, NumericDType::F32);
    let inp_b = g.add_input_tensor(GlobalId(1), 1, NumericDType::F32);
    let out = g.push_group(
        1,
        NumericDType::F32,
        ScalarOp::Binary {
            op: ScalarBinOp::Add,
            compute_dtype: NumericDType::F32,
        },
        vec![big_b, big_s],
        vec![
            GroupInput {
                input_ref: InputRef::affine(inp_a, 1),
                sym_dim_map: vec![SymDimMap::Identity(0), SymDimMap::Identity(1)],
            },
            GroupInput {
                input_ref: InputRef::affine(inp_b, 1),
                sym_dim_map: vec![SymDimMap::Identity(0), SymDimMap::Broadcast],
            },
        ],
    );

    // B=2, S=3. A row-major (jB,jS): [10..60 step 10].
    // B (one per jB): [100, 200].
    // Expected out[jB*3+jS] = A[jB*3+jS] + B[jB]:
    // row 0 (jB=0): 10+100, 20+100, 30+100 = 110, 120, 130
    // row 1 (jB=1): 40+200, 50+200, 60+200 = 240, 250, 260
    let a_bytes = f32_input_bytes(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
    let b_bytes = f32_input_bytes(&[100.0, 200.0]);
    let mut bindings = std::collections::HashMap::new();
    bindings.insert(big_b, 2u64);
    bindings.insert(big_s, 3u64);

    let outs = ab_test_bytes_sym(
        &g,
        &[
            (inp_a, NumericDType::F32, a_bytes),
            (inp_b, NumericDType::F32, b_bytes),
        ],
        &[AtomRange {
            base: out,
            count: 1,
            dtype: NumericDType::F32,
        }],
        &[(inp_a, vec![big_b, big_s]), (inp_b, vec![big_b])],
        &bindings,
    );

    let expected = f32_input_bytes(&[110.0, 120.0, 130.0, 240.0, 250.0, 260.0]);
    assert_eq!(
        outs[0], expected,
        "Binary Add with [Identity(0), Broadcast] map (outer-axis broadcast)"
    );
}

/// Consumer `[a, b, c]` + producer `[a, c]` (drop middle axis). Map =
/// `[Identity(0), Broadcast, Identity(1)]`. Producer sym flat =
/// `ja * c_ext + jc`. Interleaved — exercises the stride-fold-over-
/// unmapped-axes path.
#[test]
fn binary_f32_sym_broadcast_middle_interleaved() {
    use crate::nano_graph::pattern::SymDimMap;
    let mut g = NanoGraph::new();
    let a = g.bounded_graph_constant("a", 4);
    let b = g.bounded_graph_constant("b", 4);
    let c = g.bounded_graph_constant("c", 4);

    let inp_a = g.add_input_tensor(GlobalId(0), 1, NumericDType::F32);
    let inp_b = g.add_input_tensor(GlobalId(1), 1, NumericDType::F32);
    let out = g.push_group(
        1,
        NumericDType::F32,
        ScalarOp::Binary {
            op: ScalarBinOp::Add,
            compute_dtype: NumericDType::F32,
        },
        vec![a, b, c],
        vec![
            GroupInput {
                input_ref: InputRef::affine(inp_a, 1),
                sym_dim_map: vec![
                    SymDimMap::Identity(0),
                    SymDimMap::Identity(1),
                    SymDimMap::Identity(2),
                ],
            },
            GroupInput {
                input_ref: InputRef::affine(inp_b, 1),
                sym_dim_map: vec![
                    SymDimMap::Identity(0),
                    SymDimMap::Broadcast,
                    SymDimMap::Identity(1),
                ],
            },
        ],
    );

    // a=2, b=2, c=3. A row-major (ja, jb, jc): sym_i = ja*6 + jb*3 + jc.
    // Values [1..12]:
    //   ja=0, jb=0: [1, 2, 3]
    //   ja=0, jb=1: [4, 5, 6]
    //   ja=1, jb=0: [7, 8, 9]
    //   ja=1, jb=1: [10, 11, 12]
    // B row-major (ja, jc): prod_sym_flat = ja*3 + jc.
    // Values [100, 200, 300, 400, 500, 600]:
    //   ja=0: [100, 200, 300]
    //   ja=1: [400, 500, 600]
    // Expected:
    //   ja=0,jb=0: [1+100, 2+200, 3+300]
    //   ja=0,jb=1: [4+100, 5+200, 6+300]
    //   ja=1,jb=0: [7+400, 8+500, 9+600]
    //   ja=1,jb=1: [10+400, 11+500, 12+600]
    let a_vals: Vec<f32> = (1..=12).map(|v| v as f32).collect();
    let b_vals: Vec<f32> = vec![100.0, 200.0, 300.0, 400.0, 500.0, 600.0];
    let a_bytes = f32_input_bytes(&a_vals);
    let b_bytes = f32_input_bytes(&b_vals);
    let mut bindings = std::collections::HashMap::new();
    bindings.insert(a, 2u64);
    bindings.insert(b, 2u64);
    bindings.insert(c, 3u64);

    let outs = ab_test_bytes_sym(
        &g,
        &[
            (inp_a, NumericDType::F32, a_bytes),
            (inp_b, NumericDType::F32, b_bytes),
        ],
        &[AtomRange {
            base: out,
            count: 1,
            dtype: NumericDType::F32,
        }],
        &[(inp_a, vec![a, b, c]), (inp_b, vec![a, c])],
        &bindings,
    );

    let expected = f32_input_bytes(&[
        101.0, 202.0, 303.0, //  ja=0,jb=0
        104.0, 205.0, 306.0, //  ja=0,jb=1
        407.0, 508.0, 609.0, //  ja=1,jb=0
        410.0, 511.0, 612.0, //  ja=1,jb=1
    ]);
    assert_eq!(
        outs[0], expected,
        "Binary Add with interleaved map [Identity(0), Broadcast, Identity(1)]"
    );
}

/// Pure outer broadcast with more atoms — exercises the atom-loop
/// interaction with the remap sequence (atom stride multiplication
/// plus per-atom sym loop).
#[test]
fn binary_f32_sym_broadcast_outer_multi_atom() {
    use crate::nano_graph::pattern::SymDimMap;
    let mut g = NanoGraph::new();
    let big_b = g.bounded_graph_constant("B", 4);
    let big_s = g.bounded_graph_constant("S", 4);

    // 2 atoms × [B=2, S=2] for A; 2 atoms × [B=2] for B.
    let inp_a = g.add_input_tensor(GlobalId(0), 2, NumericDType::F32);
    let inp_b = g.add_input_tensor(GlobalId(1), 2, NumericDType::F32);
    let out = g.push_group(
        2,
        NumericDType::F32,
        ScalarOp::Binary {
            op: ScalarBinOp::Add,
            compute_dtype: NumericDType::F32,
        },
        vec![big_b, big_s],
        vec![
            GroupInput {
                input_ref: InputRef::affine(inp_a, 1),
                sym_dim_map: vec![SymDimMap::Identity(0), SymDimMap::Identity(1)],
            },
            GroupInput {
                input_ref: InputRef::affine(inp_b, 1),
                sym_dim_map: vec![SymDimMap::Identity(0), SymDimMap::Broadcast],
            },
        ],
    );

    // B=2, S=2.
    // A[atom][jB][jS]:
    //   atom 0: [[1, 2], [3, 4]]
    //   atom 1: [[5, 6], [7, 8]]
    // B[atom][jB]:
    //   atom 0: [10, 20]
    //   atom 1: [30, 40]
    // Expected out[atom][jB][jS] = A + B[atom][jB]:
    //   atom 0: [[1+10, 2+10], [3+20, 4+20]] = [[11, 12], [23, 24]]
    //   atom 1: [[5+30, 6+30], [7+40, 8+40]] = [[35, 36], [47, 48]]
    let a_bytes = f32_input_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let b_bytes = f32_input_bytes(&[10.0, 20.0, 30.0, 40.0]);
    let mut bindings = std::collections::HashMap::new();
    bindings.insert(big_b, 2u64);
    bindings.insert(big_s, 2u64);

    let outs = ab_test_bytes_sym(
        &g,
        &[
            (inp_a, NumericDType::F32, a_bytes),
            (inp_b, NumericDType::F32, b_bytes),
        ],
        &[AtomRange {
            base: out,
            count: 2,
            dtype: NumericDType::F32,
        }],
        &[(inp_a, vec![big_b, big_s]), (inp_b, vec![big_b])],
        &bindings,
    );

    let expected = f32_input_bytes(&[11.0, 12.0, 23.0, 24.0, 35.0, 36.0, 47.0, 48.0]);
    assert_eq!(
        outs[0], expected,
        "Binary Add [Identity(0), Broadcast] with 2 atoms"
    );
}
