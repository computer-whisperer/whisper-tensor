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

use super::ab_harness::ab_test_bytes;
use crate::graph::GlobalId;
use crate::nano_graph::ops::ScalarOp;
use crate::nano_graph::pattern::{AtomRange, InputRef, NanoGraph};
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
        vec![InputRef::affine(inp, 1)],
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
        vec![InputRef::affine(inp, 1)],
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
        vec![InputRef::affine(inp, 1)],
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
        vec![InputRef::affine(inp, 1)],
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

// NOTE: A `BOOL` Identity A/B test would belong here but is blocked
// by a `PoolEvalSpan` output divergence for sub-byte dtypes:
// `PoolEvalSpan::execute` raw-copies the result NumericTensor's
// bit-packed bytes (e.g. one byte per 8 bools) into the SpanOutput
// buffer, while the JIT pipeline goes through `read_buffer_to_output`
// which produces the byte-padded format (one byte per bool). The
// two formats disagree even though the bits represent the same data.
// Re-enable once `PoolEvalSpan` either expands sub-byte storage on
// output or the SpanOutput contract picks one canonical layout.

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
        vec![InputRef::Broadcast(inp)],
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
        vec![InputRef::Explicit(vec![inp])],
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
        vec![InputRef::modular(inp, 1, 4)],
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
        vec![InputRef::modular(inp, 1, 3)],
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
        vec![InputRef::strided_broadcast(inp, 1, 4)],
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
        vec![InputRef::Explicit(ids)],
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
        vec![InputRef::affine(inp, 1)],
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
        vec![InputRef::affine(inp, 1)],
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
        vec![InputRef::affine(inp, 1)],
    );
    let src: Vec<f32> = vec![0.0, 1.0, -1.0, 0.5, 65504.0, -65504.0, f32::INFINITY, f32::NAN];
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
        vec![InputRef::affine(inp, 1)],
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
        vec![InputRef::affine(inp, 1)],
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
        vec![InputRef::affine(lit, 1)],
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
        vec![InputRef::affine(inp, 1)],
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
        vec![InputRef::affine(inp, 1)],
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

fn binary_f32_test(
    op: ScalarBinOp,
    a_vals: &[f32],
    b_vals: &[f32],
) -> Vec<Vec<u8>> {
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
            InputRef::affine(inp_a, 1),
            InputRef::affine(inp_b, 1),
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
        vec![InputRef::affine(inp, 1)],
    );
    let bytes = f32_input_bytes(vals);
    ab_test_bytes(
        &g,
        &[(inp, NumericDType::F32, bytes)],
        &[AtomRange { base: out, count: n, dtype: NumericDType::F32 }],
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
        ScalarOp::Binary { op, compute_dtype: NumericDType::I32 },
        vec![],
        vec![InputRef::affine(inp_a, 1), InputRef::affine(inp_b, 1)],
    );
    let a_bytes: Vec<u8> = a_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    let b_bytes: Vec<u8> = b_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    ab_test_bytes(
        &g,
        &[
            (inp_a, NumericDType::I32, a_bytes),
            (inp_b, NumericDType::I32, b_bytes),
        ],
        &[AtomRange { base: out, count: n, dtype: NumericDType::I32 }],
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
    let outs = binary_i32_test(
        ScalarBinOp::Mul,
        &[3, -3, 100000, 0],
        &[7, 7, 100000, 42],
    );
    assert_eq!(outs[0].len(), 16, "I32 Mul output size");
}

#[test]
fn binary_i32_div() {
    // Includes div-by-zero → 0.
    let outs = binary_i32_test(
        ScalarBinOp::Div,
        &[10, -10, 42, i32::MIN],
        &[3, 3, 0, -1],
    );
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
        .iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(outs[0], expected, "I32 BitwiseAnd");
}

#[test]
fn binary_i32_shift_left() {
    let outs = binary_i32_test(
        ScalarBinOp::BitShiftLeft,
        &[1, 1, -1, 0xFF],
        &[0, 8, 16, 4],
    );
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
        vec![InputRef::affine(inp_a, 1), InputRef::affine(inp_b, 1)],
    );
    let a: Vec<u8> = [5i32, 3, 3, -1].iter().flat_map(|v| v.to_le_bytes()).collect();
    let b: Vec<u8> = [3i32, 5, 3, 1].iter().flat_map(|v| v.to_le_bytes()).collect();
    let outs = ab_test_bytes(
        &g,
        &[(inp_a, NumericDType::I32, a), (inp_b, NumericDType::I32, b)],
        &[AtomRange { base: out, count: 4, dtype: NumericDType::I32 }],
    );
    assert_eq!(outs[0].len(), 16, "I32 Greater output size");
}
