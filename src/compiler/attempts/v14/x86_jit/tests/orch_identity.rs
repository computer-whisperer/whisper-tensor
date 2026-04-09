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
