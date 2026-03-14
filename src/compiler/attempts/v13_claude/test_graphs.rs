//! Test NanoGraph builders for benchmarking partition strategies.
//!
//! These build NanoGraphs that match the structure the real lowering produces,
//! so partition quality on these graphs is representative of real workloads.

use crate::dtype::DType;
use crate::nano_graph::{AtomId, InputRef, NanoGraph, ScalarBinOp, ScalarOp, ScalarUnaryOp};
use crate::numeric_scalar::NumericScalar;

/// Elementwise binary op: C[N] = A[N] op B[N].
/// Returns (graph, a_base, b_base, c_base).
pub fn elementwise_binary(n: u32, op: ScalarBinOp) -> (NanoGraph, AtomId, AtomId, AtomId) {
    let mut g = NanoGraph::new();
    let a = g.push_group(
        n,
        ScalarOp::Literal(NumericScalar::F32(0.0)),
        vec![], vec![], vec![],
    );
    let b = g.push_group(
        n,
        ScalarOp::Literal(NumericScalar::F32(0.0)),
        vec![], vec![], vec![],
    );
    let c = g.push_group(
        n,
        ScalarOp::Binary { op, compute_dtype: DType::F32, output_dtype: DType::F32 },
        vec![], vec![],
        vec![
            InputRef::Affine { base: a, stride: 1 },
            InputRef::Affine { base: b, stride: 1 },
        ],
    );
    g.outputs = vec![c];
    (g, a, b, c)
}

/// Unary chain: out = op_n(op_{n-1}(...op_1(input))).
/// Returns (graph, input_base, output_base).
pub fn unary_chain(n: u32, ops: &[ScalarUnaryOp]) -> (NanoGraph, AtomId, AtomId) {
    let mut g = NanoGraph::new();
    let input = g.push_group(
        n,
        ScalarOp::Literal(NumericScalar::F32(0.0)),
        vec![], vec![], vec![],
    );
    let mut prev = input;
    for &op in ops {
        prev = g.push_group(
            n,
            ScalarOp::Unary { op, compute_dtype: DType::F32, output_dtype: DType::F32 },
            vec![], vec![],
            vec![InputRef::Affine { base: prev, stride: 1 }],
        );
    }
    g.outputs = vec![prev];
    (g, input, prev)
}

/// Broadcast-add: C[N] = A[N] + scalar_b.
/// Returns (graph, a_base, scalar_b, c_base).
pub fn broadcast_add(n: u32) -> (NanoGraph, AtomId, AtomId, AtomId) {
    let mut g = NanoGraph::new();
    let a = g.push_group(
        n,
        ScalarOp::Literal(NumericScalar::F32(0.0)),
        vec![], vec![], vec![],
    );
    let b = g.push_atom(
        ScalarOp::Literal(NumericScalar::F32(0.0)),
        vec![], vec![], vec![],
    );
    let c = g.push_group(
        n,
        ScalarOp::Binary { op: ScalarBinOp::Add, compute_dtype: DType::F32, output_dtype: DType::F32 },
        vec![], vec![],
        vec![
            InputRef::Affine { base: a, stride: 1 },
            InputRef::Broadcast(b),
        ],
    );
    g.outputs = vec![c];
    (g, a, b, c)
}

/// MatMul: C[M, N] = A[M, K] @ B[K, N].
///
/// Matches the real lowering structure:
/// - M*K Mul groups, each of count N. Group (m,k) has:
///   - Input 0: Broadcast(A[m, k])
///   - Input 1: Affine(B[k, 0], stride=1)
/// - M ReduceSum groups, each of count N, with SymAffine(stride_i=1, stride_k=N).
///
/// Returns (graph, a_base, b_base, reduce_base) where reduce_base is the
/// output matmul result.
pub fn matmul(m: u32, k: u32, n: u32) -> (NanoGraph, AtomId, AtomId, AtomId) {
    let mut g = NanoGraph::new();

    // A[M, K] — laid out row-major, so A[m, k] = a_base + m*K + k
    let a = g.push_group(
        m * k,
        ScalarOp::Literal(NumericScalar::F32(0.0)),
        vec![], vec![], vec![],
    );

    // B[K, N] — laid out row-major, so B[k, j] = b_base + k*N + j
    let b = g.push_group(
        k * n,
        ScalarOp::Literal(NumericScalar::F32(0.0)),
        vec![], vec![], vec![],
    );

    // K sym dim for the contraction.
    let k_sym = g.bounded_sym_dim("k", k as u64);

    // M*K Mul groups, each of count N.
    let mut mul_base: Option<AtomId> = None;
    for mi in 0..m {
        for ki in 0..k {
            let a_atom = a.offset(mi * k + ki);           // A[m, k]
            let b_row_start = b.offset(ki * n);            // B[k, 0]

            let base = g.push_group(
                n,
                ScalarOp::Binary {
                    op: ScalarBinOp::Mul,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![], vec![],
                vec![
                    InputRef::Broadcast(a_atom),
                    InputRef::Affine { base: b_row_start, stride: 1 },
                ],
            );
            if mul_base.is_none() {
                mul_base = Some(base);
            }
        }
    }
    let mul_base = mul_base.unwrap();

    // M ReduceSum groups, each of count N.
    let mut reduce_base: Option<AtomId> = None;
    for mi in 0..m {
        let row_mul_base = AtomId(mul_base.0 + mi * k * n);

        let base = g.push_group(
            n,
            ScalarOp::ReduceSum {
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![k_sym],
            vec![InputRef::SymAffine {
                base: row_mul_base,
                stride_i: 1,
                stride_k: n as i32,
            }],
        );
        if reduce_base.is_none() {
            reduce_base = Some(base);
        }
    }
    let reduce_base = reduce_base.unwrap();
    g.outputs = vec![reduce_base];

    (g, a, b, reduce_base)
}

/// MatMul followed by elementwise activation: out = activation(A @ B).
/// Returns (graph, a_base, b_base, out_base).
pub fn matmul_activation(
    m: u32, k: u32, n: u32,
    activation: ScalarUnaryOp,
) -> (NanoGraph, AtomId, AtomId, AtomId) {
    let mut g = NanoGraph::new();

    let a = g.push_group(
        m * k,
        ScalarOp::Literal(NumericScalar::F32(0.0)),
        vec![], vec![], vec![],
    );
    let b = g.push_group(
        k * n,
        ScalarOp::Literal(NumericScalar::F32(0.0)),
        vec![], vec![], vec![],
    );

    let k_sym = g.bounded_sym_dim("k", k as u64);

    let mut mul_base: Option<AtomId> = None;
    for mi in 0..m {
        for ki in 0..k {
            let a_atom = a.offset(mi * k + ki);
            let b_row_start = b.offset(ki * n);
            let base = g.push_group(
                n,
                ScalarOp::Binary {
                    op: ScalarBinOp::Mul,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![], vec![],
                vec![
                    InputRef::Broadcast(a_atom),
                    InputRef::Affine { base: b_row_start, stride: 1 },
                ],
            );
            if mul_base.is_none() {
                mul_base = Some(base);
            }
        }
    }
    let mul_base = mul_base.unwrap();

    let mut reduce_base: Option<AtomId> = None;
    for mi in 0..m {
        let row_mul_base = AtomId(mul_base.0 + mi * k * n);
        let base = g.push_group(
            n,
            ScalarOp::ReduceSum {
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![k_sym],
            vec![InputRef::SymAffine {
                base: row_mul_base,
                stride_i: 1,
                stride_k: n as i32,
            }],
        );
        if reduce_base.is_none() {
            reduce_base = Some(base);
        }
    }
    let reduce_base = reduce_base.unwrap();

    // Elementwise activation on the matmul output.
    let out = g.push_group(
        m * n,
        ScalarOp::Unary {
            op: activation,
            compute_dtype: DType::F32,
            output_dtype: DType::F32,
        },
        vec![], vec![],
        vec![InputRef::Affine { base: reduce_base, stride: 1 }],
    );
    g.outputs = vec![out];

    (g, a, b, out)
}

/// Two chained matmuls: out = (A @ B) @ C.
/// Mimics the common MLP pattern where one matmul feeds another.
/// Returns (graph, a_base, b_base, c_base, out_base).
pub fn matmul_chain(
    m: u32, k1: u32, n1: u32, k2: u32, n2: u32,
) -> (NanoGraph, AtomId, AtomId, AtomId, AtomId) {
    // Note: n1 == k2 for the chain to be valid.
    assert_eq!(n1, k2, "inner dimensions must match: n1={} != k2={}", n1, k2);

    let mut g = NanoGraph::new();

    // A[M, K1]
    let a = g.push_group(
        m * k1,
        ScalarOp::Literal(NumericScalar::F32(0.0)),
        vec![], vec![], vec![],
    );
    // B[K1, N1]
    let b = g.push_group(
        k1 * n1,
        ScalarOp::Literal(NumericScalar::F32(0.0)),
        vec![], vec![], vec![],
    );
    // C[K2, N2] (K2 == N1)
    let c = g.push_group(
        k2 * n2,
        ScalarOp::Literal(NumericScalar::F32(0.0)),
        vec![], vec![], vec![],
    );

    // First matmul: AB = A @ B  →  [M, N1]
    let k1_sym = g.bounded_sym_dim("k1", k1 as u64);

    let mut mul1_base: Option<AtomId> = None;
    for mi in 0..m {
        for ki in 0..k1 {
            let a_atom = a.offset(mi * k1 + ki);
            let b_row_start = b.offset(ki * n1);
            let base = g.push_group(
                n1,
                ScalarOp::Binary {
                    op: ScalarBinOp::Mul,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![], vec![],
                vec![
                    InputRef::Broadcast(a_atom),
                    InputRef::Affine { base: b_row_start, stride: 1 },
                ],
            );
            if mul1_base.is_none() {
                mul1_base = Some(base);
            }
        }
    }
    let mul1_base = mul1_base.unwrap();

    let mut ab_base: Option<AtomId> = None;
    for mi in 0..m {
        let row_mul_base = AtomId(mul1_base.0 + mi * k1 * n1);
        let base = g.push_group(
            n1,
            ScalarOp::ReduceSum {
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![k1_sym],
            vec![InputRef::SymAffine {
                base: row_mul_base,
                stride_i: 1,
                stride_k: n1 as i32,
            }],
        );
        if ab_base.is_none() {
            ab_base = Some(base);
        }
    }
    let ab_base = ab_base.unwrap();

    // Second matmul: out = AB @ C  →  [M, N2]
    let k2_sym = g.bounded_sym_dim("k2", k2 as u64);

    let mut mul2_base: Option<AtomId> = None;
    for mi in 0..m {
        for ki in 0..k2 {
            // AB[m, ki] = ab_base + m*N1 + ki  (N1 == K2)
            let ab_atom = ab_base.offset(mi * n1 + ki);
            let c_row_start = c.offset(ki * n2);
            let base = g.push_group(
                n2,
                ScalarOp::Binary {
                    op: ScalarBinOp::Mul,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![], vec![],
                vec![
                    InputRef::Broadcast(ab_atom),
                    InputRef::Affine { base: c_row_start, stride: 1 },
                ],
            );
            if mul2_base.is_none() {
                mul2_base = Some(base);
            }
        }
    }
    let mul2_base = mul2_base.unwrap();

    let mut out_base: Option<AtomId> = None;
    for mi in 0..m {
        let row_mul_base = AtomId(mul2_base.0 + mi * k2 * n2);
        let base = g.push_group(
            n2,
            ScalarOp::ReduceSum {
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![k2_sym],
            vec![InputRef::SymAffine {
                base: row_mul_base,
                stride_i: 1,
                stride_k: n2 as i32,
            }],
        );
        if out_base.is_none() {
            out_base = Some(base);
        }
    }
    let out_base = out_base.unwrap();
    g.outputs = vec![out_base];

    (g, a, b, c, out_base)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elementwise_valid() {
        let (g, _, _, _) = elementwise_binary(1024, ScalarBinOp::Add);
        assert!(g.validate().is_empty(), "{:?}", g.validate());
        assert_eq!(g.num_atoms(), 3 * 1024);
    }

    #[test]
    fn test_matmul_valid() {
        let (g, _, _, _) = matmul(4, 8, 16);
        let errors = g.validate();
        assert!(errors.is_empty(), "{:?}", errors);
        // 4*8 A + 8*16 B + 4*8*16 Mul + 4*16 Reduce = 32 + 128 + 512 + 64 = 736
        assert_eq!(g.num_atoms(), 4 * 8 + 8 * 16 + 4 * 8 * 16 + 4 * 16);
    }

    #[test]
    fn test_matmul_activation_valid() {
        let (g, _, _, _) = matmul_activation(4, 8, 16, ScalarUnaryOp::Tanh);
        let errors = g.validate();
        assert!(errors.is_empty(), "{:?}", errors);
        // Same as matmul + M*N activation atoms
        assert_eq!(g.num_atoms(), 4 * 8 + 8 * 16 + 4 * 8 * 16 + 4 * 16 + 4 * 16);
    }

    #[test]
    fn test_matmul_chain_valid() {
        let (g, _, _, _, _) = matmul_chain(4, 8, 16, 16, 32);
        let errors = g.validate();
        assert!(errors.is_empty(), "{:?}", errors);
    }

    #[test]
    fn test_unary_chain_valid() {
        let (g, _, _) = unary_chain(256, &[ScalarUnaryOp::Exp, ScalarUnaryOp::Neg, ScalarUnaryOp::Tanh]);
        assert!(g.validate().is_empty());
        assert_eq!(g.num_atoms(), 4 * 256); // input + 3 unary stages
    }
}
