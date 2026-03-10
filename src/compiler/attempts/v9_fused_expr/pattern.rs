//! Pattern recovery from inlined expression trees.
//!
//! After inlining, each output binding has a fully-expanded ScalarExpr tree.
//! This pass:
//! 1. Groups bindings by output tensor
//! 2. Extracts a structural "pattern" (the tree shape with load indices abstracted)
//! 3. Recovers multi-dimensional affine access coefficients from sampled instances
//! 4. Classifies each group as pointwise or reduction

use crate::compiler::common::v2_frontend::{
    OutputBinding, ScalarBinOp, ScalarExpr, ScalarUnaryOp,
};
use crate::graph::GlobalId;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Pattern extraction
// ---------------------------------------------------------------------------

/// Structural pattern of an expression tree with load indices abstracted.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PatternExpr {
    Load {
        tensor: GlobalId,
        load_ordinal: usize,
    },
    Literal {
        value_bits: u64,
    },
    Binary {
        op: ScalarBinOp,
        a: Box<PatternExpr>,
        b: Box<PatternExpr>,
    },
    Unary {
        op: ScalarUnaryOp,
        input: Box<PatternExpr>,
    },
}

pub fn extract_pattern(expr: &ScalarExpr) -> (PatternExpr, Vec<(GlobalId, usize)>) {
    let mut loads = Vec::new();
    let pattern = extract_pattern_inner(expr, &mut loads);
    (pattern, loads)
}

fn extract_pattern_inner(
    expr: &ScalarExpr,
    loads: &mut Vec<(GlobalId, usize)>,
) -> PatternExpr {
    match expr {
        ScalarExpr::Element {
            tensor, flat_index, ..
        } => {
            let ordinal = loads.len();
            loads.push((*tensor, *flat_index));
            PatternExpr::Load {
                tensor: *tensor,
                load_ordinal: ordinal,
            }
        }
        ScalarExpr::Literal { value } => PatternExpr::Literal {
            value_bits: value.to_bits(),
        },
        ScalarExpr::Binary { op, a, b } => PatternExpr::Binary {
            op: *op,
            a: Box::new(extract_pattern_inner(a, loads)),
            b: Box::new(extract_pattern_inner(b, loads)),
        },
        ScalarExpr::Unary { op, input } => PatternExpr::Unary {
            op: *op,
            input: Box::new(extract_pattern_inner(input, loads)),
        },
    }
}

// ---------------------------------------------------------------------------
// Multi-dimensional affine access
// ---------------------------------------------------------------------------

/// Multi-dimensional affine access:
/// `flat_index = base + sum_d(axis_coeffs[d] * axis_var[d]) + reduction_coeff * k`
#[derive(Debug, Clone)]
pub struct AffineAccess {
    pub tensor: GlobalId,
    pub base: i64,
    /// Coefficient per output axis
    pub axis_coeffs: Vec<i64>,
}

/// A recovered loop pattern for a group of output bindings.
#[derive(Debug, Clone)]
pub struct RecoveredPattern {
    pub output_tensor: GlobalId,
    pub pattern: PatternExpr,
    /// Output shape (loop extents per axis)
    pub output_shape: Vec<usize>,
    /// Per-load affine access
    pub accesses: Vec<AffineAccess>,
    pub kind: PatternKind,
}

#[derive(Debug, Clone)]
pub enum PatternKind {
    Pointwise,
    Reduction {
        accum_op: ScalarBinOp,
        identity: f64,
        depth: usize,
        /// Per-load coefficient for the reduction variable
        reduction_coeffs: Vec<i64>,
    },
}

/// Recover patterns from a set of inlined bindings.
///
/// `output_shapes` maps tensor IDs to their shapes (needed to decompose
/// flat indices into multi-dimensional coordinates).
pub fn recover_patterns(
    bindings: &[OutputBinding],
    output_shapes: &HashMap<GlobalId, Vec<usize>>,
) -> Vec<RecoveredPattern> {
    let mut by_output: HashMap<GlobalId, Vec<&OutputBinding>> = HashMap::new();
    for b in bindings {
        by_output.entry(b.output_tensor).or_default().push(b);
    }

    let mut patterns = Vec::new();
    for (out_tensor, group) in &by_output {
        if group.is_empty() {
            continue;
        }
        let shape = output_shapes.get(out_tensor).cloned().unwrap_or_else(|| {
            // Fallback: infer 1D shape from count
            vec![group.len()]
        });
        if let Some(pat) = recover_one_group(*out_tensor, group, &shape) {
            patterns.push(pat);
        }
    }

    patterns.sort_by_key(|p| p.output_tensor);
    patterns
}

fn recover_one_group(
    out_tensor: GlobalId,
    group: &[&OutputBinding],
    output_shape: &[usize],
) -> Option<RecoveredPattern> {
    let (pattern, first_loads) = extract_pattern(&group[0].expr);
    let num_loads = first_loads.len();

    // Verify all bindings have the same pattern, collect all load indices
    let mut all_loads: Vec<Vec<(GlobalId, usize)>> = Vec::with_capacity(group.len());
    all_loads.push(first_loads);
    for b in &group[1..] {
        let (p, loads) = extract_pattern(&b.expr);
        if p != pattern {
            return None;
        }
        all_loads.push(loads);
    }

    // Detect reduction vs pointwise
    let mut loads_per_tensor: HashMap<GlobalId, usize> = HashMap::new();
    for (t, _) in &all_loads[0] {
        *loads_per_tensor.entry(*t).or_default() += 1;
    }
    let is_reduction = loads_per_tensor.values().any(|&c| c > 1);

    if is_reduction {
        recover_reduction(out_tensor, pattern, group, &all_loads, output_shape)
    } else {
        recover_pointwise(out_tensor, pattern, group, &all_loads, output_shape)
    }
}

fn recover_pointwise(
    out_tensor: GlobalId,
    pattern: PatternExpr,
    group: &[&OutputBinding],
    all_loads: &[Vec<(GlobalId, usize)>],
    output_shape: &[usize],
) -> Option<RecoveredPattern> {
    let ndim = output_shape.len();
    let num_loads = all_loads[0].len();

    // For each load position, recover multi-dimensional affine coefficients.
    // load_flat = base + sum_d(axis_coeffs[d] * out_axis[d])
    //
    // Strategy: use multiple samples to set up a system of equations.
    // With N output dims, we need N+1 samples minimum. Since we have
    // group.len() samples, pick ones that vary each axis independently.

    // Decompose all output flat indices to multi-dimensional
    let out_multis: Vec<Vec<usize>> = group
        .iter()
        .map(|b| flat_to_multi(b.flat_index, output_shape))
        .collect();

    let mut accesses = Vec::with_capacity(num_loads);
    for load_pos in 0..num_loads {
        let coeffs = solve_affine_coeffs(
            &out_multis,
            &all_loads.iter().map(|l| l[load_pos].1 as i64).collect::<Vec<_>>(),
            ndim,
        );
        accesses.push(AffineAccess {
            tensor: all_loads[0][load_pos].0,
            base: coeffs.0,
            axis_coeffs: coeffs.1,
        });
    }

    Some(RecoveredPattern {
        output_tensor: out_tensor,
        pattern,
        output_shape: output_shape.to_vec(),
        accesses,
        kind: PatternKind::Pointwise,
    })
}

fn recover_reduction(
    out_tensor: GlobalId,
    pattern: PatternExpr,
    group: &[&OutputBinding],
    all_loads: &[Vec<(GlobalId, usize)>],
    output_shape: &[usize],
) -> Option<RecoveredPattern> {
    let first_loads = &all_loads[0];
    let num_loads = first_loads.len();
    let ndim = output_shape.len();

    // Find the period (loads per k-iteration)
    let mut period = 0;
    'outer: for p in 1..=num_loads {
        if num_loads % p != 0 {
            continue;
        }
        for i in 0..(num_loads - p) {
            if first_loads[i].0 != first_loads[i + p].0 {
                continue 'outer;
            }
        }
        period = p;
        break;
    }
    if period == 0 {
        return None;
    }

    let depth = num_loads / period;
    let accum_op = detect_accum_op(&pattern)?;
    let identity = detect_identity(&pattern).unwrap_or(0.0);

    let out_multis: Vec<Vec<usize>> = group
        .iter()
        .map(|b| flat_to_multi(b.flat_index, output_shape))
        .collect();

    let mut accesses = Vec::with_capacity(period);
    let mut reduction_coeffs = Vec::with_capacity(period);

    for pos in 0..period {
        // Reduction coefficient: delta between k=0 and k=1 within first sample
        let idx_k0 = first_loads[pos].1 as i64;
        let red_coeff = if depth > 1 {
            let idx_k1 = first_loads[pos + period].1 as i64;
            idx_k1 - idx_k0
        } else {
            0
        };

        // Output coefficients: solve from cross-sample variation at k=0
        let load_vals: Vec<i64> = all_loads.iter().map(|l| l[pos].1 as i64).collect();
        let (base, axis_coeffs) = solve_affine_coeffs(&out_multis, &load_vals, ndim);

        accesses.push(AffineAccess {
            tensor: first_loads[pos].0,
            base,
            axis_coeffs,
        });
        reduction_coeffs.push(red_coeff);
    }

    Some(RecoveredPattern {
        output_tensor: out_tensor,
        pattern,
        output_shape: output_shape.to_vec(),
        accesses,
        kind: PatternKind::Reduction {
            accum_op,
            identity,
            depth,
            reduction_coeffs,
        },
    })
}

/// Solve for affine coefficients: load_flat = base + sum(coeff[d] * axis[d])
///
/// Given samples of (multi_index, load_flat_index), recover base and coefficients.
/// Uses the first sample as the reference, then differences to recover each coeff.
fn solve_affine_coeffs(
    out_multis: &[Vec<usize>],
    load_flats: &[i64],
    ndim: usize,
) -> (i64, Vec<i64>) {
    if out_multis.is_empty() {
        return (0, vec![0; ndim]);
    }

    let ref_multi = &out_multis[0];
    let ref_load = load_flats[0];

    let mut coeffs = vec![0i64; ndim];

    // For each axis, find a sample that differs only (or primarily) on that axis
    for d in 0..ndim {
        // Find a sample where axis d differs from reference
        for (i, multi) in out_multis.iter().enumerate().skip(1) {
            let axis_delta = multi[d] as i64 - ref_multi[d] as i64;
            if axis_delta != 0 {
                let load_delta = load_flats[i] - ref_load;
                // Subtract contributions from already-solved axes
                let mut other_contribution = 0i64;
                for d2 in 0..d {
                    other_contribution +=
                        coeffs[d2] * (multi[d2] as i64 - ref_multi[d2] as i64);
                }
                coeffs[d] = (load_delta - other_contribution) / axis_delta;
                break;
            }
        }
    }

    // base = ref_load - sum(coeffs[d] * ref_multi[d])
    let base = ref_load
        - coeffs
            .iter()
            .zip(ref_multi.iter())
            .map(|(&c, &m)| c * m as i64)
            .sum::<i64>();

    (base, coeffs)
}

fn detect_accum_op(pattern: &PatternExpr) -> Option<ScalarBinOp> {
    match pattern {
        PatternExpr::Binary { op, a, .. } => {
            if let PatternExpr::Binary { op: inner_op, .. } = a.as_ref() {
                if inner_op == op {
                    return Some(*op);
                }
            }
            if let PatternExpr::Literal { .. } = a.as_ref() {
                return Some(*op);
            }
            Some(*op)
        }
        _ => None,
    }
}

fn detect_identity(pattern: &PatternExpr) -> Option<f64> {
    match pattern {
        PatternExpr::Literal { value_bits } => Some(f64::from_bits(*value_bits)),
        PatternExpr::Binary { a, .. } => detect_identity(a),
        _ => None,
    }
}

fn flat_to_multi(flat: usize, shape: &[usize]) -> Vec<usize> {
    let ndim = shape.len();
    if ndim == 0 {
        return vec![];
    }
    let mut indices = vec![0usize; ndim];
    let mut remaining = flat;
    for i in (0..ndim).rev() {
        indices[i] = remaining % shape[i];
        remaining /= shape[i];
    }
    indices
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::common::v2_frontend::*;

    #[test]
    fn test_pattern_extraction() {
        let expr = ScalarExpr::Binary {
            op: ScalarBinOp::Add,
            a: Box::new(ScalarExpr::Element {
                tensor: GlobalId(1),
                flat_index: 5,
            }),
            b: Box::new(ScalarExpr::Element {
                tensor: GlobalId(2),
                flat_index: 10,
            }),
        };
        let (pattern, loads) = extract_pattern(&expr);
        assert_eq!(loads.len(), 2);
        assert_eq!(loads[0], (GlobalId(1), 5));
        assert_eq!(loads[1], (GlobalId(2), 10));
        match &pattern {
            PatternExpr::Binary {
                op: ScalarBinOp::Add,
                a,
                b,
            } => {
                assert!(matches!(a.as_ref(), PatternExpr::Load { load_ordinal: 0, .. }));
                assert!(matches!(b.as_ref(), PatternExpr::Load { load_ordinal: 1, .. }));
            }
            _ => panic!("Expected Binary::Add"),
        }
    }

    #[test]
    fn test_pointwise_recovery_1d() {
        let a = GlobalId(1);
        let b = GlobalId(2);
        let out = GlobalId(3);

        let bindings: Vec<OutputBinding> = (0..4)
            .map(|i| OutputBinding {
                output_tensor: out,
                flat_index: i,
                expr: ScalarExpr::Binary {
                    op: ScalarBinOp::Add,
                    a: Box::new(ScalarExpr::Element { tensor: a, flat_index: i }),
                    b: Box::new(ScalarExpr::Element { tensor: b, flat_index: i }),
                },
            })
            .collect();

        let mut shapes = HashMap::new();
        shapes.insert(out, vec![4]);
        let patterns = recover_patterns(&bindings, &shapes);
        assert_eq!(patterns.len(), 1);
        let p = &patterns[0];
        assert!(matches!(p.kind, PatternKind::Pointwise));
        assert_eq!(p.accesses[0].axis_coeffs, vec![1]);
        assert_eq!(p.accesses[0].base, 0);
        assert_eq!(p.accesses[1].axis_coeffs, vec![1]);
    }

    #[test]
    fn test_broadcast_recovery() {
        let a = GlobalId(1);
        let b = GlobalId(2);
        let out = GlobalId(3);

        let bindings: Vec<OutputBinding> = (0..4)
            .map(|i| OutputBinding {
                output_tensor: out,
                flat_index: i,
                expr: ScalarExpr::Binary {
                    op: ScalarBinOp::Add,
                    a: Box::new(ScalarExpr::Element { tensor: a, flat_index: i }),
                    b: Box::new(ScalarExpr::Element { tensor: b, flat_index: 0 }),
                },
            })
            .collect();

        let mut shapes = HashMap::new();
        shapes.insert(out, vec![4]);
        let patterns = recover_patterns(&bindings, &shapes);
        assert_eq!(patterns.len(), 1);
        let p = &patterns[0];
        assert_eq!(p.accesses[0].axis_coeffs, vec![1]);
        assert_eq!(p.accesses[1].axis_coeffs, vec![0]);
        assert_eq!(p.accesses[1].base, 0);
    }

    #[test]
    fn test_matmul_recovery_2d() {
        // out[i,j] = sum_k(a[i,k] * b[k,j]) for 2x3 * 3x4 -> 2x4
        let a = GlobalId(1);
        let b = GlobalId(2);
        let out = GlobalId(3);
        let m = 2;
        let n = 4;
        let k = 3;

        let mut bindings = Vec::new();
        for i in 0..m {
            for j in 0..n {
                let mut expr = ScalarExpr::Literal { value: 0.0 };
                for kk in 0..k {
                    let a_elem = ScalarExpr::Element {
                        tensor: a,
                        flat_index: i * k + kk,
                    };
                    let b_elem = ScalarExpr::Element {
                        tensor: b,
                        flat_index: kk * n + j,
                    };
                    let prod = ScalarExpr::Binary {
                        op: ScalarBinOp::Mul,
                        a: Box::new(a_elem),
                        b: Box::new(b_elem),
                    };
                    expr = ScalarExpr::Binary {
                        op: ScalarBinOp::Add,
                        a: Box::new(expr),
                        b: Box::new(prod),
                    };
                }
                bindings.push(OutputBinding {
                    output_tensor: out,
                    flat_index: i * n + j,
                    expr,
                });
            }
        }

        let mut shapes = HashMap::new();
        shapes.insert(out, vec![m, n]);
        let patterns = recover_patterns(&bindings, &shapes);
        assert_eq!(patterns.len(), 1);
        let p = &patterns[0];
        assert_eq!(p.output_shape, vec![m, n]);

        match &p.kind {
            PatternKind::Reduction {
                accum_op,
                depth,
                reduction_coeffs,
                ..
            } => {
                assert_eq!(*accum_op, ScalarBinOp::Add);
                assert_eq!(*depth, k);
                assert_eq!(p.accesses.len(), 2);
                // A access: base=0, axis_coeffs=[K, 0], red_coeff=1
                // A[i*K+kk]: varies with i (coeff K=3), not with j (coeff 0)
                assert_eq!(p.accesses[0].axis_coeffs, vec![k as i64, 0]);
                assert_eq!(reduction_coeffs[0], 1);
                // B access: base=0, axis_coeffs=[0, 1], red_coeff=N
                // B[kk*N+j]: varies with j (coeff 1), not with i (coeff 0)
                assert_eq!(p.accesses[1].axis_coeffs, vec![0, 1]);
                assert_eq!(reduction_coeffs[1], n as i64);
            }
            _ => panic!("Expected Reduction"),
        }
    }
}
