//! Pattern recovery from inlined expression trees.
//!
//! After inlining, each output binding has a fully-expanded ScalarExpr tree.
//! This pass:
//! 1. Groups bindings by output tensor
//! 2. Extracts structural patterns, collapsing left-fold chains into Reduce nodes
//! 3. Recovers multi-dimensional affine access coefficients
//!
//! Reductions are interior Reduce nodes in the pattern tree, not a top-level
//! classification. This enables patterns like `tanh(matmul + bias)` where a
//! reduction is embedded inside pointwise operations.

use crate::compiler::common::v2_frontend::{
    OutputBinding, ScalarBinOp, ScalarExpr, ScalarUnaryOp,
};
use crate::graph::GlobalId;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Pattern expression tree
// ---------------------------------------------------------------------------

/// Structural pattern of an expression tree with load indices abstracted.
/// Reductions appear as interior `Reduce` nodes rather than a top-level kind.
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
    /// A left-fold reduction collapsed from a repeating chain.
    /// The body pattern is evaluated once per k-iteration.
    Reduce {
        accum_op: ScalarBinOp,
        identity_bits: u64,
        body: Box<PatternExpr>,
        /// Index into RecoveredPattern::reductions
        reduce_idx: usize,
    },
}

// ---------------------------------------------------------------------------
// Pattern extraction
// ---------------------------------------------------------------------------

/// Result of extracting a pattern from one ScalarExpr.
pub struct ExtractionResult {
    pub pattern: PatternExpr,
    /// Loads outside any reduction (pointwise loads).
    pub outer_loads: Vec<(GlobalId, usize)>,
    /// One entry per Reduce node, in pre-order traversal order.
    pub reductions: Vec<ReductionExtraction>,
}

/// Load info for one Reduce node across k iterations.
pub struct ReductionExtraction {
    pub depth: usize,
    pub body_load_count: usize,
    /// loads_by_k[k][load_pos] = (tensor, flat_index)
    pub loads_by_k: Vec<Vec<(GlobalId, usize)>>,
}

pub fn extract_pattern(expr: &ScalarExpr) -> ExtractionResult {
    let mut outer_loads = Vec::new();
    let mut reductions = Vec::new();
    let pattern = extract_inner(expr, &mut outer_loads, &mut reductions);
    ExtractionResult {
        pattern,
        outer_loads,
        reductions,
    }
}

/// Detect a left-fold reduction chain in a subtree.
struct FoldInfo<'a> {
    op: ScalarBinOp,
    identity: f64,
    /// Terms in order [k=0, k=1, ..., k=depth-1]
    terms: Vec<&'a ScalarExpr>,
}

fn try_detect_fold(expr: &ScalarExpr) -> Option<FoldInfo<'_>> {
    let ScalarExpr::Binary { op, .. } = expr else {
        return None;
    };

    // Walk down the left spine, collecting right children
    let mut terms = Vec::new();
    let mut current: &ScalarExpr = expr;
    while let ScalarExpr::Binary {
        op: inner_op,
        a: left,
        b: right,
    } = current
    {
        if inner_op != op {
            break;
        }
        terms.push(right.as_ref());
        current = left.as_ref();
    }

    // Bottom of spine must be identity literal
    let ScalarExpr::Literal { value: identity } = current else {
        return None;
    };

    // Need at least 2 repeating terms
    if terms.len() < 2 {
        return None;
    }

    // Reverse so terms are in order [k=0, k=1, ...]
    terms.reverse();

    // Verify all terms have identical pattern structure
    let mut dummy = Vec::new();
    let ref_pattern = extract_body_inner(terms[0], &mut dummy);
    for term in &terms[1..] {
        let mut d2 = Vec::new();
        let p = extract_body_inner(term, &mut d2);
        if p != ref_pattern {
            return None;
        }
    }

    Some(FoldInfo {
        op: *op,
        identity: *identity,
        terms,
    })
}

fn extract_inner(
    expr: &ScalarExpr,
    outer_loads: &mut Vec<(GlobalId, usize)>,
    reductions: &mut Vec<ReductionExtraction>,
) -> PatternExpr {
    // Check for left-fold reduction first
    if let Some(fold) = try_detect_fold(expr) {
        let reduce_idx = reductions.len();

        // Extract body pattern from first term
        let mut first_body_loads = Vec::new();
        let body_pattern = extract_body_inner(fold.terms[0], &mut first_body_loads);
        let body_load_count = first_body_loads.len();

        // Collect loads from all terms
        let mut loads_by_k = vec![first_body_loads];
        for term in &fold.terms[1..] {
            let mut term_loads = Vec::new();
            extract_body_inner(term, &mut term_loads);
            loads_by_k.push(term_loads);
        }

        reductions.push(ReductionExtraction {
            depth: fold.terms.len(),
            body_load_count,
            loads_by_k,
        });

        return PatternExpr::Reduce {
            accum_op: fold.op,
            identity_bits: fold.identity.to_bits(),
            body: Box::new(body_pattern),
            reduce_idx,
        };
    }

    // Normal (non-fold) extraction
    match expr {
        ScalarExpr::Element {
            tensor, flat_index, ..
        } => {
            let ordinal = outer_loads.len();
            outer_loads.push((*tensor, *flat_index));
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
            a: Box::new(extract_inner(a, outer_loads, reductions)),
            b: Box::new(extract_inner(b, outer_loads, reductions)),
        },
        ScalarExpr::Unary { op, input } => PatternExpr::Unary {
            op: *op,
            input: Box::new(extract_inner(input, outer_loads, reductions)),
        },
    }
}

/// Extract pattern for a reduction body term (no fold detection).
fn extract_body_inner(
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
            a: Box::new(extract_body_inner(a, loads)),
            b: Box::new(extract_body_inner(b, loads)),
        },
        ScalarExpr::Unary { op, input } => PatternExpr::Unary {
            op: *op,
            input: Box::new(extract_body_inner(input, loads)),
        },
    }
}

// ---------------------------------------------------------------------------
// Affine access recovery
// ---------------------------------------------------------------------------

/// Multi-dimensional affine access:
/// `flat_index = base + sum_d(axis_coeffs[d] * axis_var[d])`
#[derive(Debug, Clone)]
pub struct AffineAccess {
    pub tensor: GlobalId,
    pub base: i64,
    /// Coefficient per output axis
    pub axis_coeffs: Vec<i64>,
}

/// Recovered access info for an embedded reduction.
#[derive(Debug, Clone)]
pub struct ReductionAccess {
    pub depth: usize,
    /// Affine accesses for each load in the reduction body.
    pub accesses: Vec<AffineAccess>,
    /// Per-load coefficient for the reduction variable k.
    pub reduction_coeffs: Vec<i64>,
}

/// A recovered pattern for a group of output bindings.
#[derive(Debug, Clone)]
pub struct RecoveredPattern {
    pub output_tensor: GlobalId,
    pub pattern: PatternExpr,
    /// Output shape (loop extents per axis)
    pub output_shape: Vec<usize>,
    /// Affine accesses for outer (pointwise) loads.
    pub accesses: Vec<AffineAccess>,
    /// Affine access info for each embedded Reduce node.
    pub reductions: Vec<ReductionAccess>,
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
        let shape = output_shapes
            .get(out_tensor)
            .cloned()
            .unwrap_or_else(|| vec![group.len()]);
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
    let ndim = output_shape.len();

    // Extract pattern from first binding
    let first = extract_pattern(&group[0].expr);
    let num_outer_loads = first.outer_loads.len();
    let num_reductions = first.reductions.len();

    // Verify all bindings have the same pattern, collect loads
    let mut all_outer_loads: Vec<Vec<(GlobalId, usize)>> = vec![first.outer_loads];
    let mut all_reduction_loads: Vec<Vec<ReductionExtraction>> = vec![first.reductions];

    for b in &group[1..] {
        let extracted = extract_pattern(&b.expr);
        if extracted.pattern != first.pattern {
            return None;
        }
        all_outer_loads.push(extracted.outer_loads);
        all_reduction_loads.push(extracted.reductions);
    }

    // Decompose output flat indices to multi-dimensional
    let out_multis: Vec<Vec<usize>> = group
        .iter()
        .map(|b| flat_to_multi(b.flat_index, output_shape))
        .collect();

    // Recover outer load affine coefficients
    let mut accesses = Vec::with_capacity(num_outer_loads);
    for pos in 0..num_outer_loads {
        let load_flats: Vec<i64> = all_outer_loads.iter().map(|l| l[pos].1 as i64).collect();
        let (base, axis_coeffs) = solve_affine_coeffs(&out_multis, &load_flats, ndim);
        accesses.push(AffineAccess {
            tensor: all_outer_loads[0][pos].0,
            base,
            axis_coeffs,
        });
    }

    // Recover reduction affine coefficients
    let mut reductions = Vec::with_capacity(num_reductions);
    for r in 0..num_reductions {
        let depth = all_reduction_loads[0][r].depth;
        let body_load_count = all_reduction_loads[0][r].body_load_count;

        let mut red_accesses = Vec::with_capacity(body_load_count);
        let mut reduction_coeffs = Vec::with_capacity(body_load_count);

        for pos in 0..body_load_count {
            // Axis coefficients: use k=0 loads across bindings
            let load_flats: Vec<i64> = all_reduction_loads
                .iter()
                .map(|rl| rl[r].loads_by_k[0][pos].1 as i64)
                .collect();
            let (base, axis_coeffs) = solve_affine_coeffs(&out_multis, &load_flats, ndim);

            // Reduction coefficient: delta between k=0 and k=1 within first binding
            let red_coeff = if depth > 1 {
                let k0 = all_reduction_loads[0][r].loads_by_k[0][pos].1 as i64;
                let k1 = all_reduction_loads[0][r].loads_by_k[1][pos].1 as i64;
                k1 - k0
            } else {
                0
            };

            red_accesses.push(AffineAccess {
                tensor: all_reduction_loads[0][r].loads_by_k[0][pos].0,
                base,
                axis_coeffs,
            });
            reduction_coeffs.push(red_coeff);
        }

        reductions.push(ReductionAccess {
            depth,
            accesses: red_accesses,
            reduction_coeffs,
        });
    }

    Some(RecoveredPattern {
        output_tensor: out_tensor,
        pattern: first.pattern,
        output_shape: output_shape.to_vec(),
        accesses,
        reductions,
    })
}

// ---------------------------------------------------------------------------
// Affine solver
// ---------------------------------------------------------------------------

/// Solve for affine coefficients: load_flat = base + sum(coeff[d] * axis[d])
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

    for d in 0..ndim {
        for (i, multi) in out_multis.iter().enumerate().skip(1) {
            let axis_delta = multi[d] as i64 - ref_multi[d] as i64;
            if axis_delta != 0 {
                let load_delta = load_flats[i] - ref_load;
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

    let base = ref_load
        - coeffs
            .iter()
            .zip(ref_multi.iter())
            .map(|(&c, &m)| c * m as i64)
            .sum::<i64>();

    (base, coeffs)
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::common::v2_frontend::*;

    #[test]
    fn test_pattern_extraction_pointwise() {
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
        let result = extract_pattern(&expr);
        assert_eq!(result.outer_loads.len(), 2);
        assert_eq!(result.outer_loads[0], (GlobalId(1), 5));
        assert_eq!(result.outer_loads[1], (GlobalId(2), 10));
        assert!(result.reductions.is_empty());
        match &result.pattern {
            PatternExpr::Binary {
                op: ScalarBinOp::Add,
                a,
                b,
            } => {
                assert!(matches!(
                    a.as_ref(),
                    PatternExpr::Load {
                        load_ordinal: 0,
                        ..
                    }
                ));
                assert!(matches!(
                    b.as_ref(),
                    PatternExpr::Load {
                        load_ordinal: 1,
                        ..
                    }
                ));
            }
            _ => panic!("Expected Binary::Add"),
        }
    }

    #[test]
    fn test_reduction_extraction() {
        // sum_k(a[k] * b[k]) with k=3
        let mut expr = ScalarExpr::Literal { value: 0.0 };
        for k in 0..3usize {
            let prod = ScalarExpr::Binary {
                op: ScalarBinOp::Mul,
                a: Box::new(ScalarExpr::Element {
                    tensor: GlobalId(1),
                    flat_index: k,
                }),
                b: Box::new(ScalarExpr::Element {
                    tensor: GlobalId(2),
                    flat_index: k,
                }),
            };
            expr = ScalarExpr::Binary {
                op: ScalarBinOp::Add,
                a: Box::new(expr),
                b: Box::new(prod),
            };
        }
        let result = extract_pattern(&expr);
        assert!(result.outer_loads.is_empty());
        assert_eq!(result.reductions.len(), 1);
        assert_eq!(result.reductions[0].depth, 3);
        assert_eq!(result.reductions[0].body_load_count, 2);
        match &result.pattern {
            PatternExpr::Reduce {
                accum_op: ScalarBinOp::Add,
                reduce_idx: 0,
                ..
            } => {}
            _ => panic!("Expected Reduce"),
        }
    }

    #[test]
    fn test_fused_reduction_extraction() {
        // tanh(sum_k(a[k] * b[k]) + c[0])
        let mut sum_expr = ScalarExpr::Literal { value: 0.0 };
        for k in 0..3usize {
            let prod = ScalarExpr::Binary {
                op: ScalarBinOp::Mul,
                a: Box::new(ScalarExpr::Element {
                    tensor: GlobalId(1),
                    flat_index: k,
                }),
                b: Box::new(ScalarExpr::Element {
                    tensor: GlobalId(2),
                    flat_index: k,
                }),
            };
            sum_expr = ScalarExpr::Binary {
                op: ScalarBinOp::Add,
                a: Box::new(sum_expr),
                b: Box::new(prod),
            };
        }
        let expr = ScalarExpr::Unary {
            op: ScalarUnaryOp::Tanh,
            input: Box::new(ScalarExpr::Binary {
                op: ScalarBinOp::Add,
                a: Box::new(sum_expr),
                b: Box::new(ScalarExpr::Element {
                    tensor: GlobalId(3),
                    flat_index: 0,
                }),
            }),
        };

        let result = extract_pattern(&expr);
        assert_eq!(result.outer_loads.len(), 1); // c[0]
        assert_eq!(result.outer_loads[0].0, GlobalId(3));
        assert_eq!(result.reductions.len(), 1);
        assert_eq!(result.reductions[0].depth, 3);

        match &result.pattern {
            PatternExpr::Unary {
                op: ScalarUnaryOp::Tanh,
                input,
            } => match input.as_ref() {
                PatternExpr::Binary {
                    op: ScalarBinOp::Add,
                    a,
                    b,
                } => {
                    assert!(matches!(a.as_ref(), PatternExpr::Reduce { .. }));
                    assert!(matches!(
                        b.as_ref(),
                        PatternExpr::Load {
                            load_ordinal: 0,
                            ..
                        }
                    ));
                }
                _ => panic!("Expected Binary::Add"),
            },
            _ => panic!("Expected Unary::Tanh"),
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
                    a: Box::new(ScalarExpr::Element {
                        tensor: a,
                        flat_index: i,
                    }),
                    b: Box::new(ScalarExpr::Element {
                        tensor: b,
                        flat_index: i,
                    }),
                },
            })
            .collect();

        let mut shapes = HashMap::new();
        shapes.insert(out, vec![4]);
        let patterns = recover_patterns(&bindings, &shapes);
        assert_eq!(patterns.len(), 1);
        let p = &patterns[0];
        assert!(p.reductions.is_empty());
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
                    a: Box::new(ScalarExpr::Element {
                        tensor: a,
                        flat_index: i,
                    }),
                    b: Box::new(ScalarExpr::Element {
                        tensor: b,
                        flat_index: 0,
                    }),
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

        // No outer loads — entire pattern is a reduction
        assert!(p.accesses.is_empty());
        assert_eq!(p.reductions.len(), 1);
        let red = &p.reductions[0];
        assert_eq!(red.depth, k);
        assert_eq!(red.accesses.len(), 2);
        // A access: axis_coeffs=[K, 0], red_coeff=1
        assert_eq!(red.accesses[0].axis_coeffs, vec![k as i64, 0]);
        assert_eq!(red.reduction_coeffs[0], 1);
        // B access: axis_coeffs=[0, 1], red_coeff=N
        assert_eq!(red.accesses[1].axis_coeffs, vec![0, 1]);
        assert_eq!(red.reduction_coeffs[1], n as i64);
    }
}
