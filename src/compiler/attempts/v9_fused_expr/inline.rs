//! Expression inlining: fuse intermediate tensors by substituting their
//! expression trees into downstream consumers.
//!
//! Given a set of OutputBindings from the v2 frontend, this pass:
//! 1. Identifies which tensors are "intermediate" (produced by some binding
//!    and consumed by Element references in other bindings) vs "final"
//!    (graph outputs or needed for external consumption).
//! 2. Detects materialization boundaries: non-leaf tensors consumed by
//!    reduction (left-fold) tensors must be materialized to prevent
//!    nested reductions.
//! 3. For remaining intermediates, substitutes the producing expression
//!    inline at every point of use.

use crate::compiler::common::v2_frontend::{OutputBinding, ScalarExpr, ScalarUnaryOp};
use crate::dtype::DType;
use crate::graph::GlobalId;
use std::collections::{HashMap, HashSet};

/// Result of the inlining pass.
pub struct InlinedGraph {
    /// Bindings for all materialized tensors (final outputs + boundary points),
    /// with eligible intermediates inlined.
    pub bindings: Vec<OutputBinding>,
    /// The set of tensor IDs that are "leaf" tensors (graph inputs + constants).
    pub leaf_tensors: HashSet<GlobalId>,
}

/// Inline intermediate tensors, producing fully-expanded expression trees.
///
/// `final_outputs` is the set of tensor IDs that must be materialized
/// (graph outputs). Additional tensors may be materialized automatically
/// to prevent nested reductions.
pub fn inline_intermediates(
    bindings: &[OutputBinding],
    final_outputs: &HashSet<GlobalId>,
    dtypes: &HashMap<GlobalId, DType>,
) -> InlinedGraph {
    let produced_tensors: HashSet<GlobalId> = bindings.iter().map(|b| b.output_tensor).collect();

    // Detect fold (reduction) tensors
    let fold_tensors = find_fold_tensors(bindings);

    // Build materialization set: final outputs + reduction boundary points.
    // For each fold tensor, any non-leaf tensor it references must be
    // materialized to prevent that tensor's expression (which may itself
    // contain a fold after inlining) from ending up inside this fold's body.
    let mut materialize: HashSet<GlobalId> = final_outputs.clone();
    for b in bindings {
        if fold_tensors.contains(&b.output_tensor) {
            collect_produced_references(&b.expr, &produced_tensors, &mut materialize);
        }
    }

    // Build lookup from (tensor_id, flat_index) -> expr
    let mut expr_map: HashMap<(GlobalId, usize), &ScalarExpr> = HashMap::new();
    for b in bindings {
        expr_map.insert((b.output_tensor, b.flat_index), &b.expr);
    }

    // For each materialized tensor, produce inlined bindings
    let mut result_bindings = Vec::new();
    for b in bindings {
        if !materialize.contains(&b.output_tensor) {
            continue;
        }
        let expanded = inline_expr(&b.expr, &expr_map, &produced_tensors, &materialize, dtypes);
        result_bindings.push(OutputBinding {
            output_tensor: b.output_tensor,
            flat_index: b.flat_index,
            expr: expanded,
        });
    }

    let mut leaf_tensors = HashSet::new();
    for b in &result_bindings {
        collect_leaves(&b.expr, &mut leaf_tensors);
    }

    InlinedGraph {
        bindings: result_bindings,
        leaf_tensors,
    }
}

/// Recursively inline an expression, substituting Element references to
/// intermediate tensors with their producing expressions.
fn inline_expr(
    expr: &ScalarExpr,
    expr_map: &HashMap<(GlobalId, usize), &ScalarExpr>,
    produced: &HashSet<GlobalId>,
    materialize: &HashSet<GlobalId>,
    dtypes: &HashMap<GlobalId, DType>,
) -> ScalarExpr {
    match expr {
        ScalarExpr::Element {
            tensor, flat_index, ..
        } => {
            // If this tensor is produced and NOT materialized, inline it.
            if produced.contains(tensor) && !materialize.contains(tensor)
                && let Some(producing_expr) = expr_map.get(&(*tensor, *flat_index))
            {
                let inlined = inline_expr(producing_expr, expr_map, produced, materialize, dtypes);
                // If the inlined tensor is BF16, wrap with RoundBf16 to preserve
                // the dtype boundary that would otherwise be lost during fusion.
                if dtypes.get(tensor) == Some(&DType::BF16) {
                    return ScalarExpr::Unary {
                        op: ScalarUnaryOp::RoundBf16,
                        input: Box::new(inlined),
                    };
                }
                return inlined;
            }
            expr.clone()
        }
        ScalarExpr::Literal { .. } => expr.clone(),
        ScalarExpr::Binary { op, a, b } => ScalarExpr::Binary {
            op: *op,
            a: Box::new(inline_expr(a, expr_map, produced, materialize, dtypes)),
            b: Box::new(inline_expr(b, expr_map, produced, materialize, dtypes)),
        },
        ScalarExpr::Unary { op, input } => ScalarExpr::Unary {
            op: *op,
            input: Box::new(inline_expr(input, expr_map, produced, materialize, dtypes)),
        },
    }
}

// ---------------------------------------------------------------------------
// Fold detection
// ---------------------------------------------------------------------------

/// Identify tensors whose bindings contain a left-fold reduction pattern.
fn find_fold_tensors(bindings: &[OutputBinding]) -> HashSet<GlobalId> {
    let mut fold_tensors = HashSet::new();
    for b in bindings {
        if contains_left_fold(&b.expr) {
            fold_tensors.insert(b.output_tensor);
        }
    }
    fold_tensors
}

/// Check if an expression tree contains a left-fold pattern anywhere.
fn contains_left_fold(expr: &ScalarExpr) -> bool {
    if is_left_fold(expr) {
        return true;
    }
    match expr {
        ScalarExpr::Binary { a, b, .. } => contains_left_fold(a) || contains_left_fold(b),
        ScalarExpr::Unary { input, .. } => contains_left_fold(input),
        _ => false,
    }
}

/// Check if this exact node is the root of a left-fold:
/// Binary(op, Binary(op, Binary(op, Literal, term), term), term)
/// with depth >= 2.
fn is_left_fold(expr: &ScalarExpr) -> bool {
    let ScalarExpr::Binary { op, .. } = expr else {
        return false;
    };
    let mut depth = 0;
    let mut current = expr;
    while let ScalarExpr::Binary {
        op: inner_op,
        a: left,
        ..
    } = current
    {
        if inner_op != op {
            break;
        }
        depth += 1;
        current = left.as_ref();
    }
    matches!(current, ScalarExpr::Literal { .. }) && depth >= 2
}

/// Collect all tensor IDs referenced by Element nodes that are in the
/// `produced` set (i.e., non-leaf tensors).
fn collect_produced_references(
    expr: &ScalarExpr,
    produced: &HashSet<GlobalId>,
    out: &mut HashSet<GlobalId>,
) {
    match expr {
        ScalarExpr::Element { tensor, .. } => {
            if produced.contains(tensor) {
                out.insert(*tensor);
            }
        }
        ScalarExpr::Binary { a, b, .. } => {
            collect_produced_references(a, produced, out);
            collect_produced_references(b, produced, out);
        }
        ScalarExpr::Unary { input, .. } => {
            collect_produced_references(input, produced, out);
        }
        ScalarExpr::Literal { .. } => {}
    }
}

/// Collect all tensor IDs referenced by Element nodes in an expression.
fn collect_leaves(expr: &ScalarExpr, out: &mut HashSet<GlobalId>) {
    match expr {
        ScalarExpr::Element { tensor, .. } => {
            out.insert(*tensor);
        }
        ScalarExpr::Literal { .. } => {}
        ScalarExpr::Binary { a, b, .. } => {
            collect_leaves(a, out);
            collect_leaves(b, out);
        }
        ScalarExpr::Unary { input, .. } => {
            collect_leaves(input, out);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::common::v2_frontend::*;

    #[test]
    fn test_inline_chain() {
        // Simulate: mul_out[i] = a[i] * b[i], neg_out[i] = -mul_out[i]
        let a = GlobalId(1);
        let b = GlobalId(2);
        let mul_out = GlobalId(3);
        let neg_out = GlobalId(4);

        let mut bindings = Vec::new();
        for i in 0..4 {
            bindings.push(OutputBinding {
                output_tensor: mul_out,
                flat_index: i,
                expr: ScalarExpr::Binary {
                    op: ScalarBinOp::Mul,
                    a: Box::new(ScalarExpr::Element {
                        tensor: a,
                        flat_index: i,
                    }),
                    b: Box::new(ScalarExpr::Element {
                        tensor: b,
                        flat_index: i,
                    }),
                },
            });
            bindings.push(OutputBinding {
                output_tensor: neg_out,
                flat_index: i,
                expr: ScalarExpr::Unary {
                    op: ScalarUnaryOp::Neg,
                    input: Box::new(ScalarExpr::Element {
                        tensor: mul_out,
                        flat_index: i,
                    }),
                },
            });
        }

        let final_outputs: HashSet<GlobalId> = [neg_out].into_iter().collect();
        let result = inline_intermediates(&bindings, &final_outputs, &HashMap::new());

        assert_eq!(result.bindings.len(), 4);
        for rb in &result.bindings {
            assert_eq!(rb.output_tensor, neg_out);
            match &rb.expr {
                ScalarExpr::Unary {
                    op: ScalarUnaryOp::Neg,
                    input,
                } => match input.as_ref() {
                    ScalarExpr::Binary {
                        op: ScalarBinOp::Mul,
                        a: ea,
                        b: eb,
                    } => {
                        assert!(matches!(ea.as_ref(), ScalarExpr::Element { tensor, .. } if *tensor == a));
                        assert!(matches!(eb.as_ref(), ScalarExpr::Element { tensor, .. } if *tensor == b));
                    }
                    other => panic!("Expected Binary::Mul, got {:?}", other),
                },
                other => panic!("Expected Unary::Neg, got {:?}", other),
            }
        }

        assert!(result.leaf_tensors.contains(&a));
        assert!(result.leaf_tensors.contains(&b));
        assert!(!result.leaf_tensors.contains(&mul_out));
    }

    #[test]
    fn test_inline_preserves_final_outputs() {
        let a = GlobalId(1);
        let b = GlobalId(2);
        let mul_out = GlobalId(3);
        let neg_out = GlobalId(4);

        let mut bindings = Vec::new();
        for i in 0..2 {
            bindings.push(OutputBinding {
                output_tensor: mul_out,
                flat_index: i,
                expr: ScalarExpr::Binary {
                    op: ScalarBinOp::Mul,
                    a: Box::new(ScalarExpr::Element {
                        tensor: a,
                        flat_index: i,
                    }),
                    b: Box::new(ScalarExpr::Element {
                        tensor: b,
                        flat_index: i,
                    }),
                },
            });
            bindings.push(OutputBinding {
                output_tensor: neg_out,
                flat_index: i,
                expr: ScalarExpr::Unary {
                    op: ScalarUnaryOp::Neg,
                    input: Box::new(ScalarExpr::Element {
                        tensor: mul_out,
                        flat_index: i,
                    }),
                },
            });
        }

        let final_outputs: HashSet<GlobalId> = [mul_out, neg_out].into_iter().collect();
        let result = inline_intermediates(&bindings, &final_outputs, &HashMap::new());

        assert_eq!(result.bindings.len(), 4);
        let neg_bindings: Vec<_> = result
            .bindings
            .iter()
            .filter(|b| b.output_tensor == neg_out)
            .collect();
        for nb in &neg_bindings {
            match &nb.expr {
                ScalarExpr::Unary { input, .. } => {
                    assert!(
                        matches!(input.as_ref(), ScalarExpr::Element { tensor, .. } if *tensor == mul_out)
                    );
                }
                _ => panic!("Expected Unary"),
            }
        }
    }

    #[test]
    fn test_materialization_boundary() {
        // Simulate two-layer matmul:
        // mm1[i] = 0 + a[0]*w1[0] + a[1]*w1[1] (fold, K=2)
        // act1[i] = tanh(mm1[i])
        // mm2[i] = 0 + act1[0]*w2[0] + act1[1]*w2[1] (fold, K=2)
        // out[i] = mm2[i]
        let a = GlobalId(1);
        let w1 = GlobalId(2);
        let w2 = GlobalId(3);
        let mm1 = GlobalId(10);
        let act1 = GlobalId(11);
        let mm2 = GlobalId(12);
        let out = GlobalId(13);

        let mut bindings = Vec::new();

        // mm1 bindings (fold: 0 + a[0]*w1[0] + a[1]*w1[1])
        for i in 0..2 {
            let mut expr = ScalarExpr::Literal { value: 0.0 };
            for k in 0..2 {
                let prod = ScalarExpr::Binary {
                    op: ScalarBinOp::Mul,
                    a: Box::new(ScalarExpr::Element {
                        tensor: a,
                        flat_index: i * 2 + k,
                    }),
                    b: Box::new(ScalarExpr::Element {
                        tensor: w1,
                        flat_index: k * 2 + i,
                    }),
                };
                expr = ScalarExpr::Binary {
                    op: ScalarBinOp::Add,
                    a: Box::new(expr),
                    b: Box::new(prod),
                };
            }
            bindings.push(OutputBinding {
                output_tensor: mm1,
                flat_index: i,
                expr,
            });
        }

        // act1 = tanh(mm1)
        for i in 0..2 {
            bindings.push(OutputBinding {
                output_tensor: act1,
                flat_index: i,
                expr: ScalarExpr::Unary {
                    op: ScalarUnaryOp::Tanh,
                    input: Box::new(ScalarExpr::Element {
                        tensor: mm1,
                        flat_index: i,
                    }),
                },
            });
        }

        // mm2 bindings (fold: 0 + act1[0]*w2[0] + act1[1]*w2[1])
        for i in 0..2 {
            let mut expr = ScalarExpr::Literal { value: 0.0 };
            for k in 0..2 {
                let prod = ScalarExpr::Binary {
                    op: ScalarBinOp::Mul,
                    a: Box::new(ScalarExpr::Element {
                        tensor: act1,
                        flat_index: i * 2 + k,
                    }),
                    b: Box::new(ScalarExpr::Element {
                        tensor: w2,
                        flat_index: k * 2 + i,
                    }),
                };
                expr = ScalarExpr::Binary {
                    op: ScalarBinOp::Add,
                    a: Box::new(expr),
                    b: Box::new(prod),
                };
            }
            bindings.push(OutputBinding {
                output_tensor: mm2,
                flat_index: i,
                expr,
            });
        }

        // out = mm2 (identity)
        for i in 0..2 {
            bindings.push(OutputBinding {
                output_tensor: out,
                flat_index: i,
                expr: ScalarExpr::Element {
                    tensor: mm2,
                    flat_index: i,
                },
            });
        }

        let final_outputs: HashSet<GlobalId> = [out].into_iter().collect();
        let result = inline_intermediates(&bindings, &final_outputs, &HashMap::new());

        // act1 should be materialized (referenced by mm2, which is a fold)
        let materialized_tensors: HashSet<GlobalId> =
            result.bindings.iter().map(|b| b.output_tensor).collect();
        assert!(
            materialized_tensors.contains(&act1),
            "act1 should be materialized as a reduction boundary"
        );
        assert!(
            materialized_tensors.contains(&out),
            "out should be materialized as a final output"
        );

        // act1's expression should contain the mm1 fold inlined
        // (mm1 is not materialized, so it gets inlined into act1)
        let act1_binding = result
            .bindings
            .iter()
            .find(|b| b.output_tensor == act1)
            .unwrap();
        // Should be: Tanh(fold(...)) — the fold from mm1 is inlined
        assert!(
            contains_left_fold(&act1_binding.expr),
            "act1 should have mm1's fold inlined"
        );

        // out's expression should reference act1 (materialized) not the fold
        let out_binding = result
            .bindings
            .iter()
            .find(|b| b.output_tensor == out)
            .unwrap();
        // mm2 is inlined into out, but its body references act1 (materialized)
        // So out should be: fold(Element(act1, ...), Element(w2, ...))
        assert!(
            contains_left_fold(&out_binding.expr),
            "out should have mm2's fold inlined"
        );
        // out's fold body should reference act1, not the raw inputs.
        // Verify by checking out's expression directly.
        let mut out_leaves = HashSet::new();
        collect_leaves(&out_binding.expr, &mut out_leaves);
        assert!(
            out_leaves.contains(&act1),
            "out should reference act1 (materialized)"
        );
        assert!(
            !out_leaves.contains(&a),
            "out should not reference a directly (it's inside act1)"
        );
    }
}
