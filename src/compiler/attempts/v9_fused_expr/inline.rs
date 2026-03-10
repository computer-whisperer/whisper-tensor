//! Expression inlining: fuse intermediate tensors by substituting their
//! expression trees into downstream consumers.
//!
//! Given a set of OutputBindings from the v2 frontend, this pass:
//! 1. Identifies which tensors are "intermediate" (produced by some binding
//!    and consumed by Element references in other bindings) vs "final"
//!    (graph outputs or needed for external consumption).
//! 2. For intermediates, substitutes the producing expression inline at
//!    every point of use.
//! 3. After inlining, final output bindings reference only graph inputs
//!    and constants — all intermediate tensors have been erased.
//!
//! This is the fusion pass: if mul_out is only consumed by neg, then
//! neg_out[i] = Neg(Element(mul_out, i)) becomes
//! neg_out[i] = Neg(Mul(Element(a, i), Element(b, i))).

use crate::compiler::common::v2_frontend::{OutputBinding, ScalarExpr};
use crate::graph::GlobalId;
use std::collections::{HashMap, HashSet};

/// Result of the inlining pass.
pub struct InlinedGraph {
    /// Bindings for final output tensors only, with all intermediates inlined.
    pub bindings: Vec<OutputBinding>,
    /// The set of tensor IDs that are "leaf" tensors (graph inputs + constants).
    pub leaf_tensors: HashSet<GlobalId>,
}

/// Inline intermediate tensors, producing fully-expanded expression trees
/// for each graph output element.
///
/// `final_outputs` is the set of tensor IDs that must be materialized
/// (graph outputs). Everything else is eligible for inlining.
pub fn inline_intermediates(
    bindings: &[OutputBinding],
    final_outputs: &HashSet<GlobalId>,
) -> InlinedGraph {
    // Phase 1: Build lookup from (tensor_id, flat_index) -> expr for all bindings
    let mut expr_map: HashMap<(GlobalId, usize), &ScalarExpr> = HashMap::new();
    for b in bindings {
        expr_map.insert((b.output_tensor, b.flat_index), &b.expr);
    }

    // Which tensors have bindings (i.e., are computed, not leaves)?
    let produced_tensors: HashSet<GlobalId> = bindings.iter().map(|b| b.output_tensor).collect();

    // Phase 2: For each final output binding, recursively inline
    let mut result_bindings = Vec::new();
    for b in bindings {
        if !final_outputs.contains(&b.output_tensor) {
            continue;
        }
        let expanded = inline_expr(&b.expr, &expr_map, &produced_tensors, final_outputs);
        result_bindings.push(OutputBinding {
            output_tensor: b.output_tensor,
            flat_index: b.flat_index,
            expr: expanded,
        });
    }

    // Collect leaf tensors: everything referenced by Element that isn't produced
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
    final_outputs: &HashSet<GlobalId>,
) -> ScalarExpr {
    match expr {
        ScalarExpr::Element {
            tensor, flat_index, ..
        } => {
            // If this tensor is produced (has bindings) and is NOT a final output,
            // it's an intermediate — inline it.
            if produced.contains(tensor) && !final_outputs.contains(tensor) {
                if let Some(producing_expr) = expr_map.get(&(*tensor, *flat_index)) {
                    // Recursively inline the producing expression too
                    return inline_expr(producing_expr, expr_map, produced, final_outputs);
                }
            }
            // Leaf tensor or final output — keep the Element reference
            expr.clone()
        }
        ScalarExpr::Literal { .. } => expr.clone(),
        ScalarExpr::Binary { op, a, b } => ScalarExpr::Binary {
            op: *op,
            a: Box::new(inline_expr(a, expr_map, produced, final_outputs)),
            b: Box::new(inline_expr(b, expr_map, produced, final_outputs)),
        },
        ScalarExpr::Unary { op, input } => ScalarExpr::Unary {
            op: *op,
            input: Box::new(inline_expr(input, expr_map, produced, final_outputs)),
        },
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
            // mul bindings
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
            // neg bindings
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
        let result = inline_intermediates(&bindings, &final_outputs);

        // mul_out should be inlined away — only neg_out bindings remain
        assert_eq!(result.bindings.len(), 4);
        for rb in &result.bindings {
            assert_eq!(rb.output_tensor, neg_out);
            // Expression should be Neg(Mul(Element(a,i), Element(b,i)))
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

        // Leaves should be {a, b} only — mul_out is gone
        assert!(result.leaf_tensors.contains(&a));
        assert!(result.leaf_tensors.contains(&b));
        assert!(!result.leaf_tensors.contains(&mul_out));
    }

    #[test]
    fn test_inline_preserves_final_outputs() {
        // If mul_out IS a final output, it should not be inlined
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

        // Both are final outputs
        let final_outputs: HashSet<GlobalId> = [mul_out, neg_out].into_iter().collect();
        let result = inline_intermediates(&bindings, &final_outputs);

        // Should have 4 bindings: 2 for mul_out, 2 for neg_out
        assert_eq!(result.bindings.len(), 4);
        let mul_bindings: Vec<_> = result
            .bindings
            .iter()
            .filter(|b| b.output_tensor == mul_out)
            .collect();
        let neg_bindings: Vec<_> = result
            .bindings
            .iter()
            .filter(|b| b.output_tensor == neg_out)
            .collect();
        assert_eq!(mul_bindings.len(), 2);
        assert_eq!(neg_bindings.len(), 2);

        // neg_out should still reference mul_out (not inlined)
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
}
