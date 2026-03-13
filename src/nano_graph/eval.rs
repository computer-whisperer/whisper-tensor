//! Naive scalar evaluator for NanoGraph integrity checks.
//!
//! Walks every atom in group order, evaluates the scalar op, and stores
//! the result. This is intentionally simple and slow — it exists only to
//! verify that the lowered NanoGraph produces the same values as the
//! original MilliOpGraph.
//!
//! Precision semantics come from each ScalarOp variant's compute_dtype
//! and output_dtype fields. Inputs are cast to compute_dtype, the op
//! executes at that precision, then the result is cast to output_dtype.

use std::collections::HashMap;

use crate::numeric_scalar::NumericScalar;

use super::ops::{ScalarBinOp, ScalarOp, ScalarUnaryOp};
use super::pattern::{AtomId, NanoGraph};

/// Flat storage of one NumericScalar per atom.
pub struct NanoEval {
    values: Vec<NumericScalar>,
}

impl NanoEval {
    /// Evaluate a NanoGraph with no symbolic dimensions.
    ///
    /// `overrides` maps atom index → NumericScalar for input atoms whose
    /// Literal(0.0) placeholder should be replaced with an actual value.
    /// Groups are walked in insertion order (topological).
    pub fn eval(graph: &NanoGraph, overrides: &HashMap<u32, NumericScalar>) -> Self {
        Self::eval_inner(graph, overrides, false)
    }

    /// Like eval, but prints the first group that produces NaN/Inf.
    pub fn eval_debug(graph: &NanoGraph, overrides: &HashMap<u32, NumericScalar>) -> Self {
        Self::eval_inner(graph, overrides, true)
    }

    fn eval_inner(
        graph: &NanoGraph,
        overrides: &HashMap<u32, NumericScalar>,
        debug_nan: bool,
    ) -> Self {
        let num_atoms = graph.num_atoms() as usize;
        let mut values: Vec<NumericScalar> = vec![NumericScalar::F32(0.0); num_atoms];
        let mut nan_reported = false;

        for (group_idx, group) in graph.groups().iter().enumerate() {
            let is_reduce = group.op.is_reduce();

            for i in 0..group.count {
                let atom_idx = group.base_id.0 + i;

                if is_reduce {
                    assert_eq!(
                        group.reduce_dims.len(),
                        1,
                        "Only single reduce_dim supported"
                    );
                    let rd = group.reduce_dims[0];
                    let bound = graph
                        .sym_dim_bounds
                        .get(&rd)
                        .copied()
                        .expect("Reduce dim must have a bound");

                    let (compute_dtype, output_dtype) = match &group.op {
                        ScalarOp::ReduceSum {
                            compute_dtype,
                            output_dtype,
                        } => (*compute_dtype, *output_dtype),
                        ScalarOp::ReduceMax {
                            compute_dtype,
                            output_dtype,
                        } => (*compute_dtype, *output_dtype),
                        _ => unreachable!(),
                    };

                    let mut acc = match &group.op {
                        ScalarOp::ReduceSum { .. } => NumericScalar::zero_of(compute_dtype),
                        ScalarOp::ReduceMax { .. } => NumericScalar::neg_infinity_of(compute_dtype),
                        _ => unreachable!(),
                    };

                    for k in 0..bound as u32 {
                        let src = group.inputs[0].resolve(i, k);
                        let val = values[src.0 as usize].cast_to(compute_dtype);
                        acc = match &group.op {
                            ScalarOp::ReduceSum { .. } => acc.add(&val),
                            ScalarOp::ReduceMax { .. } => acc.scalar_max(&val),
                            _ => unreachable!(),
                        };
                    }
                    if debug_nan && !nan_reported && !acc.to_f64().is_finite() {
                        eprintln!(
                            "[NaN-debug] group {} (reduce {:?}): atom {} = {:?}",
                            group_idx, group.op, atom_idx, acc
                        );
                        nan_reported = true;
                    }
                    // Cast accumulated result to output dtype before storing.
                    values[atom_idx as usize] = acc.cast_to(output_dtype);
                } else {
                    let val = match &group.op {
                        ScalarOp::Literal(scalar) => {
                            if let Some(ov) = overrides.get(&atom_idx) {
                                // Override: cast to the literal's dtype.
                                ov.cast_to(scalar.dtype())
                            } else {
                                scalar.clone()
                            }
                        }
                        ScalarOp::Identity {
                            compute_dtype,
                            output_dtype,
                        } => {
                            let src = group.inputs[0].resolve(i, 0);
                            let x = values[src.0 as usize].cast_to(*compute_dtype);
                            x.cast_to(*output_dtype)
                        }
                        ScalarOp::Binary {
                            op,
                            compute_dtype,
                            output_dtype,
                        } => {
                            let a = values[group.inputs[0].resolve(i, 0).0 as usize]
                                .cast_to(*compute_dtype);
                            let b = values[group.inputs[1].resolve(i, 0).0 as usize]
                                .cast_to(*compute_dtype);
                            let result = match op {
                                ScalarBinOp::Add => a.add(&b),
                                ScalarBinOp::Sub => a.sub(&b),
                                ScalarBinOp::Mul => a.mul(&b),
                                ScalarBinOp::Div => a.div(&b),
                                ScalarBinOp::Max => a.scalar_max(&b),
                                ScalarBinOp::Min => a.scalar_min(&b),
                                ScalarBinOp::Mod => a.modulo(&b),
                                ScalarBinOp::Pow => a.pow(&b),
                            };
                            result.cast_to(*output_dtype)
                        }
                        ScalarOp::Unary {
                            op,
                            compute_dtype,
                            output_dtype,
                        } => {
                            let x = values[group.inputs[0].resolve(i, 0).0 as usize]
                                .cast_to(*compute_dtype);
                            let result = match op {
                                ScalarUnaryOp::Neg => x.neg(),
                                ScalarUnaryOp::Abs => x.abs(),
                                ScalarUnaryOp::Exp => x.exp(),
                                ScalarUnaryOp::Ln => x.ln(),
                                ScalarUnaryOp::Sqrt => x.sqrt(),
                                ScalarUnaryOp::Reciprocal => x.recip(),
                                ScalarUnaryOp::Tanh => x.tanh(),
                                ScalarUnaryOp::Floor => x.floor(),
                                ScalarUnaryOp::Ceil => x.ceil(),
                            };
                            result.cast_to(*output_dtype)
                        }
                        ScalarOp::Select {
                            compute_dtype,
                            output_dtype,
                        } => {
                            let cond = values[group.inputs[0].resolve(i, 0).0 as usize]
                                .cast_to(*compute_dtype);
                            let result = if cond.is_nonzero() {
                                values[group.inputs[1].resolve(i, 0).0 as usize]
                                    .cast_to(*compute_dtype)
                            } else {
                                values[group.inputs[2].resolve(i, 0).0 as usize]
                                    .cast_to(*compute_dtype)
                            };
                            result.cast_to(*output_dtype)
                        }
                        ScalarOp::ReduceSum { .. } | ScalarOp::ReduceMax { .. } => unreachable!(),
                    };
                    if debug_nan && !nan_reported && !val.to_f64().is_finite() {
                        let input_detail: Vec<String> = group
                            .inputs
                            .iter()
                            .enumerate()
                            .map(|(j, inp)| {
                                let src = inp.resolve(i, 0);
                                format!("inp[{}]=atom{}={:?}", j, src.0, values[src.0 as usize])
                            })
                            .collect();
                        eprintln!(
                            "[NaN-debug] group {} ({:?}): atom {} = {:?} inputs: {}",
                            group_idx,
                            group.op,
                            atom_idx,
                            val,
                            input_detail.join(", ")
                        );
                        nan_reported = true;
                    }
                    values[atom_idx as usize] = val;
                }
            }
        }

        NanoEval { values }
    }

    /// Get the value of an atom as f64 (for comparison).
    pub fn get(&self, id: AtomId) -> f64 {
        self.values[id.0 as usize].to_f64()
    }

    /// Get the raw NumericScalar for an atom.
    pub fn get_scalar(&self, id: AtomId) -> &NumericScalar {
        &self.values[id.0 as usize]
    }

    /// Get values for a contiguous range of atoms as f64.
    pub fn get_range(&self, base: AtomId, count: u32) -> Vec<f64> {
        let start = base.0 as usize;
        self.values[start..start + count as usize]
            .iter()
            .map(|v| v.to_f64())
            .collect()
    }
}
