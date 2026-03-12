//! Naive scalar evaluator for NanoGraph integrity checks.
//!
//! Walks every atom in group order, evaluates the scalar op, and stores
//! the result. This is intentionally simple and slow — it exists only to
//! verify that the lowered NanoGraph produces the same values as the
//! original MilliOpGraph.
//!
//! All arithmetic is performed in the group's declared dtype, matching
//! the precision semantics of the MilliOpGraph interpreter.

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
        // Initialize all atoms to F32 zero; actual dtype comes from the group.
        let mut values: Vec<NumericScalar> = vec![NumericScalar::F32(0.0); num_atoms];
        let mut nan_reported = false;

        for (group_idx, group) in graph.groups().iter().enumerate() {
            let is_reduce = matches!(group.op, ScalarOp::ReduceSum | ScalarOp::ReduceMax);
            let dtype = group.dtype;

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

                    let mut acc = match group.op {
                        ScalarOp::ReduceSum => NumericScalar::zero_of(dtype),
                        ScalarOp::ReduceMax => NumericScalar::neg_infinity_of(dtype),
                        _ => unreachable!(),
                    };

                    for k in 0..bound as u32 {
                        let src = group.inputs[0].resolve(i, k);
                        let val = &values[src.0 as usize];
                        // Cast input to this group's dtype before accumulating.
                        let val = val.cast_to(dtype);
                        acc = match group.op {
                            ScalarOp::ReduceSum => acc.add(&val),
                            ScalarOp::ReduceMax => acc.scalar_max(&val),
                            _ => unreachable!(),
                        };
                    }
                    if debug_nan && !nan_reported && !acc.to_f64().is_finite() {
                        eprintln!(
                            "[NaN-debug] group {} (reduce {:?}, {:?}): atom {} = {:?}",
                            group_idx, group.op, dtype, atom_idx, acc
                        );
                        nan_reported = true;
                    }
                    values[atom_idx as usize] = acc;
                } else {
                    let resolve_input = |inp: &super::pattern::InputRef| -> NumericScalar {
                        let src = inp.resolve(i, 0);
                        values[src.0 as usize].cast_to(dtype)
                    };

                    let val = match &group.op {
                        ScalarOp::Literal(lit) => {
                            if let Some(ov) = overrides.get(&atom_idx) {
                                ov.cast_to(dtype)
                            } else {
                                // Interpret the literal bits in this group's dtype.
                                NumericScalar::F64(lit.as_f64()).cast_to(dtype)
                            }
                        }
                        ScalarOp::Identity => resolve_input(&group.inputs[0]),
                        ScalarOp::Binary(bin) => {
                            let a = resolve_input(&group.inputs[0]);
                            let b = resolve_input(&group.inputs[1]);
                            match bin {
                                ScalarBinOp::Add => a.add(&b),
                                ScalarBinOp::Sub => a.sub(&b),
                                ScalarBinOp::Mul => a.mul(&b),
                                ScalarBinOp::Div => a.div(&b),
                                ScalarBinOp::Max => a.scalar_max(&b),
                                ScalarBinOp::Min => a.scalar_min(&b),
                                ScalarBinOp::Mod => a.modulo(&b),
                                ScalarBinOp::Pow => a.pow(&b),
                            }
                        }
                        ScalarOp::Unary(un) => {
                            let x = resolve_input(&group.inputs[0]);
                            match un {
                                ScalarUnaryOp::Neg => x.neg(),
                                ScalarUnaryOp::Abs => x.abs(),
                                ScalarUnaryOp::Exp => x.exp(),
                                ScalarUnaryOp::Ln => x.ln(),
                                ScalarUnaryOp::Sqrt => x.sqrt(),
                                ScalarUnaryOp::Reciprocal => x.recip(),
                                ScalarUnaryOp::Tanh => x.tanh(),
                                ScalarUnaryOp::Floor => x.floor(),
                                ScalarUnaryOp::Ceil => x.ceil(),
                            }
                        }
                        ScalarOp::Select => {
                            let cond = resolve_input(&group.inputs[0]);
                            if cond.is_nonzero() {
                                resolve_input(&group.inputs[1])
                            } else {
                                resolve_input(&group.inputs[2])
                            }
                        }
                        ScalarOp::ReduceSum | ScalarOp::ReduceMax => unreachable!(),
                    };
                    if debug_nan && !nan_reported && !val.to_f64().is_finite() {
                        let input_detail: Vec<String> = group
                            .inputs
                            .iter()
                            .enumerate()
                            .map(|(j, inp)| {
                                let src = inp.resolve(i, 0);
                                format!(
                                    "inp[{}]=atom{}={:?}",
                                    j,
                                    src.0,
                                    values[src.0 as usize]
                                )
                            })
                            .collect();
                        eprintln!(
                            "[NaN-debug] group {} ({:?}, {:?}): atom {} = {:?} inputs: {}",
                            group_idx,
                            group.op,
                            dtype,
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
