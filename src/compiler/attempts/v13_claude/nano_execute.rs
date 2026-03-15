#![allow(clippy::all, dead_code, unreachable_patterns, unused)]
//! Naive NanoGraph executor using NumericScalar for dtype-correct evaluation.
//!
//! This is the reference implementation for correctness testing.
//! Use the v2c lane planner + nano_codegen_v2 for actual execution.

use std::collections::HashMap;

use crate::nano_graph::{AtomGroup, AtomId, InputRef, NanoGraph, ScalarBinOp, ScalarOp, ScalarUnaryOp};
use crate::numeric_scalar::NumericScalar;

/// Execute all groups sequentially in index order (topological order).
/// No partitioning. This is the correctness reference.
pub fn execute_nanograph_naive(
    graph: &NanoGraph,
    inputs: &HashMap<u64, NumericScalar>,
) -> HashMap<u64, NumericScalar> {
    let num_atoms = graph.num_atoms() as usize;
    let mut values: Vec<NumericScalar> = vec![NumericScalar::F32(0.0); num_atoms];

    // Pre-fill Literal atoms from ScalarOp values.
    for group in graph.groups() {
        if let ScalarOp::Literal(scalar) = &group.op {
            for i in 0..group.count {
                values[(group.base_id.0 + i) as usize] = scalar.clone();
            }
        }
    }

    // Apply overrides (weights, user inputs).
    for (&idx, val) in inputs {
        values[idx as usize] = val.clone();
    }

    // Execute each group in topological order.
    for group in graph.groups() {
        eval_group(graph, group, inputs, &mut values);
    }

    let mut result = HashMap::new();
    for i in 0..num_atoms {
        result.insert(i as u64, values[i].clone());
    }
    result
}

fn eval_group(
    _graph: &NanoGraph,
    group: &AtomGroup,
    inputs: &HashMap<u64, NumericScalar>,
    values: &mut [NumericScalar],
) {
    let is_reduce = group.op.is_reduce();

    if is_reduce {
        let (reduce_count, reduce_stride, compute_dtype, output_dtype) = match &group.op {
            ScalarOp::ReduceSum { reduce_count, reduce_stride, compute_dtype, output_dtype } =>
                (*reduce_count, *reduce_stride, *compute_dtype, *output_dtype),
            ScalarOp::ReduceMax { reduce_count, reduce_stride, compute_dtype, output_dtype } =>
                (*reduce_count, *reduce_stride, *compute_dtype, *output_dtype),
            _ => unreachable!(),
        };
        let is_sum = matches!(&group.op, ScalarOp::ReduceSum { .. });

        for i in 0..group.count {
            let atom_idx = (group.base_id.0 + i) as usize;
            let base = group.inputs[0].resolve(i, 0);
            let mut acc = if is_sum {
                NumericScalar::zero_of(compute_dtype)
            } else {
                NumericScalar::neg_infinity_of(compute_dtype)
            };
            for k in 0..reduce_count {
                let src = (base.0 as i64 + k as i64 * reduce_stride) as usize;
                let val = values[src].cast_to(compute_dtype);
                acc = if is_sum { acc.add(&val) } else { acc.scalar_max(&val) };
            }
            values[atom_idx] = acc.cast_to(output_dtype);
        }
    } else {
        for i in 0..group.count {
            let atom_idx = (group.base_id.0 + i) as usize;
            let val = match &group.op {
                ScalarOp::Literal(scalar) => {
                    if let Some(ov) = inputs.get(&(atom_idx as u64)) {
                        ov.clone()
                    } else {
                        scalar.clone()
                    }
                }
                ScalarOp::Identity { compute_dtype, output_dtype } => {
                    let src = group.inputs[0].resolve(i, 0);
                    values[src.0 as usize].cast_to(*compute_dtype).cast_to(*output_dtype)
                }
                ScalarOp::Binary { op, compute_dtype, output_dtype } => {
                    let a = values[group.inputs[0].resolve(i, 0).0 as usize].cast_to(*compute_dtype);
                    let b = values[group.inputs[1].resolve(i, 0).0 as usize].cast_to(*compute_dtype);
                    let result = match op {
                        ScalarBinOp::Add => a.add(&b),
                        ScalarBinOp::Sub => a.sub(&b),
                        ScalarBinOp::Mul => a.mul(&b),
                        ScalarBinOp::Div => a.div(&b),
                        ScalarBinOp::Max => a.scalar_max(&b),
                        ScalarBinOp::Min => a.scalar_min(&b),
                        ScalarBinOp::Mod => a.modulo(&b),
                        ScalarBinOp::Pow => a.pow(&b),
                        ScalarBinOp::Equal => if a.to_f64() == b.to_f64() { NumericScalar::F32(1.0) } else { NumericScalar::F32(0.0) },
                        ScalarBinOp::Greater => if a.to_f64() > b.to_f64() { NumericScalar::F32(1.0) } else { NumericScalar::F32(0.0) },
                        ScalarBinOp::GreaterOrEqual => if a.to_f64() >= b.to_f64() { NumericScalar::F32(1.0) } else { NumericScalar::F32(0.0) },
                        ScalarBinOp::Less => if a.to_f64() < b.to_f64() { NumericScalar::F32(1.0) } else { NumericScalar::F32(0.0) },
                        ScalarBinOp::LessOrEqual => if a.to_f64() <= b.to_f64() { NumericScalar::F32(1.0) } else { NumericScalar::F32(0.0) },
                        ScalarBinOp::And => if a.to_f64() != 0.0 && b.to_f64() != 0.0 { NumericScalar::F32(1.0) } else { NumericScalar::F32(0.0) },
                        ScalarBinOp::Or => if a.to_f64() != 0.0 || b.to_f64() != 0.0 { NumericScalar::F32(1.0) } else { NumericScalar::F32(0.0) },
                        ScalarBinOp::Xor => if (a.to_f64() != 0.0) ^ (b.to_f64() != 0.0) { NumericScalar::F32(1.0) } else { NumericScalar::F32(0.0) },
                    };
                    result.cast_to(*output_dtype)
                }
                ScalarOp::Unary { op, compute_dtype, output_dtype } => {
                    let x = values[group.inputs[0].resolve(i, 0).0 as usize].cast_to(*compute_dtype);
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
                ScalarOp::Select { compute_dtype, output_dtype } => {
                    let cond = values[group.inputs[0].resolve(i, 0).0 as usize].cast_to(*compute_dtype);
                    let result = if cond.is_nonzero() {
                        values[group.inputs[1].resolve(i, 0).0 as usize].cast_to(*compute_dtype)
                    } else {
                        values[group.inputs[2].resolve(i, 0).0 as usize].cast_to(*compute_dtype)
                    };
                    result.cast_to(*output_dtype)
                }
                ScalarOp::IndirectLoad { table_base, output_dtype } => {
                    let idx = values[group.inputs[0].resolve(i, 0).0 as usize].to_f64() as usize;
                    values[table_base.0 as usize + idx].cast_to(*output_dtype)
                }
                ScalarOp::ReduceSum { .. } | ScalarOp::ReduceMax { .. } => unreachable!(),
            };
            values[atom_idx] = val;
        }
    }
}
