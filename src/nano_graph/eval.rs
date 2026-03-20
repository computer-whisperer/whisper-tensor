//! Scalar evaluator for NanoGraph integrity checks.
//!
//! Evaluates groups in topological order using per-group buffers with
//! refcount-based freeing. Peak memory is bounded by the max live set
//! rather than total atoms.
//!
//! Precision semantics: ops with a `compute_dtype` cast inputs to that
//! precision, execute the op, then cast the result to the group's
//! `output_dtype`. Ops without `compute_dtype` (Identity, Select,
//! IndirectLoad) use the group's `output_dtype` directly.

use crate::DynRank;
use crate::backends::ndarray_backend::numeric_tensor::NDArrayNumericTensor;
use crate::numeric_scalar::NumericScalar;

use super::ops::{ReduceKind, ScalarBinOp, ScalarOp, ScalarUnaryOp};
use super::pattern::{AtomId, NanoGraph};

/// Evaluate a NanoGraph, returning one tensor per requested output range.
///
/// Each `(AtomId, tensor)` input pair fills a contiguous atom range starting
/// at the given AtomId. These typically correspond to `input_tensors` entries.
///
/// Peak memory is bounded by the max live set (groups whose values are
/// needed by some future group) rather than total atoms.
pub fn eval(
    graph: &NanoGraph,
    inputs: &[(AtomId, &NDArrayNumericTensor<DynRank>)],
    output_ranges: &[super::pattern::AtomRange],
) -> Vec<NDArrayNumericTensor<DynRank>> {
    use std::collections::HashSet;

    let groups = graph.groups();
    let n = groups.len();
    let input_tensors = graph.input_tensors();

    // --- Step 1: Populate input tensor buffers ---
    let mut input_buffers: Vec<Vec<NumericScalar>> = input_tensors
        .iter()
        .map(|it| vec![NumericScalar::F32(0.0); it.count as usize])
        .collect();

    for &(base, tensor) in inputs {
        if let Some((ti, offset)) = graph.find_input_idx(base) {
            let buf = &mut input_buffers[ti];
            populate_from_tensor(buf, offset as usize, tensor);
        }
    }

    // --- Step 2: Compute producer sets and use counts ---
    let mut producers: Vec<Vec<usize>> = Vec::with_capacity(n);
    let mut remaining: Vec<u32> = vec![0; n];

    for (gi, group) in groups.iter().enumerate() {
        let mut seen = HashSet::<usize>::new();
        graph.collect_all_producer_indices(group, gi, &mut seen);

        let deps: Vec<usize> = seen.into_iter().collect();
        for &pi in &deps {
            remaining[pi] += 1;
        }
        producers.push(deps);
    }

    // --- Step 3: Mark output groups with extra use count ---
    for range in output_ranges {
        let base = range.base.0;
        let end = base + range.count;
        let mut id = base;
        while id < end {
            if let Some(gi) = graph.find_group_idx(AtomId(id)) {
                remaining[gi] += 1;
                let g = &groups[gi];
                id = g.base_id.0 + g.count;
            } else {
                id += 1;
            }
        }
    }

    // --- Step 4: Evaluate groups in order ---
    let mut group_buffers: Vec<Option<Vec<NumericScalar>>> = vec![None; n];

    for (gi, group) in groups.iter().enumerate() {
        // Skip dead groups (unless they're literals that might be needed).
        if remaining[gi] == 0 && !matches!(&group.op, ScalarOp::Literal(_)) {
            continue;
        }

        let count = group.count as usize;
        let mut buf = vec![NumericScalar::F32(0.0); count];
        let output_dtype = group.output_dtype;

        for i in 0..group.count {
            let ri = i + group.atom_offset;

            if let ScalarOp::Reduce {
                kind,
                reduce_count,
                reduce_stride,
                compute_dtype,
            } = &group.op
            {
                let mut acc = match kind {
                    ReduceKind::Sum => NumericScalar::zero_of(*compute_dtype),
                    ReduceKind::Max => NumericScalar::neg_infinity_of(*compute_dtype),
                };

                let base_atom = group.inputs[0].resolve(ri);
                for k in 0..*reduce_count {
                    let src_id = AtomId((base_atom.0 as i64 + k as i64 * reduce_stride) as u64);
                    let val = lookup_atom(src_id, graph, &group_buffers, &input_buffers)
                        .cast_to(*compute_dtype);
                    acc = match kind {
                        ReduceKind::Sum => acc.add(&val),
                        ReduceKind::Max => acc.scalar_max(&val),
                    };
                }
                buf[i as usize] = acc.cast_to(output_dtype);
            } else {
                let val = match &group.op {
                    ScalarOp::Literal(scalar) => scalar.clone(),
                    ScalarOp::Identity => {
                        let src = group.inputs[0].resolve(ri);
                        lookup_atom(src, graph, &group_buffers, &input_buffers)
                            .cast_to(output_dtype)
                    }
                    ScalarOp::Binary { op, compute_dtype } => {
                        let a = lookup_atom(
                            group.inputs[0].resolve(ri),
                            graph,
                            &group_buffers,
                            &input_buffers,
                        )
                        .cast_to(*compute_dtype);
                        let b = lookup_atom(
                            group.inputs[1].resolve(ri),
                            graph,
                            &group_buffers,
                            &input_buffers,
                        )
                        .cast_to(*compute_dtype);
                        eval_binop(op, &a, &b).cast_to(output_dtype)
                    }
                    ScalarOp::Unary { op, compute_dtype } => {
                        let x = lookup_atom(
                            group.inputs[0].resolve(ri),
                            graph,
                            &group_buffers,
                            &input_buffers,
                        )
                        .cast_to(*compute_dtype);
                        eval_unaryop(op, &x).cast_to(output_dtype)
                    }
                    ScalarOp::Select => {
                        let cond = lookup_atom(
                            group.inputs[0].resolve(ri),
                            graph,
                            &group_buffers,
                            &input_buffers,
                        );
                        if cond.is_nonzero() {
                            lookup_atom(
                                group.inputs[1].resolve(ri),
                                graph,
                                &group_buffers,
                                &input_buffers,
                            )
                            .cast_to(output_dtype)
                        } else {
                            lookup_atom(
                                group.inputs[2].resolve(ri),
                                graph,
                                &group_buffers,
                                &input_buffers,
                            )
                            .cast_to(output_dtype)
                        }
                    }
                    ScalarOp::IndirectLoad { table_base } => {
                        let src = group.inputs[0].resolve(ri);
                        let index =
                            lookup_atom(src, graph, &group_buffers, &input_buffers).to_f64() as u64;
                        let table_atom = AtomId(table_base.0 + index);
                        lookup_atom(table_atom, graph, &group_buffers, &input_buffers)
                            .cast_to(output_dtype)
                    }
                    ScalarOp::Reduce { .. } => unreachable!(),
                };
                buf[i as usize] = val;
            }
        }

        group_buffers[gi] = Some(buf);

        // Decrement producer refcounts and free spent buffers.
        for &pi in &producers[gi] {
            remaining[pi] -= 1;
            if remaining[pi] == 0 {
                group_buffers[pi] = None;
            }
        }
    }

    // --- Step 5: Assemble output tensors ---
    output_ranges
        .iter()
        .map(|range| {
            let mut scalars = Vec::with_capacity(range.count as usize);
            for offset in 0..range.count {
                let atom_id = AtomId(range.base.0 + offset);
                scalars.push(lookup_atom(atom_id, graph, &group_buffers, &input_buffers));
            }
            scalars_to_tensor(&scalars, range.dtype)
        })
        .collect()
}

/// Look up an atom's value from per-group or per-input-tensor buffers.
fn lookup_atom(
    atom_id: AtomId,
    graph: &NanoGraph,
    group_buffers: &[Option<Vec<NumericScalar>>],
    input_buffers: &[Vec<NumericScalar>],
) -> NumericScalar {
    if let Some(gi) = graph.find_group_idx(atom_id) {
        let group = &graph.groups()[gi];
        let offset = (atom_id.0 - group.base_id.0) as usize;
        return group_buffers[gi].as_ref().unwrap_or_else(|| {
            panic!(
                "group {} (base={}) buffer already freed when reading atom {}",
                gi, group.base_id, atom_id
            )
        })[offset]
            .clone();
    }
    if let Some((ti, offset)) = graph.find_input_idx(atom_id) {
        return input_buffers[ti][offset as usize].clone();
    }
    panic!("atom {} not found in any group or input tensor", atom_id);
}

/// Evaluate a binary op.
fn eval_binop(op: &ScalarBinOp, a: &NumericScalar, b: &NumericScalar) -> NumericScalar {
    match op {
        ScalarBinOp::Add => a.add(b),
        ScalarBinOp::Sub => a.sub(b),
        ScalarBinOp::Mul => a.mul(b),
        ScalarBinOp::Div => a.div(b),
        ScalarBinOp::Max => a.scalar_max(b),
        ScalarBinOp::Min => a.scalar_min(b),
        ScalarBinOp::Mod => a.modulo(b),
        ScalarBinOp::Pow => a.pow(b),
        ScalarBinOp::Equal => NumericScalar::F32(if a.to_f64() == b.to_f64() { 1.0 } else { 0.0 }),
        ScalarBinOp::Greater => NumericScalar::F32(if a.to_f64() > b.to_f64() { 1.0 } else { 0.0 }),
        ScalarBinOp::GreaterOrEqual => {
            NumericScalar::F32(if a.to_f64() >= b.to_f64() { 1.0 } else { 0.0 })
        }
        ScalarBinOp::Less => NumericScalar::F32(if a.to_f64() < b.to_f64() { 1.0 } else { 0.0 }),
        ScalarBinOp::LessOrEqual => {
            NumericScalar::F32(if a.to_f64() <= b.to_f64() { 1.0 } else { 0.0 })
        }
        ScalarBinOp::And => NumericScalar::F32(if a.to_f64() != 0.0 && b.to_f64() != 0.0 {
            1.0
        } else {
            0.0
        }),
        ScalarBinOp::Or => NumericScalar::F32(if a.to_f64() != 0.0 || b.to_f64() != 0.0 {
            1.0
        } else {
            0.0
        }),
        ScalarBinOp::Xor => NumericScalar::F32(if (a.to_f64() != 0.0) ^ (b.to_f64() != 0.0) {
            1.0
        } else {
            0.0
        }),
    }
}

/// Evaluate a unary op.
fn eval_unaryop(op: &ScalarUnaryOp, x: &NumericScalar) -> NumericScalar {
    match op {
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

/// Populate a buffer from an NDArrayNumericTensor.
fn populate_from_tensor(
    buf: &mut [NumericScalar],
    offset: usize,
    tensor: &NDArrayNumericTensor<DynRank>,
) {
    let limit = buf.len();
    match tensor {
        NDArrayNumericTensor::F32(a) => {
            for (i, &v) in a.iter().enumerate() {
                if offset + i >= limit {
                    break;
                }
                buf[offset + i] = NumericScalar::F32(v);
            }
        }
        NDArrayNumericTensor::F64(a) => {
            for (i, &v) in a.iter().enumerate() {
                if offset + i >= limit {
                    break;
                }
                buf[offset + i] = NumericScalar::F64(v);
            }
        }
        NDArrayNumericTensor::BF16(a) => {
            for (i, &v) in a.iter().enumerate() {
                if offset + i >= limit {
                    break;
                }
                buf[offset + i] = NumericScalar::BF16(v);
            }
        }
        NDArrayNumericTensor::F16(a) => {
            for (i, &v) in a.iter().enumerate() {
                if offset + i >= limit {
                    break;
                }
                buf[offset + i] = NumericScalar::F16(v);
            }
        }
        NDArrayNumericTensor::I64(a) => {
            for (i, &v) in a.iter().enumerate() {
                if offset + i >= limit {
                    break;
                }
                buf[offset + i] = NumericScalar::I64(v);
            }
        }
        NDArrayNumericTensor::U64(a) => {
            for (i, &v) in a.iter().enumerate() {
                if offset + i >= limit {
                    break;
                }
                buf[offset + i] = NumericScalar::U64(v);
            }
        }
        NDArrayNumericTensor::I32(a) => {
            for (i, &v) in a.iter().enumerate() {
                if offset + i >= limit {
                    break;
                }
                buf[offset + i] = NumericScalar::I32(v);
            }
        }
        NDArrayNumericTensor::U32(a) => {
            for (i, &v) in a.iter().enumerate() {
                if offset + i >= limit {
                    break;
                }
                buf[offset + i] = NumericScalar::U32(v);
            }
        }
        NDArrayNumericTensor::I16(a) => {
            for (i, &v) in a.iter().enumerate() {
                if offset + i >= limit {
                    break;
                }
                buf[offset + i] = NumericScalar::I16(v);
            }
        }
        NDArrayNumericTensor::U16(a) => {
            for (i, &v) in a.iter().enumerate() {
                if offset + i >= limit {
                    break;
                }
                buf[offset + i] = NumericScalar::U16(v);
            }
        }
        NDArrayNumericTensor::I8(a) => {
            for (i, &v) in a.iter().enumerate() {
                if offset + i >= limit {
                    break;
                }
                buf[offset + i] = NumericScalar::I8(v);
            }
        }
        NDArrayNumericTensor::U8(a) => {
            for (i, &v) in a.iter().enumerate() {
                if offset + i >= limit {
                    break;
                }
                buf[offset + i] = NumericScalar::U8(v);
            }
        }
        NDArrayNumericTensor::F8E4M3FN(a) => {
            for (i, &v) in a.iter().enumerate() {
                if offset + i >= limit {
                    break;
                }
                buf[offset + i] = NumericScalar::F8E4M3FN(v);
            }
        }
        NDArrayNumericTensor::F8E5M2(a) => {
            for (i, &v) in a.iter().enumerate() {
                if offset + i >= limit {
                    break;
                }
                buf[offset + i] = NumericScalar::F8E5M2(v);
            }
        }
        NDArrayNumericTensor::BOOL(a) => {
            for (i, &v) in a.iter().enumerate() {
                if offset + i >= limit {
                    break;
                }
                buf[offset + i] = NumericScalar::BOOL(v);
            }
        }
        NDArrayNumericTensor::STRING(_) => panic!("Cannot populate values from string tensor"),
    }
}

fn scalars_to_tensor(
    scalars: &[NumericScalar],
    dtype: crate::dtype::DType,
) -> NDArrayNumericTensor<DynRank> {
    use ndarray::{ArcArray, IxDyn};
    let shape = IxDyn(&[scalars.len()]);
    match dtype {
        crate::dtype::DType::F32 => {
            let data: Vec<f32> = scalars.iter().map(|s| s.to_f64() as f32).collect();
            NDArrayNumericTensor::F32(ArcArray::from_shape_vec(shape, data).unwrap())
        }
        crate::dtype::DType::F64 => {
            let data: Vec<f64> = scalars.iter().map(|s| s.to_f64()).collect();
            NDArrayNumericTensor::F64(ArcArray::from_shape_vec(shape, data).unwrap())
        }
        crate::dtype::DType::BF16 => {
            let data: Vec<half::bf16> = scalars
                .iter()
                .map(|s| half::bf16::from_f64(s.to_f64()))
                .collect();
            NDArrayNumericTensor::BF16(ArcArray::from_shape_vec(shape, data).unwrap())
        }
        crate::dtype::DType::F16 => {
            let data: Vec<half::f16> = scalars
                .iter()
                .map(|s| half::f16::from_f64(s.to_f64()))
                .collect();
            NDArrayNumericTensor::F16(ArcArray::from_shape_vec(shape, data).unwrap())
        }
        crate::dtype::DType::I64 => {
            let data: Vec<i64> = scalars.iter().map(|s| s.to_f64() as i64).collect();
            NDArrayNumericTensor::I64(ArcArray::from_shape_vec(shape, data).unwrap())
        }
        crate::dtype::DType::I32 => {
            let data: Vec<i32> = scalars.iter().map(|s| s.to_f64() as i32).collect();
            NDArrayNumericTensor::I32(ArcArray::from_shape_vec(shape, data).unwrap())
        }
        crate::dtype::DType::U64 => {
            let data: Vec<u64> = scalars.iter().map(|s| s.to_f64() as u64).collect();
            NDArrayNumericTensor::U64(ArcArray::from_shape_vec(shape, data).unwrap())
        }
        crate::dtype::DType::U32 => {
            let data: Vec<u32> = scalars.iter().map(|s| s.to_f64() as u32).collect();
            NDArrayNumericTensor::U32(ArcArray::from_shape_vec(shape, data).unwrap())
        }
        crate::dtype::DType::I8 => {
            let data: Vec<i8> = scalars.iter().map(|s| s.to_f64() as i8).collect();
            NDArrayNumericTensor::I8(ArcArray::from_shape_vec(shape, data).unwrap())
        }
        crate::dtype::DType::U8 => {
            let data: Vec<u8> = scalars.iter().map(|s| s.to_f64() as u8).collect();
            NDArrayNumericTensor::U8(ArcArray::from_shape_vec(shape, data).unwrap())
        }
        crate::dtype::DType::I16 => {
            let data: Vec<i16> = scalars.iter().map(|s| s.to_f64() as i16).collect();
            NDArrayNumericTensor::I16(ArcArray::from_shape_vec(shape, data).unwrap())
        }
        crate::dtype::DType::U16 => {
            let data: Vec<u16> = scalars.iter().map(|s| s.to_f64() as u16).collect();
            NDArrayNumericTensor::U16(ArcArray::from_shape_vec(shape, data).unwrap())
        }
        crate::dtype::DType::BOOL => {
            let data: Vec<bool> = scalars.iter().map(|s| s.to_f64() != 0.0).collect();
            NDArrayNumericTensor::BOOL(ArcArray::from_shape_vec(shape, data).unwrap())
        }
        other => panic!("scalars_to_tensor: unsupported dtype {:?}", other),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::graph::GlobalId;
    use crate::nano_graph::ops::{ReduceKind, ScalarBinOp, ScalarOp, ScalarUnaryOp};
    use crate::nano_graph::pattern::{AtomRange, InputRef, NanoGraph};
    use crate::numeric_scalar::NumericScalar;
    use ndarray::{ArcArray, IxDyn};

    fn make_f32_tensor(data: &[f32]) -> NDArrayNumericTensor<DynRank> {
        NDArrayNumericTensor::F32(
            ArcArray::from_shape_vec(IxDyn(&[data.len()]), data.to_vec()).unwrap(),
        )
    }

    fn eval_f32(
        graph: &NanoGraph,
        inputs: &[(AtomId, &NDArrayNumericTensor<DynRank>)],
        output_range: AtomRange,
    ) -> Vec<f64> {
        let result = eval(graph, inputs, &[output_range]);
        match &result[0] {
            NDArrayNumericTensor::F32(a) => a.iter().map(|&v| v as f64).collect(),
            _ => panic!("expected F32"),
        }
    }

    #[test]
    fn test_elementwise_mul_add() {
        let mut g = NanoGraph::new();
        let gid = GlobalId(1);

        let inp = g.add_input_tensor(gid, 4, DType::F32);

        let two = g.push_atom(
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(2.0)),
            vec![],
            vec![],
        );

        let mul = g.push_group(
            4,
            DType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![
                InputRef::Affine {
                    base: inp,
                    stride: 1,
                },
                InputRef::Broadcast(two),
            ],
        );

        let bias = g.push_group(
            4,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(10.0)),
            vec![],
            vec![],
        );

        let out = g.push_group(
            4,
            DType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![
                InputRef::Affine {
                    base: mul,
                    stride: 1,
                },
                InputRef::Affine {
                    base: bias,
                    stride: 1,
                },
            ],
        );

        let input_data = make_f32_tensor(&[1.0, 2.0, 3.0, 4.0]);
        let inputs = vec![(inp, &input_data)];
        let output_range = AtomRange {
            base: out,
            count: 4,
            dtype: DType::F32,
        };

        let vals = eval_f32(&g, &inputs, output_range);
        assert_eq!(vals, vec![12.0, 14.0, 16.0, 18.0]);
    }

    #[test]
    fn test_reduce_sum() {
        let mut g = NanoGraph::new();
        let gid = GlobalId(1);

        let inp = g.add_input_tensor(gid, 8, DType::F32);

        let red = g.push_group(
            2,
            DType::F32,
            ScalarOp::Reduce {
                kind: ReduceKind::Sum,
                reduce_count: 4,
                reduce_stride: 1,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine {
                base: inp,
                stride: 4,
            }],
        );

        let input_data = make_f32_tensor(&[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]);
        let inputs = vec![(inp, &input_data)];
        let output_range = AtomRange {
            base: red,
            count: 2,
            dtype: DType::F32,
        };

        let vals = eval_f32(&g, &inputs, output_range);
        assert_eq!(vals, vec![10.0, 100.0]);
    }

    #[test]
    fn test_frees_intermediates() {
        let mut g = NanoGraph::new();
        let gid = GlobalId(1);

        let inp = g.add_input_tensor(gid, 4, DType::F32);

        let neg = g.push_group(
            4,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine {
                base: inp,
                stride: 1,
            }],
        );

        let exp = g.push_group(
            4,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Exp,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine {
                base: neg,
                stride: 1,
            }],
        );

        let _dead = g.push_group(
            4,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(999.0)),
            vec![],
            vec![],
        );

        let input_data = make_f32_tensor(&[0.0, 1.0, 2.0, 3.0]);
        let inputs = vec![(inp, &input_data)];
        let output_range = AtomRange {
            base: exp,
            count: 4,
            dtype: DType::F32,
        };

        let vals = eval_f32(&g, &inputs, output_range);
        // exp(-x) for x = 0, 1, 2, 3
        for (i, &v) in vals.iter().enumerate() {
            let expected = (-(i as f64)).exp();
            assert!(
                (v - expected).abs() < 1e-6,
                "element {}: got {} expected {}",
                i,
                v,
                expected
            );
        }
    }

    #[test]
    fn test_indirect_load() {
        let mut g = NanoGraph::new();
        let table_gid = GlobalId(1);
        let idx_gid = GlobalId(2);

        let table = g.add_input_tensor(table_gid, 4, DType::F32);
        let indices = g.add_input_tensor(idx_gid, 2, DType::F32);

        let gathered = g.push_group(
            2,
            DType::F32,
            ScalarOp::IndirectLoad { table_base: table },
            vec![],
            vec![InputRef::Affine {
                base: indices,
                stride: 1,
            }],
        );

        let table_data = make_f32_tensor(&[10.0, 20.0, 30.0, 40.0]);
        let idx_data = make_f32_tensor(&[2.0, 0.0]);
        let inputs = vec![(table, &table_data), (indices, &idx_data)];
        let output_range = AtomRange {
            base: gathered,
            count: 2,
            dtype: DType::F32,
        };

        let vals = eval_f32(&g, &inputs, output_range);
        assert_eq!(vals, vec![30.0, 10.0]);
    }
}
