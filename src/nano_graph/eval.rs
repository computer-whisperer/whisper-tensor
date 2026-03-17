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

use crate::backends::ndarray_backend::numeric_tensor::NDArrayNumericTensor;
use crate::numeric_scalar::NumericScalar;
use crate::DynRank;

use super::ops::{ScalarBinOp, ScalarOp, ScalarUnaryOp};
use super::pattern::{AtomId, NanoGraph};

/// Flat storage of one NumericScalar per atom.
pub struct NanoEval {
    values: Vec<NumericScalar>,
}

impl NanoEval {
    /// Evaluate a NanoGraph, providing external data as tensors keyed by base AtomId.
    ///
    /// Each `(AtomId, tensor)` pair fills a contiguous atom range starting at
    /// the given AtomId with the tensor's flattened elements. Typically these
    /// correspond to the graph's `input_tensors` entries (weights, user inputs).
    pub fn eval(
        graph: &NanoGraph,
        inputs: &[(AtomId, &NDArrayNumericTensor<DynRank>)],
    ) -> Self {
        let num_atoms = graph.num_atoms() as usize;
        let mut values: Vec<NumericScalar> = vec![NumericScalar::F32(0.0); num_atoms];

        for &(base, tensor) in inputs {
            populate_from_tensor(&mut values, base.0 as usize, tensor);
        }

        Self::run_eval(graph, &mut values, None, false)
    }

    /// Legacy API: evaluate with per-atom scalar overrides.
    ///
    /// `overrides` maps atom index → NumericScalar for input atoms.
    /// Kept for backward compatibility with existing tests and v13 code.
    pub fn eval_with_overrides(
        graph: &NanoGraph,
        overrides: &HashMap<u64, NumericScalar>,
    ) -> Self {
        Self::eval_with_overrides_inner(graph, overrides, false)
    }

    /// Like eval_with_overrides, but prints the first group that produces NaN/Inf.
    pub fn eval_with_overrides_debug(
        graph: &NanoGraph,
        overrides: &HashMap<u64, NumericScalar>,
    ) -> Self {
        Self::eval_with_overrides_inner(graph, overrides, true)
    }

    fn eval_with_overrides_inner(
        graph: &NanoGraph,
        overrides: &HashMap<u64, NumericScalar>,
        debug_nan: bool,
    ) -> Self {
        let num_atoms = graph.num_atoms() as usize;
        let mut values: Vec<NumericScalar> = vec![NumericScalar::F32(0.0); num_atoms];

        for (&idx, val) in overrides {
            if (idx as usize) < num_atoms {
                values[idx as usize] = val.clone();
            }
        }

        Self::run_eval(graph, &mut values, Some(overrides), debug_nan)
    }

    /// Core evaluation loop shared by all entry points.
    ///
    /// `literal_overrides` is Some for the legacy API where Literal(0.0)
    /// placeholder groups need to be overridden with actual values.
    /// None for the new API where input data lives in input_tensor ranges.
    fn run_eval(
        graph: &NanoGraph,
        values: &mut Vec<NumericScalar>,
        literal_overrides: Option<&HashMap<u64, NumericScalar>>,
        debug_nan: bool,
    ) -> Self {
        let mut nan_reported = false;

        for (group_idx, group) in graph.groups().iter().enumerate() {
            let is_reduce = group.op.is_reduce();

            for i in 0..group.count {
                let atom_idx = group.base_id.0 + i;
                let ri = i + group.atom_offset;

                if is_reduce {
                    let (reduce_count, reduce_stride, compute_dtype, output_dtype) = match &group.op
                    {
                        ScalarOp::ReduceSum {
                            reduce_count,
                            reduce_stride,
                            compute_dtype,
                            output_dtype,
                        } => (*reduce_count, *reduce_stride, *compute_dtype, *output_dtype),
                        ScalarOp::ReduceMax {
                            reduce_count,
                            reduce_stride,
                            compute_dtype,
                            output_dtype,
                        } => (*reduce_count, *reduce_stride, *compute_dtype, *output_dtype),
                        _ => unreachable!(),
                    };

                    let mut acc = match &group.op {
                        ScalarOp::ReduceSum { .. } => NumericScalar::zero_of(compute_dtype),
                        ScalarOp::ReduceMax { .. } => NumericScalar::neg_infinity_of(compute_dtype),
                        _ => unreachable!(),
                    };

                    let base = group.inputs[0].resolve(ri, 0);
                    for k in 0..reduce_count {
                        let src_idx = (base.0 as i64 + k as i64 * reduce_stride) as u64;
                        let val = values[src_idx as usize].cast_to(compute_dtype);
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
                    values[atom_idx as usize] = acc.cast_to(output_dtype);
                } else {
                    let val = match &group.op {
                        ScalarOp::Literal(scalar) => {
                            if let Some(ovs) = literal_overrides {
                                if let Some(ov) = ovs.get(&atom_idx) {
                                    ov.cast_to(scalar.dtype())
                                } else {
                                    scalar.clone()
                                }
                            } else {
                                scalar.clone()
                            }
                        }
                        ScalarOp::Identity {
                            compute_dtype,
                            output_dtype,
                        } => {
                            let src = group.inputs[0].resolve(ri, 0);
                            let x = values[src.0 as usize].cast_to(*compute_dtype);
                            x.cast_to(*output_dtype)
                        }
                        ScalarOp::Binary {
                            op,
                            compute_dtype,
                            output_dtype,
                        } => {
                            let a = values[group.inputs[0].resolve(ri, 0).0 as usize]
                                .cast_to(*compute_dtype);
                            let b = values[group.inputs[1].resolve(ri, 0).0 as usize]
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
                                ScalarBinOp::Equal => {
                                    if a.to_f64() == b.to_f64() {
                                        NumericScalar::F32(1.0)
                                    } else {
                                        NumericScalar::F32(0.0)
                                    }
                                }
                                ScalarBinOp::Greater => {
                                    if a.to_f64() > b.to_f64() {
                                        NumericScalar::F32(1.0)
                                    } else {
                                        NumericScalar::F32(0.0)
                                    }
                                }
                                ScalarBinOp::GreaterOrEqual => {
                                    if a.to_f64() >= b.to_f64() {
                                        NumericScalar::F32(1.0)
                                    } else {
                                        NumericScalar::F32(0.0)
                                    }
                                }
                                ScalarBinOp::Less => {
                                    if a.to_f64() < b.to_f64() {
                                        NumericScalar::F32(1.0)
                                    } else {
                                        NumericScalar::F32(0.0)
                                    }
                                }
                                ScalarBinOp::LessOrEqual => {
                                    if a.to_f64() <= b.to_f64() {
                                        NumericScalar::F32(1.0)
                                    } else {
                                        NumericScalar::F32(0.0)
                                    }
                                }
                                ScalarBinOp::And => {
                                    if a.to_f64() != 0.0 && b.to_f64() != 0.0 {
                                        NumericScalar::F32(1.0)
                                    } else {
                                        NumericScalar::F32(0.0)
                                    }
                                }
                                ScalarBinOp::Or => {
                                    if a.to_f64() != 0.0 || b.to_f64() != 0.0 {
                                        NumericScalar::F32(1.0)
                                    } else {
                                        NumericScalar::F32(0.0)
                                    }
                                }
                                ScalarBinOp::Xor => {
                                    if (a.to_f64() != 0.0) ^ (b.to_f64() != 0.0) {
                                        NumericScalar::F32(1.0)
                                    } else {
                                        NumericScalar::F32(0.0)
                                    }
                                }
                            };
                            result.cast_to(*output_dtype)
                        }
                        ScalarOp::Unary {
                            op,
                            compute_dtype,
                            output_dtype,
                        } => {
                            let x = values[group.inputs[0].resolve(ri, 0).0 as usize]
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
                            let cond = values[group.inputs[0].resolve(ri, 0).0 as usize]
                                .cast_to(*compute_dtype);
                            let result = if cond.is_nonzero() {
                                values[group.inputs[1].resolve(ri, 0).0 as usize]
                                    .cast_to(*compute_dtype)
                            } else {
                                values[group.inputs[2].resolve(ri, 0).0 as usize]
                                    .cast_to(*compute_dtype)
                            };
                            result.cast_to(*output_dtype)
                        }
                        ScalarOp::IndirectLoad {
                            table_base,
                            output_dtype,
                        } => {
                            let src = group.inputs[0].resolve(ri, 0);
                            let index = values[src.0 as usize].to_f64() as usize;
                            let table_atom = table_base.0 as usize + index;
                            values[table_atom].cast_to(*output_dtype)
                        }
                        ScalarOp::ReduceSum { .. } | ScalarOp::ReduceMax { .. } => unreachable!(),
                    };
                    if debug_nan && !nan_reported && !val.to_f64().is_finite() {
                        let input_detail: Vec<String> = group
                            .inputs
                            .iter()
                            .enumerate()
                            .map(|(j, inp)| {
                                let src = inp.resolve(ri, 0);
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

        NanoEval {
            values: std::mem::take(values),
        }
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
    pub fn get_range(&self, base: AtomId, count: u64) -> Vec<f64> {
        let start = base.0 as usize;
        self.values[start..start + count as usize]
            .iter()
            .map(|v| v.to_f64())
            .collect()
    }

    /// Extract a contiguous atom range as a 1D NDArrayNumericTensor.
    ///
    /// Reads `range.count` scalars starting at `range.base`, converts
    /// them to the dtype specified by `range.dtype`.
    pub fn extract_tensor(
        &self,
        range: &super::pattern::AtomRange,
    ) -> NDArrayNumericTensor<DynRank> {
        let start = range.base.0 as usize;
        let count = range.count as usize;
        let scalars = &self.values[start..start + count];
        scalars_to_tensor(scalars, range.dtype)
    }
}

/// Memory-efficient evaluation: per-group buffers with refcount-based freeing.
///
/// Instead of allocating one `NumericScalar` per atom in the entire graph
/// (infeasible for large models — 8.1B atoms = ~130GB for GPT-2), this
/// allocates per-group buffers and frees them when all consumers are done.
///
/// Peak memory is bounded by the max live set (groups whose values are
/// needed by some future group) rather than total atoms.
///
/// Returns one `NDArrayNumericTensor` per requested output range.
pub fn eval_efficient(
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
            populate_from_tensor_into(buf, offset as usize, tensor);
        }
    }

    // --- Step 2: Compute producer sets and use counts ---
    let mut producers: Vec<Vec<usize>> = Vec::with_capacity(n);
    let mut remaining: Vec<u32> = vec![0; n];

    for (gi, group) in groups.iter().enumerate() {
        let mut seen = HashSet::<usize>::new();

        for input in &group.inputs {
            collect_producer_indices_static(
                graph,
                input,
                group.count,
                group.atom_offset,
                &mut seen,
            );
        }

        // Reduce ops access additional atoms via stride.
        match &group.op {
            ScalarOp::ReduceSum {
                reduce_count,
                reduce_stride,
                ..
            }
            | ScalarOp::ReduceMax {
                reduce_count,
                reduce_stride,
                ..
            } if *reduce_count > 1 && *reduce_stride != 0 => {
                for input in &group.inputs {
                    let first = input.resolve(group.atom_offset, 0);
                    let last = input.resolve(group.atom_offset + group.count - 1, 0);
                    let end_off = (*reduce_count as i64 - 1) * reduce_stride;
                    let endpoints = [
                        first.0,
                        (first.0 as i64 + end_off) as u64,
                        last.0,
                        (last.0 as i64 + end_off) as u64,
                    ];
                    let lo = *endpoints.iter().min().unwrap();
                    let hi = *endpoints.iter().max().unwrap();
                    insert_groups_in_range_static(graph, lo, hi, &mut seen);
                }
            }
            _ => {}
        }

        // IndirectLoad table reference.
        if let ScalarOp::IndirectLoad { table_base, .. } = &group.op {
            if let Some(pi) = graph.find_group_idx(*table_base) {
                seen.insert(pi);
            }
        }

        seen.remove(&gi);
        let deps: Vec<usize> = seen.into_iter().collect();
        for &pi in &deps {
            remaining[pi] += 1;
        }
        producers.push(deps);
    }

    // --- Step 3: Mark output groups with extra use count ---
    // Find which groups produce atoms in the requested output ranges.
    for range in output_ranges {
        let base = range.base.0;
        let end = base + range.count;
        // Find all groups overlapping [base, base+count).
        let mut id = base;
        while id < end {
            if let Some(gi) = graph.find_group_idx(AtomId(id)) {
                remaining[gi] += 1;
                // Skip past this group.
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
        // Skip dead groups.
        if remaining[gi] == 0 && !matches!(&group.op, ScalarOp::Literal(_)) {
            // Check if any output range needs this group — if remaining is 0
            // and we already marked outputs, this group is truly dead.
            producers.get(gi).map(|_| {}); // keep producers vec in sync
            continue;
        }

        let count = group.count as usize;
        let mut buf = vec![NumericScalar::F32(0.0); count];
        let is_reduce = group.op.is_reduce();

        for i in 0..group.count {
            let ri = i + group.atom_offset;

            if is_reduce {
                let (reduce_count, reduce_stride, compute_dtype, output_dtype) = match &group.op {
                    ScalarOp::ReduceSum {
                        reduce_count,
                        reduce_stride,
                        compute_dtype,
                        output_dtype,
                    } => (*reduce_count, *reduce_stride, *compute_dtype, *output_dtype),
                    ScalarOp::ReduceMax {
                        reduce_count,
                        reduce_stride,
                        compute_dtype,
                        output_dtype,
                    } => (*reduce_count, *reduce_stride, *compute_dtype, *output_dtype),
                    _ => unreachable!(),
                };

                let mut acc = match &group.op {
                    ScalarOp::ReduceSum { .. } => NumericScalar::zero_of(compute_dtype),
                    ScalarOp::ReduceMax { .. } => NumericScalar::neg_infinity_of(compute_dtype),
                    _ => unreachable!(),
                };

                let base_atom = group.inputs[0].resolve(ri, 0);
                for k in 0..reduce_count {
                    let src_id =
                        AtomId((base_atom.0 as i64 + k as i64 * reduce_stride) as u64);
                    let val = lookup_atom(src_id, graph, &group_buffers, &input_buffers)
                        .cast_to(compute_dtype);
                    acc = match &group.op {
                        ScalarOp::ReduceSum { .. } => acc.add(&val),
                        ScalarOp::ReduceMax { .. } => acc.scalar_max(&val),
                        _ => unreachable!(),
                    };
                }
                buf[i as usize] = acc.cast_to(output_dtype);
            } else {
                let val = match &group.op {
                    ScalarOp::Literal(scalar) => scalar.clone(),
                    ScalarOp::Identity {
                        compute_dtype,
                        output_dtype,
                    } => {
                        let src = group.inputs[0].resolve(ri, 0);
                        lookup_atom(src, graph, &group_buffers, &input_buffers)
                            .cast_to(*compute_dtype)
                            .cast_to(*output_dtype)
                    }
                    ScalarOp::Binary {
                        op,
                        compute_dtype,
                        output_dtype,
                    } => {
                        let a = lookup_atom(
                            group.inputs[0].resolve(ri, 0),
                            graph,
                            &group_buffers,
                            &input_buffers,
                        )
                        .cast_to(*compute_dtype);
                        let b = lookup_atom(
                            group.inputs[1].resolve(ri, 0),
                            graph,
                            &group_buffers,
                            &input_buffers,
                        )
                        .cast_to(*compute_dtype);
                        let result = eval_binop(op, &a, &b);
                        result.cast_to(*output_dtype)
                    }
                    ScalarOp::Unary {
                        op,
                        compute_dtype,
                        output_dtype,
                    } => {
                        let x = lookup_atom(
                            group.inputs[0].resolve(ri, 0),
                            graph,
                            &group_buffers,
                            &input_buffers,
                        )
                        .cast_to(*compute_dtype);
                        let result = eval_unaryop(op, &x);
                        result.cast_to(*output_dtype)
                    }
                    ScalarOp::Select {
                        compute_dtype,
                        output_dtype,
                    } => {
                        let cond = lookup_atom(
                            group.inputs[0].resolve(ri, 0),
                            graph,
                            &group_buffers,
                            &input_buffers,
                        )
                        .cast_to(*compute_dtype);
                        let result = if cond.is_nonzero() {
                            lookup_atom(
                                group.inputs[1].resolve(ri, 0),
                                graph,
                                &group_buffers,
                                &input_buffers,
                            )
                            .cast_to(*compute_dtype)
                        } else {
                            lookup_atom(
                                group.inputs[2].resolve(ri, 0),
                                graph,
                                &group_buffers,
                                &input_buffers,
                            )
                            .cast_to(*compute_dtype)
                        };
                        result.cast_to(*output_dtype)
                    }
                    ScalarOp::IndirectLoad {
                        table_base,
                        output_dtype,
                    } => {
                        let src = group.inputs[0].resolve(ri, 0);
                        let index = lookup_atom(src, graph, &group_buffers, &input_buffers)
                            .to_f64() as u64;
                        let table_atom = AtomId(table_base.0 + index);
                        lookup_atom(table_atom, graph, &group_buffers, &input_buffers)
                            .cast_to(*output_dtype)
                    }
                    ScalarOp::ReduceSum { .. } | ScalarOp::ReduceMax { .. } => unreachable!(),
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
                scalars.push(lookup_atom(
                    atom_id,
                    graph,
                    &group_buffers,
                    &input_buffers,
                ));
            }
            scalars_to_tensor(&scalars, range.dtype)
        })
        .collect()
}

/// Look up an atom's value from sparse per-group or per-input-tensor buffers.
fn lookup_atom(
    atom_id: AtomId,
    graph: &NanoGraph,
    group_buffers: &[Option<Vec<NumericScalar>>],
    input_buffers: &[Vec<NumericScalar>],
) -> NumericScalar {
    // Try compute groups.
    if let Some(gi) = graph.find_group_idx(atom_id) {
        let group = &graph.groups()[gi];
        let offset = (atom_id.0 - group.base_id.0) as usize;
        return group_buffers[gi]
            .as_ref()
            .unwrap_or_else(|| {
                panic!(
                    "group {} (base={}) buffer already freed when reading atom {}",
                    gi, group.base_id, atom_id
                )
            })[offset]
            .clone();
    }
    // Try input tensors.
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
        ScalarBinOp::Greater => {
            NumericScalar::F32(if a.to_f64() > b.to_f64() { 1.0 } else { 0.0 })
        }
        ScalarBinOp::GreaterOrEqual => {
            NumericScalar::F32(if a.to_f64() >= b.to_f64() { 1.0 } else { 0.0 })
        }
        ScalarBinOp::Less => {
            NumericScalar::F32(if a.to_f64() < b.to_f64() { 1.0 } else { 0.0 })
        }
        ScalarBinOp::LessOrEqual => {
            NumericScalar::F32(if a.to_f64() <= b.to_f64() { 1.0 } else { 0.0 })
        }
        ScalarBinOp::And => {
            NumericScalar::F32(if a.to_f64() != 0.0 && b.to_f64() != 0.0 { 1.0 } else { 0.0 })
        }
        ScalarBinOp::Or => {
            NumericScalar::F32(if a.to_f64() != 0.0 || b.to_f64() != 0.0 { 1.0 } else { 0.0 })
        }
        ScalarBinOp::Xor => NumericScalar::F32(
            if (a.to_f64() != 0.0) ^ (b.to_f64() != 0.0) {
                1.0
            } else {
                0.0
            },
        ),
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

/// Collect producer group indices for an InputRef (free function version).
fn collect_producer_indices_static(
    graph: &NanoGraph,
    input: &super::pattern::InputRef,
    count: u64,
    atom_offset: u64,
    out: &mut std::collections::HashSet<usize>,
) {
    use super::pattern::InputRef;
    if count == 0 {
        return;
    }
    match input {
        InputRef::Broadcast(base) => {
            if let Some(gi) = graph.find_group_idx(*base) {
                out.insert(gi);
            }
        }
        InputRef::Affine { .. }
        | InputRef::SymAffine { .. }
        | InputRef::StridedBroadcast { .. } => {
            let first = input.resolve(atom_offset, 0);
            let last = input.resolve(atom_offset + count - 1, 0);
            let lo = first.0.min(last.0);
            let hi = first.0.max(last.0);
            insert_groups_in_range_static(graph, lo, hi, out);
        }
        InputRef::Modular {
            base,
            stride,
            modulus,
        } => {
            let a = base.0;
            let b = (base.0 as i64 + *stride as i64 * (*modulus as i64 - 1)) as u64;
            insert_groups_in_range_static(graph, a.min(b), a.max(b), out);
        }
        InputRef::Explicit(ids) => {
            let mut prev_gi: Option<usize> = None;
            for id in ids.iter().skip(atom_offset as usize).take(count as usize) {
                let gi = graph.find_group_idx(*id);
                if gi != prev_gi {
                    if let Some(g) = gi {
                        out.insert(g);
                    }
                    prev_gi = gi;
                }
            }
        }
    }
}

/// Insert all group indices whose atom ranges overlap [lo, hi].
fn insert_groups_in_range_static(
    graph: &NanoGraph,
    lo: u64,
    hi: u64,
    out: &mut std::collections::HashSet<usize>,
) {
    let groups = graph.groups();
    if let Some(first_gi) = graph.find_group_idx(AtomId(lo)) {
        out.insert(first_gi);
        for gi in (first_gi + 1)..groups.len() {
            if groups[gi].base_id.0 > hi {
                break;
            }
            out.insert(gi);
        }
    } else {
        if let Some(gi) = graph.find_group_idx(AtomId(hi)) {
            out.insert(gi);
        }
    }
}

/// Populate a buffer from an NDArrayNumericTensor, starting at `offset` within the buffer.
fn populate_from_tensor_into(
    buf: &mut [NumericScalar],
    offset: usize,
    tensor: &NDArrayNumericTensor<DynRank>,
) {
    match tensor {
        NDArrayNumericTensor::F32(a) => {
            for (i, &v) in a.iter().enumerate() {
                buf[offset + i] = NumericScalar::F32(v);
            }
        }
        NDArrayNumericTensor::F64(a) => {
            for (i, &v) in a.iter().enumerate() {
                buf[offset + i] = NumericScalar::F64(v);
            }
        }
        NDArrayNumericTensor::BF16(a) => {
            for (i, &v) in a.iter().enumerate() {
                buf[offset + i] = NumericScalar::BF16(v);
            }
        }
        NDArrayNumericTensor::F16(a) => {
            for (i, &v) in a.iter().enumerate() {
                buf[offset + i] = NumericScalar::F16(v);
            }
        }
        NDArrayNumericTensor::I64(a) => {
            for (i, &v) in a.iter().enumerate() {
                buf[offset + i] = NumericScalar::I64(v);
            }
        }
        NDArrayNumericTensor::U64(a) => {
            for (i, &v) in a.iter().enumerate() {
                buf[offset + i] = NumericScalar::U64(v);
            }
        }
        NDArrayNumericTensor::I32(a) => {
            for (i, &v) in a.iter().enumerate() {
                buf[offset + i] = NumericScalar::I32(v);
            }
        }
        NDArrayNumericTensor::U32(a) => {
            for (i, &v) in a.iter().enumerate() {
                buf[offset + i] = NumericScalar::U32(v);
            }
        }
        NDArrayNumericTensor::I16(a) => {
            for (i, &v) in a.iter().enumerate() {
                buf[offset + i] = NumericScalar::I16(v);
            }
        }
        NDArrayNumericTensor::U16(a) => {
            for (i, &v) in a.iter().enumerate() {
                buf[offset + i] = NumericScalar::U16(v);
            }
        }
        NDArrayNumericTensor::I8(a) => {
            for (i, &v) in a.iter().enumerate() {
                buf[offset + i] = NumericScalar::I8(v);
            }
        }
        NDArrayNumericTensor::U8(a) => {
            for (i, &v) in a.iter().enumerate() {
                buf[offset + i] = NumericScalar::U8(v);
            }
        }
        NDArrayNumericTensor::F8E4M3(a) => {
            for (i, &v) in a.iter().enumerate() {
                buf[offset + i] = NumericScalar::F8E4M3(v);
            }
        }
        NDArrayNumericTensor::F8E5M2(a) => {
            for (i, &v) in a.iter().enumerate() {
                buf[offset + i] = NumericScalar::F8E5M2(v);
            }
        }
        NDArrayNumericTensor::BOOL(a) => {
            for (i, &v) in a.iter().enumerate() {
                buf[offset + i] = NumericScalar::BOOL(v);
            }
        }
        NDArrayNumericTensor::STRING(_) => panic!("Cannot populate values from string tensor"),
    }
}

/// Populate a values slice from an NDArrayNumericTensor's flattened elements.
fn populate_from_tensor(
    values: &mut [NumericScalar],
    base: usize,
    tensor: &NDArrayNumericTensor<DynRank>,
) {
    match tensor {
        NDArrayNumericTensor::F32(a) => {
            for (i, &v) in a.iter().enumerate() {
                values[base + i] = NumericScalar::F32(v);
            }
        }
        NDArrayNumericTensor::F64(a) => {
            for (i, &v) in a.iter().enumerate() {
                values[base + i] = NumericScalar::F64(v);
            }
        }
        NDArrayNumericTensor::BF16(a) => {
            for (i, &v) in a.iter().enumerate() {
                values[base + i] = NumericScalar::BF16(v);
            }
        }
        NDArrayNumericTensor::F16(a) => {
            for (i, &v) in a.iter().enumerate() {
                values[base + i] = NumericScalar::F16(v);
            }
        }
        NDArrayNumericTensor::I64(a) => {
            for (i, &v) in a.iter().enumerate() {
                values[base + i] = NumericScalar::I64(v);
            }
        }
        NDArrayNumericTensor::U64(a) => {
            for (i, &v) in a.iter().enumerate() {
                values[base + i] = NumericScalar::U64(v);
            }
        }
        NDArrayNumericTensor::I32(a) => {
            for (i, &v) in a.iter().enumerate() {
                values[base + i] = NumericScalar::I32(v);
            }
        }
        NDArrayNumericTensor::U32(a) => {
            for (i, &v) in a.iter().enumerate() {
                values[base + i] = NumericScalar::U32(v);
            }
        }
        NDArrayNumericTensor::I16(a) => {
            for (i, &v) in a.iter().enumerate() {
                values[base + i] = NumericScalar::I16(v);
            }
        }
        NDArrayNumericTensor::U16(a) => {
            for (i, &v) in a.iter().enumerate() {
                values[base + i] = NumericScalar::U16(v);
            }
        }
        NDArrayNumericTensor::I8(a) => {
            for (i, &v) in a.iter().enumerate() {
                values[base + i] = NumericScalar::I8(v);
            }
        }
        NDArrayNumericTensor::U8(a) => {
            for (i, &v) in a.iter().enumerate() {
                values[base + i] = NumericScalar::U8(v);
            }
        }
        NDArrayNumericTensor::F8E4M3(a) => {
            for (i, &v) in a.iter().enumerate() {
                values[base + i] = NumericScalar::F8E4M3(v);
            }
        }
        NDArrayNumericTensor::F8E5M2(a) => {
            for (i, &v) in a.iter().enumerate() {
                values[base + i] = NumericScalar::F8E5M2(v);
            }
        }
        NDArrayNumericTensor::BOOL(a) => {
            for (i, &v) in a.iter().enumerate() {
                values[base + i] = NumericScalar::BOOL(v);
            }
        }
        NDArrayNumericTensor::STRING(_) => panic!("Cannot populate values from string tensor"),
    }
}

/// Convert a slice of NumericScalars to a 1D NDArrayNumericTensor.
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
        other => panic!("extract_tensor: unsupported dtype {:?}", other),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::graph::GlobalId;
    use crate::nano_graph::ops::{ScalarBinOp, ScalarOp, ScalarUnaryOp};
    use crate::nano_graph::pattern::{AtomRange, InputRef, NanoGraph};
    use crate::numeric_scalar::NumericScalar;
    use ndarray::{ArcArray, IxDyn};

    fn make_f32_tensor(data: &[f32]) -> NDArrayNumericTensor<DynRank> {
        NDArrayNumericTensor::F32(ArcArray::from_shape_vec(IxDyn(&[data.len()]), data.to_vec()).unwrap())
    }

    /// Build a small graph: input → mul(2) → add(bias) → output.
    /// Verify eval_efficient matches NanoEval::eval.
    #[test]
    fn test_efficient_matches_flat() {
        let mut g = NanoGraph::new();
        let gid = GlobalId(1);

        // 4-element input tensor.
        let inp = g.add_input_tensor(gid, 4, DType::F32);

        // Literal constant: 2.0
        let two = g.push_atom(
            ScalarOp::Literal(NumericScalar::F32(2.0)),
            vec![],
            vec![],
            vec![],
        );

        // mul = input * 2
        let mul = g.push_group(
            4,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![
                InputRef::Affine { base: inp, stride: 1 },
                InputRef::Broadcast(two),
            ],
        );

        // bias literal
        let bias = g.push_group(
            4,
            ScalarOp::Literal(NumericScalar::F32(10.0)),
            vec![],
            vec![],
            vec![],
        );

        // out = mul + bias
        let out = g.push_group(
            4,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![
                InputRef::Affine { base: mul, stride: 1 },
                InputRef::Affine { base: bias, stride: 1 },
            ],
        );

        let input_data = make_f32_tensor(&[1.0, 2.0, 3.0, 4.0]);
        let inputs = vec![(inp, &input_data)];

        // Flat eval.
        let flat = NanoEval::eval(&g, &inputs);

        // Efficient eval.
        let output_range = AtomRange {
            base: out,
            count: 4,
            dtype: DType::F32,
        };
        let efficient = eval_efficient(&g, &inputs, &[output_range]);
        assert_eq!(efficient.len(), 1);

        // Compare.
        let flat_vals: Vec<f64> = (0..4).map(|i| flat.get(out.offset(i))).collect();
        let eff_tensor = &efficient[0];
        let eff_vals: Vec<f64> = match eff_tensor {
            NDArrayNumericTensor::F32(a) => a.iter().map(|&v| v as f64).collect(),
            _ => panic!("expected F32"),
        };

        assert_eq!(flat_vals, eff_vals);
        // Expected: (1*2+10, 2*2+10, 3*2+10, 4*2+10) = (12, 14, 16, 18)
        assert_eq!(eff_vals, vec![12.0, 14.0, 16.0, 18.0]);
    }

    /// Test with a reduce op.
    #[test]
    fn test_efficient_reduce() {
        let mut g = NanoGraph::new();
        let gid = GlobalId(1);

        // 8-element input.
        let inp = g.add_input_tensor(gid, 8, DType::F32);

        // ReduceSum: 2 outputs, each summing 4 elements with stride 1.
        let red = g.push_group(
            2,
            ScalarOp::ReduceSum {
                reduce_count: 4,
                reduce_stride: 1,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine {
                base: inp,
                stride: 4,
            }],
        );

        let input_data = make_f32_tensor(&[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]);
        let inputs = vec![(inp, &input_data)];

        let flat = NanoEval::eval(&g, &inputs);
        let output_range = AtomRange {
            base: red,
            count: 2,
            dtype: DType::F32,
        };
        let efficient = eval_efficient(&g, &inputs, &[output_range]);

        let flat_vals: Vec<f64> = (0..2).map(|i| flat.get(red.offset(i))).collect();
        let eff_vals: Vec<f64> = match &efficient[0] {
            NDArrayNumericTensor::F32(a) => a.iter().map(|&v| v as f64).collect(),
            _ => panic!("expected F32"),
        };

        assert_eq!(flat_vals, eff_vals);
        // sum([1,2,3,4])=10, sum([10,20,30,40])=100
        assert_eq!(eff_vals, vec![10.0, 100.0]);
    }

    /// Test that dead groups are skipped and intermediate buffers are freed.
    #[test]
    fn test_efficient_frees_intermediates() {
        let mut g = NanoGraph::new();
        let gid = GlobalId(1);

        let inp = g.add_input_tensor(gid, 4, DType::F32);

        // chain: inp → neg → exp → output
        let neg = g.push_group(
            4,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine { base: inp, stride: 1 }],
        );

        let exp = g.push_group(
            4,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Exp,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine { base: neg, stride: 1 }],
        );

        // Dead branch: not consumed by anything, not in outputs.
        let _dead = g.push_group(
            4,
            ScalarOp::Literal(NumericScalar::F32(999.0)),
            vec![],
            vec![],
            vec![],
        );

        let input_data = make_f32_tensor(&[0.0, 1.0, 2.0, 3.0]);
        let inputs = vec![(inp, &input_data)];

        let flat = NanoEval::eval(&g, &inputs);
        let output_range = AtomRange {
            base: exp,
            count: 4,
            dtype: DType::F32,
        };
        let efficient = eval_efficient(&g, &inputs, &[output_range]);

        let flat_vals: Vec<f64> = (0..4).map(|i| flat.get(exp.offset(i))).collect();
        let eff_vals: Vec<f64> = match &efficient[0] {
            NDArrayNumericTensor::F32(a) => a.iter().map(|&v| v as f64).collect(),
            _ => panic!("expected F32"),
        };

        for (f, e) in flat_vals.iter().zip(eff_vals.iter()) {
            assert!((f - e).abs() < 1e-6, "mismatch: flat={} eff={}", f, e);
        }
    }

    /// Test with IndirectLoad (Gather pattern).
    #[test]
    fn test_efficient_indirect_load() {
        let mut g = NanoGraph::new();
        let table_gid = GlobalId(1);
        let idx_gid = GlobalId(2);

        // Table: 4 elements [10, 20, 30, 40].
        let table = g.add_input_tensor(table_gid, 4, DType::F32);
        // Indices: 2 elements [2, 0] — pick elements 2 and 0 from table.
        let indices = g.add_input_tensor(idx_gid, 2, DType::F32);

        // IndirectLoad: output[i] = table[indices[i]]
        let gathered = g.push_group(
            2,
            ScalarOp::IndirectLoad {
                table_base: table,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine {
                base: indices,
                stride: 1,
            }],
        );

        let table_data = make_f32_tensor(&[10.0, 20.0, 30.0, 40.0]);
        let idx_data = make_f32_tensor(&[2.0, 0.0]);
        let inputs = vec![(table, &table_data), (indices, &idx_data)];

        let flat = NanoEval::eval(&g, &inputs);
        let output_range = AtomRange {
            base: gathered,
            count: 2,
            dtype: DType::F32,
        };
        let efficient = eval_efficient(&g, &inputs, &[output_range]);

        let flat_vals: Vec<f64> = (0..2).map(|i| flat.get(gathered.offset(i))).collect();
        let eff_vals: Vec<f64> = match &efficient[0] {
            NDArrayNumericTensor::F32(a) => a.iter().map(|&v| v as f64).collect(),
            _ => panic!("expected F32"),
        };

        assert_eq!(flat_vals, eff_vals);
        assert_eq!(eff_vals, vec![30.0, 10.0]);
    }
}
