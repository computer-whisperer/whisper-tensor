//! Pool-based evaluator for NanoGraph using the new numeric types.
//!
//! Each AtomGroup is backed by a 1D [`NumericTensor`] allocated from the pool.
//! Elements are stored at their native dtype width (e.g. 2 bytes per BF16 atom
//! instead of 24 bytes per `NumericScalar` enum variant).
//!
//! Group buffers are freed as soon as their last consumer finishes, returning
//! memory to the pool. Output buffers are returned to the caller.

use std::collections::{HashMap, HashSet};

use crate::numeric_dtype::NumericDType;
use crate::numeric_scalar::NumericScalar;
use crate::numeric_tensor::{NumericTensor, NumericTensorView, TensorLayout};
use crate::pool::{AllocationError, Pool};
use crate::tensor_rank::DynRank;

use super::ops::{ReduceKind, ScalarBinOp, ScalarOp, ScalarUnaryOp};
use super::pattern::{AtomId, AtomRange, NanoGraph};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Evaluate a NanoGraph using pool-managed buffers.
///
/// Inputs are `NumericTensorView`s (pool-type-erased). Each output range
/// produces a `NumericTensor` allocated from `pool`.
pub fn pool_eval<'p, P: Pool + 'p>(
    graph: &NanoGraph,
    inputs: &[(AtomId, &NumericTensorView<'_, DynRank>)],
    output_ranges: &[AtomRange],
    pool: &'p P,
) -> Result<Vec<NumericTensor<'p, DynRank, P>>, PoolEvalError> {
    let groups = graph.groups();
    let n = groups.len();
    let input_tensors = graph.input_tensors();

    // --- Step 1: Populate input buffers as 1D NumericTensors ---
    let input_buffers: Vec<InputBuffer> = input_tensors
        .iter()
        .map(|it| {
            let dtype = it.dtype;
            InputBuffer {
                base: it.base_id,
                count: it.count,
                dtype,
                data: vec![0u8; dtype.bytes_per_element() * it.count as usize],
            }
        })
        .collect();

    // Fill input buffers from the provided tensor views.
    let mut input_buffers = input_buffers;
    for &(base, view) in inputs {
        if let Some((ti, offset)) = graph.find_input_idx(base) {
            let buf = &mut input_buffers[ti];
            for i in 0..view.numel() {
                let scalar = view.read_element(i);
                let cast = scalar.cast_to(buf.dtype);
                let bit_offset = (offset as usize + i) * buf.dtype.total_bits() as usize;
                crate::numeric_scalar::conversions::write_raw_bits(
                    &mut buf.data,
                    bit_offset,
                    buf.dtype.total_bits(),
                    crate::numeric_scalar::conversions::read_raw_bits(
                        cast.raw_bits(),
                        0,
                        buf.dtype.total_bits(),
                    ),
                );
            }
        }
    }

    // --- Step 2: Compute producer dependencies and use counts ---
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

    // Mark output groups with extra refcount so they survive.
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

    // --- Step 3: Evaluate groups in topological order ---
    let mut group_buffers: Vec<Option<GroupBuffer<'p, P>>> = (0..n).map(|_| None).collect();

    // Cache for opaque op results. Key: opaque_idx, Value: output tensors.
    let mut opaque_cache: HashMap<usize, Vec<NumericTensor<'p, DynRank, P>>> = HashMap::new();

    for (gi, group) in groups.iter().enumerate() {
        if remaining[gi] == 0 && !matches!(&group.op, ScalarOp::Literal(_)) {
            continue;
        }

        // --- Handle OpaqueOutput: dispatch to opaque eval function ---
        if let ScalarOp::OpaqueOutput {
            opaque_idx,
            output_idx,
        } = &group.op
        {
            // Evaluate the opaque op if not cached.
            if !opaque_cache.contains_key(opaque_idx) {
                let opaque_op = &graph.opaque_ops()[*opaque_idx];

                // Assemble input tensors from atom buffers.
                let mut input_tensors = Vec::with_capacity(opaque_op.inputs.len());
                for inp in &opaque_op.inputs {
                    let inp_layout =
                        TensorLayout::<DynRank>::row_major(inp.shape.clone(), inp.dtype);
                    let mut inp_buf = pool
                        .allocate(inp_layout.buffer_size_bytes())
                        .map_err(PoolEvalError::Allocation)?;
                    let mut inp_tensor: NumericTensor<'p, DynRank, P> =
                        NumericTensor::from_parts(inp_buf, inp_layout);
                    for elem in 0..inp.count as usize {
                        let atom_id = AtomId(inp.base.0 + elem as u64);
                        let val = lookup_atom_raw(atom_id, graph, &group_buffers, &input_buffers);
                        let val_dtype =
                            lookup_atom_dtype(atom_id, graph, &group_buffers, &input_buffers);
                        let scalar = NumericScalar {
                            bits: val.to_le_bytes(),
                            dtype: val_dtype,
                        };
                        inp_tensor.write_element(elem, scalar.cast_to(inp.dtype));
                    }
                    input_tensors.push(inp_tensor);
                }

                let input_views: Vec<_> = input_tensors.iter().map(|t| t.view()).collect();
                let sys_results = opaque_op.eval_fn.eval(&input_views)?;
                // Copy SystemPool results into the caller's pool.
                let results: Vec<NumericTensor<'p, DynRank, P>> = sys_results
                    .iter()
                    .map(|sys_t| {
                        let layout = sys_t.layout().clone();
                        let mut buf = pool
                            .allocate(layout.buffer_size_bytes())
                            .expect("pool alloc failed copying opaque result");
                        let mut t = NumericTensor::from_parts(buf, layout);
                        for i in 0..sys_t.numel() {
                            t.write_element(i, sys_t.read_element(i));
                        }
                        t
                    })
                    .collect();
                opaque_cache.insert(*opaque_idx, results);
            }

            // Read this output from the cache.
            let cached = &opaque_cache[opaque_idx];
            let result_tensor = &cached[*output_idx];

            // Copy result into the group buffer.
            let count = group.count as usize;
            let output_dtype = group.output_dtype;
            let layout = TensorLayout::<DynRank>::row_major(vec![count as u64], output_dtype);
            let buffer = pool
                .allocate(layout.buffer_size_bytes())
                .map_err(PoolEvalError::Allocation)?;
            let mut tensor = NumericTensor::from_parts(buffer, layout);
            for i in 0..count {
                let scalar = result_tensor.read_element(i);
                tensor.write_element(i, scalar);
            }

            group_buffers[gi] = Some(GroupBuffer {
                tensor,
                dtype: output_dtype,
            });

            // Free spent producers.
            for &pi in &producers[gi] {
                remaining[pi] -= 1;
                if remaining[pi] == 0 {
                    group_buffers[pi] = None;
                }
            }
            continue;
        }

        let count = group.count as usize;
        let output_dtype = group.output_dtype;
        let buf_size = output_dtype.bytes_per_element() * count;
        let layout = TensorLayout::<DynRank>::row_major(vec![count as u64], output_dtype);

        // Allocate group buffer from the pool.
        let buffer = pool.allocate(buf_size).map_err(PoolEvalError::Allocation)?;
        let mut tensor = NumericTensor::from_parts(buffer, layout);

        for i in 0..group.count {
            let ri = i + group.atom_offset;

            if let ScalarOp::Reduce {
                kind,
                reduce_count,
                reduce_stride,
                compute_dtype,
            } = &group.op
            {
                let mut acc_raw = match kind {
                    ReduceKind::Sum => compute_dtype.encode_from_f64(0.0),
                    ReduceKind::Max => compute_dtype.encode_from_f64(f64::NEG_INFINITY),
                    ReduceKind::Min => compute_dtype.encode_from_f64(f64::INFINITY),
                    ReduceKind::Prod => compute_dtype.encode_from_f64(1.0),
                };

                let base_atom = group.inputs[0].resolve(ri);
                for k in 0..*reduce_count {
                    let src_id = AtomId((base_atom.0 as i64 + k as i64 * reduce_stride) as u64);
                    let val = lookup_atom_raw(src_id, graph, &group_buffers, &input_buffers);
                    let val_dtype =
                        lookup_atom_dtype(src_id, graph, &group_buffers, &input_buffers);
                    let cast_raw = val_dtype.cast_raw(val, *compute_dtype);

                    acc_raw = match kind {
                        ReduceKind::Sum => {
                            let sum = compute_dtype.decode_to_f64(acc_raw)
                                + compute_dtype.decode_to_f64(cast_raw);
                            compute_dtype.encode_from_f64(sum)
                        }
                        ReduceKind::Max => {
                            let a = compute_dtype.decode_to_f64(acc_raw);
                            let b = compute_dtype.decode_to_f64(cast_raw);
                            compute_dtype.encode_from_f64(a.max(b))
                        }
                        ReduceKind::Min => {
                            let a = compute_dtype.decode_to_f64(acc_raw);
                            let b = compute_dtype.decode_to_f64(cast_raw);
                            compute_dtype.encode_from_f64(a.min(b))
                        }
                        ReduceKind::Prod => {
                            let prod = compute_dtype.decode_to_f64(acc_raw)
                                * compute_dtype.decode_to_f64(cast_raw);
                            compute_dtype.encode_from_f64(prod)
                        }
                    };
                }
                let result = compute_dtype.cast_raw(acc_raw, output_dtype);
                write_atom(&mut tensor, i as usize, result, output_dtype);
            } else {
                let result_raw = match &group.op {
                    ScalarOp::Literal(scalar) => {
                        let raw = scalar.view().read_raw();
                        scalar.dtype().cast_raw(raw, output_dtype)
                    }
                    ScalarOp::Identity => {
                        let src = group.inputs[0].resolve(ri);
                        let val = lookup_atom_raw(src, graph, &group_buffers, &input_buffers);
                        let val_dtype =
                            lookup_atom_dtype(src, graph, &group_buffers, &input_buffers);
                        val_dtype.cast_raw(val, output_dtype)
                    }
                    ScalarOp::Binary { op, compute_dtype } => {
                        let a_src = group.inputs[0].resolve(ri);
                        let b_src = group.inputs[1].resolve(ri);
                        let a_raw = lookup_atom_raw(a_src, graph, &group_buffers, &input_buffers);
                        let b_raw = lookup_atom_raw(b_src, graph, &group_buffers, &input_buffers);
                        let a_dtype =
                            lookup_atom_dtype(a_src, graph, &group_buffers, &input_buffers);
                        let b_dtype =
                            lookup_atom_dtype(b_src, graph, &group_buffers, &input_buffers);
                        let a_cast = a_dtype.cast_raw(a_raw, *compute_dtype);
                        let b_cast = b_dtype.cast_raw(b_raw, *compute_dtype);
                        let result = eval_binop(op, a_cast, b_cast, *compute_dtype);
                        compute_dtype.cast_raw(result, output_dtype)
                    }
                    ScalarOp::Unary { op, compute_dtype } => {
                        let src = group.inputs[0].resolve(ri);
                        let val = lookup_atom_raw(src, graph, &group_buffers, &input_buffers);
                        let val_dtype =
                            lookup_atom_dtype(src, graph, &group_buffers, &input_buffers);
                        let x = val_dtype.cast_raw(val, *compute_dtype);
                        let result = eval_unaryop(op, x, *compute_dtype);
                        compute_dtype.cast_raw(result, output_dtype)
                    }
                    ScalarOp::Select => {
                        let cond_src = group.inputs[0].resolve(ri);
                        let cond_raw =
                            lookup_atom_raw(cond_src, graph, &group_buffers, &input_buffers);
                        let cond_dtype =
                            lookup_atom_dtype(cond_src, graph, &group_buffers, &input_buffers);
                        let is_true = cond_dtype.decode_to_f64(cond_raw) != 0.0;
                        let chosen_src = if is_true {
                            group.inputs[1].resolve(ri)
                        } else {
                            group.inputs[2].resolve(ri)
                        };
                        let val =
                            lookup_atom_raw(chosen_src, graph, &group_buffers, &input_buffers);
                        let val_dtype =
                            lookup_atom_dtype(chosen_src, graph, &group_buffers, &input_buffers);
                        val_dtype.cast_raw(val, output_dtype)
                    }
                    ScalarOp::IndirectLoad { table_base } => {
                        let idx_src = group.inputs[0].resolve(ri);
                        let idx_raw =
                            lookup_atom_raw(idx_src, graph, &group_buffers, &input_buffers);
                        let idx_dtype =
                            lookup_atom_dtype(idx_src, graph, &group_buffers, &input_buffers);
                        let index = idx_dtype.decode_to_f64(idx_raw) as u64;
                        let table_atom = AtomId(table_base.0 + index);
                        let val =
                            lookup_atom_raw(table_atom, graph, &group_buffers, &input_buffers);
                        let val_dtype =
                            lookup_atom_dtype(table_atom, graph, &group_buffers, &input_buffers);
                        val_dtype.cast_raw(val, output_dtype)
                    }
                    ScalarOp::Reduce { .. } | ScalarOp::OpaqueOutput { .. } => unreachable!(),
                };
                write_atom(&mut tensor, i as usize, result_raw, output_dtype);
            }
        }

        group_buffers[gi] = Some(GroupBuffer {
            tensor,
            dtype: output_dtype,
        });

        // Free spent producer buffers.
        for &pi in &producers[gi] {
            remaining[pi] -= 1;
            if remaining[pi] == 0 {
                group_buffers[pi] = None;
            }
        }
    }

    // --- Step 4: Assemble output tensors ---
    let mut outputs = Vec::with_capacity(output_ranges.len());
    for range in output_ranges {
        let count = range.count as usize;
        let output_dtype = range.dtype;
        let layout = TensorLayout::<DynRank>::row_major(vec![count as u64], output_dtype);
        let buffer = pool
            .allocate(layout.buffer_size_bytes())
            .map_err(PoolEvalError::Allocation)?;
        let mut out_tensor = NumericTensor::from_parts(buffer, layout);

        for offset in 0..range.count {
            let atom_id = AtomId(range.base.0 + offset);
            let val = lookup_atom_raw(atom_id, graph, &group_buffers, &input_buffers);
            let val_dtype = lookup_atom_dtype(atom_id, graph, &group_buffers, &input_buffers);
            let cast = val_dtype.cast_raw(val, output_dtype);
            write_atom(&mut out_tensor, offset as usize, cast, output_dtype);
        }
        outputs.push(out_tensor);
    }

    Ok(outputs)
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum PoolEvalError {
    #[error("allocation failed: {0}")]
    Allocation(#[from] AllocationError),
    #[error("unsupported op: {0}")]
    Unsupported(String),
}

// ---------------------------------------------------------------------------
// Internal data structures
// ---------------------------------------------------------------------------

/// Input tensor buffer — owns a Vec<u8> with elements at native dtype width.
struct InputBuffer {
    base: AtomId,
    count: u64,
    dtype: NumericDType,
    data: Vec<u8>,
}

/// Group output buffer — a pool-managed 1D NumericTensor.
struct GroupBuffer<'p, P: Pool + 'p> {
    tensor: NumericTensor<'p, DynRank, P>,
    dtype: NumericDType,
}

// ---------------------------------------------------------------------------
// Atom access
// ---------------------------------------------------------------------------

/// Read an atom's raw bits from group buffers or input buffers.
fn lookup_atom_raw<P: Pool>(
    atom_id: AtomId,
    graph: &NanoGraph,
    group_buffers: &[Option<GroupBuffer<'_, P>>],
    input_buffers: &[InputBuffer],
) -> u64 {
    if let Some(gi) = graph.find_group_idx(atom_id) {
        let gb = group_buffers[gi]
            .as_ref()
            .unwrap_or_else(|| panic!("group {gi} buffer freed when reading atom {atom_id}"));
        let group = &graph.groups()[gi];
        let offset = (atom_id.0 - group.base_id.0) as usize;
        return gb.tensor.read_element(offset).view().read_raw();
    }
    if let Some((ti, offset)) = graph.find_input_idx(atom_id) {
        let ib = &input_buffers[ti];
        let bit_off = offset as usize * ib.dtype.total_bits() as usize;
        return crate::numeric_scalar::conversions::read_raw_bits(
            &ib.data,
            bit_off,
            ib.dtype.total_bits(),
        );
    }
    panic!("atom {atom_id} not found in any group or input");
}

/// Get the dtype of an atom.
fn lookup_atom_dtype<P: Pool>(
    atom_id: AtomId,
    graph: &NanoGraph,
    group_buffers: &[Option<GroupBuffer<'_, P>>],
    input_buffers: &[InputBuffer],
) -> NumericDType {
    if let Some(gi) = graph.find_group_idx(atom_id) {
        return group_buffers[gi].as_ref().unwrap().dtype;
    }
    if let Some((ti, _)) = graph.find_input_idx(atom_id) {
        return input_buffers[ti].dtype;
    }
    panic!("atom {atom_id} not found");
}

/// Write a raw u64 value to a tensor at a flat index.
fn write_atom<P: Pool>(
    tensor: &mut NumericTensor<'_, DynRank, P>,
    index: usize,
    raw: u64,
    dtype: NumericDType,
) {
    let mut bits = [0u8; 8];
    let nbytes = dtype.bytes_per_element();
    bits[..nbytes].copy_from_slice(&raw.to_le_bytes()[..nbytes]);
    tensor.write_element(index, NumericScalar { bits, dtype });
}

// ---------------------------------------------------------------------------
// Op dispatch
// ---------------------------------------------------------------------------

fn eval_binop(op: &ScalarBinOp, a: u64, b: u64, dtype: NumericDType) -> u64 {
    match dtype {
        NumericDType::Float(ft) => match op {
            ScalarBinOp::Add => crate::scalar_ops::add::float_add(a, b, &ft),
            ScalarBinOp::Sub => crate::scalar_ops::sub::float_sub(a, b, &ft),
            ScalarBinOp::Mul => crate::scalar_ops::mul::float_mul(a, b, &ft),
            ScalarBinOp::Div => crate::scalar_ops::div::float_div(a, b, &ft),
            ScalarBinOp::Max => crate::scalar_ops::max::float_max(a, b, &ft),
            ScalarBinOp::Min => crate::scalar_ops::min::float_min(a, b, &ft),
            ScalarBinOp::Mod => crate::scalar_ops::modulo::float_mod(a, b, &ft),
            ScalarBinOp::Pow => crate::scalar_ops::pow::float_pow(a, b, &ft),
            ScalarBinOp::Equal => {
                bool_to_dtype_raw(crate::scalar_ops::cmp::float_equal(a, b, &ft), dtype)
            }
            ScalarBinOp::Greater => {
                bool_to_dtype_raw(crate::scalar_ops::cmp::float_greater(a, b, &ft), dtype)
            }
            ScalarBinOp::GreaterOrEqual => bool_to_dtype_raw(
                crate::scalar_ops::cmp::float_greater_or_equal(a, b, &ft),
                dtype,
            ),
            ScalarBinOp::Less => {
                bool_to_dtype_raw(crate::scalar_ops::cmp::float_less(a, b, &ft), dtype)
            }
            ScalarBinOp::LessOrEqual => bool_to_dtype_raw(
                crate::scalar_ops::cmp::float_less_or_equal(a, b, &ft),
                dtype,
            ),
            ScalarBinOp::And => {
                bool_to_dtype_raw(crate::scalar_ops::logical::logical_and(a, b), dtype)
            }
            ScalarBinOp::Or => {
                bool_to_dtype_raw(crate::scalar_ops::logical::logical_or(a, b), dtype)
            }
            ScalarBinOp::Xor => {
                bool_to_dtype_raw(crate::scalar_ops::logical::logical_xor(a, b), dtype)
            }
        },
        NumericDType::SignedInt(it) => match op {
            ScalarBinOp::Add => crate::scalar_ops::add::signed_add_wrapping(a, b, &it),
            ScalarBinOp::Sub => crate::scalar_ops::sub::signed_sub_wrapping(a, b, &it),
            ScalarBinOp::Mul => crate::scalar_ops::mul::signed_mul_wrapping(a, b, &it),
            ScalarBinOp::Div => crate::scalar_ops::div::signed_div_wrapping(a, b, &it),
            ScalarBinOp::Max => crate::scalar_ops::max::signed_max(a, b, &it),
            ScalarBinOp::Min => crate::scalar_ops::min::signed_min(a, b, &it),
            ScalarBinOp::Mod => crate::scalar_ops::modulo::signed_mod(a, b, &it),
            ScalarBinOp::Pow => crate::scalar_ops::pow::signed_pow(a, b, &it),
            ScalarBinOp::Equal => {
                bool_to_dtype_raw(crate::scalar_ops::cmp::signed_equal(a, b, &it), dtype)
            }
            ScalarBinOp::Greater => {
                bool_to_dtype_raw(crate::scalar_ops::cmp::signed_greater(a, b, &it), dtype)
            }
            ScalarBinOp::GreaterOrEqual => bool_to_dtype_raw(
                crate::scalar_ops::cmp::signed_greater_or_equal(a, b, &it),
                dtype,
            ),
            ScalarBinOp::Less => {
                bool_to_dtype_raw(crate::scalar_ops::cmp::signed_less(a, b, &it), dtype)
            }
            ScalarBinOp::LessOrEqual => bool_to_dtype_raw(
                crate::scalar_ops::cmp::signed_less_or_equal(a, b, &it),
                dtype,
            ),
            ScalarBinOp::And => {
                bool_to_dtype_raw(crate::scalar_ops::logical::logical_and(a, b), dtype)
            }
            ScalarBinOp::Or => {
                bool_to_dtype_raw(crate::scalar_ops::logical::logical_or(a, b), dtype)
            }
            ScalarBinOp::Xor => {
                bool_to_dtype_raw(crate::scalar_ops::logical::logical_xor(a, b), dtype)
            }
        },
        NumericDType::UnsignedInt(it) => match op {
            ScalarBinOp::Add => crate::scalar_ops::add::unsigned_add_wrapping(a, b, &it),
            ScalarBinOp::Sub => crate::scalar_ops::sub::unsigned_sub_wrapping(a, b, &it),
            ScalarBinOp::Mul => crate::scalar_ops::mul::unsigned_mul_wrapping(a, b, &it),
            ScalarBinOp::Div => crate::scalar_ops::div::unsigned_div_wrapping(a, b, &it),
            ScalarBinOp::Max => crate::scalar_ops::max::unsigned_max(a, b, &it),
            ScalarBinOp::Min => crate::scalar_ops::min::unsigned_min(a, b, &it),
            ScalarBinOp::Mod => crate::scalar_ops::modulo::unsigned_mod(a, b, &it),
            ScalarBinOp::Pow => crate::scalar_ops::pow::unsigned_pow(a, b, &it),
            ScalarBinOp::Equal => {
                bool_to_dtype_raw(crate::scalar_ops::cmp::unsigned_equal(a, b, &it), dtype)
            }
            ScalarBinOp::Greater => {
                bool_to_dtype_raw(crate::scalar_ops::cmp::unsigned_greater(a, b, &it), dtype)
            }
            ScalarBinOp::GreaterOrEqual => bool_to_dtype_raw(
                crate::scalar_ops::cmp::unsigned_greater_or_equal(a, b, &it),
                dtype,
            ),
            ScalarBinOp::Less => {
                bool_to_dtype_raw(crate::scalar_ops::cmp::unsigned_less(a, b, &it), dtype)
            }
            ScalarBinOp::LessOrEqual => bool_to_dtype_raw(
                crate::scalar_ops::cmp::unsigned_less_or_equal(a, b, &it),
                dtype,
            ),
            ScalarBinOp::And => {
                bool_to_dtype_raw(crate::scalar_ops::logical::logical_and(a, b), dtype)
            }
            ScalarBinOp::Or => {
                bool_to_dtype_raw(crate::scalar_ops::logical::logical_or(a, b), dtype)
            }
            ScalarBinOp::Xor => {
                bool_to_dtype_raw(crate::scalar_ops::logical::logical_xor(a, b), dtype)
            }
        },
        NumericDType::Bool => {
            // Comparisons and logical ops on Bool
            match op {
                ScalarBinOp::Equal => bool_to_dtype_raw(if a == b { 1 } else { 0 }, dtype),
                ScalarBinOp::And => {
                    bool_to_dtype_raw(crate::scalar_ops::logical::logical_and(a, b), dtype)
                }
                ScalarBinOp::Or => {
                    bool_to_dtype_raw(crate::scalar_ops::logical::logical_or(a, b), dtype)
                }
                ScalarBinOp::Xor => {
                    bool_to_dtype_raw(crate::scalar_ops::logical::logical_xor(a, b), dtype)
                }
                _ => panic!("unsupported binop {op:?} for Bool"),
            }
        }
    }
}

fn eval_unaryop(op: &ScalarUnaryOp, x: u64, dtype: NumericDType) -> u64 {
    match dtype {
        NumericDType::Float(ft) => match op {
            ScalarUnaryOp::Neg => crate::scalar_ops::neg::float_neg(x, &ft),
            ScalarUnaryOp::Abs => crate::scalar_ops::abs::float_abs(x, &ft),
            ScalarUnaryOp::Exp => crate::scalar_ops::exp::float_exp(x, &ft),
            ScalarUnaryOp::Ln => crate::scalar_ops::ln::float_ln(x, &ft),
            ScalarUnaryOp::Sqrt => crate::scalar_ops::sqrt::float_sqrt(x, &ft),
            ScalarUnaryOp::Reciprocal => crate::scalar_ops::reciprocal::float_reciprocal(x, &ft),
            ScalarUnaryOp::Tanh => crate::scalar_ops::trig::float_tanh(x, &ft),
            ScalarUnaryOp::Floor => crate::scalar_ops::floor::float_floor(x, &ft),
            ScalarUnaryOp::Ceil => crate::scalar_ops::ceil::float_ceil(x, &ft),
            ScalarUnaryOp::Round => crate::scalar_ops::round::float_round(x, &ft),
            ScalarUnaryOp::Sign => crate::scalar_ops::sign::float_sign(x, &ft),
            ScalarUnaryOp::Erf => crate::scalar_ops::erf::float_erf(x, &ft),
            ScalarUnaryOp::Sin => crate::scalar_ops::trig::float_sin(x, &ft),
            ScalarUnaryOp::Cos => crate::scalar_ops::trig::float_cos(x, &ft),
            ScalarUnaryOp::Not => {
                // Logical NOT for float: nonzero → 0.0, zero → 1.0
                let val = ft.decode_f64(x);
                ft.encode_f64(if val != 0.0 && !val.is_nan() {
                    0.0
                } else {
                    1.0
                })
            }
            ScalarUnaryOp::IsNan => {
                let val = ft.decode_f64(x);
                // IsNan returns in the compute dtype (typically BOOL after cast)
                dtype.encode_from_f64(if val.is_nan() { 1.0 } else { 0.0 })
            }
            ScalarUnaryOp::IsInf {
                detect_positive,
                detect_negative,
            } => {
                let result = crate::scalar_ops::is_inf::float_is_inf(
                    x,
                    &ft,
                    *detect_positive,
                    *detect_negative,
                );
                dtype.encode_from_f64(result as f64)
            }
            ScalarUnaryOp::BitwiseNot => {
                panic!("BitwiseNot is not applicable to float types")
            }
            ScalarUnaryOp::Log1p => ft.encode_f64((ft.decode_f64(x) + 1.0).ln()),
            ScalarUnaryOp::Tan => crate::scalar_ops::trig::float_tan(x, &ft),
            ScalarUnaryOp::Asin => crate::scalar_ops::trig::float_asin(x, &ft),
            ScalarUnaryOp::Acos => crate::scalar_ops::trig::float_acos(x, &ft),
            ScalarUnaryOp::Atan => crate::scalar_ops::trig::float_atan(x, &ft),
            ScalarUnaryOp::Sinh => crate::scalar_ops::trig::float_sinh(x, &ft),
            ScalarUnaryOp::Cosh => crate::scalar_ops::trig::float_cosh(x, &ft),
            ScalarUnaryOp::Asinh => crate::scalar_ops::trig::float_asinh(x, &ft),
            ScalarUnaryOp::Acosh => crate::scalar_ops::trig::float_acosh(x, &ft),
            ScalarUnaryOp::Atanh => crate::scalar_ops::trig::float_atanh(x, &ft),
        },
        NumericDType::SignedInt(it) => match op {
            ScalarUnaryOp::Neg => crate::scalar_ops::neg::signed_neg_wrapping(x, &it),
            ScalarUnaryOp::Abs => crate::scalar_ops::abs::signed_abs_wrapping(x, &it),
            ScalarUnaryOp::Sign => crate::scalar_ops::sign::signed_sign(x, &it),
            ScalarUnaryOp::Not => {
                // Logical NOT for int: nonzero → 0, zero → 1
                let val = it.decode_signed(x);
                it.encode_signed(if val != 0 { 0 } else { 1 })
            }
            ScalarUnaryOp::BitwiseNot => crate::scalar_ops::bitwise::bitwise_not(x, &it),
            _ => panic!("unsupported unary op {op:?} for signed int"),
        },
        NumericDType::UnsignedInt(it) => match op {
            ScalarUnaryOp::Abs => x,
            ScalarUnaryOp::Not => {
                let val = it.decode_unsigned(x);
                it.encode_unsigned(if val != 0 { 0 } else { 1 })
            }
            ScalarUnaryOp::BitwiseNot => crate::scalar_ops::bitwise::bitwise_not(x, &it),
            _ => panic!("unsupported unary op {op:?} for unsigned int"),
        },
        NumericDType::Bool => match op {
            ScalarUnaryOp::Not => dtype.encode_from_f64(if x != 0 { 0.0 } else { 1.0 }),
            _ => panic!("unsupported unary op {op:?} for Bool"),
        },
    }
}

/// Encode a boolean comparison/logical result as raw bits in the given dtype.
/// The old eval returns F32(1.0)/F32(0.0) for comparisons. We match that by
/// encoding 1.0 or 0.0 in the compute dtype.
fn bool_to_dtype_raw(val: u64, dtype: NumericDType) -> u64 {
    let f = if val != 0 { 1.0 } else { 0.0 };
    dtype.encode_from_f64(f)
}
