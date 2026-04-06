//! Pool-based evaluator for NanoGraph using the new numeric types.
//!
//! Each AtomGroup is backed by a [`NumericTensor`] allocated from the pool.
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

use super::lower::TensorAtomMapInfo;
use super::ops::{ReduceKind, ScalarBinOp, ScalarOp, ScalarUnaryOp};
use super::pattern::{AtomId, NanoGraph};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Evaluate a NanoGraph using pool-managed buffers.
///
/// Inputs and outputs are described by [`TensorAtomMapInfo`] which maps
/// logical tensor elements to atom IDs via strides. Pool_eval uses this
/// mapping to populate input stores and assemble correctly-shaped output
/// tensors — callers don't need post-eval reassembly.
pub fn pool_eval<'p, P: Pool + 'p>(
    graph: &NanoGraph<'_, impl Pool>,
    inputs: &[(&TensorAtomMapInfo, &NumericTensorView<'_, DynRank>)],
    outputs: &[&TensorAtomMapInfo],
    pool: &'p P,
) -> Result<Vec<NumericTensor<'p, DynRank, P>>, PoolEvalError> {
    let groups = graph.groups();
    let n = groups.len();
    let input_tensors = graph.input_tensors();

    // --- Step 1: Populate input stores via relayout ---
    //
    // Build the TensorLayout the TAMI expects, then relayout the view to match.
    // If strides already agree, relayout borrows (zero-copy). Otherwise it copies.
    let mut input_stores: Vec<Option<AtomStore<'_, 'p, P>>> =
        (0..input_tensors.len()).map(|_| None).collect();

    for &(tam, view) in inputs {
        if let Some((ti, _offset)) = graph.find_input_idx(tam.base_id) {
            let element_bits = tam.dtype.total_bits() as u64;
            let strides_bits: Vec<u64> = tam
                .known_strides
                .iter()
                .map(|&s| s * element_bits)
                .collect();
            let target = TensorLayout::<DynRank>::ElementStrided {
                shape: tam.known_dims.clone(),
                dtype: tam.dtype,
                strides: strides_bits,
                offset_bits: 0,
            };
            let cow = view
                .relayout(target, pool)
                .expect("pool alloc for input relayout");
            match cow {
                crate::numeric_tensor::NumericTensorCOW::Borrowed(flat) => {
                    input_stores[ti] = Some(AtomStore::View(flat));
                }
                crate::numeric_tensor::NumericTensorCOW::Owned(tensor) => {
                    input_stores[ti] = Some(AtomStore::Owned(tensor));
                }
            }
        }
    }

    // Fill any remaining input stores that weren't provided by the caller.
    let input_stores: Vec<AtomStore<'_, 'p, P>> = input_stores
        .into_iter()
        .enumerate()
        .map(|(ti, store)| {
            store.unwrap_or_else(|| {
                let it = &input_tensors[ti];
                let layout = TensorLayout::<DynRank>::row_major(vec![it.count], it.dtype);
                let buf = pool
                    .allocate(layout.buffer_size_bytes())
                    .expect("pool alloc for input buffer");
                AtomStore::Owned(NumericTensor::from_parts(buf, layout))
            })
        })
        .collect();

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

    // Mark groups that contribute to outputs with extra refcount so they survive.
    for tam in outputs {
        let ranges = tam.atom_ranges(graph);
        for range in &ranges {
            let mut id = range.base.0;
            let end = range.base.0 + range.count;
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
    }

    let mut group_stores: Vec<Option<AtomStore<'_, 'p, P>>> = (0..n).map(|_| None).collect();

    // --- Step 3: Evaluate groups in topological order ---

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
                let mut op_input_tensors = Vec::with_capacity(opaque_op.inputs.len());
                for inp in &opaque_op.inputs {
                    let inp_layout =
                        TensorLayout::<DynRank>::row_major(inp.shape.clone(), inp.dtype);
                    let inp_buf = pool
                        .allocate(inp_layout.buffer_size_bytes())
                        .map_err(PoolEvalError::Allocation)?;
                    let mut inp_tensor: NumericTensor<'p, DynRank, P> =
                        NumericTensor::from_parts(inp_buf, inp_layout);
                    for elem in 0..inp.count as usize {
                        let atom_id = AtomId(inp.base.0 + elem as u64);
                        let scalar =
                            lookup_atom_scalar(atom_id, graph, &group_stores, &input_stores);
                        inp_tensor.write_element(elem, scalar.cast_to(inp.dtype));
                    }
                    op_input_tensors.push(inp_tensor);
                }

                let input_views: Vec<_> = op_input_tensors.iter().map(|t| t.view()).collect();
                let sys_results = opaque_op.eval_fn.eval(&input_views)?;
                // Copy SystemPool results into the caller's pool.
                let results: Vec<NumericTensor<'p, DynRank, P>> = sys_results
                    .iter()
                    .map(|sys_t| {
                        let layout = sys_t.layout().clone();
                        let buf = pool
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

            // Copy result into the group store (or pre-allocated output store).
            let count = group.count as usize;
            let output_dtype = group.output_dtype;
            let store = group_stores[gi].get_or_insert_with(|| {
                let layout = TensorLayout::<DynRank>::row_major(vec![count as u64], output_dtype);
                let buffer = pool
                    .allocate(layout.buffer_size_bytes())
                    .expect("pool alloc for opaque group");
                AtomStore::Owned(NumericTensor::from_parts(buffer, layout))
            });
            for i in 0..count {
                let scalar = result_tensor.read_element(i);
                store.write_element(i, scalar);
            }

            // Free spent producers.
            for &pi in &producers[gi] {
                remaining[pi] -= 1;
                if remaining[pi] == 0 {
                    group_stores[pi] = None;
                }
            }
            continue;
        }

        // --- LiteralSpan: borrow the tensor directly (zero-copy) ---
        if let ScalarOp::LiteralSpan(ref tensor) = group.op
            && let Some(flat) = tensor.view().flatten()
        {
            group_stores[gi] = Some(AtomStore::View(flat));

            // Free spent producers.
            for &pi in &producers[gi] {
                remaining[pi] -= 1;
                if remaining[pi] == 0 {
                    group_stores[pi] = None;
                }
            }
            continue;
        }

        let count = group.count as usize;
        let output_dtype = group.output_dtype;

        // Take the store out so we can mutate it while reading other stores.
        let mut store = group_stores[gi].take().unwrap_or_else(|| {
            let layout = TensorLayout::<DynRank>::row_major(vec![count as u64], output_dtype);
            let buffer = pool
                .allocate(layout.buffer_size_bytes())
                .expect("pool alloc for group");
            AtomStore::Owned(NumericTensor::from_parts(buffer, layout))
        });

        // For LiteralSpan that couldn't take the zero-copy path (non-flattenable),
        // copy element-by-element into the store.
        if let ScalarOp::LiteralSpan(ref span_tensor) = group.op {
            for i in 0..count {
                store.write_element(i, span_tensor.read_element(i));
            }

            group_stores[gi] = Some(store);
            for &pi in &producers[gi] {
                remaining[pi] -= 1;
                if remaining[pi] == 0 {
                    group_stores[pi] = None;
                }
            }
            continue;
        }

        {
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
                        let (val, val_dtype) =
                            lookup_atom_raw_dtype(src_id, graph, &group_stores, &input_stores);
                        let cast_raw = val_dtype.cast_raw(val, *compute_dtype);

                        let binop = match kind {
                            ReduceKind::Sum => ScalarBinOp::Add,
                            ReduceKind::Max => ScalarBinOp::Max,
                            ReduceKind::Min => ScalarBinOp::Min,
                            ReduceKind::Prod => ScalarBinOp::Mul,
                        };
                        acc_raw = eval_binop(&binop, acc_raw, cast_raw, *compute_dtype);
                    }
                    let result = compute_dtype.cast_raw(acc_raw, output_dtype);
                    write_atom(&mut store, i as usize, result, output_dtype);
                } else {
                    let result_raw = match &group.op {
                        ScalarOp::Literal(scalar) => {
                            let raw = scalar.view().read_raw();
                            scalar.dtype().cast_raw(raw, output_dtype)
                        }
                        ScalarOp::Identity => {
                            let src = group.inputs[0].resolve(ri);
                            let (val, val_dtype) =
                                lookup_atom_raw_dtype(src, graph, &group_stores, &input_stores);
                            val_dtype.cast_raw(val, output_dtype)
                        }
                        ScalarOp::Cast { saturating } => {
                            let src = group.inputs[0].resolve(ri);
                            let (val, val_dtype) =
                                lookup_atom_raw_dtype(src, graph, &group_stores, &input_stores);
                            let raw = val_dtype.cast_raw(val, output_dtype);
                            if *saturating {
                                output_dtype.saturate_inf(raw)
                            } else {
                                raw
                            }
                        }
                        ScalarOp::Binary { op, compute_dtype } => {
                            let a_src = group.inputs[0].resolve(ri);
                            let b_src = group.inputs[1].resolve(ri);
                            let (a_raw, a_dtype) =
                                lookup_atom_raw_dtype(a_src, graph, &group_stores, &input_stores);
                            let (b_raw, b_dtype) =
                                lookup_atom_raw_dtype(b_src, graph, &group_stores, &input_stores);
                            let a_cast = a_dtype.cast_raw(a_raw, *compute_dtype);
                            let b_cast = b_dtype.cast_raw(b_raw, *compute_dtype);
                            let result = eval_binop(op, a_cast, b_cast, *compute_dtype);
                            compute_dtype.cast_raw(result, output_dtype)
                        }
                        ScalarOp::Unary { op, compute_dtype } => {
                            let src = group.inputs[0].resolve(ri);
                            let (val, val_dtype) =
                                lookup_atom_raw_dtype(src, graph, &group_stores, &input_stores);
                            let x = val_dtype.cast_raw(val, *compute_dtype);
                            let result = eval_unaryop(op, x, *compute_dtype);
                            compute_dtype.cast_raw(result, output_dtype)
                        }
                        ScalarOp::Select => {
                            let cond_src = group.inputs[0].resolve(ri);
                            let cond_scalar =
                                lookup_atom_scalar(cond_src, graph, &group_stores, &input_stores);
                            let cond_raw = cond_scalar.view().read_raw();
                            let is_true = cond_scalar.dtype().decode_to_f64(cond_raw) != 0.0;
                            let chosen_src = if is_true {
                                group.inputs[1].resolve(ri)
                            } else {
                                group.inputs[2].resolve(ri)
                            };
                            let val =
                                lookup_atom_raw(chosen_src, graph, &group_stores, &input_stores);
                            let val_dtype =
                                lookup_atom_dtype(chosen_src, graph, &group_stores, &input_stores);
                            val_dtype.cast_raw(val, output_dtype)
                        }
                        ScalarOp::IndirectLoad { table_base } => {
                            let idx_src = group.inputs[0].resolve(ri);
                            let idx_scalar =
                                lookup_atom_scalar(idx_src, graph, &group_stores, &input_stores);
                            let index = idx_scalar
                                .dtype()
                                .decode_to_f64(idx_scalar.view().read_raw())
                                as u64;
                            let table_atom = AtomId(table_base.0 + index);
                            let val =
                                lookup_atom_raw(table_atom, graph, &group_stores, &input_stores);
                            let val_dtype =
                                lookup_atom_dtype(table_atom, graph, &group_stores, &input_stores);
                            val_dtype.cast_raw(val, output_dtype)
                        }
                        ScalarOp::LiteralSpan(tensor) => {
                            // Fallback for non-flattenable LiteralSpan.
                            let scalar = tensor.read_element(i as usize);
                            let raw = scalar.view().read_raw();
                            scalar.dtype().cast_raw(raw, output_dtype)
                        }
                        ScalarOp::Reduce { .. } | ScalarOp::OpaqueOutput { .. } => unreachable!(),
                    };
                    write_atom(&mut store, i as usize, result_raw, output_dtype);
                }
            }
        }

        // Put the store back.
        group_stores[gi] = Some(store);

        // Free spent producer buffers.
        for &pi in &producers[gi] {
            remaining[pi] -= 1;
            if remaining[pi] == 0 {
                group_stores[pi] = None;
            }
        }
    }

    // --- Step 4: Assemble output tensors using TAMI stride mapping ---
    //
    // For each output TAMI, allocate the correctly-shaped tensor and
    // populate it by reading atoms from group/input stores.
    // Contiguous TAMIs get a fast path (take group store + reshape).
    let mut result_tensors: Vec<NumericTensor<'p, DynRank, P>> = Vec::with_capacity(outputs.len());

    for tam in outputs {
        // Fast path: non-segmented, all atoms in a single group → take store + set layout.
        // Works for both contiguous (row-major strides) and strided (transpose) cases.
        if tam.segments.is_empty()
            && let Some(gi) = graph.find_group_idx(tam.base_id)
            && groups[gi].base_id == tam.base_id
            && groups[gi].count == tam.count
            && let Some(store) = group_stores[gi].take()
        {
            // Build a strided layout using the TAMI's strides (atom units → bits).
            let element_bits = tam.dtype.total_bits() as u64;
            let strides_bits: Vec<u64> = tam
                .known_strides
                .iter()
                .map(|&s| s * element_bits)
                .collect();
            let strided_layout = TensorLayout::<DynRank>::ElementStrided {
                shape: tam.known_dims.clone(),
                dtype: tam.dtype,
                strides: strides_bits,
                offset_bits: 0,
            };
            let tensor = match store {
                AtomStore::Owned(t) => t.into_layout(strided_layout),
                AtomStore::View(v) => v
                    .to_tensor(pool)
                    .map_err(PoolEvalError::Allocation)?
                    .into_layout(strided_layout),
            };
            result_tensors.push(tensor);
            continue;
        }

        // General path: iterate elements, use TAMI to map each to an atom,
        // read from group/input stores.
        let target_layout = TensorLayout::<DynRank>::row_major(tam.known_dims.clone(), tam.dtype);
        let buffer = pool
            .allocate(target_layout.buffer_size_bytes())
            .map_err(PoolEvalError::Allocation)?;
        let mut out_tensor = NumericTensor::from_parts(buffer, target_layout);
        for elem in 0..tam.count {
            let atom = tam.atom_id_for_element(elem);
            let scalar = lookup_atom_scalar(atom, graph, &group_stores, &input_stores);
            out_tensor.write_element(elem as usize, scalar.cast_to(tam.dtype));
        }
        result_tensors.push(out_tensor);
    }

    Ok(result_tensors)
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

/// Unified atom storage — either a borrowed view (zero-copy) or a
/// pool-allocated tensor (for computed values).
enum AtomStore<'a, 'p, P: Pool + 'p> {
    /// Borrowed view — zero-copy for input tensors and LiteralSpan groups.
    View(NumericTensorView<'a, DynRank>),
    /// Owned tensor — allocated from pool during group evaluation.
    Owned(NumericTensor<'p, DynRank, P>),
}

impl<'a, 'p, P: Pool + 'p> AtomStore<'a, 'p, P> {
    fn read_element(&self, index: usize) -> NumericScalar {
        match self {
            AtomStore::View(v) => v.read_element(index),
            AtomStore::Owned(t) => t.read_element(index),
        }
    }

    fn dtype(&self) -> NumericDType {
        match self {
            AtomStore::View(v) => v.dtype(),
            AtomStore::Owned(t) => t.dtype(),
        }
    }

    fn write_element(&mut self, index: usize, value: NumericScalar) {
        match self {
            AtomStore::Owned(t) => t.write_element(index, value),
            AtomStore::View(_) => panic!("cannot write to a borrowed view"),
        }
    }
}

// ---------------------------------------------------------------------------
// Atom access
// ---------------------------------------------------------------------------

/// Read an atom as a NumericScalar from group stores or input stores.
fn lookup_atom_scalar<P: Pool>(
    atom_id: AtomId,
    graph: &NanoGraph<'_, impl Pool>,
    group_stores: &[Option<AtomStore<'_, '_, P>>],
    input_stores: &[AtomStore<'_, '_, P>],
) -> NumericScalar {
    if let Some(gi) = graph.find_group_idx(atom_id) {
        let store = group_stores[gi]
            .as_ref()
            .unwrap_or_else(|| panic!("group {gi} buffer freed when reading atom {atom_id}"));
        let group = &graph.groups()[gi];
        let offset = (atom_id.0 - group.base_id.0) as usize;
        return store.read_element(offset);
    }
    if let Some((ti, offset)) = graph.find_input_idx(atom_id) {
        return input_stores[ti].read_element(offset as usize);
    }
    panic!("atom {atom_id} not found in any group or input");
}

/// Read an atom's raw bits from group stores or input stores.
fn lookup_atom_raw<P: Pool>(
    atom_id: AtomId,
    graph: &NanoGraph<'_, impl Pool>,
    group_stores: &[Option<AtomStore<'_, '_, P>>],
    input_stores: &[AtomStore<'_, '_, P>],
) -> u64 {
    let scalar = lookup_atom_scalar(atom_id, graph, group_stores, input_stores);
    scalar.view().read_raw()
}

/// Read an atom's raw bits and dtype in a single lookup (one binary search).
fn lookup_atom_raw_dtype<P: Pool>(
    atom_id: AtomId,
    graph: &NanoGraph<'_, impl Pool>,
    group_stores: &[Option<AtomStore<'_, '_, P>>],
    input_stores: &[AtomStore<'_, '_, P>],
) -> (u64, NumericDType) {
    if let Some(gi) = graph.find_group_idx(atom_id) {
        let store = group_stores[gi]
            .as_ref()
            .unwrap_or_else(|| panic!("group {gi} buffer freed when reading atom {atom_id}"));
        let group = &graph.groups()[gi];
        let offset = (atom_id.0 - group.base_id.0) as usize;
        let scalar = store.read_element(offset);
        return (scalar.view().read_raw(), scalar.dtype());
    }
    if let Some((ti, offset)) = graph.find_input_idx(atom_id) {
        let scalar = input_stores[ti].read_element(offset as usize);
        return (scalar.view().read_raw(), scalar.dtype());
    }
    panic!("atom {atom_id} not found in any group or input");
}

/// Get the dtype of an atom.
fn lookup_atom_dtype<P: Pool>(
    atom_id: AtomId,
    graph: &NanoGraph<'_, impl Pool>,
    group_stores: &[Option<AtomStore<'_, '_, P>>],
    input_stores: &[AtomStore<'_, '_, P>],
) -> NumericDType {
    if let Some(gi) = graph.find_group_idx(atom_id) {
        return group_stores[gi].as_ref().unwrap().dtype();
    }
    if let Some((ti, _)) = graph.find_input_idx(atom_id) {
        return input_stores[ti].dtype();
    }
    panic!("atom {atom_id} not found");
}

/// Write a raw u64 value to a tensor at a flat index.
fn write_atom<P: Pool>(
    store: &mut AtomStore<'_, '_, P>,
    index: usize,
    raw: u64,
    dtype: NumericDType,
) {
    let mut bits = [0u8; 8];
    let nbytes = dtype.bytes_per_element();
    bits[..nbytes].copy_from_slice(&raw.to_le_bytes()[..nbytes]);
    store.write_element(index, NumericScalar { bits, dtype });
}

// ---------------------------------------------------------------------------
// Op dispatch
// ---------------------------------------------------------------------------

/// Apply a bitwise operation on raw integer bits, masked to the type's bit width.
fn bitwise_op_int(
    a: u64,
    b: u64,
    it: &crate::numeric_dtype::IntType,
    f: impl Fn(u64, u64) -> u64,
) -> u64 {
    let mask = if it.bits >= 64 {
        u64::MAX
    } else {
        (1u64 << it.bits) - 1
    };
    f(a & mask, b & mask) & mask
}
fn eval_binop(op: &ScalarBinOp, a: u64, b: u64, dtype: NumericDType) -> u64 {
    match dtype {
        NumericDType::Float(ft) => match op {
            ScalarBinOp::Add => crate::scalar_ops::add::float_add(a, b, &ft),
            ScalarBinOp::Sub => crate::scalar_ops::sub::float_sub(a, b, &ft),
            ScalarBinOp::Mul => crate::scalar_ops::mul::float_mul(a, b, &ft),
            ScalarBinOp::Div => crate::scalar_ops::div::float_div(a, b, &ft),
            ScalarBinOp::Max => crate::scalar_ops::max::float_max(a, b, &ft),
            ScalarBinOp::Min => crate::scalar_ops::min::float_min(a, b, &ft),
            ScalarBinOp::Mod | ScalarBinOp::IMod => crate::scalar_ops::modulo::float_mod(a, b, &ft),
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
            ScalarBinOp::BitwiseAnd
            | ScalarBinOp::BitwiseOr
            | ScalarBinOp::BitwiseXor
            | ScalarBinOp::BitShiftLeft
            | ScalarBinOp::BitShiftRight => {
                panic!("bitwise ops not supported on float type {ft:?}")
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
            ScalarBinOp::IMod => crate::scalar_ops::modulo::signed_imod(a, b, &it),
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
            ScalarBinOp::BitwiseAnd => bitwise_op_int(a, b, &it, |x, y| x & y),
            ScalarBinOp::BitwiseOr => bitwise_op_int(a, b, &it, |x, y| x | y),
            ScalarBinOp::BitwiseXor => bitwise_op_int(a, b, &it, |x, y| x ^ y),
            ScalarBinOp::BitShiftLeft => bitwise_op_int(a, b, &it, |x, y| x << y),
            ScalarBinOp::BitShiftRight => bitwise_op_int(a, b, &it, |x, y| x >> y),
        },
        NumericDType::UnsignedInt(it) => match op {
            ScalarBinOp::Add => crate::scalar_ops::add::unsigned_add_wrapping(a, b, &it),
            ScalarBinOp::Sub => crate::scalar_ops::sub::unsigned_sub_wrapping(a, b, &it),
            ScalarBinOp::Mul => crate::scalar_ops::mul::unsigned_mul_wrapping(a, b, &it),
            ScalarBinOp::Div => crate::scalar_ops::div::unsigned_div_wrapping(a, b, &it),
            ScalarBinOp::Max => crate::scalar_ops::max::unsigned_max(a, b, &it),
            ScalarBinOp::Min => crate::scalar_ops::min::unsigned_min(a, b, &it),
            ScalarBinOp::Mod | ScalarBinOp::IMod => {
                crate::scalar_ops::modulo::unsigned_mod(a, b, &it)
            }
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
            ScalarBinOp::BitwiseAnd => bitwise_op_int(a, b, &it, |x, y| x & y),
            ScalarBinOp::BitwiseOr => bitwise_op_int(a, b, &it, |x, y| x | y),
            ScalarBinOp::BitwiseXor => bitwise_op_int(a, b, &it, |x, y| x ^ y),
            ScalarBinOp::BitShiftLeft => bitwise_op_int(a, b, &it, |x, y| x << y),
            ScalarBinOp::BitShiftRight => bitwise_op_int(a, b, &it, |x, y| x >> y),
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
                ScalarBinOp::Xor | ScalarBinOp::BitwiseXor => {
                    bool_to_dtype_raw(crate::scalar_ops::logical::logical_xor(a, b), dtype)
                }
                ScalarBinOp::BitwiseAnd => {
                    bool_to_dtype_raw(crate::scalar_ops::logical::logical_and(a, b), dtype)
                }
                ScalarBinOp::BitwiseOr => {
                    bool_to_dtype_raw(crate::scalar_ops::logical::logical_or(a, b), dtype)
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
                let val = ft.decode_f64(x);
                ft.encode_f64(if val != 0.0 && !val.is_nan() {
                    0.0
                } else {
                    1.0
                })
            }
            ScalarUnaryOp::IsNan => {
                let val = ft.decode_f64(x);
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
fn bool_to_dtype_raw(val: u64, dtype: NumericDType) -> u64 {
    let f = if val != 0 { 1.0 } else { 0.0 };
    dtype.encode_from_f64(f)
}
