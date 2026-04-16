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
use super::pattern::{AtomId, NanoGraph, SymDimMap};

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
    bindings: &HashMap<super::pattern::GraphConstantId, u64>,
    pool: &'p P,
) -> Result<Vec<NumericTensor<'p, DynRank, P>>, PoolEvalError> {
    // Resolve every graph constant into a dense index-addressed Vec so the
    // hot-path code below can use infallible `[]` access. Missing bindings
    // surface as `UnboundGraphConstant` — there is no fallback default
    // because `0` is a legal dim extent and we must not silently pretend
    // an unbound constant is zero.
    let gc_values: Vec<u64> = {
        let mut resolved = Vec::with_capacity(graph.graph_constants.len());
        for (i, info) in graph.graph_constants.iter().enumerate() {
            let id = super::pattern::GraphConstantId(i as u16);
            let Some(&v) = bindings.get(&id) else {
                return Err(PoolEvalError::UnboundGraphConstant {
                    id,
                    name: info.name.clone(),
                });
            };
            resolved.push(v);
        }
        resolved
    };
    let gc_values = gc_values.as_slice();

    let groups = graph.groups();
    let n = groups.len();
    let input_tensors = graph.input_tensors();

    // Build a lookup from input tensor index → sym_dims (from TAMIs).
    // This lets resolve_producer_sym determine sym_dims for input atoms.
    let mut input_sym_dims: Vec<Vec<super::pattern::GraphConstantId>> =
        (0..input_tensors.len()).map(|_| vec![]).collect();
    for &(tam, _) in inputs {
        let input_index = graph
            .find_input_idx_by_base(tam.base_id)
            .or_else(|| graph.find_input_idx(tam.base_id).map(|(ti, _)| ti));
        if let Some(ti) = input_index {
            input_sym_dims[ti] = tam.sym_dims();
        }
    }

    // --- Step 1: Populate input stores ---
    //
    // If the input has no symbolic dims, relayout to the TAMI's known_dims layout
    // (zero-copy when strides agree). If the input has symbolic dims, decompose
    // each element via dim_layout into (atom_idx, sym_point) and write into an
    // n-d atom store shaped [count, ext0, ext1, ...].
    let mut input_stores: Vec<Option<AtomStore<'_, 'p, P>>> =
        (0..input_tensors.len()).map(|_| None).collect();

    // Multiple main-graph groups may read overlapping sub-ranges of the
    // same input tensor, and partitioner_m records each access as a
    // separate `insert_input_tensor_at_allow_overlap` call. That produces
    // multiple graph `input_tensors` entries sharing a base_id or sitting
    // inside each other's range. The caller (PoolEvalSpan) in turn
    // supplies one input tuple per declared access, so we must match
    // each tuple to a *distinct* input_tensor index. Exact-match on
    // (base, count, dtype) wins first; then fall back to the first
    // unfilled slot whose range contains tam.base_id. Without this the
    // first matching slot absorbs every tuple and later slots get
    // zero-filled.
    for &(tam, view) in inputs {
        let input_index = input_tensors
            .iter()
            .enumerate()
            .find(|(ti, it)| {
                input_stores[*ti].is_none()
                    && it.base_id == tam.base_id
                    && it.count == tam.count
                    && it.dtype == tam.dtype
            })
            .map(|(ti, _)| ti)
            .or_else(|| {
                input_tensors
                    .iter()
                    .enumerate()
                    .find(|(ti, it)| {
                        input_stores[*ti].is_none()
                            && it.base_id.0 <= tam.base_id.0
                            && tam.base_id.0 < it.base_id.0 + it.count
                    })
                    .map(|(ti, _)| ti)
            });
        if let Some(ti) = input_index {
            let sym_dims_v = tam.sym_dims();
            if sym_dims_v.is_empty() {
                // No symbolic dims — relayout to TAMI's known_dims shape.
                let element_bits = tam.dtype.total_bits() as u64;
                let known_strides_v = tam.known_strides();
                let strides_bits: Vec<u64> =
                    known_strides_v.iter().map(|&s| s * element_bits).collect();
                let target = TensorLayout::<DynRank>::ElementStrided {
                    shape: tam.known_dims(),
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
            } else {
                // Symbolic dims present — decompose input tensor elements
                // into (atom_idx, sym_point) via dims and populate
                // an n-d atom store.
                let sym_extents: Vec<u64> = sym_dims_v
                    .iter()
                    .map(|gc| gc_values[gc.0 as usize])
                    .collect();
                let sym_prod: u64 = sym_extents.iter().product::<u64>().max(1);
                let _total = tam.count * sym_prod;

                let mut store_shape = Vec::with_capacity(1 + sym_extents.len());
                store_shape.push(tam.count);
                store_shape.extend_from_slice(&sym_extents);
                let layout = TensorLayout::<DynRank>::row_major(store_shape, tam.dtype);
                let buffer = pool
                    .allocate(layout.buffer_size_bytes())
                    .expect("pool alloc for sym input");
                let mut store = NumericTensor::from_parts(buffer, layout);

                // Build the full shape from dims: Known → size, Sym → extent.
                let full_shape: Vec<u64> = tam
                    .dims
                    .iter()
                    .map(|dk| match dk {
                        super::lower::DimKind::Known { size, .. } => *size,
                        super::lower::DimKind::Sym { axis, .. } => sym_extents[*axis],
                    })
                    .collect();

                // Iterate over all elements using the full shape.
                let full_numel: u64 = full_shape.iter().product();
                for flat_elem in 0..full_numel {
                    // Decompose flat_elem into per-dim coordinates.
                    let mut remaining = flat_elem;
                    let mut coords = vec![0u64; full_shape.len()];
                    for d in (0..full_shape.len()).rev() {
                        if d == 0 {
                            coords[0] = remaining;
                        } else {
                            coords[d] = remaining % full_shape[d];
                            remaining /= full_shape[d];
                        }
                    }

                    // Split into atom_idx (from known dims) and sym_point (from sym dims).
                    let mut atom_idx = 0u64;
                    let mut atom_stride = 1u64;
                    let mut sym_flat = 0u64;
                    let mut sym_stride = 1u64;

                    // Build atom_idx from known dims (row-major) and sym_flat from sym dims.
                    for d in (0..full_shape.len()).rev() {
                        match &tam.dims[d] {
                            super::lower::DimKind::Known { .. } => {
                                atom_idx += coords[d] * atom_stride;
                                atom_stride *= full_shape[d];
                            }
                            super::lower::DimKind::Sym { .. } => {
                                sym_flat += coords[d] * sym_stride;
                                sym_stride *= full_shape[d];
                            }
                        }
                    }

                    let write_idx = (atom_idx * sym_prod + sym_flat) as usize;
                    let scalar = view.read_element(flat_elem as usize);
                    store.write_element(write_idx, scalar.cast_to(tam.dtype));
                }

                input_stores[ti] = Some(AtomStore::Owned(store));
            }
        }
    }

    // Fill any remaining input stores that weren't provided by the caller.
    let n_unfilled: usize = input_stores.iter().filter(|s| s.is_none()).count();
    if n_unfilled > 0 {
        let unfilled_atoms: u64 = input_stores
            .iter()
            .enumerate()
            .filter(|(_, s)| s.is_none())
            .map(|(ti, _)| input_tensors[ti].count)
            .sum();
        let detail: Vec<String> = input_stores
            .iter()
            .enumerate()
            .filter(|(_, s)| s.is_none())
            .map(|(ti, _)| {
                let it = &input_tensors[ti];
                format!("ti={ti} base={} count={}", it.base_id.0, it.count)
            })
            .collect();
        eprintln!(
            "[pool_eval] WARNING: {n_unfilled} of {} input tensors unfilled ({unfilled_atoms} atoms) — zero-filling [{}]",
            input_tensors.len(),
            detail.join(", "),
        );
    }
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
        if remaining[gi] == 0 && !matches!(&group.op, ScalarOp::Literal(_) | ScalarOp::GcLiteral(_))
        {
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
                // Resolve symbolic dims via gc_values to build full-shaped inputs.
                let mut op_input_tensors = Vec::with_capacity(opaque_op.inputs.len());
                for inp in &opaque_op.inputs {
                    let full_shape = inp.full_shape(gc_values);
                    // If any dim resolved to 0 (unpopulated gc_value), the input
                    // has no elements — create an empty tensor.
                    let any_zero = full_shape.contains(&0);
                    let full_numel: u64 = if any_zero {
                        0
                    } else {
                        full_shape.iter().product::<u64>().max(1)
                    };

                    // The input's dim_layout may reorder sym dims relative to
                    // the producing group's storage (e.g. after transpose),
                    // so resolve strides against the producer's sym_dims and
                    // index via each DimKind::Sym.axis.
                    let producer_sym_dims: Vec<super::pattern::GraphConstantId> =
                        if let Some(gi2) = graph.find_group_idx(inp.base) {
                            graph.groups()[gi2].sym_dims.clone()
                        } else if let Some((ti, _)) = graph.find_input_idx(inp.base) {
                            input_sym_dims[ti].clone()
                        } else {
                            inp.sym_dims()
                        };
                    let producer_sym_extents: Vec<u64> = producer_sym_dims
                        .iter()
                        .map(|gc| gc_values[gc.0 as usize])
                        .collect();
                    let producer_sym_prod: u64 =
                        producer_sym_extents.iter().product::<u64>().max(1);
                    let mut producer_sym_strides = vec![1u64; producer_sym_extents.len()];
                    for i in (0..producer_sym_extents.len().saturating_sub(1)).rev() {
                        producer_sym_strides[i] =
                            producer_sym_strides[i + 1] * producer_sym_extents[i + 1];
                    }

                    let inp_layout =
                        TensorLayout::<DynRank>::row_major(full_shape.clone(), inp.dtype);
                    let inp_buf = pool
                        .allocate(inp_layout.buffer_size_bytes())
                        .map_err(PoolEvalError::Allocation)?;
                    let mut inp_tensor: NumericTensor<'p, DynRank, P> =
                        NumericTensor::from_parts(inp_buf, inp_layout);

                    for flat_elem in 0..full_numel {
                        // Decompose flat_elem into per-dim coordinates and
                        // split into (atom_idx, sym_flat). Sym_flat is in the
                        // producer's sym-dim storage order.
                        let mut remaining = flat_elem;
                        let mut atom_idx = 0u64;
                        let mut atom_stride = 1u64;
                        let mut sym_flat = 0u64;

                        for d in (0..full_shape.len()).rev() {
                            let coord = remaining % full_shape[d];
                            remaining /= full_shape[d];
                            match &inp.dims[d] {
                                super::lower::DimKind::Known { .. } => {
                                    atom_idx += coord * atom_stride;
                                    atom_stride *= full_shape[d];
                                }
                                super::lower::DimKind::Sym { axis, .. } => {
                                    sym_flat += coord * producer_sym_strides[*axis];
                                }
                            }
                        }

                        let atom_id = AtomId(inp.base.0 + atom_idx);
                        let scalar = lookup_atom_scalar(
                            atom_id,
                            sym_flat,
                            producer_sym_prod,
                            graph,
                            &group_stores,
                            &input_stores,
                        );
                        inp_tensor.write_element(flat_elem as usize, scalar.cast_to(inp.dtype));
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

            // Copy result into the group store. For opaque ops with sym dims,
            // the result tensor has the full shape and we decompose each element
            // into (atom_idx, sym_flat) for the store layout.
            let count = group.count as usize;
            let output_dtype = group.output_dtype;
            let opaque_sym_prod = sym_product_for(&group.sym_dims, gc_values);
            let store = group_stores[gi].get_or_insert_with(|| {
                let total = count as u64 * opaque_sym_prod;
                let layout = TensorLayout::<DynRank>::row_major(vec![total], output_dtype);
                let buffer = pool
                    .allocate(layout.buffer_size_bytes())
                    .expect("pool alloc for opaque group");
                AtomStore::Owned(NumericTensor::from_parts(buffer, layout))
            });

            if group.sym_dims.is_empty() {
                // No sym dims — simple sequential copy.
                for i in 0..count {
                    let scalar = result_tensor.read_element(i);
                    store.write_element(i, scalar);
                }
            } else {
                // Has sym dims — decompose result elements into atom + sym positions.
                // The out_mapping's dim_layout may reorder sym dims relative
                // to this group's sym_dims storage order, so strides for sym
                // axes come from group.sym_dims and DimKind::Sym.axis picks
                // the right one.
                let opaque_op = &graph.opaque_ops()[*opaque_idx];
                let out_mapping = &opaque_op.outputs[*output_idx];
                let result_shape = result_tensor.shape();
                let full_numel = result_tensor.numel();

                let group_sym_extents: Vec<u64> = group
                    .sym_dims
                    .iter()
                    .map(|gc| gc_values[gc.0 as usize])
                    .collect();
                let mut group_sym_strides = vec![1u64; group_sym_extents.len()];
                for i in (0..group_sym_extents.len().saturating_sub(1)).rev() {
                    group_sym_strides[i] = group_sym_strides[i + 1] * group_sym_extents[i + 1];
                }

                for flat_elem in 0..full_numel {
                    let mut remaining = flat_elem as u64;
                    let mut atom_idx = 0u64;
                    let mut atom_stride = 1u64;
                    let mut sf = 0u64;

                    for d in (0..result_shape.len()).rev() {
                        let coord = remaining % result_shape[d];
                        remaining /= result_shape[d];
                        match &out_mapping.dims[d] {
                            super::lower::DimKind::Known { .. } => {
                                atom_idx += coord * atom_stride;
                                atom_stride *= result_shape[d];
                            }
                            super::lower::DimKind::Sym { axis, .. } => {
                                sf += coord * group_sym_strides[*axis];
                            }
                        }
                    }

                    let write_idx = (atom_idx * opaque_sym_prod + sf) as usize;
                    let scalar = result_tensor.read_element(flat_elem);
                    store.write_element(write_idx, scalar);
                }
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
        let sym_prod = sym_product_for(&group.sym_dims, gc_values);

        // Compute sym_dim extents for this group.
        let sym_extents: Vec<u64> = group
            .sym_dims
            .iter()
            .map(|gc| gc_values[gc.0 as usize])
            .collect();

        // Take the store out so we can mutate it while reading other stores.
        // Shape is [count, ext0, ext1, ...] — dim 0 is atom_id, remaining
        // dims are sym_dim axes in order.
        let mut store = group_stores[gi].take().unwrap_or_else(|| {
            let mut shape = Vec::with_capacity(1 + sym_extents.len());
            shape.push(count as u64);
            shape.extend_from_slice(&sym_extents);
            let layout = TensorLayout::<DynRank>::row_major(shape, output_dtype);
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

        // Helper: resolve an input's producer sym_flat given a consumer sym_flat.
        let resolve_producer_sym =
            |input: &super::pattern::GroupInput, consumer_sf: u64| -> (u64, u64) {
                if input.sym_dim_map.is_empty() {
                    return (0, 1);
                }
                let src_atom = input.input_ref.resolve(group.atom_offset);
                let producer_sym_dims: &[super::pattern::GraphConstantId] =
                    if let Some(pgi) = graph.find_group_idx(src_atom) {
                        &graph.groups()[pgi].sym_dims
                    } else if let Some((ti, _)) = graph.find_input_idx(src_atom) {
                        &input_sym_dims[ti]
                    } else {
                        return (0, 1);
                    };
                let p_sym_prod = sym_product_for(producer_sym_dims, gc_values);
                let p_sf = map_sym_flat(
                    consumer_sf,
                    &sym_extents,
                    &input.sym_dim_map,
                    producer_sym_dims,
                    gc_values,
                );
                (p_sf, p_sym_prod)
            };

        {
            for i in 0..group.count {
                let ri = i + group.atom_offset;

                for sym_flat in 0..sym_prod {
                    let write_idx = (i * sym_prod + sym_flat) as usize;

                    if let ScalarOp::Reduce {
                        kind,
                        reduce_count,
                        reduce_stride,
                        compute_dtype,
                    } = &group.op
                    {
                        // Known-dim reduce: sym_flat passes through to producer.
                        let (p_sf, p_sp) = if !group.inputs[0].sym_dim_map.is_empty() {
                            resolve_producer_sym(&group.inputs[0], sym_flat)
                        } else {
                            (0, 1)
                        };

                        let mut acc_raw = match kind {
                            ReduceKind::Sum => compute_dtype.encode_from_f64(0.0),
                            ReduceKind::Max => compute_dtype.encode_from_f64(f64::NEG_INFINITY),
                            ReduceKind::Min => compute_dtype.encode_from_f64(f64::INFINITY),
                            ReduceKind::Prod => compute_dtype.encode_from_f64(1.0),
                        };

                        let base_atom = group.inputs[0].input_ref.resolve(ri);
                        for k in 0..*reduce_count {
                            let src_id =
                                AtomId((base_atom.0 as i64 + k as i64 * reduce_stride) as u64);
                            let (val, val_dtype) = lookup_atom_raw_dtype(
                                src_id,
                                p_sf,
                                p_sp,
                                graph,
                                &group_stores,
                                &input_stores,
                            );
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
                        write_atom(&mut store, write_idx, result, output_dtype);
                    } else if let ScalarOp::SymReduce {
                        kind,
                        axis,
                        compute_dtype,
                    } = &group.op
                    {
                        // SymReduce: iterate over one axis of the input's sym_dims.
                        // The output group has one fewer sym_dim than the input.
                        let input = &group.inputs[0];
                        let src_atom = input.input_ref.resolve(ri);
                        let producer_sym_dims = if let Some(pgi) = graph.find_group_idx(src_atom) {
                            graph.groups()[pgi].sym_dims.clone()
                        } else if let Some((ti, _)) = graph.find_input_idx(src_atom) {
                            input_sym_dims[ti].clone()
                        } else {
                            vec![]
                        };
                        let reduce_extent = if *axis < producer_sym_dims.len() {
                            gc_values[producer_sym_dims[*axis].0 as usize]
                        } else {
                            1
                        };
                        let p_sp = sym_product_for(&producer_sym_dims, gc_values);

                        let mut acc_raw = match kind {
                            ReduceKind::Sum => compute_dtype.encode_from_f64(0.0),
                            ReduceKind::Max => compute_dtype.encode_from_f64(f64::NEG_INFINITY),
                            ReduceKind::Min => compute_dtype.encode_from_f64(f64::INFINITY),
                            ReduceKind::Prod => compute_dtype.encode_from_f64(1.0),
                        };

                        let binop = match kind {
                            ReduceKind::Sum => ScalarBinOp::Add,
                            ReduceKind::Max => ScalarBinOp::Max,
                            ReduceKind::Min => ScalarBinOp::Min,
                            ReduceKind::Prod => ScalarBinOp::Mul,
                        };

                        for k in 0..reduce_extent {
                            // Build producer sym_flat: same as consumer point but
                            // with the reduced axis set to k.
                            // Consumer has N-1 sym_dims, producer has N.
                            // The sym_dim_map tells us how consumer axes map to
                            // producer axes; the reduced axis is the one NOT covered.
                            let p_sf = map_sym_flat_with_reduce(
                                sym_flat,
                                &sym_extents,
                                &input.sym_dim_map,
                                &producer_sym_dims,
                                gc_values,
                                *axis,
                                k,
                            );
                            let (val, val_dtype) = lookup_atom_raw_dtype(
                                src_atom,
                                p_sf,
                                p_sp,
                                graph,
                                &group_stores,
                                &input_stores,
                            );
                            let cast_raw = val_dtype.cast_raw(val, *compute_dtype);
                            acc_raw = eval_binop(&binop, acc_raw, cast_raw, *compute_dtype);
                        }
                        let result = compute_dtype.cast_raw(acc_raw, output_dtype);
                        write_atom(&mut store, write_idx, result, output_dtype);
                    } else {
                        let result_raw = match &group.op {
                            ScalarOp::Literal(scalar) => {
                                let raw = scalar.view().read_raw();
                                scalar.dtype().cast_raw(raw, output_dtype)
                            }
                            ScalarOp::GcLiteral(gc) => {
                                // Resolve the GraphConstant at eval time and
                                // cast its u64 extent to the output dtype.
                                let value = gc_values[gc.0 as usize];
                                NumericDType::U64.cast_raw(value, output_dtype)
                            }
                            ScalarOp::Identity => {
                                let src = group.inputs[0].input_ref.resolve(ri);
                                let (p_sf, p_sp) = if !group.inputs[0].sym_dim_map.is_empty() {
                                    resolve_producer_sym(&group.inputs[0], sym_flat)
                                } else {
                                    (0, 1)
                                };
                                let (val, val_dtype) = lookup_atom_raw_dtype(
                                    src,
                                    p_sf,
                                    p_sp,
                                    graph,
                                    &group_stores,
                                    &input_stores,
                                );
                                val_dtype.cast_raw(val, output_dtype)
                            }
                            ScalarOp::Cast { saturating } => {
                                let src = group.inputs[0].input_ref.resolve(ri);
                                let (p_sf, p_sp) = if !group.inputs[0].sym_dim_map.is_empty() {
                                    resolve_producer_sym(&group.inputs[0], sym_flat)
                                } else {
                                    (0, 1)
                                };
                                let (val, val_dtype) = lookup_atom_raw_dtype(
                                    src,
                                    p_sf,
                                    p_sp,
                                    graph,
                                    &group_stores,
                                    &input_stores,
                                );
                                let raw = val_dtype.cast_raw(val, output_dtype);
                                if *saturating {
                                    output_dtype.saturate_inf(raw)
                                } else {
                                    raw
                                }
                            }
                            ScalarOp::Binary { op, compute_dtype } => {
                                let a_src = group.inputs[0].input_ref.resolve(ri);
                                let b_src = group.inputs[1].input_ref.resolve(ri);
                                let (a_sf, a_sp) = if !group.inputs[0].sym_dim_map.is_empty() {
                                    resolve_producer_sym(&group.inputs[0], sym_flat)
                                } else {
                                    (0, 1)
                                };
                                let (b_sf, b_sp) = if !group.inputs[1].sym_dim_map.is_empty() {
                                    resolve_producer_sym(&group.inputs[1], sym_flat)
                                } else {
                                    (0, 1)
                                };
                                let (a_raw, a_dtype) = lookup_atom_raw_dtype(
                                    a_src,
                                    a_sf,
                                    a_sp,
                                    graph,
                                    &group_stores,
                                    &input_stores,
                                );
                                let (b_raw, b_dtype) = lookup_atom_raw_dtype(
                                    b_src,
                                    b_sf,
                                    b_sp,
                                    graph,
                                    &group_stores,
                                    &input_stores,
                                );
                                let a_cast = a_dtype.cast_raw(a_raw, *compute_dtype);
                                let b_cast = b_dtype.cast_raw(b_raw, *compute_dtype);
                                let result = eval_binop(op, a_cast, b_cast, *compute_dtype);
                                compute_dtype.cast_raw(result, output_dtype)
                            }
                            ScalarOp::Unary { op, compute_dtype } => {
                                let src = group.inputs[0].input_ref.resolve(ri);
                                let (p_sf, p_sp) = if !group.inputs[0].sym_dim_map.is_empty() {
                                    resolve_producer_sym(&group.inputs[0], sym_flat)
                                } else {
                                    (0, 1)
                                };
                                let (val, val_dtype) = lookup_atom_raw_dtype(
                                    src,
                                    p_sf,
                                    p_sp,
                                    graph,
                                    &group_stores,
                                    &input_stores,
                                );
                                let x = val_dtype.cast_raw(val, *compute_dtype);
                                let result = eval_unaryop(op, x, *compute_dtype);
                                compute_dtype.cast_raw(result, output_dtype)
                            }
                            ScalarOp::Select => {
                                let cond_src = group.inputs[0].input_ref.resolve(ri);
                                let (c_sf, c_sp) = if !group.inputs[0].sym_dim_map.is_empty() {
                                    resolve_producer_sym(&group.inputs[0], sym_flat)
                                } else {
                                    (0, 1)
                                };
                                let cond_scalar = lookup_atom_scalar(
                                    cond_src,
                                    c_sf,
                                    c_sp,
                                    graph,
                                    &group_stores,
                                    &input_stores,
                                );
                                let cond_raw = cond_scalar.view().read_raw();
                                let is_true = cond_scalar.dtype().decode_to_f64(cond_raw) != 0.0;
                                let (chosen_inp_idx, chosen_src) = if is_true {
                                    (1, group.inputs[1].input_ref.resolve(ri))
                                } else {
                                    (2, group.inputs[2].input_ref.resolve(ri))
                                };
                                let (ch_sf, ch_sp) = if !group.inputs[chosen_inp_idx]
                                    .sym_dim_map
                                    .is_empty()
                                {
                                    resolve_producer_sym(&group.inputs[chosen_inp_idx], sym_flat)
                                } else {
                                    (0, 1)
                                };
                                let val = lookup_atom_raw(
                                    chosen_src,
                                    ch_sf,
                                    ch_sp,
                                    graph,
                                    &group_stores,
                                    &input_stores,
                                );
                                let val_dtype = lookup_atom_dtype(
                                    chosen_src,
                                    graph,
                                    &group_stores,
                                    &input_stores,
                                );
                                val_dtype.cast_raw(val, output_dtype)
                            }
                            ScalarOp::IndirectLoad { table_base, .. } => {
                                // IndirectLoad: index input may have sym_dims but
                                // the table itself is atom-addressed with no sym_dims.
                                let idx_src = group.inputs[0].input_ref.resolve(ri);
                                let (p_sf, p_sp) = if !group.inputs[0].sym_dim_map.is_empty() {
                                    resolve_producer_sym(&group.inputs[0], sym_flat)
                                } else {
                                    (0, 1)
                                };
                                let idx_scalar = lookup_atom_scalar(
                                    idx_src,
                                    p_sf,
                                    p_sp,
                                    graph,
                                    &group_stores,
                                    &input_stores,
                                );
                                let index = idx_scalar
                                    .dtype()
                                    .decode_to_f64(idx_scalar.view().read_raw())
                                    as u64;
                                let table_atom = AtomId(table_base.0 + index);
                                let val = lookup_atom_raw(
                                    table_atom,
                                    0,
                                    1,
                                    graph,
                                    &group_stores,
                                    &input_stores,
                                );
                                let val_dtype = lookup_atom_dtype(
                                    table_atom,
                                    graph,
                                    &group_stores,
                                    &input_stores,
                                );
                                val_dtype.cast_raw(val, output_dtype)
                            }
                            ScalarOp::LiteralSpan(tensor) => {
                                // LiteralSpan: no sym_dims, read by atom index.
                                let scalar = tensor.read_element(i as usize);
                                let raw = scalar.view().read_raw();
                                scalar.dtype().cast_raw(raw, output_dtype)
                            }
                            ScalarOp::Reduce { .. }
                            | ScalarOp::SymReduce { .. }
                            | ScalarOp::OpaqueOutput { .. } => unreachable!(),
                        };
                        write_atom(&mut store, write_idx, result_raw, output_dtype);
                    }
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
        // Fast path: non-segmented, no sym_dims, all atoms in a single group → take
        // store + set layout.  Sym_dim outputs go through the general path below.
        let sym_dims_v = tam.sym_dims();
        if sym_dims_v.is_empty()
            && tam.segments.is_empty()
            && let Some(gi) = graph.find_group_idx(tam.base_id)
            && groups[gi].base_id == tam.base_id
            && groups[gi].count == tam.count
            && let Some(store) = group_stores[gi].take()
        {
            // Build a strided layout using the TAMI's strides (atom units → bits).
            let element_bits = tam.dtype.total_bits() as u64;
            let known_strides_v = tam.known_strides();
            let strides_bits: Vec<u64> =
                known_strides_v.iter().map(|&s| s * element_bits).collect();
            let strided_layout = TensorLayout::<DynRank>::ElementStrided {
                shape: tam.known_dims(),
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

        if !sym_dims_v.is_empty() {
            // Output has symbolic dims — reconstruct full tensor shape from
            // dims and read each element from the n-d atom store.
            //
            // The TAM's dim_layout may reorder sym dims relative to the
            // producing group's storage layout (e.g. after transpose). To
            // compute the correct sym_flat index into the group store, we
            // resolve strides against the producer's sym_dims order (found
            // via the TAM's base atom), and use each DimKind::Sym's `axis`
            // as an index into that producer sym_dims list.
            let producer_sym_dims: Vec<super::pattern::GraphConstantId> =
                if let Some(gi) = graph.find_group_idx(tam.base_id) {
                    groups[gi].sym_dims.clone()
                } else if let Some((ti, _)) = graph.find_input_idx(tam.base_id) {
                    input_sym_dims[ti].clone()
                } else {
                    // Fall back to the TAM's own order — at worst, this
                    // matches the pre-fix behavior.
                    sym_dims_v.clone()
                };
            let producer_sym_extents: Vec<u64> = producer_sym_dims
                .iter()
                .map(|gc| gc_values[gc.0 as usize])
                .collect();
            let producer_sym_prod: u64 = producer_sym_extents.iter().product::<u64>().max(1);
            // Row-major strides over the producer's sym-dim axes: last axis
            // has stride 1, each earlier axis is that stride times the
            // following extent.
            let mut producer_sym_strides = vec![1u64; producer_sym_extents.len()];
            for i in (0..producer_sym_extents.len().saturating_sub(1)).rev() {
                producer_sym_strides[i] = producer_sym_strides[i + 1] * producer_sym_extents[i + 1];
            }

            // Full output shape: resolve each dim's extent directly from
            // its gc, so permuted dim_layouts produce correctly permuted
            // output shapes.
            let full_shape: Vec<u64> = tam
                .dims
                .iter()
                .map(|dk| match dk {
                    super::lower::DimKind::Known { size, .. } => *size,
                    super::lower::DimKind::Sym { gc, .. } => gc_values[gc.0 as usize],
                })
                .collect();
            let full_numel: u64 = full_shape.iter().product();

            let target_layout = TensorLayout::<DynRank>::row_major(full_shape.clone(), tam.dtype);
            let buffer = pool
                .allocate(target_layout.buffer_size_bytes())
                .map_err(PoolEvalError::Allocation)?;
            let mut out_tensor = NumericTensor::from_parts(buffer, target_layout);

            for flat_elem in 0..full_numel {
                // Decompose flat_elem into coordinates.
                let mut remaining = flat_elem;
                let mut coords = vec![0u64; full_shape.len()];
                for d in (0..full_shape.len()).rev() {
                    if d == 0 {
                        coords[0] = remaining;
                    } else {
                        coords[d] = remaining % full_shape[d];
                        remaining /= full_shape[d];
                    }
                }

                // Split into atom_idx (from known dims, row-major) and
                // sym_flat (indexing into the producer's sym storage).
                let mut atom_idx = 0u64;
                let mut atom_stride = 1u64;
                let mut sym_flat = 0u64;

                for d in (0..full_shape.len()).rev() {
                    match &tam.dims[d] {
                        super::lower::DimKind::Known { .. } => {
                            atom_idx += coords[d] * atom_stride;
                            atom_stride *= full_shape[d];
                        }
                        super::lower::DimKind::Sym { axis, .. } => {
                            sym_flat += coords[d] * producer_sym_strides[*axis];
                        }
                    }
                }

                let atom = tam.atom_id_for_element(atom_idx);
                let scalar = lookup_atom_scalar(
                    atom,
                    sym_flat,
                    producer_sym_prod,
                    graph,
                    &group_stores,
                    &input_stores,
                );
                out_tensor.write_element(flat_elem as usize, scalar.cast_to(tam.dtype));
            }
            result_tensors.push(out_tensor);
            continue;
        }

        // General path (no sym_dims): iterate elements, use TAMI to map each
        // to an atom, read from group/input stores.
        let target_layout = TensorLayout::<DynRank>::row_major(tam.known_dims(), tam.dtype);
        let buffer = pool
            .allocate(target_layout.buffer_size_bytes())
            .map_err(PoolEvalError::Allocation)?;
        let mut out_tensor = NumericTensor::from_parts(buffer, target_layout);
        for elem in 0..tam.count {
            let atom = tam.atom_id_for_element(elem);
            let scalar = lookup_atom_scalar(atom, 0, 1, graph, &group_stores, &input_stores);
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
    /// The caller didn't supply a runtime value for one of the graph's
    /// symbolic dim constants. Every `GraphConstantId` in
    /// `NanoGraph::graph_constants` must have a binding — there's no
    /// safe default (`0` is a legal extent, so we can't silently fill).
    /// `name` is the GC's debug name when available (e.g. "batch").
    #[error("graph constant {id:?} ({name:?}) has no runtime binding")]
    UnboundGraphConstant {
        id: super::pattern::GraphConstantId,
        name: Option<String>,
    },
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

/// Compute the flat element index for an atom with sym_dim coordinates.
///
/// `atom_id` identifies the atom within its group/input.
/// `sym_flat` is the row-major flat index within the sym_dim space
/// (0 for groups with no sym_dims).
/// `sym_product` is the product of all sym_dim extents (1 for no sym_dims).
fn elem_index(atom_id: AtomId, base_id: AtomId, sym_flat: u64, sym_product: u64) -> usize {
    let atom_offset = atom_id.0 - base_id.0;
    (atom_offset * sym_product + sym_flat) as usize
}

/// Compute sym_product for a group's sym_dims given gc_values.
fn sym_product_for(sym_dims: &[super::pattern::GraphConstantId], gc_values: &[u64]) -> u64 {
    sym_dims
        .iter()
        .map(|gc| gc_values[gc.0 as usize])
        .product::<u64>()
        .max(1)
}

/// Map consumer sym_dim coordinates to a producer's flat sym_dim index.
///
/// `consumer_sym_flat` is decomposed into coordinates using `consumer_extents`,
/// then mapped through `sym_dim_map` to the producer's space, and flattened
/// using `producer_extents`.
fn map_sym_flat(
    consumer_sym_flat: u64,
    consumer_extents: &[u64],
    sym_dim_map: &[SymDimMap],
    producer_sym_dims: &[super::pattern::GraphConstantId],
    gc_values: &[u64],
) -> u64 {
    if consumer_extents.is_empty() || producer_sym_dims.is_empty() {
        return 0;
    }
    // Decompose consumer_sym_flat into coordinates.
    let mut remaining = consumer_sym_flat;
    let mut consumer_coords = vec![0u64; consumer_extents.len()];
    for d in (0..consumer_extents.len()).rev() {
        if d == 0 {
            consumer_coords[0] = remaining;
        } else {
            consumer_coords[d] = remaining % consumer_extents[d];
            remaining /= consumer_extents[d];
        }
    }
    // Map to producer coordinates.
    let num_producer_axes = producer_sym_dims.len();
    let mut producer_coords = vec![0u64; num_producer_axes];
    for (j, mapping) in sym_dim_map.iter().enumerate() {
        if let SymDimMap::Identity(k) = mapping
            && *k < num_producer_axes
        {
            producer_coords[*k] = consumer_coords[j];
        }
    }
    // Flatten producer coordinates.
    let producer_extents: Vec<u64> = producer_sym_dims
        .iter()
        .map(|gc| gc_values[gc.0 as usize])
        .collect();
    let mut flat = 0u64;
    let mut stride = 1u64;
    for d in (0..num_producer_axes).rev() {
        flat += producer_coords[d] * stride;
        stride *= producer_extents[d];
    }
    flat
}

/// Like `map_sym_flat` but for SymReduce: inserts the reduce coordinate
/// at `reduce_axis` in the producer's sym_dim space.
///
/// The consumer has N-1 sym_dims (the reduced axis is absent).
/// The producer has N sym_dims. The sym_dim_map has N-1 entries
/// (parallel to consumer). We build producer_coords as usual from the
/// map, then set producer_coords[reduce_axis] = reduce_k.
fn map_sym_flat_with_reduce(
    consumer_sym_flat: u64,
    consumer_extents: &[u64],
    sym_dim_map: &[SymDimMap],
    producer_sym_dims: &[super::pattern::GraphConstantId],
    gc_values: &[u64],
    reduce_axis: usize,
    reduce_k: u64,
) -> u64 {
    let num_producer_axes = producer_sym_dims.len();
    let producer_extents: Vec<u64> = producer_sym_dims
        .iter()
        .map(|gc| gc_values[gc.0 as usize])
        .collect();

    // Decompose consumer_sym_flat into coordinates.
    let mut consumer_coords = vec![0u64; consumer_extents.len()];
    if !consumer_extents.is_empty() {
        let mut remaining = consumer_sym_flat;
        for d in (0..consumer_extents.len()).rev() {
            if d == 0 {
                consumer_coords[0] = remaining;
            } else {
                consumer_coords[d] = remaining % consumer_extents[d];
                remaining /= consumer_extents[d];
            }
        }
    }

    // Map to producer coordinates via sym_dim_map.
    let mut producer_coords = vec![0u64; num_producer_axes];
    for (j, mapping) in sym_dim_map.iter().enumerate() {
        if let SymDimMap::Identity(k) = mapping
            && *k < num_producer_axes
        {
            producer_coords[*k] = consumer_coords[j];
        }
    }
    // Insert the reduce coordinate.
    if reduce_axis < num_producer_axes {
        producer_coords[reduce_axis] = reduce_k;
    }

    // Flatten.
    let mut flat = 0u64;
    let mut stride = 1u64;
    for d in (0..num_producer_axes).rev() {
        flat += producer_coords[d] * stride;
        stride *= producer_extents[d];
    }
    flat
}

/// Read an atom as a NumericScalar from group stores or input stores.
fn lookup_atom_scalar<P: Pool>(
    atom_id: AtomId,
    sym_flat: u64,
    sym_product: u64,
    graph: &NanoGraph<'_, impl Pool>,
    group_stores: &[Option<AtomStore<'_, '_, P>>],
    input_stores: &[AtomStore<'_, '_, P>],
) -> NumericScalar {
    if let Some(gi) = graph.find_group_idx(atom_id) {
        let store = group_stores[gi]
            .as_ref()
            .unwrap_or_else(|| panic!("group {gi} buffer freed when reading atom {atom_id}"));
        let group = &graph.groups()[gi];
        let idx = elem_index(atom_id, group.base_id, sym_flat, sym_product);
        return store.read_element(idx);
    }
    if let Some((ti, offset)) = graph.find_input_idx(atom_id) {
        let idx = offset * sym_product + sym_flat;
        return input_stores[ti].read_element(idx as usize);
    }
    panic!("atom {atom_id} not found in any group or input");
}

/// Read an atom's raw bits from group stores or input stores.
fn lookup_atom_raw<P: Pool>(
    atom_id: AtomId,
    sym_flat: u64,
    sym_product: u64,
    graph: &NanoGraph<'_, impl Pool>,
    group_stores: &[Option<AtomStore<'_, '_, P>>],
    input_stores: &[AtomStore<'_, '_, P>],
) -> u64 {
    let scalar = lookup_atom_scalar(
        atom_id,
        sym_flat,
        sym_product,
        graph,
        group_stores,
        input_stores,
    );
    scalar.view().read_raw()
}

/// Read an atom's raw bits and dtype in a single lookup (one binary search).
fn lookup_atom_raw_dtype<P: Pool>(
    atom_id: AtomId,
    sym_flat: u64,
    sym_product: u64,
    graph: &NanoGraph<'_, impl Pool>,
    group_stores: &[Option<AtomStore<'_, '_, P>>],
    input_stores: &[AtomStore<'_, '_, P>],
) -> (u64, NumericDType) {
    if let Some(gi) = graph.find_group_idx(atom_id) {
        let store = group_stores[gi]
            .as_ref()
            .unwrap_or_else(|| panic!("group {gi} buffer freed when reading atom {atom_id}"));
        let group = &graph.groups()[gi];
        let idx = elem_index(atom_id, group.base_id, sym_flat, sym_product);
        let scalar = store.read_element(idx);
        return (scalar.view().read_raw(), scalar.dtype());
    }
    if let Some((ti, offset)) = graph.find_input_idx(atom_id) {
        let idx = offset * sym_product + sym_flat;
        let scalar = input_stores[ti].read_element(idx as usize);
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
            // Per docs/dtype_contract.md §5.4: Mod is C-style truncated
            // (sign matches dividend), IMod is Euclidean (sign matches
            // divisor) for both floats and integers.
            ScalarBinOp::Mod => crate::scalar_ops::modulo::float_mod(a, b, &ft),
            ScalarBinOp::IMod => crate::scalar_ops::modulo::float_imod(a, b, &ft),
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
            // Logical ops use dtype-aware truthiness (handles -0.0 and NaN
            // correctly). See docs/dtype_contract.md §5.7.
            ScalarBinOp::And => {
                bool_to_dtype_raw((is_truthy(a, dtype) && is_truthy(b, dtype)) as u64, dtype)
            }
            ScalarBinOp::Or => {
                bool_to_dtype_raw((is_truthy(a, dtype) || is_truthy(b, dtype)) as u64, dtype)
            }
            ScalarBinOp::Xor => {
                bool_to_dtype_raw((is_truthy(a, dtype) ^ is_truthy(b, dtype)) as u64, dtype)
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
            // Logical ops use dtype-aware truthiness. For ints this is
            // equivalent to `raw != 0` (no sign-bit ambiguity), but we
            // route through is_truthy for consistency with the float arm.
            ScalarBinOp::And => {
                bool_to_dtype_raw((is_truthy(a, dtype) && is_truthy(b, dtype)) as u64, dtype)
            }
            ScalarBinOp::Or => {
                bool_to_dtype_raw((is_truthy(a, dtype) || is_truthy(b, dtype)) as u64, dtype)
            }
            ScalarBinOp::Xor => {
                bool_to_dtype_raw((is_truthy(a, dtype) ^ is_truthy(b, dtype)) as u64, dtype)
            }
            ScalarBinOp::BitwiseAnd => bitwise_op_int(a, b, &it, |x, y| x & y),
            ScalarBinOp::BitwiseOr => bitwise_op_int(a, b, &it, |x, y| x | y),
            ScalarBinOp::BitwiseXor => bitwise_op_int(a, b, &it, |x, y| x ^ y),
            ScalarBinOp::BitShiftLeft => crate::scalar_ops::bitwise::shift_left(a, b, &it),
            // Signed BitShiftRight is *arithmetic* (sign-extends from the
            // left). Per docs/dtype_contract.md §5.8.
            ScalarBinOp::BitShiftRight => crate::scalar_ops::bitwise::signed_shift_right(a, b, &it),
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
            // Logical ops use dtype-aware truthiness (consistent with the
            // float and signed-int arms).
            ScalarBinOp::And => {
                bool_to_dtype_raw((is_truthy(a, dtype) && is_truthy(b, dtype)) as u64, dtype)
            }
            ScalarBinOp::Or => {
                bool_to_dtype_raw((is_truthy(a, dtype) || is_truthy(b, dtype)) as u64, dtype)
            }
            ScalarBinOp::Xor => {
                bool_to_dtype_raw((is_truthy(a, dtype) ^ is_truthy(b, dtype)) as u64, dtype)
            }
            ScalarBinOp::BitwiseAnd => bitwise_op_int(a, b, &it, |x, y| x & y),
            ScalarBinOp::BitwiseOr => bitwise_op_int(a, b, &it, |x, y| x | y),
            ScalarBinOp::BitwiseXor => bitwise_op_int(a, b, &it, |x, y| x ^ y),
            ScalarBinOp::BitShiftLeft => crate::scalar_ops::bitwise::shift_left(a, b, &it),
            // Unsigned BitShiftRight is *logical* (zero-fill).
            ScalarBinOp::BitShiftRight => {
                crate::scalar_ops::bitwise::unsigned_shift_right(a, b, &it)
            }
        },
        NumericDType::Bool => {
            // Comparisons, logical, and bitwise ops on Bool. For Bool the
            // raw bits ARE the truthiness (0 = false, 1 = true), so the
            // raw-bits and decoded-truthiness paths agree. Logical and
            // bitwise variants are equivalent for Bool.
            match op {
                ScalarBinOp::Equal => bool_to_dtype_raw(if a == b { 1 } else { 0 }, dtype),
                ScalarBinOp::And | ScalarBinOp::BitwiseAnd => {
                    bool_to_dtype_raw((is_truthy(a, dtype) && is_truthy(b, dtype)) as u64, dtype)
                }
                ScalarBinOp::Or | ScalarBinOp::BitwiseOr => {
                    bool_to_dtype_raw((is_truthy(a, dtype) || is_truthy(b, dtype)) as u64, dtype)
                }
                ScalarBinOp::Xor | ScalarBinOp::BitwiseXor => {
                    bool_to_dtype_raw((is_truthy(a, dtype) ^ is_truthy(b, dtype)) as u64, dtype)
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
                // Per docs/dtype_contract.md §5.7: NaN is truthy (NaN != 0
                // is true), -0.0 is falsy (-0.0 == 0.0). Both fall out of
                // `decode_f64(x) != 0.0` correctly.
                let truthy = ft.decode_f64(x) != 0.0;
                ft.encode_f64(if truthy { 0.0 } else { 1.0 })
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

/// Truthiness of `raw` interpreted as `dtype`, per
/// `docs/dtype_contract.md` §5.7. A float is truthy iff its decoded
/// real value is not exactly `0.0` — `-0.0` is falsy because
/// `-0.0 == 0.0`, NaN is truthy because `NaN != 0.0`. Integers and
/// Bool are truthy iff nonzero. Implemented via `decode_to_f64`
/// because that helper already handles every dtype correctly.
fn is_truthy(raw: u64, dtype: NumericDType) -> bool {
    dtype.decode_to_f64(raw) != 0.0
}
