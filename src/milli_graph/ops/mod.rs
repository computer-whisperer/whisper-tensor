mod argmax;
mod argmin;
pub(crate) mod binary;
mod cast;
mod cast_like;
mod concat;
mod constant;
mod conv;
mod cumsum;
mod expand;
mod eye_like;
mod gather;
mod nonzero;
mod pad;

mod random_normal_like;
mod range;
mod reduce_max;
mod reduce_mean;
mod reduce_min;
mod reduce_prod;
mod reduce_sum;
mod reshape;
mod resize;
mod shape;
mod slice;
mod split;
mod squeeze;
mod sum_to;
mod topk;
mod transpose;
pub(crate) mod unary;
mod unsqueeze;
mod where_op;

pub use argmax::*;
pub use argmin::*;
pub use binary::*;
pub use cast::*;
pub use cast_like::*;
pub use concat::*;
pub use constant::*;
pub use conv::*;
pub use cumsum::*;
pub use expand::*;
pub use eye_like::*;
pub use gather::*;
pub use nonzero::*;
pub use pad::*;

pub use random_normal_like::*;
pub use range::*;
pub use reduce_max::*;
pub use reduce_mean::*;
pub use reduce_min::*;
pub use reduce_prod::*;
pub use reduce_sum::*;
pub use reshape::*;
pub use resize::*;
pub use shape::*;
pub use slice::*;
pub use split::*;
pub use squeeze::*;
pub use sum_to::*;
pub use topk::*;
pub use transpose::*;
pub use unary::*;
pub use unsqueeze::*;
pub use where_op::*;

use crate::pool::Pool;
use crate::backends::eval_backend::EvalBackend;
use crate::backends::ndarray_backend::NDArrayNumericTensor;
use crate::graph::{GlobalId, Node, NodeMetadata, NodeSlotEditError, SlotDirection};
use crate::milli_graph::MilliOpGraphError;
use crate::migration::numeric_tensor::NumericTensor;
use crate::scalar_info::ScalarInfoTyped;
use crate::symbolic_scalar::{SymbolicResolver, SymbolicScalarTyped};
use crate::tensor_info::{TensorInfo, TensorInfoTypedRanked};
use crate::tensor_rank::DynRank;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use typenum::P1;

pub(crate) fn remap(id: &mut GlobalId, map: &HashMap<GlobalId, GlobalId>) {
    if let Some(&new) = map.get(id) {
        *id = new;
    }
}

pub(crate) fn remap_opt(id: &mut Option<GlobalId>, map: &HashMap<GlobalId, GlobalId>) {
    if let Some(inner) = id {
        remap(inner, map);
    }
}

/// Prescribes the accumulation order for reduction operations (ReduceSum,
/// ReduceMean, ReduceProd, and MatMul's internal contraction).
///
/// Both milli-eval and nano-eval must produce bit-identical results for the
/// same mode.  When lowering from the symbolic graph, all reduce ops default
/// to `Sequential` — the simplest strategy and the one the nano scalar
/// evaluator naturally implements.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccumulationMode {
    /// Left-to-right sequential accumulation: `acc = init; for v in values { acc = op(acc, v); }`.
    /// Deterministic and portable — the reference semantics for correctness testing.
    #[default]
    Sequential,
    /// Pairwise (recursive halving) accumulation.
    ///
    /// Recursively splits the input in half and reduces each half, then combines.
    /// Base case: length 0 returns init, length 1 returns that element.
    /// This matches PyTorch's CPU reduction strategy and gives better numerical
    /// accuracy than sequential for floating-point addition.
    ///
    /// Precisely: for `values[0..n]`:
    /// - `n == 0` → `init`
    /// - `n == 1` → `values[0]`
    /// - otherwise → `op(pairwise(values[0..n/2]), pairwise(values[n/2..n]))`
    Pairwise,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MilliOpTensorIDOrLiteral {
    TensorID(GlobalId),
    Literal(NDArrayNumericTensor<DynRank>),
}

/// Configuration for milli-op evaluation.
#[derive(Debug, Clone, Default)]
pub struct MilliEvalConfig {
    /// When true, reductions and matmuls may use BLAS or other fast paths
    /// that don't guarantee a specific accumulation order. Results may
    /// differ from the op's `AccumulationMode` but will be faster.
    ///
    /// When false (default), all ops respect their specified `AccumulationMode`
    /// exactly, producing deterministic, bit-reproducible results.
    pub relaxed_accumulation: bool,
}

pub type EvalResult =
    Result<Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>, MilliOpGraphError>;
/// Try to constant-fold an op by lowering to nano-ops and evaluating via pool_eval.
///
/// Returns `Some(results)` if all inputs are concrete and lowering succeeds.
/// Returns `None` if any inputs are symbolic or lowering/eval fails.
///
/// `output_hints` provides dtype+shape info for the op's outputs (needed by
/// lower_to_nano to classify output dimensions). Callers compute this from
/// their symbolic inference path before calling constant_fold.
pub fn constant_fold<'p, P: Pool + 'p>(
    op: &(impl MilliOp + Sized),
    known_inputs: &HashMap<GlobalId, TensorInfo<'p, P>>,
    output_hints: &[(GlobalId, TensorInfo<'p, P>)],
    pool: &'p P,
) -> Option<Vec<(GlobalId, TensorInfo<'p, P>)>> {
    use crate::nano_graph::lower::{NanoLoweringContext, new_numeric_to_legacy};
    use crate::nano_graph::pattern::AtomRange;
    use crate::nano_graph::pool_eval;
    use crate::pool::SystemPool;

    type LowerTensorInfo = TensorInfo<'static, SystemPool>;
    static SYS_POOL: SystemPool = SystemPool;

    // 1. Check all inputs are concrete.
    let input_ids: Vec<GlobalId> = op.inputs().collect();
    for &id in &input_ids {
        known_inputs.get(&id)?.as_concrete()?;
    }

    // Output hints are required — without them the lowering can't classify
    // output dimensions. All ops should provide hints from their symbolic
    // inference path.
    if output_hints.is_empty() {
        return None;
    }

    // 2. Build LowerTensorInfo map with inputs + output hints.
    let mut sys_infos: HashMap<GlobalId, LowerTensorInfo> = HashMap::new();
    for &id in &input_ids {
        let concrete = known_inputs.get(&id)?.as_concrete()?;
        let legacy = new_numeric_to_legacy(concrete);
        sys_infos.insert(id, LowerTensorInfo::from_legacy(&legacy, &SYS_POOL));
    }
    for (id, hint) in output_hints {
        // Output hints are dtype+shape only (no concrete data).
        // Reconstruct as a symbolic TensorInfo for the lowering context.
        let dtype = hint.dtype();
        if let Some(rank) = hint.rank_if_known() {
            let dims: Vec<crate::scalar_info::ScalarInfoTyped<u64>> = (0..rank)
                .map(|i| match hint.dim_if_known(i) {
                    Some(d) => crate::scalar_info::ScalarInfoTyped::Numeric(d),
                    None => crate::scalar_info::ScalarInfoTyped::Symbolic(
                        crate::symbolic_scalar::SymbolicScalarTyped::new(
                            &mut crate::symbolic_scalar::SymbolicResolver::new(),
                        ),
                    ),
                })
                .collect();
            sys_infos.insert(*id, LowerTensorInfo::from_dtype_and_shape_scalars(dtype, &dims));
        } else {
            sys_infos.insert(
                *id,
                LowerTensorInfo::Minimal(crate::tensor_info::MinimalTensor::new(
                    crate::scalar_info::ScalarInfo::Numeric(
                        crate::numeric_scalar::NumericScalar::zero(dtype),
                    ),
                    crate::symbolic_scalar::SymbolicScalarTyped::new(
                        &mut crate::symbolic_scalar::SymbolicResolver::new(),
                    ),
                )),
            );
        }
    }

    // 3. Lower this single op with constants embedded as Literal nano-ops.
    let mut ctx = NanoLoweringContext::new(&sys_infos);
    for &id in &input_ids {
        ctx.register_constant(id, &sys_infos[&id]);
    }
    op.lower_to_nano(&mut ctx);

    // If the op wasn't lowered (unsupported or fell back to boundary), bail out.
    if !ctx.unsupported.is_empty() {
        return None;
    }
    let output_ids: Vec<GlobalId> = op.outputs().collect();
    for &out_id in &output_ids {
        if !ctx.tensor_map.contains_key(&out_id) {
            return None;
        }
    }

    // 4. Build output AtomRanges.
    let mut output_ranges: Vec<AtomRange> = Vec::new();
    for &out_id in &output_ids {
        let tam = ctx.tensor_map.get(&out_id)?;
        let mut seen = std::collections::HashSet::new();
        for i in 0..tam.count {
            let atom = tam.atom_id_for_element(i);
            if let Some(gi) = ctx.nano.find_group_idx(atom) {
                if seen.insert(gi) {
                    let g = &ctx.nano.groups()[gi];
                    output_ranges.push(AtomRange {
                        base: g.base_id,
                        count: g.count,
                        dtype: g.output_dtype,
                    });
                }
            }
            if let Some((ti, _)) = ctx.nano.find_input_idx(atom) {
                let it = &ctx.nano.input_tensors()[ti];
                let fake_gi = usize::MAX - ti;
                if seen.insert(fake_gi) {
                    output_ranges.push(AtomRange {
                        base: it.base_id,
                        count: it.count,
                        dtype: it.dtype,
                    });
                }
            }
        }
    }

    // 5. Run pool_eval (no external inputs — all data in Literal groups).
    let eval_results =
        pool_eval::pool_eval(&ctx.nano, &[], &output_ranges, pool).ok()?;

    // 6. Build result TensorInfos.
    //
    // Output atoms may span multiple eval result ranges (e.g. when Literal
    // coalescing splits input data into several groups, and the output is a
    // zero-cost view over those groups). For each atom we need to find which
    // result range it belongs to.
    let find_range = |atom: crate::nano_graph::pattern::AtomId| -> Option<(usize, usize)> {
        for (ri, range) in output_ranges.iter().enumerate() {
            if atom.0 >= range.base.0 && atom.0 < range.base.0 + range.count {
                return Some((ri, (atom.0 - range.base.0) as usize));
            }
        }
        None
    };

    let mut results = Vec::new();
    for &out_id in &output_ids {
        let tam = ctx.tensor_map.get(&out_id)?;
        let shape = tam.known_dims();

        // Determine dtype from the first atom's result.
        let first_atom = tam.atom_id_for_element(0);
        let (first_ri, _) = find_range(first_atom)?;
        let dtype = eval_results[first_ri].dtype();

        let layout = crate::numeric_tensor::TensorLayout::row_major(shape, dtype);
        let buf = pool.allocate(layout.buffer_size_bytes()).ok()?;
        let mut out_tensor = crate::numeric_tensor::NumericTensor::from_parts(buf, layout);
        let numel = out_tensor.numel();
        for i in 0..numel {
            let atom = tam.atom_id_for_element(i as u64);
            let (ri, offset) = find_range(atom)?;
            out_tensor.write_element(i, eval_results[ri].read_element(offset));
        }
        results.push((out_id, TensorInfo::from(out_tensor)));
    }

    Some(results)
}

/// Shared eval_new logic for all reduction ops (ReduceSum, ReduceMax, ReduceMin, etc.).
///
/// `init` is the identity element (zero for sum, -inf for max, +inf for min, etc.).
/// `acc` is the accumulation function applied per element.
/// `finalize` is applied to each output element after accumulation (e.g., divide by count for mean).
pub(crate) fn reduce_eval_new<'p, P2: Pool + 'p>(
    inputs: &[crate::numeric_tensor::NumericTensorView<'_, DynRank>],
    axes_input_idx: Option<usize>,
    keepdims: bool,
    noop_with_empty_axes: bool,
    init: crate::numeric_scalar::NumericScalar,
    acc: impl Fn(crate::numeric_scalar::NumericScalar, crate::numeric_scalar::NumericScalar) -> crate::numeric_scalar::NumericScalar,
    finalize: impl Fn(crate::numeric_scalar::NumericScalar, u64) -> crate::numeric_scalar::NumericScalar,
    pool: &'p P2,
) -> Result<Vec<crate::numeric_tensor::NumericTensor<'p, DynRank, P2>>, crate::nano_graph::pool_eval::PoolEvalError> {
    use crate::numeric_tensor::{NumericTensor, TensorLayout};

    let data = &inputs[0];
    let shape = data.shape();
    let rank = shape.len();
    let dtype = data.dtype();

    // Extract axes.
    let axes: Vec<usize> = if let Some(ax_idx) = axes_input_idx {
        if ax_idx < inputs.len() {
            let ax_view = &inputs[ax_idx];
            let raw: Vec<i64> = (0..ax_view.numel()).map(|i| ax_view.read_element(i).to_i64()).collect();
            if raw.is_empty() && noop_with_empty_axes {
                let layout = TensorLayout::<DynRank>::row_major(shape.clone(), dtype);
                let buf = pool.allocate(layout.buffer_size_bytes())
                    .map_err(crate::nano_graph::pool_eval::PoolEvalError::Allocation)?;
                let mut out = NumericTensor::from_parts(buf, layout);
                for i in 0..data.numel() { out.write_element(i, data.read_element(i)); }
                return Ok(vec![out]);
            }
            if raw.is_empty() {
                (0..rank).collect()
            } else {
                raw.iter().map(|&a| if a < 0 { (a + rank as i64) as usize } else { a as usize }).collect()
            }
        } else {
            (0..rank).collect()
        }
    } else {
        (0..rank).collect()
    };

    // Compute output shape.
    let mut out_shape = Vec::new();
    for (i, &dim) in shape.iter().enumerate() {
        if axes.contains(&i) {
            if keepdims { out_shape.push(1u64); }
        } else {
            out_shape.push(dim);
        }
    }
    if out_shape.is_empty() { out_shape.push(1); }

    // Compute reduce count (product of reduced dims) for finalize.
    let reduce_count: u64 = axes.iter().map(|&a| shape[a]).product();

    let out_numel: usize = out_shape.iter().product::<u64>() as usize;
    let layout = TensorLayout::<DynRank>::row_major(out_shape.clone(), dtype);
    let buf = pool.allocate(layout.buffer_size_bytes())
        .map_err(crate::nano_graph::pool_eval::PoolEvalError::Allocation)?;
    let mut out = NumericTensor::from_parts(buf, layout);

    // Initialize all output elements.
    for i in 0..out_numel {
        out.write_element(i, init);
    }

    // Strides for index decomposition.
    let in_strides = {
        let mut s = vec![1usize; rank];
        for i in (0..rank.saturating_sub(1)).rev() { s[i] = s[i + 1] * shape[i + 1] as usize; }
        s
    };
    let out_strides = {
        let mut s = vec![1usize; out_shape.len()];
        for i in (0..out_shape.len().saturating_sub(1)).rev() { s[i] = s[i + 1] * out_shape[i + 1] as usize; }
        s
    };

    // Accumulate.
    for flat_in in 0..data.numel() {
        let mut rem = flat_in;
        let mut out_flat = 0usize;
        let mut out_dim_idx = 0;
        for i in 0..rank {
            let idx = rem / in_strides[i];
            rem %= in_strides[i];
            if !axes.contains(&i) {
                out_flat += idx * out_strides[out_dim_idx];
                out_dim_idx += 1;
            } else if keepdims {
                out_dim_idx += 1;
            }
        }
        let val = data.read_element(flat_in);
        let cur = out.read_element(out_flat);
        out.write_element(out_flat, acc(cur, val));
    }

    // Finalize (e.g. divide by count for mean).
    for i in 0..out_numel {
        out.write_element(i, finalize(out.read_element(i), reduce_count));
    }

    Ok(vec![out])
}

/// Result of attempting to lower a milli op into nano ops.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LowerResult {
    /// Op was fully decomposed into nano scalar ops.
    Lowered,
    /// Op (or this specific configuration) cannot be decomposed.
    /// The caller should fall through to the opaque eval_new path.
    Unsupported,
}

pub trait MilliOp: Node<OpKind = String> {
    fn infer<'p, P: Pool + 'p>(
        &self,
        known_inputs: &HashMap<GlobalId, TensorInfo<'p, P>>,
        _symbolic_resolver: &mut SymbolicResolver,
        pool: &'p P,
    ) -> Result<Vec<(GlobalId, TensorInfo<'p, P>)>, MilliOpGraphError>
    where
        Self: Sized,
    {
        constant_fold(self, known_inputs, &[], pool).ok_or(MilliOpGraphError::UnableToInfer)
    }

    fn eval(
        &self,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
        config: &MilliEvalConfig,
        _backend: &mut EvalBackend,
    ) -> EvalResult;

    /// Generate backward ops for this milli op.
    /// `output_grads` maps each output tensor ID to its gradient tensor ID.
    /// New gradient ops are added directly to `graph`.
    /// Returns input_id → gradient_id for each differentiable input.
    fn backward(
        &self,
        _output_grads: &HashMap<GlobalId, GlobalId>,
        _graph: &mut crate::milli_graph::MilliOpGraph,
        _rng: &mut impl Rng,
    ) -> Option<HashMap<GlobalId, GlobalId>> {
        None // default: not differentiable
    }

    /// Evaluate this op on new pool-backed types.
    ///
    /// Used by the OpaqueOp system for ops that can't be decomposed into
    /// scalar nano-ops. The default returns `Unsupported` — ops must override
    /// this to be executable through pool_eval without nano decomposition.
    fn eval_new<'p, P2: Pool + 'p>(
        &self,
        _inputs: &[crate::numeric_tensor::NumericTensorView<'_, DynRank>],
        _pool: &'p P2,
    ) -> Result<Vec<crate::numeric_tensor::NumericTensor<'p, DynRank, P2>>, crate::nano_graph::pool_eval::PoolEvalError>
    where
        Self: Sized,
    {
        Err(crate::nano_graph::pool_eval::PoolEvalError::Unsupported(
            format!("{} does not implement eval_new", self.op_kind()),
        ))
    }

    /// Lower this op to nano-graph representation.
    ///
    /// Returns `Lowered` if the op was decomposed into nano ops.
    /// Returns `Unsupported` if this op/configuration can't be lowered —
    /// the caller will fall through to the opaque eval_new path.
    fn lower_to_nano(&self, _ctx: &mut crate::nano_graph::NanoLoweringContext) -> LowerResult
    where
        Self: Sized,
    {
        LowerResult::Unsupported
    }
}

#[allow(dead_code)]
fn infer_multidirectional_broadcasting_rank(
    shapes: &[TensorInfoTypedRanked<u64, P1>],
    symbolic_resolver: &mut SymbolicResolver,
) -> Result<ScalarInfoTyped<u32>, MilliOpGraphError> {
    let mut output_rank: Option<usize> = None;
    for shape in shapes {
        match shape {
            TensorInfoTypedRanked::Shaped(x) => {
                let this_rank = x.shape()[0] as usize;
                if let Some(o) = output_rank {
                    output_rank = Some(o.max(this_rank));
                } else {
                    output_rank = Some(this_rank)
                }
            }
            TensorInfoTypedRanked::Ranked(_x) => {
                output_rank = None;
                break;
            }
        }
    }
    match output_rank {
        Some(x) => Ok(ScalarInfoTyped::Numeric(x as u32)),
        None => Ok(ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(
            symbolic_resolver,
        ))),
    }
}

#[allow(dead_code)]
fn infer_multidirectional_broadcasting_shape(
    shapes: &[Vec<ScalarInfoTyped<u64>>],
    symbolic_resolver: &mut SymbolicResolver,
) -> Result<Vec<ScalarInfoTyped<u64>>, MilliOpGraphError> {
    if shapes.is_empty() {
        return Err(MilliOpGraphError::InvalidInput(
            "Cannot broadcast empty input".to_string(),
        ));
    }

    let output_rank = shapes.iter().map(|x| x.len()).max().unwrap();

    let mut output_shape = vec![];
    for i in 0..output_rank {
        let mut dim = ScalarInfoTyped::<u64>::Numeric(1);
        for shape in shapes {
            let rank = shape.len();
            let local_i = (i as i64 - output_rank as i64) + rank as i64;
            if local_i < 0 {
                // Infer dim of size 1, and pass
            } else {
                let local_dim = shape[local_i as usize].clone();
                match local_dim {
                    ScalarInfoTyped::Numeric(x) => {
                        if x == 1 {
                            // Do not modify the dimension, pass it through.
                        } else {
                            match dim {
                                ScalarInfoTyped::Numeric(y) => {
                                    if y == 1 || x == y {
                                        dim = ScalarInfoTyped::Numeric(y.max(x));
                                    } else {
                                        return Err(MilliOpGraphError::InvalidInput(
                                            "Cannot broadcast input shape".to_string(),
                                        ));
                                    }
                                }
                                _ => {
                                    // The only way this is valid is if the unknown dim matches the known one, so be optimistic here
                                    dim = local_dim.clone();
                                }
                            }
                        }
                    }
                    _ => {
                        // Incoming dim is unknown
                        match dim {
                            ScalarInfoTyped::Numeric(y) => {
                                if y == 1 {
                                    // Use the existing unknown dim
                                    dim = local_dim.clone();
                                } else {
                                    // The only way this is valid is if the unknown dim matches the known one, so be optimistic here
                                }
                            }
                            _ => {
                                // Two unknown dimensions
                                match dim.try_eq(&local_dim) {
                                    None => {
                                        // Must use new unknown dimension
                                        dim = ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(
                                            symbolic_resolver,
                                        ))
                                    }
                                    Some(is_same) => {
                                        if is_same {
                                            // Ok, use the unknown dim already in there
                                        } else {
                                            // Must use new unknown dimension
                                            dim = ScalarInfoTyped::Symbolic(
                                                SymbolicScalarTyped::new(symbolic_resolver),
                                            )
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        output_shape.push(dim);
    }
    Ok(output_shape)
}

/// Compute per-dim output shape for a reduce op when input shape and axes are (partially) known.
/// Returns None if we can't compute the shape (fall back to rank-only inference).
fn infer_reduce_output_shape<'p, P: Pool + 'p>(
    data_info: &crate::tensor_info::TensorInfo<'p, P>,
    axes_id: Option<GlobalId>,
    keepdims: bool,
    noop_with_empty_axes: bool,
    known_inputs: &HashMap<GlobalId, crate::tensor_info::TensorInfo<'p, P>>,
    symbolic_resolver: &mut SymbolicResolver,
) -> Option<Vec<ScalarInfoTyped<u64>>> {
    // Need input with known per-dim shape.
    let data_ranked = data_info.as_ranked()?;
    let data_shape = data_ranked.shape();
    let rank = data_shape.len();

    // Get concrete axes values.
    let axes: Vec<usize> = if let Some(ax_id) = axes_id {
        let ax_info = known_inputs.get(&ax_id)?;
        let vals = ax_info.to_i64_vec()?;
        vals.iter()
            .map(|&a| {
                if a < 0 {
                    (a + rank as i64) as usize
                } else {
                    a as usize
                }
            })
            .collect()
    } else {
        // No axes → reduce all.
        (0..rank).collect()
    };

    if axes.is_empty() && noop_with_empty_axes {
        return Some(data_shape.clone());
    }
    if axes.is_empty() {
        // Empty axes, not noop → reduce all.
        let axes: Vec<usize> = (0..rank).collect();
        return infer_reduce_dims(&data_shape, &axes, keepdims, symbolic_resolver);
    }

    infer_reduce_dims(&data_shape, &axes, keepdims, symbolic_resolver)
}

fn infer_reduce_dims(
    data_shape: &[ScalarInfoTyped<u64>],
    axes: &[usize],
    keepdims: bool,
    _symbolic_resolver: &mut SymbolicResolver,
) -> Option<Vec<ScalarInfoTyped<u64>>> {
    let mut out_dims = Vec::new();
    for (i, dim) in data_shape.iter().enumerate() {
        if axes.contains(&i) {
            if keepdims {
                out_dims.push(ScalarInfoTyped::Numeric(1));
            }
            // else: dim is removed
        } else {
            out_dims.push(dim.clone());
        }
    }
    Some(out_dims)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnyMilliOp {
    Constant(Constant),
    ConstantOfShape(ConstantOfShape),
    SimpleBinary(SimpleBinary),
    MatMul(MatMul),
    Pow(Pow),
    SimpleUnary(SimpleUnaryOp),
    ClampMin(ClampMin),
    NonZero(NonZero),
    CumSum(CumSum),
    Shape(Shape),
    Reshape(Reshape),
    Slice(Slice),
    ReduceSum(ReduceSum),
    ReduceMin(ReduceMin),
    ReduceMax(ReduceMax),
    ReduceProd(ReduceProd),
    ReduceMean(ReduceMean),
    Cast(Cast),
    CastLike(CastLike),
    Transpose(Transpose),
    Squeeze(Squeeze),
    Unsqueeze(Unsqueeze),
    Gather(Gather),
    GatherGrad(GatherGrad),
    Concat(Concat),
    Split(Split),
    Where(Where),
    Range(Range),
    Expand(Expand),
    EyeLike(EyeLike),
    SumTo(SumTo),
    ArgMax(ArgMax),
    ArgMin(ArgMin),
    Resize(Resize),
    Conv(Conv),
    ConvInputGrad(ConvInputGrad),
    ConvWeightGrad(ConvWeightGrad),
    ConvBiasGrad(ConvBiasGrad),
    Pad(Pad),

    TopK(TopK),
    RandomNormalLike(RandomNormalLike),
}

impl AnyMilliOp {
    pub fn lower_to_nano(&self, ctx: &mut crate::nano_graph::NanoLoweringContext) -> LowerResult {
        match self {
            AnyMilliOp::SimpleBinary(x) => x.lower_to_nano(ctx),
            AnyMilliOp::SimpleUnary(x) => x.lower_to_nano(ctx),
            AnyMilliOp::ClampMin(x) => x.lower_to_nano(ctx),
            AnyMilliOp::Cast(x) => x.lower_to_nano(ctx),
            AnyMilliOp::CastLike(x) => x.lower_to_nano(ctx),
            AnyMilliOp::Constant(x) => x.lower_to_nano(ctx),
            AnyMilliOp::ConstantOfShape(x) => x.lower_to_nano(ctx),
            AnyMilliOp::Shape(x) => x.lower_to_nano(ctx),
            AnyMilliOp::Reshape(x) => x.lower_to_nano(ctx),
            AnyMilliOp::Squeeze(x) => x.lower_to_nano(ctx),
            AnyMilliOp::Unsqueeze(x) => x.lower_to_nano(ctx),
            AnyMilliOp::Transpose(x) => x.lower_to_nano(ctx),
            AnyMilliOp::Expand(x) => x.lower_to_nano(ctx),
            AnyMilliOp::EyeLike(x) => x.lower_to_nano(ctx),
            AnyMilliOp::Pow(x) => x.lower_to_nano(ctx),
            AnyMilliOp::Where(x) => x.lower_to_nano(ctx),
            AnyMilliOp::Concat(x) => x.lower_to_nano(ctx),
            AnyMilliOp::Split(x) => x.lower_to_nano(ctx),
            AnyMilliOp::Slice(x) => x.lower_to_nano(ctx),
            AnyMilliOp::MatMul(x) => x.lower_to_nano(ctx),
            AnyMilliOp::ReduceSum(x) => x.lower_to_nano(ctx),
            AnyMilliOp::ReduceMax(x) => x.lower_to_nano(ctx),
            AnyMilliOp::ReduceMean(x) => x.lower_to_nano(ctx),
            AnyMilliOp::Gather(x) => x.lower_to_nano(ctx),
            AnyMilliOp::ReduceMin(x) => x.lower_to_nano(ctx),
            AnyMilliOp::ReduceProd(x) => x.lower_to_nano(ctx),
            AnyMilliOp::Conv(x) => x.lower_to_nano(ctx),
            AnyMilliOp::Pad(x) => x.lower_to_nano(ctx),
            // Everything else: no nano decomposition, fall through to opaque.
            _ => LowerResult::Unsupported,
        }
    }

    pub fn stored_label(&self) -> Option<String> {
        match self {
            AnyMilliOp::Constant(x) => x.label.clone(),
            AnyMilliOp::ConstantOfShape(x) => x.label.clone(),
            AnyMilliOp::SimpleBinary(x) => x.label.clone(),
            AnyMilliOp::MatMul(x) => x.label.clone(),
            AnyMilliOp::Pow(x) => x.label.clone(),
            AnyMilliOp::SimpleUnary(x) => x.label.clone(),
            AnyMilliOp::ClampMin(x) => x.label.clone(),
            AnyMilliOp::NonZero(x) => x.label.clone(),
            AnyMilliOp::CumSum(x) => x.label.clone(),
            AnyMilliOp::Shape(x) => x.label.clone(),
            AnyMilliOp::Reshape(x) => x.label.clone(),
            AnyMilliOp::Slice(x) => x.label.clone(),
            AnyMilliOp::ReduceSum(x) => x.label.clone(),
            AnyMilliOp::ReduceMin(x) => x.label.clone(),
            AnyMilliOp::ReduceMax(x) => x.label.clone(),
            AnyMilliOp::ReduceProd(x) => x.label.clone(),
            AnyMilliOp::ReduceMean(x) => x.label.clone(),
            AnyMilliOp::Cast(x) => x.label.clone(),
            AnyMilliOp::CastLike(x) => x.label.clone(),
            AnyMilliOp::Transpose(x) => x.label.clone(),
            AnyMilliOp::Squeeze(x) => x.label.clone(),
            AnyMilliOp::Unsqueeze(x) => x.label.clone(),
            AnyMilliOp::Gather(x) => x.label.clone(),
            AnyMilliOp::GatherGrad(x) => x.label.clone(),
            AnyMilliOp::Concat(x) => x.label.clone(),
            AnyMilliOp::Split(x) => x.label.clone(),
            AnyMilliOp::Where(x) => x.label.clone(),
            AnyMilliOp::Range(x) => x.label.clone(),
            AnyMilliOp::Expand(x) => x.label.clone(),
            AnyMilliOp::EyeLike(x) => x.label.clone(),
            AnyMilliOp::SumTo(x) => x.label.clone(),
            AnyMilliOp::ArgMax(x) => x.label.clone(),
            AnyMilliOp::ArgMin(x) => x.label.clone(),
            AnyMilliOp::Resize(x) => x.label.clone(),
            AnyMilliOp::Conv(x) => x.label.clone(),
            AnyMilliOp::ConvInputGrad(x) => x.label.clone(),
            AnyMilliOp::ConvWeightGrad(x) => x.label.clone(),
            AnyMilliOp::ConvBiasGrad(x) => x.label.clone(),
            AnyMilliOp::Pad(x) => x.label.clone(),

            AnyMilliOp::TopK(x) => x.label.clone(),
            AnyMilliOp::RandomNormalLike(x) => x.label.clone(),
        }
    }

    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl Rng) {
        match self {
            AnyMilliOp::Constant(x) => x.remap_tensors(map, rng),
            AnyMilliOp::ConstantOfShape(x) => x.remap_tensors(map, rng),
            AnyMilliOp::SimpleBinary(x) => x.remap_tensors(map, rng),
            AnyMilliOp::MatMul(x) => x.remap_tensors(map, rng),
            AnyMilliOp::Pow(x) => x.remap_tensors(map, rng),
            AnyMilliOp::SimpleUnary(x) => x.remap_tensors(map, rng),
            AnyMilliOp::ClampMin(x) => x.remap_tensors(map, rng),
            AnyMilliOp::NonZero(x) => x.remap_tensors(map, rng),
            AnyMilliOp::CumSum(x) => x.remap_tensors(map, rng),
            AnyMilliOp::Shape(x) => x.remap_tensors(map, rng),
            AnyMilliOp::Reshape(x) => x.remap_tensors(map, rng),
            AnyMilliOp::Slice(x) => x.remap_tensors(map, rng),
            AnyMilliOp::ReduceSum(x) => x.remap_tensors(map, rng),
            AnyMilliOp::ReduceMin(x) => x.remap_tensors(map, rng),
            AnyMilliOp::ReduceMax(x) => x.remap_tensors(map, rng),
            AnyMilliOp::ReduceProd(x) => x.remap_tensors(map, rng),
            AnyMilliOp::ReduceMean(x) => x.remap_tensors(map, rng),
            AnyMilliOp::Cast(x) => x.remap_tensors(map, rng),
            AnyMilliOp::CastLike(x) => x.remap_tensors(map, rng),
            AnyMilliOp::Transpose(x) => x.remap_tensors(map, rng),
            AnyMilliOp::Squeeze(x) => x.remap_tensors(map, rng),
            AnyMilliOp::Unsqueeze(x) => x.remap_tensors(map, rng),
            AnyMilliOp::Gather(x) => x.remap_tensors(map, rng),
            AnyMilliOp::GatherGrad(x) => x.remap_tensors(map, rng),
            AnyMilliOp::Concat(x) => x.remap_tensors(map, rng),
            AnyMilliOp::Split(x) => x.remap_tensors(map, rng),
            AnyMilliOp::Where(x) => x.remap_tensors(map, rng),
            AnyMilliOp::Range(x) => x.remap_tensors(map, rng),
            AnyMilliOp::Expand(x) => x.remap_tensors(map, rng),
            AnyMilliOp::EyeLike(x) => x.remap_tensors(map, rng),
            AnyMilliOp::SumTo(x) => x.remap_tensors(map, rng),
            AnyMilliOp::ArgMax(x) => x.remap_tensors(map, rng),
            AnyMilliOp::ArgMin(x) => x.remap_tensors(map, rng),
            AnyMilliOp::Resize(x) => x.remap_tensors(map, rng),
            AnyMilliOp::Conv(x) => x.remap_tensors(map, rng),
            AnyMilliOp::ConvInputGrad(x) => x.remap_tensors(map, rng),
            AnyMilliOp::ConvWeightGrad(x) => x.remap_tensors(map, rng),
            AnyMilliOp::ConvBiasGrad(x) => x.remap_tensors(map, rng),
            AnyMilliOp::Pad(x) => x.remap_tensors(map, rng),

            AnyMilliOp::TopK(x) => x.remap_tensors(map, rng),
            AnyMilliOp::RandomNormalLike(x) => x.remap_tensors(map, rng),
        }
    }
}

macro_rules! delegate {
    ($name:ident($($arg:ident: $ty:ty),*) -> $ret:ty) => {
        fn $name(&self, $($arg: $ty),*) -> $ret {
            match self {
                AnyMilliOp::Constant(x) => x.$name($($arg),*),
                AnyMilliOp::ConstantOfShape(x) => x.$name($($arg),*),
                AnyMilliOp::SimpleBinary(x) => x.$name($($arg),*),
                AnyMilliOp::MatMul(x) => x.$name($($arg),*),
                AnyMilliOp::Pow(x) => x.$name($($arg),*),
                AnyMilliOp::SimpleUnary(x) => x.$name($($arg),*),
                AnyMilliOp::ClampMin(x) => x.$name($($arg),*),
                AnyMilliOp::NonZero(x) => x.$name($($arg),*),
                AnyMilliOp::CumSum(x) => x.$name($($arg),*),
                AnyMilliOp::Shape(x) => x.$name($($arg),*),
                AnyMilliOp::Reshape(x) => x.$name($($arg),*),
                AnyMilliOp::Slice(x) => x.$name($($arg),*),
                AnyMilliOp::ReduceSum(x) => x.$name($($arg),*),
                AnyMilliOp::ReduceMin(x) => x.$name($($arg),*),
                AnyMilliOp::ReduceMax(x) => x.$name($($arg),*),
                AnyMilliOp::ReduceProd(x) => x.$name($($arg),*),
                AnyMilliOp::ReduceMean(x) => x.$name($($arg),*),
                AnyMilliOp::Cast(x) => x.$name($($arg),*),
                AnyMilliOp::CastLike(x) => x.$name($($arg),*),
                AnyMilliOp::Transpose(x) => x.$name($($arg),*),
                AnyMilliOp::Squeeze(x) => x.$name($($arg),*),
                AnyMilliOp::Unsqueeze(x) => x.$name($($arg),*),
                AnyMilliOp::Gather(x) => x.$name($($arg),*),
                AnyMilliOp::GatherGrad(x) => x.$name($($arg),*),
                AnyMilliOp::Concat(x) => x.$name($($arg),*),
                AnyMilliOp::Split(x) => x.$name($($arg),*),
                AnyMilliOp::Where(x) => x.$name($($arg),*),
                AnyMilliOp::Range(x) => x.$name($($arg),*),
                AnyMilliOp::Expand(x) => x.$name($($arg),*),
                AnyMilliOp::EyeLike(x) => x.$name($($arg),*),
                AnyMilliOp::SumTo(x) => x.$name($($arg),*),
                AnyMilliOp::ArgMax(x) => x.$name($($arg),*),
                AnyMilliOp::ArgMin(x) => x.$name($($arg),*),
                AnyMilliOp::Resize(x) => x.$name($($arg),*),
                AnyMilliOp::Conv(x) => x.$name($($arg),*),
                AnyMilliOp::ConvInputGrad(x) => x.$name($($arg),*),
                AnyMilliOp::ConvWeightGrad(x) => x.$name($($arg),*),
                AnyMilliOp::ConvBiasGrad(x) => x.$name($($arg),*),
                AnyMilliOp::Pad(x) => x.$name($($arg),*),

                AnyMilliOp::TopK(x) => x.$name($($arg),*),
                AnyMilliOp::RandomNormalLike(x) => x.$name($($arg),*),
            }
        }
    }
}

impl MilliOp for AnyMilliOp {
    delegate!(eval(
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
        config: &MilliEvalConfig,
        backend: &mut EvalBackend
    ) -> Result<
        Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>,
        MilliOpGraphError,
    > );

    fn infer<'p, P: Pool + 'p>(
        &self,
        known_inputs: &HashMap<GlobalId, TensorInfo<'p, P>>,
        symbolic_resolver: &mut SymbolicResolver,
        pool: &'p P,
    ) -> Result<Vec<(GlobalId, TensorInfo<'p, P>)>, MilliOpGraphError> {
        match self {
            AnyMilliOp::Constant(x) => x.infer(known_inputs, symbolic_resolver, pool),
            AnyMilliOp::ConstantOfShape(x) => x.infer(known_inputs, symbolic_resolver, pool),
            AnyMilliOp::SimpleBinary(x) => x.infer(known_inputs, symbolic_resolver, pool),
            AnyMilliOp::MatMul(x) => x.infer(known_inputs, symbolic_resolver, pool),
            AnyMilliOp::Pow(x) => x.infer(known_inputs, symbolic_resolver, pool),
            AnyMilliOp::SimpleUnary(x) => x.infer(known_inputs, symbolic_resolver, pool),
            AnyMilliOp::ClampMin(x) => x.infer(known_inputs, symbolic_resolver, pool),
            AnyMilliOp::NonZero(x) => x.infer(known_inputs, symbolic_resolver, pool),
            AnyMilliOp::CumSum(x) => x.infer(known_inputs, symbolic_resolver, pool),
            AnyMilliOp::Shape(x) => x.infer(known_inputs, symbolic_resolver, pool),
            AnyMilliOp::Reshape(x) => x.infer(known_inputs, symbolic_resolver, pool),
            AnyMilliOp::Slice(x) => x.infer(known_inputs, symbolic_resolver, pool),
            AnyMilliOp::ReduceSum(x) => x.infer(known_inputs, symbolic_resolver, pool),
            AnyMilliOp::ReduceMin(x) => x.infer(known_inputs, symbolic_resolver, pool),
            AnyMilliOp::ReduceMax(x) => x.infer(known_inputs, symbolic_resolver, pool),
            AnyMilliOp::ReduceProd(x) => x.infer(known_inputs, symbolic_resolver, pool),
            AnyMilliOp::ReduceMean(x) => x.infer(known_inputs, symbolic_resolver, pool),
            AnyMilliOp::Cast(x) => x.infer(known_inputs, symbolic_resolver, pool),
            AnyMilliOp::CastLike(x) => x.infer(known_inputs, symbolic_resolver, pool),
            AnyMilliOp::Transpose(x) => x.infer(known_inputs, symbolic_resolver, pool),
            AnyMilliOp::Squeeze(x) => x.infer(known_inputs, symbolic_resolver, pool),
            AnyMilliOp::Unsqueeze(x) => x.infer(known_inputs, symbolic_resolver, pool),
            AnyMilliOp::Gather(x) => x.infer(known_inputs, symbolic_resolver, pool),
            AnyMilliOp::GatherGrad(x) => x.infer(known_inputs, symbolic_resolver, pool),
            AnyMilliOp::Concat(x) => x.infer(known_inputs, symbolic_resolver, pool),
            AnyMilliOp::Split(x) => x.infer(known_inputs, symbolic_resolver, pool),
            AnyMilliOp::Where(x) => x.infer(known_inputs, symbolic_resolver, pool),
            AnyMilliOp::Range(x) => x.infer(known_inputs, symbolic_resolver, pool),
            AnyMilliOp::Expand(x) => x.infer(known_inputs, symbolic_resolver, pool),
            AnyMilliOp::EyeLike(x) => x.infer(known_inputs, symbolic_resolver, pool),
            AnyMilliOp::SumTo(x) => x.infer(known_inputs, symbolic_resolver, pool),
            AnyMilliOp::ArgMax(x) => x.infer(known_inputs, symbolic_resolver, pool),
            AnyMilliOp::ArgMin(x) => x.infer(known_inputs, symbolic_resolver, pool),
            AnyMilliOp::Resize(x) => x.infer(known_inputs, symbolic_resolver, pool),
            AnyMilliOp::Conv(x) => x.infer(known_inputs, symbolic_resolver, pool),
            AnyMilliOp::ConvInputGrad(x) => x.infer(known_inputs, symbolic_resolver, pool),
            AnyMilliOp::ConvWeightGrad(x) => x.infer(known_inputs, symbolic_resolver, pool),
            AnyMilliOp::ConvBiasGrad(x) => x.infer(known_inputs, symbolic_resolver, pool),
            AnyMilliOp::Pad(x) => x.infer(known_inputs, symbolic_resolver, pool),
            AnyMilliOp::TopK(x) => x.infer(known_inputs, symbolic_resolver, pool),
            AnyMilliOp::RandomNormalLike(x) => x.infer(known_inputs, symbolic_resolver, pool),
        }
    }

    fn backward(
        &self,
        output_grads: &HashMap<GlobalId, GlobalId>,
        graph: &mut crate::milli_graph::MilliOpGraph,
        rng: &mut impl Rng,
    ) -> Option<HashMap<GlobalId, GlobalId>> {
        match self {
            AnyMilliOp::Constant(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::ConstantOfShape(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::SimpleBinary(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::MatMul(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::Pow(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::SimpleUnary(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::ClampMin(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::NonZero(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::CumSum(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::Shape(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::Reshape(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::Slice(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::ReduceSum(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::ReduceMin(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::ReduceMax(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::ReduceProd(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::ReduceMean(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::Cast(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::CastLike(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::Transpose(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::Squeeze(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::Unsqueeze(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::Gather(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::GatherGrad(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::Concat(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::Split(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::Where(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::Range(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::Expand(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::EyeLike(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::SumTo(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::ArgMax(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::ArgMin(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::Resize(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::Conv(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::ConvInputGrad(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::ConvWeightGrad(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::ConvBiasGrad(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::Pad(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::RandomNormalLike(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::TopK(x) => x.backward(output_grads, graph, rng),
        }
    }

    fn lower_to_nano(&self, ctx: &mut crate::nano_graph::NanoLoweringContext) -> LowerResult {
        AnyMilliOp::lower_to_nano(self, ctx)
    }

    fn eval_new<'p, P2: Pool + 'p>(
        &self,
        inputs: &[crate::numeric_tensor::NumericTensorView<'_, DynRank>],
        pool: &'p P2,
    ) -> Result<Vec<crate::numeric_tensor::NumericTensor<'p, DynRank, P2>>, crate::nano_graph::pool_eval::PoolEvalError> {
        match self {
            AnyMilliOp::SimpleBinary(x) => x.eval_new(inputs, pool),
            AnyMilliOp::MatMul(x) => x.eval_new(inputs, pool),
            AnyMilliOp::Pow(x) => x.eval_new(inputs, pool),
            AnyMilliOp::SimpleUnary(x) => x.eval_new(inputs, pool),
            AnyMilliOp::ClampMin(x) => x.eval_new(inputs, pool),
            AnyMilliOp::NonZero(x) => x.eval_new(inputs, pool),
            AnyMilliOp::CumSum(x) => x.eval_new(inputs, pool),
            AnyMilliOp::Shape(x) => x.eval_new(inputs, pool),
            AnyMilliOp::Reshape(x) => x.eval_new(inputs, pool),
            AnyMilliOp::Slice(x) => x.eval_new(inputs, pool),
            AnyMilliOp::ReduceSum(x) => x.eval_new(inputs, pool),
            AnyMilliOp::ReduceMin(x) => x.eval_new(inputs, pool),
            AnyMilliOp::ReduceMax(x) => x.eval_new(inputs, pool),
            AnyMilliOp::ReduceProd(x) => x.eval_new(inputs, pool),
            AnyMilliOp::ReduceMean(x) => x.eval_new(inputs, pool),
            AnyMilliOp::Cast(x) => x.eval_new(inputs, pool),
            AnyMilliOp::CastLike(x) => x.eval_new(inputs, pool),
            AnyMilliOp::Transpose(x) => x.eval_new(inputs, pool),
            AnyMilliOp::Squeeze(x) => x.eval_new(inputs, pool),
            AnyMilliOp::Unsqueeze(x) => x.eval_new(inputs, pool),
            AnyMilliOp::Gather(x) => x.eval_new(inputs, pool),
            AnyMilliOp::GatherGrad(x) => x.eval_new(inputs, pool),
            AnyMilliOp::Concat(x) => x.eval_new(inputs, pool),
            AnyMilliOp::Split(x) => x.eval_new(inputs, pool),
            AnyMilliOp::Where(x) => x.eval_new(inputs, pool),
            AnyMilliOp::Range(x) => x.eval_new(inputs, pool),
            AnyMilliOp::Expand(x) => x.eval_new(inputs, pool),
            AnyMilliOp::EyeLike(x) => x.eval_new(inputs, pool),
            AnyMilliOp::SumTo(x) => x.eval_new(inputs, pool),
            AnyMilliOp::ArgMax(x) => x.eval_new(inputs, pool),
            AnyMilliOp::ArgMin(x) => x.eval_new(inputs, pool),
            AnyMilliOp::Resize(x) => x.eval_new(inputs, pool),
            AnyMilliOp::Conv(x) => x.eval_new(inputs, pool),
            AnyMilliOp::ConvInputGrad(x) => x.eval_new(inputs, pool),
            AnyMilliOp::ConvWeightGrad(x) => x.eval_new(inputs, pool),
            AnyMilliOp::ConvBiasGrad(x) => x.eval_new(inputs, pool),
            AnyMilliOp::Pad(x) => x.eval_new(inputs, pool),
            AnyMilliOp::Constant(x) => x.eval_new(inputs, pool),
            AnyMilliOp::ConstantOfShape(x) => x.eval_new(inputs, pool),
            AnyMilliOp::RandomNormalLike(x) => x.eval_new(inputs, pool),
            AnyMilliOp::TopK(x) => x.eval_new(inputs, pool),
        }
    }
}

impl Node for AnyMilliOp {
    type OpKind = String;
    delegate!(op_kind() -> String);
    delegate!(inputs() ->  Box<dyn Iterator<Item = GlobalId> + '_>);
    delegate!(outputs() -> Box<dyn Iterator<Item = GlobalId> + '_>);
    delegate!(global_id() -> GlobalId);
    fn input_slots(&self) -> Box<dyn Iterator<Item = Option<GlobalId>> + '_> {
        Box::new(self.inputs().map(Some))
    }
    fn output_slots(&self) -> Box<dyn Iterator<Item = Option<GlobalId>> + '_> {
        Box::new(self.outputs().map(Some))
    }
    fn set_input_slot(
        &mut self,
        slot_index: usize,
        link: Option<GlobalId>,
    ) -> Result<(), NodeSlotEditError> {
        let Some(new_link_id) = link else {
            return Err(NodeSlotEditError::missing_slot_kind(
                self.op_kind(),
                SlotDirection::Input,
                slot_index,
            ));
        };

        let op_kind = self.op_kind();
        let old_inputs = self.inputs().collect::<Vec<_>>();
        let old_outputs = self.outputs().collect::<Vec<_>>();
        if slot_index >= old_inputs.len() {
            return Err(NodeSlotEditError::invalid_slot_index(
                op_kind,
                SlotDirection::Input,
                slot_index,
                old_inputs.len(),
            ));
        }
        if old_inputs[slot_index] == new_link_id {
            return Ok(());
        }

        let mut new_inputs = old_inputs.clone();
        new_inputs[slot_index] = new_link_id;

        let mut remap = HashMap::<GlobalId, GlobalId>::new();
        for (old_id, new_id) in old_inputs.iter().zip(new_inputs.iter()) {
            if let Some(existing) = remap.insert(*old_id, *new_id)
                && existing != *new_id
            {
                return Err(NodeSlotEditError::unsupported(
                    self.op_kind(),
                    SlotDirection::Input,
                    slot_index,
                ));
            }
        }
        for old_id in &old_outputs {
            if let Some(existing) = remap.insert(*old_id, *old_id)
                && existing != *old_id
            {
                return Err(NodeSlotEditError::unsupported(
                    self.op_kind(),
                    SlotDirection::Input,
                    slot_index,
                ));
            }
        }
        remap.retain(|old_id, new_id| old_id != new_id);
        if remap.is_empty() {
            return Ok(());
        }

        let mut rng = rand::rng();
        self.remap_tensors(&remap, &mut rng);
        Ok(())
    }
    fn set_output_slot(
        &mut self,
        slot_index: usize,
        link: Option<GlobalId>,
    ) -> Result<(), NodeSlotEditError> {
        let Some(new_link_id) = link else {
            return Err(NodeSlotEditError::missing_slot_kind(
                self.op_kind(),
                SlotDirection::Output,
                slot_index,
            ));
        };

        let op_kind = self.op_kind();
        let old_inputs = self.inputs().collect::<Vec<_>>();
        let old_outputs = self.outputs().collect::<Vec<_>>();
        if slot_index >= old_outputs.len() {
            return Err(NodeSlotEditError::invalid_slot_index(
                op_kind,
                SlotDirection::Output,
                slot_index,
                old_outputs.len(),
            ));
        }
        if old_outputs[slot_index] == new_link_id {
            return Ok(());
        }

        let mut new_outputs = old_outputs.clone();
        new_outputs[slot_index] = new_link_id;

        let mut remap = HashMap::<GlobalId, GlobalId>::new();
        for old_id in &old_inputs {
            if let Some(existing) = remap.insert(*old_id, *old_id)
                && existing != *old_id
            {
                return Err(NodeSlotEditError::unsupported(
                    self.op_kind(),
                    SlotDirection::Output,
                    slot_index,
                ));
            }
        }
        for (old_id, new_id) in old_outputs.iter().zip(new_outputs.iter()) {
            if let Some(existing) = remap.insert(*old_id, *new_id)
                && existing != *new_id
            {
                return Err(NodeSlotEditError::unsupported(
                    self.op_kind(),
                    SlotDirection::Output,
                    slot_index,
                ));
            }
        }
        remap.retain(|old_id, new_id| old_id != new_id);
        if remap.is_empty() {
            return Ok(());
        }

        let mut rng = rand::rng();
        self.remap_tensors(&remap, &mut rng);
        Ok(())
    }
    fn label(&self) -> Option<String> {
        self.stored_label()
    }
}

impl NodeMetadata for AnyMilliOp {
    // MilliOp nodes currently don't expose parameters via introspection
    // This can be expanded later by adding parameters() to MilliOp trait
}
