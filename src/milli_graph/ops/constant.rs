use crate::DynRank;
use crate::graph::{GlobalId, Node};
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError};
use crate::numeric_dtype::NumericDType;
use crate::numeric_scalar::NumericScalar as NewScalar;
use crate::pool::Pool;
use crate::symbolic_graph::InlineConstantTensor;

use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::numeric_dtype::NumericPrimitive;

/// Trait for Rust types that can be stored as constant tensor elements.
/// Blanket-implemented for all `NumericPrimitive` types.
pub trait ConstantValue: Copy {
    fn to_scalar(self) -> NewScalar;
    fn dtype() -> NumericDType;
}

impl<T: NumericPrimitive> ConstantValue for T {
    fn to_scalar(self) -> NewScalar {
        NumericPrimitive::to_scalar(self)
    }
    fn dtype() -> NumericDType {
        T::NUMERIC_DTYPE
    }
}

/// Build an inline constant tensor from values + shape on the system pool.
pub fn build_inline_constant<T: ConstantValue>(
    values: &[T],
    shape: Vec<u64>,
) -> InlineConstantTensor {
    use crate::numeric_tensor::{NumericTensor, TensorLayout};
    use crate::pool::{Pool, SystemPool};

    let layout = TensorLayout::<DynRank>::row_major(shape, T::dtype());
    let buf = SystemPool
        .allocate(layout.buffer_size_bytes())
        .expect("system pool allocation for constant");
    let mut tensor = NumericTensor::from_parts(buf, layout);
    for (i, v) in values.iter().enumerate() {
        tensor.write_element(i, v.to_scalar());
    }
    InlineConstantTensor(std::sync::Arc::new(tensor))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Constant {
    global_id: GlobalId,
    pub(crate) label: Option<String>,
    output: GlobalId,
    data: InlineConstantTensor,
}

impl Constant {
    #[allow(dead_code)]
    pub(crate) fn pool_data(
        &self,
    ) -> &crate::numeric_tensor::NumericTensor<'static, DynRank, crate::pool::SystemPool> {
        self.data.inner()
    }

    /// Push a 1D constant tensor from a vec of typed values.
    pub fn from_vec<T: ConstantValue>(
        graph: &mut MilliOpGraph,
        values: Vec<T>,
        rng: &mut impl Rng,
    ) -> GlobalId {
        Self::from_vec_with_label(graph, values, None, rng)
    }

    /// Push a 1D constant tensor with a debug label.
    pub fn from_vec_with_label<T: ConstantValue>(
        graph: &mut MilliOpGraph,
        values: Vec<T>,
        label: Option<String>,
        rng: &mut impl Rng,
    ) -> GlobalId {
        let len = values.len() as u64;
        let data = build_inline_constant(&values, vec![len]);
        Self::push_new_pool(graph, data, label, rng)
    }

    /// Push a shaped constant tensor from values + shape.
    pub fn from_vec_shape<T: ConstantValue>(
        graph: &mut MilliOpGraph,
        values: Vec<T>,
        shape: Vec<u64>,
        rng: &mut impl Rng,
    ) -> GlobalId {
        let data = build_inline_constant(&values, shape);
        Self::push_new_pool(graph, data, None, rng)
    }

    /// Push a scalar (shape [1]) constant.
    pub fn new_scalar<T: ConstantValue>(
        graph: &mut MilliOpGraph,
        v: T,
        rng: &mut impl Rng,
    ) -> GlobalId {
        Self::new_scalar_with_label(graph, v, None, rng)
    }

    /// Push a scalar constant with a debug label.
    pub fn new_scalar_with_label<T: ConstantValue>(
        graph: &mut MilliOpGraph,
        v: T,
        label: Option<String>,
        rng: &mut impl Rng,
    ) -> GlobalId {
        let data = build_inline_constant(&[v], vec![1]);
        Self::push_new_pool(graph, data, label, rng)
    }

    /// Push a constant from an inline constant tensor directly.
    pub fn push_new_pool(
        graph: &mut MilliOpGraph,
        data: InlineConstantTensor,
        label: Option<String>,
        rng: &mut impl Rng,
    ) -> GlobalId {
        let node = Self {
            global_id: GlobalId::new(rng),
            label,
            output: graph.get_new_tensor_id(rng),
            data,
        };
        let out = node.output;
        graph.push_op(AnyMilliOp::Constant(node));
        out
    }
}

impl Constant {
    pub fn lower_to_nano(
        &self,
        ctx: &mut crate::nano_graph::NanoLoweringContext,
    ) -> crate::milli_graph::ops::LowerResult {
        let out_id = self.output;
        if let Some(info) = ctx.all_infos.get(&out_id) {
            ctx.register_constant(out_id, info);
        } else {
            ctx.register_opaque(out_id);
        }
        crate::milli_graph::ops::LowerResult::Lowered
    }

    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl Rng) {
        self.global_id = GlobalId::new(rng);
        super::remap(&mut self.output, map);
    }
}

impl Node for Constant {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "Constant".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::empty())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.output].into_iter())
    }
}

impl MilliOp for Constant {
    fn infer<'p, P: Pool + 'p>(
        &self,
        _known_inputs: &HashMap<GlobalId, crate::tensor_info::TensorInfo<'p, P>>,
        _symbolic_resolver: &mut crate::symbolic_scalar::SymbolicResolver,
        pool: &'p P,
    ) -> Result<Vec<(GlobalId, crate::tensor_info::TensorInfo<'p, P>)>, MilliOpGraphError> {
        use crate::tensor_info::TensorInfo;
        Ok(vec![(
            self.output,
            TensorInfo::from_view(&self.data.inner().view(), pool),
        )])
    }

    fn eval_new<'p, P2: crate::pool::Pool + 'p>(
        &self,
        _inputs: &[crate::numeric_tensor::NumericTensorView<'_, crate::tensor_rank::DynRank>],
        pool: &'p P2,
    ) -> Result<
        Vec<crate::numeric_tensor::NumericTensor<'p, crate::tensor_rank::DynRank, P2>>,
        crate::nano_graph::pool_eval::PoolEvalError,
    > {
        let out = self
            .data
            .inner()
            .view()
            .to_tensor(pool)
            .map_err(crate::nano_graph::pool_eval::PoolEvalError::Allocation)?;
        Ok(vec![out])
    }

    fn lower_to_nano(
        &self,
        ctx: &mut crate::nano_graph::NanoLoweringContext,
    ) -> crate::milli_graph::ops::LowerResult {
        Constant::lower_to_nano(self, ctx)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstantOfShape {
    global_id: GlobalId,
    pub(crate) label: Option<String>,
    output: GlobalId,
    value: NewScalar,
    shape: GlobalId,
}

impl ConstantOfShape {
    pub fn push_new(
        graph: &mut MilliOpGraph,
        value: NewScalar,
        shape: GlobalId,
        rng: &mut impl Rng,
    ) -> GlobalId {
        Self::push_new_with_label(graph, value, shape, None, rng)
    }

    pub fn push_new_with_label(
        graph: &mut MilliOpGraph,
        value: NewScalar,
        shape: GlobalId,
        label: Option<String>,
        rng: &mut impl Rng,
    ) -> GlobalId {
        let output = graph.get_new_tensor_id(rng);
        let node = Self {
            global_id: GlobalId::new(rng),
            label,
            output,
            value,
            shape,
        };
        graph.push_op(AnyMilliOp::ConstantOfShape(node));
        output
    }
}

impl ConstantOfShape {
    pub fn lower_to_nano(
        &self,
        ctx: &mut crate::nano_graph::NanoLoweringContext,
    ) -> crate::milli_graph::ops::LowerResult {
        let out_id = self.output;
        if let Some(info) = ctx.all_infos.get(&out_id) {
            ctx.register_constant(out_id, info);
        } else {
            ctx.register_opaque(out_id);
        }
        crate::milli_graph::ops::LowerResult::Lowered
    }

    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl Rng) {
        self.global_id = GlobalId::new(rng);
        super::remap(&mut self.output, map);
        super::remap(&mut self.shape, map);
    }
}

impl Node for ConstantOfShape {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "ConstantOfShape".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.shape].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.output].into_iter())
    }
}

impl MilliOp for ConstantOfShape {
    fn infer<'p, P: Pool + 'p>(
        &self,
        known_inputs: &HashMap<GlobalId, crate::tensor_info::TensorInfo<'p, P>>,
        _symbolic_resolver: &mut crate::symbolic_scalar::SymbolicResolver,
        pool: &'p P,
    ) -> Result<Vec<(GlobalId, crate::tensor_info::TensorInfo<'p, P>)>, MilliOpGraphError> {
        use crate::scalar_info::ScalarInfoTyped;
        use crate::tensor_info::TensorInfo;

        let shape_info = known_inputs
            .get(&self.shape)
            .ok_or(MilliOpGraphError::UnableToInfer)?;

        // If shape is concrete, produce the filled tensor directly.
        if let Some(shape_values) = shape_info.to_i64_vec() {
            let shape_u64 = shape_values.iter().map(|x| *x as u64).collect::<Vec<_>>();
            let numel = shape_u64.iter().product::<u64>() as usize;
            let ndt = self.value.dtype();
            let layout = crate::numeric_tensor::TensorLayout::<DynRank>::row_major(shape_u64, ndt);
            if let Ok(buf) = pool.allocate(layout.buffer_size_bytes()) {
                let mut tensor: crate::numeric_tensor::NumericTensor<'_, DynRank, P> =
                    crate::numeric_tensor::NumericTensor::from_parts(buf, layout);
                for i in 0..numel {
                    tensor.write_element(i, self.value);
                }
                return Ok(vec![(
                    self.output,
                    TensorInfo::from_view(&tensor.view(), pool),
                )]);
            }
        }

        let out_dtype = self.value.dtype();
        if let Some(rank) = shape_info.rank_if_known() {
            // shape is 1D — its first dim tells us the output rank.
            if rank == 1
                && let Some(out_rank) = shape_info.dim_if_known(0)
            {
                // We know the output rank. Build dims from shape tensor's values.
                // The shape tensor is [d0, d1, ...] — extract concrete dim values if available.
                // For a Shaped tensor, individual values may be known from the scalar_info.
                let out_rank = out_rank as usize;
                let mut out_dims: Vec<ScalarInfoTyped<u64>> = Vec::with_capacity(out_rank);
                for i in 0..out_rank {
                    // Try to read the i-th element of the shape tensor.
                    if let Some(crate::scalar_info::ScalarInfo::Numeric(n)) =
                        shape_info.get(&vec![i as u64], _symbolic_resolver)
                    {
                        // n is new NumericScalar — extract as u64 via to_i64
                        let v = n.to_i64() as u64;
                        out_dims.push(ScalarInfoTyped::Numeric(v));
                    } else {
                        out_dims.push(ScalarInfoTyped::Symbolic(
                            crate::symbolic_scalar::SymbolicScalarTyped::new(_symbolic_resolver),
                        ));
                    }
                }
                let out_info = TensorInfo::from_dtype_and_shape_scalars(out_dtype, &out_dims);
                return Ok(vec![(self.output, out_info)]);
            }
        }

        Err(MilliOpGraphError::UnableToInfer)
    }

    fn eval_new<'p, P2: crate::pool::Pool + 'p>(
        &self,
        inputs: &[crate::numeric_tensor::NumericTensorView<'_, crate::tensor_rank::DynRank>],
        pool: &'p P2,
    ) -> Result<
        Vec<crate::numeric_tensor::NumericTensor<'p, crate::tensor_rank::DynRank, P2>>,
        crate::nano_graph::pool_eval::PoolEvalError,
    > {
        use crate::numeric_tensor::NumericTensor;
        use crate::tensor_rank::DynRank;

        let shape_tensor = &inputs[0];
        let shape: Vec<u64> = (0..shape_tensor.numel())
            .map(|i| shape_tensor.read_element(i).to_i64() as u64)
            .collect();

        let fill_val = self.value;
        let out =
            NumericTensor::<DynRank, P2>::from_fn(shape, fill_val.dtype(), pool, |_| fill_val)
                .map_err(crate::nano_graph::pool_eval::PoolEvalError::Allocation)?;

        Ok(vec![out])
    }

    fn lower_to_nano(
        &self,
        ctx: &mut crate::nano_graph::NanoLoweringContext,
    ) -> crate::milli_graph::ops::LowerResult {
        ConstantOfShape::lower_to_nano(self, ctx)
    }
}
