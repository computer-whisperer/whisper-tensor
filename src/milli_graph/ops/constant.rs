use crate::pool::Pool;
use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::backends::ndarray_backend::NDArrayNumericTensor;
use crate::backends::ndarray_backend::conversions::NDArrayNumericTensorType;
use crate::graph::{GlobalId, Node};
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError};
use crate::migration::numeric_scalar::NumericScalar;
use crate::migration::numeric_tensor::NumericTensor;
use crate::numeric_scalar::NumericScalar as NewScalar;
use crate::symbolic_graph::SharedPoolTensor;

use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use typenum::P1;

type PoolTensor = crate::numeric_tensor::NumericTensor<'static, DynRank, crate::pool::SystemPool>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Constant {
    global_id: GlobalId,
    pub(crate) label: Option<String>,
    output: GlobalId,
    data: SharedPoolTensor,
}

impl Constant {
    #[allow(dead_code)]
    pub(crate) fn pool_data(&self) -> &PoolTensor {
        &self.data.0
    }

    /// Push a constant from a legacy NDArrayNumericTensor (bridges internally).
    pub fn push_new(
        graph: &mut MilliOpGraph,
        a: NDArrayNumericTensor<DynRank>,
        rng: &mut impl Rng,
    ) -> GlobalId {
        Self::push_new_with_label(graph, a, None, rng)
    }

    pub fn push_new_with_label(
        graph: &mut MilliOpGraph,
        a: NDArrayNumericTensor<DynRank>,
        label: Option<String>,
        rng: &mut impl Rng,
    ) -> GlobalId {
        // Bridge legacy → pool tensor.
        let pool_tensor = crate::symbolic_graph::tensor_proto_to_pool_tensor_from_ndarray(&a)
            .expect("bridge constant to pool tensor");
        Self::push_new_pool(graph, SharedPoolTensor(std::sync::Arc::new(pool_tensor)), label, rng)
    }

    /// Push a constant from a pool tensor directly.
    pub fn push_new_pool(
        graph: &mut MilliOpGraph,
        data: SharedPoolTensor,
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

    pub(crate) fn new_scalar<T>(graph: &mut MilliOpGraph, v: T, rng: &mut impl Rng) -> GlobalId
    where
        T: NDArrayNumericTensorType,
    {
        Self::new_scalar_with_label(graph, v, None, rng)
    }

    pub(crate) fn new_scalar_with_label<T>(
        graph: &mut MilliOpGraph,
        v: T,
        label: Option<String>,
        rng: &mut impl Rng,
    ) -> GlobalId
    where
        T: NDArrayNumericTensorType,
    {
        let data = NDArrayNumericTensor::<DynRank>::from_vec_shape(vec![v], &vec![1]).unwrap();
        Self::push_new_with_label(graph, data, label, rng)
    }
}

impl Constant {
    pub fn lower_to_nano(&self, ctx: &mut crate::nano_graph::NanoLoweringContext) -> crate::milli_graph::ops::LowerResult {
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
    ) -> Result<
        Vec<(GlobalId, crate::tensor_info::TensorInfo<'p, P>)>,
        MilliOpGraphError,
    > {
        use crate::tensor_info::TensorInfo;
        Ok(vec![(self.output, TensorInfo::from_view(&self.data.0.view(), pool))])
    }

    fn eval(
        &self,
        _inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
        _config: &super::MilliEvalConfig,
        _backend: &mut EvalBackend,
    ) -> Result<Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>, MilliOpGraphError>
    {
        // Bridge to legacy for old eval path.
        let legacy = crate::nano_graph::lower::new_numeric_to_legacy(&*self.data.0);
        Ok(Box::new([(self.output, legacy)].into_iter()))
    }

    fn eval_new<'p, P2: crate::pool::Pool + 'p>(
        &self,
        _inputs: &[crate::numeric_tensor::NumericTensorView<'_, crate::tensor_rank::DynRank>],
        pool: &'p P2,
    ) -> Result<Vec<crate::numeric_tensor::NumericTensor<'p, crate::tensor_rank::DynRank, P2>>, crate::nano_graph::pool_eval::PoolEvalError> {
        use crate::numeric_tensor::{NumericTensor, TensorLayout};
        use crate::tensor_rank::DynRank;

        let src = &*self.data.0;
        let ndt = src.dtype();
        let shape = src.shape().clone();

        let layout = TensorLayout::<DynRank>::row_major(shape, ndt);
        let buf_size = layout.buffer_size_bytes();
        let buf = pool.allocate(buf_size)
            .map_err(crate::nano_graph::pool_eval::PoolEvalError::Allocation)?;

        let mut out = NumericTensor::from_parts(buf, layout);
        for i in 0..src.numel() {
            out.write_element(i, src.read_element(i));
        }
        Ok(vec![out])
    }

    fn lower_to_nano(&self, ctx: &mut crate::nano_graph::NanoLoweringContext) -> crate::milli_graph::ops::LowerResult {
        Constant::lower_to_nano(self, ctx)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstantOfShape {
    global_id: GlobalId,
    pub(crate) label: Option<String>,
    output: GlobalId,
    value: NumericScalar,
    shape: GlobalId,
}

impl ConstantOfShape {
    pub fn push_new(
        graph: &mut MilliOpGraph,
        value: NumericScalar,
        shape: GlobalId,
        rng: &mut impl Rng,
    ) -> GlobalId {
        Self::push_new_with_label(graph, value, shape, None, rng)
    }

    pub fn push_new_with_label(
        graph: &mut MilliOpGraph,
        value: NumericScalar,
        shape: GlobalId,
        label: Option<String>,
        rng: &mut impl Rng,
    ) -> GlobalId {
        let node = Self {
            global_id: GlobalId::new(rng),
            label,
            output: graph.get_new_tensor_id(rng),
            value,
            shape,
        };
        graph.push_op(AnyMilliOp::ConstantOfShape(node))
    }
}

impl ConstantOfShape {
    pub fn lower_to_nano(&self, ctx: &mut crate::nano_graph::NanoLoweringContext) -> crate::milli_graph::ops::LowerResult {
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
    ) -> Result<
        Vec<(GlobalId, crate::tensor_info::TensorInfo<'p, P>)>,
        MilliOpGraphError,
    > {
        use crate::scalar_info::ScalarInfoTyped;
        use crate::tensor_info::TensorInfo;

        let shape_info = known_inputs
            .get(&self.shape)
            .ok_or(MilliOpGraphError::UnableToInfer)?;

        // If shape is concrete, produce the filled tensor directly.
        if let Some(shape_values) = shape_info.to_i64_vec() {
            let shape_usize = shape_values.iter().map(|x| *x as u64).collect::<Vec<_>>();
            let out: NumericTensor<DynRank> =
                NDArrayNumericTensor::<DynRank>::fill(self.value.clone(), &shape_usize)
                    .map_err(|_| MilliOpGraphError::UnableToInfer)?
                    .into();
            return Ok(vec![(self.output, TensorInfo::from_legacy(&out, pool))]);
        }

        // Shape-only inference: the shape tensor's VALUES are the output dims.
        // If shape tensor is Ranked with known dims, each dim value tells us
        // the rank of the output but not the actual dim sizes.
        // If shape tensor is Shaped (1D with known length), we at least know the output rank.
        let out_dtype = crate::numeric_dtype::NumericDType::from_legacy(self.value.dtype())
            .expect("unsupported ConstantOfShape value dtype");
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
                return Ok(vec![((self.output, out_info))]);
            }
        }

        Err(MilliOpGraphError::UnableToInfer)
    }

    fn eval(
        &self,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
        _config: &super::MilliEvalConfig,
        _backend: &mut EvalBackend,
    ) -> Result<Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>, MilliOpGraphError>
    {
        let shape: Vec<i64> = inputs[&self.shape].try_to_rank::<P1>()?.try_into()?;
        let shape_usize = shape.iter().map(|x| *x as u64).collect::<Vec<_>>();
        let out: NumericTensor<DynRank> =
            NDArrayNumericTensor::<DynRank>::fill(self.value.clone(), &shape_usize)?.into();
        Ok(Box::new([(self.output, out)].into_iter()))
    }

    fn eval_new<'p, P2: crate::pool::Pool + 'p>(
        &self,
        inputs: &[crate::numeric_tensor::NumericTensorView<'_, crate::tensor_rank::DynRank>],
        pool: &'p P2,
    ) -> Result<Vec<crate::numeric_tensor::NumericTensor<'p, crate::tensor_rank::DynRank, P2>>, crate::nano_graph::pool_eval::PoolEvalError> {
        use crate::numeric_tensor::{NumericTensor, TensorLayout};
        use crate::tensor_rank::DynRank;

        // Read shape from inputs[0]
        let shape_tensor = &inputs[0];
        let shape: Vec<u64> = (0..shape_tensor.numel())
            .map(|i| shape_tensor.read_element(i).to_i64() as u64)
            .collect();

        let legacy_dtype = self.value.dtype();
        let ndt = crate::numeric_dtype::NumericDType::from_legacy(legacy_dtype)
            .ok_or_else(|| crate::nano_graph::pool_eval::PoolEvalError::Unsupported(
                format!("ConstantOfShape: unsupported dtype {:?}", legacy_dtype),
            ))?;

        let fill_val = crate::numeric_scalar::NumericScalar::from_f64(self.value.to_f64()).cast_to(ndt);
        let numel: usize = shape.iter().product::<u64>() as usize;

        let layout = TensorLayout::<DynRank>::row_major(shape, ndt);
        let buf = pool.allocate(layout.buffer_size_bytes())
            .map_err(crate::nano_graph::pool_eval::PoolEvalError::Allocation)?;
        let mut out = NumericTensor::from_parts(buf, layout);
        for i in 0..numel {
            out.write_element(i, fill_val);
        }

        Ok(vec![out])
    }

    fn lower_to_nano(&self, ctx: &mut crate::nano_graph::NanoLoweringContext) -> crate::milli_graph::ops::LowerResult {
        ConstantOfShape::lower_to_nano(self, ctx)
    }
}
