use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::backends::ndarray_backend::NDArrayNumericTensor;
use crate::backends::ndarray_backend::conversions::NDArrayNumericTensorType;
use crate::graph::{GlobalId, Node};
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError};
use crate::numeric_scalar::NumericScalar;
use crate::numeric_tensor::NumericTensor;

use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use typenum::P1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Constant {
    global_id: GlobalId,
    pub(crate) label: Option<String>,
    output: GlobalId,
    data: NDArrayNumericTensor<DynRank>,
}

impl Constant {
    #[allow(dead_code)] // used by compiler (cranelift feature)
    pub(crate) fn data(&self) -> &NDArrayNumericTensor<DynRank> {
        &self.data
    }

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
        let node = Self {
            global_id: GlobalId::new(rng),
            label,
            output: graph.get_new_tensor_id(rng),
            data: a,
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
    fn eval(
        &self,
        _inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
        _backend: &mut EvalBackend,
    ) -> Result<Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>, MilliOpGraphError>
    {
        Ok(Box::new(
            [(self.output, self.data.clone().into())].into_iter(),
        ))
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
    fn infer(
        &self,
        known_inputs: &HashMap<GlobalId, crate::tensor_info::TensorInfo>,
        _symbolic_resolver: &mut crate::symbolic_scalar::SymbolicResolver,
        backend: &mut EvalBackend,
    ) -> Result<
        Box<dyn Iterator<Item = (GlobalId, crate::tensor_info::TensorInfo)>>,
        MilliOpGraphError,
    > {
        use crate::scalar_info::ScalarInfoTyped;
        use crate::tensor_info::TensorInfo;

        let shape_info = known_inputs
            .get(&self.shape)
            .ok_or(MilliOpGraphError::UnableToInfer)?;

        // If shape is concrete, fall back to eval.
        if shape_info.as_numeric().is_some() {
            let mut resolved = HashMap::new();
            resolved.insert(self.shape, shape_info.as_numeric().unwrap().clone());
            let collected: Vec<(GlobalId, TensorInfo)> = self
                .eval(&resolved, backend)?
                .map(|(a, b)| (a, TensorInfo::from(b)))
                .collect();
            return Ok(Box::new(collected.into_iter()));
        }

        // Shape-only inference: the shape tensor's VALUES are the output dims.
        // If shape tensor is Ranked with known dims, each dim value tells us
        // the rank of the output but not the actual dim sizes.
        // If shape tensor is Shaped (1D with known length), we at least know the output rank.
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
                        use crate::numeric_scalar::NumericScalar;
                        let v = match n {
                            NumericScalar::I64(x) => x as u64,
                            NumericScalar::I32(x) => x as u64,
                            NumericScalar::U64(x) => x,
                            NumericScalar::U32(x) => x as u64,
                            NumericScalar::F32(x) => x as u64,
                            NumericScalar::F64(x) => x as u64,
                            _ => {
                                out_dims.push(ScalarInfoTyped::Symbolic(
                                    crate::symbolic_scalar::SymbolicScalarTyped::new(
                                        _symbolic_resolver,
                                    ),
                                ));
                                continue;
                            }
                        };
                        out_dims.push(ScalarInfoTyped::Numeric(v));
                    } else {
                        out_dims.push(ScalarInfoTyped::Symbolic(
                            crate::symbolic_scalar::SymbolicScalarTyped::new(_symbolic_resolver),
                        ));
                    }
                }
                let out_info = TensorInfo::from_dtype_and_shape_scalars(out_dtype, &out_dims);
                return Ok(Box::new([(self.output, out_info)].into_iter()));
            }
        }

        Err(MilliOpGraphError::UnableToInfer)
    }

    fn eval(
        &self,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
        _backend: &mut EvalBackend,
    ) -> Result<Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>, MilliOpGraphError>
    {
        let shape: Vec<i64> = inputs[&self.shape].try_to_rank::<P1>()?.try_into()?;
        let shape_usize = shape.iter().map(|x| *x as u64).collect::<Vec<_>>();
        let out: NumericTensor<DynRank> =
            NDArrayNumericTensor::<DynRank>::fill(self.value.clone(), &shape_usize)?.into();
        Ok(Box::new([(self.output, out)].into_iter()))
    }
}
