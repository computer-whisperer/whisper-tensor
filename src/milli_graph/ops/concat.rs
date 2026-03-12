use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::graph::GlobalId;
use crate::milli_graph::ops::{AnyMilliOp, MilliOp, MilliOpTensorIDOrLiteral};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError};
use crate::numeric_tensor::NumericTensor;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Concat {
    global_id: GlobalId,
    output: GlobalId,
    inputs: Vec<GlobalId>,
    axis: i64,
}

impl Concat {
    pub fn push_new(
        graph: &mut MilliOpGraph,
        inputs: Vec<GlobalId>,
        axis: i64,
        rng: &mut impl Rng,
    ) -> GlobalId {
        let output = graph.get_new_tensor_id(rng);
        let node = Self {
            global_id: GlobalId::new(rng),
            output,
            inputs,
            axis,
        };
        graph.push_op(AnyMilliOp::Concat(node));
        output
    }
}

impl Concat {
    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl rand::Rng) {
        self.global_id = GlobalId::new(rng);
        super::remap(&mut self.output, map);
        for id in &mut self.inputs {
            super::remap(id, map);
        }
    }
}

impl crate::graph::Node for Concat {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "Concat".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(self.inputs.clone().into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.output].into_iter())
    }
}

impl MilliOp for Concat {
    fn infer(
        &self,
        known_inputs: &HashMap<GlobalId, crate::tensor_info::TensorInfo>,
        symbolic_resolver: &mut crate::symbolic_scalar::SymbolicResolver,
        backend: &mut EvalBackend,
    ) -> Result<
        Box<dyn Iterator<Item = (GlobalId, crate::tensor_info::TensorInfo)>>,
        MilliOpGraphError,
    > {
        use crate::tensor_info::TensorInfo;

        // Collect input infos.
        let mut input_infos = Vec::new();
        for id in &self.inputs {
            let info = known_inputs
                .get(id)
                .ok_or(MilliOpGraphError::UnableToInfer)?;
            input_infos.push(info);
        }

        // If all inputs are concrete, fall back to eval.
        if input_infos.iter().all(|info| info.as_numeric().is_some()) {
            let mut resolved = HashMap::new();
            for (id, info) in self.inputs.iter().zip(input_infos.iter()) {
                resolved.insert(*id, info.as_numeric().unwrap().clone());
            }
            let collected: Vec<(GlobalId, TensorInfo)> = self
                .eval(&resolved, backend)?
                .map(|(a, b)| (a, TensorInfo::from(b)))
                .collect();
            return Ok(Box::new(collected.into_iter()));
        }

        // Concat: output rank = input rank, output dtype = first input dtype.
        let first = input_infos[0];
        let out_dtype = first.dtype();
        let out_rank = first.rank();

        let first_elem = crate::scalar_info::ScalarInfo::Symbolic(
            crate::symbolic_scalar::SymbolicScalar::new(out_dtype, symbolic_resolver),
        );
        let out_info =
            TensorInfo::new_from_first_element_and_rank(first_elem, out_rank, symbolic_resolver);
        Ok(Box::new([(self.output, out_info)].into_iter()))
    }

    fn backward(
        &self,
        output_grads: &HashMap<GlobalId, GlobalId>,
        graph: &mut MilliOpGraph,
        rng: &mut impl Rng,
    ) -> Option<HashMap<GlobalId, GlobalId>> {
        let grad_output = *output_grads.get(&self.output)?;
        let n = self.inputs.len();

        // Build split sizes by gathering each input's dim along the concat axis
        let axis_idx = super::Constant::push_new(
            graph,
            crate::backends::ndarray_backend::NDArrayNumericTensor::<DynRank>::from_vec_shape(
                vec![self.axis],
                &vec![1],
            )
            .unwrap(),
            rng,
        );
        let mut size_tensors = Vec::new();
        for &input_id in &self.inputs {
            let shape = super::Shape::push_new(graph, input_id, rng);
            let size = super::Gather::push_new(graph, shape, axis_idx, 0, rng);
            size_tensors.push(size);
        }
        let split_sizes = super::Concat::push_new(graph, size_tensors, 0, rng);

        // Split the gradient along the same axis
        let mut result = HashMap::new();
        for (i, &input_id) in self.inputs.iter().enumerate() {
            let grad_i = super::Split::push_new(
                graph,
                grad_output,
                Some(MilliOpTensorIDOrLiteral::TensorID(split_sizes)),
                self.axis,
                Some(n),
                i,
                rng,
            );
            result
                .entry(input_id)
                .and_modify(|existing: &mut GlobalId| {
                    *existing = super::SimpleBinary::add(graph, *existing, grad_i, rng);
                })
                .or_insert(grad_i);
        }
        Some(result)
    }

    fn eval(
        &self,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>, MilliOpGraphError>
    {
        let mut resolved_inputs = vec![];
        for input in &self.inputs {
            resolved_inputs.push(&inputs[input]);
        }
        let axis = if self.axis < 0 {
            resolved_inputs[0].shape().len() as i64 + self.axis
        } else {
            self.axis
        } as usize;
        let out = NumericTensor::<DynRank>::concat(resolved_inputs.as_slice(), axis, backend)?;
        Ok(Box::new([(self.output, out)].into_iter()))
    }
}
