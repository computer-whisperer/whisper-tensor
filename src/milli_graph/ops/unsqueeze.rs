use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::backends::ndarray_backend::NDArrayNumericTensor;
use crate::dtype::DType;
use crate::graph::{GlobalId, Node};
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError};
use crate::numeric_tensor::NumericTensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use typenum::P1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Unsqueeze {
    global_id: GlobalId,
    output: GlobalId,
    data: GlobalId,
    axes: GlobalId,
}

impl Unsqueeze {
    pub fn push_new(
        graph: &mut MilliOpGraph,
        data: GlobalId,
        axes: GlobalId,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        let output = graph.get_new_tensor_id(rng);
        let node = Self {
            output,
            data,
            axes,
            global_id: GlobalId::new(rng),
        };
        graph.push_op(AnyMilliOp::Unsqueeze(node));
        output
    }
}

impl Unsqueeze {
    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl rand::Rng) {
        self.global_id = GlobalId::new(rng);
        super::remap(&mut self.output, map);
        super::remap(&mut self.data, map);
        super::remap(&mut self.axes, map);
    }
}

impl Node for Unsqueeze {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> String {
        "Unsqueeze".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.data, self.axes].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.output].into_iter())
    }
}

impl MilliOp for Unsqueeze {
    fn infer(
        &self,
        known_inputs: &HashMap<GlobalId, crate::tensor_info::TensorInfo>,
        symbolic_resolver: &mut crate::symbolic_scalar::SymbolicResolver,
        backend: &mut EvalBackend,
    ) -> Result<
        Box<dyn Iterator<Item = (GlobalId, crate::tensor_info::TensorInfo)>>,
        MilliOpGraphError,
    > {
        use crate::scalar_info::ScalarInfoTyped;
        use crate::symbolic_scalar::SymbolicScalarTyped;
        use crate::tensor_info::TensorInfo;

        let data_info = known_inputs
            .get(&self.data)
            .ok_or(MilliOpGraphError::UnableToInfer)?;
        let axes_info = known_inputs
            .get(&self.axes)
            .ok_or(MilliOpGraphError::UnableToInfer)?;

        // If both inputs are concrete, delegate to eval
        if let (Some(data_num), Some(axes_num)) = (data_info.as_numeric(), axes_info.as_numeric()) {
            let inputs =
                HashMap::from([(self.data, data_num.clone()), (self.axes, axes_num.clone())]);
            let out: Vec<_> = self
                .eval(&inputs, backend)?
                .map(|(id, t)| (id, TensorInfo::from(t)))
                .collect();
            return Ok(Box::new(out.into_iter()));
        }

        let first_elem = data_info.first_element();

        // If we know the input shape and axes are concrete, compute output shape
        if let (Some(data_ranked), Some(axes_num)) = (data_info.as_ranked(), axes_info.as_numeric())
        {
            let axes_values: Vec<i64> = axes_num
                .cast(DType::I64, backend)?
                .try_to_rank::<P1>()?
                .try_into()?;

            let input_shape = data_ranked.shape();
            let output_rank = input_shape.len() + axes_values.len();

            // ONNX: negative axes refer to the output rank
            let normalized_axes: Vec<usize> = axes_values
                .iter()
                .map(|&a| {
                    if a < 0 {
                        (output_rank as i64 + a) as usize
                    } else {
                        a as usize
                    }
                })
                .collect();

            let mut output_shape: Vec<ScalarInfoTyped<u64>> = Vec::new();
            let mut input_idx = 0;
            for i in 0..output_rank {
                if normalized_axes.contains(&i) {
                    output_shape.push(ScalarInfoTyped::Numeric(1));
                } else {
                    output_shape.push(input_shape[input_idx].clone());
                    input_idx += 1;
                }
            }

            let out = TensorInfo::Ranked(crate::tensor_info::TensorInfoRanked::new(
                first_elem,
                output_shape,
                symbolic_resolver,
            ));
            return Ok(Box::new([(self.output, out)].into_iter()));
        }

        // If axes are concrete, we can at least compute output rank
        if let Some(axes_num) = axes_info.as_numeric() {
            let axes_values: Vec<i64> = axes_num
                .cast(DType::I64, backend)?
                .try_to_rank::<P1>()?
                .try_into()?;
            let num_axes = axes_values.len() as u32;

            if let ScalarInfoTyped::Numeric(input_rank) = data_info.rank() {
                let output_rank = input_rank + num_axes;
                let out = TensorInfo::new_from_first_element_and_rank(
                    first_elem,
                    ScalarInfoTyped::Numeric(output_rank),
                    symbolic_resolver,
                );
                return Ok(Box::new([(self.output, out)].into_iter()));
            }
        }

        // Fallback: propagate dtype only
        let out = TensorInfo::new_from_first_element_and_rank(
            first_elem,
            ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(symbolic_resolver)),
            symbolic_resolver,
        );
        Ok(Box::new([(self.output, out)].into_iter()))
    }

    fn backward(
        &self,
        output_grads: &HashMap<GlobalId, GlobalId>,
        graph: &mut MilliOpGraph,
        rng: &mut impl rand::Rng,
    ) -> Option<HashMap<GlobalId, GlobalId>> {
        let grad_output = *output_grads.get(&self.output)?;
        // Unsqueeze backward = Squeeze along the same axes
        let grad_input = super::Squeeze::push_new(graph, grad_output, self.axes, rng);
        let mut result = HashMap::new();
        result.insert(self.data, grad_input);
        Some(result)
    }

    fn eval(
        &self,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>, MilliOpGraphError>
    {
        let axes_ndarray = NDArrayNumericTensor::<DynRank>::try_from(
            inputs[&self.axes].cast(DType::I64, backend)?,
        )?;
        let axes = Vec::<i64>::try_from(axes_ndarray.try_to_rank::<P1>()?)?;
        let input_shape = inputs[&self.data].shape();
        let output_rank = input_shape.len() + axes.len();
        if axes.len() == 1 {
            let axis = axes[0];
            // ONNX: negative axes refer to the output rank, not the input rank
            let axis = if axis >= 0 {
                axis as usize
            } else {
                (output_rank as i64 + axis) as usize
            };
            let output = inputs[&self.data].unsqueeze(axis)?;
            Ok(Box::new([(self.output, output)].into_iter()))
        } else {
            // Multiple axes (use reshape)
            // ONNX: negative axes refer to the output rank
            let mut output_shape = Vec::new();
            let mut input_index = 0;
            for i in 0..output_rank {
                let mut is_selected = false;
                for axis in &axes {
                    // Negative axes relative to output rank
                    let axis = if *axis < 0 {
                        output_rank as i64 + *axis
                    } else {
                        *axis
                    };
                    if axis == i as i64 {
                        is_selected = true;
                        break;
                    }
                }
                if is_selected {
                    output_shape.push(1);
                } else {
                    output_shape.push(input_shape[input_index]);
                    input_index += 1;
                }
            }
            let output = inputs[&self.data].reshape(output_shape, backend)?;
            Ok(Box::new([(self.output, output)].into_iter()))
        }
    }
}
