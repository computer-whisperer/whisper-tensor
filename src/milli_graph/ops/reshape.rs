use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::dtype::DType;
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError, MilliOpGraphTensorId};
use crate::numeric_tensor::NumericTensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use typenum::P1;
use crate::graph::Node;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilliOpReshape {
    output: MilliOpGraphTensorId,
    data: MilliOpGraphTensorId,
    shape: MilliOpGraphTensorId,
    allowzero: bool,
}

impl MilliOpReshape {
    pub fn new<T: std::hash::Hash + Clone + Eq>(graph: &mut MilliOpGraph<T>, data: MilliOpGraphTensorId, shape: MilliOpGraphTensorId, allowzero: bool) -> MilliOpGraphTensorId {
        let output = graph.get_new_tensor_id();
        let node = Self {
            output,
            data,
            shape,
            allowzero,
        };
        graph.push_op(AnyMilliOp::Reshape(node));
        output
    }

    fn calculate_new_shape(
        &self,
        data_input_shape: &[u64],
        shape_input_value: &[i64],
    ) -> Result<Vec<u64>, MilliOpGraphError> {
        let mut new_shape_dims = vec![];
        let mut backfill_dim: Option<usize> = None;
        for i in 0..shape_input_value.len() {
            new_shape_dims.push(if shape_input_value[i] == 0 {
                data_input_shape[i]
            } else if shape_input_value[i] == -1 {
                if backfill_dim.is_some() {
                    // Only one dimension can be inferred
                    Err(MilliOpGraphError::InvalidInput("Reshape".to_string()))?
                }
                backfill_dim = Some(i);
                1
            } else if shape_input_value[i] < -1 {
                Err(MilliOpGraphError::InvalidInput("Reshape".to_string()))?
            } else {
                shape_input_value[i] as u64
            });
        }

        // Backfill the inferred dimension
        if let Some(i) = backfill_dim {
            let total_input_size = data_input_shape.iter().product::<u64>();

            // Calculate the current product of the dimensions
            let mut current_product = 1;
            for (j, dim) in new_shape_dims.iter().enumerate() {
                if j != i {
                    current_product *= dim;
                }
            }
            // Calculate the inferred dimension size
            let inferred_size = total_input_size / current_product;
            new_shape_dims[i] = inferred_size;
        }
        let output_shape = new_shape_dims;

        // Verify that the dimensions are compatible
        if output_shape.iter().product::<u64>() != data_input_shape.iter().product::<u64>() {
            Err(MilliOpGraphError::InvalidInput("Reshape".to_string()))?
        }

        Ok(output_shape)
    }
}

impl Node<MilliOpGraphTensorId> for MilliOpReshape {
    fn inputs(&self) -> impl Iterator<Item=MilliOpGraphTensorId> { vec![self.data, self.shape].into_iter() }
    fn outputs(&self) -> impl Iterator<Item=MilliOpGraphTensorId> { vec![self.output].into_iter() }
}

impl MilliOp for MilliOpReshape {
    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<impl Iterator<Item=(MilliOpGraphTensorId, NumericTensor<DynRank>)>, MilliOpGraphError> {
        let data_input = &inputs[&self.data];
        let shape_input = &inputs[&self.shape];
        let shape_input_value: Vec<i64> = shape_input
            .cast(DType::I64, backend)?
            .try_to_rank::<P1>()?
            .try_into()?;

        let output_shape = self.calculate_new_shape(&data_input.shape(), &shape_input_value)?;

        let output_value = data_input.reshape(output_shape, backend)?;

        Ok([(self.output, output_value)].into_iter())
    }

    fn get_name(&self) -> String {
        "Reshape".to_string()
    }
}
