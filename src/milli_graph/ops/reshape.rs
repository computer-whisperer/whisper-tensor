use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::dtype::DType;
use crate::milli_graph::ops::MilliOp;
use crate::milli_graph::{MilliOpGraphError, MilliOpGraphTensorId};
use crate::numeric_tensor::NumericTensor;
use crate::scalar_info::ScalarInfoTyped;
use crate::symbolic_scalar::{SymbolicResolver, SymbolicScalarTyped};
use crate::tensor_info::{TensorInfo, TensorInfoRanked, TensorInfoShaped};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use typenum::P1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilliOpReshape {
    data: MilliOpGraphTensorId,
    shape: MilliOpGraphTensorId,
    allowzero: bool,
}

impl MilliOpReshape {
    pub fn new(data: MilliOpGraphTensorId, shape: MilliOpGraphTensorId, allowzero: bool) -> Self {
        Self {
            data,
            shape,
            allowzero,
        }
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

impl MilliOp for MilliOpReshape {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {
        vec![self.data, self.shape]
    }

    fn infer(
        &self,
        known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>,
        symbolic_resolver: &mut SymbolicResolver,
        backend: &mut EvalBackend,
    ) -> Result<TensorInfo, MilliOpGraphError> {
        let data_input = &known_inputs[&self.data];
        let shape_input = known_inputs[&self.shape]
            .try_to_rank::<P1>(symbolic_resolver)?
            .try_to_type::<i64>()?;

        if let Some(shape) = shape_input.as_numeric() {
            if let Some(data) = data_input.as_shaped() {
                match data {
                    TensorInfoShaped::Numeric(data) => {
                        let inputs = HashMap::from([
                            (self.shape, shape.to_dyn_rank().to_dyn_type()),
                            (self.data, data.clone()),
                        ]);
                        Ok(TensorInfo::from(self.eval(&inputs, backend)?))
                    }
                    TensorInfoShaped::Symbolic(data) => {
                        let new_shape = self.calculate_new_shape(data.shape(), &shape.to_vec())?;
                        Ok(TensorInfo::from(data.reshape(new_shape)))
                    }
                }
            } else {
                let mut new_shape = vec![];
                for (i, dim) in shape.to_vec().into_iter().enumerate() {
                    if dim > 0 {
                        new_shape.push(ScalarInfoTyped::Numeric(dim as u64));
                    } else if dim == 0 {
                        new_shape.push(
                            data_input
                                .shape(symbolic_resolver)
                                .get(&[i as u64], symbolic_resolver)
                                .unwrap(),
                        );
                    } else {
                        new_shape.push(ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(
                            symbolic_resolver,
                        )));
                    }
                }
                Ok(TensorInfo::from(TensorInfoRanked::<DynRank>::new(
                    data_input.first_element(),
                    new_shape,
                    symbolic_resolver,
                )))
            }
        } else if let Some(shape) = shape_input.as_shaped() {
            let output_rank = shape.shape()[0];
            let mut new_shape = vec![];
            for i in 0..output_rank {
                let dim = shape.get(&[i]).unwrap();
                match dim {
                    ScalarInfoTyped::Numeric(dim) => {
                        if dim > 0 {
                            new_shape.push(ScalarInfoTyped::Numeric(dim as u64));
                        } else if dim == 0 {
                            new_shape.push(
                                data_input
                                    .shape(symbolic_resolver)
                                    .get(&[i], symbolic_resolver)
                                    .unwrap(),
                            );
                        } else {
                            new_shape.push(ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(
                                symbolic_resolver,
                            )));
                        }
                    }
                    ScalarInfoTyped::Symbolic(_x) => {
                        // Could be negative or zero, so have to use new symbol
                        new_shape.push(ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(
                            symbolic_resolver,
                        )));
                    }
                }
            }
            Ok(TensorInfo::from(TensorInfoRanked::<DynRank>::new(
                data_input.first_element(),
                new_shape,
                symbolic_resolver,
            )))
        } else {
            // We don't even know the rank of the output
            Ok(TensorInfo::new_from_first_element_and_rank(
                data_input.first_element(),
                shape_input.shape()[0].cast(),
                symbolic_resolver,
            ))
        }
    }

    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        let data_input = &inputs[&self.data];
        let shape_input = &inputs[&self.shape];
        let shape_input_value: Vec<i64> = shape_input
            .cast(DType::I64, backend)?
            .try_to_rank::<P1>()?
            .try_into()?;

        let output_shape = self.calculate_new_shape(&data_input.shape(), &shape_input_value)?;

        let output_value = data_input.reshape(output_shape, backend)?;

        Ok(output_value)
    }

    fn get_name(&self) -> String {
        "Reshape".to_string()
    }
}
