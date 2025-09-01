use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::backends::ndarray_backend::NDArrayNumericTensor;
use crate::dtype::DType;
use crate::milli_graph::ops::MilliOp;
use crate::milli_graph::{MilliOpGraphError, MilliOpGraphTensorId};
use crate::numeric_tensor::NumericTensor;
use crate::scalar_info::ScalarInfoTyped;
use crate::symbolic_scalar::{SymbolicResolver, SymbolicScalarTyped};
use crate::tensor_info::TensorInfo;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use typenum::P1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilliOpUnsqueeze {
    data: MilliOpGraphTensorId,
    axes: MilliOpGraphTensorId,
}

impl MilliOpUnsqueeze {
    pub fn new(data: MilliOpGraphTensorId, axes: MilliOpGraphTensorId) -> Self {
        Self { data, axes }
    }
}

impl MilliOp for MilliOpUnsqueeze {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {
        vec![self.data, self.axes]
    }

    fn infer(
        &self,
        known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>,
        symbolic_resolver: &mut SymbolicResolver,
        backend: &mut EvalBackend,
    ) -> Result<TensorInfo, MilliOpGraphError> {
        let data_input = &known_inputs[&self.data];
        let axes_input = &known_inputs[&self.axes]
            .try_to_rank::<P1>(symbolic_resolver)?
            .try_to_type::<i64>()?;

        if let Some(axes) = axes_input.as_numeric() {
            if let Some(data) = data_input.as_numeric() {
                let inputs = HashMap::from([
                    (self.data, data.clone()),
                    (self.axes, axes.to_dyn_type().to_dyn_rank()),
                ]);
                Ok(TensorInfo::from(self.eval(&inputs, backend)?))
            } else {
                let axes = axes.to_vec();
                if let Some(data) = data_input.as_ranked() {
                    let mut new_shape = vec![];
                    let new_rank = data.rank() + axes.len();
                    let mut input_i = 0;
                    for i in 0..new_rank {
                        let mut found = false;
                        for axis in &axes {
                            if (axis >= &0 && i == *axis as usize)
                                || (axis < &0 && i == (new_rank as i64 + axis) as usize)
                            {
                                // Skip dim
                                found = true;
                                break;
                            } else {
                                // keep dim
                            }
                        }
                        if found {
                            new_shape.push(ScalarInfoTyped::Numeric(1));
                        } else {
                            new_shape.push(data.shape()[input_i].clone());
                            input_i += 1;
                        }
                    }
                    Ok(TensorInfo::from(data.reshape(
                        new_shape,
                        symbolic_resolver,
                        backend,
                    )?))
                } else {
                    // Data has no defined rank, just decrease the rank by the number of axes.
                    let new_rank = data_input.rank().add_offset(axes.len() as i64);
                    Ok(TensorInfo::new_from_first_element_and_rank(
                        data_input.first_element(),
                        new_rank,
                        symbolic_resolver,
                    ))
                }
            }
        } else if let Some(axes) = axes_input.as_shaped() {
            // Data has no defined rank, just decrease the rank by the number of axes.
            let new_rank = data_input.rank().add_offset(axes.shape()[0] as i64);
            Ok(TensorInfo::new_from_first_element_and_rank(
                data_input.first_element(),
                new_rank,
                symbolic_resolver,
            ))
        } else {
            // Can't infer the output shape at all
            Ok(TensorInfo::new_from_first_element_and_rank(
                data_input.first_element(),
                ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(symbolic_resolver)),
                symbolic_resolver,
            ))
        }
    }

    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        let axes_ndarray = NDArrayNumericTensor::<DynRank>::try_from(
            inputs[&self.axes].cast(DType::I64, backend)?,
        )?;
        let axes = Vec::<i64>::try_from(axes_ndarray.try_to_rank::<P1>()?)?;
        if axes.len() == 1 {
            let axis = axes[0];
            let input_shape = inputs[&self.data].shape();
            let axis = if axis >= 0 {
                axis as usize
            } else {
                (input_shape.len() as i64 + axis) as usize
            };
            let output = inputs[&self.data].unsqueeze(axis)?;
            Ok(output)
        } else {
            // Multiple axes (use reshape)
            let input_shape = inputs[&self.data].shape();
            let mut output_shape = Vec::new();
            let mut input_index = 0;
            for i in 0..(input_shape.len() + axes.len()) {
                let mut is_selected = false;
                for axis in &axes {
                    let axis = if *axis < 0 {
                        input_shape.len() as i64 + *axis
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
            Ok(output)
        }
    }

    fn get_name(&self) -> String {
        "Unsqueeze".to_string()
    }
}
