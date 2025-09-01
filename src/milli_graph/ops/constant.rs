use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::backends::ndarray_backend::NDArrayNumericTensor;
use crate::backends::ndarray_backend::conversions::NDArrayNumericTensorType;
use crate::milli_graph::ops::MilliOp;
use crate::milli_graph::{MilliOpGraphError, MilliOpGraphTensorId};
use crate::numeric_scalar::NumericScalar;
use crate::numeric_tensor::NumericTensor;
use crate::scalar_info::ScalarInfo;
use crate::symbolic_scalar::SymbolicResolver;
use crate::tensor_info::{
    TensorInfo, TensorInfoRanked, TensorInfoTypedRanked, TensorInfoTypedShaped,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use typenum::P1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilliOpConstant {
    data: NDArrayNumericTensor<DynRank>,
}

impl MilliOpConstant {
    pub fn new(a: NDArrayNumericTensor<DynRank>) -> Self {
        Self { data: a }
    }

    pub(crate) fn new_scalar<T>(v: T) -> Self
    where
        T: NDArrayNumericTensorType,
    {
        Self {
            data: NDArrayNumericTensor::<DynRank>::from_vec_shape(vec![v], &vec![1]).unwrap(),
        }
    }
}

impl MilliOp for MilliOpConstant {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {
        vec![]
    }

    fn infer(
        &self,
        _known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>,
        _symbolic_resolver: &mut SymbolicResolver,
        _backend: &mut EvalBackend,
    ) -> Result<TensorInfo, MilliOpGraphError> {
        Ok(TensorInfo::from(NumericTensor::NDArray(self.data.clone())))
    }
    fn eval(
        &self,
        _inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        _backend: &mut EvalBackend,
    ) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        Ok(self.data.clone().into())
    }

    fn get_name(&self) -> String {
        "Constant".to_string()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilliOpConstantOfShape {
    value: NumericScalar,
    shape: MilliOpGraphTensorId,
}

impl MilliOpConstantOfShape {
    pub fn new(value: NumericScalar, shape: MilliOpGraphTensorId) -> Self {
        Self { value, shape }
    }
}

impl MilliOp for MilliOpConstantOfShape {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {
        vec![]
    }

    fn infer(
        &self,
        known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>,
        symbolic_resolver: &mut SymbolicResolver,
        backend: &mut EvalBackend,
    ) -> Result<TensorInfo, MilliOpGraphError> {
        let input = &known_inputs[&self.shape];
        let input = input
            .try_to_rank::<P1>(symbolic_resolver)?
            .try_to_type::<u64>()?;

        match input {
            TensorInfoTypedRanked::Shaped(tensor) => match tensor {
                TensorInfoTypedShaped::Numeric(tensor) => {
                    let inputs = HashMap::from([(self.shape, tensor.to_dyn_rank().to_dyn_type())]);
                    Ok(TensorInfo::from(self.eval(&inputs, backend)?))
                }
                TensorInfoTypedShaped::Shaped(tensor) => {
                    let mut new_shape = vec![];
                    for i in 0..tensor.shape()[0] {
                        new_shape.push(tensor.get(&[i]).unwrap().clone());
                    }
                    Ok(TensorInfo::from(TensorInfoRanked::<DynRank>::new(
                        ScalarInfo::Numeric(self.value.clone()),
                        new_shape,
                        symbolic_resolver,
                    )))
                }
            },
            TensorInfoTypedRanked::Ranked(tensor) => {
                Ok(TensorInfo::new_from_first_element_and_rank(
                    ScalarInfo::Numeric(self.value.clone()),
                    tensor.shape()[0].cast(),
                    symbolic_resolver,
                ))
            }
        }
    }

    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        _backend: &mut EvalBackend,
    ) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        let shape: Vec<i64> = inputs[&self.shape].try_to_rank::<P1>()?.try_into()?;
        let shape_usize = shape.iter().map(|x| *x as u64).collect::<Vec<_>>();
        Ok(NDArrayNumericTensor::<DynRank>::fill(self.value.clone(), &shape_usize)?.into())
    }

    fn get_name(&self) -> String {
        "Constant of Shape".to_string()
    }
}
