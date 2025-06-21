use std::collections::HashMap;
use std::hash::Hash;
use serde::{Deserialize, Serialize};
use crate::backends::eval_backend::EvalBackend;
use crate::dtype::DTypeError;
use crate::DynRank;
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::numeric_tensor::NumericTensor;
use crate::tensor_info::TensorInfoError;

pub(crate) mod ops;
pub(crate) mod ops_helpers;

#[derive(Debug, thiserror::Error)]
pub enum MilliOpGraphError {
    #[error(transparent)]
    NumericTensorError(#[from] crate::numeric_tensor::NumericTensorError),
    #[error(transparent)]
    NDArrayNumericTensorError(#[from] crate::backends::ndarray_backend::NDArrayNumericTensorError),
    #[error("Unimplemented milli operator: {0}")]
    UnimplementedOperatorError(String),
    #[error("Invalid input for operation {0}")]
    InvalidInput(String),
    #[error(transparent)]
    DTypeError(#[from] DTypeError),
    #[error(transparent)]
    TensorInfoError(#[from] TensorInfoError),
    #[error("Unable to do any type if inference")]
    UnableToInfer
}

#[derive(Debug, Clone, Copy, Hash, Ord, PartialOrd, Eq, PartialEq, Serialize, Deserialize)]
pub struct MilliOpGraphTensorId {
    inner: usize,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilliOpGraph<ID: Hash + Clone + Eq> {
    input_map: HashMap<ID, MilliOpGraphTensorId>,
    output_map: Option<HashMap<MilliOpGraphTensorId, ID>>,
    ops: HashMap<MilliOpGraphTensorId, AnyMilliOp>,
    next_op_id: usize
}

impl<ID: Hash + Clone + Eq> MilliOpGraph<ID> {
    pub fn new(inputs: &[ID]) -> (Self, HashMap<ID, MilliOpGraphTensorId>) {
        let mut next_op_id = 0;
        let mut input_map = HashMap::new();
        for input in inputs {
            input_map.insert(input.clone(), MilliOpGraphTensorId{inner:next_op_id});
            next_op_id += 1;
        }
        (Self{
            input_map: input_map.clone(),
            ops: HashMap::new(),
            output_map: None,
            next_op_id
        }, input_map)
    }

    pub fn set_output_map(&mut self, output_map: HashMap<MilliOpGraphTensorId, ID>) {
        assert!(self.output_map.is_none());
        self.output_map = Some(output_map)
    }

    pub fn push_op(&mut self, op: AnyMilliOp) -> MilliOpGraphTensorId {
        let new_tensor_id = MilliOpGraphTensorId{inner:self.next_op_id};
        self.next_op_id += 1;
        self.ops.insert(new_tensor_id, op);
        new_tensor_id
    }

    pub(crate) fn eval(&self, inputs: &HashMap<ID, NumericTensor<DynRank>>, backend: &mut EvalBackend) -> Result<HashMap<ID, NumericTensor<DynRank>>, MilliOpGraphError> {
        assert!(self.output_map.is_some());

        let op_ids_to_eval: Vec<_> = {
            let mut x = self.ops.keys().collect::<Vec<_>>();
            x.sort();
            x
        };

        let mut intermediate_values = HashMap::new();
        for (tensor_id, tensor_value) in inputs {
            intermediate_values.insert(self.input_map[tensor_id], tensor_value.clone());
        }

        for op_id in op_ids_to_eval {
            let op = &self.ops[op_id];
            let out = op.eval(&intermediate_values, backend)?;
            //assert_eq!(out.has_nan()?, false);
            intermediate_values.insert(*op_id, out);
        }

        let mut outputs = HashMap::new();
        for (a, b) in self.output_map.as_ref().unwrap() {
            outputs.insert(b.clone(), intermediate_values[a].clone());
        }

        Ok(outputs)
    }
}