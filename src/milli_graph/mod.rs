use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::dtype::DTypeError;
use crate::milli_graph::observer::MilliOpGraphObserver;
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::numeric_tensor::NumericTensor;
use crate::tensor_info::TensorInfoError;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::time::Instant;

pub mod observer;
pub mod ops;
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
    UnableToInfer,
}

#[derive(Debug, Clone, Copy, Hash, Ord, PartialOrd, Eq, PartialEq, Serialize, Deserialize)]
pub struct MilliOpGraphTensorId {
    inner: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilliOpGraph<ID: Hash + Clone + Eq> {
    pub input_map: HashMap<ID, MilliOpGraphTensorId>,
    pub input_ordering: Vec<ID>,
    pub output_map: Option<HashMap<MilliOpGraphTensorId, ID>>,
    pub output_ordering: Option<Vec<ID>>,
    ops: HashMap<MilliOpGraphTensorId, AnyMilliOp>,
    next_op_id: usize,
}

impl<ID: Hash + Clone + Eq> MilliOpGraph<ID> {
    pub fn new(inputs: &[ID]) -> (Self, HashMap<ID, MilliOpGraphTensorId>) {
        let mut next_op_id = 0;
        let mut input_map = HashMap::new();
        let mut input_ordering = Vec::new();
        for input in inputs {
            input_ordering.push(input.clone());
            input_map.insert(input.clone(), MilliOpGraphTensorId { inner: next_op_id });
            next_op_id += 1;
        }
        (
            Self {
                input_ordering,
                output_ordering: None,
                input_map: input_map.clone(),
                ops: HashMap::new(),
                output_map: None,
                next_op_id,
            },
            input_map,
        )
    }

    pub fn get_op(&self, id: &MilliOpGraphTensorId) -> Option<&AnyMilliOp> {
        self.ops.get(id)
    }

    pub fn get_inputs(&self) -> Vec<ID> {
        self.input_ordering.clone()
    }

    pub fn get_all_tensors(&self) -> HashSet<MilliOpGraphTensorId> {
        let mut result = HashSet::new();
        result.extend(self.input_map.values());
        result.extend(self.ops.keys());
        result
    }

    pub fn get_all_ops(&self) -> &HashMap<MilliOpGraphTensorId, AnyMilliOp> {
        &self.ops
    }

    pub fn get_outputs(&self) -> Vec<ID> {
        if let Some(x) = &self.output_ordering {
            x.clone()
        } else if let Some(x) = &self.output_map {
            x.values().cloned().collect()
        } else {
            vec![]
        }
    }

    pub fn set_output_map(&mut self, output_map: HashMap<MilliOpGraphTensorId, ID>) {
        assert!(self.output_map.is_none());
        self.output_map = Some(output_map)
    }

    pub fn set_output_map_ordered(
        &mut self,
        output_map: HashMap<MilliOpGraphTensorId, ID>,
        output_ordering: Vec<ID>,
    ) {
        assert!(self.output_map.is_none());
        assert!(self.output_ordering.is_none());
        self.output_map = Some(output_map);
        self.output_ordering = Some(output_ordering);
    }

    pub fn push_op(&mut self, op: AnyMilliOp) -> MilliOpGraphTensorId {
        let new_tensor_id = MilliOpGraphTensorId {
            inner: self.next_op_id,
        };
        self.next_op_id += 1;
        self.ops.insert(new_tensor_id, op);
        new_tensor_id
    }

    pub(crate) fn eval<T: MilliOpGraphObserver>(
        &self,
        inputs: &HashMap<ID, NumericTensor<DynRank>>,
        observer: &mut T,
        backend: &mut EvalBackend,
    ) -> Result<HashMap<ID, NumericTensor<DynRank>>, MilliOpGraphError> {
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
            let start_instant = Instant::now();
            let out = op.eval(&intermediate_values, backend)?;
            let end_instant = Instant::now();
            observer.on_node_executed(
                &MilliOpGraphNodePath::Op(*op_id),
                start_instant,
                end_instant,
                backend,
            );
            observer.on_tensor_assigned(&MilliOpGraphTensorPath::Tensor(*op_id), &out, backend);
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

#[derive(Clone, Debug, Serialize, Deserialize, Hash, PartialEq, Eq)]
pub enum MilliOpGraphTensorPath {
    Tensor(MilliOpGraphTensorId),
}

#[derive(Clone, Debug, Serialize, Deserialize, Hash, PartialEq, Eq)]
pub enum MilliOpGraphNodePath {
    Op(MilliOpGraphTensorId),
}
