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
use crate::graph::{Graph, GraphPath, InnerGraph, LinkPath, Node, NodePath};

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

#[derive(Debug, Clone, Copy, Hash, Ord, PartialOrd, Eq, PartialEq, Serialize, Deserialize)]
pub struct MilliOpGraphNodeId {
    inner: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilliOpGraph<ID: Hash + Clone + Eq> {
    pub input_map: HashMap<ID, MilliOpGraphTensorId>,
    pub input_ordering: Vec<ID>,
    pub output_map: Option<HashMap<MilliOpGraphTensorId, ID>>,
    pub output_ordering: Option<Vec<ID>>,
    ops: HashMap<MilliOpGraphNodeId, AnyMilliOp>,
    next_tensor_id: usize,
    next_node_id: usize,
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
                next_tensor_id: next_op_id,
                next_node_id: 0,
            },
            input_map,
        )
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

    pub fn get_new_tensor_id(&mut self) -> MilliOpGraphTensorId {
        let new_id = MilliOpGraphTensorId {
            inner: self.next_tensor_id,
        };
        self.next_tensor_id += 1;
        new_id
    }

    pub fn push_op(&mut self, op: AnyMilliOp) -> MilliOpGraphNodeId {
        let new_node_id = MilliOpGraphNodeId {
            inner: self.next_node_id,
        };
        self.next_node_id += 1;
        self.ops.insert(new_node_id, op);
        new_node_id
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
            for (tensor_id, value) in out {
                observer.on_tensor_assigned(&MilliOpGraphTensorPath::Tensor(tensor_id), &value, backend);
                intermediate_values.insert(*tensor_id, value);
            }
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
    Op(MilliOpGraphNodeId),
}

impl GraphPath for () {
}

impl NodePath for MilliOpGraphNodePath {
}

impl LinkPath for MilliOpGraphTensorPath {
}


impl<IoLinkId: Hash + Clone + Eq> Graph for MilliOpGraph<IoLinkId> {
    type GraphPath = ();
    type NodePath = MilliOpGraphNodePath;
    type LinkPath = MilliOpGraphTensorPath;
    type Inner = Self;
    type AnySubGraph = Self;

    fn inner(&self, _: &Self::GraphPath) -> &Self::AnySubGraph {
        self
    }
}

impl<IoLinkId: Hash + Clone + Eq> InnerGraph for MilliOpGraph<IoLinkId> {
    type NodeId = MilliOpGraphNodeId;
    type LinkId = MilliOpGraphTensorId;
    type Error = ();
    type AnyNode = AnyMilliOp;
    type AnyLink = ();
    type InputLinkId = IoLinkId;
    type OutputLinkId = IoLinkId;

    fn nodes(&self) -> impl Iterator<Item=Self::NodeId> {
        self.ops.keys().cloned()
    }

    fn links(&self) -> impl Iterator<Item=Self::LinkId> {
        self.ops.keys().cloned()
    }

    fn get_node(&self, id: &Self::NodeId) -> Option<&Self::AnyNode> {
        self.ops.get(id)
    }

    fn get_link(&self, _id: &Self::LinkId) -> Option<&Self::AnyLink> {
        Some(&())
    }

    fn input_links(&self) -> impl Iterator<Item=(Self::InputLinkId, Self::LinkId)> {
        self.input_ordering.map(|x| (x, self.input_map[&x]))
    }

    fn output_links(&self) -> impl Iterator<Item=(Self::OutputLinkId, Self::LinkId)> {
        self.output_ordering.as_ref().unwrap().map(|x| (x, self.output_map.unwrap()[&x]))
    }
}
