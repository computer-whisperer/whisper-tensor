use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::dtype::DTypeError;
use crate::graph::{GlobalId, Graph, Link, Node};
use crate::milli_graph::observer::MilliOpGraphObserver;
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::numeric_tensor::NumericTensor;
use crate::tensor_info::TensorInfoError;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::time::Instant;
use rand::Rng;

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
pub struct MilliOpGraphTensor {
    global_id: GlobalId,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilliOpGraph {
    global_id: GlobalId,
    pub input_map: HashMap<GlobalId, GlobalId>,
    pub input_ordering: Vec<GlobalId>,
    pub output_map: Option<HashMap<GlobalId, GlobalId>>,
    pub output_ordering: Option<Vec<GlobalId>>,
    ops: HashMap<GlobalId, AnyMilliOp>,
    op_ordering: Vec<GlobalId>,
    tensors: HashMap<GlobalId, MilliOpGraphTensor>,
}

impl MilliOpGraph {
    pub fn new(inputs: impl IntoIterator<Item = GlobalId>, rng: &mut impl Rng) -> (Self, HashMap<GlobalId, GlobalId>) {
        let mut input_map = HashMap::new();
        let mut input_ordering = Vec::new();
        let mut tensors = HashMap::new();
        for input in inputs {
            input_ordering.push(input.clone());
            let global_id = GlobalId::new(rng);
            input_map.insert(input.clone(), global_id);
            tensors.insert(global_id, MilliOpGraphTensor { global_id });
        }
        (
            Self {
                global_id: GlobalId::new(rng),
                tensors,
                input_ordering,
                output_ordering: None,
                input_map: input_map.clone(),
                ops: HashMap::new(),
                output_map: None,
                op_ordering: vec![],
            },
            input_map,
        )
    }

    pub fn get_inputs(&self) -> Vec<GlobalId> {
        self.input_ordering.clone()
    }

    pub fn get_all_tensors(&self) -> HashSet<GlobalId> {
        let mut result = HashSet::new();
        for v in self.input_map.values() {
            result.insert(*v);
        }
        for op in self.ops.values() {
            for tid in op.outputs() {
                result.insert(tid);
            }
        }
        result
    }

    pub fn get_outputs(&self) -> Vec<GlobalId> {
        if let Some(x) = &self.output_ordering {
            x.clone()
        } else if let Some(x) = &self.output_map {
            x.values().cloned().collect()
        } else {
            vec![]
        }
    }

    pub fn set_output_map(
        &mut self,
        output_map: impl IntoIterator<Item = (GlobalId, GlobalId)>,
    ) {
        assert!(self.output_map.is_none());
        self.output_map = Some(output_map.into_iter().collect());
    }

    pub fn set_output_map_ordered(
        &mut self,
        output_map: HashMap<GlobalId, GlobalId>,
        output_ordering: Vec<GlobalId>,
    ) {
        assert!(self.output_map.is_none());
        assert!(self.output_ordering.is_none());
        self.output_map = Some(output_map);
        self.output_ordering = Some(output_ordering);
    }

    pub fn get_new_tensor_id(&mut self, rng: &mut impl Rng) -> GlobalId {
        let global_id = GlobalId::new(rng);
        self.tensors.insert(global_id, MilliOpGraphTensor { global_id });
        global_id
    }

    pub fn push_op(&mut self, op: AnyMilliOp) -> GlobalId {
        let id = op.global_id();
        self.ops.insert(id, op);
        self.op_ordering.push(id);
        id
    }

    #[allow(clippy::type_complexity)]
    pub(crate) fn eval<T: MilliOpGraphObserver>(
        &self,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
        observer: &mut T,
        backend: &mut EvalBackend,
    ) -> Result<Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>, MilliOpGraphError> {
        assert!(self.output_map.is_some());

        let mut intermediate_values = HashMap::new();
        for (tensor_id, tensor_value) in inputs {
            intermediate_values.insert(self.input_map[tensor_id], tensor_value.clone());
        }

        for op_id in &self.op_ordering {
            let op = &self.ops[op_id];
            let start_instant = Instant::now();
            let out_vec: Vec<_> = op.eval(&intermediate_values, backend)?.collect();
            let end_instant = Instant::now();
            observer.on_node_executed(
                &[op.global_id()],
                start_instant,
                end_instant,
                backend,
            );
            for (tensor_id, value) in out_vec {
                observer.on_tensor_assigned(
                    &[self.tensors[&tensor_id].global_id()],
                    &value,
                    backend,
                );
                intermediate_values.insert(tensor_id, value);
            }
        }

        let mut outputs = HashMap::new();
        for (a, b) in self.output_map.as_ref().unwrap() {
            outputs.insert(b.clone(), intermediate_values[a].clone());
        }

        Ok(Box::new(outputs.into_iter()))
    }
}

impl Graph for MilliOpGraph
{
    type Error = ();
    type AnyNode = AnyMilliOp;
    type AnyLink = MilliOpGraphTensor;

    fn global_id(&self) -> GlobalId {
        self.global_id
    }

    fn node_ids(&self) -> impl Iterator<Item = GlobalId> {
        self.ops.keys().cloned()
    }

    fn inner_link_ids(&self) -> impl Iterator<Item = GlobalId> {
        self.get_all_tensors().into_iter()
    }

    fn get_node_by_id(&self, id: &GlobalId) -> Option<&Self::AnyNode> {
        self.ops.get(id)
    }

    fn get_link_by_id(&self, id: &GlobalId) -> Option<&Self::AnyLink> {
        self.tensors.get(id)
    }

    fn input_link_ids(&self) -> impl Iterator<Item = (GlobalId, GlobalId)> {
        self.input_ordering
            .iter()
            .map(|x| (x.clone(), self.input_map[x]))
    }

    fn constant_link_ids(&self) -> impl Iterator<Item = GlobalId> {
        core::iter::empty()
    }

    fn output_link_ids(&self) -> impl Iterator<Item = (GlobalId, GlobalId)> {
        let mut output = vec![];
        if let Some(ordering) = &self.output_ordering {
            let map = self.output_map.as_ref().unwrap();
            output.extend(ordering.iter().cloned().map(move |x| {
                let tid = map
                    .iter()
                    .find(|(_, id)| **id == x)
                    .map(|(tid, _)| *tid)
                    .expect("output id not found in map");
                (x, tid)
            }))
        } else {
            output.extend(
                self.output_map
                    .as_ref()
                    .unwrap()
                    .iter()
                    .map(|(tid, id)| (id.clone(), *tid)),
            )
        }
        output.into_iter()
    }
}


impl crate::graph::Link for MilliOpGraphTensor {
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
}

impl crate::graph::LinkMetadata for MilliOpGraphTensor {
    // MilliOpGraph tensors don't have rich metadata like dtype/shape at this level
}
