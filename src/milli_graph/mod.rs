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

    pub fn new_empty(rng: &mut impl Rng) -> Self {
        Self {
            global_id: GlobalId::new(rng),
            tensors: HashMap::new(),
            input_ordering: Vec::new(),
            output_ordering: None,
            input_map: HashMap::new(),
            ops: HashMap::new(),
            output_map: None,
            op_ordering: Vec::new(),
        }
    }

    /// Add an input with a known external ID. Returns the internal tensor ID.
    pub fn add_input_with_id(&mut self, external_id: GlobalId, rng: &mut impl Rng) -> GlobalId {
        let internal_id = GlobalId::new(rng);
        self.input_map.insert(external_id, internal_id);
        self.input_ordering.push(external_id);
        self.tensors.insert(internal_id, MilliOpGraphTensor { global_id: internal_id });
        internal_id
    }

    /// Mark an internal tensor as an output with a given external ID.
    pub fn add_output(&mut self, internal_id: GlobalId, external_id: GlobalId) {
        let output_map = self.output_map.get_or_insert_with(HashMap::new);
        output_map.insert(internal_id, external_id);
        let output_ordering = self.output_ordering.get_or_insert_with(Vec::new);
        output_ordering.push(external_id);
    }

    /// Merge another MilliOpGraph's ops into self.
    /// `sym_to_combined` maps symbolic-graph tensor IDs to combined-graph internal tensor IDs.
    /// Updated in-place: output tensors from `other` are added to the mapping.
    pub fn merge_graph(
        &mut self,
        other: MilliOpGraph,
        sym_to_combined: &mut HashMap<GlobalId, GlobalId>,
        rng: &mut impl Rng,
    ) {
        // Build tensor_map: other's internal IDs → self's internal IDs
        let mut tensor_map: HashMap<GlobalId, GlobalId> = HashMap::new();

        // Map other's inputs: other_internal → sym_to_combined[external]
        for (external, other_internal) in &other.input_map {
            let combined_id = sym_to_combined[external];
            tensor_map.insert(*other_internal, combined_id);
        }

        // For each non-input tensor in other, create a fresh ID in self
        let input_internals: HashSet<GlobalId> = other.input_map.values().copied().collect();
        for &tensor_id in other.tensors.keys() {
            if !input_internals.contains(&tensor_id) {
                let fresh_id = self.get_new_tensor_id(rng);
                tensor_map.insert(tensor_id, fresh_id);
            }
        }

        // Clone + remap each op, push to self
        for op_id in &other.op_ordering {
            let mut op = other.ops[op_id].clone();
            op.remap_tensors(&tensor_map, rng);
            self.push_op(op);
        }

        // Map other's outputs back to sym_to_combined
        if let Some(output_map) = &other.output_map {
            for (other_internal, external) in output_map {
                sym_to_combined.insert(*external, tensor_map[other_internal]);
            }
        }
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


#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::eval_backend::EvalBackend;
    use crate::backends::ndarray_backend::NDArrayNumericTensor;
    use crate::milli_graph::ops::{Constant, SimpleBinary};

    #[test]
    fn test_merge_graph_two_small_graphs() {
        // Graph A: computes add(x, y)
        // Graph B: computes add(z, const(10))
        // Merge: x and z are the same external tensor; B depends on A's output.

        let rng = &mut rand::rng();

        // External tensor IDs (symbolic graph IDs)
        let ext_x = GlobalId::new(rng);
        let ext_y = GlobalId::new(rng);
        let ext_sum_xy = GlobalId::new(rng); // output of graph A

        // Build graph A: add(x, y) -> sum_xy
        let (mut graph_a, a_input_map) =
            MilliOpGraph::new([ext_x, ext_y], rng);
        let a_x = a_input_map[&ext_x];
        let a_y = a_input_map[&ext_y];
        let a_out = SimpleBinary::add(&mut graph_a, a_x, a_y, rng);
        let mut a_output_map = HashMap::new();
        a_output_map.insert(a_out, ext_sum_xy);
        graph_a.set_output_map(a_output_map);

        // Build graph B: add(sum_xy, const(10)) -> final_out
        let ext_final = GlobalId::new(rng);
        let (mut graph_b, b_input_map) =
            MilliOpGraph::new([ext_sum_xy], rng);
        let b_in = b_input_map[&ext_sum_xy];
        let b_const = Constant::push_new(
            &mut graph_b,
            NDArrayNumericTensor::<DynRank>::from_vec_shape(vec![10.0f32], &vec![1]).unwrap(),
            rng,
        );
        let b_out = SimpleBinary::add(&mut graph_b, b_in, b_const, rng);
        let mut b_output_map = HashMap::new();
        b_output_map.insert(b_out, ext_final);
        graph_b.set_output_map(b_output_map);

        // Now merge into a combined graph
        let mut combined = MilliOpGraph::new_empty(rng);
        let mut sym_to_combined: HashMap<GlobalId, GlobalId> = HashMap::new();

        // Add external inputs
        let c_x = combined.add_input_with_id(ext_x, rng);
        sym_to_combined.insert(ext_x, c_x);
        let c_y = combined.add_input_with_id(ext_y, rng);
        sym_to_combined.insert(ext_y, c_y);

        // Merge graph A
        combined.merge_graph(graph_a, &mut sym_to_combined, rng);

        // After merging A, ext_sum_xy should be in sym_to_combined
        assert!(sym_to_combined.contains_key(&ext_sum_xy));

        // Merge graph B
        combined.merge_graph(graph_b, &mut sym_to_combined, rng);

        // After merging B, ext_final should be in sym_to_combined
        assert!(sym_to_combined.contains_key(&ext_final));

        // Set output
        combined.add_output(sym_to_combined[&ext_final], ext_final);

        // Evaluate: x=3, y=5 → add(3,5)=8 → add(8,10)=18
        let mut inputs = HashMap::new();
        inputs.insert(
            ext_x,
            NumericTensor::<DynRank>::from_vec_shape(vec![3.0f32], vec![1]).unwrap(),
        );
        inputs.insert(
            ext_y,
            NumericTensor::<DynRank>::from_vec_shape(vec![5.0f32], vec![1]).unwrap(),
        );

        let mut backend = EvalBackend::NDArray;
        let results: HashMap<_, _> = combined.eval(&inputs, &mut (), &mut backend).unwrap().collect();
        let result = &results[&ext_final];
        let values: Vec<f32> = result.flatten().unwrap().try_into().unwrap();
        assert_eq!(values, vec![18.0f32]);
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
