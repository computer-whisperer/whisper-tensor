pub mod cache;
pub mod data;
pub mod links;
pub mod nodes;
pub mod observer;

use crate::backends::eval_backend::EvalBackend;
use crate::milli_graph::{
    MilliOpGraphError, MilliOpGraphNodePath, MilliOpGraphTensorId, MilliOpGraphTensorPath,
};
use crate::model::ModelError;
use crate::numeric_tensor::NumericTensorError;
use crate::numeric_tensor_typed::TypedNumericTensorError;
use crate::super_graph::cache::{SuperGraphCache, SuperGraphTensorCache};
use crate::super_graph::data::SuperGraphData;
pub use crate::super_graph::links::{
    SuperGraphAnyLink, SuperGraphLinkHash, SuperGraphLinkId, SuperGraphLinkModel,
    SuperGraphLinkString, SuperGraphLinkTensor, SuperGraphLinkTokenizer,
};
use crate::super_graph::nodes::SuperGraphAnyNode;
use crate::super_graph::observer::SuperGraphObserver;
use crate::symbolic_graph::{
    SymbolicGraphNodePath, SymbolicGraphOperationId, SymbolicGraphTensorId, SymbolicGraphTensorPath,
};
use crate::tokenizer::TokenizerError;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::Instant;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SuperGraphNodeId(pub(crate) u32);

#[derive(Debug, thiserror::Error)]
pub enum SuperGraphError {
    #[error(transparent)]
    ModelError(#[from] ModelError),
    #[error(transparent)]
    TokenizerError(#[from] TokenizerError),
    #[error(transparent)]
    MilliOpGraphError(#[from] MilliOpGraphError),
    #[error(transparent)]
    NumericTensorError(#[from] NumericTensorError),
    #[error(transparent)]
    TypedNumericTensorError(#[from] TypedNumericTensorError),
    #[error("Missing link")]
    MissingLinkError(),
}

pub type SuperGraphHash = u64;

pub struct SuperGraphContext<'short, 'model, 'c, 'd, T: SuperGraphObserver> {
    pub observer: &'short mut T,
    pub eval_backend: &'c mut EvalBackend<'d>,
    pub caches: Option<&'short mut SuperGraphCache>,
    pub super_graph_tensor_cache: &'short mut SuperGraphTensorCache<'model>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraph {
    pub inner: SuperGraphInner,
}

impl SuperGraph {
    pub fn run<'short, 'model, 'c, 'd, T: SuperGraphObserver>(
        &'short self,
        data: SuperGraphData<'model>,
        context: &mut SuperGraphContext<'short, 'model, 'c, 'd, T>,
    ) -> Result<SuperGraphData<'model>, SuperGraphError> {
        self.inner.eval(&[], data, context)
    }

    pub fn get_all_links(&self) -> HashSet<SuperGraphAnyLink> {
        self.inner.get_all_links()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphInner {
    pub input_links: HashSet<SuperGraphAnyLink>,
    pub output_links: HashSet<SuperGraphAnyLink>,
    pub nodes: HashMap<SuperGraphNodeId, SuperGraphAnyNode>,
}

impl SuperGraphInner {
    pub fn eval<'a, 'b, 'c, 'd, T: SuperGraphObserver>(
        &'a self,
        node_path: &[SuperGraphNodeId],
        data: SuperGraphData<'b>,
        context: &mut SuperGraphContext<'a, 'b, 'c, 'd, T>,
    ) -> Result<SuperGraphData<'b>, SuperGraphError> {
        let mut data = data;

        let mut remaining_ops = self.nodes.keys().cloned().collect::<Vec<_>>();

        loop {
            let op_id_to_use = {
                let mut op_id_to_use = None;
                for op_id in &remaining_ops {
                    let op = self.nodes.get(op_id).unwrap();
                    let mut all_inputs_ready = true;
                    for input in &op.get_inputs() {
                        match input {
                            SuperGraphAnyLink::Tensor(x) => {
                                if !data.tensors.contains_key(x) {
                                    all_inputs_ready = false;
                                    break;
                                }
                            }
                            SuperGraphAnyLink::String(x) => {
                                if !data.strings.contains_key(x) {
                                    all_inputs_ready = false;
                                    break;
                                }
                            }
                            SuperGraphAnyLink::Model(x) => {
                                if !data.models.contains_key(x) {
                                    all_inputs_ready = false;
                                    break;
                                }
                            }
                            SuperGraphAnyLink::Tokenizer(x) => {
                                if !data.tokenizers.contains_key(x) {
                                    all_inputs_ready = false;
                                    break;
                                }
                            }
                            SuperGraphAnyLink::Hash(x) => {
                                if !data.hashes.contains_key(x) {
                                    all_inputs_ready = false;
                                    break;
                                }
                            }
                        }
                    }
                    if all_inputs_ready {
                        op_id_to_use = Some(op_id);
                        break;
                    }
                }
                op_id_to_use
            };
            if let Some(op_id) = op_id_to_use {
                let op_id = *op_id;
                let mut this_path = node_path.to_vec();
                this_path.push(op_id);
                let op = self.nodes.get(&op_id).unwrap();
                let start_instant = Instant::now();
                op.eval(&this_path, &mut data, context)?;
                let end_instant = Instant::now();
                context.observer.on_node_executed(
                    &SuperGraphNodePath::SuperGraphNode(this_path),
                    start_instant,
                    end_instant,
                    context.eval_backend,
                );
                remaining_ops.retain(|x| *x != op_id);
            } else {
                break;
            }
        }

        let output_data = data.select(
            self.output_links
                .iter()
                .cloned()
                .collect::<Vec<_>>()
                .as_slice(),
        )?;

        Ok(output_data)
    }

    pub fn get_all_links(&self) -> HashSet<SuperGraphAnyLink> {
        let mut links = HashSet::new();
        links.extend(self.input_links.iter().cloned());
        links.extend(self.output_links.iter().cloned());
        for node in self.nodes.values() {
            links.extend(node.get_inputs());
            links.extend(node.get_outputs());
        }
        links
    }
}

pub struct SuperGraphBuilder {
    next_node_id: u32,
    next_link_id: u32,
    nodes: HashMap<SuperGraphNodeId, SuperGraphAnyNode>,
}

impl SuperGraphBuilder {
    pub fn new() -> Self {
        Self {
            next_node_id: 0,
            next_link_id: 0,
            nodes: HashMap::new(),
        }
    }

    fn get_next_node_id(&mut self) -> SuperGraphNodeId {
        let id = self.next_node_id;
        self.next_node_id += 1;
        SuperGraphNodeId(id)
    }

    pub fn get_next_link_id(&mut self) -> SuperGraphLinkId {
        let id = self.next_link_id;
        self.next_link_id += 1;
        SuperGraphLinkId(id)
    }

    pub fn add_node(&mut self, node: SuperGraphAnyNode) -> SuperGraphNodeId {
        let id = self.get_next_node_id();
        self.nodes.insert(id, node);
        id
    }

    pub fn build(
        self,
        input_links: &[SuperGraphAnyLink],
        output_links: &[SuperGraphAnyLink],
    ) -> SuperGraph {
        SuperGraph {
            inner: Self::build_inner(self, input_links, output_links),
        }
    }

    pub fn build_inner(
        self,
        input_links: &[SuperGraphAnyLink],
        output_links: &[SuperGraphAnyLink],
    ) -> SuperGraphInner {
        // Validate that all input and output links are present in the graph
        let mut sourced_links = HashSet::new();
        let mut sinked_links = HashSet::new();
        sinked_links.extend(output_links.iter().cloned());
        sourced_links.extend(input_links.iter().cloned());

        for node in self.nodes.values() {
            for link in node.get_outputs() {
                if !sourced_links.insert(link) {
                    panic!("Link {link:?} is sourced multiple times");
                }
            }
            sinked_links.extend(node.get_inputs().iter().cloned());
        }

        for link in sinked_links {
            if !sourced_links.contains(&link) {
                panic!("Link {link:?} is not sourced");
            }
        }

        SuperGraphInner {
            nodes: self.nodes,
            input_links: HashSet::from_iter(input_links.iter().cloned()),
            output_links: HashSet::from_iter(output_links.iter().cloned()),
        }
    }

    pub fn new_tensor_link(&mut self) -> SuperGraphLinkTensor {
        SuperGraphLinkTensor::new(self.get_next_link_id())
    }

    pub fn new_model_link(&mut self) -> SuperGraphLinkModel {
        SuperGraphLinkModel::new(self.get_next_link_id())
    }

    pub fn new_tokenizer_link(&mut self) -> SuperGraphLinkTokenizer {
        SuperGraphLinkTokenizer::new(self.get_next_link_id())
    }

    pub fn new_string_link(&mut self) -> SuperGraphLinkString {
        SuperGraphLinkString::new(self.get_next_link_id())
    }

    pub fn new_hash_link(&mut self) -> SuperGraphLinkHash {
        SuperGraphLinkHash::new(self.get_next_link_id())
    }
}

impl Default for SuperGraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Hash, PartialEq, Eq)]
pub enum SuperGraphTensorPath {
    SuperGraphLink(Vec<SuperGraphNodeId>, SuperGraphLinkTensor),
    SymbolicGraphTensor(Vec<SuperGraphNodeId>, SymbolicGraphTensorPath),
    MilliOpGraphTensor(Vec<SuperGraphNodeId>, MilliOpGraphTensorPath),
}

impl SuperGraphTensorPath {
    pub fn prepend_super_graph_node(&self, node: SuperGraphNodeId) -> Self {
        let mut path = vec![node];
        match self {
            SuperGraphTensorPath::SuperGraphLink(a, b) => {
                path.extend(a);
                SuperGraphTensorPath::SuperGraphLink(path, *b)
            }
            SuperGraphTensorPath::SymbolicGraphTensor(a, b) => {
                path.extend(a);
                SuperGraphTensorPath::SymbolicGraphTensor(path, b.clone())
            }
            SuperGraphTensorPath::MilliOpGraphTensor(a, b) => {
                path.extend(a);
                SuperGraphTensorPath::MilliOpGraphTensor(path, b.clone())
            }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Hash, PartialEq, Eq)]
pub enum SuperGraphNodePath {
    SuperGraphNode(Vec<SuperGraphNodeId>),
    SymbolicGraphNode(Vec<SuperGraphNodeId>, SymbolicGraphNodePath),
    MilliOpGraphNode(Vec<SuperGraphNodeId>, MilliOpGraphNodePath),
}

impl SuperGraphNodePath {
    pub fn prepend_super_graph_node(&self, node: SuperGraphNodeId) -> Self {
        let mut path = vec![node];
        match self {
            SuperGraphNodePath::SuperGraphNode(a) => {
                path.extend(a);
                SuperGraphNodePath::SuperGraphNode(path)
            }
            SuperGraphNodePath::SymbolicGraphNode(a, b) => {
                path.extend(a);
                SuperGraphNodePath::SymbolicGraphNode(path, b.clone())
            }
            SuperGraphNodePath::MilliOpGraphNode(a, b) => {
                path.extend(a);
                SuperGraphNodePath::MilliOpGraphNode(path, b.clone())
            }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Hash, PartialEq, Eq)]
pub enum SuperGraphGraphPath {
    SymbolicGraph(Vec<SuperGraphNodeId>),
    SuperGraph(Vec<SuperGraphNodeId>),
    MilliOpGraph(Vec<SuperGraphNodeId>),
}

impl SuperGraphGraphPath {
    pub fn push_symbolic_node(&self, node: SymbolicGraphOperationId) -> SuperGraphNodePath {
        if let SuperGraphGraphPath::SymbolicGraph(path) = self {
            SuperGraphNodePath::SymbolicGraphNode(path.clone(), SymbolicGraphNodePath::Node(node))
        } else {
            panic!();
        }
    }

    pub fn push_symbolic_tensor(&self, tensor: SymbolicGraphTensorId) -> SuperGraphTensorPath {
        if let SuperGraphGraphPath::SymbolicGraph(path) = self {
            SuperGraphTensorPath::SymbolicGraphTensor(
                path.clone(),
                SymbolicGraphTensorPath::Tensor(tensor),
            )
        } else {
            panic!();
        }
    }

    pub fn push_super_node(&self, node: SuperGraphNodeId) -> SuperGraphNodePath {
        if let SuperGraphGraphPath::SuperGraph(a) = self {
            let mut path = a.clone();
            path.push(node);
            SuperGraphNodePath::SuperGraphNode(path)
        } else {
            panic!();
        }
    }

    pub fn push_super_node_graph(&self, node: SuperGraphNodeId) -> SuperGraphGraphPath {
        if let SuperGraphGraphPath::SuperGraph(a) = self {
            let mut path = a.clone();
            path.push(node);
            SuperGraphGraphPath::SuperGraph(path)
        } else {
            panic!();
        }
    }

    pub fn push_model_execution_node_graph(&self, node: SuperGraphNodeId) -> SuperGraphGraphPath {
        if let SuperGraphGraphPath::SuperGraph(a) = self {
            let mut path = a.clone();
            path.push(node);
            SuperGraphGraphPath::SymbolicGraph(path)
        } else {
            panic!();
        }
    }

    pub fn push_super_milli_op_node_graph(&self, node: SuperGraphNodeId) -> SuperGraphGraphPath {
        if let SuperGraphGraphPath::SuperGraph(a) = self {
            let mut path = a.clone();
            path.push(node);
            SuperGraphGraphPath::MilliOpGraph(path)
        } else {
            panic!();
        }
    }

    pub fn push_super_tensor(&self, link: SuperGraphLinkTensor) -> SuperGraphTensorPath {
        if let SuperGraphGraphPath::SuperGraph(a) = self {
            SuperGraphTensorPath::SuperGraphLink(a.clone(), link)
        } else {
            panic!();
        }
    }

    pub fn push_milli_op_node(&self, id: MilliOpGraphTensorId) -> SuperGraphNodePath {
        if let SuperGraphGraphPath::MilliOpGraph(path) = self {
            SuperGraphNodePath::MilliOpGraphNode(path.clone(), MilliOpGraphNodePath::Op(id))
        } else {
            panic!();
        }
    }

    pub fn push_milli_op_tensor(&self, id: MilliOpGraphTensorId) -> SuperGraphTensorPath {
        if let SuperGraphGraphPath::MilliOpGraph(path) = self {
            SuperGraphTensorPath::MilliOpGraphTensor(
                path.clone(),
                MilliOpGraphTensorPath::Tensor(id),
            )
        } else {
            panic!();
        }
    }
}
