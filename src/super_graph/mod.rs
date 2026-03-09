pub mod cache;
pub mod data;
pub mod links;
pub mod nodes;
pub mod observer;

use crate::backends::eval_backend::{EvalBackend, EvalRuntimeError};
use crate::compiler::{CompiledProgram, CompilerError};
use crate::graph::{GlobalId, Graph, Link};
use crate::milli_graph::MilliOpGraphError;
use crate::model::{Model, ModelError};
use crate::numeric_tensor::NumericTensorError;
use crate::numeric_tensor_typed::TypedNumericTensorError;
use crate::super_graph::cache::{SuperGraphCache, SuperGraphTensorCache};
use crate::super_graph::data::SuperGraphData;
use crate::super_graph::links::SuperGraphLinkTensorMap;
pub use crate::super_graph::links::{
    SuperGraphAnyLink, SuperGraphLinkHash, SuperGraphLinkString, SuperGraphLinkTensor,
    SuperGraphLinkTokenizer,
};
use crate::super_graph::nodes::{SuperGraphAnyNode, SuperGraphNode};
use crate::super_graph::observer::SuperGraphObserver;
use crate::symbolic_graph::SymbolicGraph;
use crate::tokenizer::TokenizerError;
use rand::{Rng, RngCore};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::Instant;

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
    #[error("Model not in compiled cache")]
    ModelNotCompiledError,
    #[error(transparent)]
    CompilerError(#[from] CompilerError),
    #[error("Missing link{0}")]
    MissingLinkError(String),
    #[error(transparent)]
    EvalRuntimeError(#[from] EvalRuntimeError),
}

pub type SuperGraphHash = u64;

pub struct SuperGraphContext<'short, 'model, 'c, 'd, T: SuperGraphObserver> {
    pub observer: &'short mut T,
    pub eval_backend: &'c mut EvalBackend<'d>,
    pub caches: Option<&'short mut SuperGraphCache>,
    pub super_graph_tensor_cache: &'short mut SuperGraphTensorCache<'model>,
    pub use_compiled_models: bool,
    pub symbolic_graphs: Vec<&'model SymbolicGraph>,
    pub compiled_models: Option<Vec<(&'model Model, &'short CompiledProgram)>>,
}

impl<'short, 'model, 'c, 'd, T: SuperGraphObserver> SuperGraphContext<'short, 'model, 'c, 'd, T> {
    /// Construct a context with only the required fields; caches, compiled
    /// models, and symbolic graphs default to empty/None.
    pub fn new(
        eval_backend: &'c mut EvalBackend<'d>,
        observer: &'short mut T,
        tensor_cache: &'short mut SuperGraphTensorCache<'model>,
    ) -> Self {
        Self {
            observer,
            eval_backend,
            caches: None,
            super_graph_tensor_cache: tensor_cache,
            use_compiled_models: false,
            symbolic_graphs: vec![],
            compiled_models: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraph {
    global_id: GlobalId,
    pub input_links: HashSet<SuperGraphAnyLink>,
    pub output_links: HashSet<SuperGraphAnyLink>,
    pub nodes: HashMap<GlobalId, SuperGraphAnyNode>,
    pub links_by_global_id: HashMap<GlobalId, SuperGraphAnyLink>,
}

impl SuperGraph {
    pub fn run<'short, 'model, 'c, 'd, T: SuperGraphObserver>(
        &'short self,
        data: SuperGraphData<'model>,
        context: &mut SuperGraphContext<'short, 'model, 'c, 'd, T>,
    ) -> Result<SuperGraphData<'model>, SuperGraphError> {
        self.eval(&[], data, context)
    }

    pub fn eval<'a, 'b, 'c, 'd, T: SuperGraphObserver>(
        &'a self,
        node_path: &[GlobalId],
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
                    for input in SuperGraphNode::inputs(op) {
                        match input {
                            SuperGraphAnyLink::Tensor(x) => {
                                if !data.tensors.contains_key(&x) {
                                    all_inputs_ready = false;
                                    break;
                                }
                            }
                            SuperGraphAnyLink::String(x) => {
                                if !data.strings.contains_key(&x) {
                                    all_inputs_ready = false;
                                    break;
                                }
                            }
                            SuperGraphAnyLink::TensorMap(x) => {
                                if !data.tensor_maps.contains_key(&x) {
                                    all_inputs_ready = false;
                                    break;
                                }
                            }
                            SuperGraphAnyLink::Tokenizer(x) => {
                                if !data.tokenizers.contains_key(&x) {
                                    all_inputs_ready = false;
                                    break;
                                }
                            }
                            SuperGraphAnyLink::Hash(x) => {
                                if !data.hashes.contains_key(&x) {
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
                let op = self.nodes.get(&op_id).unwrap();
                let start_instant = Instant::now();
                op.eval(&this_path, &mut data, context)?;
                this_path.push(op.global_id());
                let end_instant = Instant::now();
                context.observer.on_node_executed(
                    &this_path,
                    &op.op_kind(),
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
            links.extend(node.inputs());
            links.extend(node.outputs());
        }
        links
    }
}

pub struct SuperGraphBuilder {
    nodes: HashMap<GlobalId, SuperGraphAnyNode>,
}

impl SuperGraphBuilder {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
        }
    }

    pub fn add_node(&mut self, node: SuperGraphAnyNode) -> GlobalId {
        let id = node.global_id();
        self.nodes.insert(id, node);
        id
    }

    pub fn build(
        self,
        rng: &mut impl Rng,
        input_links: &[SuperGraphAnyLink],
        output_links: &[SuperGraphAnyLink],
    ) -> SuperGraph {
        // Validate that all input and output links are present in the graph
        let mut sourced_links = HashSet::new();
        let mut sinked_links = HashSet::new();
        sinked_links.extend(output_links.iter().cloned());
        sourced_links.extend(input_links.iter().cloned());

        for node in self.nodes.values() {
            for link in node.outputs() {
                if !sourced_links.insert(link) {
                    panic!("Link {link:?} is sourced multiple times");
                }
            }
            sinked_links.extend(node.inputs());
        }

        for link in sinked_links {
            if !sourced_links.contains(&link) {
                panic!("Link {link:?} is not sourced");
            }
        }

        let links_by_global_id = sourced_links
            .iter()
            .map(|link| (link.global_id(), *link))
            .collect::<HashMap<_, _>>();

        SuperGraph {
            global_id: GlobalId::new(rng),
            nodes: self.nodes,
            input_links: HashSet::from_iter(input_links.iter().cloned()),
            output_links: HashSet::from_iter(output_links.iter().cloned()),
            links_by_global_id,
        }
    }

    pub fn new_tensor_link(&mut self, rng: &mut impl RngCore) -> SuperGraphLinkTensor {
        SuperGraphLinkTensor::new(rng)
    }

    pub fn new_model_link(&mut self, rng: &mut impl RngCore) -> SuperGraphLinkTensorMap {
        SuperGraphLinkTensorMap::new(rng)
    }

    pub fn new_tokenizer_link(&mut self, rng: &mut impl RngCore) -> SuperGraphLinkTokenizer {
        SuperGraphLinkTokenizer::new(rng)
    }

    pub fn new_string_link(&mut self, rng: &mut impl RngCore) -> SuperGraphLinkString {
        SuperGraphLinkString::new(rng)
    }

    pub fn new_hash_link(&mut self, rng: &mut impl RngCore) -> SuperGraphLinkHash {
        SuperGraphLinkHash::new(rng)
    }
}

impl Default for SuperGraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl Link for SuperGraphAnyLink {
    fn global_id(&self) -> GlobalId {
        self.global_id()
    }
}

impl Graph for SuperGraph {
    type Error = ();
    type AnyNode = SuperGraphAnyNode;
    type AnyLink = SuperGraphAnyLink;

    fn global_id(&self) -> GlobalId {
        self.global_id
    }

    fn node_ids(&self) -> impl Iterator<Item = GlobalId> {
        self.nodes.keys().cloned()
    }

    fn inner_link_ids(&self) -> impl Iterator<Item = GlobalId> {
        let mut links = HashSet::new();
        for node in self.nodes.values() {
            links.extend(node.inputs().map(|x| x.global_id()));
            links.extend(node.outputs().map(|x| x.global_id()));
        }
        links.into_iter()
    }

    fn get_node_by_id(&self, id: &GlobalId) -> Option<&Self::AnyNode> {
        self.nodes.get(id)
    }

    fn get_link_by_id(&self, id: &GlobalId) -> Option<&Self::AnyLink> {
        self.links_by_global_id.get(id)
    }

    fn input_link_ids(&self) -> impl Iterator<Item = (GlobalId, GlobalId)> {
        self.input_links
            .iter()
            .map(|x| (x.global_id(), x.global_id()))
    }

    fn output_link_ids(&self) -> impl Iterator<Item = (GlobalId, GlobalId)> {
        self.output_links
            .iter()
            .map(|x| (x.global_id(), x.global_id()))
    }

    fn constant_link_ids(&self) -> impl Iterator<Item = GlobalId> {
        core::iter::empty()
    }
}
