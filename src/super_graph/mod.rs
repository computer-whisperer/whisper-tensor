pub mod cache;
pub mod data;
pub mod links;
pub mod nodes;

use crate::backends::eval_backend::EvalBackend;
use crate::milli_graph::MilliOpGraphError;
use crate::model::ModelError;
use crate::numeric_tensor::NumericTensorError;
use crate::numeric_tensor_typed::TypedNumericTensorError;
use crate::super_graph::cache::SuperGraphCache;
use crate::super_graph::data::SuperGraphData;
use crate::super_graph::links::{
    SuperGraphAnyLink, SuperGraphLinkHash, SuperGraphLinkId, SuperGraphLinkModel,
    SuperGraphLinkString, SuperGraphLinkTensor, SuperGraphLinkTokenizer,
};
use crate::super_graph::nodes::SuperGraphAnyNode;
use crate::tokenizer::TokenizerError;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraph {
    inner: SuperGraphInner,
}

impl SuperGraph {
    pub fn run<'a>(
        &'a self,
        data: SuperGraphData<'a>,
        caches: Option<&mut SuperGraphCache>,
        backend: &mut EvalBackend,
    ) -> Result<SuperGraphData<'a>, SuperGraphError> {
        self.inner.eval(data, caches, backend)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphInner {
    input_links: HashSet<SuperGraphAnyLink>,
    output_links: HashSet<SuperGraphAnyLink>,
    nodes: HashMap<SuperGraphNodeId, SuperGraphAnyNode>,
}

impl SuperGraphInner {
    pub fn eval<'a>(
        &'a self,
        data: SuperGraphData<'a>,
        mut caches: Option<&mut SuperGraphCache>,
        backend: &mut EvalBackend,
    ) -> Result<SuperGraphData<'a>, SuperGraphError> {
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
                        op_id_to_use = Some(op_id.clone());
                        break;
                    }
                }
                op_id_to_use
            };
            if let Some(op_id) = op_id_to_use {
                let op = self.nodes.get(&op_id).unwrap();
                op.eval(&mut data, caches.as_deref_mut(), backend)?;
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
        self.nodes.insert(id.clone(), node);
        id
    }

    pub fn build(
        self,
        input_links: &[SuperGraphAnyLink],
        output_links: &[SuperGraphAnyLink],
    ) -> SuperGraph {
        SuperGraph {
            inner: SuperGraphInner {
                nodes: self.nodes,
                input_links: HashSet::from_iter(input_links.iter().cloned()),
                output_links: HashSet::from_iter(output_links.iter().cloned()),
            },
        }
    }

    pub fn build_inner(
        self,
        input_links: &[SuperGraphAnyLink],
        output_links: &[SuperGraphAnyLink],
    ) -> SuperGraphInner {
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
