pub mod links;
pub mod nodes;

use std::collections::HashMap;
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use crate::backends::eval_backend::EvalBackend;
use crate::DynRank;
use crate::milli_graph::MilliOpGraphError;
use crate::model::{Model, ModelError};
use crate::numeric_tensor::{NumericTensor, NumericTensorError};
use crate::numeric_tensor_typed::TypedNumericTensorError;
use crate::super_graph::links::{SuperGraphAnyLink, SuperGraphLinkId, SuperGraphLinkModel, SuperGraphLinkString, SuperGraphLinkTensor, SuperGraphLinkTokenizer};
use crate::super_graph::nodes::{SuperGraphAnyNode};
use crate::tokenizer::{AnyTokenizer, TokenizerError};

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SuperGraphNodeId (pub(crate) u32);

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
}

struct SuperGraphData {
    tensors: HashMap<SuperGraphLinkTensor, NumericTensor<DynRank>>,
    strings: HashMap<SuperGraphLinkString, String>,
    tokenizers: HashMap<SuperGraphLinkTokenizer, AnyTokenizer>,
    models: HashMap<SuperGraphLinkModel, Arc<Model>>,
    text_input: String,
    text_output: Option<String>,
    loaded_models: Vec<Arc<Model>>
}

impl SuperGraphData {
    pub fn new(text_input: String, loaded_models: Vec<Arc<Model>>) -> Self {
        Self {
            tensors: HashMap::new(),
            strings: HashMap::new(),
            tokenizers: HashMap::new(),
            models: HashMap::new(),
            text_input,
            text_output: None,
            loaded_models
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraph {
    nodes: HashMap<SuperGraphNodeId, SuperGraphAnyNode>,
    links: Vec<SuperGraphAnyLink>
}

impl SuperGraph {
    pub fn run(&self, models: &[Arc<Model>], input: String, backend: &mut EvalBackend) -> Result<Option<String>, SuperGraphError> {
        let mut data = SuperGraphData::new(input, models.to_vec());

        let mut remaining_ops = self.nodes.keys().map(|x| x.clone()).collect::<Vec<_>>();

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
                                }
                            }
                            SuperGraphAnyLink::String(x) => {
                                if !data.strings.contains_key(x) {
                                    all_inputs_ready = false;
                                }
                            }
                            SuperGraphAnyLink::Model(x) => {
                                if !data.models.contains_key(x) {
                                    all_inputs_ready = false;
                                }
                            }
                            SuperGraphAnyLink::Tokenizer(x) => {
                                if !data.tokenizers.contains_key(x) {
                                    all_inputs_ready = false;
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
                op.eval(&mut data, backend)?;
                remaining_ops.retain(|x| *x != op_id);
            }
            else {
                break;
            }
        }

        Ok(data.text_output)
    }
}

pub struct SuperGraphBuilder {
    next_node_id: u32,
    next_link_id: u32,
    nodes: HashMap<SuperGraphNodeId, SuperGraphAnyNode>,
    links: Vec<SuperGraphAnyLink>
}

impl SuperGraphBuilder {
    pub fn new() -> Self {
        Self {
            next_node_id: 0,
            next_link_id: 0,
            nodes: HashMap::new(),
            links: Vec::new()
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

    pub fn build(self) -> SuperGraph {
        SuperGraph {
            nodes: self.nodes,
            links: self.links
        }
    }
}