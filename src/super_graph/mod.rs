mod links;
mod nodes;

use std::collections::HashMap;
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use whisper_tensor_import::onnx_graph::TokenizerInfo;
use crate::model::Model;
use crate::super_graph::links::{SuperGraphAnyLink, SuperGraphLinkId, SuperGraphLinkTensor};
use crate::super_graph::nodes::{SuperGraphAnyNode, SuperGraphNodeModelExecution, SuperGraphNodeModelLoad, SuperGraphNodeStringInput, SuperGraphNodeStringOutput, SuperGraphNodeTokenizerDecode, SuperGraphNodeTokenizerEncode, SuperGraphNodeTokenizerLoad};

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(crate) struct SuperGraphNodeId (pub(crate) u32);

#[derive(Clone, Debug, Serialize, Deserialize)]
struct SuperGraph {
    nodes: HashMap<SuperGraphNodeId, SuperGraphAnyNode>,
    links: Vec<SuperGraphAnyLink>
}

struct SuperGraphBuilder {
    next_node_id: u32,
    next_link_id: u32,
    nodes: HashMap<SuperGraphNodeId, SuperGraphAnyNode>,
    links: Vec<SuperGraphAnyLink>
}

impl SuperGraphBuilder {
    fn new() -> Self {
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

    fn get_next_link_id(&mut self) -> SuperGraphLinkId {
        let id = self.next_link_id;
        self.next_link_id += 1;
        SuperGraphLinkId(id)
    }

    fn add_node(&mut self, node: SuperGraphAnyNode) -> SuperGraphNodeId {
        let id = self.get_next_node_id();
        self.nodes.insert(id.clone(), node);
        id
    }

    fn build(self) -> SuperGraph {
        SuperGraph {
            nodes: self.nodes,
            links: self.links
        }
    }
}

fn build_basic_supergraph() -> SuperGraph {
    let mut builder = SuperGraphBuilder::new();

    let model_load = SuperGraphNodeModelLoad::new_and_add(&mut builder, "test".to_string());

    let text_input = SuperGraphNodeStringInput::new_and_add(&mut builder);

    let tokenizer_link = SuperGraphNodeTokenizerLoad::new_and_add(&mut builder, TokenizerInfo::RWKVWorld);

    let tokens = SuperGraphNodeTokenizerEncode::new_and_add(&mut builder, tokenizer_link.clone(), text_input);

    let logit_output = {
        let inputs = {
            let mut inputs = HashMap::new();
            inputs.insert(tokens.clone(), "tokens".to_string());
            inputs
        };
        let (outputs, logit_output) = {
            let mut outputs = HashMap::new();
            let tensor = SuperGraphLinkTensor::new(builder.get_next_link_id());
            outputs.insert("logits".to_string(), tensor.clone());
            (outputs, tensor)
        };
        let node = SuperGraphNodeModelExecution::new(model_load, inputs, outputs);
        builder.add_node(node.into());
        logit_output
    };

    let text_output = SuperGraphNodeTokenizerDecode::new_and_add(&mut builder, tokenizer_link, logit_output);

    SuperGraphNodeStringOutput::new_and_add(&mut builder, text_output);

    builder.build()
}