use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use whisper_tensor_import::onnx_graph::TokenizerInfo;
use crate::milli_graph::MilliOpGraph;
use crate::super_graph::links::{SuperGraphAnyLink, SuperGraphLink, SuperGraphLinkModel, SuperGraphLinkString, SuperGraphLinkTensor, SuperGraphLinkTokenizer};
use crate::super_graph::SuperGraphBuilder;

trait SuperGraphNode {
    fn get_inputs(&self) -> Vec<SuperGraphAnyLink>;
    fn get_outputs(&self) -> Vec<SuperGraphAnyLink>;
    fn to_any(self) -> SuperGraphAnyNode;
}

impl <T: SuperGraphNode> From<T> for SuperGraphAnyNode {
    fn from(value: T) -> Self {
        value.to_any()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelReference {
    name: String
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeModelLoad {
    model: ModelReference,
    output: SuperGraphLinkModel
}

impl SuperGraphNodeModelLoad {
    pub fn new(builder: &mut SuperGraphBuilder, name: String) -> Self {
        Self {
            model: ModelReference { name },
            output: SuperGraphLinkModel::new(builder.get_next_link_id())
        }
    }

    pub fn new_and_add(builder: &mut SuperGraphBuilder, name: String) -> SuperGraphLinkModel {
        let node = Self::new(builder, name);
        let output = node.get_model_output();
        builder.add_node(node.into());
        output
    }

    pub fn get_model_output(&self) -> SuperGraphLinkModel {
        self.output.clone()
    }
}

impl SuperGraphNode for SuperGraphNodeModelLoad {
    fn get_inputs(&self) -> Vec<SuperGraphAnyLink> {
        vec![]
    }
    fn get_outputs(&self) -> Vec<SuperGraphAnyLink> {
        vec![self.output.to_any()]
    }
    fn to_any(self) -> SuperGraphAnyNode {
        SuperGraphAnyNode::ModelLoad(self)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeModelExecution {
    model: SuperGraphLinkModel,
    tensor_inputs: HashMap<SuperGraphLinkTensor, String>,
    tensor_outputs: HashMap<String, SuperGraphLinkTensor>
}

impl SuperGraphNodeModelExecution {
    pub fn new(model: SuperGraphLinkModel, tensor_inputs: HashMap<SuperGraphLinkTensor, String>, tensor_outputs: HashMap<String, SuperGraphLinkTensor>) -> Self {
        Self {
            model,
            tensor_inputs,
            tensor_outputs
        }
    }
}

impl SuperGraphNode for SuperGraphNodeModelExecution {
    fn get_inputs(&self) -> Vec<SuperGraphAnyLink> {
        self.tensor_inputs.keys().map(|x| x.to_any()).collect()
    }
    fn get_outputs(&self) -> Vec<SuperGraphAnyLink> {
        self.tensor_outputs.values().map(|x| x.to_any()).collect()
    }
    fn to_any(self) -> SuperGraphAnyNode {
        SuperGraphAnyNode::ModelExecution(self)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeTokenizerLoad {
    info: TokenizerInfo,
    output: SuperGraphLinkTokenizer
}

impl SuperGraphNodeTokenizerLoad {
    pub fn new(builder: &mut SuperGraphBuilder, info: TokenizerInfo) -> Self {
        Self {
            info,
            output: SuperGraphLinkTokenizer::new(builder.get_next_link_id())
        }
    }

    pub fn new_and_add(builder: &mut SuperGraphBuilder, info: TokenizerInfo) -> SuperGraphLinkTokenizer {
        let node = Self::new(builder, info);
        let output = node.get_tokenizer_output();
        builder.add_node(node.into());
        output
    }

    pub fn get_tokenizer_output(&self) -> SuperGraphLinkTokenizer {
        self.output.clone()
    }
}

impl SuperGraphNode for SuperGraphNodeTokenizerLoad {
    fn get_inputs(&self) -> Vec<SuperGraphAnyLink> {
        vec![]
    }
    fn get_outputs(&self) -> Vec<SuperGraphAnyLink> {
        vec![self.output.to_any()]
    }
    fn to_any(self) -> SuperGraphAnyNode {
        SuperGraphAnyNode::TokenizerLoad(self)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeTokenizerEncode {
    tokenizer: SuperGraphLinkTokenizer,
    text_input: SuperGraphLinkString,
    tensor_output: SuperGraphLinkTensor
}

impl SuperGraphNodeTokenizerEncode {
    pub fn new(builder: &mut SuperGraphBuilder, tokenizer: SuperGraphLinkTokenizer, text_input: SuperGraphLinkString) -> Self {
        Self {
            tokenizer,
            text_input,
            tensor_output: SuperGraphLinkTensor::new(builder.get_next_link_id())
        }
    }

    pub fn new_and_add(builder: &mut SuperGraphBuilder, tokenizer: SuperGraphLinkTokenizer, text_input: SuperGraphLinkString) -> SuperGraphLinkTensor {
        let node = Self::new(builder, tokenizer, text_input);
        let output = node.get_tensor_output();
        builder.add_node(node.into());
        output
    }

    pub fn get_tensor_output(&self) -> SuperGraphLinkTensor {
        self.tensor_output.clone()
    }
}

impl SuperGraphNode for SuperGraphNodeTokenizerEncode {
    fn get_inputs(&self) -> Vec<SuperGraphAnyLink> {
        vec![self.tokenizer.to_any(), self.text_input.to_any()]
    }

    fn get_outputs(&self) -> Vec<SuperGraphAnyLink> {
        vec![self.tensor_output.to_any()]
    }

    fn to_any(self) -> SuperGraphAnyNode {
        SuperGraphAnyNode::TokenizerEncode(self)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeStringInput {
    text_output: SuperGraphLinkString
}

impl SuperGraphNodeStringInput {
    pub fn new(builder: &mut SuperGraphBuilder) -> Self {
        Self {
            text_output: SuperGraphLinkString::new(builder.get_next_link_id())
        }
    }

    pub fn new_and_add(builder: &mut SuperGraphBuilder) -> SuperGraphLinkString {
        let node = Self::new(builder);
        let output = node.get_string_output();
        builder.add_node(node.into());
        output
    }

    pub fn get_string_output(&self) -> SuperGraphLinkString {
        self.text_output.clone()
    }
}

impl SuperGraphNode for SuperGraphNodeStringInput {
    fn get_inputs(&self) -> Vec<SuperGraphAnyLink> {
        vec![]
    }
    fn get_outputs(&self) -> Vec<SuperGraphAnyLink> {
        vec![self.text_output.to_any()]
    }
    fn to_any(self) -> SuperGraphAnyNode {
        SuperGraphAnyNode::StringInput(self)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeStringOutput {
    text_input: SuperGraphLinkString
}

impl SuperGraphNodeStringOutput {
    pub fn new(text_input: SuperGraphLinkString) -> Self {
        Self {
            text_input
        }
    }

    pub fn new_and_add(builder: &mut SuperGraphBuilder, text_input: SuperGraphLinkString) {
        let node = Self::new(text_input);
        builder.add_node(node.into());
    }
}

impl SuperGraphNode for SuperGraphNodeStringOutput {
    fn get_inputs(&self) -> Vec<SuperGraphAnyLink> {
        vec![self.text_input.to_any()]
    }
    fn get_outputs(&self) -> Vec<SuperGraphAnyLink> {
        vec![]
    }
    fn to_any(self) -> SuperGraphAnyNode {
        SuperGraphAnyNode::StringOutput(self)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeTokenizerDecode {
    tokenizer: SuperGraphLinkTokenizer,
    tensor_input: SuperGraphLinkTensor,
    text_output: SuperGraphLinkString
}

impl SuperGraphNodeTokenizerDecode {
    pub fn new(builder: &mut SuperGraphBuilder, tokenizer: SuperGraphLinkTokenizer, tensor_input: SuperGraphLinkTensor) -> Self {
        Self {
            tokenizer,
            tensor_input,
            text_output: SuperGraphLinkString::new(builder.get_next_link_id())
        }
    }
    pub fn get_string_output(&self) -> SuperGraphLinkString {
        self.text_output.clone()
    }
    pub fn new_and_add(builder: &mut SuperGraphBuilder, tokenizer: SuperGraphLinkTokenizer, tensor_input: SuperGraphLinkTensor) -> SuperGraphLinkString {
        let node = Self::new(builder, tokenizer, tensor_input);
        let output = node.get_string_output();
        builder.add_node(node.to_any());
        output
    }
}

impl SuperGraphNode for SuperGraphNodeTokenizerDecode {
    fn get_inputs(&self) -> Vec<SuperGraphAnyLink> {
        vec![self.tensor_input.to_any()]
    }
    fn get_outputs(&self) -> Vec<SuperGraphAnyLink> {
        vec![self.text_output.to_any()]
    }
    fn to_any(self) -> SuperGraphAnyNode {
        SuperGraphAnyNode::TokenizerDecode(self)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeMilliOpGraph {
    graph: MilliOpGraph<SuperGraphLinkTensor>
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SuperGraphAnyNode {
    ModelExecution(SuperGraphNodeModelExecution),
    TokenizerEncode(SuperGraphNodeTokenizerEncode),
    TokenizerDecode(SuperGraphNodeTokenizerDecode),
    ModelLoad(SuperGraphNodeModelLoad),
    TokenizerLoad(SuperGraphNodeTokenizerLoad),
    StringInput(SuperGraphNodeStringInput),
    StringOutput(SuperGraphNodeStringOutput)
}

impl SuperGraphAnyNode {
    pub fn get_outputs(&self) -> Vec<SuperGraphAnyLink> {
        match self {
            SuperGraphAnyNode::ModelExecution(node) => node.get_outputs(),
            SuperGraphAnyNode::TokenizerEncode(node) => node.get_outputs(),
            SuperGraphAnyNode::TokenizerDecode(node) => node.get_outputs(),
            SuperGraphAnyNode::ModelLoad(node) => node.get_outputs(),
            SuperGraphAnyNode::TokenizerLoad(node) => node.get_outputs(),
            SuperGraphAnyNode::StringInput(node) => node.get_outputs(),
            SuperGraphAnyNode::StringOutput(node) => node.get_outputs(),
        }
    }

    pub fn get_inputs(&self) -> Vec<SuperGraphAnyLink> {
        match self {
            SuperGraphAnyNode::ModelExecution(node) => node.get_inputs(),
            SuperGraphAnyNode::TokenizerEncode(node) => node.get_inputs(),
            SuperGraphAnyNode::TokenizerDecode(node) => node.get_inputs(),
            SuperGraphAnyNode::ModelLoad(node) => node.get_inputs(),
            SuperGraphAnyNode::TokenizerLoad(node) => node.get_inputs(),
            SuperGraphAnyNode::StringInput(node) => node.get_inputs(),
            SuperGraphAnyNode::StringOutput(node) => node.get_inputs()
        }
    }
}