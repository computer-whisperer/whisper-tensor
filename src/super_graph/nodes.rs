use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use typenum::P1;
use rwkv_tokenizer::WorldTokenizer;
use whisper_tensor_import::onnx_graph::TokenizerInfo;
use crate::backends::eval_backend::EvalBackend;
use crate::milli_graph::MilliOpGraph;
use crate::milli_graph::ops::MilliOp;
use crate::numeric_tensor::NumericTensor;
use crate::super_graph::links::{SuperGraphAnyLink, SuperGraphLink, SuperGraphLinkModel, SuperGraphLinkString, SuperGraphLinkTensor, SuperGraphLinkTokenizer};
use crate::super_graph::{SuperGraphBuilder, SuperGraphData, SuperGraphError};
use crate::tokenizer::{AnyTokenizer, Tokenizer};

trait SuperGraphNode {
    fn get_inputs(&self) -> Vec<SuperGraphAnyLink>;
    fn get_outputs(&self) -> Vec<SuperGraphAnyLink>;
    fn to_any(self) -> SuperGraphAnyNode;
    fn eval(&self, data: &mut SuperGraphData, backend: &mut EvalBackend) -> Result<(), SuperGraphError>;
}

impl <T: SuperGraphNode> From<T> for SuperGraphAnyNode {
    fn from(value: T) -> Self {
        value.to_any()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelReference {
    idx: usize
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeModelLoad {
    model: ModelReference,
    output: SuperGraphLinkModel
}

impl SuperGraphNodeModelLoad {
    pub fn new(builder: &mut SuperGraphBuilder, idx: usize) -> Self {
        Self {
            model: ModelReference { idx },
            output: SuperGraphLinkModel::new(builder.get_next_link_id())
        }
    }

    pub fn new_and_add(builder: &mut SuperGraphBuilder, idx: usize) -> SuperGraphLinkModel {
        let node = Self::new(builder, idx);
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
    fn eval(&self, data: &mut SuperGraphData, _backend: &mut EvalBackend) -> Result<(), SuperGraphError> {
        let model = data.loaded_models.get(self.model.idx).unwrap();
        data.models.insert(self.output.clone(), model.clone());
        Ok(())
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
        let mut ret: Vec<_> = self.tensor_inputs.keys().map(|x| x.to_any()).collect();
        ret.push(self.model.to_any());
        ret
    }
    fn get_outputs(&self) -> Vec<SuperGraphAnyLink> {
        self.tensor_outputs.values().map(|x| x.to_any()).collect()
    }
    fn to_any(self) -> SuperGraphAnyNode {
        SuperGraphAnyNode::ModelExecution(self)
    }
    fn eval(&self, data: &mut SuperGraphData, backend: &mut EvalBackend) -> Result<(), SuperGraphError> {
        let model = data.models.get(&self.model).unwrap();

        let inputs = {
            let mut inputs = HashMap::new();
            for (link, name) in &self.tensor_inputs {
                inputs.insert(name.clone(), data.tensors.get(link).unwrap().clone());
            }
            inputs
        };
        let outputs = model.eval(inputs, backend)?;
        for (name, link) in &self.tensor_outputs {
            data.tensors.insert(link.clone(), outputs.get(name).unwrap().clone());
        }
        Ok(())
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
    fn eval(&self, data: &mut SuperGraphData, _backend: &mut EvalBackend) -> Result<(), SuperGraphError> {
        let tokenizer = match &self.info {
            TokenizerInfo::HFTokenizer(name) => {
                #[cfg(all(feature = "tokenizers", feature = "http"))]
                {
                    AnyTokenizer::Tokenizers(tokenizers::Tokenizer::from_pretrained(name, None).unwrap())
                }
            }
            TokenizerInfo::RWKVWorld => {
                #[cfg(feature = "rwkv-tokenizer")]
                {
                    AnyTokenizer::Rwkv(WorldTokenizer::new(None).unwrap())
                }
            }
        };
        data.tokenizers.insert(self.output.clone(), tokenizer);
        Ok(())
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
    fn eval(&self, data: &mut SuperGraphData, _backend: &mut EvalBackend) -> Result<(), SuperGraphError> {
        let text = data.strings.get(&self.text_input).unwrap();
        let tokenizer = data.tokenizers.get(&self.tokenizer).unwrap();
        let tokens = tokenizer.encode(text).iter().map(|x| *x as i64).collect::<Vec<_>>();;
        let input_tensor = NumericTensor::from_vec(tokens).to_dyn_rank().unsqueeze(0).unwrap().unsqueeze(0).unwrap();
        data.tensors.insert(self.tensor_output.clone(), input_tensor);
        Ok(())
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
    fn eval(&self, data: &mut SuperGraphData, _backend: &mut EvalBackend) -> Result<(), SuperGraphError> {
        data.strings.insert(self.text_output.clone(), data.text_input.clone());
        Ok(())
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
    fn eval(&self, data: &mut SuperGraphData, _backend: &mut EvalBackend) -> Result<(), SuperGraphError> {
        let string = data.strings.get(&self.text_input).unwrap();
        data.text_output = Some(string.clone());
        Ok(())
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
    fn eval(&self, data: &mut SuperGraphData, _backend: &mut EvalBackend) -> Result<(), SuperGraphError> {
        let tensor = data.tensors.get(&self.tensor_input).unwrap();
        let tokenizer = data.tokenizers.get(&self.tokenizer).unwrap();
        let tensor = tensor.try_to_type::<u32>()?.try_to_rank::<P1>()?;
        let output_values: Vec<u32> = tensor.to_vec();
        let text = tokenizer.decode(&output_values)?;
        data.strings.insert(self.text_output.clone(), text);
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeMilliOpGraph {
    graph: MilliOpGraph<SuperGraphLinkTensor>
}

impl SuperGraphNodeMilliOpGraph {
    pub fn new(graph: MilliOpGraph<SuperGraphLinkTensor>) -> Self {
        Self {
            graph
        }
    }
}

impl SuperGraphNode for SuperGraphNodeMilliOpGraph {
    fn get_inputs(&self) -> Vec<SuperGraphAnyLink> {
        self.graph.get_inputs().iter().map(|x| x.to_any()).collect()
    }
    fn get_outputs(&self) -> Vec<SuperGraphAnyLink> {
        self.graph.get_outputs().iter().map(|x| x.to_any()).collect()
    }
    fn to_any(self) -> SuperGraphAnyNode {
        SuperGraphAnyNode::MilliOpGraph(self)
    }
    fn eval(&self, data: &mut SuperGraphData, backend: &mut EvalBackend) -> Result<(), SuperGraphError> {
        let inputs = {
            let mut inputs = HashMap::new();
            for input in &self.graph.get_inputs() {
                inputs.insert(input.clone(), data.tensors.get(&input).unwrap().clone());
            }
            inputs
        };
        let res = self.graph.eval(&inputs, backend)?;
        data.tensors.extend(res);
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SuperGraphAnyNode {
    ModelExecution(SuperGraphNodeModelExecution),
    TokenizerEncode(SuperGraphNodeTokenizerEncode),
    TokenizerDecode(SuperGraphNodeTokenizerDecode),
    ModelLoad(SuperGraphNodeModelLoad),
    TokenizerLoad(SuperGraphNodeTokenizerLoad),
    StringInput(SuperGraphNodeStringInput),
    StringOutput(SuperGraphNodeStringOutput),
    MilliOpGraph(SuperGraphNodeMilliOpGraph),
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
            SuperGraphAnyNode::MilliOpGraph(node) => node.get_outputs()
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
            SuperGraphAnyNode::StringOutput(node) => node.get_inputs(),
            SuperGraphAnyNode::MilliOpGraph(node) => node.get_inputs()
        }
    }

    pub(crate) fn eval(&self, data: &mut SuperGraphData, backend: &mut EvalBackend) -> Result<(), SuperGraphError> {
        match self {
            SuperGraphAnyNode::ModelExecution(node) => node.eval(data, backend),
            SuperGraphAnyNode::TokenizerEncode(node) => node.eval(data, backend),
            SuperGraphAnyNode::TokenizerDecode(node) => node.eval(data, backend),
            SuperGraphAnyNode::ModelLoad(node) => node.eval(data, backend),
            SuperGraphAnyNode::TokenizerLoad(node) => node.eval(data, backend),
            SuperGraphAnyNode::StringInput(node) => node.eval(data, backend),
            SuperGraphAnyNode::StringOutput(node) => node.eval(data, backend),
            SuperGraphAnyNode::MilliOpGraph(node) => node.eval(data, backend)
        }
    }
}