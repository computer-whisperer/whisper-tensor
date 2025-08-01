use std::collections::{HashMap};
use serde::{Deserialize, Serialize};
use typenum::P1;
use rwkv_tokenizer::WorldTokenizer;
use whisper_tensor_import::onnx_graph::TokenizerInfo;
use crate::backends::eval_backend::EvalBackend;
use crate::backends::ndarray_backend::NDArrayNumericTensor;
use crate::DynRank;
use crate::milli_graph::MilliOpGraph;
use crate::numeric_tensor::NumericTensor;
use crate::super_graph::links::{SuperGraphAnyLink, SuperGraphLink, SuperGraphLinkDouble, SuperGraphLinkHash, SuperGraphLinkModel, SuperGraphLinkString, SuperGraphLinkTensor, SuperGraphLinkTokenizer, SuperGraphLinkTriple};
use crate::super_graph::{SuperGraphBuilder, SuperGraphData, SuperGraphError, SuperGraphInner};
use crate::super_graph::cache::SuperGraphCache;
use crate::tokenizer::{AnyTokenizer, Tokenizer};

pub trait SuperGraphNode {
    fn get_inputs(&self) -> Vec<SuperGraphAnyLink>;
    fn get_outputs(&self) -> Vec<SuperGraphAnyLink>;
    fn to_any(self) -> SuperGraphAnyNode;
    fn eval<'a>(&'a self, data: &mut SuperGraphData<'a>, caches: Option<&mut SuperGraphCache>, backend: &mut EvalBackend) -> Result<(), SuperGraphError>;
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
    fn eval(&self, data: &mut SuperGraphData, _caches: Option<&mut SuperGraphCache>, backend: &mut EvalBackend) -> Result<(), SuperGraphError> {
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
    fn eval(&self, data: &mut SuperGraphData, _caches: Option<&mut SuperGraphCache>, _backend: &mut EvalBackend) -> Result<(), SuperGraphError> {
        let tokenizer = match &self.info {
            TokenizerInfo::HFTokenizer(name) => {
                #[cfg(all(feature = "tokenizers", feature = "http"))]
                {
                    AnyTokenizer::Tokenizers(tokenizers::Tokenizer::from_pretrained(name, None).unwrap())
                }
                #[cfg(not(all(feature = "tokenizers", feature = "http")))]
                {
                    panic!("Huggingface tokenizer not supported")
                }
            }
            TokenizerInfo::RWKVWorld => {
                #[cfg(feature = "rwkv-tokenizer")]
                {
                    AnyTokenizer::Rwkv(WorldTokenizer::new(None).unwrap())
                }
                #[cfg(not(feature = "rwkv-tokenizer"))]
                {
                    panic!("RWKV tokenizer not supported")
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
    fn eval(&self, data: &mut SuperGraphData, _caches: Option<&mut SuperGraphCache>, _backend: &mut EvalBackend) -> Result<(), SuperGraphError> {
        let text = data.strings.get(&self.text_input).unwrap();
        let tokenizer = data.tokenizers.get(&self.tokenizer).unwrap();
        let tokens = tokenizer.encode(text).iter().map(|x| *x as i64).collect::<Vec<_>>();
        let input_tensor = NumericTensor::from_vec(tokens).to_dyn_rank().unsqueeze(0).unwrap().unsqueeze(0).unwrap();
        data.tensors.insert(self.tensor_output.clone(), input_tensor);
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
    fn eval(&self, data: &mut SuperGraphData, _caches: Option<&mut SuperGraphCache>, _backend: &mut EvalBackend) -> Result<(), SuperGraphError> {
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
    fn eval(&self, data: &mut SuperGraphData, _caches: Option<&mut SuperGraphCache>, backend: &mut EvalBackend) -> Result<(), SuperGraphError> {
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
pub struct SuperGraphNodeScan {
    inner_graph: SuperGraphInner,
    iteration_count: SuperGraphLinkTensor,
    simple_inputs: Vec<SuperGraphLinkDouble>,
    state_links: Vec<SuperGraphLinkTriple>,
    scan_inputs: Vec<(SuperGraphLinkTensor, SuperGraphLinkTensor, u32)>,
    scan_outputs: Vec<(SuperGraphLinkTensor, SuperGraphLinkTensor, u32)>,
    simple_outputs: Vec<SuperGraphLinkDouble>,
}

impl SuperGraphNodeScan {
    pub fn new(
        inner_graph: SuperGraphInner,
        iteration_count: SuperGraphLinkTensor,
        simple_inputs: Vec<SuperGraphLinkDouble>,
        state_links: Vec<SuperGraphLinkTriple>,
        scan_inputs: Vec<(SuperGraphLinkTensor, SuperGraphLinkTensor, u32)>,
        scan_outputs: Vec<(SuperGraphLinkTensor, SuperGraphLinkTensor, u32)>,
        simple_outputs: Vec<SuperGraphLinkDouble>,
    ) -> Self {
        Self {
            inner_graph,
            iteration_count,
            simple_inputs,
            state_links,
            scan_inputs,
            scan_outputs,
            simple_outputs,
        }
    }
}

impl SuperGraphNode for SuperGraphNodeScan {
    fn get_inputs(&self) -> Vec<SuperGraphAnyLink> {
        let mut inputs = Vec::new();
        inputs.push(self.iteration_count.to_any());
        for link in &self.simple_inputs {
            inputs.push(link.first());
        }
        for link in &self.state_links {
            inputs.push(link.first());
        }
        for (input, _, _) in &self.scan_inputs {
            inputs.push(input.to_any());
        }
        inputs
    }

    fn get_outputs(&self) -> Vec<SuperGraphAnyLink> {
        let mut outputs = Vec::new();
        for link in &self.simple_outputs {
            outputs.push(link.second());
        }
        for (_, output, _) in &self.scan_outputs {
            outputs.push(output.to_any());
        }
        outputs
    }

    fn to_any(self) -> SuperGraphAnyNode {
        SuperGraphAnyNode::Scan(self)
    }

    fn eval<'a>(&'a self, data: &mut SuperGraphData<'a>, mut caches: Option<&mut SuperGraphCache>, backend: &mut EvalBackend) -> Result<(), SuperGraphError> {
        let iteration_count_tensor = data.tensors.get(&self.iteration_count).ok_or(SuperGraphError::MissingLinkError())?;
        let iteration_count: i64 = iteration_count_tensor.first_element().into();

        let simple_inputs = {
            let mut simple_inputs = SuperGraphData::new();
            for link in &self.simple_inputs {
                match link {
                    SuperGraphLinkDouble::Tensor(input, output) => {
                        simple_inputs.tensors.insert(output.clone(), data.tensors.get(input).ok_or(SuperGraphError::MissingLinkError())?.clone());
                    }
                    SuperGraphLinkDouble::String(input, output) => {
                        simple_inputs.strings.insert(output.clone(), data.strings.get(input).ok_or(SuperGraphError::MissingLinkError())?.clone());
                    }
                    SuperGraphLinkDouble::Model(input, output) => {
                        simple_inputs.models.insert(output.clone(), *data.models.get(input).ok_or(SuperGraphError::MissingLinkError())?);
                    }
                    SuperGraphLinkDouble::Tokenizer(input, output) => {
                        simple_inputs.tokenizers.insert(output.clone(), data.tokenizers.get(input).ok_or(SuperGraphError::MissingLinkError())?.clone());
                    }
                    SuperGraphLinkDouble::Hash(input, output) => {
                        simple_inputs.hashes.insert(output.clone(), data.hashes.get(input).ok_or(SuperGraphError::MissingLinkError())?.clone());
                    }
                }
            }
            simple_inputs
        };

        let mut state_values = {
            let mut state_values = SuperGraphData::new();
            for link in &self.state_links {
                match link {
                    SuperGraphLinkTriple::Tensor(initial, inner_input, _inner_output) => {
                        state_values.tensors.insert(inner_input.clone(), data.tensors.get(initial).ok_or(SuperGraphError::MissingLinkError())?.clone());
                    }
                    SuperGraphLinkTriple::String(initial, inner_input, _inner_output) => {
                        state_values.strings.insert(inner_input.clone(), data.strings.get(initial).ok_or(SuperGraphError::MissingLinkError())?.clone());
                    }
                    SuperGraphLinkTriple::Model(initial, inner_input, _inner_output) => {
                        state_values.models.insert(inner_input.clone(), *data.models.get(initial).ok_or(SuperGraphError::MissingLinkError())?);
                    }
                    SuperGraphLinkTriple::Tokenizer(initial, inner_input, _inner_output) => {
                        state_values.tokenizers.insert(inner_input.clone(), data.tokenizers.get(initial).ok_or(SuperGraphError::MissingLinkError())?.clone());
                    }
                    SuperGraphLinkTriple::Hash(initial, inner_input, _inner_output) => {
                        state_values.hashes.insert(inner_input.clone(), data.hashes.get(initial).ok_or(SuperGraphError::MissingLinkError())?.clone());
                    }
                }
            }
            state_values
        };

        let mut prev_iter_outputs: Option<SuperGraphData> = None;

        let mut output_scan_tensor_parts = HashMap::new();
        for (_inner, outer, scan_axis) in &self.scan_outputs {
            output_scan_tensor_parts.insert(outer.clone(), (Vec::new(), *scan_axis as usize));
        }

        for i in 0..iteration_count as u64 {
            let iter_inputs = {
                let mut iter_inputs = simple_inputs.clone();
                iter_inputs.extend(&state_values);
                for (outer, inner, scan_axis) in &self.scan_inputs {
                    let tensor = data.tensors.get(&outer).ok_or(SuperGraphError::MissingLinkError())?;
                    let slice_arg = {
                        let mut slice_ranges = Vec::new();
                        for j in 0..tensor.rank() {
                            if j == *scan_axis as usize {
                                slice_ranges.push(i..i+1);
                            }
                            else {
                                slice_ranges.push(0..tensor.shape()[j]);
                            }
                        }
                        slice_ranges
                    };
                    let sliced = tensor.slice(slice_arg.as_slice(), backend)?;
                    let squeezed = sliced.squeeze(*scan_axis as usize)?;
                    iter_inputs.tensors.insert(inner.clone(), squeezed);
                }
                iter_inputs
            };
            let iter_outputs = self.inner_graph.eval(iter_inputs, caches.as_deref_mut(), backend)?;


            for (inner, outer, _scan_axis) in &self.scan_outputs {
                let tensor = iter_outputs.tensors.get(inner).ok_or(SuperGraphError::MissingLinkError())?;
                let (tensors, _) = output_scan_tensor_parts.get_mut(outer).ok_or(SuperGraphError::MissingLinkError())?;
                tensors.push(tensor.clone());
            }

            state_values = {
                let mut state_values = SuperGraphData::new();
                for link in &self.state_links {
                    match link {
                        SuperGraphLinkTriple::Tensor(_initial, inner_input, inner_output) => {
                            state_values.tensors.insert(inner_input.clone(), iter_outputs.tensors.get(inner_output).ok_or(SuperGraphError::MissingLinkError())?.clone());
                        }
                        SuperGraphLinkTriple::String(_initial, inner_input, inner_output) => {
                            state_values.strings.insert(inner_input.clone(), iter_outputs.strings.get(inner_output).ok_or(SuperGraphError::MissingLinkError())?.clone());
                        }
                        SuperGraphLinkTriple::Tokenizer(_initial, inner_input, inner_output) => {
                            state_values.tokenizers.insert(inner_input.clone(), iter_outputs.tokenizers.get(inner_output).ok_or(SuperGraphError::MissingLinkError())?.clone());
                        }
                        SuperGraphLinkTriple::Model(_initial, inner_input, inner_output) => {
                            state_values.models.insert(inner_input.clone(), *iter_outputs.models.get(inner_output).ok_or(SuperGraphError::MissingLinkError())?);
                        }
                        SuperGraphLinkTriple::Hash(_initial, inner_input, inner_output) => {
                            state_values.hashes.insert(inner_input.clone(), iter_outputs.hashes.get(inner_output).ok_or(SuperGraphError::MissingLinkError())?.clone());
                        }
                    }
                }
                state_values
            };


            prev_iter_outputs = Some(iter_outputs);
        }

        let mut output_data = SuperGraphData::new();

        for link in &self.simple_outputs {
            match link {
                SuperGraphLinkDouble::Tensor(input, output) => {
                    output_data.tensors
                        .insert(output.clone(),
                                prev_iter_outputs.as_ref().unwrap()
                                    .tensors.get(input).ok_or(SuperGraphError::MissingLinkError())?.clone());
                }
                SuperGraphLinkDouble::String(input, output) => {
                    output_data.strings
                       .insert(output.clone(), prev_iter_outputs.as_ref().unwrap().strings.get(input).ok_or(SuperGraphError::MissingLinkError())?.clone());
                }
                SuperGraphLinkDouble::Tokenizer(input, output) => {
                    output_data.tokenizers
                       .insert(output.clone(), prev_iter_outputs.as_ref().unwrap().tokenizers.get(input).ok_or(SuperGraphError::MissingLinkError())?.clone());
                }
                SuperGraphLinkDouble::Model(input, output) => {
                    output_data.models
                       .insert(output.clone(), *prev_iter_outputs.as_ref().unwrap().models.get(input).ok_or(SuperGraphError::MissingLinkError())?);
                }
                SuperGraphLinkDouble::Hash(input, output) => {
                    output_data.hashes
                       .insert(output.clone(), prev_iter_outputs.as_ref().unwrap().hashes.get(input).ok_or(SuperGraphError::MissingLinkError())?.clone());
                }
            }
        };

        for (link, (parts, axis)) in output_scan_tensor_parts {
            let unsqueezed = parts.into_iter().map(|tensor| tensor.unsqueeze(axis)).collect::<Result<Vec<_>, _>>()?;
            let unsqueezed_ref = unsqueezed.iter().collect::<Vec<_>>();
            output_data.tensors.insert(link, NumericTensor::<DynRank>::concat(unsqueezed_ref.as_slice(), axis, backend)?);
        }

        data.extend(&output_data);
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeRNNCacheRead {
    key_input: SuperGraphLinkHash,
    tokens_input: SuperGraphLinkTensor,
    tokens_output: SuperGraphLinkTensor,
    state_outputs: HashMap<String, SuperGraphLinkTensor>,
    default_state_inputs: HashMap<String, SuperGraphLinkTensor>
}

impl SuperGraphNodeRNNCacheRead {
    pub fn new(key_input: SuperGraphLinkHash, tokens_input: SuperGraphLinkTensor, tokens_output: SuperGraphLinkTensor, state_outputs: HashMap<String, SuperGraphLinkTensor>, default_state_inputs: HashMap<String, SuperGraphLinkTensor>) -> Self {
        Self {
            key_input,
            tokens_input,
            tokens_output,
            state_outputs,
            default_state_inputs
        }
    }
}

impl SuperGraphNode for SuperGraphNodeRNNCacheRead {
    fn get_inputs(&self) -> Vec<SuperGraphAnyLink> {
        vec![self.key_input.to_any(), self.tokens_input.to_any()].into_iter().chain(self.default_state_inputs.values().map(|x| x.to_any())).collect()
    }

    fn get_outputs(&self) -> Vec<SuperGraphAnyLink> {
        vec![self.tokens_output.to_any()].into_iter().chain(self.state_outputs.values().map(|x| x.to_any())).collect()
    }

    fn to_any(self) -> SuperGraphAnyNode {
        SuperGraphAnyNode::RNNCacheRead(self)
    }

    fn eval<'a>(&'a self, data: &mut SuperGraphData<'a>, caches: Option<&mut SuperGraphCache>, _backend: &mut EvalBackend) -> Result<(), SuperGraphError> {
        let tokens_input = data.tensors.get(&self.tokens_input).ok_or(SuperGraphError::MissingLinkError())?.clone();
        let mut found = false;
        if let Some(caches) = caches {
            let key_input = *data.hashes.get(&self.key_input).ok_or(SuperGraphError::MissingLinkError())?;
            if let Some(rnn_cache) = caches.rnn_cache.get(&key_input) {
                let tokens_vec: Vec<u32> = tokens_input.to_ndarray()?.try_to_rank::<P1>().unwrap().try_to_vec().unwrap();
                // Try to match as many tokens as possible

                for i in (1..tokens_vec.len()).rev() {
                    let matched_tokens = &tokens_vec[..i].to_vec();
                    if let Some(state) = rnn_cache.get(matched_tokens) {
                        found = true;
                        for (key, value) in state.iter() {
                            if let Some(output_link) = self.state_outputs.get(key) {
                                data.tensors.insert(output_link.clone(), value.clone());
                            }
                        }
                        // Emit remaining tokens
                        let remaining_tokens = &tokens_vec[i..];
                        let remaining_tokens_tensor = NumericTensor::from(NDArrayNumericTensor::from_vec(remaining_tokens.to_vec()).to_dyn());
                        data.tensors.insert(self.tokens_output.clone(), remaining_tokens_tensor.clone());
                        break;
                    }
                }
            }
        }

        if !found {
            // Emit default state
            for (key, value) in self.default_state_inputs.iter() {
                let value = data.tensors.get(value).ok_or(SuperGraphError::MissingLinkError())?;
                if let Some(output_link) = self.state_outputs.get(key) {
                    data.tensors.insert(output_link.clone(), value.clone());
                }
            }
            data.tensors.insert(self.tokens_output.clone(), tokens_input.clone());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeRNNCacheWrite {
    key_input: SuperGraphLinkHash,
    tokens_input: SuperGraphLinkTensor,
    state_inputs: HashMap<String, SuperGraphLinkTensor>
}

impl SuperGraphNodeRNNCacheWrite {
    pub fn new(key_input: SuperGraphLinkHash, tokens_input: SuperGraphLinkTensor, state_inputs: HashMap<String, SuperGraphLinkTensor>) -> Self {
        Self {
            key_input,
            tokens_input,
            state_inputs
        }
    }
}

impl SuperGraphNode for SuperGraphNodeRNNCacheWrite {
    fn get_inputs(&self) -> Vec<SuperGraphAnyLink> {
        vec![self.key_input.to_any(), self.tokens_input.to_any()].into_iter().chain(self.state_inputs.values().map(|x| x.to_any())).collect()
    }

    fn get_outputs(&self) -> Vec<SuperGraphAnyLink> {
        vec![]
    }

    fn to_any(self) -> SuperGraphAnyNode {
        SuperGraphAnyNode::RNNCacheWrite(self)
    }

    fn eval<'a>(&'a self, data: &mut SuperGraphData<'a>, caches: Option<&mut SuperGraphCache>, _backend: &mut EvalBackend) -> Result<(), SuperGraphError> {
        if let Some(caches) = caches {
            let key_input = *data.hashes.get(&self.key_input).ok_or(SuperGraphError::MissingLinkError())?;
            let tokens_input = data.tensors.get(&self.tokens_input).ok_or(SuperGraphError::MissingLinkError())?;
            let tokens_vec: Vec<u32> = tokens_input.to_ndarray()?.try_to_rank::<P1>().unwrap().try_to_vec().unwrap();
            let state_inputs: HashMap<String, &NumericTensor<DynRank>> = self.state_inputs
                .iter().map(|(k, v)| (k.clone(), data.tensors.get(v).unwrap())).collect();
            caches.rnn_cache.entry(key_input).or_default().insert(tokens_vec, state_inputs.into_iter().map(|(k, v)| (k, v.clone())).collect());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SuperGraphAnyNode {
    ModelExecution(SuperGraphNodeModelExecution),
    TokenizerEncode(SuperGraphNodeTokenizerEncode),
    TokenizerDecode(SuperGraphNodeTokenizerDecode),
    TokenizerLoad(SuperGraphNodeTokenizerLoad),
    MilliOpGraph(SuperGraphNodeMilliOpGraph),
    Scan(SuperGraphNodeScan),
    RNNCacheWrite(SuperGraphNodeRNNCacheWrite),
    RNNCacheRead(SuperGraphNodeRNNCacheRead)
}

impl SuperGraphAnyNode {
    pub fn get_outputs(&self) -> Vec<SuperGraphAnyLink> {
        match self {
            SuperGraphAnyNode::ModelExecution(node) => node.get_outputs(),
            SuperGraphAnyNode::TokenizerEncode(node) => node.get_outputs(),
            SuperGraphAnyNode::TokenizerDecode(node) => node.get_outputs(),
            SuperGraphAnyNode::TokenizerLoad(node) => node.get_outputs(),
            SuperGraphAnyNode::MilliOpGraph(node) => node.get_outputs(),
            SuperGraphAnyNode::Scan(node) => node.get_outputs(),
            SuperGraphAnyNode::RNNCacheWrite(node) => node.get_outputs(),
            SuperGraphAnyNode::RNNCacheRead(node) => node.get_outputs()
        }
    }

    pub fn get_inputs(&self) -> Vec<SuperGraphAnyLink> {
        match self {
            SuperGraphAnyNode::ModelExecution(node) => node.get_inputs(),
            SuperGraphAnyNode::TokenizerEncode(node) => node.get_inputs(),
            SuperGraphAnyNode::TokenizerDecode(node) => node.get_inputs(),
            SuperGraphAnyNode::TokenizerLoad(node) => node.get_inputs(),
            SuperGraphAnyNode::MilliOpGraph(node) => node.get_inputs(),
            SuperGraphAnyNode::Scan(node) => node.get_inputs(),
            SuperGraphAnyNode::RNNCacheWrite(node) => node.get_inputs(),
            SuperGraphAnyNode::RNNCacheRead(node) => node.get_inputs()
        }
    }

    pub(crate) fn eval<'a>(&'a self, data: &mut SuperGraphData<'a>, caches: Option<&mut SuperGraphCache>, backend: &mut EvalBackend) -> Result<(), SuperGraphError> {
        match self {
            SuperGraphAnyNode::ModelExecution(node) => node.eval(data, caches, backend),
            SuperGraphAnyNode::TokenizerEncode(node) => node.eval(data, caches, backend),
            SuperGraphAnyNode::TokenizerDecode(node) => node.eval(data, caches, backend),
            SuperGraphAnyNode::TokenizerLoad(node) => node.eval(data, caches, backend),
            SuperGraphAnyNode::MilliOpGraph(node) => node.eval(data, caches, backend),
            SuperGraphAnyNode::Scan(node) => node.eval(data, caches, backend),
            SuperGraphAnyNode::RNNCacheWrite(node) => node.eval(data, caches, backend),
            SuperGraphAnyNode::RNNCacheRead(node) => node.eval(data, caches, backend)
        }
    }
}