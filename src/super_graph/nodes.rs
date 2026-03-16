use crate::DynRank;
use crate::backends::eval_backend;
use crate::backends::eval_backend::EvalBackend;
use crate::backends::ndarray_backend::NDArrayNumericTensor;
use crate::compiler::CompiledProgramObserver;
use crate::dtype::DType;
use crate::graph::{GlobalId, Graph, Node, NodeMetadata, Property, PropertyValue};
use crate::metadata::TokenizerInfo;
use crate::milli_graph::MilliOpGraph;
use crate::milli_graph::observer::MilliOpGraphObserver;
use crate::numeric_scalar::NumericScalar;
use crate::numeric_tensor::NumericTensor;
use crate::phonemization::{text_to_kokoro_phonemes, text_to_piper_phonemes};
use crate::super_graph::data::{SuperGraphAudioClip, SuperGraphImage};
use crate::super_graph::links::{
    SuperGraphAnyLink, SuperGraphLink, SuperGraphLinkDouble, SuperGraphLinkKind,
    SuperGraphLinkTriple,
};
use crate::super_graph::observer::SuperGraphObserver;
use crate::super_graph::{
    SuperGraph, SuperGraphBuilder, SuperGraphContext, SuperGraphData, SuperGraphError,
};
use crate::symbolic_graph::observer::SymbolicGraphObserver;
use crate::tokenizer::{AnyTokenizer, Tokenizer};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ptr;
use std::time::Instant;
use typenum::P1;

pub trait SuperGraphNode {
    fn to_any(self) -> SuperGraphAnyNode;

    fn op_kind(&self) -> String;
    fn label(&self) -> Option<String> {
        None
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_>;
    fn outputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_>;
    fn global_id(&self) -> GlobalId;

    fn eval<'a, 'b, 'c, 'd, T: SuperGraphObserver>(
        &'a self,
        this_path: &[GlobalId],
        data: &mut SuperGraphData<'b>,
        context: &mut SuperGraphContext<'a, 'b, 'c, 'd, T>,
    ) -> Result<(), SuperGraphError>;
}

impl<T: SuperGraphNode> Node for T {
    type OpKind = String;

    fn global_id(&self) -> GlobalId {
        <Self as SuperGraphNode>::global_id(self)
    }

    fn op_kind(&self) -> Self::OpKind {
        <Self as SuperGraphNode>::op_kind(self)
    }

    fn label(&self) -> Option<String> {
        <Self as SuperGraphNode>::label(self)
    }

    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId> + '_> {
        Box::new(<Self as SuperGraphNode>::inputs(self).map(|x| x.global_id()))
    }

    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId> + '_> {
        Box::new(<Self as SuperGraphNode>::outputs(self).map(|x| x.global_id()))
    }
}

fn tensor_bool_scalar(value: bool) -> Result<NumericTensor<DynRank>, SuperGraphError> {
    Ok(NumericTensor::from_vec_shape(vec![value], vec![])?)
}

fn read_rank0_bool_tensor(
    tensor: &NumericTensor<DynRank>,
    input_name: &str,
) -> Result<bool, SuperGraphError> {
    if tensor.rank() != 0 {
        return Err(SuperGraphError::InvalidInputError(format!(
            "{} must be a rank-0 bool tensor, got rank={} shape={:?}",
            input_name,
            tensor.rank(),
            tensor.shape()
        )));
    }
    if tensor.dtype() != DType::BOOL {
        return Err(SuperGraphError::InvalidInputError(format!(
            "{} must have BOOL dtype, got {:?}",
            input_name,
            tensor.dtype()
        )));
    }
    Ok(tensor.first_element().into())
}

fn read_rank0_i64_tensor(
    tensor: &NumericTensor<DynRank>,
    input_name: &str,
    backend: &mut EvalBackend,
) -> Result<i64, SuperGraphError> {
    if tensor.rank() > 1
        || (tensor.rank() == 1 && tensor.shape().first().copied().unwrap_or(0) != 1)
    {
        return Err(SuperGraphError::InvalidInputError(format!(
            "{} must be a scalar (rank-0 or rank-1 with 1 element), got rank={} shape={:?}",
            input_name,
            tensor.rank(),
            tensor.shape()
        )));
    }
    if tensor.dtype() == DType::BOOL {
        let value: bool = tensor.first_element().into();
        return Ok(if value { 1 } else { 0 });
    }
    let cast_tensor = tensor.cast(DType::I64, backend)?;
    Ok(cast_tensor.first_element().into())
}

fn read_rank0_f64_tensor(
    tensor: &NumericTensor<DynRank>,
    input_name: &str,
    backend: &mut EvalBackend,
) -> Result<f64, SuperGraphError> {
    if tensor.rank() > 1
        || (tensor.rank() == 1 && tensor.shape().first().copied().unwrap_or(0) != 1)
    {
        return Err(SuperGraphError::InvalidInputError(format!(
            "{} must be a scalar (rank-0 or rank-1 with 1 element), got rank={} shape={:?}",
            input_name,
            tensor.rank(),
            tensor.shape()
        )));
    }
    if tensor.dtype() == DType::BOOL {
        let value: bool = tensor.first_element().into();
        return Ok(if value { 1.0 } else { 0.0 });
    }
    let cast_tensor = tensor.cast(DType::F64, backend)?;
    if let NumericScalar::F64(value) = cast_tensor.first_element() {
        Ok(value)
    } else {
        Err(SuperGraphError::InvalidInputError(format!(
            "{} could not be converted into F64",
            input_name
        )))
    }
}

fn parse_piper_phoneme_id_map(json: &str) -> Result<HashMap<char, Vec<i64>>, SuperGraphError> {
    let value: serde_json::Value = serde_json::from_str(json).map_err(|e| {
        SuperGraphError::InvalidInputError(format!("invalid Piper phoneme_id_map JSON: {e}"))
    })?;
    let obj = value.as_object().ok_or(SuperGraphError::InvalidInputError(
        "Piper phoneme_id_map is not an object".to_string(),
    ))?;
    let mut map = HashMap::new();
    for (key, val) in obj {
        if key.chars().count() != 1 {
            continue;
        }
        let ch = key.chars().next().unwrap();
        let ids = val
            .as_array()
            .ok_or(SuperGraphError::InvalidInputError(format!(
                "Piper phoneme_id_map[{key}] is not an array"
            )))?
            .iter()
            .map(|v| {
                v.as_i64().ok_or(SuperGraphError::InvalidInputError(format!(
                    "Piper phoneme_id_map[{key}] contains non-i64 value"
                )))
            })
            .collect::<Result<Vec<_>, _>>()?;
        map.insert(ch, ids);
    }
    Ok(map)
}

fn load_kokoro_vocab(info: &TokenizerInfo) -> Result<HashMap<char, u32>, SuperGraphError> {
    let path = match info {
        TokenizerInfo::HFTokenizerLocal(path) => path,
        _ => {
            return Err(SuperGraphError::InvalidInputError(
                "Kokoro phoneme tokenizer must be HFTokenizerLocal".to_string(),
            ));
        }
    };
    let json = std::fs::read_to_string(path).map_err(|e| {
        SuperGraphError::InvalidInputError(format!(
            "failed to read Kokoro tokenizer file {path}: {e}"
        ))
    })?;
    let value: serde_json::Value = serde_json::from_str(&json).map_err(|e| {
        SuperGraphError::InvalidInputError(format!(
            "invalid JSON in Kokoro tokenizer file {path}: {e}"
        ))
    })?;
    let vocab_obj =
        value["model"]["vocab"]
            .as_object()
            .ok_or(SuperGraphError::InvalidInputError(format!(
                "missing model.vocab object in Kokoro tokenizer file {path}"
            )))?;
    let mut vocab = HashMap::new();
    for (key, val) in vocab_obj {
        if key.chars().count() != 1 {
            continue;
        }
        let id = val
            .as_u64()
            .ok_or(SuperGraphError::InvalidInputError(format!(
                "non-u64 vocab id for key {key} in {path}"
            )))? as u32;
        vocab.insert(key.chars().next().unwrap(), id);
    }
    Ok(vocab)
}

fn build_f5_vocab(vocab_text: &str) -> HashMap<char, i32> {
    let mut map = HashMap::new();
    for (id, line) in vocab_text.lines().enumerate() {
        if line.chars().count() == 1 {
            map.insert(line.chars().next().unwrap(), id as i32);
        } else if line.is_empty() {
            // Line 0 is space in F5 vocab.
            map.insert(' ', id as i32);
        }
    }
    map
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelReference {
    idx: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeModelExecution {
    global_id: GlobalId,
    pub label: Option<String>,
    tensor_map: SuperGraphLink,
    pub symbolic_graph_id: usize, // Which graph (passed to
    tensor_inputs: Vec<(SuperGraphLink, String)>,
    tensor_outputs: Vec<(String, SuperGraphLink)>,
}

impl SuperGraphNodeModelExecution {
    pub fn new(
        rng: &mut impl Rng,
        tensor_map: SuperGraphLink,
        symbolic_graph_id: usize,
        tensor_inputs: Vec<(SuperGraphLink, String)>,
        tensor_outputs: Vec<(String, SuperGraphLink)>,
    ) -> Self {
        Self {
            global_id: GlobalId::new(rng),
            label: None,
            tensor_map,
            symbolic_graph_id,
            tensor_inputs,
            tensor_outputs,
        }
    }
}

struct SymbolicGraphObserverWrapper<'a, T: SuperGraphObserver> {
    inner: &'a mut T,
    path: Vec<GlobalId>,
}

impl<'a, T: SuperGraphObserver> SymbolicGraphObserverWrapper<'a, T> {
    fn new(inner: &'a mut T, path: &[GlobalId]) -> Self {
        Self {
            inner,
            path: path.to_vec(),
        }
    }
}

impl<'a, T: SuperGraphObserver> SymbolicGraphObserver for SymbolicGraphObserverWrapper<'a, T> {
    fn on_op_executed(
        &mut self,
        node_path: &[GlobalId],
        start_instant: Instant,
        end_instant: Instant,
        backend: &mut EvalBackend,
    ) {
        let node_path = self
            .path
            .clone()
            .into_iter()
            .chain(node_path.iter().cloned())
            .collect::<Vec<_>>();
        self.inner.on_node_executed(
            node_path.as_slice(),
            "",
            start_instant,
            end_instant,
            backend,
        )
    }

    fn on_tensor_assigned(
        &mut self,
        tensor_path: &[GlobalId],
        tensor: &NumericTensor<DynRank>,
        backend: &mut EvalBackend,
    ) {
        let tensor_path = self
            .path
            .clone()
            .into_iter()
            .chain(tensor_path.iter().cloned())
            .collect::<Vec<_>>();
        self.inner
            .on_tensor_assigned(tensor_path.as_slice(), tensor, backend)
    }

    fn on_loading_weight(&mut self, path: &[GlobalId], weight_name: Option<String>) {
        let path = self
            .path
            .clone()
            .into_iter()
            .chain(path.iter().cloned())
            .collect::<Vec<_>>();
        self.inner.on_loading_weight(path.as_slice(), weight_name);
    }

    fn should_cancel(&mut self) -> bool {
        self.inner.should_cancel()
    }
}

impl<'a, T: SuperGraphObserver> CompiledProgramObserver for SymbolicGraphObserverWrapper<'a, T> {
    fn on_op_executed(
        &mut self,
        node_path: &[GlobalId],
        start_instant: Instant,
        end_instant: Instant,
        backend: &mut EvalBackend,
    ) {
        let node_path = self
            .path
            .clone()
            .into_iter()
            .chain(node_path.iter().cloned())
            .collect::<Vec<_>>();
        self.inner.on_node_executed(
            node_path.as_slice(),
            "",
            start_instant,
            end_instant,
            backend,
        )
    }

    fn on_tensor_assigned(
        &mut self,
        tensor_path: &[GlobalId],
        tensor: &NumericTensor<DynRank>,
        backend: &mut EvalBackend,
    ) {
        let tensor_path = self
            .path
            .clone()
            .into_iter()
            .chain(tensor_path.iter().cloned())
            .collect::<Vec<_>>();
        self.inner
            .on_tensor_assigned(tensor_path.as_slice(), tensor, backend)
    }

    fn on_loading_weight(&mut self, path: &[GlobalId], weight_name: Option<String>) {
        let path = self
            .path
            .clone()
            .into_iter()
            .chain(path.iter().cloned())
            .collect::<Vec<_>>();
        self.inner.on_loading_weight(path.as_slice(), weight_name);
    }

    fn should_cancel(&mut self) -> bool {
        self.inner.should_cancel()
    }
}

impl SuperGraphNode for SuperGraphNodeModelExecution {
    fn to_any(self) -> SuperGraphAnyNode {
        SuperGraphAnyNode::ModelExecution(self)
    }
    fn eval<T: SuperGraphObserver>(
        &self,
        node_path: &[GlobalId],
        data: &mut SuperGraphData,
        context: &mut SuperGraphContext<T>,
    ) -> Result<(), SuperGraphError> {
        let tensor_store = data.tensor_maps.get(&self.tensor_map).unwrap();

        let inputs = {
            let mut inputs = HashMap::new();
            for (link, name) in &self.tensor_inputs {
                inputs.insert(name.clone(), data.tensors.get(link).unwrap().clone());
            }
            inputs
        };
        let tensor_cache = {
            let mut res = None;
            for (a, b) in &mut context.super_graph_tensor_cache.caches {
                if ptr::eq(*a, *tensor_store) {
                    res = Some(b)
                }
            }
            res
        };

        let symbolic_graph = context.symbolic_graphs[self.symbolic_graph_id];

        let global_id = node_path
            .iter()
            .chain(core::iter::once(&self.global_id))
            .cloned()
            .collect::<Vec<_>>();
        let mut observer =
            SymbolicGraphObserverWrapper::new(context.observer, global_id.as_slice());
        if context.use_compiled_models
            && let Some(compiled_models) = &context.compiled_models
        {
            let compiled_model = &compiled_models
                .iter()
                .find(|(x, _y)| core::ptr::addr_eq(*x, symbolic_graph))
                .ok_or(SuperGraphError::ModelNotCompiledError)?
                .1;
            let outputs = compiled_model.run(
                context.eval_backend,
                tensor_store,
                tensor_cache,
                inputs,
                &mut observer,
            )?;
            let outputs = outputs.collect::<HashMap<_, _>>();
            for (name, link) in &self.tensor_outputs {
                data.tensors
                    .insert(*link, outputs.get(name).unwrap().clone());
            }
        } else {
            let outputs = eval_backend::run(
                symbolic_graph,
                tensor_store,
                tensor_cache,
                context.eval_backend,
                &mut observer,
                inputs,
            )?;
            for (name, link) in &self.tensor_outputs {
                data.tensors
                    .insert(*link, outputs.get(name).unwrap().clone());
            }
        };

        Ok(())
    }

    fn op_kind(&self) -> String {
        "Model Execution".to_string()
    }
    fn label(&self) -> Option<String> {
        self.label.clone()
    }

    fn inputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        let ret = self.tensor_inputs.clone().into_iter().map(|x| x.0.to_any());
        Box::new(ret.chain(std::iter::once(self.tensor_map.to_any())))
    }

    fn outputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(
            self.tensor_outputs
                .clone()
                .into_iter()
                .map(|x| x.1.to_any()),
        )
    }

    fn global_id(&self) -> GlobalId {
        self.global_id
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeTokenizerLoad {
    global_id: GlobalId,
    pub label: Option<String>,
    info: TokenizerInfo,
    output: SuperGraphLink,
}

impl SuperGraphNodeTokenizerLoad {
    pub fn new(_builder: &mut SuperGraphBuilder, info: TokenizerInfo, rng: &mut impl Rng) -> Self {
        Self {
            global_id: GlobalId::new(rng),
            label: None,
            info,
            output: SuperGraphLink::new(SuperGraphLinkKind::Tokenizer, rng),
        }
    }

    pub fn new_and_add(
        builder: &mut SuperGraphBuilder,
        info: TokenizerInfo,
        rng: &mut impl Rng,
    ) -> SuperGraphLink {
        let node = Self::new(builder, info, rng);
        let output = node.get_tokenizer_output();
        builder.add_node(node.to_any());
        output
    }

    pub fn get_tokenizer_output(&self) -> SuperGraphLink {
        self.output
    }
}

impl SuperGraphNode for SuperGraphNodeTokenizerLoad {
    fn to_any(self) -> SuperGraphAnyNode {
        SuperGraphAnyNode::TokenizerLoad(self)
    }
    fn eval<T: SuperGraphObserver>(
        &self,
        _this_path: &[GlobalId],
        data: &mut SuperGraphData,
        _context: &mut SuperGraphContext<T>,
    ) -> Result<(), SuperGraphError> {
        let tokenizer = AnyTokenizer::from_tokenizer_info(&self.info);
        data.tokenizers.insert(self.output, tokenizer);
        Ok(())
    }

    fn op_kind(&self) -> String {
        "Tokenizer Load".to_string()
    }
    fn label(&self) -> Option<String> {
        self.label.clone()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(std::iter::empty())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(std::iter::once(self.output.to_any()))
    }
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SuperGraphNodeTokenizerEncodeMode {
    Plain,
    ClipStyle {
        seq_len: usize,
        bos: u32,
        eos: u32,
        pad: u32,
    },
    RawPad {
        seq_len: usize,
        pad: u32,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeTokenizerEncode {
    global_id: GlobalId,
    pub label: Option<String>,
    tokenizer: SuperGraphLink,
    text_input: SuperGraphLink,
    tensor_output: SuperGraphLink,
    mode: SuperGraphNodeTokenizerEncodeMode,
}

impl SuperGraphNodeTokenizerEncode {
    pub fn new(
        _builder: &mut SuperGraphBuilder,
        tokenizer: SuperGraphLink,
        text_input: SuperGraphLink,
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            global_id: GlobalId::new(rng),
            label: None,
            tokenizer,
            text_input,
            tensor_output: SuperGraphLink::new(SuperGraphLinkKind::Tensor, rng),
            mode: SuperGraphNodeTokenizerEncodeMode::Plain,
        }
    }

    pub fn new_with_mode(
        _builder: &mut SuperGraphBuilder,
        tokenizer: SuperGraphLink,
        text_input: SuperGraphLink,
        mode: SuperGraphNodeTokenizerEncodeMode,
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            global_id: GlobalId::new(rng),
            label: None,
            tokenizer,
            text_input,
            tensor_output: SuperGraphLink::new(SuperGraphLinkKind::Tensor, rng),
            mode,
        }
    }

    pub fn new_and_add(
        builder: &mut SuperGraphBuilder,
        tokenizer: SuperGraphLink,
        text_input: SuperGraphLink,
        rng: &mut impl Rng,
    ) -> SuperGraphLink {
        let node = Self::new(builder, tokenizer, text_input, rng);
        let output = node.get_tensor_output();
        builder.add_node(node.to_any());
        output
    }

    pub fn new_with_mode_and_add(
        builder: &mut SuperGraphBuilder,
        tokenizer: SuperGraphLink,
        text_input: SuperGraphLink,
        mode: SuperGraphNodeTokenizerEncodeMode,
        rng: &mut impl Rng,
    ) -> SuperGraphLink {
        let node = Self::new_with_mode(builder, tokenizer, text_input, mode, rng);
        let output = node.get_tensor_output();
        builder.add_node(node.to_any());
        output
    }

    pub fn get_tensor_output(&self) -> SuperGraphLink {
        self.tensor_output
    }
}

impl SuperGraphNode for SuperGraphNodeTokenizerEncode {
    fn to_any(self) -> SuperGraphAnyNode {
        SuperGraphAnyNode::TokenizerEncode(self)
    }
    fn eval<T: SuperGraphObserver>(
        &self,
        _this_path: &[GlobalId],
        data: &mut SuperGraphData,
        _context: &mut SuperGraphContext<T>,
    ) -> Result<(), SuperGraphError> {
        let text = data.strings.get(&self.text_input).unwrap();
        let tokenizer = data.tokenizers.get(&self.tokenizer).unwrap();
        let input_tensor = match &self.mode {
            SuperGraphNodeTokenizerEncodeMode::Plain => {
                let tokens = tokenizer
                    .encode(text)
                    .iter()
                    .map(|x| *x as i64)
                    .collect::<Vec<_>>();
                NumericTensor::from_vec(tokens)
                    .to_dyn_rank()
                    .unsqueeze(0)
                    .unwrap()
                    .unsqueeze(0)
                    .unwrap()
            }
            SuperGraphNodeTokenizerEncodeMode::ClipStyle {
                seq_len,
                bos,
                eos,
                pad,
            } => {
                let mut encoded = tokenizer.encode(text);
                if encoded.first() == Some(bos) {
                    encoded.remove(0);
                }
                if encoded.last() == Some(eos) {
                    encoded.pop();
                }
                let max_text_tokens = seq_len.saturating_sub(2);
                let mut ids: Vec<i32> = Vec::with_capacity(*seq_len);
                ids.push(*bos as i32);
                for &id in encoded.iter().take(max_text_tokens) {
                    ids.push(id as i32);
                }
                ids.push(*eos as i32);
                ids.resize(*seq_len, *pad as i32);
                NumericTensor::from_vec_shape(ids, vec![1, *seq_len])?
            }
            SuperGraphNodeTokenizerEncodeMode::RawPad { seq_len, pad } => {
                let encoded = tokenizer.encode(text);
                let mut ids: Vec<i32> =
                    encoded.iter().take(*seq_len).map(|&id| id as i32).collect();
                ids.resize(*seq_len, *pad as i32);
                NumericTensor::from_vec_shape(ids, vec![1, *seq_len])?
            }
        };
        data.tensors.insert(self.tensor_output, input_tensor);
        Ok(())
    }

    fn op_kind(&self) -> String {
        "Tokenizer Encode".to_string()
    }
    fn label(&self) -> Option<String> {
        self.label.clone()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(
            std::iter::once(self.tokenizer.to_any())
                .chain(std::iter::once(self.text_input.to_any())),
        )
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(std::iter::once(self.tensor_output.to_any()))
    }
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeTokenizerDecode {
    global_id: GlobalId,
    pub label: Option<String>,
    tokenizer: SuperGraphLink,
    tensor_input: SuperGraphLink,
    text_output: SuperGraphLink,
}

impl SuperGraphNodeTokenizerDecode {
    pub fn new(
        _builder: &mut SuperGraphBuilder,
        tokenizer: SuperGraphLink,
        tensor_input: SuperGraphLink,
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            global_id: GlobalId::new(rng),
            label: None,
            tokenizer,
            tensor_input,
            text_output: SuperGraphLink::new(SuperGraphLinkKind::String, rng),
        }
    }
    pub fn get_string_output(&self) -> SuperGraphLink {
        self.text_output
    }
    pub fn new_and_add(
        builder: &mut SuperGraphBuilder,
        tokenizer: SuperGraphLink,
        tensor_input: SuperGraphLink,
        rng: &mut impl Rng,
    ) -> SuperGraphLink {
        let node = Self::new(builder, tokenizer, tensor_input, rng);
        let output = node.get_string_output();
        builder.add_node(node.to_any());
        output
    }
}

impl SuperGraphNode for SuperGraphNodeTokenizerDecode {
    fn to_any(self) -> SuperGraphAnyNode {
        SuperGraphAnyNode::TokenizerDecode(self)
    }
    fn eval<T: SuperGraphObserver>(
        &self,
        _node_path: &[GlobalId],
        data: &mut SuperGraphData,
        _context: &mut SuperGraphContext<T>,
    ) -> Result<(), SuperGraphError> {
        let tensor = data.tensors.get(&self.tensor_input).unwrap();
        let tokenizer = data.tokenizers.get(&self.tokenizer).unwrap();
        let tensor = tensor.try_to_type::<u32>()?.try_to_rank::<P1>()?;
        let output_values: Vec<u32> = tensor.to_vec();
        let text = tokenizer.decode(&output_values)?;
        data.strings.insert(self.text_output, text);
        Ok(())
    }
    fn op_kind(&self) -> String {
        "Tokenizer Decode".to_string()
    }
    fn label(&self) -> Option<String> {
        self.label.clone()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(
            std::iter::once(self.tokenizer.to_any())
                .chain(std::iter::once(self.tensor_input.to_any())),
        )
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(std::iter::once(self.text_output.to_any()))
    }
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SuperGraphNodeTextToPhonemesMode {
    Piper { voice: String },
    Kokoro { voice: String },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeTextToPhonemes {
    global_id: GlobalId,
    pub label: Option<String>,
    text_input: SuperGraphLink,
    phonemes_output: SuperGraphLink,
    mode: SuperGraphNodeTextToPhonemesMode,
}

impl SuperGraphNodeTextToPhonemes {
    pub fn new(
        _builder: &mut SuperGraphBuilder,
        text_input: SuperGraphLink,
        mode: SuperGraphNodeTextToPhonemesMode,
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            global_id: GlobalId::new(rng),
            label: None,
            text_input,
            phonemes_output: SuperGraphLink::new(SuperGraphLinkKind::String, rng),
            mode,
        }
    }

    pub fn new_and_add(
        builder: &mut SuperGraphBuilder,
        text_input: SuperGraphLink,
        mode: SuperGraphNodeTextToPhonemesMode,
        rng: &mut impl Rng,
    ) -> SuperGraphLink {
        let node = Self::new(builder, text_input, mode, rng);
        let output = node.get_phonemes_output();
        builder.add_node(node.to_any());
        output
    }

    pub fn get_phonemes_output(&self) -> SuperGraphLink {
        self.phonemes_output
    }
}

impl SuperGraphNode for SuperGraphNodeTextToPhonemes {
    fn to_any(self) -> SuperGraphAnyNode {
        SuperGraphAnyNode::TextToPhonemes(self)
    }

    fn eval<T: SuperGraphObserver>(
        &self,
        _this_path: &[GlobalId],
        data: &mut SuperGraphData,
        _context: &mut SuperGraphContext<T>,
    ) -> Result<(), SuperGraphError> {
        let text = data
            .strings
            .get(&self.text_input)
            .ok_or(SuperGraphError::MissingLinkError(format!(
                ": missing text input {:?}",
                self.text_input
            )))?;
        let phonemes = match &self.mode {
            SuperGraphNodeTextToPhonemesMode::Piper { voice } => {
                text_to_piper_phonemes(text, voice)
            }
            SuperGraphNodeTextToPhonemesMode::Kokoro { voice } => {
                text_to_kokoro_phonemes(text, voice)
            }
        }
        .map_err(SuperGraphError::InvalidInputError)?;
        data.strings.insert(self.phonemes_output, phonemes);
        Ok(())
    }

    fn op_kind(&self) -> String {
        "TextToPhonemes".to_string()
    }
    fn label(&self) -> Option<String> {
        self.label.clone()
    }

    fn inputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(std::iter::once(self.text_input.to_any()))
    }

    fn outputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(std::iter::once(self.phonemes_output.to_any()))
    }

    fn global_id(&self) -> GlobalId {
        self.global_id
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodePiperPhonemesToTensor {
    global_id: GlobalId,
    pub label: Option<String>,
    phonemes_input: SuperGraphLink,
    token_ids_output: SuperGraphLink,
    input_lengths_output: SuperGraphLink,
    phoneme_id_map_json: String,
}

impl SuperGraphNodePiperPhonemesToTensor {
    pub fn new(
        _builder: &mut SuperGraphBuilder,
        phonemes_input: SuperGraphLink,
        phoneme_id_map_json: String,
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            global_id: GlobalId::new(rng),
            label: None,
            phonemes_input,
            token_ids_output: SuperGraphLink::new(SuperGraphLinkKind::Tensor, rng),
            input_lengths_output: SuperGraphLink::new(SuperGraphLinkKind::Tensor, rng),
            phoneme_id_map_json,
        }
    }

    pub fn new_and_add(
        builder: &mut SuperGraphBuilder,
        phonemes_input: SuperGraphLink,
        phoneme_id_map_json: String,
        rng: &mut impl Rng,
    ) -> (SuperGraphLink, SuperGraphLink) {
        let node = Self::new(builder, phonemes_input, phoneme_id_map_json, rng);
        let token_ids = node.get_token_ids_output();
        let input_lengths = node.get_input_lengths_output();
        builder.add_node(node.to_any());
        (token_ids, input_lengths)
    }

    pub fn get_token_ids_output(&self) -> SuperGraphLink {
        self.token_ids_output
    }

    pub fn get_input_lengths_output(&self) -> SuperGraphLink {
        self.input_lengths_output
    }
}

impl SuperGraphNode for SuperGraphNodePiperPhonemesToTensor {
    fn to_any(self) -> SuperGraphAnyNode {
        SuperGraphAnyNode::PiperPhonemesToTensor(self)
    }

    fn eval<T: SuperGraphObserver>(
        &self,
        _this_path: &[GlobalId],
        data: &mut SuperGraphData,
        _context: &mut SuperGraphContext<T>,
    ) -> Result<(), SuperGraphError> {
        let phonemes =
            data.strings
                .get(&self.phonemes_input)
                .ok_or(SuperGraphError::MissingLinkError(format!(
                    ": missing phoneme input {:?}",
                    self.phonemes_input
                )))?;
        let phoneme_id_map = parse_piper_phoneme_id_map(&self.phoneme_id_map_json)?;

        let mut token_ids: Vec<i64> = vec![1, 0];
        for ch in phonemes.chars() {
            if let Some(ids) = phoneme_id_map.get(&ch) {
                token_ids.extend(ids.iter().copied());
            }
            token_ids.push(0);
        }
        token_ids.push(2);

        let num_tokens = token_ids.len();
        let token_ids_tensor =
            NumericTensor::<DynRank>::from_vec_shape(token_ids, vec![1, num_tokens])?;
        let input_lengths_tensor =
            NumericTensor::<DynRank>::from_vec_shape(vec![num_tokens as i64], vec![1])?;
        data.tensors.insert(self.token_ids_output, token_ids_tensor);
        data.tensors
            .insert(self.input_lengths_output, input_lengths_tensor);
        Ok(())
    }

    fn op_kind(&self) -> String {
        "PiperPhonemesToTensor".to_string()
    }
    fn label(&self) -> Option<String> {
        self.label.clone()
    }

    fn inputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(std::iter::once(self.phonemes_input.to_any()))
    }

    fn outputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(
            [
                self.token_ids_output.to_any(),
                self.input_lengths_output.to_any(),
            ]
            .into_iter(),
        )
    }

    fn global_id(&self) -> GlobalId {
        self.global_id
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeKokoroPhonemesToTensor {
    global_id: GlobalId,
    pub label: Option<String>,
    phonemes_input: SuperGraphLink,
    token_ids_output: SuperGraphLink,
    tokenizer: TokenizerInfo,
}

impl SuperGraphNodeKokoroPhonemesToTensor {
    pub fn new(
        _builder: &mut SuperGraphBuilder,
        phonemes_input: SuperGraphLink,
        tokenizer: TokenizerInfo,
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            global_id: GlobalId::new(rng),
            label: None,
            phonemes_input,
            token_ids_output: SuperGraphLink::new(SuperGraphLinkKind::Tensor, rng),
            tokenizer,
        }
    }

    pub fn new_and_add(
        builder: &mut SuperGraphBuilder,
        phonemes_input: SuperGraphLink,
        tokenizer: TokenizerInfo,
        rng: &mut impl Rng,
    ) -> SuperGraphLink {
        let node = Self::new(builder, phonemes_input, tokenizer, rng);
        let output = node.get_token_ids_output();
        builder.add_node(node.to_any());
        output
    }

    pub fn get_token_ids_output(&self) -> SuperGraphLink {
        self.token_ids_output
    }
}

impl SuperGraphNode for SuperGraphNodeKokoroPhonemesToTensor {
    fn to_any(self) -> SuperGraphAnyNode {
        SuperGraphAnyNode::KokoroPhonemesToTensor(self)
    }

    fn eval<T: SuperGraphObserver>(
        &self,
        _this_path: &[GlobalId],
        data: &mut SuperGraphData,
        _context: &mut SuperGraphContext<T>,
    ) -> Result<(), SuperGraphError> {
        let phonemes =
            data.strings
                .get(&self.phonemes_input)
                .ok_or(SuperGraphError::MissingLinkError(format!(
                    ": missing phoneme input {:?}",
                    self.phonemes_input
                )))?;
        let vocab = load_kokoro_vocab(&self.tokenizer)?;
        let mut token_ids: Vec<i64> = vec![0]; // BOS ($)
        for ch in phonemes.chars() {
            if let Some(&id) = vocab.get(&ch) {
                token_ids.push(id as i64);
            }
        }
        token_ids.push(0); // EOS ($)
        let num_tokens = token_ids.len();
        let token_ids_tensor =
            NumericTensor::<DynRank>::from_vec_shape(token_ids, vec![1, num_tokens])?;
        data.tensors.insert(self.token_ids_output, token_ids_tensor);
        Ok(())
    }

    fn op_kind(&self) -> String {
        "KokoroPhonemesToTensor".to_string()
    }
    fn label(&self) -> Option<String> {
        self.label.clone()
    }

    fn inputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(std::iter::once(self.phonemes_input.to_any()))
    }

    fn outputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(std::iter::once(self.token_ids_output.to_any()))
    }

    fn global_id(&self) -> GlobalId {
        self.global_id
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeF5TextToTensor {
    global_id: GlobalId,
    pub label: Option<String>,
    text_input: SuperGraphLink,
    token_ids_output: SuperGraphLink,
    vocab: String,
}

impl SuperGraphNodeF5TextToTensor {
    pub fn new(
        _builder: &mut SuperGraphBuilder,
        text_input: SuperGraphLink,
        vocab: String,
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            global_id: GlobalId::new(rng),
            label: None,
            text_input,
            token_ids_output: SuperGraphLink::new(SuperGraphLinkKind::Tensor, rng),
            vocab,
        }
    }

    pub fn new_and_add(
        builder: &mut SuperGraphBuilder,
        text_input: SuperGraphLink,
        vocab: String,
        rng: &mut impl Rng,
    ) -> SuperGraphLink {
        let node = Self::new(builder, text_input, vocab, rng);
        let output = node.get_token_ids_output();
        builder.add_node(node.to_any());
        output
    }

    pub fn get_token_ids_output(&self) -> SuperGraphLink {
        self.token_ids_output
    }
}

impl SuperGraphNode for SuperGraphNodeF5TextToTensor {
    fn to_any(self) -> SuperGraphAnyNode {
        SuperGraphAnyNode::F5TextToTensor(self)
    }

    fn eval<T: SuperGraphObserver>(
        &self,
        _this_path: &[GlobalId],
        data: &mut SuperGraphData,
        _context: &mut SuperGraphContext<T>,
    ) -> Result<(), SuperGraphError> {
        let text = data
            .strings
            .get(&self.text_input)
            .ok_or(SuperGraphError::MissingLinkError(format!(
                ": missing text input {:?}",
                self.text_input
            )))?;
        let vocab_map = build_f5_vocab(&self.vocab);
        let mut token_ids: Vec<i32> = Vec::new();
        for ch in text.chars() {
            if let Some(&id) = vocab_map.get(&ch) {
                token_ids.push(id);
            }
        }
        let num_tokens = token_ids.len();
        let token_ids_tensor =
            NumericTensor::<DynRank>::from_vec_shape(token_ids, vec![1, num_tokens])?;
        data.tensors.insert(self.token_ids_output, token_ids_tensor);
        Ok(())
    }

    fn op_kind(&self) -> String {
        "F5TextToTensor".to_string()
    }
    fn label(&self) -> Option<String> {
        self.label.clone()
    }

    fn inputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(std::iter::once(self.text_input.to_any()))
    }

    fn outputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(std::iter::once(self.token_ids_output.to_any()))
    }

    fn global_id(&self) -> GlobalId {
        self.global_id
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeTensorToImage {
    global_id: GlobalId,
    pub label: Option<String>,
    tensor_input: SuperGraphLink,
    image_output: SuperGraphLink,
}

impl SuperGraphNodeTensorToImage {
    pub fn new(
        _builder: &mut SuperGraphBuilder,
        tensor_input: SuperGraphLink,
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            global_id: GlobalId::new(rng),
            label: None,
            tensor_input,
            image_output: SuperGraphLink::new(SuperGraphLinkKind::Image, rng),
        }
    }

    pub fn new_and_add(
        builder: &mut SuperGraphBuilder,
        tensor_input: SuperGraphLink,
        rng: &mut impl Rng,
    ) -> SuperGraphLink {
        let node = Self::new(builder, tensor_input, rng);
        let output = node.get_image_output();
        builder.add_node(node.to_any());
        output
    }

    pub fn get_image_output(&self) -> SuperGraphLink {
        self.image_output
    }
}

impl SuperGraphNode for SuperGraphNodeTensorToImage {
    fn to_any(self) -> SuperGraphAnyNode {
        SuperGraphAnyNode::TensorToImage(self)
    }

    fn eval<T: SuperGraphObserver>(
        &self,
        _node_path: &[GlobalId],
        data: &mut SuperGraphData,
        _context: &mut SuperGraphContext<T>,
    ) -> Result<(), SuperGraphError> {
        let tensor = data
            .tensors
            .get(&self.tensor_input)
            .ok_or(SuperGraphError::MissingLinkError(format!(
                ": missing tensor input link {:?}",
                self.tensor_input
            )))?
            .clone();
        data.images
            .insert(self.image_output, SuperGraphImage::new(tensor));
        Ok(())
    }

    fn op_kind(&self) -> String {
        "TensorToImage".to_string()
    }
    fn label(&self) -> Option<String> {
        self.label.clone()
    }

    fn inputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(std::iter::once(self.tensor_input.to_any()))
    }

    fn outputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(std::iter::once(self.image_output.to_any()))
    }

    fn global_id(&self) -> GlobalId {
        self.global_id
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeTensorToAudioClip {
    global_id: GlobalId,
    pub label: Option<String>,
    tensor_input: SuperGraphLink,
    audio_output: SuperGraphLink,
    sample_rate_hz: u32,
}

impl SuperGraphNodeTensorToAudioClip {
    pub fn new(
        _builder: &mut SuperGraphBuilder,
        tensor_input: SuperGraphLink,
        sample_rate_hz: u32,
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            global_id: GlobalId::new(rng),
            label: None,
            tensor_input,
            audio_output: SuperGraphLink::new(SuperGraphLinkKind::AudioClip, rng),
            sample_rate_hz,
        }
    }

    pub fn new_and_add(
        builder: &mut SuperGraphBuilder,
        tensor_input: SuperGraphLink,
        sample_rate_hz: u32,
        rng: &mut impl Rng,
    ) -> SuperGraphLink {
        let node = Self::new(builder, tensor_input, sample_rate_hz, rng);
        let output = node.get_audio_output();
        builder.add_node(node.to_any());
        output
    }

    pub fn get_audio_output(&self) -> SuperGraphLink {
        self.audio_output
    }
}

impl SuperGraphNode for SuperGraphNodeTensorToAudioClip {
    fn to_any(self) -> SuperGraphAnyNode {
        SuperGraphAnyNode::TensorToAudioClip(self)
    }

    fn eval<T: SuperGraphObserver>(
        &self,
        _node_path: &[GlobalId],
        data: &mut SuperGraphData,
        _context: &mut SuperGraphContext<T>,
    ) -> Result<(), SuperGraphError> {
        let tensor = data
            .tensors
            .get(&self.tensor_input)
            .ok_or(SuperGraphError::MissingLinkError(format!(
                ": missing tensor input link {:?}",
                self.tensor_input
            )))?
            .clone();
        data.audio_clips.insert(
            self.audio_output,
            SuperGraphAudioClip::new(tensor, self.sample_rate_hz),
        );
        Ok(())
    }

    fn op_kind(&self) -> String {
        "TensorToAudioClip".to_string()
    }
    fn label(&self) -> Option<String> {
        self.label.clone()
    }

    fn inputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(std::iter::once(self.tensor_input.to_any()))
    }

    fn outputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(std::iter::once(self.audio_output.to_any()))
    }

    fn global_id(&self) -> GlobalId {
        self.global_id
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeAudioClipToTensor {
    global_id: GlobalId,
    pub label: Option<String>,
    audio_input: SuperGraphLink,
    tensor_output: SuperGraphLink,
    expected_sample_rate_hz: Option<u32>,
}

impl SuperGraphNodeAudioClipToTensor {
    pub fn new(
        _builder: &mut SuperGraphBuilder,
        audio_input: SuperGraphLink,
        expected_sample_rate_hz: Option<u32>,
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            global_id: GlobalId::new(rng),
            label: None,
            audio_input,
            tensor_output: SuperGraphLink::new(SuperGraphLinkKind::Tensor, rng),
            expected_sample_rate_hz,
        }
    }

    pub fn new_and_add(
        builder: &mut SuperGraphBuilder,
        audio_input: SuperGraphLink,
        expected_sample_rate_hz: Option<u32>,
        rng: &mut impl Rng,
    ) -> SuperGraphLink {
        let node = Self::new(builder, audio_input, expected_sample_rate_hz, rng);
        let output = node.get_tensor_output();
        builder.add_node(node.to_any());
        output
    }

    pub fn get_tensor_output(&self) -> SuperGraphLink {
        self.tensor_output
    }
}

impl SuperGraphNode for SuperGraphNodeAudioClipToTensor {
    fn to_any(self) -> SuperGraphAnyNode {
        SuperGraphAnyNode::AudioClipToTensor(self)
    }

    fn eval<T: SuperGraphObserver>(
        &self,
        _node_path: &[GlobalId],
        data: &mut SuperGraphData,
        _context: &mut SuperGraphContext<T>,
    ) -> Result<(), SuperGraphError> {
        let clip =
            data.audio_clips
                .get(&self.audio_input)
                .ok_or(SuperGraphError::MissingLinkError(format!(
                    ": missing audio input link {:?}",
                    self.audio_input
                )))?;
        if let Some(expected) = self.expected_sample_rate_hz
            && clip.sample_rate_hz != expected
        {
            return Err(SuperGraphError::InvalidInputError(format!(
                "audio sample rate mismatch for {:?}: expected {}, got {}",
                self.audio_input, expected, clip.sample_rate_hz
            )));
        }
        data.tensors
            .insert(self.tensor_output, clip.samples.clone());
        Ok(())
    }

    fn op_kind(&self) -> String {
        "AudioClipToTensor".to_string()
    }
    fn label(&self) -> Option<String> {
        self.label.clone()
    }

    fn inputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(std::iter::once(self.audio_input.to_any()))
    }

    fn outputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(std::iter::once(self.tensor_output.to_any()))
    }

    fn global_id(&self) -> GlobalId {
        self.global_id
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeAudioToMelConfig {
    pub expected_sample_rate_hz: Option<u32>,
    pub n_fft: u32,
    pub hop_length: u32,
    pub center_padding: u32,
    pub max_samples: Option<u32>,
    pub drop_last_frame: bool,
    pub num_mel_bins: u32,
    pub mel_filters: Vec<f32>,
    pub log_floor: f32,
    pub clamp_dynamic_range: Option<f32>,
    pub normalize_add: Option<f32>,
    pub normalize_div: Option<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeAudioClipToMelSpectrogram {
    global_id: GlobalId,
    pub label: Option<String>,
    audio_input: SuperGraphLink,
    tensor_output: SuperGraphLink,
    config: SuperGraphNodeAudioToMelConfig,
}

impl SuperGraphNodeAudioClipToMelSpectrogram {
    pub fn new(
        _builder: &mut SuperGraphBuilder,
        audio_input: SuperGraphLink,
        config: SuperGraphNodeAudioToMelConfig,
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            global_id: GlobalId::new(rng),
            label: None,
            audio_input,
            tensor_output: SuperGraphLink::new(SuperGraphLinkKind::Tensor, rng),
            config,
        }
    }

    pub fn new_and_add(
        builder: &mut SuperGraphBuilder,
        audio_input: SuperGraphLink,
        config: SuperGraphNodeAudioToMelConfig,
        rng: &mut impl Rng,
    ) -> SuperGraphLink {
        let node = Self::new(builder, audio_input, config, rng);
        let output = node.get_tensor_output();
        builder.add_node(node.to_any());
        output
    }

    pub fn get_tensor_output(&self) -> SuperGraphLink {
        self.tensor_output
    }
}

impl SuperGraphNode for SuperGraphNodeAudioClipToMelSpectrogram {
    fn to_any(self) -> SuperGraphAnyNode {
        SuperGraphAnyNode::AudioClipToMelSpectrogram(self)
    }

    fn eval<T: SuperGraphObserver>(
        &self,
        _node_path: &[GlobalId],
        data: &mut SuperGraphData,
        context: &mut SuperGraphContext<T>,
    ) -> Result<(), SuperGraphError> {
        let clip =
            data.audio_clips
                .get(&self.audio_input)
                .ok_or(SuperGraphError::MissingLinkError(format!(
                    ": missing audio input link {:?}",
                    self.audio_input
                )))?;
        if let Some(expected) = self.config.expected_sample_rate_hz
            && clip.sample_rate_hz != expected
        {
            return Err(SuperGraphError::InvalidInputError(format!(
                "audio sample rate mismatch for {:?}: expected {}, got {}",
                self.audio_input, expected, clip.sample_rate_hz
            )));
        }
        if self.config.n_fft == 0 {
            return Err(SuperGraphError::InvalidInputError(
                "audio->mel n_fft must be > 0".to_string(),
            ));
        }
        if self.config.hop_length == 0 {
            return Err(SuperGraphError::InvalidInputError(
                "audio->mel hop_length must be > 0".to_string(),
            ));
        }
        if self.config.num_mel_bins == 0 {
            return Err(SuperGraphError::InvalidInputError(
                "audio->mel num_mel_bins must be > 0".to_string(),
            ));
        }
        if self.config.log_floor <= 0.0 {
            return Err(SuperGraphError::InvalidInputError(
                "audio->mel log_floor must be > 0".to_string(),
            ));
        }
        if let Some(dynamic) = self.config.clamp_dynamic_range
            && dynamic <= 0.0
        {
            return Err(SuperGraphError::InvalidInputError(
                "audio->mel clamp_dynamic_range must be > 0 when set".to_string(),
            ));
        }
        if (self.config.normalize_add.is_some() && self.config.normalize_div.is_none())
            || (self.config.normalize_add.is_none() && self.config.normalize_div.is_some())
        {
            return Err(SuperGraphError::InvalidInputError(
                "audio->mel normalize_add and normalize_div must be set together".to_string(),
            ));
        }
        if let Some(div) = self.config.normalize_div
            && div == 0.0
        {
            return Err(SuperGraphError::InvalidInputError(
                "audio->mel normalize_div must be non-zero".to_string(),
            ));
        }

        let n_fft = self.config.n_fft as usize;
        let hop_length = self.config.hop_length as usize;
        let center_padding = self.config.center_padding as usize;
        let num_mel_bins = self.config.num_mel_bins as usize;
        let n_freqs = n_fft / 2 + 1;
        if self.config.mel_filters.len() != num_mel_bins * n_freqs {
            return Err(SuperGraphError::InvalidInputError(format!(
                "audio->mel mel_filters length mismatch: expected {}, got {}",
                num_mel_bins * n_freqs,
                self.config.mel_filters.len()
            )));
        }

        let audio_f32 = clip.samples.cast(DType::F32, context.eval_backend)?;
        let audio_nd = audio_f32.to_ndarray()?;
        let mut samples: Vec<f32> = audio_nd.flatten().try_into().map_err(|_| {
            SuperGraphError::InvalidInputError(
                "audio->mel failed to flatten audio tensor into f32 samples".to_string(),
            )
        })?;

        if let Some(max_samples) = self.config.max_samples {
            let max_samples = max_samples as usize;
            if samples.len() > max_samples {
                samples.truncate(max_samples);
            } else if samples.len() < max_samples {
                samples.resize(max_samples, 0.0);
            }
        }
        if samples.is_empty() {
            return Err(SuperGraphError::InvalidInputError(
                "audio->mel input has no samples".to_string(),
            ));
        }

        let mut padded = vec![0.0f32; center_padding + samples.len() + center_padding];
        padded[center_padding..center_padding + samples.len()].copy_from_slice(&samples);

        if padded.len() < n_fft {
            return Err(SuperGraphError::InvalidInputError(format!(
                "audio->mel padded input too short: len={} n_fft={}",
                padded.len(),
                n_fft
            )));
        }

        let stft_frames = (padded.len() - n_fft) / hop_length + 1;
        let num_frames = if self.config.drop_last_frame {
            stft_frames.saturating_sub(1)
        } else {
            stft_frames
        };
        if num_frames == 0 {
            return Err(SuperGraphError::InvalidInputError(
                "audio->mel produced zero frames".to_string(),
            ));
        }

        let window: Vec<f32> = (0..n_fft)
            .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / n_fft as f32).cos()))
            .collect();

        let mut cos_table = vec![0.0f32; n_freqs * n_fft];
        let mut sin_table = vec![0.0f32; n_freqs * n_fft];
        for k in 0..n_freqs {
            for n in 0..n_fft {
                let angle = -2.0 * std::f32::consts::PI * k as f32 * n as f32 / n_fft as f32;
                let idx = k * n_fft + n;
                cos_table[idx] = angle.cos();
                sin_table[idx] = angle.sin();
            }
        }

        let mut frame_buf = vec![0.0f32; n_fft];
        let mut magnitudes = vec![0.0f32; n_freqs * num_frames];
        for frame in 0..num_frames {
            let start = frame * hop_length;
            for i in 0..n_fft {
                frame_buf[i] = padded[start + i] * window[i];
            }
            for k in 0..n_freqs {
                let mut re = 0.0f32;
                let mut im = 0.0f32;
                let trig_base = k * n_fft;
                for n in 0..n_fft {
                    let x = frame_buf[n];
                    re += x * cos_table[trig_base + n];
                    im += x * sin_table[trig_base + n];
                }
                magnitudes[k * num_frames + frame] = re * re + im * im;
            }
        }

        let mut mel_spec = vec![0.0f32; num_mel_bins * num_frames];
        for mel_idx in 0..num_mel_bins {
            let filter_row = &self.config.mel_filters[mel_idx * n_freqs..(mel_idx + 1) * n_freqs];
            for frame in 0..num_frames {
                let mut sum = 0.0f32;
                for freq in 0..n_freqs {
                    sum += filter_row[freq] * magnitudes[freq * num_frames + frame];
                }
                mel_spec[mel_idx * num_frames + frame] = sum;
            }
        }

        let mut output = mel_spec
            .into_iter()
            .map(|x| x.max(self.config.log_floor).log10())
            .collect::<Vec<_>>();

        if let Some(dynamic) = self.config.clamp_dynamic_range {
            let max_val = output.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let min_val = max_val - dynamic;
            for x in &mut output {
                if *x < min_val {
                    *x = min_val;
                }
            }
        }

        if let (Some(add), Some(div)) = (self.config.normalize_add, self.config.normalize_div) {
            for x in &mut output {
                *x = (*x + add) / div;
            }
        }

        let mel_tensor =
            NumericTensor::<DynRank>::from_vec_shape(output, vec![1, num_mel_bins, num_frames])?;
        data.tensors.insert(self.tensor_output, mel_tensor);
        Ok(())
    }

    fn op_kind(&self) -> String {
        "AudioClipToMelSpectrogram".to_string()
    }
    fn label(&self) -> Option<String> {
        self.label.clone()
    }

    fn inputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(std::iter::once(self.audio_input.to_any()))
    }

    fn outputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(std::iter::once(self.tensor_output.to_any()))
    }

    fn global_id(&self) -> GlobalId {
        self.global_id
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeMilliOpGraph {
    global_id: GlobalId,
    pub label: Option<String>,
    pub graph: MilliOpGraph,
}

impl SuperGraphNodeMilliOpGraph {
    pub fn new(graph: MilliOpGraph, rng: &mut impl Rng) -> Self {
        Self {
            graph,
            global_id: GlobalId::new(rng),
            label: None,
        }
    }
}

struct MilliOpGraphObserverWrapper<'a, T: SuperGraphObserver> {
    inner: &'a mut T,
    node_path: Vec<GlobalId>,
}

impl<'a, T: SuperGraphObserver> MilliOpGraphObserverWrapper<'a, T> {
    fn new(inner: &'a mut T, node_path: &[GlobalId]) -> Self {
        Self {
            inner,
            node_path: node_path.to_vec(),
        }
    }
}

impl<'a, T: SuperGraphObserver> MilliOpGraphObserver for MilliOpGraphObserverWrapper<'a, T> {
    fn on_tensor_assigned(
        &mut self,
        tensor_path: &[GlobalId],
        tensor: &NumericTensor<DynRank>,
        backend: &mut EvalBackend,
    ) {
        let tensor_path = self
            .node_path
            .iter()
            .chain(tensor_path.iter())
            .copied()
            .collect::<Vec<_>>();
        self.inner
            .on_tensor_assigned(tensor_path.as_slice(), tensor, backend);
    }

    fn on_node_executed(
        &mut self,
        node_path: &[GlobalId],
        start_instant: Instant,
        end_instant: Instant,
        backend: &mut EvalBackend,
    ) {
        let node_path = self
            .node_path
            .iter()
            .chain(node_path.iter())
            .copied()
            .collect::<Vec<_>>();
        self.inner.on_node_executed(
            node_path.as_slice(),
            "",
            start_instant,
            end_instant,
            backend,
        );
    }

    fn should_cancel(&mut self) -> bool {
        self.inner.should_cancel()
    }
}

impl SuperGraphNode for SuperGraphNodeMilliOpGraph {
    fn to_any(self) -> SuperGraphAnyNode {
        SuperGraphAnyNode::MilliOpGraph(self)
    }
    fn eval<T: SuperGraphObserver>(
        &self,
        node_path: &[GlobalId],
        data: &mut SuperGraphData,
        context: &mut SuperGraphContext<T>,
    ) -> Result<(), SuperGraphError> {
        let inputs = {
            let mut inputs = HashMap::new();
            for input in self.graph.get_inputs() {
                inputs.insert(
                    input,
                    data.tensors
                        .get(&SuperGraphLink::tensor(input))
                        .unwrap()
                        .clone(),
                );
            }
            inputs
        };
        let node_path = node_path
            .iter()
            .chain(core::iter::once(&self.global_id))
            .copied()
            .collect::<Vec<_>>();
        let mut observer = MilliOpGraphObserverWrapper::new(context.observer, node_path.as_slice());
        let res = self
            .graph
            .eval(&inputs, &mut observer, context.eval_backend)?;
        data.tensors
            .extend(res.map(|(k, v)| (SuperGraphLink::tensor(k), v)));
        Ok(())
    }
    fn op_kind(&self) -> String {
        "MilliOpGraph".to_string()
    }
    fn label(&self) -> Option<String> {
        self.label.clone()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(
            self.graph
                .input_link_ids()
                .map(|(a, _b)| SuperGraphLink::tensor(a).to_any()),
        )
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(
            self.graph
                .output_link_ids()
                .map(|(a, _b)| SuperGraphLink::tensor(a).to_any()),
        )
    }
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeScan {
    global_id: GlobalId,
    pub label: Option<String>,
    inner_graph: SuperGraph,
    iteration_count: SuperGraphLink,
    simple_inputs: Vec<SuperGraphLinkDouble>,
    state_links: Vec<SuperGraphLinkTriple>,
    scan_inputs: Vec<(SuperGraphLink, SuperGraphLink, u32)>,
    scan_outputs: Vec<(SuperGraphLink, SuperGraphLink, u32)>,
    simple_outputs: Vec<SuperGraphLinkDouble>,
}

impl SuperGraphNodeScan {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        inner_graph: SuperGraph,
        iteration_count: SuperGraphLink,
        simple_inputs: Vec<SuperGraphLinkDouble>,
        state_links: Vec<SuperGraphLinkTriple>,
        scan_inputs: Vec<(SuperGraphLink, SuperGraphLink, u32)>,
        scan_outputs: Vec<(SuperGraphLink, SuperGraphLink, u32)>,
        simple_outputs: Vec<SuperGraphLinkDouble>,
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            inner_graph,
            iteration_count,
            simple_inputs,
            state_links,
            scan_inputs,
            scan_outputs,
            simple_outputs,
            global_id: GlobalId::new(rng),
            label: None,
        }
    }
}

impl SuperGraphNode for SuperGraphNodeScan {
    fn to_any(self) -> SuperGraphAnyNode {
        SuperGraphAnyNode::Scan(self)
    }

    fn eval<'a, 'b, 'c, 'd, T: SuperGraphObserver>(
        &'a self,
        node_path: &[GlobalId],
        data: &mut SuperGraphData<'b>,
        context: &mut SuperGraphContext<'a, 'b, 'c, 'd, T>,
    ) -> Result<(), SuperGraphError> {
        let iteration_count_tensor =
            data.tensors
                .get(&self.iteration_count)
                .ok_or(SuperGraphError::MissingLinkError(format!(
                    ": scan iteration_count {:?}",
                    self.iteration_count
                )))?;
        let iteration_count: i64 = iteration_count_tensor.first_element().into();
        if iteration_count < 0 {
            return Err(SuperGraphError::InvalidInputError(format!(
                "scan iteration_count must be non-negative, got {iteration_count}"
            )));
        }
        let iteration_count = iteration_count as u64;

        let node_path = node_path
            .iter()
            .chain(core::iter::once(&self.global_id))
            .copied()
            .collect::<Vec<_>>();

        let simple_inputs = {
            let mut simple_inputs = SuperGraphData::new();
            for link in &self.simple_inputs {
                simple_inputs.copy_link_from(data, link.first(), link.second())?;
            }
            simple_inputs
        };

        let mut state_values = {
            let mut state_values = SuperGraphData::new();
            for link in &self.state_links {
                state_values.copy_link_from(data, link.first(), link.second())?;
            }
            state_values
        };

        let mut prev_iter_outputs: Option<SuperGraphData> = None;

        let mut output_scan_tensor_parts = HashMap::new();
        for (_inner, outer, scan_axis) in &self.scan_outputs {
            output_scan_tensor_parts.insert(*outer, (Vec::new(), *scan_axis as usize));
        }

        for i in 0..iteration_count {
            if context.observer.should_cancel() {
                return Err(SuperGraphError::Cancelled);
            }
            let iter_inputs = {
                let mut iter_inputs = simple_inputs.clone();
                iter_inputs.extend(&state_values);
                for (outer, inner, scan_axis) in &self.scan_inputs {
                    let tensor =
                        data.tensors
                            .get(outer)
                            .ok_or(SuperGraphError::MissingLinkError(format!(
                                ": scan_input outer={:?} inner={:?}",
                                outer, inner
                            )))?;
                    let slice_arg = {
                        let mut slice_ranges = Vec::new();
                        for j in 0..tensor.rank() {
                            if j == *scan_axis as usize {
                                slice_ranges.push(i..i + 1);
                            } else {
                                slice_ranges.push(0..tensor.shape()[j]);
                            }
                        }
                        slice_ranges
                    };
                    let sliced = tensor.slice(slice_arg.as_slice(), context.eval_backend)?;
                    let squeezed = sliced.squeeze(*scan_axis as usize)?;
                    iter_inputs.tensors.insert(*inner, squeezed);
                }
                iter_inputs
            };
            let iter_outputs = self
                .inner_graph
                .eval(node_path.as_slice(), iter_inputs, context)?;

            for (inner, outer, _scan_axis) in &self.scan_outputs {
                let tensor = iter_outputs
                    .tensors
                    .get(inner)
                    .ok_or(SuperGraphError::MissingLinkError(String::new()))?;
                let (tensors, _) = output_scan_tensor_parts
                    .get_mut(outer)
                    .ok_or(SuperGraphError::MissingLinkError(String::new()))?;
                tensors.push(tensor.clone());
            }

            state_values = {
                let mut state_values = SuperGraphData::new();
                for link in &self.state_links {
                    state_values.copy_link_from(&iter_outputs, link.third(), link.second())?;
                }
                state_values
            };

            prev_iter_outputs = Some(iter_outputs);
        }

        let mut output_data = SuperGraphData::new();
        let prev_iter_outputs = if iteration_count == 0 {
            None
        } else {
            Some(
                prev_iter_outputs
                    .as_ref()
                    .ok_or(SuperGraphError::InvalidInputError(
                        "scan had iterations but no outputs were produced".to_string(),
                    ))?,
            )
        };

        for link in &self.simple_outputs {
            output_data.copy_link_from(
                prev_iter_outputs.ok_or(SuperGraphError::InvalidInputError(
                    "scan simple_outputs are unavailable when iteration_count is 0".to_string(),
                ))?,
                link.first(),
                link.second(),
            )?;
        }

        for (link, (parts, axis)) in output_scan_tensor_parts {
            if parts.is_empty() {
                return Err(SuperGraphError::InvalidInputError(format!(
                    "scan output {:?} has no parts; iteration_count is likely 0",
                    link
                )));
            }
            let unsqueezed = parts
                .into_iter()
                .map(|tensor| tensor.unsqueeze(axis))
                .collect::<Result<Vec<_>, _>>()?;
            let unsqueezed_ref = unsqueezed.iter().collect::<Vec<_>>();
            output_data.tensors.insert(
                link,
                NumericTensor::<DynRank>::concat(
                    unsqueezed_ref.as_slice(),
                    axis,
                    context.eval_backend,
                )?,
            );
        }

        data.extend(&output_data);
        Ok(())
    }
    fn op_kind(&self) -> String {
        "Scan".to_string()
    }
    fn label(&self) -> Option<String> {
        self.label.clone()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
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
        Box::new(inputs.into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        let mut outputs = Vec::new();
        for link in &self.simple_outputs {
            outputs.push(link.second());
        }
        for (_, output, _) in &self.scan_outputs {
            outputs.push(output.to_any());
        }
        Box::new(outputs.into_iter())
    }
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeReportProgress {
    global_id: GlobalId,
    pub label: Option<String>,
    tier_input: SuperGraphLink,
    numerator_input: SuperGraphLink,
    denominator_input: SuperGraphLink,
}

impl SuperGraphNodeReportProgress {
    pub fn new(
        tier_input: SuperGraphLink,
        numerator_input: SuperGraphLink,
        denominator_input: SuperGraphLink,
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            global_id: GlobalId::new(rng),
            label: None,
            tier_input,
            numerator_input,
            denominator_input,
        }
    }
}

impl SuperGraphNode for SuperGraphNodeReportProgress {
    fn to_any(self) -> SuperGraphAnyNode {
        SuperGraphAnyNode::ReportProgress(self)
    }

    fn eval<T: SuperGraphObserver>(
        &self,
        node_path: &[GlobalId],
        data: &mut SuperGraphData,
        context: &mut SuperGraphContext<T>,
    ) -> Result<(), SuperGraphError> {
        let tier = read_rank0_i64_tensor(
            data.tensors
                .get(&self.tier_input)
                .ok_or(SuperGraphError::MissingLinkError(format!(
                    ": report_progress tier_input {:?}",
                    self.tier_input
                )))?,
            "ReportProgress.tier_input",
            context.eval_backend,
        )?;
        let numerator = read_rank0_f64_tensor(
            data.tensors
                .get(&self.numerator_input)
                .ok_or(SuperGraphError::MissingLinkError(format!(
                    ": report_progress numerator_input {:?}",
                    self.numerator_input
                )))?,
            "ReportProgress.numerator_input",
            context.eval_backend,
        )?;
        let denominator = read_rank0_f64_tensor(
            data.tensors
                .get(&self.denominator_input)
                .ok_or(SuperGraphError::MissingLinkError(format!(
                    ": report_progress denominator_input {:?}",
                    self.denominator_input
                )))?,
            "ReportProgress.denominator_input",
            context.eval_backend,
        )?;
        let path = node_path
            .iter()
            .chain(core::iter::once(&self.global_id))
            .copied()
            .collect::<Vec<_>>();
        context
            .observer
            .on_progress(path.as_slice(), tier, numerator, denominator);
        Ok(())
    }

    fn op_kind(&self) -> String {
        "ReportProgress".to_string()
    }
    fn label(&self) -> Option<String> {
        self.label.clone()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(
            [
                self.tier_input.to_any(),
                self.numerator_input.to_any(),
                self.denominator_input.to_any(),
            ]
            .into_iter(),
        )
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(std::iter::empty())
    }
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeRNNCacheRead {
    global_id: GlobalId,
    pub label: Option<String>,
    key_input: SuperGraphLink,
    tokens_input: SuperGraphLink,
    tokens_output: SuperGraphLink,
    state_outputs: Vec<(String, SuperGraphLink)>,
    default_state_inputs: Vec<(String, SuperGraphLink)>,
}

impl SuperGraphNodeRNNCacheRead {
    pub fn new(
        key_input: SuperGraphLink,
        tokens_input: SuperGraphLink,
        tokens_output: SuperGraphLink,
        state_outputs: Vec<(String, SuperGraphLink)>,
        default_state_inputs: Vec<(String, SuperGraphLink)>,
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            global_id: GlobalId::new(rng),
            label: None,
            key_input,
            tokens_input,
            tokens_output,
            state_outputs,
            default_state_inputs,
        }
    }
}

impl SuperGraphNode for SuperGraphNodeRNNCacheRead {
    fn to_any(self) -> SuperGraphAnyNode {
        SuperGraphAnyNode::RNNCacheRead(self)
    }

    fn eval<T: SuperGraphObserver>(
        &self,
        _node_path: &[GlobalId],
        data: &mut SuperGraphData,
        context: &mut SuperGraphContext<T>,
    ) -> Result<(), SuperGraphError> {
        let tokens_input = data
            .tensors
            .get(&self.tokens_input)
            .ok_or(SuperGraphError::MissingLinkError(String::new()))?
            .clone();
        let mut found = false;
        if let Some(caches) = &mut context.caches {
            let key_input = *data
                .hashes
                .get(&self.key_input)
                .ok_or(SuperGraphError::MissingLinkError(String::new()))?;
            if let Some(rnn_cache) = caches.rnn_cache.get(&key_input) {
                let tokens_vec: Vec<u32> = tokens_input
                    .to_ndarray()?
                    .try_to_rank::<P1>()
                    .unwrap()
                    .try_to_vec()
                    .unwrap();
                // Try to match as many tokens as possible

                for i in (1..tokens_vec.len()).rev() {
                    let matched_tokens = &tokens_vec[..i].to_vec();
                    if let Some(state) = rnn_cache.get(matched_tokens) {
                        found = true;
                        for (key, value) in state.iter() {
                            if let Some((_, output_link)) =
                                self.state_outputs.iter().find(|(k, _)| k == key)
                            {
                                data.tensors.insert(*output_link, value.clone());
                            }
                        }
                        // Emit remaining tokens
                        let remaining_tokens = &tokens_vec[i..];
                        let remaining_tokens_tensor = NumericTensor::from(
                            NDArrayNumericTensor::from_vec(remaining_tokens.to_vec()).to_dyn(),
                        );
                        data.tensors
                            .insert(self.tokens_output, remaining_tokens_tensor.clone());
                        break;
                    }
                }
            }
        }

        if !found {
            // Emit default state
            for (key, value) in self.default_state_inputs.iter() {
                let value = data
                    .tensors
                    .get(value)
                    .ok_or(SuperGraphError::MissingLinkError(String::new()))?;
                if let Some((_, output_link)) = self.state_outputs.iter().find(|x| x.0 == *key) {
                    data.tensors.insert(*output_link, value.clone());
                }
            }
            data.tensors
                .insert(self.tokens_output, tokens_input.clone());
        }
        Ok(())
    }

    fn op_kind(&self) -> String {
        "RNNCacheRead".to_string()
    }
    fn label(&self) -> Option<String> {
        self.label.clone()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(
            [self.key_input.to_any(), self.tokens_input.to_any()]
                .into_iter()
                .chain(self.default_state_inputs.iter().map(|x| x.1.to_any())),
        )
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(
            std::iter::once(self.tokens_output.to_any())
                .chain(self.state_outputs.iter().map(|x| x.1.to_any())),
        )
    }
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeRNNCacheWrite {
    global_id: GlobalId,
    pub label: Option<String>,
    key_input: SuperGraphLink,
    tokens_input: SuperGraphLink,
    state_inputs: Vec<(String, SuperGraphLink)>,
}

impl SuperGraphNodeRNNCacheWrite {
    pub fn new(
        key_input: SuperGraphLink,
        tokens_input: SuperGraphLink,
        state_inputs: Vec<(String, SuperGraphLink)>,
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            global_id: GlobalId::new(rng),
            label: None,
            key_input,
            tokens_input,
            state_inputs,
        }
    }
}

impl SuperGraphNode for SuperGraphNodeRNNCacheWrite {
    fn to_any(self) -> SuperGraphAnyNode {
        SuperGraphAnyNode::RNNCacheWrite(self)
    }

    fn eval<T: SuperGraphObserver>(
        &self,
        _node_path: &[GlobalId],
        data: &mut SuperGraphData,
        context: &mut SuperGraphContext<T>,
    ) -> Result<(), SuperGraphError> {
        if let Some(caches) = &mut context.caches {
            let key_input = *data
                .hashes
                .get(&self.key_input)
                .ok_or(SuperGraphError::MissingLinkError(String::new()))?;
            let tokens_input = data
                .tensors
                .get(&self.tokens_input)
                .ok_or(SuperGraphError::MissingLinkError(String::new()))?;
            let tokens_vec: Vec<u32> = tokens_input
                .to_ndarray()?
                .try_to_rank::<P1>()
                .unwrap()
                .try_to_vec()
                .unwrap();
            let state_inputs: HashMap<String, &NumericTensor<DynRank>> = self
                .state_inputs
                .iter()
                .map(|(k, v)| (k.clone(), data.tensors.get(v).unwrap()))
                .collect();
            caches.rnn_cache.entry(key_input).or_default().insert(
                tokens_vec,
                state_inputs
                    .into_iter()
                    .map(|(k, v)| (k, v.clone()))
                    .collect(),
            );
        }
        Ok(())
    }
    fn op_kind(&self) -> String {
        "RNNCacheWrite".to_string()
    }
    fn label(&self) -> Option<String> {
        self.label.clone()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(
            [self.key_input.to_any(), self.tokens_input.to_any()]
                .into_iter()
                .chain(self.state_inputs.iter().map(|x| x.1.to_any())),
        )
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(std::iter::empty())
    }
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeTensorCacheRead {
    global_id: GlobalId,
    pub label: Option<String>,
    key_input: SuperGraphLink,
    default_input: SuperGraphLink,
    value_output: SuperGraphLink,
    hit_output: SuperGraphLink,
}

impl SuperGraphNodeTensorCacheRead {
    pub fn new(
        key_input: SuperGraphLink,
        default_input: SuperGraphLink,
        value_output: SuperGraphLink,
        hit_output: SuperGraphLink,
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            global_id: GlobalId::new(rng),
            label: None,
            key_input,
            default_input,
            value_output,
            hit_output,
        }
    }
}

impl SuperGraphNode for SuperGraphNodeTensorCacheRead {
    fn to_any(self) -> SuperGraphAnyNode {
        SuperGraphAnyNode::TensorCacheRead(self)
    }

    fn eval<T: SuperGraphObserver>(
        &self,
        _node_path: &[GlobalId],
        data: &mut SuperGraphData,
        context: &mut SuperGraphContext<T>,
    ) -> Result<(), SuperGraphError> {
        let key_input = *data
            .hashes
            .get(&self.key_input)
            .ok_or(SuperGraphError::MissingLinkError(String::new()))?;
        let default_input = data
            .tensors
            .get(&self.default_input)
            .ok_or(SuperGraphError::MissingLinkError(String::new()))?
            .clone();
        let mut output = default_input;
        let mut hit = false;
        if let Some(caches) = &mut context.caches
            && let Some(value) = caches.tensor_cache.get(&key_input)
        {
            output = value.clone();
            hit = true;
        }
        data.tensors.insert(self.value_output, output);
        data.tensors
            .insert(self.hit_output, tensor_bool_scalar(hit)?);
        Ok(())
    }

    fn op_kind(&self) -> String {
        "TensorCacheRead".to_string()
    }
    fn label(&self) -> Option<String> {
        self.label.clone()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new([self.key_input.to_any(), self.default_input.to_any()].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new([self.value_output.to_any(), self.hit_output.to_any()].into_iter())
    }
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeTensorCacheWrite {
    global_id: GlobalId,
    pub label: Option<String>,
    key_input: SuperGraphLink,
    value_input: SuperGraphLink,
    write_enable_input: SuperGraphLink,
}

impl SuperGraphNodeTensorCacheWrite {
    pub fn new(
        key_input: SuperGraphLink,
        value_input: SuperGraphLink,
        write_enable_input: SuperGraphLink,
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            global_id: GlobalId::new(rng),
            label: None,
            key_input,
            value_input,
            write_enable_input,
        }
    }
}

impl SuperGraphNode for SuperGraphNodeTensorCacheWrite {
    fn to_any(self) -> SuperGraphAnyNode {
        SuperGraphAnyNode::TensorCacheWrite(self)
    }

    fn eval<T: SuperGraphObserver>(
        &self,
        _node_path: &[GlobalId],
        data: &mut SuperGraphData,
        context: &mut SuperGraphContext<T>,
    ) -> Result<(), SuperGraphError> {
        let write_enable = read_rank0_bool_tensor(
            data.tensors
                .get(&self.write_enable_input)
                .ok_or(SuperGraphError::MissingLinkError(String::new()))?,
            "TensorCacheWrite.write_enable_input",
        )?;
        if write_enable && let Some(caches) = &mut context.caches {
            let key_input = *data
                .hashes
                .get(&self.key_input)
                .ok_or(SuperGraphError::MissingLinkError(String::new()))?;
            let value_input = data
                .tensors
                .get(&self.value_input)
                .ok_or(SuperGraphError::MissingLinkError(String::new()))?
                .clone();
            caches.tensor_cache.insert(key_input, value_input);
        }
        Ok(())
    }
    fn op_kind(&self) -> String {
        "TensorCacheWrite".to_string()
    }
    fn label(&self) -> Option<String> {
        self.label.clone()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(
            [
                self.key_input.to_any(),
                self.value_input.to_any(),
                self.write_enable_input.to_any(),
            ]
            .into_iter(),
        )
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(std::iter::empty())
    }
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeTensorPackCacheRead {
    global_id: GlobalId,
    pub label: Option<String>,
    key_input: SuperGraphLink,
    value_outputs: Vec<(String, SuperGraphLink)>,
    default_value_inputs: Vec<(String, SuperGraphLink)>,
    hit_output: SuperGraphLink,
}

impl SuperGraphNodeTensorPackCacheRead {
    pub fn new(
        key_input: SuperGraphLink,
        value_outputs: Vec<(String, SuperGraphLink)>,
        default_value_inputs: Vec<(String, SuperGraphLink)>,
        hit_output: SuperGraphLink,
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            global_id: GlobalId::new(rng),
            label: None,
            key_input,
            value_outputs,
            default_value_inputs,
            hit_output,
        }
    }
}

impl SuperGraphNode for SuperGraphNodeTensorPackCacheRead {
    fn to_any(self) -> SuperGraphAnyNode {
        SuperGraphAnyNode::TensorPackCacheRead(self)
    }

    fn eval<T: SuperGraphObserver>(
        &self,
        _node_path: &[GlobalId],
        data: &mut SuperGraphData,
        context: &mut SuperGraphContext<T>,
    ) -> Result<(), SuperGraphError> {
        let key_input = *data
            .hashes
            .get(&self.key_input)
            .ok_or(SuperGraphError::MissingLinkError(String::new()))?;
        let default_values_by_name: HashMap<String, NumericTensor<DynRank>> = self
            .default_value_inputs
            .iter()
            .map(|(name, input)| {
                Ok((
                    name.clone(),
                    data.tensors
                        .get(input)
                        .ok_or(SuperGraphError::MissingLinkError(String::new()))?
                        .clone(),
                ))
            })
            .collect::<Result<_, SuperGraphError>>()?;

        let mut hit = false;
        if let Some(caches) = &mut context.caches
            && let Some(cached_values) = caches.tensor_pack_cache.get(&key_input)
        {
            let full_hit = self
                .value_outputs
                .iter()
                .all(|(name, _)| cached_values.contains_key(name));
            if full_hit {
                for (name, output) in &self.value_outputs {
                    data.tensors
                        .insert(*output, cached_values.get(name).unwrap().clone());
                }
                hit = true;
            }
        }

        if !hit {
            for (name, output) in &self.value_outputs {
                let default_value =
                    default_values_by_name
                        .get(name)
                        .ok_or(SuperGraphError::InvalidInputError(format!(
                            "TensorPackCacheRead missing default input for key '{name}'"
                        )))?;
                data.tensors.insert(*output, default_value.clone());
            }
        }

        data.tensors
            .insert(self.hit_output, tensor_bool_scalar(hit)?);
        Ok(())
    }

    fn op_kind(&self) -> String {
        "TensorPackCacheRead".to_string()
    }
    fn label(&self) -> Option<String> {
        self.label.clone()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(
            std::iter::once(self.key_input.to_any())
                .chain(self.default_value_inputs.iter().map(|(_, x)| x.to_any())),
        )
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(
            self.value_outputs
                .iter()
                .map(|(_, x)| x.to_any())
                .chain(std::iter::once(self.hit_output.to_any())),
        )
    }
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeTensorPackCacheWrite {
    global_id: GlobalId,
    pub label: Option<String>,
    key_input: SuperGraphLink,
    value_inputs: Vec<(String, SuperGraphLink)>,
    write_enable_input: SuperGraphLink,
}

impl SuperGraphNodeTensorPackCacheWrite {
    pub fn new(
        key_input: SuperGraphLink,
        value_inputs: Vec<(String, SuperGraphLink)>,
        write_enable_input: SuperGraphLink,
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            global_id: GlobalId::new(rng),
            label: None,
            key_input,
            value_inputs,
            write_enable_input,
        }
    }
}

impl SuperGraphNode for SuperGraphNodeTensorPackCacheWrite {
    fn to_any(self) -> SuperGraphAnyNode {
        SuperGraphAnyNode::TensorPackCacheWrite(self)
    }

    fn eval<T: SuperGraphObserver>(
        &self,
        _node_path: &[GlobalId],
        data: &mut SuperGraphData,
        context: &mut SuperGraphContext<T>,
    ) -> Result<(), SuperGraphError> {
        let write_enable = read_rank0_bool_tensor(
            data.tensors
                .get(&self.write_enable_input)
                .ok_or(SuperGraphError::MissingLinkError(String::new()))?,
            "TensorPackCacheWrite.write_enable_input",
        )?;
        if write_enable && let Some(caches) = &mut context.caches {
            let key_input = *data
                .hashes
                .get(&self.key_input)
                .ok_or(SuperGraphError::MissingLinkError(String::new()))?;
            let values = self
                .value_inputs
                .iter()
                .map(|(name, input)| {
                    Ok((
                        name.clone(),
                        data.tensors
                            .get(input)
                            .ok_or(SuperGraphError::MissingLinkError(String::new()))?
                            .clone(),
                    ))
                })
                .collect::<Result<HashMap<String, NumericTensor<DynRank>>, SuperGraphError>>()?;
            caches.tensor_pack_cache.insert(key_input, values);
        }
        Ok(())
    }
    fn op_kind(&self) -> String {
        "TensorPackCacheWrite".to_string()
    }
    fn label(&self) -> Option<String> {
        self.label.clone()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(
            std::iter::once(self.key_input.to_any())
                .chain(std::iter::once(self.write_enable_input.to_any()))
                .chain(self.value_inputs.iter().map(|(_, x)| x.to_any())),
        )
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(std::iter::empty())
    }
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[allow(clippy::large_enum_variant)]
pub enum SuperGraphAnyNode {
    ModelExecution(SuperGraphNodeModelExecution),
    TokenizerEncode(SuperGraphNodeTokenizerEncode),
    TokenizerDecode(SuperGraphNodeTokenizerDecode),
    TokenizerLoad(SuperGraphNodeTokenizerLoad),
    TextToPhonemes(SuperGraphNodeTextToPhonemes),
    PiperPhonemesToTensor(SuperGraphNodePiperPhonemesToTensor),
    KokoroPhonemesToTensor(SuperGraphNodeKokoroPhonemesToTensor),
    F5TextToTensor(SuperGraphNodeF5TextToTensor),
    TensorToImage(SuperGraphNodeTensorToImage),
    TensorToAudioClip(SuperGraphNodeTensorToAudioClip),
    AudioClipToTensor(SuperGraphNodeAudioClipToTensor),
    AudioClipToMelSpectrogram(SuperGraphNodeAudioClipToMelSpectrogram),
    MilliOpGraph(SuperGraphNodeMilliOpGraph),
    Scan(SuperGraphNodeScan),
    ReportProgress(SuperGraphNodeReportProgress),
    RNNCacheWrite(SuperGraphNodeRNNCacheWrite),
    RNNCacheRead(SuperGraphNodeRNNCacheRead),
    TensorCacheRead(SuperGraphNodeTensorCacheRead),
    TensorCacheWrite(SuperGraphNodeTensorCacheWrite),
    TensorPackCacheRead(SuperGraphNodeTensorPackCacheRead),
    TensorPackCacheWrite(SuperGraphNodeTensorPackCacheWrite),
}

macro_rules! delegate {
    ($name:ident($($arg:ident: $ty:ty),*) -> $ret:ty) => {
        fn $name(&self, $($arg: $ty),*) -> $ret {
            match self {
                SuperGraphAnyNode::ModelExecution(x) => SuperGraphNode::$name(x,$($arg),*),
                SuperGraphAnyNode::TokenizerEncode(x) => SuperGraphNode::$name(x,$($arg),*),
                SuperGraphAnyNode::TokenizerDecode(x) => SuperGraphNode::$name(x,$($arg),*),
                SuperGraphAnyNode::TokenizerLoad(x) => SuperGraphNode::$name(x,$($arg),*),
                SuperGraphAnyNode::TextToPhonemes(x) => SuperGraphNode::$name(x,$($arg),*),
                SuperGraphAnyNode::PiperPhonemesToTensor(x) => SuperGraphNode::$name(x,$($arg),*),
                SuperGraphAnyNode::KokoroPhonemesToTensor(x) => SuperGraphNode::$name(x,$($arg),*),
                SuperGraphAnyNode::F5TextToTensor(x) => SuperGraphNode::$name(x,$($arg),*),
                SuperGraphAnyNode::TensorToImage(x) => SuperGraphNode::$name(x,$($arg),*),
                SuperGraphAnyNode::TensorToAudioClip(x) => SuperGraphNode::$name(x,$($arg),*),
                SuperGraphAnyNode::AudioClipToTensor(x) => SuperGraphNode::$name(x,$($arg),*),
                SuperGraphAnyNode::AudioClipToMelSpectrogram(x) => SuperGraphNode::$name(x,$($arg),*),
                SuperGraphAnyNode::MilliOpGraph(x) => SuperGraphNode::$name(x,$($arg),*),
                SuperGraphAnyNode::Scan(x) => SuperGraphNode::$name(x,$($arg),*),
                SuperGraphAnyNode::ReportProgress(x) => SuperGraphNode::$name(x,$($arg),*),
                SuperGraphAnyNode::RNNCacheRead(x) => SuperGraphNode::$name(x,$($arg),*),
                SuperGraphAnyNode::RNNCacheWrite(x) => SuperGraphNode::$name(x,$($arg),*),
                SuperGraphAnyNode::TensorCacheRead(x) => SuperGraphNode::$name(x,$($arg),*),
                SuperGraphAnyNode::TensorCacheWrite(x) => SuperGraphNode::$name(x,$($arg),*),
                SuperGraphAnyNode::TensorPackCacheRead(x) => SuperGraphNode::$name(x,$($arg),*),
                SuperGraphAnyNode::TensorPackCacheWrite(x) => SuperGraphNode::$name(x,$($arg),*),
            }
        }
    }
}

impl SuperGraphAnyNode {
    pub fn get_sub_graph(&self) -> Option<&SuperGraph> {
        match self {
            SuperGraphAnyNode::Scan(x) => Some(&x.inner_graph),
            _ => None,
        }
    }
}

impl SuperGraphNode for SuperGraphAnyNode {
    fn to_any(self) -> SuperGraphAnyNode {
        self
    }

    fn eval<'a, 'b, 'c, 'd, T: SuperGraphObserver>(
        &'a self,
        node_path: &[GlobalId],
        data: &mut SuperGraphData<'b>,
        context: &mut SuperGraphContext<'a, 'b, 'c, 'd, T>,
    ) -> Result<(), SuperGraphError> {
        match self {
            SuperGraphAnyNode::ModelExecution(node) => node.eval(node_path, data, context),
            SuperGraphAnyNode::TokenizerEncode(node) => node.eval(node_path, data, context),
            SuperGraphAnyNode::TokenizerDecode(node) => node.eval(node_path, data, context),
            SuperGraphAnyNode::TokenizerLoad(node) => node.eval(node_path, data, context),
            SuperGraphAnyNode::TextToPhonemes(node) => node.eval(node_path, data, context),
            SuperGraphAnyNode::PiperPhonemesToTensor(node) => node.eval(node_path, data, context),
            SuperGraphAnyNode::KokoroPhonemesToTensor(node) => node.eval(node_path, data, context),
            SuperGraphAnyNode::F5TextToTensor(node) => node.eval(node_path, data, context),
            SuperGraphAnyNode::TensorToImage(node) => node.eval(node_path, data, context),
            SuperGraphAnyNode::TensorToAudioClip(node) => node.eval(node_path, data, context),
            SuperGraphAnyNode::AudioClipToTensor(node) => node.eval(node_path, data, context),
            SuperGraphAnyNode::AudioClipToMelSpectrogram(node) => {
                node.eval(node_path, data, context)
            }
            SuperGraphAnyNode::MilliOpGraph(node) => node.eval(node_path, data, context),
            SuperGraphAnyNode::Scan(node) => node.eval(node_path, data, context),
            SuperGraphAnyNode::ReportProgress(node) => node.eval(node_path, data, context),
            SuperGraphAnyNode::RNNCacheWrite(node) => node.eval(node_path, data, context),
            SuperGraphAnyNode::RNNCacheRead(node) => node.eval(node_path, data, context),
            SuperGraphAnyNode::TensorCacheRead(node) => node.eval(node_path, data, context),
            SuperGraphAnyNode::TensorCacheWrite(node) => node.eval(node_path, data, context),
            SuperGraphAnyNode::TensorPackCacheRead(node) => node.eval(node_path, data, context),
            SuperGraphAnyNode::TensorPackCacheWrite(node) => node.eval(node_path, data, context),
        }
    }

    delegate!(op_kind() -> String);
    delegate!(label() -> Option<String>);
    delegate!(inputs() -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_>);
    delegate!(outputs() -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_>);
    delegate!(global_id() -> GlobalId);
}

impl NodeMetadata for SuperGraphAnyNode {
    fn parameters(&self) -> Vec<Property> {
        match self {
            SuperGraphAnyNode::ModelExecution(node) => {
                vec![Property::new(
                    "symbolic_graph_id",
                    PropertyValue::Int(node.symbolic_graph_id as i64),
                )]
            }
            SuperGraphAnyNode::Scan(node) => {
                vec![Property::new(
                    "num_scan_inputs",
                    PropertyValue::Int(node.scan_inputs.len() as i64),
                )]
            }
            _ => Vec::new(),
        }
    }

    fn has_subgraph(&self) -> bool {
        matches!(self, SuperGraphAnyNode::Scan(_))
    }
}
