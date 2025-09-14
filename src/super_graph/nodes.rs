use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::backends::ndarray_backend::NDArrayNumericTensor;
use crate::compiler::CompiledProgramObserver;
use crate::graph::{InnerGraph, Node};
use crate::milli_graph::observer::MilliOpGraphObserver;
use crate::milli_graph::{MilliOpGraph, MilliOpGraphNodePath, MilliOpGraphTensorPath};
use crate::numeric_tensor::NumericTensor;
use crate::super_graph::links::{
    SuperGraphAnyLink, SuperGraphLink, SuperGraphLinkDouble, SuperGraphLinkHash,
    SuperGraphLinkModel, SuperGraphLinkString, SuperGraphLinkTensor, SuperGraphLinkTokenizer,
    SuperGraphLinkTriple,
};
use crate::super_graph::observer::SuperGraphObserver;
use crate::super_graph::{
    SuperGraphBuilder, SuperGraphContext, SuperGraphData, SuperGraphError, SuperGraphInner,
    SuperGraphNodeId, SuperGraphNodePath, SuperGraphTensorPath,
};
use crate::symbolic_graph::observer::SymbolicGraphObserver;
use crate::symbolic_graph::{SymbolicGraphNodePath, SymbolicGraphTensorPath};
use crate::tokenizer::{AnyTokenizer, Tokenizer};
use rwkv_tokenizer::WorldTokenizer;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ptr;
use std::time::Instant;
use typenum::P1;
use whisper_tensor_import::onnx_graph::TokenizerInfo;

pub trait SuperGraphNode {
    fn to_any(self) -> SuperGraphAnyNode;
    fn eval<'a, 'b, 'c, 'd, T: SuperGraphObserver>(
        &'a self,
        this_path: &[SuperGraphNodeId],
        data: &mut SuperGraphData<'b>,
        context: &mut SuperGraphContext<'a, 'b, 'c, 'd, T>,
    ) -> Result<(), SuperGraphError>;
}

impl<T: SuperGraphNode> From<T> for SuperGraphAnyNode {
    fn from(value: T) -> Self {
        value.to_any()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelReference {
    idx: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeModelExecution {
    model: SuperGraphLinkModel,
    tensor_inputs: Vec<(SuperGraphLinkTensor, String)>,
    tensor_outputs: Vec<(String, SuperGraphLinkTensor)>,
}

impl SuperGraphNodeModelExecution {
    pub fn new(
        model: SuperGraphLinkModel,
        tensor_inputs: Vec<(SuperGraphLinkTensor, String)>,
        tensor_outputs: Vec<(String, SuperGraphLinkTensor)>,
    ) -> Self {
        Self {
            model,
            tensor_inputs,
            tensor_outputs,
        }
    }
}

impl Node<SuperGraphAnyLink> for SuperGraphNodeModelExecution {
    type OpKind = String;
    fn op_kind(&self) -> Self::OpKind {
        "Model Execution".to_string()
    }

    fn inputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        let ret = self.tensor_inputs.clone().into_iter().map(|x| x.0.to_any());
        Box::new(ret.chain(std::iter::once(self.model.to_any())))
    }

    fn outputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(
            self.tensor_outputs
                .clone()
                .into_iter()
                .map(|x| x.1.to_any()),
        )
    }
}

struct SymbolicGraphObserverWrapper<'a, T: SuperGraphObserver> {
    inner: &'a mut T,
    node_path: Vec<SuperGraphNodeId>,
}

impl<'a, T: SuperGraphObserver> SymbolicGraphObserverWrapper<'a, T> {
    fn new(inner: &'a mut T, node_path: &[SuperGraphNodeId]) -> Self {
        Self {
            inner,
            node_path: node_path.to_vec(),
        }
    }
}

impl<'a, T: SuperGraphObserver> SymbolicGraphObserver for SymbolicGraphObserverWrapper<'a, T> {
    fn on_op_executed(
        &mut self,
        node_path: &SymbolicGraphNodePath,
        start_instant: Instant,
        end_instant: Instant,
        backend: &mut EvalBackend,
    ) {
        self.inner.on_node_executed(
            &SuperGraphNodePath::SymbolicGraphNode(self.node_path.clone(), node_path.clone()),
            start_instant,
            end_instant,
            backend,
        )
    }

    fn on_tensor_assigned(
        &mut self,
        tensor_path: &SymbolicGraphTensorPath,
        tensor: &NumericTensor<DynRank>,
        backend: &mut EvalBackend,
    ) {
        self.inner.on_tensor_assigned(
            &SuperGraphTensorPath::SymbolicGraphTensor(self.node_path.clone(), tensor_path.clone()),
            tensor,
            backend,
        )
    }
}

impl<'a, T: SuperGraphObserver> CompiledProgramObserver for SymbolicGraphObserverWrapper<'a, T> {
    fn on_op_executed(
        &mut self,
        node_path: &SymbolicGraphNodePath,
        start_instant: Instant,
        end_instant: Instant,
        backend: &mut EvalBackend,
    ) {
        self.inner.on_node_executed(
            &SuperGraphNodePath::SymbolicGraphNode(self.node_path.clone(), node_path.clone()),
            start_instant,
            end_instant,
            backend,
        )
    }

    fn on_tensor_assigned(
        &mut self,
        tensor_path: &SymbolicGraphTensorPath,
        tensor: &NumericTensor<DynRank>,
        backend: &mut EvalBackend,
    ) {
        self.inner.on_tensor_assigned(
            &SuperGraphTensorPath::SymbolicGraphTensor(self.node_path.clone(), tensor_path.clone()),
            tensor,
            backend,
        )
    }
}

impl SuperGraphNode for SuperGraphNodeModelExecution {
    fn to_any(self) -> SuperGraphAnyNode {
        SuperGraphAnyNode::ModelExecution(self)
    }
    fn eval<T: SuperGraphObserver>(
        &self,
        node_path: &[SuperGraphNodeId],
        data: &mut SuperGraphData,
        context: &mut SuperGraphContext<T>,
    ) -> Result<(), SuperGraphError> {
        let model = data.models.get(&self.model).unwrap();

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
                if ptr::eq(*a, *model) {
                    res = Some(b)
                }
            }
            res
        };

        let mut observer = SymbolicGraphObserverWrapper::new(context.observer, node_path);
        if context.use_compiled_models
            && let Some(compiled_models) = &context.compiled_models
        {
            let compiled_model = &compiled_models
                .iter()
                .find(|(x, _y)| core::ptr::addr_eq(*x, *model))
                .ok_or(SuperGraphError::ModelNotCompiledError)?
                .1;
            let outputs =
                compiled_model.run(context.eval_backend, tensor_cache, inputs, &mut observer)?;
            let outputs = outputs.collect::<HashMap<_, _>>();
            for (name, link) in &self.tensor_outputs {
                data.tensors
                    .insert(*link, outputs.get(name).unwrap().clone());
            }
        } else {
            let outputs = model.eval(inputs, &mut observer, tensor_cache, context.eval_backend)?;
            for (name, link) in &self.tensor_outputs {
                data.tensors
                    .insert(*link, outputs.get(name).unwrap().clone());
            }
        };

        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeTokenizerLoad {
    info: TokenizerInfo,
    output: SuperGraphLinkTokenizer,
}

impl SuperGraphNodeTokenizerLoad {
    pub fn new(builder: &mut SuperGraphBuilder, info: TokenizerInfo) -> Self {
        Self {
            info,
            output: SuperGraphLinkTokenizer::new(builder.get_next_link_id()),
        }
    }

    pub fn new_and_add(
        builder: &mut SuperGraphBuilder,
        info: TokenizerInfo,
    ) -> SuperGraphLinkTokenizer {
        let node = Self::new(builder, info);
        let output = node.get_tokenizer_output();
        builder.add_node(node.into());
        output
    }

    pub fn get_tokenizer_output(&self) -> SuperGraphLinkTokenizer {
        self.output
    }
}

impl Node<SuperGraphAnyLink> for SuperGraphNodeTokenizerLoad {
    type OpKind = String;
    fn op_kind(&self) -> Self::OpKind {
        "Tokenizer Load".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(std::iter::empty())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(std::iter::once(self.output.to_any()))
    }
}
impl SuperGraphNode for SuperGraphNodeTokenizerLoad {
    fn to_any(self) -> SuperGraphAnyNode {
        SuperGraphAnyNode::TokenizerLoad(self)
    }
    fn eval<T: SuperGraphObserver>(
        &self,
        _this_path: &[SuperGraphNodeId],
        data: &mut SuperGraphData,
        _context: &mut SuperGraphContext<T>,
    ) -> Result<(), SuperGraphError> {
        let tokenizer = match &self.info {
            TokenizerInfo::HFTokenizer(name) => {
                #[cfg(all(feature = "tokenizers", feature = "http"))]
                {
                    AnyTokenizer::Tokenizers(
                        tokenizers::Tokenizer::from_pretrained(name, None).unwrap(),
                    )
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
        data.tokenizers.insert(self.output, tokenizer);
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeTokenizerEncode {
    tokenizer: SuperGraphLinkTokenizer,
    text_input: SuperGraphLinkString,
    tensor_output: SuperGraphLinkTensor,
}

impl SuperGraphNodeTokenizerEncode {
    pub fn new(
        builder: &mut SuperGraphBuilder,
        tokenizer: SuperGraphLinkTokenizer,
        text_input: SuperGraphLinkString,
    ) -> Self {
        Self {
            tokenizer,
            text_input,
            tensor_output: SuperGraphLinkTensor::new(builder.get_next_link_id()),
        }
    }

    pub fn new_and_add(
        builder: &mut SuperGraphBuilder,
        tokenizer: SuperGraphLinkTokenizer,
        text_input: SuperGraphLinkString,
    ) -> SuperGraphLinkTensor {
        let node = Self::new(builder, tokenizer, text_input);
        let output = node.get_tensor_output();
        builder.add_node(node.into());
        output
    }

    pub fn get_tensor_output(&self) -> SuperGraphLinkTensor {
        self.tensor_output
    }
}

impl Node<SuperGraphAnyLink> for SuperGraphNodeTokenizerEncode {
    type OpKind = String;
    fn op_kind(&self) -> Self::OpKind {
        "Tokenizer Encode".to_string()
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
}
impl SuperGraphNode for SuperGraphNodeTokenizerEncode {
    fn to_any(self) -> SuperGraphAnyNode {
        SuperGraphAnyNode::TokenizerEncode(self)
    }
    fn eval<T: SuperGraphObserver>(
        &self,
        _this_path: &[SuperGraphNodeId],
        data: &mut SuperGraphData,
        _context: &mut SuperGraphContext<T>,
    ) -> Result<(), SuperGraphError> {
        let text = data.strings.get(&self.text_input).unwrap();
        let tokenizer = data.tokenizers.get(&self.tokenizer).unwrap();
        let tokens = tokenizer
            .encode(text)
            .iter()
            .map(|x| *x as i64)
            .collect::<Vec<_>>();
        let input_tensor = NumericTensor::from_vec(tokens)
            .to_dyn_rank()
            .unsqueeze(0)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        data.tensors.insert(self.tensor_output, input_tensor);
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeTokenizerDecode {
    tokenizer: SuperGraphLinkTokenizer,
    tensor_input: SuperGraphLinkTensor,
    text_output: SuperGraphLinkString,
}

impl SuperGraphNodeTokenizerDecode {
    pub fn new(
        builder: &mut SuperGraphBuilder,
        tokenizer: SuperGraphLinkTokenizer,
        tensor_input: SuperGraphLinkTensor,
    ) -> Self {
        Self {
            tokenizer,
            tensor_input,
            text_output: SuperGraphLinkString::new(builder.get_next_link_id()),
        }
    }
    pub fn get_string_output(&self) -> SuperGraphLinkString {
        self.text_output
    }
    pub fn new_and_add(
        builder: &mut SuperGraphBuilder,
        tokenizer: SuperGraphLinkTokenizer,
        tensor_input: SuperGraphLinkTensor,
    ) -> SuperGraphLinkString {
        let node = Self::new(builder, tokenizer, tensor_input);
        let output = node.get_string_output();
        builder.add_node(node.to_any());
        output
    }
}

impl Node<SuperGraphAnyLink> for SuperGraphNodeTokenizerDecode {
    type OpKind = String;
    fn op_kind(&self) -> Self::OpKind {
        "Tokenizer Decode".to_string()
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
}

impl SuperGraphNode for SuperGraphNodeTokenizerDecode {
    fn to_any(self) -> SuperGraphAnyNode {
        SuperGraphAnyNode::TokenizerDecode(self)
    }
    fn eval<T: SuperGraphObserver>(
        &self,
        _node_path: &[SuperGraphNodeId],
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
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeMilliOpGraph {
    pub graph: MilliOpGraph<SuperGraphLinkTensor>,
}

impl SuperGraphNodeMilliOpGraph {
    pub fn new(graph: MilliOpGraph<SuperGraphLinkTensor>) -> Self {
        Self { graph }
    }
}

struct MilliOpGraphObserverWrapper<'a, T: SuperGraphObserver> {
    inner: &'a mut T,
    node_path: Vec<SuperGraphNodeId>,
}

impl<'a, T: SuperGraphObserver> MilliOpGraphObserverWrapper<'a, T> {
    fn new(inner: &'a mut T, node_path: &[SuperGraphNodeId]) -> Self {
        Self {
            inner,
            node_path: node_path.to_vec(),
        }
    }
}

impl<'a, T: SuperGraphObserver> MilliOpGraphObserver for MilliOpGraphObserverWrapper<'a, T> {
    fn on_tensor_assigned(
        &mut self,
        tensor_path: &MilliOpGraphTensorPath,
        tensor: &NumericTensor<DynRank>,
        backend: &mut EvalBackend,
    ) {
        self.inner.on_tensor_assigned(
            &SuperGraphTensorPath::MilliOpGraphTensor(self.node_path.clone(), tensor_path.clone()),
            tensor,
            backend,
        );
    }

    fn on_node_executed(
        &mut self,
        node_path: &MilliOpGraphNodePath,
        start_instant: Instant,
        end_instant: Instant,
        backend: &mut EvalBackend,
    ) {
        self.inner.on_node_executed(
            &SuperGraphNodePath::MilliOpGraphNode(self.node_path.clone(), node_path.clone()),
            start_instant,
            end_instant,
            backend,
        );
    }
}

impl Node<SuperGraphAnyLink> for SuperGraphNodeMilliOpGraph {
    type OpKind = String;
    fn op_kind(&self) -> Self::OpKind {
        "MilliOpGraph".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(self.graph.input_links().map(|(a, _b)| a.to_any()))
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(self.graph.output_links().map(|(a, _b)| a.to_any()))
    }
}

impl SuperGraphNode for SuperGraphNodeMilliOpGraph {
    fn to_any(self) -> SuperGraphAnyNode {
        SuperGraphAnyNode::MilliOpGraph(self)
    }
    fn eval<T: SuperGraphObserver>(
        &self,
        node_path: &[SuperGraphNodeId],
        data: &mut SuperGraphData,
        context: &mut SuperGraphContext<T>,
    ) -> Result<(), SuperGraphError> {
        let inputs = {
            let mut inputs = HashMap::new();
            for input in &self.graph.get_inputs() {
                inputs.insert(*input, data.tensors.get(input).unwrap().clone());
            }
            inputs
        };
        let mut observer = MilliOpGraphObserverWrapper::new(context.observer, node_path);
        let res = self
            .graph
            .eval(&inputs, &mut observer, context.eval_backend)?;
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

impl Node<SuperGraphAnyLink> for SuperGraphNodeScan {
    type OpKind = String;
    fn op_kind(&self) -> Self::OpKind {
        "Scan".to_string()
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
}
impl SuperGraphNode for SuperGraphNodeScan {
    fn to_any(self) -> SuperGraphAnyNode {
        SuperGraphAnyNode::Scan(self)
    }

    fn eval<'a, 'b, 'c, 'd, T: SuperGraphObserver>(
        &'a self,
        node_path: &[SuperGraphNodeId],
        data: &mut SuperGraphData<'b>,
        context: &mut SuperGraphContext<'a, 'b, 'c, 'd, T>,
    ) -> Result<(), SuperGraphError> {
        let iteration_count_tensor = data
            .tensors
            .get(&self.iteration_count)
            .ok_or(SuperGraphError::MissingLinkError())?;
        let iteration_count: i64 = iteration_count_tensor.first_element().into();

        let simple_inputs = {
            let mut simple_inputs = SuperGraphData::new();
            for link in &self.simple_inputs {
                match link {
                    SuperGraphLinkDouble::Tensor(input, output) => {
                        simple_inputs.tensors.insert(
                            *output,
                            data.tensors
                                .get(input)
                                .ok_or(SuperGraphError::MissingLinkError())?
                                .clone(),
                        );
                    }
                    SuperGraphLinkDouble::String(input, output) => {
                        simple_inputs.strings.insert(
                            *output,
                            data.strings
                                .get(input)
                                .ok_or(SuperGraphError::MissingLinkError())?
                                .clone(),
                        );
                    }
                    SuperGraphLinkDouble::Model(input, output) => {
                        simple_inputs.models.insert(
                            *output,
                            *data
                                .models
                                .get(input)
                                .ok_or(SuperGraphError::MissingLinkError())?,
                        );
                    }
                    SuperGraphLinkDouble::Tokenizer(input, output) => {
                        simple_inputs.tokenizers.insert(
                            *output,
                            data.tokenizers
                                .get(input)
                                .ok_or(SuperGraphError::MissingLinkError())?
                                .clone(),
                        );
                    }
                    SuperGraphLinkDouble::Hash(input, output) => {
                        simple_inputs.hashes.insert(
                            *output,
                            *data
                                .hashes
                                .get(input)
                                .ok_or(SuperGraphError::MissingLinkError())?,
                        );
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
                        state_values.tensors.insert(
                            *inner_input,
                            data.tensors
                                .get(initial)
                                .ok_or(SuperGraphError::MissingLinkError())?
                                .clone(),
                        );
                    }
                    SuperGraphLinkTriple::String(initial, inner_input, _inner_output) => {
                        state_values.strings.insert(
                            *inner_input,
                            data.strings
                                .get(initial)
                                .ok_or(SuperGraphError::MissingLinkError())?
                                .clone(),
                        );
                    }
                    SuperGraphLinkTriple::Model(initial, inner_input, _inner_output) => {
                        state_values.models.insert(
                            *inner_input,
                            *data
                                .models
                                .get(initial)
                                .ok_or(SuperGraphError::MissingLinkError())?,
                        );
                    }
                    SuperGraphLinkTriple::Tokenizer(initial, inner_input, _inner_output) => {
                        state_values.tokenizers.insert(
                            *inner_input,
                            data.tokenizers
                                .get(initial)
                                .ok_or(SuperGraphError::MissingLinkError())?
                                .clone(),
                        );
                    }
                    SuperGraphLinkTriple::Hash(initial, inner_input, _inner_output) => {
                        state_values.hashes.insert(
                            *inner_input,
                            *data
                                .hashes
                                .get(initial)
                                .ok_or(SuperGraphError::MissingLinkError())?,
                        );
                    }
                }
            }
            state_values
        };

        let mut prev_iter_outputs: Option<SuperGraphData> = None;

        let mut output_scan_tensor_parts = HashMap::new();
        for (_inner, outer, scan_axis) in &self.scan_outputs {
            output_scan_tensor_parts.insert(*outer, (Vec::new(), *scan_axis as usize));
        }

        for i in 0..iteration_count as u64 {
            let iter_inputs = {
                let mut iter_inputs = simple_inputs.clone();
                iter_inputs.extend(&state_values);
                for (outer, inner, scan_axis) in &self.scan_inputs {
                    let tensor = data
                        .tensors
                        .get(outer)
                        .ok_or(SuperGraphError::MissingLinkError())?;
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
            let iter_outputs = self.inner_graph.eval(node_path, iter_inputs, context)?;

            for (inner, outer, _scan_axis) in &self.scan_outputs {
                let tensor = iter_outputs
                    .tensors
                    .get(inner)
                    .ok_or(SuperGraphError::MissingLinkError())?;
                let (tensors, _) = output_scan_tensor_parts
                    .get_mut(outer)
                    .ok_or(SuperGraphError::MissingLinkError())?;
                tensors.push(tensor.clone());
            }

            state_values = {
                let mut state_values = SuperGraphData::new();
                for link in &self.state_links {
                    match link {
                        SuperGraphLinkTriple::Tensor(_initial, inner_input, inner_output) => {
                            state_values.tensors.insert(
                                *inner_input,
                                iter_outputs
                                    .tensors
                                    .get(inner_output)
                                    .ok_or(SuperGraphError::MissingLinkError())?
                                    .clone(),
                            );
                        }
                        SuperGraphLinkTriple::String(_initial, inner_input, inner_output) => {
                            state_values.strings.insert(
                                *inner_input,
                                iter_outputs
                                    .strings
                                    .get(inner_output)
                                    .ok_or(SuperGraphError::MissingLinkError())?
                                    .clone(),
                            );
                        }
                        SuperGraphLinkTriple::Tokenizer(_initial, inner_input, inner_output) => {
                            state_values.tokenizers.insert(
                                *inner_input,
                                iter_outputs
                                    .tokenizers
                                    .get(inner_output)
                                    .ok_or(SuperGraphError::MissingLinkError())?
                                    .clone(),
                            );
                        }
                        SuperGraphLinkTriple::Model(_initial, inner_input, inner_output) => {
                            state_values.models.insert(
                                *inner_input,
                                *iter_outputs
                                    .models
                                    .get(inner_output)
                                    .ok_or(SuperGraphError::MissingLinkError())?,
                            );
                        }
                        SuperGraphLinkTriple::Hash(_initial, inner_input, inner_output) => {
                            state_values.hashes.insert(
                                *inner_input,
                                *iter_outputs
                                    .hashes
                                    .get(inner_output)
                                    .ok_or(SuperGraphError::MissingLinkError())?,
                            );
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
                    output_data.tensors.insert(
                        *output,
                        prev_iter_outputs
                            .as_ref()
                            .unwrap()
                            .tensors
                            .get(input)
                            .ok_or(SuperGraphError::MissingLinkError())?
                            .clone(),
                    );
                }
                SuperGraphLinkDouble::String(input, output) => {
                    output_data.strings.insert(
                        *output,
                        prev_iter_outputs
                            .as_ref()
                            .unwrap()
                            .strings
                            .get(input)
                            .ok_or(SuperGraphError::MissingLinkError())?
                            .clone(),
                    );
                }
                SuperGraphLinkDouble::Tokenizer(input, output) => {
                    output_data.tokenizers.insert(
                        *output,
                        prev_iter_outputs
                            .as_ref()
                            .unwrap()
                            .tokenizers
                            .get(input)
                            .ok_or(SuperGraphError::MissingLinkError())?
                            .clone(),
                    );
                }
                SuperGraphLinkDouble::Model(input, output) => {
                    output_data.models.insert(
                        *output,
                        *prev_iter_outputs
                            .as_ref()
                            .unwrap()
                            .models
                            .get(input)
                            .ok_or(SuperGraphError::MissingLinkError())?,
                    );
                }
                SuperGraphLinkDouble::Hash(input, output) => {
                    output_data.hashes.insert(
                        *output,
                        *prev_iter_outputs
                            .as_ref()
                            .unwrap()
                            .hashes
                            .get(input)
                            .ok_or(SuperGraphError::MissingLinkError())?,
                    );
                }
            }
        }

        for (link, (parts, axis)) in output_scan_tensor_parts {
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
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeRNNCacheRead {
    key_input: SuperGraphLinkHash,
    tokens_input: SuperGraphLinkTensor,
    tokens_output: SuperGraphLinkTensor,
    state_outputs: Vec<(String, SuperGraphLinkTensor)>,
    default_state_inputs: Vec<(String, SuperGraphLinkTensor)>,
}

impl SuperGraphNodeRNNCacheRead {
    pub fn new(
        key_input: SuperGraphLinkHash,
        tokens_input: SuperGraphLinkTensor,
        tokens_output: SuperGraphLinkTensor,
        state_outputs: Vec<(String, SuperGraphLinkTensor)>,
        default_state_inputs: Vec<(String, SuperGraphLinkTensor)>,
    ) -> Self {
        Self {
            key_input,
            tokens_input,
            tokens_output,
            state_outputs,
            default_state_inputs,
        }
    }
}

impl Node<SuperGraphAnyLink> for SuperGraphNodeRNNCacheRead {
    type OpKind = String;
    fn op_kind(&self) -> Self::OpKind {
        "RNNCacheRead".to_string()
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
}

impl SuperGraphNode for SuperGraphNodeRNNCacheRead {
    fn to_any(self) -> SuperGraphAnyNode {
        SuperGraphAnyNode::RNNCacheRead(self)
    }

    fn eval<T: SuperGraphObserver>(
        &self,
        _node_path: &[SuperGraphNodeId],
        data: &mut SuperGraphData,
        context: &mut SuperGraphContext<T>,
    ) -> Result<(), SuperGraphError> {
        let tokens_input = data
            .tensors
            .get(&self.tokens_input)
            .ok_or(SuperGraphError::MissingLinkError())?
            .clone();
        let mut found = false;
        if let Some(caches) = &mut context.caches {
            let key_input = *data
                .hashes
                .get(&self.key_input)
                .ok_or(SuperGraphError::MissingLinkError())?;
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
                    .ok_or(SuperGraphError::MissingLinkError())?;
                if let Some((_, output_link)) = self.state_outputs.iter().find(|x| x.0 == *key) {
                    data.tensors.insert(*output_link, value.clone());
                }
            }
            data.tensors
                .insert(self.tokens_output, tokens_input.clone());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeRNNCacheWrite {
    key_input: SuperGraphLinkHash,
    tokens_input: SuperGraphLinkTensor,
    state_inputs: Vec<(String, SuperGraphLinkTensor)>,
}

impl SuperGraphNodeRNNCacheWrite {
    pub fn new(
        key_input: SuperGraphLinkHash,
        tokens_input: SuperGraphLinkTensor,
        state_inputs: Vec<(String, SuperGraphLinkTensor)>,
    ) -> Self {
        Self {
            key_input,
            tokens_input,
            state_inputs,
        }
    }
}

impl Node<SuperGraphAnyLink> for SuperGraphNodeRNNCacheWrite {
    type OpKind = String;
    fn op_kind(&self) -> Self::OpKind {
        "RNNCacheWrite".to_string()
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
}
impl SuperGraphNode for SuperGraphNodeRNNCacheWrite {
    fn to_any(self) -> SuperGraphAnyNode {
        SuperGraphAnyNode::RNNCacheWrite(self)
    }

    fn eval<T: SuperGraphObserver>(
        &self,
        _node_path: &[SuperGraphNodeId],
        data: &mut SuperGraphData,
        context: &mut SuperGraphContext<T>,
    ) -> Result<(), SuperGraphError> {
        if let Some(caches) = &mut context.caches {
            let key_input = *data
                .hashes
                .get(&self.key_input)
                .ok_or(SuperGraphError::MissingLinkError())?;
            let tokens_input = data
                .tensors
                .get(&self.tokens_input)
                .ok_or(SuperGraphError::MissingLinkError())?;
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
    RNNCacheRead(SuperGraphNodeRNNCacheRead),
}

macro_rules! delegate {
    ($name:ident($($arg:ident: $ty:ty),*) -> $ret:ty) => {
        fn $name(&self, $($arg: $ty),*) -> $ret {
            match self {
                SuperGraphAnyNode::ModelExecution(x) => x.$name($($arg),*),
                SuperGraphAnyNode::TokenizerEncode(x) => x.$name($($arg),*),
                SuperGraphAnyNode::TokenizerDecode(x) => x.$name($($arg),*),
                SuperGraphAnyNode::TokenizerLoad(x) => x.$name($($arg),*),
                SuperGraphAnyNode::MilliOpGraph(x) => x.$name($($arg),*),
                SuperGraphAnyNode::Scan(x) => x.$name($($arg),*),
                SuperGraphAnyNode::RNNCacheRead(x) => x.$name($($arg),*),
                SuperGraphAnyNode::RNNCacheWrite(x) => x.$name($($arg),*),
            }
        }
    }
}

impl SuperGraphAnyNode {
    pub(crate) fn eval<'a, 'b, 'c, 'd, T: SuperGraphObserver>(
        &'a self,
        node_path: &[SuperGraphNodeId],
        data: &mut SuperGraphData<'b>,
        context: &mut SuperGraphContext<'a, 'b, 'c, 'd, T>,
    ) -> Result<(), SuperGraphError> {
        match self {
            SuperGraphAnyNode::ModelExecution(node) => node.eval(node_path, data, context),
            SuperGraphAnyNode::TokenizerEncode(node) => node.eval(node_path, data, context),
            SuperGraphAnyNode::TokenizerDecode(node) => node.eval(node_path, data, context),
            SuperGraphAnyNode::TokenizerLoad(node) => node.eval(node_path, data, context),
            SuperGraphAnyNode::MilliOpGraph(node) => node.eval(node_path, data, context),
            SuperGraphAnyNode::Scan(node) => node.eval(node_path, data, context),
            SuperGraphAnyNode::RNNCacheWrite(node) => node.eval(node_path, data, context),
            SuperGraphAnyNode::RNNCacheRead(node) => node.eval(node_path, data, context),
        }
    }

    pub fn get_sub_graph(&self) -> Option<&SuperGraphInner> {
        match self {
            SuperGraphAnyNode::Scan(x) => Some(&x.inner_graph),
            _ => None,
        }
    }
}

impl Node<SuperGraphAnyLink> for SuperGraphAnyNode {
    type OpKind = String;
    delegate!(op_kind() -> Self::OpKind);
    delegate!(inputs() -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_>);
    delegate!(outputs() -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_>);
}
