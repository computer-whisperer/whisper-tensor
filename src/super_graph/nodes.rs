use crate::DynRank;
use crate::backends::eval_backend;
use crate::backends::eval_backend::EvalBackend;
use crate::backends::ndarray_backend::NDArrayNumericTensor;
use crate::compiler::CompiledProgramObserver;
use crate::graph::{GlobalId, Graph, Node, NodeMetadata, Property, PropertyValue};
use crate::milli_graph::MilliOpGraph;
use crate::milli_graph::observer::MilliOpGraphObserver;
use crate::numeric_tensor::NumericTensor;
use crate::super_graph::links::{
    SuperGraphAnyLink, SuperGraphLink, SuperGraphLinkDouble, SuperGraphLinkHash,
    SuperGraphLinkString, SuperGraphLinkTensor, SuperGraphLinkTensorMap, SuperGraphLinkTokenizer,
    SuperGraphLinkTriple,
};
use crate::super_graph::observer::SuperGraphObserver;
use crate::super_graph::{
    SuperGraph, SuperGraphBuilder, SuperGraphContext, SuperGraphData, SuperGraphError,
};
use crate::symbolic_graph::observer::SymbolicGraphObserver;
use crate::tokenizer::{AnyTokenizer, Tokenizer};
use rand::RngCore;
use rwkv_tokenizer::WorldTokenizer;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ptr;
use std::time::Instant;
use typenum::P1;
use whisper_tensor_import::onnx_graph::TokenizerInfo;

pub trait SuperGraphNode {
    fn to_any(self) -> SuperGraphAnyNode;

    fn op_kind(&self) -> String;
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

    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId> + '_> {
        Box::new(<Self as SuperGraphNode>::inputs(self).map(|x| x.global_id()))
    }

    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId> + '_> {
        Box::new(<Self as SuperGraphNode>::outputs(self).map(|x| x.global_id()))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelReference {
    idx: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeModelExecution {
    global_id: GlobalId,
    tensor_map: SuperGraphLinkTensorMap,
    pub symbolic_graph_id: usize, // Which graph (passed to
    tensor_inputs: Vec<(SuperGraphLinkTensor, String)>,
    tensor_outputs: Vec<(String, SuperGraphLinkTensor)>,
}

impl SuperGraphNodeModelExecution {
    pub fn new(
        rng: &mut impl RngCore,
        tensor_map: SuperGraphLinkTensorMap,
        symbolic_graph_id: usize,
        tensor_inputs: Vec<(SuperGraphLinkTensor, String)>,
        tensor_outputs: Vec<(String, SuperGraphLinkTensor)>,
    ) -> Self {
        Self {
            global_id: GlobalId::new(rng),
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
        self.inner
            .on_node_executed(node_path.as_slice(), start_instant, end_instant, backend)
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
        self.inner
            .on_node_executed(node_path.as_slice(), start_instant, end_instant, backend)
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
    info: TokenizerInfo,
    output: SuperGraphLinkTokenizer,
}

impl SuperGraphNodeTokenizerLoad {
    pub fn new(
        _builder: &mut SuperGraphBuilder,
        info: TokenizerInfo,
        rng: &mut impl RngCore,
    ) -> Self {
        Self {
            global_id: GlobalId::new(rng),
            info,
            output: SuperGraphLinkTokenizer::new(rng),
        }
    }

    pub fn new_and_add(
        builder: &mut SuperGraphBuilder,
        info: TokenizerInfo,
        rng: &mut impl RngCore,
    ) -> SuperGraphLinkTokenizer {
        let node = Self::new(builder, info, rng);
        let output = node.get_tokenizer_output();
        builder.add_node(node.to_any());
        output
    }

    pub fn get_tokenizer_output(&self) -> SuperGraphLinkTokenizer {
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
        let tokenizer = match &self.info {
            #[allow(unused_variables)]
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

    fn op_kind(&self) -> String {
        "Tokenizer Load".to_string()
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
pub struct SuperGraphNodeTokenizerEncode {
    global_id: GlobalId,
    tokenizer: SuperGraphLinkTokenizer,
    text_input: SuperGraphLinkString,
    tensor_output: SuperGraphLinkTensor,
}

impl SuperGraphNodeTokenizerEncode {
    pub fn new(
        _builder: &mut SuperGraphBuilder,
        tokenizer: SuperGraphLinkTokenizer,
        text_input: SuperGraphLinkString,
        rng: &mut impl RngCore,
    ) -> Self {
        Self {
            global_id: GlobalId::new(rng),
            tokenizer,
            text_input,
            tensor_output: SuperGraphLinkTensor::new(rng),
        }
    }

    pub fn new_and_add(
        builder: &mut SuperGraphBuilder,
        tokenizer: SuperGraphLinkTokenizer,
        text_input: SuperGraphLinkString,
        rng: &mut impl RngCore,
    ) -> SuperGraphLinkTensor {
        let node = Self::new(builder, tokenizer, text_input, rng);
        let output = node.get_tensor_output();
        builder.add_node(node.to_any());
        output
    }

    pub fn get_tensor_output(&self) -> SuperGraphLinkTensor {
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

    fn op_kind(&self) -> String {
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
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeTokenizerDecode {
    global_id: GlobalId,
    tokenizer: SuperGraphLinkTokenizer,
    tensor_input: SuperGraphLinkTensor,
    text_output: SuperGraphLinkString,
}

impl SuperGraphNodeTokenizerDecode {
    pub fn new(
        _builder: &mut SuperGraphBuilder,
        tokenizer: SuperGraphLinkTokenizer,
        tensor_input: SuperGraphLinkTensor,
        rng: &mut impl RngCore,
    ) -> Self {
        Self {
            global_id: GlobalId::new(rng),
            tokenizer,
            tensor_input,
            text_output: SuperGraphLinkString::new(rng),
        }
    }
    pub fn get_string_output(&self) -> SuperGraphLinkString {
        self.text_output
    }
    pub fn new_and_add(
        builder: &mut SuperGraphBuilder,
        tokenizer: SuperGraphLinkTokenizer,
        tensor_input: SuperGraphLinkTensor,
        rng: &mut impl RngCore,
    ) -> SuperGraphLinkString {
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
pub struct SuperGraphNodeMilliOpGraph {
    global_id: GlobalId,
    pub graph: MilliOpGraph,
}

impl SuperGraphNodeMilliOpGraph {
    pub fn new(graph: MilliOpGraph, rng: &mut impl RngCore) -> Self {
        Self {
            graph,
            global_id: GlobalId::new(rng),
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
        self.inner
            .on_node_executed(node_path.as_slice(), start_instant, end_instant, backend);
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
                        .get(&SuperGraphLinkTensor(input))
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
            .extend(res.map(|(k, v)| (SuperGraphLinkTensor(k), v)));
        Ok(())
    }
    fn op_kind(&self) -> String {
        "MilliOpGraph".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(
            self.graph
                .input_link_ids()
                .map(|(a, _b)| SuperGraphLinkTensor(a).to_any()),
        )
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(
            self.graph
                .output_link_ids()
                .map(|(a, _b)| SuperGraphLinkTensor(a).to_any()),
        )
    }
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeScan {
    global_id: GlobalId,
    inner_graph: SuperGraph,
    iteration_count: SuperGraphLinkTensor,
    simple_inputs: Vec<SuperGraphLinkDouble>,
    state_links: Vec<SuperGraphLinkTriple>,
    scan_inputs: Vec<(SuperGraphLinkTensor, SuperGraphLinkTensor, u32)>,
    scan_outputs: Vec<(SuperGraphLinkTensor, SuperGraphLinkTensor, u32)>,
    simple_outputs: Vec<SuperGraphLinkDouble>,
}

impl SuperGraphNodeScan {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        inner_graph: SuperGraph,
        iteration_count: SuperGraphLinkTensor,
        simple_inputs: Vec<SuperGraphLinkDouble>,
        state_links: Vec<SuperGraphLinkTriple>,
        scan_inputs: Vec<(SuperGraphLinkTensor, SuperGraphLinkTensor, u32)>,
        scan_outputs: Vec<(SuperGraphLinkTensor, SuperGraphLinkTensor, u32)>,
        simple_outputs: Vec<SuperGraphLinkDouble>,
        rng: &mut impl RngCore,
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

        let node_path = node_path
            .iter()
            .chain(core::iter::once(&self.global_id))
            .copied()
            .collect::<Vec<_>>();

        let simple_inputs = {
            let mut simple_inputs = SuperGraphData::new();
            for link in &self.simple_inputs {
                match link {
                    SuperGraphLinkDouble::Tensor(input, output) => {
                        simple_inputs.tensors.insert(
                            *output,
                            data.tensors
                                .get(input)
                                .ok_or(SuperGraphError::MissingLinkError(format!(
                                    ": scan simple_input tensor {:?}",
                                    input
                                )))?
                                .clone(),
                        );
                    }
                    SuperGraphLinkDouble::String(input, output) => {
                        simple_inputs.strings.insert(
                            *output,
                            data.strings
                                .get(input)
                                .ok_or(SuperGraphError::MissingLinkError(format!(
                                    ": scan simple_input string {:?}",
                                    input
                                )))?
                                .clone(),
                        );
                    }
                    SuperGraphLinkDouble::TensorMap(input, output) => {
                        simple_inputs.tensor_maps.insert(
                            *output,
                            *data.tensor_maps.get(input).ok_or(
                                SuperGraphError::MissingLinkError(format!(
                                    ": scan simple_input tensor_map {:?}",
                                    input
                                )),
                            )?,
                        );
                    }
                    SuperGraphLinkDouble::Tokenizer(input, output) => {
                        simple_inputs.tokenizers.insert(
                            *output,
                            data.tokenizers
                                .get(input)
                                .ok_or(SuperGraphError::MissingLinkError(String::new()))?
                                .clone(),
                        );
                    }
                    SuperGraphLinkDouble::Hash(input, output) => {
                        simple_inputs.hashes.insert(
                            *output,
                            *data
                                .hashes
                                .get(input)
                                .ok_or(SuperGraphError::MissingLinkError(String::new()))?,
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
                                .ok_or(SuperGraphError::MissingLinkError(String::new()))?
                                .clone(),
                        );
                    }
                    SuperGraphLinkTriple::String(initial, inner_input, _inner_output) => {
                        state_values.strings.insert(
                            *inner_input,
                            data.strings
                                .get(initial)
                                .ok_or(SuperGraphError::MissingLinkError(String::new()))?
                                .clone(),
                        );
                    }
                    SuperGraphLinkTriple::TensorMap(initial, inner_input, _inner_output) => {
                        state_values.tensor_maps.insert(
                            *inner_input,
                            *data
                                .tensor_maps
                                .get(initial)
                                .ok_or(SuperGraphError::MissingLinkError(String::new()))?,
                        );
                    }
                    SuperGraphLinkTriple::Tokenizer(initial, inner_input, _inner_output) => {
                        state_values.tokenizers.insert(
                            *inner_input,
                            data.tokenizers
                                .get(initial)
                                .ok_or(SuperGraphError::MissingLinkError(String::new()))?
                                .clone(),
                        );
                    }
                    SuperGraphLinkTriple::Hash(initial, inner_input, _inner_output) => {
                        state_values.hashes.insert(
                            *inner_input,
                            *data
                                .hashes
                                .get(initial)
                                .ok_or(SuperGraphError::MissingLinkError(String::new()))?,
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
                    match link {
                        SuperGraphLinkTriple::Tensor(_initial, inner_input, inner_output) => {
                            state_values.tensors.insert(
                                *inner_input,
                                iter_outputs
                                    .tensors
                                    .get(inner_output)
                                    .ok_or(SuperGraphError::MissingLinkError(String::new()))?
                                    .clone(),
                            );
                        }
                        SuperGraphLinkTriple::String(_initial, inner_input, inner_output) => {
                            state_values.strings.insert(
                                *inner_input,
                                iter_outputs
                                    .strings
                                    .get(inner_output)
                                    .ok_or(SuperGraphError::MissingLinkError(String::new()))?
                                    .clone(),
                            );
                        }
                        SuperGraphLinkTriple::Tokenizer(_initial, inner_input, inner_output) => {
                            state_values.tokenizers.insert(
                                *inner_input,
                                iter_outputs
                                    .tokenizers
                                    .get(inner_output)
                                    .ok_or(SuperGraphError::MissingLinkError(String::new()))?
                                    .clone(),
                            );
                        }
                        SuperGraphLinkTriple::TensorMap(_initial, inner_input, inner_output) => {
                            state_values.tensor_maps.insert(
                                *inner_input,
                                *iter_outputs
                                    .tensor_maps
                                    .get(inner_output)
                                    .ok_or(SuperGraphError::MissingLinkError(String::new()))?,
                            );
                        }
                        SuperGraphLinkTriple::Hash(_initial, inner_input, inner_output) => {
                            state_values.hashes.insert(
                                *inner_input,
                                *iter_outputs
                                    .hashes
                                    .get(inner_output)
                                    .ok_or(SuperGraphError::MissingLinkError(String::new()))?,
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
                            .ok_or(SuperGraphError::MissingLinkError(String::new()))?
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
                            .ok_or(SuperGraphError::MissingLinkError(String::new()))?
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
                            .ok_or(SuperGraphError::MissingLinkError(String::new()))?
                            .clone(),
                    );
                }
                SuperGraphLinkDouble::TensorMap(input, output) => {
                    output_data.tensor_maps.insert(
                        *output,
                        *prev_iter_outputs
                            .as_ref()
                            .unwrap()
                            .tensor_maps
                            .get(input)
                            .ok_or(SuperGraphError::MissingLinkError(String::new()))?,
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
                            .ok_or(SuperGraphError::MissingLinkError(String::new()))?,
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
    fn op_kind(&self) -> String {
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
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeRNNCacheRead {
    global_id: GlobalId,
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
        rng: &mut impl RngCore,
    ) -> Self {
        Self {
            global_id: GlobalId::new(rng),
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
    key_input: SuperGraphLinkHash,
    tokens_input: SuperGraphLinkTensor,
    state_inputs: Vec<(String, SuperGraphLinkTensor)>,
}

impl SuperGraphNodeRNNCacheWrite {
    pub fn new(
        key_input: SuperGraphLinkHash,
        tokens_input: SuperGraphLinkTensor,
        state_inputs: Vec<(String, SuperGraphLinkTensor)>,
        rng: &mut impl RngCore,
    ) -> Self {
        Self {
            global_id: GlobalId::new(rng),
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
#[allow(clippy::large_enum_variant)]
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
                SuperGraphAnyNode::ModelExecution(x) => SuperGraphNode::$name(x,$($arg),*),
                SuperGraphAnyNode::TokenizerEncode(x) => SuperGraphNode::$name(x,$($arg),*),
                SuperGraphAnyNode::TokenizerDecode(x) => SuperGraphNode::$name(x,$($arg),*),
                SuperGraphAnyNode::TokenizerLoad(x) => SuperGraphNode::$name(x,$($arg),*),
                SuperGraphAnyNode::MilliOpGraph(x) => SuperGraphNode::$name(x,$($arg),*),
                SuperGraphAnyNode::Scan(x) => SuperGraphNode::$name(x,$($arg),*),
                SuperGraphAnyNode::RNNCacheRead(x) => SuperGraphNode::$name(x,$($arg),*),
                SuperGraphAnyNode::RNNCacheWrite(x) => SuperGraphNode::$name(x,$($arg),*),
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
            SuperGraphAnyNode::MilliOpGraph(node) => node.eval(node_path, data, context),
            SuperGraphAnyNode::Scan(node) => node.eval(node_path, data, context),
            SuperGraphAnyNode::RNNCacheWrite(node) => node.eval(node_path, data, context),
            SuperGraphAnyNode::RNNCacheRead(node) => node.eval(node_path, data, context),
        }
    }

    delegate!(op_kind() -> String);
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
