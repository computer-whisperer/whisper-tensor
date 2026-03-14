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
use crate::numeric_tensor::NumericTensor;
use crate::super_graph::data::SuperGraphImage;
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
use rand::RngCore;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ptr;
use std::time::Instant;
use typenum::P1;

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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelReference {
    idx: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeModelExecution {
    global_id: GlobalId,
    tensor_map: SuperGraphLink,
    pub symbolic_graph_id: usize, // Which graph (passed to
    tensor_inputs: Vec<(SuperGraphLink, String)>,
    tensor_outputs: Vec<(String, SuperGraphLink)>,
}

impl SuperGraphNodeModelExecution {
    pub fn new(
        rng: &mut impl RngCore,
        tensor_map: SuperGraphLink,
        symbolic_graph_id: usize,
        tensor_inputs: Vec<(SuperGraphLink, String)>,
        tensor_outputs: Vec<(String, SuperGraphLink)>,
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
    output: SuperGraphLink,
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
            output: SuperGraphLink::new(SuperGraphLinkKind::Tokenizer, rng),
        }
    }

    pub fn new_and_add(
        builder: &mut SuperGraphBuilder,
        info: TokenizerInfo,
        rng: &mut impl RngCore,
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
        rng: &mut impl RngCore,
    ) -> Self {
        Self {
            global_id: GlobalId::new(rng),
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
        rng: &mut impl RngCore,
    ) -> Self {
        Self {
            global_id: GlobalId::new(rng),
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
        rng: &mut impl RngCore,
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
        rng: &mut impl RngCore,
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
    tokenizer: SuperGraphLink,
    tensor_input: SuperGraphLink,
    text_output: SuperGraphLink,
}

impl SuperGraphNodeTokenizerDecode {
    pub fn new(
        _builder: &mut SuperGraphBuilder,
        tokenizer: SuperGraphLink,
        tensor_input: SuperGraphLink,
        rng: &mut impl RngCore,
    ) -> Self {
        Self {
            global_id: GlobalId::new(rng),
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
        rng: &mut impl RngCore,
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
pub struct SuperGraphNodeTensorToImage {
    global_id: GlobalId,
    tensor_input: SuperGraphLink,
    image_output: SuperGraphLink,
}

impl SuperGraphNodeTensorToImage {
    pub fn new(
        _builder: &mut SuperGraphBuilder,
        tensor_input: SuperGraphLink,
        rng: &mut impl RngCore,
    ) -> Self {
        Self {
            global_id: GlobalId::new(rng),
            tensor_input,
            image_output: SuperGraphLink::new(SuperGraphLinkKind::Image, rng),
        }
    }

    pub fn new_and_add(
        builder: &mut SuperGraphBuilder,
        tensor_input: SuperGraphLink,
        rng: &mut impl RngCore,
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
        self.inner.on_node_executed(
            node_path.as_slice(),
            "",
            start_instant,
            end_instant,
            backend,
        );
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
    key_input: SuperGraphLink,
    tokens_input: SuperGraphLink,
    state_inputs: Vec<(String, SuperGraphLink)>,
}

impl SuperGraphNodeRNNCacheWrite {
    pub fn new(
        key_input: SuperGraphLink,
        tokens_input: SuperGraphLink,
        state_inputs: Vec<(String, SuperGraphLink)>,
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
pub struct SuperGraphNodeTensorCacheRead {
    global_id: GlobalId,
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
        rng: &mut impl RngCore,
    ) -> Self {
        Self {
            global_id: GlobalId::new(rng),
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
    key_input: SuperGraphLink,
    value_input: SuperGraphLink,
    write_enable_input: SuperGraphLink,
}

impl SuperGraphNodeTensorCacheWrite {
    pub fn new(
        key_input: SuperGraphLink,
        value_input: SuperGraphLink,
        write_enable_input: SuperGraphLink,
        rng: &mut impl RngCore,
    ) -> Self {
        Self {
            global_id: GlobalId::new(rng),
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
        rng: &mut impl RngCore,
    ) -> Self {
        Self {
            global_id: GlobalId::new(rng),
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
    key_input: SuperGraphLink,
    value_inputs: Vec<(String, SuperGraphLink)>,
    write_enable_input: SuperGraphLink,
}

impl SuperGraphNodeTensorPackCacheWrite {
    pub fn new(
        key_input: SuperGraphLink,
        value_inputs: Vec<(String, SuperGraphLink)>,
        write_enable_input: SuperGraphLink,
        rng: &mut impl RngCore,
    ) -> Self {
        Self {
            global_id: GlobalId::new(rng),
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
    TensorToImage(SuperGraphNodeTensorToImage),
    MilliOpGraph(SuperGraphNodeMilliOpGraph),
    Scan(SuperGraphNodeScan),
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
                SuperGraphAnyNode::TensorToImage(x) => SuperGraphNode::$name(x,$($arg),*),
                SuperGraphAnyNode::MilliOpGraph(x) => SuperGraphNode::$name(x,$($arg),*),
                SuperGraphAnyNode::Scan(x) => SuperGraphNode::$name(x,$($arg),*),
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
            SuperGraphAnyNode::TensorToImage(node) => node.eval(node_path, data, context),
            SuperGraphAnyNode::MilliOpGraph(node) => node.eval(node_path, data, context),
            SuperGraphAnyNode::Scan(node) => node.eval(node_path, data, context),
            SuperGraphAnyNode::RNNCacheWrite(node) => node.eval(node_path, data, context),
            SuperGraphAnyNode::RNNCacheRead(node) => node.eval(node_path, data, context),
            SuperGraphAnyNode::TensorCacheRead(node) => node.eval(node_path, data, context),
            SuperGraphAnyNode::TensorCacheWrite(node) => node.eval(node_path, data, context),
            SuperGraphAnyNode::TensorPackCacheRead(node) => node.eval(node_path, data, context),
            SuperGraphAnyNode::TensorPackCacheWrite(node) => node.eval(node_path, data, context),
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
