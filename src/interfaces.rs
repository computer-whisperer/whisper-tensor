use crate::backends::ModelLoadedTensorCache;
use crate::backends::eval_backend::EvalBackend;
use crate::backends::ndarray_backend::NDArrayNumericTensor;
use crate::compiler::CompiledProgram;
use crate::dtype::DType;
use crate::interfaces::Error::GeneralError;
use crate::milli_graph::MilliOpGraph;
use crate::milli_graph::ops::{
    Cast, Constant, Shape, SimpleBinary, SimpleUnaryOp, Squeeze, Unsqueeze,
};
use crate::model::Model;
use crate::numeric_tensor::NumericTensor;
use crate::super_graph::cache::{SuperGraphCache, SuperGraphTensorCache};
use crate::super_graph::data::SuperGraphData;
use crate::super_graph::links::{
    SuperGraphLink, SuperGraphLinkDouble, SuperGraphLinkHash, SuperGraphLinkTensor,
    SuperGraphLinkTensorMap, SuperGraphLinkTriple,
};
use crate::super_graph::nodes::{
    SuperGraphNode, SuperGraphNodeMilliOpGraph, SuperGraphNodeModelExecution,
    SuperGraphNodeRNNCacheRead, SuperGraphNodeRNNCacheWrite, SuperGraphNodeScan,
};
use crate::super_graph::{SuperGraph, SuperGraphBuilder, SuperGraphContext, SuperGraphError};
use crate::symbolic_graph::SymbolicGraph;
use crate::tensor_rank::DynRank;
use crate::tokenizer::{AnyTokenizer, Tokenizer};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use whisper_tensor_import::onnx_graph::{
    InputMetadata, ModelInputType, ModelMetadata, ModelOutputType, OutputMetadata, TokenizerInfo,
};

pub fn get_automatic_interfaces_from_model(model: &Model) -> Vec<AnyInterface> {
    let mut interfaces = Vec::new();

    if let Some(x) = model.text_inference_tokens_in_logits_out_interface.clone() {
        interfaces.push(x.to_any());
    }

    interfaces
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum AnyInterface {
    TextInferenceTokensInLogitOutInterface(TextInferenceTokensInLogitOutInterface),
    StableDiffusionInterface(StableDiffusionInterface),
}

impl AnyInterface {
    pub fn name(&self) -> String {
        match self {
            AnyInterface::TextInferenceTokensInLogitOutInterface(_) => {
                "TextInferenceTokensInLogitsOut".to_string()
            }
            AnyInterface::StableDiffusionInterface(_) => "StableDiffusion".to_string(),
        }
    }

    pub fn get_super_graph(&self) -> &SuperGraph {
        match self {
            AnyInterface::TextInferenceTokensInLogitOutInterface(x) => &x.super_graph,
            AnyInterface::StableDiffusionInterface(x) => &x.super_graph,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TextInferenceTokensInLogitOutInterface {
    pub cache_key_input_link: SuperGraphLinkHash,
    pub token_context_input_link: SuperGraphLinkTensor,
    pub model_input_link: SuperGraphLinkTensorMap,
    pub logit_output_link: SuperGraphLinkTensor,
    pub super_graph: SuperGraph,
    pub tokenizer: TokenizerInfo,
}

pub enum Error {
    GeneralError,
}

impl TextInferenceTokensInLogitOutInterface {
    pub fn try_from_onnx_metadata(
        model_metadata: &ModelMetadata,
        model_inputs: &HashMap<String, Option<InputMetadata>>,
        model_outputs: &HashMap<String, Option<OutputMetadata>>,
        symbolic_graph: &SymbolicGraph,
        rng: &mut impl Rng,
    ) -> Result<Self, Error> {
        let tokenizer = model_metadata
            .tokenizer_infos
            .first()
            .ok_or(Error::GeneralError)?
            .clone();

        let (tokens_input_name, state_chain_input_names, state_chain_ids) = {
            let mut tokens_input_name = None;
            let mut state_chain_inputs = HashMap::new();
            let mut state_chain_ids = Vec::new();
            for (name, input) in model_inputs {
                if let Some(input) = input {
                    match input.model_input_type {
                        ModelInputType::TokenID(_) => {
                            tokens_input_name = Some(name.clone());
                        }
                        ModelInputType::PreviousInternal(id) => {
                            state_chain_inputs.insert(id, name.clone());
                            state_chain_ids.push(id);
                        }
                    }
                }
            }
            (
                tokens_input_name.ok_or(Error::GeneralError)?,
                state_chain_inputs,
                state_chain_ids,
            )
        };

        let (logit_output_name, state_chain_output_names) = {
            let mut logit_output_name = None;
            let mut state_chain_outputs = HashMap::new();
            for (name, output) in model_outputs {
                if let Some(output) = output {
                    match output.model_output_type {
                        ModelOutputType::TokenID(_) => {
                            logit_output_name = Some(name.clone());
                        }
                        ModelOutputType::NextInternal(id) => {
                            state_chain_outputs.insert(id, name.clone());
                        }
                    }
                }
            }
            (
                logit_output_name.ok_or(Error::GeneralError)?,
                state_chain_outputs,
            )
        };

        // Build super graph
        let mut super_graph_builder = SuperGraphBuilder::new();
        let token_context_input_link = super_graph_builder.new_tensor_link(rng);

        let model_input_link = super_graph_builder.new_model_link(rng);
        let raw_logit_output_link = super_graph_builder.new_tensor_link(rng);

        let cache_key = super_graph_builder.new_hash_link(rng);

        let processed_logit_output_link = if Some(1) == model_metadata.max_token_batch {
            // RNN compatibility mode

            let state_init_links = state_chain_ids
                .iter()
                .map(|id| (*id, super_graph_builder.new_tensor_link(rng)))
                .collect::<Vec<(_, _)>>();

            // Cache
            let (post_cache_tokens_input, post_cache_state_init_links) = {
                let post_cache_state_init_links = state_chain_ids
                    .iter()
                    .map(|id| (*id, super_graph_builder.new_tensor_link(rng)))
                    .collect::<Vec<(_, _)>>();
                let post_cache_tokens = super_graph_builder.new_tensor_link(rng);
                let node = SuperGraphNodeRNNCacheRead::new(
                    cache_key,
                    token_context_input_link,
                    post_cache_tokens,
                    post_cache_state_init_links
                        .iter()
                        .map(|(id, link)| (id.to_string(), *link))
                        .collect::<Vec<(_, _)>>(),
                    state_init_links
                        .iter()
                        .map(|(id, link)| (id.to_string(), *link))
                        .collect::<Vec<(_, _)>>(),
                    rng,
                );
                super_graph_builder.add_node(node.to_any());
                (post_cache_tokens, post_cache_state_init_links)
            };

            let loop_count_link = {
                let loop_count_link = SuperGraphLinkTensor::new(rng);

                let (mut milli_graph, input_map) =
                    MilliOpGraph::new(std::iter::once(post_cache_tokens_input.global_id()), rng);
                let milli_op_graph_input =
                    *input_map.get(&post_cache_tokens_input.global_id()).unwrap();

                let shape_out = Shape::push_new(&mut milli_graph, milli_op_graph_input, rng);
                milli_graph
                    .set_output_map(std::iter::once((shape_out, loop_count_link.global_id())));

                let node = SuperGraphNodeMilliOpGraph::new(milli_graph, rng);
                super_graph_builder.add_node(node.to_any());
                loop_count_link
            };

            // Sub graph
            let (processed_logit_output_link, final_state_outputs) = {
                // State initialization
                {
                    let (mut milli_graph, _) = MilliOpGraph::new(std::iter::empty(), rng);
                    let mut output_map = HashMap::new();
                    let mut output_order = vec![];
                    for (id, link) in state_init_links.iter() {
                        let input_name = state_chain_input_names.get(id).unwrap();
                        let input_tensor_id = *symbolic_graph
                            .get_tensors_by_name()
                            .get(input_name)
                            .unwrap();
                        let input_tensor_info =
                            symbolic_graph.get_tensor_info(input_tensor_id).unwrap();
                        let input_tensor_shape = input_tensor_info
                            .shape
                            .clone()
                            .unwrap()
                            .iter()
                            .map(|x| x.as_numeric().ok_or(GeneralError).cloned())
                            .collect::<Result<Vec<_>, Error>>()?;
                        let input_tensor_dtype = input_tensor_info.dtype.unwrap();
                        let num_elements = input_tensor_shape.iter().product::<u64>();
                        let input_tensor = NDArrayNumericTensor::from_vec_shape(
                            vec![0.0; num_elements as usize],
                            &input_tensor_shape,
                        )
                        .unwrap()
                        .cast(input_tensor_dtype)
                        .unwrap();
                        let input_tensor_tid =
                            Constant::push_new(&mut milli_graph, input_tensor, rng);
                        output_map.insert(input_tensor_tid, link.global_id());
                        output_order.push(link.global_id());
                    }
                    milli_graph.set_output_map_ordered(output_map, output_order);

                    let node = SuperGraphNodeMilliOpGraph::new(milli_graph, rng);
                    super_graph_builder.add_node(node.to_any());
                }

                let mut sub_builder = SuperGraphBuilder::new();

                let sub_model_input_link = sub_builder.new_model_link(rng);
                let sub_token_input = sub_builder.new_tensor_link(rng);

                let state_input_links = state_chain_ids
                    .iter()
                    .map(|id| (*id, sub_builder.new_tensor_link(rng)))
                    .collect::<HashMap<_, _>>();
                let state_output_links = state_chain_ids
                    .iter()
                    .map(|id| (*id, sub_builder.new_tensor_link(rng)))
                    .collect::<HashMap<_, _>>();
                let final_state_output_links = state_chain_ids
                    .iter()
                    .map(|id| (*id, super_graph_builder.new_tensor_link(rng)))
                    .collect::<Vec<(_, _)>>();

                // Input processing
                let adjusted_token_context = {
                    // Convert dtype and reshape according to model
                    let input_tensor_id = *symbolic_graph
                        .get_tensors_by_name()
                        .get(&tokens_input_name)
                        .unwrap();
                    let input_tensor_info =
                        symbolic_graph.get_tensor_info(input_tensor_id).unwrap();
                    let input_tensor_rank = input_tensor_info.shape.clone().unwrap().len();
                    let input_tensor_dtype = input_tensor_info.dtype.unwrap();

                    let (mut milli_graph, input_map) =
                        MilliOpGraph::new(std::iter::once(sub_token_input.global_id()), rng);
                    let milli_op_graph_input =
                        *input_map.get(&sub_token_input.global_id()).unwrap();
                    let mut x = Cast::push_new(
                        &mut milli_graph,
                        milli_op_graph_input,
                        input_tensor_dtype,
                        rng,
                    );
                    let zero_tid = Constant::push_new(
                        &mut milli_graph,
                        NDArrayNumericTensor::from_vec_shape(vec![0i64], &vec![1]).unwrap(),
                        rng,
                    );
                    for _ in 0..input_tensor_rank {
                        x = Unsqueeze::push_new(&mut milli_graph, x, zero_tid, rng);
                    }

                    let processed_input_link = sub_builder.new_tensor_link(rng);
                    milli_graph
                        .set_output_map(std::iter::once((x, processed_input_link.global_id())));

                    let node = SuperGraphNodeMilliOpGraph::new(milli_graph, rng);
                    sub_builder.add_node(node.to_any());
                    processed_input_link
                };

                let tensor_inputs = {
                    let mut tensor_inputs = Vec::new();
                    tensor_inputs.push((adjusted_token_context, tokens_input_name));
                    for id in &state_chain_ids {
                        tensor_inputs.push((
                            *state_input_links.get(id).unwrap(),
                            state_chain_input_names.get(id).unwrap().clone(),
                        ));
                    }
                    tensor_inputs
                };

                let sub_logit_output = sub_builder.new_tensor_link(rng);
                let tensor_outputs = {
                    let mut tensor_outputs = Vec::new();
                    tensor_outputs.push((logit_output_name.clone(), sub_logit_output));
                    for id in &state_chain_ids {
                        tensor_outputs.push((
                            state_chain_output_names.get(id).unwrap().clone(),
                            *state_output_links.get(id).unwrap(),
                        ));
                    }
                    tensor_outputs
                };

                sub_builder.add_node(
                    SuperGraphNodeModelExecution::new(
                        rng,
                        sub_model_input_link,
                        0,
                        tensor_inputs,
                        tensor_outputs,
                    )
                    .to_any(),
                );

                let processed_logit_output_link = {
                    // Convert dtype and reshape according to model
                    let output_tensor_id = *symbolic_graph
                        .get_tensors_by_name()
                        .get(&logit_output_name)
                        .unwrap();
                    let output_tensor_info =
                        symbolic_graph.get_tensor_info(output_tensor_id).unwrap();
                    let output_tensor_rank = output_tensor_info.shape.clone().unwrap().len();

                    let (mut milli_graph, input_map) =
                        MilliOpGraph::new(std::iter::once(sub_logit_output.global_id()), rng);
                    let milli_op_graph_input =
                        *input_map.get(&sub_logit_output.global_id()).unwrap();
                    let mut x = milli_op_graph_input;
                    let zero_tid = Constant::push_new(
                        &mut milli_graph,
                        NDArrayNumericTensor::from_vec_shape(vec![0i64], &vec![1]).unwrap(),
                        rng,
                    );
                    for _ in 0..(output_tensor_rank - 1) {
                        x = Squeeze::push_new(&mut milli_graph, x, zero_tid, rng);
                    }
                    x = Cast::push_new(&mut milli_graph, x, DType::F32, rng);

                    let processed_logit_output_link = sub_builder.new_tensor_link(rng);
                    milli_graph.set_output_map(std::iter::once((
                        x,
                        processed_logit_output_link.global_id(),
                    )));

                    let node = SuperGraphNodeMilliOpGraph::new(milli_graph, rng);
                    sub_builder.add_node(node.to_any());
                    processed_logit_output_link
                };

                let state_links = {
                    let mut state_links = vec![];

                    for (id, init_link) in &post_cache_state_init_links {
                        let link_triple = SuperGraphLinkTriple::Tensor(
                            *init_link,
                            *state_input_links.get(id).unwrap(),
                            *state_output_links.get(id).unwrap(),
                        );

                        state_links.push(link_triple);
                    }

                    state_links
                };

                let outer_logit_output_link = super_graph_builder.new_tensor_link(rng);

                let input_links = {
                    let mut input_links =
                        vec![sub_model_input_link.to_any(), sub_token_input.to_any()];
                    for &id in &state_chain_ids {
                        input_links.push(state_input_links.get(&id).unwrap().to_any());
                    }
                    input_links
                };
                let output_links = {
                    let mut output_links = vec![processed_logit_output_link.to_any()];
                    for id in &state_chain_ids {
                        output_links.push(state_output_links.get(id).unwrap().to_any());
                    }
                    output_links
                };

                let sub_graph_inner = sub_builder.build(rng, &input_links, &output_links);

                let final_state_outputs = {
                    let mut final_state_outputs = vec![];
                    for (id, link) in &final_state_output_links {
                        final_state_outputs.push(SuperGraphLinkDouble::Tensor(
                            *state_output_links.get(id).unwrap(),
                            *link,
                        ))
                    }
                    final_state_outputs
                };

                let scan_node = SuperGraphNodeScan::new(
                    sub_graph_inner,
                    loop_count_link,
                    vec![SuperGraphLinkDouble::TensorMap(
                        model_input_link,
                        sub_model_input_link,
                    )],
                    state_links,
                    vec![(post_cache_tokens_input, sub_token_input, 0)],
                    vec![(processed_logit_output_link, outer_logit_output_link, 0)],
                    final_state_outputs,
                    rng,
                );
                super_graph_builder.add_node(scan_node.to_any());

                (outer_logit_output_link, final_state_output_links)
            };

            // Cache write
            {
                let node = SuperGraphNodeRNNCacheWrite::new(
                    cache_key,
                    token_context_input_link,
                    final_state_outputs
                        .iter()
                        .map(|(id, link)| (id.to_string(), *link))
                        .collect::<Vec<(_, _)>>(),
                    rng,
                );
                super_graph_builder.add_node(node.to_any());
            }

            processed_logit_output_link
        } else {
            // Input processing
            let adjusted_token_context = {
                // Convert dtype and reshape according to model
                let input_tensor_id = *symbolic_graph
                    .get_tensors_by_name()
                    .get(&tokens_input_name)
                    .unwrap();
                let input_tensor_info = symbolic_graph.get_tensor_info(input_tensor_id).unwrap();
                let input_tensor_rank = input_tensor_info.shape.clone().unwrap().len();
                let input_tensor_dtype = input_tensor_info.dtype.unwrap();

                let (mut milli_graph, input_map) =
                    MilliOpGraph::new(std::iter::once(token_context_input_link.global_id()), rng);
                let milli_op_graph_input = *input_map
                    .get(&token_context_input_link.global_id())
                    .unwrap();
                let mut x = Cast::push_new(
                    &mut milli_graph,
                    milli_op_graph_input,
                    input_tensor_dtype,
                    rng,
                );
                let zero_tid = Constant::push_new(
                    &mut milli_graph,
                    NDArrayNumericTensor::from_vec_shape(vec![0i64], &vec![1]).unwrap(),
                    rng,
                );
                for _ in 0..(input_tensor_rank - 1) {
                    x = Unsqueeze::push_new(&mut milli_graph, x, zero_tid, rng);
                }

                let processed_input_link = SuperGraphLinkTensor::new(rng);
                let mut output_map = HashMap::new();
                output_map.insert(x, processed_input_link.global_id());
                milli_graph.set_output_map(output_map);

                let node = SuperGraphNodeMilliOpGraph::new(milli_graph, rng);
                super_graph_builder.add_node(node.to_any());
                processed_input_link
            };

            let tensor_inputs = vec![(adjusted_token_context, tokens_input_name)];

            let tensor_outputs = vec![(logit_output_name.clone(), raw_logit_output_link)];

            super_graph_builder.add_node(
                SuperGraphNodeModelExecution::new(
                    rng,
                    model_input_link,
                    0,
                    tensor_inputs,
                    tensor_outputs,
                )
                .to_any(),
            );

            // Output processing
            {
                // Convert dtype and reshape according to model
                let output_tensor_id = *symbolic_graph
                    .get_tensors_by_name()
                    .get(&logit_output_name)
                    .unwrap();
                let output_tensor_info = symbolic_graph.get_tensor_info(output_tensor_id).unwrap();
                let output_tensor_rank = output_tensor_info.shape.clone().unwrap().len();

                let (mut milli_graph, input_map) =
                    MilliOpGraph::new(std::iter::once(raw_logit_output_link.global_id()), rng);
                let milli_op_graph_input =
                    *input_map.get(&raw_logit_output_link.global_id()).unwrap();
                let mut x = milli_op_graph_input;
                let zero_tid = Constant::push_new(
                    &mut milli_graph,
                    NDArrayNumericTensor::from_vec_shape(vec![0i64], &vec![1]).unwrap(),
                    rng,
                );
                for _ in 0..(output_tensor_rank - 2) {
                    x = Squeeze::push_new(&mut milli_graph, x, zero_tid, rng);
                }
                x = Cast::push_new(&mut milli_graph, x, DType::F32, rng);

                let processed_logit_output_link = super_graph_builder.new_tensor_link(rng);
                let mut output_map = HashMap::new();
                output_map.insert(x, processed_logit_output_link.global_id());
                milli_graph.set_output_map(output_map);

                let node = SuperGraphNodeMilliOpGraph::new(milli_graph, rng);
                super_graph_builder.add_node(node.to_any());
                processed_logit_output_link
            }
        };

        let super_graph_inputs = vec![
            cache_key.to_any(),
            model_input_link.to_any(),
            token_context_input_link.to_any(),
        ];
        let super_graph_outputs = vec![processed_logit_output_link.to_any()];
        let super_graph = super_graph_builder.build(
            rng,
            super_graph_inputs.as_slice(),
            super_graph_outputs.as_slice(),
        );

        Ok(Self {
            tokenizer,
            model_input_link,
            token_context_input_link,
            logit_output_link: processed_logit_output_link,
            super_graph,
            cache_key_input_link: cache_key,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run_string_in_string_out(
        &self,
        model: &Model,
        compiled_model: Option<&CompiledProgram>,
        text_in: String,
        tokenizer_cache: &mut HashMap<TokenizerInfo, Arc<AnyTokenizer>>,
        tensor_cache: Option<&mut ModelLoadedTensorCache>,
        super_graph_caches: Option<&mut SuperGraphCache>,
        backend: &mut EvalBackend,
    ) -> Result<String, SuperGraphError> {
        let tokenizer = {
            if let Some(x) = tokenizer_cache.get(&self.tokenizer) {
                x.clone()
            } else {
                let x = Arc::new(AnyTokenizer::from_tokenizer_info(&self.tokenizer));
                tokenizer_cache.insert(self.tokenizer.clone(), x.clone());
                x
            }
        };
        let tokens = tokenizer.encode(text_in.as_str());
        let tokens_tensor = NumericTensor::from_vec(tokens.clone()).to_dyn_rank();

        let super_graph_data = {
            let mut super_graph_data = SuperGraphData::new();
            super_graph_data
                .tensor_maps
                .insert(self.model_input_link, model.get_tensor_store());
            super_graph_data
                .tensors
                .insert(self.token_context_input_link, tokens_tensor);
            super_graph_data.hashes.insert(self.cache_key_input_link, 0);
            super_graph_data
        };
        let super_graph_output = {
            let mut observer = ();
            let mut super_graph_tensor_cache = SuperGraphTensorCache::new();
            if let Some(tensor_cache) = &tensor_cache {
                super_graph_tensor_cache
                    .caches
                    .push((model.get_tensor_store(), (*tensor_cache).clone()))
            }
            let compiled_models = {
                let mut compiled_models = Vec::new();
                if let Some(compiled_model) = compiled_model {
                    compiled_models.push((model, compiled_model));
                }
                compiled_models
            };
            let mut context = SuperGraphContext {
                observer: &mut observer,
                eval_backend: backend,
                super_graph_tensor_cache: &mut super_graph_tensor_cache,
                caches: super_graph_caches,
                symbolic_graphs: vec![model.get_symbolic_graph()],
                use_compiled_models: compiled_model.is_some(),
                compiled_models: Some(compiled_models),
            };
            let res = self.super_graph.run(super_graph_data, &mut context)?;
            if let Some(tensor_cache) = tensor_cache {
                *tensor_cache = context.super_graph_tensor_cache.caches.remove(0).1
            }
            res
        };
        let logits = super_graph_output
            .tensors
            .get(&self.logit_output_link)
            .unwrap();
        let logits_shape = logits.shape();
        // Select last position
        let logits = logits.slice(
            &[logits_shape[0] - 1..logits_shape[0], 0..logits_shape[1]],
            backend,
        )?;
        let logits = logits.squeeze(0)?;
        let token_id = logits.argmax(0, true, false, backend)?;

        let token_id: u32 = token_id.first_element().into();
        let token_str = tokenizer.decode(&[token_id])?;
        Ok(token_str)
    }

    pub fn get_tokenizer(&self) -> &TokenizerInfo {
        &self.tokenizer
    }

    pub fn to_any(self) -> AnyInterface {
        AnyInterface::TextInferenceTokensInLogitOutInterface(self)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StableDiffusionInterface {
    pub super_graph: SuperGraph,
    // Inputs
    pub cond_ids_input: SuperGraphLinkTensor,
    pub uncond_ids_input: SuperGraphLinkTensor,
    pub initial_latent_input: SuperGraphLinkTensor,
    pub timesteps_input: SuperGraphLinkTensor,
    pub dt_input: SuperGraphLinkTensor,
    pub sigmas_input: SuperGraphLinkTensor,
    pub iteration_count_input: SuperGraphLinkTensor,
    pub guidance_scale_input: SuperGraphLinkTensor,
    // Model weight maps
    pub text_encoder_weights: SuperGraphLinkTensorMap,
    pub unet_weights: SuperGraphLinkTensorMap,
    pub vae_decoder_weights: SuperGraphLinkTensorMap,
    // Output
    pub image_output: SuperGraphLinkTensor,
}

impl StableDiffusionInterface {
    pub fn new(rng: &mut impl Rng) -> Self {
        let mut builder = SuperGraphBuilder::new();

        // Create input links
        let cond_ids_input = builder.new_tensor_link(rng);
        let uncond_ids_input = builder.new_tensor_link(rng);
        let initial_latent_input = builder.new_tensor_link(rng);
        let timesteps_input = builder.new_tensor_link(rng);
        let dt_input = builder.new_tensor_link(rng);
        let sigmas_input = builder.new_tensor_link(rng);
        let iteration_count_input = builder.new_tensor_link(rng);
        let guidance_scale_input = builder.new_tensor_link(rng);
        let text_encoder_weights = builder.new_model_link(rng);
        let unet_weights = builder.new_model_link(rng);
        let vae_decoder_weights = builder.new_model_link(rng);

        // Node 1: Text encoder (conditional)
        let cond_hidden = builder.new_tensor_link(rng);
        builder.add_node(
            SuperGraphNodeModelExecution::new(
                rng,
                text_encoder_weights,
                0,
                vec![(cond_ids_input, "input_ids".to_string())],
                vec![("last_hidden_state".to_string(), cond_hidden)],
            )
            .to_any(),
        );

        // Node 2: Text encoder (unconditional)
        let uncond_hidden = builder.new_tensor_link(rng);
        builder.add_node(
            SuperGraphNodeModelExecution::new(
                rng,
                text_encoder_weights,
                0,
                vec![(uncond_ids_input, "input_ids".to_string())],
                vec![("last_hidden_state".to_string(), uncond_hidden)],
            )
            .to_any(),
        );

        // Node 3: Scan (denoising loop)
        let outer_final_latent = builder.new_tensor_link(rng);
        {
            let mut inner_builder = SuperGraphBuilder::new();

            // Inner links
            let inner_unet_weights = inner_builder.new_model_link(rng);
            let inner_cond_hidden = inner_builder.new_tensor_link(rng);
            let inner_uncond_hidden = inner_builder.new_tensor_link(rng);
            let inner_guidance_scale = inner_builder.new_tensor_link(rng);
            let inner_latent_in = inner_builder.new_tensor_link(rng);
            let inner_latent_out = inner_builder.new_tensor_link(rng);
            let inner_timestep = inner_builder.new_tensor_link(rng);
            let inner_dt = inner_builder.new_tensor_link(rng);
            let inner_sigma = inner_builder.new_tensor_link(rng);

            // Inner node 1: Prep — scale latent by 1/sqrt(sigma²+1), cast f32→f16, cast timestep f32→f16, reshape to [1]
            let f16_latent = inner_builder.new_tensor_link(rng);
            let f16_timestep = inner_builder.new_tensor_link(rng);
            {
                let (mut mg, input_map) = MilliOpGraph::new(
                    [
                        inner_latent_in.global_id(),
                        inner_timestep.global_id(),
                        inner_sigma.global_id(),
                    ],
                    rng,
                );
                let lat_in = *input_map.get(&inner_latent_in.global_id()).unwrap();
                let ts_in = *input_map.get(&inner_timestep.global_id()).unwrap();
                let sigma_in = *input_map.get(&inner_sigma.global_id()).unwrap();

                // scale = 1 / sqrt(sigma^2 + 1)
                let sigma_sq = SimpleBinary::mul(&mut mg, sigma_in, sigma_in, rng);
                let one = Constant::push_new(
                    &mut mg,
                    NDArrayNumericTensor::from_vec_shape(vec![1.0f32], &vec![1]).unwrap(),
                    rng,
                );
                let sigma_sq_plus_1 = SimpleBinary::add(&mut mg, sigma_sq, one, rng);
                let sqrt_val = SimpleUnaryOp::sqrt(&mut mg, sigma_sq_plus_1, rng);
                let inv_scale = SimpleBinary::div(&mut mg, one, sqrt_val, rng);
                let scaled_lat = SimpleBinary::mul(&mut mg, lat_in, inv_scale, rng);

                let lat_f16 = Cast::push_new(&mut mg, scaled_lat, DType::F16, rng);
                let ts_f16 = Cast::push_new(&mut mg, ts_in, DType::F16, rng);
                let zero_axis = Constant::push_new(
                    &mut mg,
                    NDArrayNumericTensor::from_vec_shape(vec![0i64], &vec![1]).unwrap(),
                    rng,
                );
                let ts_reshaped = Unsqueeze::push_new(&mut mg, ts_f16, zero_axis, rng);

                mg.set_output_map([
                    (lat_f16, f16_latent.global_id()),
                    (ts_reshaped, f16_timestep.global_id()),
                ]);
                inner_builder.add_node(SuperGraphNodeMilliOpGraph::new(mg, rng).to_any());
            }

            // Inner node 2: UNet unconditional
            let uncond_noise = inner_builder.new_tensor_link(rng);
            inner_builder.add_node(
                SuperGraphNodeModelExecution::new(
                    rng,
                    inner_unet_weights,
                    1,
                    vec![
                        (f16_latent, "sample".to_string()),
                        (f16_timestep, "timestep".to_string()),
                        (inner_uncond_hidden, "encoder_hidden_states".to_string()),
                    ],
                    vec![("out_sample".to_string(), uncond_noise)],
                )
                .to_any(),
            );

            // Inner node 3: UNet conditional
            let cond_noise = inner_builder.new_tensor_link(rng);
            inner_builder.add_node(
                SuperGraphNodeModelExecution::new(
                    rng,
                    inner_unet_weights,
                    1,
                    vec![
                        (f16_latent, "sample".to_string()),
                        (f16_timestep, "timestep".to_string()),
                        (inner_cond_hidden, "encoder_hidden_states".to_string()),
                    ],
                    vec![("out_sample".to_string(), cond_noise)],
                )
                .to_any(),
            );

            // Inner node 4: CFG + Euler step
            {
                let (mut mg, input_map) = MilliOpGraph::new(
                    [
                        uncond_noise.global_id(),
                        cond_noise.global_id(),
                        inner_latent_in.global_id(),
                        inner_guidance_scale.global_id(),
                        inner_dt.global_id(),
                    ],
                    rng,
                );
                let uncond_in = *input_map.get(&uncond_noise.global_id()).unwrap();
                let cond_in = *input_map.get(&cond_noise.global_id()).unwrap();
                let lat_in = *input_map.get(&inner_latent_in.global_id()).unwrap();
                let gs_in = *input_map.get(&inner_guidance_scale.global_id()).unwrap();
                let dt_in = *input_map.get(&inner_dt.global_id()).unwrap();

                // Cast noises to f32
                let uncond_f32 = Cast::push_new(&mut mg, uncond_in, DType::F32, rng);
                let cond_f32 = Cast::push_new(&mut mg, cond_in, DType::F32, rng);

                // CFG: uncond + scale * (cond - uncond)
                let diff = SimpleBinary::sub(&mut mg, cond_f32, uncond_f32, rng);
                let scaled = SimpleBinary::mul(&mut mg, diff, gs_in, rng);
                let guided = SimpleBinary::add(&mut mg, uncond_f32, scaled, rng);

                // Euler step: latent + guided * dt
                let step = SimpleBinary::mul(&mut mg, guided, dt_in, rng);
                let latent_next = SimpleBinary::add(&mut mg, lat_in, step, rng);

                mg.set_output_map(std::iter::once((latent_next, inner_latent_out.global_id())));
                inner_builder.add_node(SuperGraphNodeMilliOpGraph::new(mg, rng).to_any());
            }

            // Build inner graph
            let inner_inputs: Vec<_> = vec![
                inner_unet_weights.to_any(),
                inner_cond_hidden.to_any(),
                inner_uncond_hidden.to_any(),
                inner_guidance_scale.to_any(),
                inner_latent_in.to_any(),
                inner_timestep.to_any(),
                inner_dt.to_any(),
                inner_sigma.to_any(),
            ];
            let inner_outputs: Vec<_> = vec![inner_latent_out.to_any()];
            let inner_graph = inner_builder.build(rng, &inner_inputs, &inner_outputs);

            // Create scan node
            let scan_node = SuperGraphNodeScan::new(
                inner_graph,
                iteration_count_input,
                // simple_inputs: constant across iterations
                vec![
                    SuperGraphLinkDouble::TensorMap(unet_weights, inner_unet_weights),
                    SuperGraphLinkDouble::Tensor(cond_hidden, inner_cond_hidden),
                    SuperGraphLinkDouble::Tensor(uncond_hidden, inner_uncond_hidden),
                    SuperGraphLinkDouble::Tensor(guidance_scale_input, inner_guidance_scale),
                ],
                // state_links: (initial, inner_in, inner_out)
                vec![SuperGraphLinkTriple::Tensor(
                    initial_latent_input,
                    inner_latent_in,
                    inner_latent_out,
                )],
                // scan_inputs: (outer, inner, axis)
                vec![
                    (timesteps_input, inner_timestep, 0),
                    (dt_input, inner_dt, 0),
                    (sigmas_input, inner_sigma, 0),
                ],
                // scan_outputs: none
                vec![],
                // simple_outputs: final latent
                vec![SuperGraphLinkDouble::Tensor(
                    inner_latent_out,
                    outer_final_latent,
                )],
                rng,
            );
            builder.add_node(scan_node.to_any());
        }

        // Node 4: Scale latent by 1/0.18215 and cast f32→f16
        let scaled_latent = builder.new_tensor_link(rng);
        {
            let (mut mg, input_map) =
                MilliOpGraph::new(std::iter::once(outer_final_latent.global_id()), rng);
            let lat_in = *input_map.get(&outer_final_latent.global_id()).unwrap();

            let scale = Constant::push_new(
                &mut mg,
                NDArrayNumericTensor::from_vec_shape(vec![1.0f32 / 0.18215], &vec![1]).unwrap(),
                rng,
            );
            let scaled = SimpleBinary::mul(&mut mg, lat_in, scale, rng);
            let scaled_f16 = Cast::push_new(&mut mg, scaled, DType::F16, rng);

            mg.set_output_map(std::iter::once((scaled_f16, scaled_latent.global_id())));
            builder.add_node(SuperGraphNodeMilliOpGraph::new(mg, rng).to_any());
        }

        // Node 5: VAE decoder
        let image_output = builder.new_tensor_link(rng);
        builder.add_node(
            SuperGraphNodeModelExecution::new(
                rng,
                vae_decoder_weights,
                2,
                vec![(scaled_latent, "latent_sample".to_string())],
                vec![("sample".to_string(), image_output)],
            )
            .to_any(),
        );

        // Build outer graph
        let input_links: Vec<_> = vec![
            cond_ids_input.to_any(),
            uncond_ids_input.to_any(),
            initial_latent_input.to_any(),
            timesteps_input.to_any(),
            dt_input.to_any(),
            sigmas_input.to_any(),
            iteration_count_input.to_any(),
            guidance_scale_input.to_any(),
            text_encoder_weights.to_any(),
            unet_weights.to_any(),
            vae_decoder_weights.to_any(),
        ];
        let output_links: Vec<_> = vec![image_output.to_any()];
        let super_graph = builder.build(rng, &input_links, &output_links);

        Self {
            super_graph,
            cond_ids_input,
            uncond_ids_input,
            initial_latent_input,
            timesteps_input,
            dt_input,
            sigmas_input,
            iteration_count_input,
            guidance_scale_input,
            text_encoder_weights,
            unet_weights,
            vae_decoder_weights,
            image_output,
        }
    }

    /// Pre-compute the Euler discrete scheduler parameters.
    /// Returns (timestep_values, dt_values, initial_sigma).
    pub fn compute_euler_schedule(
        num_inference_steps: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, f32) {
        let num_train_timesteps = 1000;
        let beta_start: f32 = 0.00085;
        let beta_end: f32 = 0.012;

        // Linear beta schedule (scaled_linear: sqrt of linear interpolation)
        let betas: Vec<f32> = (0..num_train_timesteps)
            .map(|i| {
                let t = i as f32 / (num_train_timesteps - 1) as f32;
                let b = beta_start.sqrt() + t * (beta_end.sqrt() - beta_start.sqrt());
                b * b
            })
            .collect();

        let mut alphas_cumprod = Vec::with_capacity(num_train_timesteps);
        let mut cumprod = 1.0f32;
        for &beta in &betas {
            cumprod *= 1.0 - beta;
            alphas_cumprod.push(cumprod);
        }

        // Evenly spaced timesteps (descending)
        let step_ratio = num_train_timesteps / num_inference_steps;
        let timesteps: Vec<usize> = (0..num_inference_steps)
            .rev()
            .map(|i| i * step_ratio + step_ratio - 1)
            .collect();

        // Compute sigmas from alphas_cumprod
        let sigmas: Vec<f32> = timesteps
            .iter()
            .map(|&t| ((1.0 - alphas_cumprod[t]) / alphas_cumprod[t]).sqrt())
            .collect();

        let init_sigma = sigmas[0];

        // Timestep values as f32
        let timestep_values: Vec<f32> = timesteps.iter().map(|&t| t as f32).collect();

        // dt[i] = sigma[i+1] - sigma[i], with sigma[num_steps] = 0
        let dt_values: Vec<f32> = (0..num_inference_steps)
            .map(|i| {
                let sigma_next = if i + 1 < num_inference_steps {
                    sigmas[i + 1]
                } else {
                    0.0
                };
                sigma_next - sigmas[i]
            })
            .collect();

        (timestep_values, dt_values, sigmas, init_sigma)
    }

    /// Run the full SD pipeline. Pre-computes scheduler params, packs inputs, runs graph.
    #[allow(clippy::too_many_arguments)]
    pub fn run<'a>(
        &self,
        text_encoder: &'a Model,
        unet: &'a Model,
        vae_decoder: &'a Model,
        cond_ids: NumericTensor<DynRank>,
        uncond_ids: NumericTensor<DynRank>,
        initial_noise: Vec<f32>,
        latent_shape: Vec<usize>,
        num_inference_steps: usize,
        guidance_scale: f32,
        backend: &mut EvalBackend,
    ) -> Result<NumericTensor<DynRank>, SuperGraphError> {
        let (timestep_values, dt_values, sigma_values, init_sigma) =
            Self::compute_euler_schedule(num_inference_steps);

        // Scale initial noise by initial sigma
        let scaled_noise: Vec<f32> = initial_noise.iter().map(|&x| x * init_sigma).collect();
        let latent_tensor =
            NumericTensor::<DynRank>::from_vec_shape(scaled_noise, latent_shape).unwrap();

        let timesteps_tensor =
            NumericTensor::<DynRank>::from_vec_shape(timestep_values, vec![num_inference_steps])
                .unwrap();
        let dt_tensor =
            NumericTensor::<DynRank>::from_vec_shape(dt_values, vec![num_inference_steps]).unwrap();
        let sigmas_tensor =
            NumericTensor::<DynRank>::from_vec_shape(sigma_values, vec![num_inference_steps])
                .unwrap();
        let iter_count =
            NumericTensor::<DynRank>::from_vec_shape(vec![num_inference_steps as i64], vec![1])
                .unwrap();
        let guidance =
            NumericTensor::<DynRank>::from_vec_shape(vec![guidance_scale], vec![]).unwrap();

        // Pack data
        let mut data = SuperGraphData::new();
        data.tensors.insert(self.cond_ids_input, cond_ids);
        data.tensors.insert(self.uncond_ids_input, uncond_ids);
        data.tensors
            .insert(self.initial_latent_input, latent_tensor);
        data.tensors.insert(self.timesteps_input, timesteps_tensor);
        data.tensors.insert(self.dt_input, dt_tensor);
        data.tensors.insert(self.sigmas_input, sigmas_tensor);
        data.tensors.insert(self.iteration_count_input, iter_count);
        data.tensors.insert(self.guidance_scale_input, guidance);
        data.tensor_maps
            .insert(self.text_encoder_weights, text_encoder.get_tensor_store());
        data.tensor_maps
            .insert(self.unet_weights, unet.get_tensor_store());
        data.tensor_maps
            .insert(self.vae_decoder_weights, vae_decoder.get_tensor_store());

        // Run
        let mut observer = ();
        let mut tensor_cache = SuperGraphTensorCache::new();
        let mut context = SuperGraphContext {
            observer: &mut observer,
            eval_backend: backend,
            super_graph_tensor_cache: &mut tensor_cache,
            caches: None,
            symbolic_graphs: vec![
                text_encoder.get_symbolic_graph(),
                unet.get_symbolic_graph(),
                vae_decoder.get_symbolic_graph(),
            ],
            use_compiled_models: false,
            compiled_models: None,
        };

        let result = self.super_graph.run(data, &mut context)?;
        let image = result.tensors.get(&self.image_output).unwrap().clone();
        Ok(image)
    }

    pub fn to_any(self) -> AnyInterface {
        AnyInterface::StableDiffusionInterface(self)
    }
}
