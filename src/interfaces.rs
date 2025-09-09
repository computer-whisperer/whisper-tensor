use crate::backends::ModelLoadedTensorCache;
use crate::backends::eval_backend::EvalBackend;
use crate::backends::ndarray_backend::NDArrayNumericTensor;
use crate::dtype::DType;
use crate::interfaces::Error::GeneralError;
use crate::milli_graph::MilliOpGraph;
use crate::milli_graph::ops::{Cast, Constant, Shape, Squeeze, Unsqueeze};
use crate::model::Model;
use crate::numeric_tensor::NumericTensor;
use crate::super_graph::cache::{SuperGraphCache, SuperGraphTensorCache};
use crate::super_graph::data::SuperGraphData;
use crate::super_graph::links::{
    SuperGraphLink, SuperGraphLinkDouble, SuperGraphLinkHash, SuperGraphLinkModel,
    SuperGraphLinkTensor, SuperGraphLinkTriple,
};
use crate::super_graph::nodes::{
    SuperGraphNode, SuperGraphNodeMilliOpGraph, SuperGraphNodeModelExecution,
    SuperGraphNodeRNNCacheRead, SuperGraphNodeRNNCacheWrite, SuperGraphNodeScan,
};
use crate::super_graph::{SuperGraph, SuperGraphBuilder, SuperGraphContext, SuperGraphError};
use crate::symbolic_graph::SymbolicGraph;
use crate::tokenizer::{AnyTokenizer, Tokenizer};
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
}

impl AnyInterface {
    pub fn name(&self) -> String {
        match self {
            AnyInterface::TextInferenceTokensInLogitOutInterface(_) => {
                "TextInferenceTokensInLogitsOut".to_string()
            }
        }
    }

    pub fn get_super_graph(&self) -> &SuperGraph {
        match self {
            AnyInterface::TextInferenceTokensInLogitOutInterface(x) => &x.super_graph,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TextInferenceTokensInLogitOutInterface {
    pub cache_key_input_link: SuperGraphLinkHash,
    pub token_context_input_link: SuperGraphLinkTensor,
    pub model_input_link: SuperGraphLinkModel,
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
        let token_context_input_link = super_graph_builder.new_tensor_link();

        let model_input_link = super_graph_builder.new_model_link();
        let raw_logit_output_link = super_graph_builder.new_tensor_link();

        let cache_key = super_graph_builder.new_hash_link();

        let processed_logit_output_link = if Some(1) == model_metadata.max_token_batch {
            // RNN compatibility mode

            let state_init_links = state_chain_ids
                .iter()
                .map(|id| (*id, super_graph_builder.new_tensor_link()))
                .collect::<Vec<(_, _)>>();

            // Cache
            let (post_cache_tokens_input, post_cache_state_init_links) = {
                let post_cache_state_init_links = state_chain_ids
                    .iter()
                    .map(|id| (*id, super_graph_builder.new_tensor_link()))
                    .collect::<Vec<(_, _)>>();
                let post_cache_tokens = super_graph_builder.new_tensor_link();
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
                );
                super_graph_builder.add_node(node.to_any());
                (post_cache_tokens, post_cache_state_init_links)
            };

            let loop_count_link = {
                let loop_count_link =
                    SuperGraphLinkTensor::new(super_graph_builder.get_next_link_id());

                let (mut milli_graph, input_map) = MilliOpGraph::new(&[post_cache_tokens_input]);
                let milli_op_graph_input = *input_map.get(&post_cache_tokens_input).unwrap();

                let shape_out = Shape::push_new(&mut milli_graph, milli_op_graph_input);
                let mut output_map = HashMap::new();
                output_map.insert(shape_out, loop_count_link);
                milli_graph.set_output_map(output_map);

                let node = SuperGraphNodeMilliOpGraph::new(milli_graph);
                super_graph_builder.add_node(node.into());
                loop_count_link
            };

            // Sub graph
            let (processed_logit_output_link, final_state_outputs) = {
                // State initialization
                {
                    let (mut milli_graph, _) = MilliOpGraph::new(&[]);
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
                        let input_tensor_tid = Constant::push_new(&mut milli_graph, input_tensor);
                        output_map.insert(input_tensor_tid, *link);
                        output_order.push(*link);
                    }
                    milli_graph.set_output_map_ordered(output_map, output_order);

                    let node = SuperGraphNodeMilliOpGraph::new(milli_graph);
                    super_graph_builder.add_node(node.into());
                }

                let mut sub_builder = SuperGraphBuilder::new();

                let sub_model_input_link = sub_builder.new_model_link();
                let sub_token_input = sub_builder.new_tensor_link();

                let state_input_links = state_chain_ids
                    .iter()
                    .map(|id| (*id, sub_builder.new_tensor_link()))
                    .collect::<HashMap<_, _>>();
                let state_output_links = state_chain_ids
                    .iter()
                    .map(|id| (*id, sub_builder.new_tensor_link()))
                    .collect::<HashMap<_, _>>();
                let final_state_output_links = state_chain_ids
                    .iter()
                    .map(|id| (*id, super_graph_builder.new_tensor_link()))
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

                    let (mut milli_graph, input_map) = MilliOpGraph::new(&[sub_token_input]);
                    let milli_op_graph_input = *input_map.get(&sub_token_input).unwrap();
                    let mut x =
                        Cast::push_new(&mut milli_graph, milli_op_graph_input, input_tensor_dtype);
                    let zero_tid = Constant::push_new(
                        &mut milli_graph,
                        NDArrayNumericTensor::from_vec_shape(vec![0i64], &vec![1]).unwrap(),
                    );
                    for _ in 0..input_tensor_rank {
                        x = Unsqueeze::push_new(&mut milli_graph, x, zero_tid);
                    }

                    let processed_input_link = sub_builder.new_tensor_link();
                    let mut output_map = HashMap::new();
                    output_map.insert(x, processed_input_link);
                    milli_graph.set_output_map(output_map);

                    let node = SuperGraphNodeMilliOpGraph::new(milli_graph);
                    sub_builder.add_node(node.into());
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

                let sub_logit_output = sub_builder.new_tensor_link();
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
                        sub_model_input_link,
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

                    let (mut milli_graph, input_map) = MilliOpGraph::new(&[sub_logit_output]);
                    let milli_op_graph_input = *input_map.get(&sub_logit_output).unwrap();
                    let mut x = milli_op_graph_input;
                    let zero_tid = Constant::push_new(
                        &mut milli_graph,
                        NDArrayNumericTensor::from_vec_shape(vec![0i64], &vec![1]).unwrap(),
                    );
                    for _ in 0..(output_tensor_rank - 1) {
                        x = Squeeze::push_new(&mut milli_graph, x, zero_tid);
                    }
                    x = Cast::push_new(&mut milli_graph, x, DType::F32);

                    let processed_logit_output_link = sub_builder.new_tensor_link();
                    let mut output_map = HashMap::new();
                    output_map.insert(x, processed_logit_output_link);
                    milli_graph.set_output_map(output_map);

                    let node = SuperGraphNodeMilliOpGraph::new(milli_graph);
                    sub_builder.add_node(node.into());
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

                let outer_logit_output_link = super_graph_builder.new_tensor_link();

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

                let sub_graph_inner = sub_builder.build_inner(&input_links, &output_links);

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
                    vec![SuperGraphLinkDouble::Model(
                        model_input_link,
                        sub_model_input_link,
                    )],
                    state_links,
                    vec![(post_cache_tokens_input, sub_token_input, 0)],
                    vec![(processed_logit_output_link, outer_logit_output_link, 0)],
                    final_state_outputs,
                );
                super_graph_builder.add_node(scan_node.into());

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
                );
                super_graph_builder.add_node(node.into());
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

                let (mut milli_graph, input_map) = MilliOpGraph::new(&[token_context_input_link]);
                let milli_op_graph_input = *input_map.get(&token_context_input_link).unwrap();
                let mut x =
                    Cast::push_new(&mut milli_graph, milli_op_graph_input, input_tensor_dtype);
                let zero_tid = Constant::push_new(
                    &mut milli_graph,
                    NDArrayNumericTensor::from_vec_shape(vec![0i64], &vec![1]).unwrap(),
                );
                for _ in 0..(input_tensor_rank - 1) {
                    x = Unsqueeze::push_new(&mut milli_graph, x, zero_tid);
                }

                let processed_input_link =
                    SuperGraphLinkTensor::new(super_graph_builder.get_next_link_id());
                let mut output_map = HashMap::new();
                output_map.insert(x, processed_input_link);
                milli_graph.set_output_map(output_map);

                let node = SuperGraphNodeMilliOpGraph::new(milli_graph);
                super_graph_builder.add_node(node.into());
                processed_input_link
            };

            let tensor_inputs = vec![(adjusted_token_context, tokens_input_name)];

            let tensor_outputs = vec![(logit_output_name.clone(), raw_logit_output_link)];

            super_graph_builder.add_node(
                SuperGraphNodeModelExecution::new(model_input_link, tensor_inputs, tensor_outputs)
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

                let (mut milli_graph, input_map) = MilliOpGraph::new(&[raw_logit_output_link]);
                let milli_op_graph_input = *input_map.get(&raw_logit_output_link).unwrap();
                let mut x = milli_op_graph_input;
                let zero_tid = Constant::push_new(
                    &mut milli_graph,
                    NDArrayNumericTensor::from_vec_shape(vec![0i64], &vec![1]).unwrap(),
                );
                for _ in 0..(output_tensor_rank - 2) {
                    x = Squeeze::push_new(&mut milli_graph, x, zero_tid);
                }
                x = Cast::push_new(&mut milli_graph, x, DType::F32);

                let processed_logit_output_link = super_graph_builder.new_tensor_link();
                let mut output_map = HashMap::new();
                output_map.insert(x, processed_logit_output_link);
                milli_graph.set_output_map(output_map);

                let node = SuperGraphNodeMilliOpGraph::new(milli_graph);
                super_graph_builder.add_node(node.into());
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

    pub fn run_string_in_string_out(
        &self,
        model: &Model,
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
            super_graph_data.models.insert(self.model_input_link, model);
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
                    .push((model, (*tensor_cache).clone()))
            }
            let mut context = SuperGraphContext {
                observer: &mut observer,
                eval_backend: backend,
                super_graph_tensor_cache: &mut super_graph_tensor_cache,
                caches: super_graph_caches,
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
