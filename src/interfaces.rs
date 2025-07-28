use std::collections::HashMap;
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use whisper_tensor_import::onnx_graph::{InputMetadata, ModelInputType, ModelMetadata, ModelOutputType, OutputMetadata, TokenizerInfo};
use crate::backends::eval_backend::EvalBackend;
use crate::dtype::DType;
use crate::model::Model;
use crate::numeric_tensor::NumericTensor;
use crate::super_graph::links::{SuperGraphLink, SuperGraphLinkModel, SuperGraphLinkTensor};
use crate::super_graph::{SuperGraph, SuperGraphBuilder, SuperGraphError};
use crate::super_graph::data::SuperGraphData;
use crate::super_graph::nodes::{SuperGraphNode, SuperGraphNodeModelExecution};
use crate::tokenizer::{AnyTokenizer, Tokenizer};

#[derive(Clone, Serialize, Deserialize)]
pub struct TextInferenceTokensInLogitOutInterface {
    token_context_input_link: SuperGraphLinkTensor,
    model_input_link: SuperGraphLinkModel,
    logit_output_link: SuperGraphLinkTensor,
    super_graph: SuperGraph,
    tokenizer: TokenizerInfo
}


impl TextInferenceTokensInLogitOutInterface {
    pub fn try_from_onnx_metadata(
        model_metadata: &ModelMetadata,
        model_inputs: &HashMap<String, Option<InputMetadata>>,
        model_outputs: &HashMap<String, Option<OutputMetadata>>) -> Result<Self, ()> {

        let tokenizer = model_metadata.tokenizer_infos.get(0).ok_or(())?.clone();

        let tokens_input_name = {
            let mut tokens_input_name = None;
            for (name, input) in model_inputs {
                if let Some(input) = input {
                    if let ModelInputType::TokenID(_) = input.model_input_type {
                        tokens_input_name = Some(name.clone());
                    }
                }
            }
            tokens_input_name.ok_or(())?
        };

        let logit_output_name = {
            let mut logit_output_name = None;
            for (name, output) in model_outputs {
                if let Some(output) = output {
                    if let ModelOutputType::TokenID(_) = output.model_output_type {
                        logit_output_name = Some(name.clone());
                    }
                }
            }
            logit_output_name.ok_or(())?
        };

        // Build super graph
        let mut super_graph_builder = SuperGraphBuilder::new();
        let model_input_link = SuperGraphLinkModel::new(super_graph_builder.get_next_link_id());
        let token_context_input_link = SuperGraphLinkTensor::new(super_graph_builder.get_next_link_id());
        let logit_output_link = SuperGraphLinkTensor::new(super_graph_builder.get_next_link_id());

        let tensor_inputs = {
            let mut tensor_inputs = HashMap::new();
            tensor_inputs.insert(token_context_input_link.clone(), tokens_input_name);
            tensor_inputs
        };

        let tensor_outputs = {
            let mut tensor_outputs = HashMap::new();
            tensor_outputs.insert(logit_output_name, logit_output_link.clone());
            tensor_outputs
        };

        super_graph_builder.add_node(SuperGraphNodeModelExecution::new(model_input_link.clone(), tensor_inputs, tensor_outputs).to_any());

        let super_graph_inputs = vec![model_input_link.to_any(), token_context_input_link.to_any()];
        let super_graph_outputs = vec![logit_output_link.to_any()];
        let super_graph = super_graph_builder.build(super_graph_inputs.as_slice(), super_graph_outputs.as_slice());

        Ok(Self {
            tokenizer,
            model_input_link,
            token_context_input_link,
            logit_output_link,
            super_graph,
            })
    }

    pub fn run_string_in_string_out(&self, model: Arc<Model>, text_in: String, tokenizer_cache: &mut HashMap<TokenizerInfo, Arc<AnyTokenizer>>, backend: &mut EvalBackend) -> Result<String, SuperGraphError> {
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
        let tokens_tensor = NumericTensor::from_vec(tokens.clone())
            .to_dyn_rank()
            .unsqueeze(0)?
            .unsqueeze(0)?
            .cast(DType::I64, backend)?;

        let super_graph_data = {
            let mut super_graph_data = SuperGraphData::new();
            super_graph_data.models.insert(self.model_input_link.clone(), model.clone());
            super_graph_data.tensors.insert(self.token_context_input_link.clone(), tokens_tensor);
            super_graph_data
        };

        let super_graph_output = self.super_graph.run(super_graph_data, backend)?;
        let logits = super_graph_output.tensors.get(&self.logit_output_link).unwrap();
        let logits_shape = logits.shape();
        let logits = logits.slice(&[0..1, 0..1, logits_shape[2]-1..logits_shape[2], 0..logits_shape[3]], backend)?;
        let logits = logits.squeeze(0)?.squeeze(0)?.squeeze(0)?;
        let token_id = logits.argmax(0, false, false, backend)?;

        let token_id: u32 = token_id.first_element().into();
        let token_str = tokenizer.decode(&[token_id])?;
        Ok(token_str)
    }
}
