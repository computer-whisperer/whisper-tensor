use std::collections::HashMap;
use std::io;
use std::sync::Arc;
use rwkv_tokenizer::WorldTokenizer;
use onnx_graph::{ModelInputType, ModelOutputType, TokenizerInfo};
use crate::numeric_tensor::NumericTensor;
use crate::tokenizer::Tokenizer;
use crate::{RuntimeError, RuntimeModel};
use crate::eval_backend::EvalBackend;
use crate::sampler::Sampler;
use crate::tensor_rank::DynRank;

#[derive(Debug, thiserror::Error)]
pub enum LanguageModelManagerError {
    #[error(transparent)]
    TokenizersError(#[from] tokenizers::Error),
    #[error(transparent)]
    RWKVWorldTokenizerError(#[from] io::Error)
}

#[derive(Debug, Clone)]
pub struct LangaugeModelIntermediateValues {
    values: HashMap<usize, NumericTensor<DynRank>>
}

pub struct LanguageModelManager {
    model: RuntimeModel,
    tokenizer: Option<Arc<dyn Tokenizer>>
}

impl LanguageModelManager {
    pub fn new(model: RuntimeModel) -> Result<Self, LanguageModelManagerError> {
        let mut tokenizer: Option<Arc<dyn Tokenizer>> = None;
        if let Some(meta) = &model.model_metadata {
            if let Some(tokenizer_info) = meta.tokenizer_infos.get(0) {
                match tokenizer_info {
                    TokenizerInfo::HFTokenizer(name) => {
                        let x = Arc::new(tokenizers::Tokenizer::from_pretrained(name, None)?);
                        tokenizer = Some(x);
                    }
                    TokenizerInfo::RWKVWorld => {
                        let x = Arc::new(WorldTokenizer::new(None)?);
                        tokenizer = Some(x);
                    }
                }
            }
        }
        Ok(Self { 
            model,
            tokenizer
        })
    }
    
    pub fn run<S: Sampler>(&mut self, input_tokens: NumericTensor<DynRank>, input_intermediate_values: Option<LangaugeModelIntermediateValues>, sampler: &mut S) -> Result<(NumericTensor<DynRank>, LangaugeModelIntermediateValues), RuntimeError> {
        let (output_tensor, output_intermediate_values) = self.forward(input_tokens, input_intermediate_values)?;
        let shape = output_tensor.shape();
        let mut output_slice = Vec::new();
        for i in 0..shape.len()-1 {
            output_slice.push(shape[i] -1..shape[i]);
        }
        output_slice.push(0..shape[shape.len()-1]);
        let sliced_output_tensor = output_tensor.slice(&output_slice, &EvalBackend::NDArray)?;
        let output_tensor_sampled = sampler.sample(&sliced_output_tensor)?;
        Ok((output_tensor_sampled, output_intermediate_values))
    } 
    
    pub fn forward(&mut self, input_tokens: NumericTensor<DynRank>, intermediate_values: Option<LangaugeModelIntermediateValues>) -> Result<(NumericTensor<DynRank>, LangaugeModelIntermediateValues), RuntimeError> {
        let input_shape = input_tokens.shape();
        let input_sequence_len = input_shape[1];
        let mut input_tokens_processed = 0;
        let mut output_chunks = vec![];
        
        let max_sequence_len = self.model.model_metadata.clone().map(|x| x.max_token_batch).flatten().unwrap_or(input_sequence_len as usize);
        let mut intermediate_values = intermediate_values.clone();
        
        let input_tensor_infos = self.model.get_input_tensor_info()?;
        
        while input_tokens_processed < input_sequence_len {
            let chunk_size = max_sequence_len.min((input_sequence_len - input_tokens_processed) as usize) as u64;
            let mut slice_indices: Vec<_> = input_shape.iter().map(|x| 0..*x).collect();
            slice_indices[1] = (input_tokens_processed..(input_tokens_processed + chunk_size));
            let input_tokens_chunk = input_tokens.slice(slice_indices.as_slice(), &EvalBackend::NDArray)?;

            let mut input_tensors = HashMap::new();
            for (name, meta) in &self.model.model_inputs {
                if let Some(meta) = meta {
                    match meta.model_input_type {
                        ModelInputType::TokenID(_) => {
                            let info = input_tensor_infos.get(name);
                            let input_chunk = if let Some((dtype, _)) = info {
                                input_tokens_chunk.cast(*dtype, &EvalBackend::NDArray)?
                            } else {
                                input_tokens_chunk.clone()
                            };
                            input_tensors.insert(name.clone(), input_chunk);
                        }
                        ModelInputType::PreviousInternal(x) => {
                            if let Some(intermediate_values) = &intermediate_values {
                                if let Some(tensor) = intermediate_values.values.get(&x) {
                                    input_tensors.insert(name.clone(), tensor.clone());
                                }
                            }
                        }
                    }
                }
            }

            let output_tensors = self.model.run(input_tensors)?;

            let mut output_tensor = None;
            let mut output_intermediate_values = HashMap::new();
            for (name, meta) in &self.model.model_outputs {
                if let (Some(meta), Some(tensor)) = (meta, output_tensors.get(name)) {
                    match meta.model_output_type {
                        ModelOutputType::TokenID(_) => {
                            output_tensor = Some(tensor.clone());
                        }
                        ModelOutputType::NextInternal(i) => {
                            output_intermediate_values.insert(i, tensor.clone());
                        }
                    }
                }
            }
            input_tokens_processed += chunk_size;
            output_chunks.push(output_tensor.unwrap());
            intermediate_values = Some(LangaugeModelIntermediateValues { values: output_intermediate_values });
        }
        let tmp: Vec<_> = output_chunks.iter().collect(); 
        let output_tensor = NumericTensor::concat(tmp.as_slice(), 1, &EvalBackend::NDArray)?;
        Ok((output_tensor, intermediate_values.unwrap()))
    }
    
    pub fn get_tokenizer(&self) -> Option<Arc<dyn Tokenizer>> {
        self.tokenizer.clone()
    }
}