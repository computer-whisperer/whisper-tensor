use std::collections::HashMap;
use crate::numeric_tensor::NumericTensor;
use crate::tokenizer::Tokenizer;
use crate::{RuntimeError, RuntimeModel};
use crate::sampler::Sampler;


pub struct LanguageModelManager {
    model: RuntimeModel,
    token_context: Option<NumericTensor>,
    input_name: String,
    output_name: String,
}

impl LanguageModelManager {
    pub fn new(model: RuntimeModel, input_name: &str, output_name: &str) -> Self {
        Self { 
            model,
            token_context: None,
            input_name: input_name.to_string(),
            output_name: output_name.to_string()
        }
    }
    
    pub fn run<S: Sampler>(&self, input_tokens: NumericTensor, sampler: &mut S) -> Result<NumericTensor, RuntimeError> {
        let input_tokens = if let Some(token_context) = &self.token_context {
            NumericTensor::concat(&[token_context, &input_tokens], input_tokens.shape().len()-1)?
        } else {
            input_tokens
        };
        
        let input_tensors = HashMap::from([(self.input_name.clone(), input_tokens)]);
        
        let output_tensors = self.model.run(input_tensors)?;
        
        let output_tensor = output_tensors.get(&self.output_name).unwrap();
        let output_tensor_sampled = sampler.sample(output_tensor)?;
        
        Ok(output_tensor_sampled)
    }
    
    pub fn push_context(&mut self, context: NumericTensor) {
        self.token_context = if let Some(token_context) = &self.token_context {
            Some(NumericTensor::concat(&[token_context, &context], context.shape().len()-1).unwrap())
        } else {
            Some(context)
        }
    }
}