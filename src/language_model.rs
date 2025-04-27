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
    
    pub fn run<S: Sampler>(&mut self, input_tokens: NumericTensor, sampler: &mut S) -> Result<NumericTensor, RuntimeError> {
        let input_tokens = if let Some(token_context) = &self.token_context {
            NumericTensor::concat(&[token_context, &input_tokens], input_tokens.shape().len()-1)?
        } else {
            input_tokens
        };
        
        let input_tensors = HashMap::from([(self.input_name.clone(), input_tokens)]);
        
        let output_tensors = self.model.run(input_tensors)?;
        
        let output_tensor = output_tensors.get(&self.output_name).unwrap();
        
        let shape = output_tensor.shape();
        let mut output_slice = Vec::new();
        for i in 0..shape.len()-1 {
            output_slice.push(shape[i]-1..shape[i]);
        }
        output_slice.push(0..shape[shape.len()-1]);
        let sliced_output_tensor = output_tensor.slice(&output_slice)?;
        let output_tensor_sampled = sampler.sample(&sliced_output_tensor)?;
        
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