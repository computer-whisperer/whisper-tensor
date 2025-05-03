use crate::dtype::DType;
use crate::eval_backend::EvalBackend;
use crate::ndarray_backend::{NDArrayNumericTensor, NDArrayNumericTensorError};
use crate::numeric_tensor::{NumericTensor, NumericTensorError};

#[derive(Debug, thiserror::Error)]
pub enum SamplerError {
    #[cfg(feature = "llm-samplers")]
    #[error(transparent)]
    LLMSamplersError(anyhow::Error),
    #[error(transparent)]
    NumericTensorError(#[from] NumericTensorError),
    #[cfg(feature = "candle")]
    #[error(transparent)]
    Candle(#[from] candle_core::Error),
}

pub trait Sampler {
    fn sample(&mut self, x: &NumericTensor) -> Result<NumericTensor, SamplerError>;
}

pub struct GreedySampler{}

impl Sampler for GreedySampler {
    fn sample(&mut self, x: &NumericTensor) -> Result<NumericTensor, SamplerError> {
        let dim = x.shape().len() - 1;
        Ok(match x {
            NumericTensor::Candle(x) => {
                NumericTensor::Candle(x.argmax(dim)?)
            }
            _ => {
                let x = NDArrayNumericTensor::try_from(x)?;
                let x = x.argmax(dim).1;
                x.into()
            }
        })
    }
}

#[cfg(feature = "llm-samplers")]
pub struct LLMSamplersBundle {
    pub chain: llm_samplers::prelude::SamplerChain,
    pub res: llm_samplers::prelude::SimpleSamplerResources
}

#[cfg(feature = "llm-samplers")]
impl Sampler for LLMSamplersBundle {
    
    fn sample(&mut self, x: &NumericTensor) -> Result<NumericTensor, SamplerError> {
        use llm_samplers::prelude::Sampler;
        
        let x_shape = x.shape();
        let x = x.reshape(vec![x_shape[x_shape.len()-1]], &EvalBackend::NDArray)?;
        let v: Vec<f32> = x.cast(DType::F32, &EvalBackend::NDArray)?.try_into()?;
        let mut logits = llm_samplers::prelude::Logits::try_from(v).map_err(|x| {SamplerError::LLMSamplersError(x.into())})?;
        let out = self.chain.sample_token(&mut self.res, &mut logits).map_err(|x| {SamplerError::LLMSamplersError(x.into())})?;
        let out = out.unwrap();
        let mut output_shape = x_shape.clone();
        output_shape.remove(output_shape.len()-1);
        let out_tensor = NumericTensor::from_vec_shape(vec![out], output_shape)?;
        Ok(out_tensor)
    }
}


