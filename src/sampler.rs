use crate::backends::eval_backend::EvalBackend;
use crate::dtype::DType;
use crate::numeric_tensor::{NumericTensor, NumericTensorError};
use crate::tensor_rank::DynRank;
use typenum::P1;

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
    fn sample(
        &mut self,
        x: &NumericTensor<DynRank>,
        backend: &mut EvalBackend,
    ) -> Result<NumericTensor<DynRank>, SamplerError>;
}

#[cfg(feature = "llm-samplers")]
pub struct LLMSamplersBundle {
    pub chain: llm_samplers::prelude::SamplerChain,
    pub res: llm_samplers::prelude::SimpleSamplerResources,
}

#[cfg(feature = "llm-samplers")]
impl Sampler for LLMSamplersBundle {
    fn sample(
        &mut self,
        x: &NumericTensor<DynRank>,
        backend: &mut EvalBackend,
    ) -> Result<NumericTensor<DynRank>, SamplerError> {
        use llm_samplers::prelude::Sampler;

        let x_shape = x.shape();
        let x = x.reshape(vec![x_shape[x_shape.len() - 1]], backend)?;
        let v: Vec<f32> = x
            .cast(DType::F32, backend)?
            .try_to_rank::<P1>()?
            .try_into()?;
        let mut logits = llm_samplers::prelude::Logits::try_from(v)
            .map_err(|x| SamplerError::LLMSamplersError(x.into()))?;
        let out = self
            .chain
            .sample_token(&mut self.res, &mut logits)
            .map_err(SamplerError::LLMSamplersError)?;
        let out = out.unwrap();
        let mut output_shape = x_shape.clone();
        output_shape.remove(output_shape.len() - 1);
        let out_tensor = NumericTensor::from_vec_shape(
            vec![out],
            output_shape.iter().map(|x| *x as usize).collect(),
        )?;
        Ok(out_tensor)
    }
}
