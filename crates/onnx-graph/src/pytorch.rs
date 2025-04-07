use std::sync::Arc;
use crate::{operators, Error};
use crate::operators::LayerNormalization;
use crate::tensor::Tensor;
use crate::weights::WeightManager;

pub fn linear(weight_manager: &impl WeightManager, input: Arc<dyn Tensor>) -> Result<Arc<dyn Tensor>, Error> {
    let bias = weight_manager.get_tensor("bias").ok();
    let mat_out = operators::MatMul::new(
        weight_manager.get_prefix().map(|x| x.to_string()),
        weight_manager.get_tensor("weight")?,
        input
    )?;
    
    if let Some(bias) = bias {
        Ok(operators::Add::new(Some(format!("{}.bias", weight_manager.get_prefix().unwrap())), mat_out, bias)?)
    } else {
        Ok(mat_out)
    }
}

pub fn layer_norm(weight_manager: &impl WeightManager, input: Arc<dyn Tensor>, epsilon: f32) -> Result<Arc<LayerNormalization>, Error> {
    LayerNormalization::new(
        weight_manager.get_prefix().map(|x| x.to_string()),
        input,
        weight_manager.get_tensor("weight")?,
        weight_manager.get_tensor("bias").ok(),
        -1,
        epsilon,
        1
    )
}
