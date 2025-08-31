use crate::DynRank;
use crate::numeric_tensor::NumericTensor;
use crate::symbolic_graph::SymbolicGraphTensorId;
use std::collections::HashMap;

#[cfg(feature = "candle")]
pub mod candle_backend;
#[cfg(feature = "onnx-reference")]
pub mod onnx_reference_backend;
#[cfg(feature = "tch")]
pub mod tch_backend;
#[cfg(feature = "vulkan")]
pub mod vulkan_backend;

pub mod eval_backend;
pub mod ndarray_backend;

#[derive(Default, Clone)]
pub struct ModelLoadedTensorCache {
    pub tensors: HashMap<SymbolicGraphTensorId, NumericTensor<DynRank>>,
}

impl ModelLoadedTensorCache {
    pub fn new() -> ModelLoadedTensorCache {
        ModelLoadedTensorCache::default()
    }
}
