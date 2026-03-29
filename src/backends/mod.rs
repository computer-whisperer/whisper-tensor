use crate::DynRank;
use crate::graph::GlobalId;
use crate::migration::numeric_tensor::NumericTensor;
use std::collections::HashMap;

pub mod eval_backend;
pub mod ndarray_backend;
#[cfg(feature = "vulkan")]
pub mod vulkan_backend;

#[derive(Default, Clone)]
pub struct ModelLoadedTensorCache {
    pub tensors: HashMap<GlobalId, NumericTensor<DynRank>>,
}

impl ModelLoadedTensorCache {
    pub fn new() -> ModelLoadedTensorCache {
        ModelLoadedTensorCache::default()
    }
}
