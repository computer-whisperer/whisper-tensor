use crate::DynRank;
use crate::backends::ModelLoadedTensorCache;
use crate::model::Model;
use crate::numeric_tensor::NumericTensor;
use std::collections::HashMap;

type RNNCache = HashMap<Vec<u32>, HashMap<String, NumericTensor<DynRank>>>;

#[derive(Clone, Debug, Default)]
pub struct SuperGraphCache {
    pub rnn_cache: HashMap<u64, RNNCache>,
}

impl SuperGraphCache {
    pub fn new() -> Self {
        Self {
            rnn_cache: HashMap::new(),
        }
    }
}

#[derive(Default)]
pub struct SuperGraphTensorCache<'model> {
    pub caches: Vec<(&'model Model, ModelLoadedTensorCache)>,
}

impl<'model> SuperGraphTensorCache<'model> {
    pub fn new() -> Self {
        Self { caches: Vec::new() }
    }
}
