use crate::numeric_tensor::NumericTensor;
use crate::pool::SystemPool;
use crate::tensor_rank::DynRank;
use std::collections::HashMap;

type CachedTensor = NumericTensor<'static, DynRank, SystemPool>;
type RNNCache = HashMap<Vec<u32>, HashMap<String, CachedTensor>>;
type TensorPackCache = HashMap<String, CachedTensor>;

#[derive(Default)]
pub struct SuperGraphCache {
    pub rnn_cache: HashMap<u64, RNNCache>,
    pub tensor_cache: HashMap<u64, CachedTensor>,
    pub tensor_pack_cache: HashMap<u64, TensorPackCache>,
}

impl SuperGraphCache {
    pub fn new() -> Self {
        Self {
            rnn_cache: HashMap::new(),
            tensor_cache: HashMap::new(),
            tensor_pack_cache: HashMap::new(),
        }
    }
}
