use std::collections::HashMap;
use crate::DynRank;
use crate::numeric_tensor::NumericTensor;

#[derive(Clone, Debug)]
pub struct SuperGraphCache {
    pub rnn_cache: HashMap<u64, HashMap<Vec<u32>, HashMap<String, NumericTensor<DynRank>>>>
}

impl SuperGraphCache {
    pub fn new() -> Self {
        Self {
            rnn_cache: HashMap::new()
        }
    }
}
