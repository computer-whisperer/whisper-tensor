use crate::graph::GlobalId;
use crate::numeric_tensor::NumericTensor;
use crate::pool::SystemPool;
use crate::super_graph::lowered_eval::CachedLoweredModel;
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
    /// Cached lowered NanoGraphs keyed by the symbolic graph's GlobalId.
    pub lowered_model_cache: HashMap<GlobalId, CachedLoweredModel>,
    /// Cached JIT-compiled execution plans keyed by the symbolic graph's GlobalId.
    #[cfg(feature = "cranelift")]
    pub compiled_plan_cache:
        HashMap<GlobalId, crate::super_graph::compiled_eval::CachedCompiledPlan>,
}

impl SuperGraphCache {
    pub fn new() -> Self {
        Self {
            rnn_cache: HashMap::new(),
            tensor_cache: HashMap::new(),
            tensor_pack_cache: HashMap::new(),
            lowered_model_cache: HashMap::new(),
            #[cfg(feature = "cranelift")]
            compiled_plan_cache: HashMap::new(),
        }
    }
}
