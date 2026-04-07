use crate::graph::GlobalId;
use crate::numeric_tensor::NumericTensor;
use crate::pool::ArcTrackedPool;
use crate::super_graph::lowered_eval::CachedLoweredModel;
use crate::tensor_rank::DynRank;
use std::collections::HashMap;

/// Cached tensor type used by the three per-request tensor caches
/// (`rnn_cache`, `tensor_cache`, `tensor_pack_cache`).
///
/// Backed by [`ArcTrackedPool`] so every cached byte is counted against a
/// shared atomic counter. The `'static` lifetime parameter is load-bearing:
/// `ArcTrackedBuffer` owns its own `Arc<TrackedPool>` clone, so the tensor
/// does not tie its lifetime to any `&pool` reference and can live in the
/// long-lived `HashMap` fields below.
pub type CachedTensor = NumericTensor<'static, DynRank, ArcTrackedPool>;
type RNNCache = HashMap<Vec<u32>, HashMap<String, CachedTensor>>;
type TensorPackCache = HashMap<String, CachedTensor>;

pub struct SuperGraphCache {
    /// Pool backing all tensor-cache allocations on this slot. The
    /// scheduler constructs one aggregate `ArcTrackedPool` at startup and
    /// hands clones to each cache; all clones share a single byte counter
    /// that the stats sampler reads via `bytes_in_use()`.
    ///
    /// Only the three tensor caches (`rnn_cache`, `tensor_cache`,
    /// `tensor_pack_cache`) currently allocate from this pool. The
    /// `lowered_model_cache` still carries `SystemPool`-backed baked
    /// constants — migrating that path requires threading the cache pool
    /// through `lower()` and is deferred.
    pub pool: ArcTrackedPool,
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
    /// Create a new cache slot backed by the given aggregate pool.
    pub fn new(pool: ArcTrackedPool) -> Self {
        Self {
            pool,
            rnn_cache: HashMap::new(),
            tensor_cache: HashMap::new(),
            tensor_pack_cache: HashMap::new(),
            lowered_model_cache: HashMap::new(),
            #[cfg(feature = "cranelift")]
            compiled_plan_cache: HashMap::new(),
        }
    }
}
