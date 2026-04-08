use crate::graph::GlobalId;
use crate::numeric_tensor::NumericTensor;
use crate::pool::ArcTrackedPool;
use crate::super_graph::lowered_eval::CachedLoweredModel;
use crate::symbolic_graph::tensor_store::TensorStoreTensorId;
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

/// Per-slot cache of model weights resolved from a `TensorStore`, keyed by
/// `TensorStoreTensorId`. Populated lazily by `execute_lowered` /
/// `execute_compiled` on the first iteration that touches a given weight,
/// then re-used as zero-copy views on subsequent iterations.
///
/// Bundles a clone of the slot's `ArcTrackedPool` so the eval functions can
/// load missing entries without the caller having to pass the pool
/// separately. All entries allocate from this pool, so cache bytes flow into
/// the same atomic counter the stats sampler reads.
///
/// Note: keyed only by `TensorStoreTensorId`. If a single cache slot is
/// reused with two different `TensorStore`s that have overlapping IDs, the
/// stale entries will leak across — same convention as `lowered_model_cache`,
/// which doesn't catch that either. One slot is meant to be one model.
pub struct LoadedTensorCache {
    pool: ArcTrackedPool,
    entries: HashMap<TensorStoreTensorId, CachedTensor>,
}

impl LoadedTensorCache {
    pub fn new(pool: ArcTrackedPool) -> Self {
        Self {
            pool,
            entries: HashMap::new(),
        }
    }

    pub fn get(&self, id: &TensorStoreTensorId) -> Option<&CachedTensor> {
        self.entries.get(id)
    }

    pub fn contains_key(&self, id: &TensorStoreTensorId) -> bool {
        self.entries.contains_key(id)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Idempotently materialize the tensor into the cache. Returns `true` if
    /// the cache contains the entry afterwards (cache hit, freshly loaded, or
    /// already present); `false` if the load failed (e.g. unsupported format,
    /// allocation error).
    pub fn ensure_loaded(
        &mut self,
        id: TensorStoreTensorId,
        store: &crate::symbolic_graph::tensor_store::TensorStore,
    ) -> bool {
        if self.entries.contains_key(&id) {
            return true;
        }
        let Some(stored) = store.get_tensor(id) else {
            return false;
        };
        let Some(loaded) = stored.load_into_cache_pool(&self.pool) else {
            return false;
        };
        self.entries.insert(id, loaded);
        true
    }
}

pub struct SuperGraphCache {
    /// Pool backing all tensor-cache allocations on this slot. The
    /// scheduler constructs one aggregate `ArcTrackedPool` at startup and
    /// hands clones to each cache; all clones share a single byte counter
    /// that the stats sampler reads via `bytes_in_use()`.
    ///
    /// The four tensor caches (`rnn_cache`, `tensor_cache`,
    /// `tensor_pack_cache`, `loaded_tensor_cache`) all allocate from this
    /// pool. The `lowered_model_cache` still carries `SystemPool`-backed
    /// baked constants — migrating that path requires threading the cache
    /// pool through `lower()` and is deferred.
    pub pool: ArcTrackedPool,
    pub rnn_cache: HashMap<u64, RNNCache>,
    pub tensor_cache: HashMap<u64, CachedTensor>,
    pub tensor_pack_cache: HashMap<u64, TensorPackCache>,
    /// Resolved model weights, keyed by `TensorStoreTensorId`. Avoids
    /// re-streaming weight bytes from disk on every supergraph execution.
    /// Populated by `execute_lowered` / `execute_compiled` on miss. Owns its
    /// own `ArcTrackedPool` clone (cheap — internal `Arc<TrackedPool>`) so
    /// the eval path doesn't need a separate pool argument.
    pub loaded_tensor_cache: LoadedTensorCache,
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
        let loaded_tensor_cache = LoadedTensorCache::new(pool.clone());
        Self {
            pool,
            rnn_cache: HashMap::new(),
            tensor_cache: HashMap::new(),
            tensor_pack_cache: HashMap::new(),
            loaded_tensor_cache,
            lowered_model_cache: HashMap::new(),
            #[cfg(feature = "cranelift")]
            compiled_plan_cache: HashMap::new(),
        }
    }
}
