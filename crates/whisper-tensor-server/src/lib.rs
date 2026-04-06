#[cfg(not(target_arch = "wasm32"))]
pub mod handler;
#[cfg(not(target_arch = "wasm32"))]
pub mod model_server;
#[cfg(not(target_arch = "wasm32"))]
pub mod scheduler;

use std::collections::HashMap;
use std::time::Duration;
use whisper_tensor::graph::GlobalId;
use whisper_tensor::interfaces::AnyInterface;
use whisper_tensor::loader::{ConfigField, ConfigValues};
use whisper_tensor::numeric_dtype::NumericDType;
use whisper_tensor::numeric_tensor::NumericTensor;
use whisper_tensor::pool::SystemPool;
use whisper_tensor::super_graph::links::SuperGraphLink;
use whisper_tensor::super_graph::{SuperGraph, SuperGraphHash};
use whisper_tensor::symbolic_graph::tensor_store::TensorStoreTensorId;
use whisper_tensor::tensor_rank::DynRank;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct LoadedModelId(pub u32);

impl core::fmt::Display for LoadedModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AbbreviatedTensorReportSettings {
    pub downsampled_size: u64,
    pub subscribed_tensors: Vec<Vec<GlobalId>>,
    pub do_all: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SuperGraphAudioInput {
    pub samples: NumericTensor<'static, DynRank, SystemPool>,
    pub sample_rate_hz: u32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SuperGraphRequest {
    pub attention_token: Option<u64>,
    pub super_graph: SuperGraph,
    pub use_cache: Option<u64>,
    pub eval_options: whisper_tensor::super_graph::SuperGraphEvalOptions,
    pub string_inputs: HashMap<SuperGraphLink, String>,
    pub hash_inputs: HashMap<SuperGraphLink, SuperGraphHash>,
    pub tensor_inputs: HashMap<SuperGraphLink, NumericTensor<'static, DynRank, SystemPool>>,
    pub audio_inputs: HashMap<SuperGraphLink, SuperGraphAudioInput>,
    pub model_inputs: HashMap<SuperGraphLink, LoadedModelId>,
    pub symbolic_graph_ids: Vec<LoadedModelId>,
    pub subscribed_tensors: Vec<Vec<GlobalId>>,
    pub do_node_execution_reports: bool,
    pub abbreviated_tensor_report_settings: Option<AbbreviatedTensorReportSettings>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SuperGraphObserverSettingsUpdate {
    pub attention_token: u64,
    pub subscribed_tensors: Vec<Vec<GlobalId>>,
    pub do_node_execution_reports: bool,
    pub abbreviated_tensor_report_settings: Option<AbbreviatedTensorReportSettings>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SuperGraphResponseData {
    pub string_outputs: HashMap<SuperGraphLink, String>,
    pub hash_outputs: HashMap<SuperGraphLink, SuperGraphHash>,
    pub tensor_outputs: HashMap<SuperGraphLink, NumericTensor<'static, DynRank, SystemPool>>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SuperGraphResponse {
    pub attention_token: Option<u64>,
    pub result: Result<SuperGraphResponseData, String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LoaderRegistryEntry {
    pub name: String,
    pub description: String,
    pub config_schema: Vec<ConfigField>,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct LoaderRegistryReport {
    pub loaders: Vec<LoaderRegistryEntry>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[allow(clippy::large_enum_variant)]
pub enum WebsocketClientServerMessage {
    Ping,
    RunLoader {
        loader_index: usize,
        config: ConfigValues,
    },
    UnloadModel(LoadedModelId),
    GetModelGraph(LoadedModelId),
    GetStoredTensor(LoadedModelId, TensorStoreTensorId),
    GetHFTokenizer(String),
    GetTokenizerFile(String),
    SuperGraphRequest(SuperGraphRequest),
    UpdateSuperGraphObserverSettings(SuperGraphObserverSettingsUpdate),
    CancelSuperGraphRequest(u64),
    CompileModel(LoadedModelId),
    GetCacheReport,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CurrentModelsReportEntry {
    pub model_id: LoadedModelId,
    pub model_name: String,
    pub num_ops: u64,
    pub model_compiled: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CurrentInterfacesReportEntry {
    pub model_ids: Vec<LoadedModelId>,
    pub interface_name: String,
    pub interface: AnyInterface,
}

#[derive(Copy, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ScaleParams {
    pub vmin: f32, // used by MinMax/Robust
    pub vmax: f32, // used by MinMax/Robust
    pub mean: f32, // used by Std (μ)
    pub std: f32,  // used by Std (σ)
}

type DigestResult = (Option<(Vec<u8>, ScaleParams)>, Option<Vec<bool>>);

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AbbreviatedTensorValue {
    pub value: Option<(Vec<u8>, ScaleParams)>,
    /// Mask of non-finite values (NaN/Inf). If present, true = non-finite.
    pub non_finite_mask: Option<Vec<bool>>,
    pub dtype: NumericDType,
    pub shape: Vec<u64>,
}

#[derive(Copy, Clone, Debug)]
pub enum ScaleMode {
    MinMax,
    Robust,
    Std,
}

impl AbbreviatedTensorValue {
    /// Compute a downsampled digest of the tensor as f32 bin-means.
    ///
    /// Divides the flattened tensor into `digest_len` equal-ish bins and
    /// returns the mean of each bin.
    fn get_digest(
        tensor: &whisper_tensor::numeric_tensor::NumericTensorView<'_, DynRank>,
        digest_len: u64,
    ) -> DigestResult {
        if !tensor.dtype().is_float() {
            return (None, None);
        }

        let num_elements = tensor.numel() as u64;
        let digest_len = digest_len.min(num_elements);
        if digest_len == 0 {
            return (None, None);
        }

        let res_vec: Vec<f32> = if num_elements > digest_len {
            // Bin-mean downsampling: divide into digest_len bins, average each
            (0..digest_len)
                .map(|i| {
                    let start = (i * num_elements / digest_len) as usize;
                    let end = ((i + 1) * num_elements / digest_len) as usize;
                    let count = (end - start).max(1);
                    let sum: f64 = (start..end).map(|j| tensor.read_element(j).to_f64()).sum();
                    (sum / count as f64) as f32
                })
                .collect()
        } else {
            (0..num_elements as usize)
                .map(|i| tensor.read_element(i).to_f64() as f32)
                .collect()
        };

        let has_non_finite = res_vec.iter().any(|v| !v.is_finite());
        let non_finite_mask = if has_non_finite {
            Some(res_vec.iter().map(|v| !v.is_finite()).collect())
        } else {
            None
        };
        (
            Some(Self::scale_and_quantize(&res_vec, ScaleMode::MinMax)),
            non_finite_mask,
        )
    }

    /// Scale + normalize + quantize to u8.
    /// Input: mean-only downsampled values (length K = S*S).
    /// Returns (quantized_u8, params_used_for_scaling)
    pub fn scale_and_quantize(y: &[f32], mode: ScaleMode) -> (Vec<u8>, ScaleParams) {
        // filter to finite values; if none, treat as zeros
        let mut finite: Vec<f32> = y.iter().copied().filter(|v| v.is_finite()).collect();
        if finite.is_empty() {
            return (
                vec![0u8; y.len()],
                ScaleParams {
                    vmin: 0.0,
                    vmax: 1.0,
                    mean: 0.0,
                    std: 1.0,
                },
            );
        }

        // compute scale params on y
        let params = match mode {
            ScaleMode::MinMax => {
                let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
                for v in &finite {
                    lo = lo.min(*v);
                    hi = hi.max(*v);
                }
                if lo == hi {
                    lo -= 1.0;
                    hi += 1.0;
                }
                ScaleParams {
                    vmin: lo,
                    vmax: hi,
                    mean: 0.0,
                    std: 1.0,
                }
            }
            ScaleMode::Robust => {
                finite
                    .sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let p01 = Self::quantile_sorted(&finite, 0.01);
                let p99 = Self::quantile_sorted(&finite, 0.99);
                let (vmin, vmax) = if p01 < p99 {
                    (p01, p99)
                } else {
                    (p01 - 1.0, p01 + 1.0)
                };
                ScaleParams {
                    vmin,
                    vmax,
                    mean: 0.0,
                    std: 1.0,
                }
            }
            ScaleMode::Std => {
                let n = finite.len() as f32;
                let mean = finite.iter().copied().sum::<f32>() / n;
                let var = finite
                    .iter()
                    .map(|v| {
                        let d = *v - mean;
                        d * d
                    })
                    .sum::<f32>()
                    / n; // population variance
                let std = var.sqrt().max(1e-12);
                ScaleParams {
                    vmin: mean - 3.0 * std,
                    vmax: mean + 3.0 * std,
                    mean,
                    std,
                }
            }
        };

        // normalize to [0,1] with clamping
        let denom = (params.vmax - params.vmin).max(1e-12);
        let mut out = Vec::with_capacity(y.len());
        for &v in y {
            let t = ((v - params.vmin) / denom).clamp(0.0, 1.0);
            out.push(Self::round_half_even_u8(t * 255.0));
        }
        (out, params)
    }

    /// Linear-interpolated quantile from a **sorted** slice.
    fn quantile_sorted(sorted: &[f32], q: f32) -> f32 {
        let n = sorted.len();
        if n == 0 {
            return 0.0;
        }
        let q = q.clamp(0.0, 1.0);
        let idx = q * (n as f32 - 1.0);
        let lo = idx.floor() as usize;
        let hi = idx.ceil() as usize;
        if lo == hi {
            sorted[lo]
        } else {
            let t = idx - lo as f32;
            sorted[lo] * (1.0 - t) + sorted[hi] * t
        }
    }

    /// Banker's rounding (half-to-even) → u8 with saturation.
    fn round_half_even_u8(x: f32) -> u8 {
        let y = x.clamp(0.0, 255.0);
        let f = y.floor();
        let r = y - f;
        let n = if r > 0.5 {
            f as i32 + 1
        } else if r < 0.5 {
            f as i32
        } else {
            let fi = f as i32;
            if fi % 2 == 0 { fi } else { fi + 1 }
        };
        n.clamp(0, 255) as u8
    }

    pub fn from_tensor_view(
        tensor: &whisper_tensor::numeric_tensor::NumericTensorView<'_, DynRank>,
        digest_len: u64,
    ) -> Self {
        let (value, non_finite_mask) = Self::get_digest(tensor, digest_len);
        Self {
            value,
            non_finite_mask,
            dtype: tensor.dtype(),
            shape: tensor.shape().to_vec(),
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SuperGraphExecutionReport {
    pub attention: Option<u64>,
    pub node_executions: Vec<(Vec<GlobalId>, String, Duration, Duration)>,
    pub loading_weight_reports: Vec<(Vec<GlobalId>, Option<String>, Duration)>,
    pub abbreviated_tensor_assignments: Vec<(Vec<GlobalId>, AbbreviatedTensorValue)>,
    pub tensor_assignments: Vec<(Vec<GlobalId>, NumericTensor<'static, DynRank, SystemPool>)>,
    pub progress_reports: Vec<(Vec<GlobalId>, i64, f64, f64)>,
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct CurrentModelsAndInterfacesReport {
    pub models: Vec<CurrentModelsReportEntry>,
    pub interfaces: Vec<CurrentInterfacesReportEntry>,
}

impl CurrentModelsAndInterfacesReport {
    pub fn new() -> Self {
        Self {
            models: Vec::new(),
            interfaces: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ServerConfigReport {}

// ---------------------------------------------------------------------------
// Build Inspector: cache report types
// ---------------------------------------------------------------------------

/// Per-lowered-model report for the Build Inspector.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LoweredModelReport {
    pub graph_id: GlobalId,
    pub info_inputs_hash: u64,
    pub num_groups: u64,
    pub total_atoms: u64,
    pub singleton_groups: u64,
    pub symbolic_groups: u64,
    /// Op name → count (String instead of &'static str for serialization).
    pub groups_by_op: HashMap<String, u64>,
    pub num_tensors: u64,
    pub num_inputs: u64,
    pub num_outputs: u64,
    /// Milli-op census: op_kind → (group_count, atom_count). Built from group_provenance.
    pub milli_op_census: HashMap<String, (u64, u64)>,
    /// Ops that could not be lowered.
    pub unsupported: Vec<(GlobalId, String)>,
    /// Human-readable detail for each unsupported op.
    pub unsupported_details: Vec<String>,
}

/// Per-compiled-plan report for the Build Inspector.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CompiledPlanReport {
    pub graph_id: GlobalId,
    pub info_inputs_hash: u64,
    pub num_outputs: u64,
    pub plan_summary: whisper_tensor::compiler::attempts::v14::report::PlanSummary,
}

/// One cache slot (keyed by the use_cache u64 in SuperGraphRequest).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CacheReportEntry {
    pub cache_key: u64,
    pub num_rnn_entries: u64,
    pub num_tensor_entries: u64,
    pub num_tensor_pack_entries: u64,
    pub lowered_models: Vec<LoweredModelReport>,
    pub compiled_plans: Vec<CompiledPlanReport>,
}

/// Full cache report returned by the scheduler.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct CacheReport {
    pub entries: Vec<CacheReportEntry>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum WebsocketServerClientMessage {
    Pong,
    ServerConfigReport(ServerConfigReport),
    LoaderRegistryReport(LoaderRegistryReport),
    ModelLoadReturn(Result<(), String>),
    CurrentModelsReport(CurrentModelsAndInterfacesReport),
    ModelGraphReturn(Result<(LoadedModelId, Vec<u8>), String>),
    TensorStoreReturn(
        LoadedModelId,
        TensorStoreTensorId,
        Result<NumericTensor<'static, DynRank, SystemPool>, String>,
    ),
    HFTokenizerReturn(String, Result<Vec<u8>, String>),
    TokenizerFileReturn(String, Result<Vec<u8>, String>),
    SuperGraphResponse(SuperGraphResponse),
    SuperGraphExecutionReport(SuperGraphExecutionReport),
    CacheReportReturn(CacheReport),
}
