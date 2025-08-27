use std::collections::HashMap;
use std::fmt::Display;
use std::time::Duration;
use whisper_tensor::DynRank;
use whisper_tensor::backends::eval_backend::EvalBackend;
use whisper_tensor::backends::ndarray_backend::NDArrayNumericTensor;
use whisper_tensor::dtype::DType;
use whisper_tensor::interfaces::AnyInterface;
use whisper_tensor::numeric_tensor::NumericTensor;
use whisper_tensor::super_graph::links::{
    SuperGraphLinkHash, SuperGraphLinkModel, SuperGraphLinkString, SuperGraphLinkTensor,
};
use whisper_tensor::super_graph::{
    SuperGraph, SuperGraphHash, SuperGraphNodePath, SuperGraphTensorPath,
};
use whisper_tensor::symbolic_graph::tensor_store::TensorStoreTensorId;
use whisper_tensor_import::ModelTypeHint;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct LoadedModelId(pub u32);

impl core::fmt::Display for LoadedModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AbbreviatedTensorReportSettings {
    pub downsampled_size: u64,
    pub subscribed_tensors: Vec<SuperGraphTensorPath>,
    pub do_all: bool,
}

#[derive(Copy, Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum SuperGraphRequestBackendMode {
    NDArray,
    Vulkan,
}

impl Display for SuperGraphRequestBackendMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SuperGraphRequestBackendMode::NDArray => write!(f, "NDArray"),
            SuperGraphRequestBackendMode::Vulkan => write!(f, "Vulkan"),
        }
    }
}

impl Default for SuperGraphRequestBackendMode {
    fn default() -> Self {
        SuperGraphRequestBackendMode::NDArray
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SuperGraphRequest {
    pub attention_token: Option<u64>,
    pub super_graph: SuperGraph,
    pub use_cache: Option<u64>,
    pub backend_mode: SuperGraphRequestBackendMode,
    pub string_inputs: HashMap<SuperGraphLinkString, String>,
    pub hash_inputs: HashMap<SuperGraphLinkHash, SuperGraphHash>,
    pub tensor_inputs: HashMap<SuperGraphLinkTensor, NDArrayNumericTensor<DynRank>>,
    pub model_inputs: HashMap<SuperGraphLinkModel, LoadedModelId>,
    pub subscribed_tensors: Vec<SuperGraphTensorPath>,
    pub do_node_execution_reports: bool,
    pub abbreviated_tensor_report_settings: Option<AbbreviatedTensorReportSettings>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SuperGraphResponseData {
    pub string_outputs: HashMap<SuperGraphLinkString, String>,
    pub hash_outputs: HashMap<SuperGraphLinkHash, SuperGraphHash>,
    pub tensor_outputs: HashMap<SuperGraphLinkTensor, NDArrayNumericTensor<DynRank>>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SuperGraphResponse {
    pub attention_token: Option<u64>,
    pub result: Result<SuperGraphResponseData, String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[allow(clippy::large_enum_variant)]
pub enum WebsocketClientServerMessage {
    Ping,
    LoadModel {
        model_path: String,
        model_type_hint: Option<ModelTypeHint>,
    },
    UnloadModel(LoadedModelId),
    GetModelGraph(LoadedModelId),
    GetStoredTensor(LoadedModelId, TensorStoreTensorId),
    GetHFTokenizer(String),
    SuperGraphRequest(SuperGraphRequest),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CurrentModelsReportEntry {
    pub model_id: LoadedModelId,
    pub model_name: String,
    pub num_ops: u64,
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

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AbbreviatedTensorValue {
    pub value: Option<(Vec<u8>, ScaleParams)>,
    pub dtype: DType,
    pub shape: Vec<u64>,
}

#[derive(Copy, Clone, Debug)]
pub enum ScaleMode {
    MinMax,
    Robust,
    Std,
}

impl AbbreviatedTensorValue {
    fn get_digest(
        tensor: &NumericTensor<DynRank>,
        digest_len: u64,
        backend: &mut EvalBackend,
    ) -> Option<(Vec<u8>, ScaleParams)> {
        match tensor.dtype() {
            DType::F32 | DType::F64 | DType::BF16 | DType::F16 => {
                // Ok
            }
            _ => {
                return None;
            }
        }
        let flattened_tensor = tensor.flatten().unwrap().to_dyn_rank();
        let num_elements = flattened_tensor.shape()[0];
        let digest_len = digest_len.min(num_elements);
        let digest = if num_elements > digest_len {
            let ones = flattened_tensor.ones_like(backend).unwrap();
            let ps = flattened_tensor
                .cumsum(Some(0), false, false, backend)
                .unwrap();
            let pc = ones.cumsum(Some(0), false, false, backend).unwrap();
            let edges_0 = (0..digest_len)
                .map(|x| (((x * num_elements) / digest_len) as i64).min(num_elements as i64 - 1))
                .collect::<Vec<_>>();
            let edges_1 = (1..digest_len + 1)
                .map(|x| (((x * num_elements) / digest_len) as i64).min(num_elements as i64 - 1))
                .collect::<Vec<_>>();
            let edges_0_array = NumericTensor::from_vec(edges_0).to_dyn_rank();
            let edges_1_array = NumericTensor::from_vec(edges_1).to_dyn_rank();
            let s_hi = NumericTensor::gather(&ps, &edges_1_array, 0, backend).unwrap();
            let s_lo = NumericTensor::gather(&ps, &edges_0_array, 0, backend).unwrap();
            let c_hi = NumericTensor::gather(&pc, &edges_1_array, 0, backend).unwrap();
            let c_lo = NumericTensor::gather(&pc, &edges_0_array, 0, backend).unwrap();
            let sum_k = NumericTensor::sub(&s_hi, &s_lo, backend).unwrap();
            let tmp = NumericTensor::sub(&c_hi, &c_lo, backend).unwrap();
            let tmp2 = tmp.ones_like(backend).unwrap();
            let cnt_k = NumericTensor::max(&tmp, &tmp2, backend).unwrap();
            NumericTensor::div(&sum_k, &cnt_k, backend).unwrap()
        } else {
            flattened_tensor
        };
        let res_vec: Vec<f32> = digest
            .to_ndarray()
            .unwrap()
            .flatten()
            .cast(DType::F32)
            .unwrap()
            .try_to_vec()
            .unwrap();

        Some(Self::scale_and_quantize(&res_vec, ScaleMode::MinMax))
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

    pub fn from_tensor(
        tensor: &NumericTensor<DynRank>,
        digest_len: u64,
        backend: &mut EvalBackend,
    ) -> Self {
        let value = Self::get_digest(tensor, digest_len, backend);
        Self {
            value,
            dtype: tensor.dtype(),
            shape: tensor.shape().to_vec(),
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SuperGraphExecutionReport {
    pub attention: Option<u64>,
    pub node_executions: Vec<(SuperGraphNodePath, Duration, Duration)>,
    pub abbreviated_tensor_assignments: Vec<(SuperGraphTensorPath, AbbreviatedTensorValue)>,
    pub tensor_assignments: Vec<(SuperGraphTensorPath, NDArrayNumericTensor<DynRank>)>,
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
pub struct ServerConfigReport {
    pub vulkan_available: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum WebsocketServerClientMessage {
    Pong,
    ServerConfigReport(ServerConfigReport),
    ModelLoadReturn(Result<(), String>),
    CurrentModelsReport(CurrentModelsAndInterfacesReport),
    ModelGraphReturn(Result<(LoadedModelId, Vec<u8>), String>),
    TensorStoreReturn(
        LoadedModelId,
        TensorStoreTensorId,
        Result<NDArrayNumericTensor<DynRank>, String>,
    ),
    HFTokenizerReturn(String, Result<Vec<u8>, String>),
    SuperGraphResponse(SuperGraphResponse),
    SuperGraphExecutionReport(SuperGraphExecutionReport),
}
