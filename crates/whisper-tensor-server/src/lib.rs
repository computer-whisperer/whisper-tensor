use whisper_tensor::DynRank;
use whisper_tensor::backends::ndarray_backend::NDArrayNumericTensor;
use whisper_tensor::symbolic_graph::tensor_store::TensorStoreTensorId;
use whisper_tensor_import::ModelTypeHint;
use whisper_tensor_import::onnx_graph::TokenizerInfo;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct LoadedModelId(pub u32);

impl core::fmt::Display for LoadedModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(
    Debug,
    Clone,
    PartialEq,
    Eq,
    Hash,
    serde::Serialize,
    serde::Deserialize,
    strum_macros::EnumIter,
    strum_macros::Display,
)]
pub enum ModelLoadType {
    LLM,
    Other,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LLMMetadata {
    pub tokenizer_info: TokenizerInfo,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum ModelTypeMetadata {
    LLM(LLMMetadata),
    Other,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ForwardLogitRequest {
    pub model_id: LoadedModelId,
    pub context_tokens: Vec<u32>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum WebsocketClientServerMessage {
    Ping,
    LoadModel {
        model_path: String,
        model_type_hint: Option<ModelTypeHint>,
        model_load_type: ModelLoadType,
    },
    UnloadModel(LoadedModelId),
    GetModelGraph(LoadedModelId),
    GetStoredTensor(LoadedModelId, TensorStoreTensorId),
    GetHFTokenizer(String),
    GetLogits(ForwardLogitRequest),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CurrentModelsReportEntry {
    pub model_id: LoadedModelId,
    pub model_name: String,
    pub num_ops: u64,
    pub model_type_metadata: ModelTypeMetadata,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum WebsocketServerClientMessage {
    Pong,
    ModelLoadReturn(Result<(), String>),
    CurrentModelsReport(Vec<CurrentModelsReportEntry>),
    ModelGraphReturn(Result<(LoadedModelId, Vec<u8>), String>),
    TensorStoreReturn(
        LoadedModelId,
        TensorStoreTensorId,
        Result<NDArrayNumericTensor<DynRank>, String>,
    ),
    HFTokenizerReturn(String, Result<Vec<u8>, String>),
    GetLogitsReturn(ForwardLogitRequest, Result<Vec<Vec<(u32, f32)>>, String>),
}
