use std::collections::HashMap;
use whisper_tensor::DynRank;
use whisper_tensor::backends::ndarray_backend::NDArrayNumericTensor;
use whisper_tensor::interfaces::AnyInterface;
use whisper_tensor::super_graph::links::{
    SuperGraphLinkHash, SuperGraphLinkModel, SuperGraphLinkString, SuperGraphLinkTensor,
};
use whisper_tensor::super_graph::{SuperGraph, SuperGraphHash};
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
pub struct SuperGraphRequest {
    pub attention_token: Option<u64>,
    pub super_graph: SuperGraph,
    pub string_inputs: HashMap<SuperGraphLinkString, String>,
    pub hash_inputs: HashMap<SuperGraphLinkHash, SuperGraphHash>,
    pub tensor_inputs: HashMap<SuperGraphLinkTensor, NDArrayNumericTensor<DynRank>>,
    pub model_inputs: HashMap<SuperGraphLinkModel, LoadedModelId>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SuperGraphResponse {
    pub attention_token: Option<u64>,
    pub string_outputs: HashMap<SuperGraphLinkString, String>,
    pub hash_outputs: HashMap<SuperGraphLinkHash, SuperGraphHash>,
    pub tensor_outputs: HashMap<SuperGraphLinkTensor, NDArrayNumericTensor<DynRank>>,
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

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct CurrentModelsAndInterfacesReport {
    pub models: Vec<CurrentModelsReportEntry>,
    pub interfaces: HashMap<String, CurrentInterfacesReportEntry>,
}

impl CurrentModelsAndInterfacesReport {
    pub fn new() -> Self {
        Self {
            models: Vec::new(),
            interfaces: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum WebsocketServerClientMessage {
    Pong,
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
}
