use std::collections::HashMap;
use whisper_tensor::DynRank;
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
pub struct SuperGraphRequest {
    pub attention_token: Option<u64>,
    pub super_graph: SuperGraph,
    pub string_inputs: HashMap<SuperGraphLinkString, String>,
    pub hash_inputs: HashMap<SuperGraphLinkHash, SuperGraphHash>,
    pub tensor_inputs: HashMap<SuperGraphLinkTensor, NDArrayNumericTensor<DynRank>>,
    pub model_inputs: HashMap<SuperGraphLinkModel, LoadedModelId>,
    pub subscribed_tensors: Vec<SuperGraphTensorPath>,
    pub do_node_execution_reports: bool,
    pub do_abbreviated_tensor_assignment_reports: Option<u32>,
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

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AbbreviatedTensorValue {
    pub digest: Vec<u8>,
    pub dtype: DType,
    pub shape: Vec<u64>,
}

impl AbbreviatedTensorValue {
    fn get_digest(tensor: &NumericTensor<DynRank>, digest_len: u32) -> Vec<u8> {
        vec![]
    }

    pub fn from_tensor(tensor: &NumericTensor<DynRank>, digest_len: u32) -> Self {
        Self {
            digest: Self::get_digest(tensor, digest_len),
            dtype: tensor.dtype(),
            shape: tensor.shape().to_vec(),
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SuperGraphExecutionReport {
    pub attention: Option<u64>,
    pub node_executions: Vec<(SuperGraphNodePath, u32)>,
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
    SuperGraphExecutionReport(SuperGraphExecutionReport),
}
