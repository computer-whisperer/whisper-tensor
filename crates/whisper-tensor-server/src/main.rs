use axum::{
    Router,
    extract::ws::{WebSocket, WebSocketUpgrade},
    http::StatusCode,
    response::Html,
    response::IntoResponse,
    routing::get,
};
use std::default::Default;
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicU32;
use std::time;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, Notify, watch};
use tokio::sync::{RwLock, mpsc};
use tokio::time::sleep;
use tower_http::{
    services::{ServeDir, ServeFile},
    trace::TraceLayer,
};
use whisper_tensor::DynRank;

mod scheduler;

struct ModelData {
    model: Arc<Model>,
    model_id: LoadedModelId,
    model_name: String,
    compiled_program: Option<Arc<CompiledProgram>>,
}

pub(crate) struct ModelServer {
    models: RwLock<Vec<ModelData>>,
    interfaces: RwLock<Vec<CurrentInterfacesReportEntry>>,
    loaders: Vec<Box<dyn Loader>>,
    loader_registry_report: LoaderRegistryReport,
    next_model_id: AtomicU32,
    models_report_watch_sender: watch::Sender<CurrentModelsAndInterfacesReport>,
    models_report_watch_receiver: watch::Receiver<CurrentModelsAndInterfacesReport>,
}

impl ModelServer {
    pub(crate) fn new(loaders: Vec<Box<dyn Loader>>) -> Self {
        let (models_report_watch_sender, models_report_watch_receiver) =
            watch::channel(CurrentModelsAndInterfacesReport::new());

        let loader_registry_report = LoaderRegistryReport {
            loaders: loaders
                .iter()
                .map(|l| LoaderRegistryEntry {
                    name: l.name().to_string(),
                    description: l.description().to_string(),
                    config_schema: l.config_schema(),
                })
                .collect(),
        };

        Self {
            models: RwLock::new(vec![]),
            interfaces: RwLock::new(vec![]),
            loaders,
            loader_registry_report,
            next_model_id: AtomicU32::new(0),
            models_report_watch_sender,
            models_report_watch_receiver,
        }
    }

    pub(crate) fn get_loader_registry_report(&self) -> &LoaderRegistryReport {
        &self.loader_registry_report
    }

    pub(crate) async fn generate_new_model_report(&self) {
        let guard = self.models.read().await;
        let mut new_report = CurrentModelsAndInterfacesReport::default();

        for model in guard.iter() {
            new_report.models.push(CurrentModelsReportEntry {
                model_id: model.model_id,
                model_name: model.model_name.clone(),
                num_ops: model.model.get_symbolic_graph().get_operations().len() as u64,
                model_compiled: model.compiled_program.is_some(),
            });
        }

        let ifaces = self.interfaces.read().await;
        for entry in ifaces.iter() {
            new_report.interfaces.push(entry.clone());
        }

        self.models_report_watch_sender.send(new_report).unwrap()
    }

    pub(crate) async fn run_loader(
        &self,
        loader_index: usize,
        config: ConfigValues,
    ) -> Result<(), anyhow::Error> {
        let loader = self
            .loaders
            .get(loader_index)
            .ok_or_else(|| anyhow::anyhow!("Invalid loader index: {}", loader_index))?;

        tracing::info!("Running loader: {}", loader.name());
        let output = loader.load(config)?;

        let mut model_ids = Vec::new();

        // Register all models
        {
            let mut guard = self.models.write().await;
            for loaded_model in output.models {
                let model_id = LoadedModelId(
                    self.next_model_id
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed),
                );
                tracing::info!("Registered model '{}' as {}", loaded_model.name, model_id);
                guard.push(ModelData {
                    model: loaded_model.model,
                    model_id,
                    model_name: loaded_model.name,
                    compiled_program: None,
                });
                model_ids.push(model_id);
            }
        }

        // Register all interfaces
        {
            let mut ifaces = self.interfaces.write().await;
            for loaded_interface in output.interfaces {
                tracing::info!("Registered interface '{}'", loaded_interface.name);
                ifaces.push(CurrentInterfacesReportEntry {
                    model_ids: model_ids.clone(),
                    interface_name: loaded_interface.name,
                    interface: loaded_interface.interface,
                });
            }
        }

        self.generate_new_model_report().await;
        Ok(())
    }

    pub(crate) async fn unload_model(&self, model_id: LoadedModelId) -> Result<(), anyhow::Error> {
        let mut guard = self.models.write().await;
        guard.retain(|model| model.model_id != model_id);
        drop(guard);
        let mut ifaces = self.interfaces.write().await;
        ifaces.retain(|entry| !entry.model_ids.contains(&model_id));
        drop(ifaces);
        self.generate_new_model_report().await;
        Ok(())
    }

    pub(crate) async fn get_model(&self, model_id: LoadedModelId) -> Option<Arc<Model>> {
        let guard = self.models.read().await;
        guard
            .iter()
            .find(|model| model.model_id == model_id)
            .map(|model| model.model.clone())
    }

    pub(crate) async fn get_compiled_model(
        &self,
        model_id: LoadedModelId,
    ) -> Option<Arc<CompiledProgram>> {
        let guard = self.models.read().await;
        guard
            .iter()
            .find(|model| model.model_id == model_id)
            .and_then(|model| model.compiled_program.clone())
    }

    pub(crate) async fn set_compiled_model(
        &self,
        model_id: LoadedModelId,
        compiled_program: Arc<CompiledProgram>,
    ) {
        let mut guard = self.models.write().await;
        if let Some(model) = guard.iter_mut().find(|model| model.model_id == model_id) {
            model.compiled_program = Some(compiled_program);
        }
        drop(guard);
        self.generate_new_model_report().await;
    }

    pub(crate) async fn with_model<T>(
        &self,
        model_id: LoadedModelId,
        f: impl FnOnce(&ModelData) -> T,
    ) -> Result<T, String> {
        let guard = self.models.read().await;
        if let Some(model) = guard.iter().find(|model| model.model_id == model_id) {
            Ok(f(model))
        } else {
            Err(format!("Model with id {model_id} not found"))
        }
    }

    pub(crate) async fn get_stored_tensor_id(
        &self,
        model_id: LoadedModelId,
        stored_tensor_id: TensorStoreTensorId,
    ) -> Result<NumericTensor<DynRank>, String> {
        let guard = self.models.read().await;
        if let Some(model) = guard.iter().find(|model| model.model_id == model_id) {
            model
                .model
                .get_tensor_store()
                .get_tensor(stored_tensor_id)
                .map(|x| x.to_numeric())
                .ok_or("Tensor not found in Tensor Store".to_string())
        } else {
            Err(format!("Model with id {model_id} not found"))
        }
    }
}

async fn websocket_handler(
    ws: WebSocketUpgrade,
    scheduler_sender: mpsc::Sender<SchedulerJob>,
    model_server: Arc<ModelServer>,
    server_config_report: ServerConfigReport,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket: WebSocket| {
        handle_socket(
            socket,
            scheduler_sender,
            model_server,
            server_config_report.clone(),
        )
    })
}

use crate::scheduler::{SchedulerJob, SchedulerReport, SchedulerReporter, scheduler};
use axum::extract::ws::Message;
use crossbeam::queue::ArrayQueue;
use hf_hub::api::tokio::ApiBuilder;
use hf_hub::{Repo, RepoType};
use tokenizers::FromPretrainedParameters;
use whisper_tensor::compiler::CompiledProgram;
use whisper_tensor::loader::{ConfigValue, ConfigValues, Loader};
use whisper_tensor::model::Model;
use whisper_tensor::numeric_tensor::NumericTensor;
use whisper_tensor::symbolic_graph::tensor_store::TensorStoreTensorId;
use whisper_tensor_server::{
    CurrentInterfacesReportEntry, CurrentModelsAndInterfacesReport, CurrentModelsReportEntry,
    LoadedModelId, LoaderRegistryEntry, LoaderRegistryReport, ServerConfigReport,
    SuperGraphExecutionReport, WebsocketClientServerMessage, WebsocketServerClientMessage,
};

const WEBUI_PKG_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../whisper-tensor-webui/pkg");
const WEBUI_ASSETS_DIR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../whisper-tensor-webui/assets"
);
const NOT_FOUND_HTML: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../whisper-tensor-webui/assets/404.html"
));

async fn send_message(socket: &mut WebSocket, message: WebsocketServerClientMessage) {
    let mut data = Vec::<u8>::new();
    ciborium::into_writer(&message, &mut data).unwrap();
    socket.send(Message::Binary(data.into())).await.unwrap();
}

async fn not_found() -> impl IntoResponse {
    (StatusCode::NOT_FOUND, Html(NOT_FOUND_HTML))
}

pub async fn hf_from_pretrained<S: AsRef<str>>(
    identifier: S,
    params: Option<FromPretrainedParameters>,
) -> Result<PathBuf, String> {
    let identifier: String = identifier.as_ref().to_string();

    let valid_chars = ['-', '_', '.', '/'];
    let is_valid_char = |x: char| x.is_alphanumeric() || valid_chars.contains(&x);

    let valid = identifier.chars().all(is_valid_char);
    let valid_chars_stringified = valid_chars
        .iter()
        .fold(vec![], |mut buf, x| {
            buf.push(format!("'{x}'"));
            buf
        })
        .join(", "); // "'/', '-', '_', '.'"
    if !valid {
        return Err(format!(
            "Model \"{identifier}\" contains invalid characters, expected only alphanumeric or {valid_chars_stringified}"
        ));
    }
    let params = params.unwrap_or_default();

    let revision = &params.revision;
    let valid_revision = revision.chars().all(is_valid_char);
    if !valid_revision {
        return Err(format!(
            "Revision \"{revision}\" contains invalid characters, expected only alphanumeric or {valid_chars_stringified}"
        ));
    }

    let mut builder = ApiBuilder::new();
    if let Some(token) = params.token {
        builder = builder.with_token(Some(token));
    }
    let api = builder.build().map_err(|x| x.to_string())?;
    let repo = Repo::with_revision(identifier, RepoType::Model, params.revision);
    let api = api.repo(repo);
    api.get("tokenizer.json").await.map_err(|x| x.to_string())
}

async fn handle_socket(
    mut socket: WebSocket,
    scheduler_sender: mpsc::Sender<SchedulerJob>,
    model_server: Arc<ModelServer>,
    server_config_report: ServerConfigReport,
) {
    // Send opening state
    let mut receiver = model_server.models_report_watch_receiver.clone();
    let report_queue = Arc::new(ArrayQueue::new(1000));
    let report_notify = Arc::new(Notify::new());
    let initial_value = receiver.borrow_and_update().clone();
    send_message(
        &mut socket,
        WebsocketServerClientMessage::CurrentModelsReport(initial_value),
    )
    .await;
    send_message(
        &mut socket,
        WebsocketServerClientMessage::ServerConfigReport(server_config_report),
    )
    .await;
    send_message(
        &mut socket,
        WebsocketServerClientMessage::LoaderRegistryReport(
            model_server.get_loader_registry_report().clone(),
        ),
    )
    .await;

    let (finished_supergraph_job_tx, mut finished_supergraph_job_rx) = mpsc::channel(100);

    let report_buffer = Mutex::new(Vec::<SchedulerReport>::new());
    let mut last_report_time = Instant::now();
    let flush_report = async |socket: &mut WebSocket| {
        let mut report_buffer = report_buffer.lock().await;
        while !report_buffer.is_empty() {
            let mut report = SuperGraphExecutionReport {
                attention: report_buffer.first().unwrap().get_attention_token(),
                node_executions: Vec::new(),
                tensor_assignments: Vec::new(),
                abbreviated_tensor_assignments: Vec::new(),
            };
            report_buffer.retain(|x| {
                if x.get_attention_token() == report.attention {
                    match x {
                        SchedulerReport::SuperGraphNodeExecuted(x) => {
                            let delta = Instant::now() - x.end_instant;
                            let duration = x.end_instant - x.start_instant;
                            report.node_executions.push((
                                x.path.clone(),
                                x.op_kind.clone(),
                                delta,
                                duration,
                            ));
                        }
                        SchedulerReport::SuperGraphTensorAssignedFull(x) => {
                            report
                                .tensor_assignments
                                .push((x.path.clone(), x.value.clone()));
                        }
                        SchedulerReport::SuperGraphTensorAssignedAbbreviated(x) => {
                            report
                                .abbreviated_tensor_assignments
                                .push((x.path.clone(), x.value.clone()));
                        }
                    }
                    false
                } else {
                    true
                }
            });
            send_message(
                socket,
                WebsocketServerClientMessage::SuperGraphExecutionReport(report),
            )
            .await;
        }
    };
    loop {
        tokio::select! {
            Some(response) = finished_supergraph_job_rx.recv() => {
                {
                    let mut report_buffer = report_buffer.lock().await;
                    while let Some(report) = report_queue.pop() {
                        report_buffer.push(report);
                    }
                }
                flush_report(&mut socket).await;
                send_message(&mut socket, WebsocketServerClientMessage::SuperGraphResponse(response)).await;
            }
            Ok(_) = receiver.changed() => {
                let current_value = receiver.borrow_and_update().clone();
                send_message(&mut socket, WebsocketServerClientMessage::CurrentModelsReport(current_value)).await;
            }
            _ = report_notify.notified() => {
                {
                    let mut report_buffer = report_buffer.lock().await;
                    while let Some(report) = report_queue.pop() {
                        report_buffer.push(report);
                    }
                }
                if Instant::now() - last_report_time > Duration::from_millis(50) {
                    flush_report(&mut socket).await;
                    last_report_time = Instant::now();
                }
            }
            Some(msg) = socket.recv() => {
                if let Ok(msg) = msg {
                    match msg {
                        Message::Binary(data) => {
                            tracing::debug!("Received a binary message, size: {} bytes", data.len());

                            match ciborium::from_reader::<WebsocketClientServerMessage, _>(data.to_vec().as_slice()) {
                                Ok(msg) => {
                                    match msg {
                                        WebsocketClientServerMessage::Ping => {
                                            tracing::debug!("Ping");
                                            send_message(&mut socket, WebsocketServerClientMessage::Pong).await;
                                        }
                                        WebsocketClientServerMessage::RunLoader{loader_index, config} => {
                                            let res = model_server.run_loader(loader_index, config).await;
                                            let msg_out = WebsocketServerClientMessage::ModelLoadReturn(res.map_err(|x| {x.to_string()}));
                                            send_message(&mut socket, msg_out).await;
                                        }
                                        WebsocketClientServerMessage::UnloadModel(model_id) => {
                                            model_server.unload_model(model_id).await.ok();
                                        }
                                        WebsocketClientServerMessage::GetStoredTensor(model_id, stored_tensor_id) => {
                                            tracing::debug!("Getting stored tensor");
                                            let res = model_server.get_stored_tensor_id(model_id, stored_tensor_id).await;
                                            let msg_out = WebsocketServerClientMessage::TensorStoreReturn(
                                                model_id, stored_tensor_id, res.map(|x| x.to_ndarray().unwrap())
                                            );
                                            send_message(&mut socket, msg_out).await;
                                        }
                                        WebsocketClientServerMessage::GetModelGraph(model_id) => {
                                            tracing::debug!("Getting model graph");
                                            let ret = match model_server.with_model(model_id, |model| {
                                                let graph = &model.model.get_symbolic_graph();
                                                let mut data = Vec::<u8>::new();
                                                ciborium::into_writer(graph, &mut data).unwrap();
                                                Ok(data)
                                            }).await {
                                                Ok(Ok(graph)) => {
                                                    WebsocketServerClientMessage::ModelGraphReturn(Ok((model_id, graph)))
                                                }
                                                Ok(Err(err)) => {
                                                    WebsocketServerClientMessage::ModelGraphReturn(Err(err))
                                                }
                                                Err(err) => {
                                                    WebsocketServerClientMessage::ModelGraphReturn(Err(err))
                                                }
                                            };
                                            send_message(&mut socket, ret).await;
                                        }
                                        WebsocketClientServerMessage::GetHFTokenizer(hf_tokenizer_path) => {
                                            tracing::debug!("Getting HF tokenizer");
                                            let msg_out = WebsocketServerClientMessage::HFTokenizerReturn(
                                                hf_tokenizer_path.clone(), hf_from_pretrained(hf_tokenizer_path, None).await.map(|x| {
                                                    let mut v = vec![];
                                                    File::open(x.clone()).unwrap().read_to_end(&mut v).unwrap();
                                                    v
                                                }));
                                            send_message(&mut socket, msg_out).await;
                                        }
                                        WebsocketClientServerMessage::GetTokenizerFile(path) => {
                                            tracing::debug!("Getting tokenizer file: {path}");
                                            let result = std::fs::read(&path)
                                                .map_err(|e| format!("Failed to read tokenizer file {path}: {e}"));
                                            let msg_out = WebsocketServerClientMessage::TokenizerFileReturn(path, result);
                                            send_message(&mut socket, msg_out).await;
                                        }
                                        WebsocketClientServerMessage::SuperGraphRequest(request) => {
                                            let job = SchedulerJob::SuperGraphRequest((request, finished_supergraph_job_tx.clone(), Some(SchedulerReporter::new(report_queue.clone(), report_notify.clone()))));
                                            scheduler_sender.send(job).await.unwrap();
                                        }
                                        WebsocketClientServerMessage::CompileModel(model_id) => {
                                            let job = SchedulerJob::CompileModelRequest{model_id};
                                            scheduler_sender.send(job).await.unwrap();
                                        }
                                        /*
                                        _ => {
                                            log::debug!("Unhandled message: {msg:?}")
                                        }*/
                                    }
                                }
                                Err(err) => {
                                    log::warn!("Failed to decode the message: {err:?}");
                                }
                            }
                        }
                        Message::Close(_) => {
                            tracing::debug!("Client disconnected");
                            break;
                        }
                        _ => {}
                    }
                } else {
                    tracing::error!("Error receiving message");
                    break;
                }
            }
        }
    }

    tracing::debug!("WebSocket connection closed");
}

#[tokio::main]
async fn main() {
    {} // initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    #[cfg(feature = "import")]
    let loaders: Vec<Box<dyn Loader>> = {
        use whisper_tensor_import::loaders::{
            AutoLoader, FluxLoader, KittenTTSLoader, KokoroLoader, OnnxLoader, PiperLoader,
            Rwkv7Loader, SD15Loader, TransformersLoader,
        };
        vec![
            Box::new(AutoLoader),
            Box::new(OnnxLoader),
            Box::new(TransformersLoader),
            Box::new(Rwkv7Loader),
            Box::new(SD15Loader),
            Box::new(FluxLoader),
            Box::new(KokoroLoader),
            Box::new(KittenTTSLoader),
            Box::new(PiperLoader),
        ]
    };
    #[cfg(not(feature = "import"))]
    let loaders: Vec<Box<dyn Loader>> = vec![];

    let model_server = Arc::new(ModelServer::new(loaders));

    // Load test models using the ONNX loader (index 1)
    {
        let mut config = std::collections::HashMap::new();
        config.insert(
            "path".to_string(),
            ConfigValue::FilePath("test_models/gpt2-lm-head-10.onnx".into()),
        );
        config.insert(
            "model_type".to_string(),
            ConfigValue::String("GPT2".to_string()),
        );
        model_server.run_loader(1, config).await.unwrap();
    }
    {
        let mut config = std::collections::HashMap::new();
        config.insert("path".to_string(), ConfigValue::FilePath("libs/onnx/onnx/backend/test/data/node/test_layer_normalization_2d_axis0_expanded/model.onnx".into()));
        model_server.run_loader(1, config).await.unwrap();
    }

    let (scheduler_tx, scheduler_rx) = mpsc::channel(100);

    #[cfg(feature = "vulkan")]
    let vulkan_available = true;
    #[cfg(not(feature = "vulkan"))]
    let vulkan_available = false;

    let server_config_report = ServerConfigReport { vulkan_available };

    tokio::spawn(scheduler(scheduler_rx, model_server.clone()));

    tokio::spawn(async move {
        let model_server = model_server.clone();
        let not_found_file = ServeFile::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../whisper-tensor-webui/assets/404.html"
        ));
        let app = Router::new()
            .route("/health", get(|| async { "ok" }))
            .route(
                "/ws",
                get(move |ws: WebSocketUpgrade| {
                    websocket_handler(
                        ws,
                        scheduler_tx.clone(),
                        model_server.clone(),
                        server_config_report.clone(),
                    )
                }),
            ) // Add WebSocket endpoint
            .nest_service(
                "/pkg",
                ServeDir::new(WEBUI_PKG_DIR).not_found_service(not_found_file.clone()),
            )
            .nest_service(
                "/assets",
                ServeDir::new(WEBUI_ASSETS_DIR).not_found_service(not_found_file.clone()),
            )
            .route_service(
                "/index.html",
                ServeFile::new(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../whisper-tensor-webui/assets/index.html"
                )),
            )
            .route_service(
                "/",
                ServeFile::new(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../whisper-tensor-webui/assets/index.html"
                )),
            )
            .fallback(not_found)
            .layer(TraceLayer::new_for_http());

        let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
        tracing::info!("WebUI server listening on port 3000");
        axum::serve(listener, app).await.unwrap();
    });

    loop {
        sleep(time::Duration::from_millis(1000)).await;
    }
}
