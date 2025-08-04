use axum::{
    Router,
    extract::ws::{WebSocket, WebSocketUpgrade},
    response::IntoResponse,
    routing::get,
};
use std::default::Default;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicU32;
use std::time;
use tokio::sync::watch;
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
}

pub(crate) struct ModelServer {
    models: RwLock<Vec<ModelData>>,
    next_model_id: AtomicU32,
    models_report_watch_sender: watch::Sender<CurrentModelsAndInterfacesReport>,
    models_report_watch_receiver: watch::Receiver<CurrentModelsAndInterfacesReport>,
}

impl ModelServer {
    pub(crate) fn new() -> Self {
        let (models_report_watch_sender, models_report_watch_receiver) =
            watch::channel(CurrentModelsAndInterfacesReport::new());
        Self {
            models: RwLock::new(vec![]),
            next_model_id: AtomicU32::new(0),
            models_report_watch_sender,
            models_report_watch_receiver,
        }
    }

    pub(crate) async fn generate_new_model_report(&self) {
        let guard = self.models.read().await;
        let mut new_report = CurrentModelsAndInterfacesReport::default();

        for model in guard.iter() {
            new_report.models.push(CurrentModelsReportEntry {
                model_id: model.model_id,
                model_name: model.model_name.clone(),
                num_ops: model.model.get_symbolic_graph().get_operations().len() as u64,
            });

            for interface in get_automatic_interfaces_from_model(&model.model) {
                let name = model.model_name.clone() + "-" + &interface.name();
                new_report.interfaces.push(CurrentInterfacesReportEntry {
                    model_ids: vec![model.model_id],
                    interface,
                    interface_name: name,
                });
            }
        }
        self.models_report_watch_sender.send(new_report).unwrap()
    }

    pub(crate) async fn load_model(
        &self,
        model_path: &Path,
        model_hint: Option<ModelTypeHint>,
    ) -> Result<(), anyhow::Error> {
        let onnx_data =
            identify_and_load(model_path, WeightStorageStrategy::EmbeddedData, model_hint)?;

        let runtime_model = Model::new_from_onnx(&onnx_data)?;
        let model_name = model_path
            .file_stem()
            .unwrap_or_default()
            .to_str()
            .unwrap_or_default()
            .to_string();
        let model_id = self
            .next_model_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let mut guard = self.models.write().await;

        guard.push(ModelData {
            model: Arc::new(runtime_model),
            model_id: LoadedModelId(model_id),
            model_name,
        });

        drop(guard);
        self.generate_new_model_report().await;
        Ok(())
    }

    pub(crate) async fn unload_model(&self, model_id: LoadedModelId) -> Result<(), anyhow::Error> {
        let mut guard = self.models.write().await;
        guard.retain(|model| model.model_id != model_id);
        drop(guard);
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
) -> impl IntoResponse {
    ws.on_upgrade(|socket: WebSocket| handle_socket(socket, scheduler_sender, model_server))
}

use crate::scheduler::{SchedulerJob, scheduler};
use axum::extract::ws::Message;
use hf_hub::api::tokio::ApiBuilder;
use hf_hub::{Repo, RepoType};
use tokenizers::FromPretrainedParameters;
use whisper_tensor::interfaces::get_automatic_interfaces_from_model;
use whisper_tensor::model::Model;
use whisper_tensor::numeric_tensor::NumericTensor;
use whisper_tensor::symbolic_graph::tensor_store::TensorStoreTensorId;
use whisper_tensor_import::onnx_graph::WeightStorageStrategy;
use whisper_tensor_import::{ModelTypeHint, identify_and_load};
use whisper_tensor_server::{
    CurrentInterfacesReportEntry, CurrentModelsAndInterfacesReport, CurrentModelsReportEntry,
    LoadedModelId, WebsocketClientServerMessage, WebsocketServerClientMessage,
};

async fn send_message(socket: &mut WebSocket, message: WebsocketServerClientMessage) {
    let mut data = Vec::<u8>::new();
    ciborium::into_writer(&message, &mut data).unwrap();
    socket.send(Message::Binary(data.into())).await.unwrap();
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
) {
    // Send opening state
    let mut receiver = model_server.models_report_watch_receiver.clone();
    let initial_value = receiver.borrow_and_update().clone();
    send_message(
        &mut socket,
        WebsocketServerClientMessage::CurrentModelsReport(initial_value),
    )
    .await;

    let (finished_supergraph_job_tx, mut finished_supergraph_job_rx) = mpsc::channel(100);

    loop {
        tokio::select! {
            Some(response) = finished_supergraph_job_rx.recv() => {
                send_message(&mut socket, WebsocketServerClientMessage::SuperGraphResponse(response)).await;
            }
            Ok(_) = receiver.changed() => {
                let current_value = receiver.borrow_and_update().clone();
                send_message(&mut socket, WebsocketServerClientMessage::CurrentModelsReport(current_value)).await;
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
                                            send_message(&mut socket, WebsocketServerClientMessage::Pong).await;
                                        }
                                        WebsocketClientServerMessage::LoadModel{model_path, model_type_hint} => {
                                            let res = model_server.load_model(Path::new(&model_path), model_type_hint).await;
                                            let msg_out = WebsocketServerClientMessage::ModelLoadReturn(res.map_err(|x| {x.to_string()}));
                                            send_message(&mut socket, msg_out).await;
                                        }
                                        WebsocketClientServerMessage::UnloadModel(model_id) => {
                                            model_server.unload_model(model_id).await.ok();
                                        }
                                        WebsocketClientServerMessage::GetStoredTensor(model_id, stored_tensor_id) => {
                                            let res = model_server.get_stored_tensor_id(model_id, stored_tensor_id).await;
                                            let msg_out = WebsocketServerClientMessage::TensorStoreReturn(
                                                model_id, stored_tensor_id, res.map(|x| x.to_ndarray().unwrap())
                                            );
                                            send_message(&mut socket, msg_out).await;
                                        }
                                        WebsocketClientServerMessage::GetModelGraph(model_id) => {
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
                                            let msg_out = WebsocketServerClientMessage::HFTokenizerReturn(
                                                hf_tokenizer_path.clone(), hf_from_pretrained(hf_tokenizer_path, None).await.map(|x| {
                                                    let mut v = vec![];
                                                    File::open(x.clone()).unwrap().read_to_end(&mut v).unwrap();
                                                    v
                                                }));
                                            send_message(&mut socket, msg_out).await;
                                        }
                                        WebsocketClientServerMessage::SuperGraphRequest(request) => {
                                            let job = SchedulerJob::SuperGraphRequest((request, finished_supergraph_job_tx.clone()));
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
    // initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let model_server = Arc::new(ModelServer::new());

    // Load test models
    model_server
        .load_model(Path::new("gpt2-lm-head-10.onnx"), Some(ModelTypeHint::GPT2))
        .await
        .unwrap();
    model_server.load_model(Path::new("libs/onnx/onnx/backend/test/data/node/test_layer_normalization_2d_axis0_expanded/model.onnx"), None).await.unwrap();

    let (scheduler_tx, scheduler_rx) = mpsc::channel(100);

    tokio::spawn(scheduler(scheduler_rx, model_server.clone()));

    tokio::spawn(async move {
        let model_server = model_server.clone();
        let app = Router::new()
            .route("/health", get(|| async { "ok" }))
            .route(
                "/ws",
                get(move |ws: WebSocketUpgrade| {
                    websocket_handler(ws, scheduler_tx.clone(), model_server.clone())
                }),
            ) // Add WebSocket endpoint
            .nest_service("/pkg", ServeDir::new("./crates/whisper-tensor-webui/pkg/"))
            .nest_service(
                "/assets",
                ServeDir::new("./crates/whisper-tensor-webui/assets/"),
            )
            .route_service(
                "/index.html",
                ServeFile::new("./crates/whisper-tensor-webui/assets/index.html"),
            )
            .route_service(
                "/",
                ServeFile::new("./crates/whisper-tensor-webui/assets/index.html"),
            )
            .layer(TraceLayer::new_for_http());

        let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
        tracing::info!("WebUI server listening on port 3000");
        axum::serve(listener, app).await.unwrap();
    });

    loop {
        sleep(time::Duration::from_millis(1000)).await;
    }
}
