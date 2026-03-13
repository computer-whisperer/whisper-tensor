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
use std::time;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, Notify, mpsc};
use tokio::time::sleep;
use tower_http::{
    services::{ServeDir, ServeFile},
    trace::TraceLayer,
};

use axum::extract::ws::Message;
use crossbeam::queue::ArrayQueue;
use hf_hub::api::tokio::ApiBuilder;
use hf_hub::{Repo, RepoType};
use tokenizers::FromPretrainedParameters;
use whisper_tensor::loader::ConfigValue;
use whisper_tensor_server::model_server::{ModelServer, default_loaders};
use whisper_tensor_server::scheduler::{
    SchedulerJob, SchedulerReport, SchedulerReporter, scheduler,
};
use whisper_tensor_server::{
    ServerConfigReport, SuperGraphExecutionReport, WebsocketClientServerMessage,
    WebsocketServerClientMessage,
};

const WEBUI_PKG_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../whisper-tensor-ui/pkg");
const WEBUI_ASSETS_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../whisper-tensor-ui/assets");
const NOT_FOUND_HTML: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../whisper-tensor-ui/assets/404.html"
));

async fn send_message(socket: &mut WebSocket, message: WebsocketServerClientMessage) {
    let mut data = Vec::<u8>::new();
    ciborium::into_writer(&message, &mut data).unwrap();
    socket.send(Message::Binary(data.into())).await.unwrap();
}

async fn not_found() -> impl IntoResponse {
    (StatusCode::NOT_FOUND, Html(NOT_FOUND_HTML))
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
        .join(", ");
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
    let mut receiver = model_server.watch_models_report();
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
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let loaders = default_loaders();
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
            "/../whisper-tensor-ui/assets/404.html"
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
            )
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
                    "/../whisper-tensor-ui/assets/index.html"
                )),
            )
            .route_service(
                "/",
                ServeFile::new(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../whisper-tensor-ui/assets/index.html"
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
