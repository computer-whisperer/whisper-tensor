use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time;
use tokio::time::sleep;
use tower_http::{
    services::{ServeDir, ServeFile},
    trace::TraceLayer,
};
use axum::{
    extract::ws::{WebSocket, WebSocketUpgrade},
    response::IntoResponse,
    routing::{get, post},
    Router,
};
use tokio::sync::RwLock;
use onnx_graph::WeightStorageStrategy;
use onnx_import::{identify_and_load, ModelTypeHint};
use whisper_tensor::{RuntimeBackend, RuntimeEnvironment, RuntimeModel};
use whisper_tensor::eval_backend::EvalBackend;
use tokio::sync::watch;

struct ModelData {
    runtime_model: RuntimeModel,
    model_id: u32,
    model_name: String
}

pub(crate) struct ModelServer {
    models: RwLock<Vec<ModelData>>,
    next_model_id: AtomicU32,
    models_report_watch_sender: watch::Sender<Vec<CurrentModelsReportEntry>>,
    models_report_watch_receiver: watch::Receiver<Vec<CurrentModelsReportEntry>>
}

impl ModelServer {
    pub(crate) fn new() -> Self {
        let (models_report_watch_sender, models_report_watch_receiver) = watch::channel(vec![]);
        Self {
            models: RwLock::new(vec![]),
            next_model_id: AtomicU32::new(0),
            models_report_watch_sender,
            models_report_watch_receiver
        }
    }
    
    pub(crate) async fn generate_new_model_report(&self) {
        let guard = self.models.read().await;
        let mut new_vec = vec![];
        for model in guard.iter() {
            new_vec.push(CurrentModelsReportEntry{
                model_id: model.model_id,
                model_name: model.model_name.clone(),
                num_ops: model.runtime_model.get_num_ops().map(|x| x as u64)
            })
        }
        self.models_report_watch_sender.send(new_vec).unwrap()
    }

    pub(crate) async fn load_model(&self, model_path: &Path, model_hint: Option<onnx_import::ModelTypeHint>) -> Result<(), anyhow::Error> {
        let onnx_data = identify_and_load(model_path, WeightStorageStrategy::EmbeddedData, model_hint)?;

        let runtime_model = RuntimeModel::load_onnx(&onnx_data, RuntimeBackend::Eval(EvalBackend::NDArray), RuntimeEnvironment::default())?;
        let model_name = model_path.file_stem().unwrap_or_default().to_str().unwrap_or_default().to_string();
        let model_id = self.next_model_id.fetch_add(1, Ordering::Relaxed);
        let mut guard = self.models.write().await;
        guard.push(ModelData{
            runtime_model,
            model_id,
            model_name
        });
        drop(guard);
        self.generate_new_model_report().await;
        Ok(())
    }

    pub(crate) async fn unload_model(&self, model_id: u32) -> Result<(), anyhow::Error> {
        let mut guard = self.models.write().await;
        guard.retain(|model| model.model_id != model_id);
        drop(guard);
        self.generate_new_model_report().await;
        Ok(())
    }
    
    pub(crate) async fn with_model<T>(&self, model_id: u32, f: impl FnOnce(&ModelData) -> T) -> Result<T, String>
    {
        let guard = self.models.read().await;
        if let Some(model) = guard.iter().find(|model| model.model_id == model_id) {
            Ok(f(model))
        }
        else {
            Err(format!("Model with id {} not found", model_id))
        }
    }
}

async fn websocket_handler(ws: WebSocketUpgrade, model_server: Arc<ModelServer>) -> impl IntoResponse {
    ws.on_upgrade(|socket: WebSocket| {handle_socket(socket, model_server)})
}

use whisper_tensor_webui::{WebsocketServerClientMessage, WebsocketClientServerMessage, CurrentModelsReportEntry};

use axum::extract::ws::Message;
use tracing::Instrument;

async fn send_message(socket: &mut WebSocket, message: WebsocketServerClientMessage) {
    let data = rmp_serde::to_vec(&message).unwrap();
    socket.send(Message::Binary(data.into())).await.unwrap();
}

async fn handle_socket(mut socket: WebSocket, model_server: Arc<ModelServer>) {


    // Send opening state
    let mut receiver = model_server.models_report_watch_receiver.clone();
    let initial_value = receiver.borrow_and_update().clone();
    send_message(&mut socket, WebsocketServerClientMessage::CurrentModelsReport(initial_value)).await;

    loop {
        tokio::select! {
            Ok(_) = receiver.changed() => {
                let current_value = receiver.borrow_and_update().clone();
                send_message(&mut socket, WebsocketServerClientMessage::CurrentModelsReport(current_value)).await;
            }
            Some(msg) = socket.recv() => {
                if let Ok(msg) = msg {
                    match msg {
                        Message::Binary(data) => {
                            tracing::debug!("Received a binary message, size: {} bytes", data.len());

                            match rmp_serde::from_slice::<WebsocketClientServerMessage>(&data) {
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
                                        WebsocketClientServerMessage::GetModelGraph(model_id) => {
                                            let ret = match model_server.with_model(model_id, |model| {
                                                if let Some(graph) = &model.runtime_model.get_symbolic_graph() {
                                                    Ok(rmp_serde::to_vec(graph).unwrap())
                                                } else {
                                                    Err("Model runtime does not support graph introspection".to_string())
                                                }
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
                                        _ => {
                                            log::debug!("Unhandled message: {:?}", msg)
                                        }
                                    }
                                }
                                Err(err) => {
                                    log::warn!("Failed to decode the message: {:?}", err);
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
        .with_max_level(tracing::Level::TRACE)
        .init();

    let model_server = Arc::new(ModelServer::new());

    // Load test models
    model_server.load_model(Path::new("gpt2-lm-head-10.onnx"), Some(ModelTypeHint::GPT2)).await.unwrap();
    model_server.load_model(Path::new("libs/onnx/onnx/backend/test/data/node/test_layer_normalization_2d_axis0_expanded/model.onnx"), None).await.unwrap();
    
    tokio::spawn(async move {
        let model_server = model_server.clone();
        let app = Router::new()
            .route("/health", get(|| async { "ok" }))
            .route("/ws", get(move |ws: WebSocketUpgrade| {websocket_handler(ws, model_server.clone())}))  // Add WebSocket endpoint
            .nest_service("/pkg", ServeDir::new("./crates/whisper-tensor-webui/pkg/"))
            .nest_service("/assets", ServeDir::new("./crates/whisper-tensor-webui/assets/"))
            .route_service("/index.html", ServeFile::new("./crates/whisper-tensor-webui/assets/index.html"))
            .route_service("/", ServeFile::new("./crates/whisper-tensor-webui/assets/index.html"))
            .layer(TraceLayer::new_for_http());

        let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
        tracing::info!("WebUI server listening on port 3000");
        axum::serve(listener, app).await.unwrap();
    });

    loop {
        sleep(time::Duration::from_millis(1000)).await;
    }
}
