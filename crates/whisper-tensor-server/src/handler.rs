use std::fs::File;
use std::io::Read;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crossbeam::queue::ArrayQueue;
use hf_hub::api::tokio::ApiBuilder;
use hf_hub::{Repo, RepoType};
use tokio::sync::{Mutex, Notify, mpsc};
use tokenizers::FromPretrainedParameters;

use crate::model_server::ModelServer;
use crate::scheduler::{SchedulerJob, SchedulerReport, SchedulerReporter};
use crate::{
    ServerConfigReport, SuperGraphExecutionReport, WebsocketClientServerMessage,
    WebsocketServerClientMessage,
};

async fn hf_from_pretrained<S: AsRef<str>>(
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

async fn flush_reports(
    report_buffer: &Mutex<Vec<SchedulerReport>>,
    server_tx: &mpsc::UnboundedSender<WebsocketServerClientMessage>,
) {
    let mut buf = report_buffer.lock().await;
    while !buf.is_empty() {
        let mut report = SuperGraphExecutionReport {
            attention: buf.first().unwrap().get_attention_token(),
            node_executions: Vec::new(),
            tensor_assignments: Vec::new(),
            abbreviated_tensor_assignments: Vec::new(),
        };
        buf.retain(|x| {
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
        server_tx
            .send(WebsocketServerClientMessage::SuperGraphExecutionReport(
                report,
            ))
            .ok();
    }
}

/// Handles a client session using in-process channels (no WebSocket/CBOR).
/// This is the native equivalent of the WebSocket `handle_socket` in main.rs.
pub async fn handle_client_session(
    mut client_rx: mpsc::UnboundedReceiver<WebsocketClientServerMessage>,
    server_tx: mpsc::UnboundedSender<WebsocketServerClientMessage>,
    scheduler_sender: mpsc::Sender<SchedulerJob>,
    model_server: Arc<ModelServer>,
    server_config_report: ServerConfigReport,
    on_message_sent: impl Fn() + Send + 'static,
) {
    let mut receiver = model_server.watch_models_report();
    let report_queue = Arc::new(ArrayQueue::new(1000));
    let report_notify = Arc::new(Notify::new());
    let initial_value = receiver.borrow_and_update().clone();

    // Send initial state
    server_tx
        .send(WebsocketServerClientMessage::CurrentModelsReport(
            initial_value,
        ))
        .ok();
    server_tx
        .send(WebsocketServerClientMessage::ServerConfigReport(
            server_config_report,
        ))
        .ok();
    server_tx
        .send(WebsocketServerClientMessage::LoaderRegistryReport(
            model_server.get_loader_registry_report().clone(),
        ))
        .ok();
    on_message_sent();

    let (finished_tx, mut finished_rx) = mpsc::channel(100);
    let report_buffer = Mutex::new(Vec::<SchedulerReport>::new());
    let mut last_report_time = Instant::now();

    loop {
        tokio::select! {
            Some(response) = finished_rx.recv() => {
                {
                    let mut buf = report_buffer.lock().await;
                    while let Some(report) = report_queue.pop() {
                        buf.push(report);
                    }
                }
                flush_reports(&report_buffer, &server_tx).await;
                server_tx.send(WebsocketServerClientMessage::SuperGraphResponse(response)).ok();
                on_message_sent();
            }
            Ok(_) = receiver.changed() => {
                let current_value = receiver.borrow_and_update().clone();
                server_tx.send(WebsocketServerClientMessage::CurrentModelsReport(current_value)).ok();
                on_message_sent();
            }
            _ = report_notify.notified() => {
                {
                    let mut buf = report_buffer.lock().await;
                    while let Some(report) = report_queue.pop() {
                        buf.push(report);
                    }
                }
                if Instant::now() - last_report_time > Duration::from_millis(50) {
                    flush_reports(&report_buffer, &server_tx).await;
                    last_report_time = Instant::now();
                    on_message_sent();
                }
            }
            Some(msg) = client_rx.recv() => {
                match msg {
                    WebsocketClientServerMessage::Ping => {
                        server_tx.send(WebsocketServerClientMessage::Pong).ok();
                        on_message_sent();
                    }
                    WebsocketClientServerMessage::RunLoader{loader_index, config} => {
                        let res = model_server.run_loader(loader_index, config).await;
                        server_tx.send(WebsocketServerClientMessage::ModelLoadReturn(
                            res.map_err(|x| x.to_string())
                        )).ok();
                        on_message_sent();
                    }
                    WebsocketClientServerMessage::UnloadModel(model_id) => {
                        model_server.unload_model(model_id).await.ok();
                    }
                    WebsocketClientServerMessage::GetStoredTensor(model_id, stored_tensor_id) => {
                        tracing::debug!("Getting stored tensor");
                        let res = model_server.get_stored_tensor_id(model_id, stored_tensor_id).await;
                        server_tx.send(WebsocketServerClientMessage::TensorStoreReturn(
                            model_id, stored_tensor_id, res.map(|x| x.to_ndarray().unwrap())
                        )).ok();
                        on_message_sent();
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
                        server_tx.send(ret).ok();
                        on_message_sent();
                    }
                    WebsocketClientServerMessage::GetHFTokenizer(hf_tokenizer_path) => {
                        tracing::debug!("Getting HF tokenizer");
                        let msg_out = WebsocketServerClientMessage::HFTokenizerReturn(
                            hf_tokenizer_path.clone(),
                            hf_from_pretrained(hf_tokenizer_path, None).await.map(|x| {
                                let mut v = vec![];
                                File::open(x).unwrap().read_to_end(&mut v).unwrap();
                                v
                            }),
                        );
                        server_tx.send(msg_out).ok();
                        on_message_sent();
                    }
                    WebsocketClientServerMessage::GetTokenizerFile(path) => {
                        tracing::debug!("Getting tokenizer file: {path}");
                        let result = std::fs::read(&path)
                            .map_err(|e| format!("Failed to read tokenizer file {path}: {e}"));
                        server_tx.send(WebsocketServerClientMessage::TokenizerFileReturn(path, result)).ok();
                        on_message_sent();
                    }
                    WebsocketClientServerMessage::SuperGraphRequest(request) => {
                        let job = SchedulerJob::SuperGraphRequest((
                            request,
                            finished_tx.clone(),
                            Some(SchedulerReporter::new(report_queue.clone(), report_notify.clone())),
                        ));
                        scheduler_sender.send(job).await.unwrap();
                    }
                    WebsocketClientServerMessage::CompileModel(model_id) => {
                        let job = SchedulerJob::CompileModelRequest { model_id };
                        scheduler_sender.send(job).await.unwrap();
                    }
                }
            }
            else => break,
        }
    }

    tracing::debug!("Client session ended");
}
