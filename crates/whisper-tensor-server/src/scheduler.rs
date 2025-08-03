use crate::ModelServer;
use log::error;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::mpsc;
use whisper_tensor::backends::eval_backend::EvalBackend;
use whisper_tensor::numeric_tensor::NumericTensor;
use whisper_tensor::super_graph::data::SuperGraphData;
use whisper_tensor_server::{SuperGraphRequest, SuperGraphResponse};

pub enum SchedulerJob {
    SuperGraphRequest((SuperGraphRequest, mpsc::Sender<SuperGraphResponse>)),
}

pub async fn scheduler(mut input: mpsc::Receiver<SchedulerJob>, model_server: Arc<ModelServer>) {
    loop {
        if let Some(x) = input.recv().await {
            match x {
                SchedulerJob::SuperGraphRequest((req, resp_sender)) => {
                    let mut super_graph_data = SuperGraphData::new();
                    for (link, tensor) in req.tensor_inputs {
                        super_graph_data
                            .tensors
                            .insert(link, NumericTensor::from(tensor));
                    }
                    // Collect links to needed models
                    let models = {
                        let mut models = HashMap::new();
                        for (link, &model_id) in &req.model_inputs {
                            let model = model_server.get_model(model_id).await;
                            if let Some(model) = model {
                                models.insert(link.clone(), model);
                            }
                        }
                        models
                    };
                    // Populate data with refs
                    for link in req.model_inputs.keys() {
                        if let Some(model) = models.get(link) {
                            super_graph_data.models.insert(link.clone(), model);
                        }
                    }
                    for (link, data) in req.string_inputs {
                        super_graph_data.strings.insert(link, data);
                    }
                    for (link, hash) in req.hash_inputs {
                        super_graph_data.hashes.insert(link, hash);
                    }
                    let res = req
                        .super_graph
                        .run(super_graph_data, None, &mut EvalBackend::NDArray)
                        .unwrap();

                    let tensor_outputs = res
                        .tensors
                        .iter()
                        .map(|(k, v)| (k.clone(), v.to_ndarray().unwrap()))
                        .collect();
                    let string_outputs = res.strings;
                    let hash_outputs = res.hashes;

                    let resp = SuperGraphResponse {
                        attention_token: req.attention_token,
                        tensor_outputs,
                        string_outputs,
                        hash_outputs,
                    };
                    if let Err(e) = resp_sender.send(resp).await {
                        error!("Failed to send response: {e}");
                    }
                }
            }
        }
    }
}
