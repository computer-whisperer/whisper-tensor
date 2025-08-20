use crate::ModelServer;
use crossbeam::queue::ArrayQueue;
use log::error;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::{Notify, mpsc};
use tokio::time::Instant;
use whisper_tensor::DynRank;
use whisper_tensor::backends::eval_backend::EvalBackend;
use whisper_tensor::backends::ndarray_backend::NDArrayNumericTensor;
use whisper_tensor::numeric_tensor::NumericTensor;
use whisper_tensor::super_graph::data::SuperGraphData;
use whisper_tensor::super_graph::observer::SuperGraphObserver;
use whisper_tensor::super_graph::{SuperGraphNodePath, SuperGraphTensorPath};
use whisper_tensor_server::{AbbreviatedTensorValue, SuperGraphRequest, SuperGraphResponse};

#[derive(Debug)]
pub struct SchedulerReportSuperGraphNodeExecuted {
    pub attention: Option<u64>,
    pub timestamp: Instant,
    pub path: SuperGraphNodePath,
}

#[derive(Debug)]
pub struct SchedulerReportSuperGraphTensorAssigned {
    pub attention: Option<u64>,
    pub path: SuperGraphTensorPath,
    pub value: NDArrayNumericTensor<DynRank>,
}

#[derive(Debug)]
pub struct SchedulerReportSuperGraphTensorAssignedAbbreviated {
    pub attention: Option<u64>,
    pub path: SuperGraphTensorPath,
    pub value: AbbreviatedTensorValue,
}

#[allow(clippy::enum_variant_names)]
#[derive(Debug)]
pub enum SchedulerReport {
    SuperGraphNodeExecuted(SchedulerReportSuperGraphNodeExecuted),
    SuperGraphTensorAssignedFull(SchedulerReportSuperGraphTensorAssigned),
    SuperGraphTensorAssignedAbbreviated(SchedulerReportSuperGraphTensorAssignedAbbreviated),
}

impl SchedulerReport {
    pub fn get_attention_token(&self) -> Option<u64> {
        match self {
            SchedulerReport::SuperGraphNodeExecuted(report) => report.attention,
            SchedulerReport::SuperGraphTensorAssignedFull(report) => report.attention,
            SchedulerReport::SuperGraphTensorAssignedAbbreviated(report) => report.attention,
        }
    }
}

pub struct SchedulerReporter {
    queue: Arc<ArrayQueue<SchedulerReport>>,
    notify: Arc<Notify>,
    notify_counter: u64,
}

impl SchedulerReporter {
    pub fn new(queue: Arc<ArrayQueue<SchedulerReport>>, notify: Arc<Notify>) -> Self {
        Self {
            queue,
            notify,
            notify_counter: 0,
        }
    }
    pub fn push_report(&mut self, report: SchedulerReport) {
        let was_empty = self.queue.is_empty();
        self.queue.push(report).ok();
        if was_empty || self.notify_counter > 100 {
            self.notify.notify_one();
            self.notify_counter = 0;
        } else {
            self.notify_counter += 1;
        }
    }
}

pub enum SchedulerJob {
    SuperGraphRequest(
        (
            SuperGraphRequest,
            mpsc::Sender<SuperGraphResponse>,
            Option<SchedulerReporter>,
        ),
    ),
}

struct LocalSuperGraphObserver {
    attention: Option<u64>,
    do_node_execute_report: bool,
    do_abbreviated_tensor_assign_report: Option<u64>,
    reporter: Option<SchedulerReporter>,
    subscribed_tensors: HashSet<SuperGraphTensorPath>,
}

impl SuperGraphObserver for LocalSuperGraphObserver {
    fn on_node_executed(&mut self, path: &SuperGraphNodePath, _backend: &mut EvalBackend) {
        if let Some(reporter) = &mut self.reporter
            && self.do_node_execute_report
        {
            let report =
                SchedulerReport::SuperGraphNodeExecuted(SchedulerReportSuperGraphNodeExecuted {
                    attention: self.attention,
                    path: path.clone(),
                    timestamp: Instant::now(),
                });
            reporter.push_report(report);
        }
    }

    fn on_tensor_assigned(
        &mut self,
        path: &SuperGraphTensorPath,
        tensor: &NumericTensor<DynRank>,
        backend: &mut EvalBackend,
    ) {
        if let Some(reporter) = &mut self.reporter {
            if self.subscribed_tensors.contains(path) {
                let report = SchedulerReport::SuperGraphTensorAssignedFull(
                    SchedulerReportSuperGraphTensorAssigned {
                        attention: self.attention,
                        path: path.clone(),
                        value: tensor.to_ndarray().unwrap(),
                    },
                );
                reporter.push_report(report);
            }
            if let Some(x) = self.do_abbreviated_tensor_assign_report {
                let report = SchedulerReport::SuperGraphTensorAssignedAbbreviated(
                    SchedulerReportSuperGraphTensorAssignedAbbreviated {
                        attention: self.attention,
                        path: path.clone(),
                        value: AbbreviatedTensorValue::from_tensor(tensor, x, backend),
                    },
                );
                reporter.push_report(report);
            }
        }
    }
}

pub async fn scheduler(mut input: mpsc::Receiver<SchedulerJob>, model_server: Arc<ModelServer>) {
    loop {
        if let Some(x) = input.recv().await {
            match x {
                SchedulerJob::SuperGraphRequest((req, resp_sender, reporter)) => {
                    // Collect links to needed models
                    let models = {
                        let mut models = HashMap::new();
                        for (link, &model_id) in &req.model_inputs {
                            let model = model_server.get_model(model_id).await;
                            if let Some(model) = model {
                                models.insert(*link, model);
                            }
                        }
                        models
                    };
                    // Dispatch tight loop
                    let resp = tokio::task::spawn_blocking(move || {
                        let mut super_graph_data = SuperGraphData::new();
                        for (link, tensor) in req.tensor_inputs {
                            super_graph_data
                                .tensors
                                .insert(link, NumericTensor::from(tensor));
                        }
                        // Populate data with refs
                        for link in req.model_inputs.keys() {
                            if let Some(model) = models.get(link) {
                                super_graph_data.models.insert(*link, model);
                            }
                        }
                        for (link, data) in req.string_inputs {
                            super_graph_data.strings.insert(link, data);
                        }
                        for (link, hash) in req.hash_inputs {
                            super_graph_data.hashes.insert(link, hash);
                        }
                        let mut observer = LocalSuperGraphObserver {
                            attention: req.attention_token,
                            do_node_execute_report: req.do_node_execution_reports,
                            do_abbreviated_tensor_assign_report: req
                                .do_abbreviated_tensor_assignment_reports,
                            subscribed_tensors: req.subscribed_tensors.iter().cloned().collect(),
                            reporter,
                        };
                        let res = req
                            .super_graph
                            .run(
                                super_graph_data,
                                None,
                                &mut observer,
                                &mut EvalBackend::NDArray,
                            )
                            .unwrap();

                        let SuperGraphData {
                            tensors,
                            strings,
                            hashes,
                            ..
                        } = res;

                        let tensor_outputs = tensors
                            .iter()
                            .map(|(k, v)| (*k, v.to_ndarray().unwrap()))
                            .collect();

                        SuperGraphResponse {
                            attention_token: req.attention_token,
                            tensor_outputs,
                            string_outputs: strings,
                            hash_outputs: hashes,
                        }
                    })
                    .await
                    .unwrap();

                    if let Err(e) = resp_sender.send(resp).await {
                        error!("Failed to send response: {e}");
                    }
                }
            }
        }
    }
}
