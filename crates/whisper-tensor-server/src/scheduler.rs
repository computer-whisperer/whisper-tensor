use crate::ModelServer;
use crossbeam::queue::ArrayQueue;
use log::error;
use std::collections::{HashMap, HashSet};
use std::ptr;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokio::sync::{Notify, mpsc};
use whisper_tensor::DynRank;
use whisper_tensor::backends::ModelLoadedTensorCache;
use whisper_tensor::backends::eval_backend::EvalBackend;
use whisper_tensor::backends::ndarray_backend::NDArrayNumericTensor;
use whisper_tensor::numeric_tensor::NumericTensor;
use whisper_tensor::super_graph::cache::{SuperGraphCache, SuperGraphTensorCache};
use whisper_tensor::super_graph::data::SuperGraphData;
use whisper_tensor::super_graph::observer::SuperGraphObserver;
use whisper_tensor::super_graph::{SuperGraphContext, SuperGraphNodePath, SuperGraphTensorPath};
use whisper_tensor_server::{
    AbbreviatedTensorReportSettings, AbbreviatedTensorValue, LoadedModelId, SuperGraphRequest,
    SuperGraphRequestBackendMode, SuperGraphResponse, SuperGraphResponseData,
};

#[derive(Debug)]
pub struct SchedulerReportSuperGraphNodeExecuted {
    pub attention: Option<u64>,
    pub start_instant: Instant,
    pub end_instant: Instant,
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
    reporter: Option<SchedulerReporter>,
    subscribed_tensors: HashSet<SuperGraphTensorPath>,
    abbreviated_tensor_settings: Option<AbbreviatedTensorReportSettings>,
    abbreviated_tensor_subscribed_table: Option<HashSet<SuperGraphTensorPath>>,
}

impl LocalSuperGraphObserver {
    pub fn new(
        attention: Option<u64>,
        do_node_execute_report: bool,
        abbreviated_tensor_settings: Option<AbbreviatedTensorReportSettings>,
        reporter: Option<SchedulerReporter>,
        subscribed_tensors: HashSet<SuperGraphTensorPath>,
    ) -> Self {
        let abbreviated_tensor_subscribed_table = abbreviated_tensor_settings
            .as_ref()
            .map(|settings| settings.subscribed_tensors.iter().cloned().collect());
        Self {
            attention,
            do_node_execute_report,
            abbreviated_tensor_settings,
            reporter,
            subscribed_tensors,
            abbreviated_tensor_subscribed_table,
        }
    }
}

impl SuperGraphObserver for LocalSuperGraphObserver {
    fn on_node_executed(
        &mut self,
        path: &SuperGraphNodePath,
        start_instant: Instant,
        end_instant: Instant,
        _backend: &mut EvalBackend,
    ) {
        if let Some(reporter) = &mut self.reporter
            && self.do_node_execute_report
        {
            let report =
                SchedulerReport::SuperGraphNodeExecuted(SchedulerReportSuperGraphNodeExecuted {
                    attention: self.attention,
                    path: path.clone(),
                    start_instant,
                    end_instant,
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
            if let Some(settings) = &self.abbreviated_tensor_settings {
                let do_it = settings.do_all
                    || if let Some(x) = &self.abbreviated_tensor_subscribed_table {
                        x.contains(path)
                    } else {
                        false
                    };
                if do_it {
                    let report = SchedulerReport::SuperGraphTensorAssignedAbbreviated(
                        SchedulerReportSuperGraphTensorAssignedAbbreviated {
                            attention: self.attention,
                            path: path.clone(),
                            value: AbbreviatedTensorValue::from_tensor(
                                tensor,
                                settings.downsampled_size,
                                backend,
                            ),
                        },
                    );
                    reporter.push_report(report);
                }
            }
        }
    }
}

pub async fn scheduler(mut input: mpsc::Receiver<SchedulerJob>, model_server: Arc<ModelServer>) {
    #[cfg(feature = "vulkan")]
    let vulkan_runtime = {
        use whisper_tensor::backends::vulkan_backend::{VulkanContext, VulkanImmediateExecutor};
        let vulkan_context = VulkanContext::new().unwrap();
        Arc::new(Mutex::new(
            VulkanImmediateExecutor::new(vulkan_context).unwrap(),
        ))
    };
    let caches = Arc::new(Mutex::new(HashMap::new()));
    let vulkan_tensor_load_caches: Arc<Mutex<HashMap<LoadedModelId, ModelLoadedTensorCache>>> =
        Arc::new(Mutex::new(HashMap::new()));
    loop {
        if let Some(x) = input.recv().await {
            #[cfg(feature = "vulkan")]
            let vulkan_runtime = vulkan_runtime.clone();
            let caches = caches.clone();
            #[cfg(feature = "vulkan")]
            let vulkan_tensor_load_caches = vulkan_tensor_load_caches.clone();
            let ndarray_tensor_load_caches = Arc::new(Mutex::new(HashMap::new()));
            match x {
                SchedulerJob::SuperGraphRequest((req, resp_sender, reporter)) => {
                    // Collect links to needed models
                    let mut model_id_map = HashMap::new();
                    let models = {
                        let mut models = HashMap::new();
                        for (link, &model_id) in &req.model_inputs {
                            let model = model_server.get_model(model_id).await;
                            if let Some(model) = model {
                                models.insert(*link, model.clone());
                                model_id_map.insert(model_id, model);
                            }
                        }
                        models
                    };
                    // Dispatch tight loop
                    let result = tokio::task::spawn_blocking(move || {
                        let mut ndarray_backend = EvalBackend::NDArray;
                        #[cfg(feature = "vulkan")]
                        let mut vulkan_runtime = vulkan_runtime.lock().unwrap();
                        #[cfg(feature = "vulkan")]
                        let mut vulkan_backend = EvalBackend::Vulkan(&mut vulkan_runtime);
                        let backend: Result<&mut EvalBackend, String> = match req.backend_mode {
                            SuperGraphRequestBackendMode::NDArray => Ok(&mut ndarray_backend),
                            SuperGraphRequestBackendMode::Vulkan => {
                                #[cfg(feature = "vulkan")]
                                let res = Ok(&mut vulkan_backend);
                                #[cfg(not(feature = "vulkan"))]
                                let res = Err("Vulkan feature not enabled!".to_string());
                                res
                            }
                        };

                        match backend {
                            Ok(backend) => {
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
                                let mut observer = LocalSuperGraphObserver::new(
                                    req.attention_token,
                                    req.do_node_execution_reports,
                                    req.abbreviated_tensor_report_settings,
                                    reporter,
                                    req.subscribed_tensors.iter().cloned().collect(),
                                );
                                let mut caches = caches.lock().unwrap();
                                #[cfg(feature = "vulkan")]
                                let mut vulkan_tensor_load_caches =
                                    vulkan_tensor_load_caches.lock().unwrap();
                                let mut ndarray_tensor_load_caches =
                                    ndarray_tensor_load_caches.lock().unwrap();
                                // setup tensor caches
                                let res = {
                                    let mut super_graph_tensor_cache = {
                                        let mut res = SuperGraphTensorCache::new();
                                        match backend {
                                            EvalBackend::NDArray => {
                                                for (a, b) in &model_id_map {
                                                    if let Some(x) =
                                                        ndarray_tensor_load_caches.remove(a)
                                                    {
                                                        res.caches.push((b.as_ref(), x));
                                                    } else {
                                                        res.caches.push((
                                                            b.as_ref(),
                                                            ModelLoadedTensorCache::default(),
                                                        ));
                                                    }
                                                }
                                            }
                                            #[cfg(feature = "vulkan")]
                                            EvalBackend::Vulkan(_) => {
                                                for (a, b) in &model_id_map {
                                                    if let Some(x) =
                                                        vulkan_tensor_load_caches.remove(a)
                                                    {
                                                        res.caches.push((b.as_ref(), x));
                                                    } else {
                                                        res.caches.push((
                                                            b.as_ref(),
                                                            ModelLoadedTensorCache::default(),
                                                        ));
                                                    }
                                                }
                                            }
                                            _ => {
                                                // No caching
                                            }
                                        }
                                        res
                                    };
                                    let cache = req.use_cache.map(|x| {
                                        caches.entry(x).or_insert_with(SuperGraphCache::new)
                                    });
                                    let mut context = SuperGraphContext {
                                        observer: &mut observer,
                                        eval_backend: backend,
                                        caches: cache,
                                        super_graph_tensor_cache: &mut super_graph_tensor_cache,
                                    };
                                    let ret = req
                                        .super_graph
                                        .run(super_graph_data, &mut context)
                                        .map_err(|x| x.to_string())?;
                                    // Re-pack tensor caches
                                    match backend {
                                        EvalBackend::NDArray => {
                                            for (a, b) in super_graph_tensor_cache.caches {
                                                for (aa, bb) in &model_id_map {
                                                    if ptr::addr_eq(a, bb.as_ref()) {
                                                        ndarray_tensor_load_caches.insert(*aa, b);
                                                        break;
                                                    }
                                                }
                                            }
                                        }
                                        #[cfg(feature = "vulkan")]
                                        EvalBackend::Vulkan(_) => {
                                            for (a, b) in super_graph_tensor_cache.caches {
                                                for (aa, bb) in &model_id_map {
                                                    if ptr::addr_eq(a, bb.as_ref()) {
                                                        vulkan_tensor_load_caches.insert(*aa, b);
                                                        break;
                                                    }
                                                }
                                            }
                                        }
                                        _ => {
                                            // No caching
                                        }
                                    }
                                    ret
                                };

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

                                Ok(SuperGraphResponseData {
                                    tensor_outputs,
                                    string_outputs: strings,
                                    hash_outputs: hashes,
                                })
                            }
                            Err(e) => Err(e),
                        }
                    })
                    .await
                    .unwrap();

                    let resp = SuperGraphResponse {
                        attention_token: req.attention_token,
                        result,
                    };

                    if let Err(e) = resp_sender.send(resp).await {
                        error!("Failed to send response: {e}");
                    }
                }
            }
        }
    }
}
