use crate::model_server::ModelServer;
use crate::{
    AbbreviatedTensorReportSettings, AbbreviatedTensorValue, LoadedModelId, SuperGraphRequest,
    SuperGraphRequestBackendMode, SuperGraphResponse, SuperGraphResponseData,
};
use crossbeam::queue::ArrayQueue;
use log::error;
use std::collections::{HashMap, HashSet};
use std::ptr;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokio::sync::{Notify, mpsc};
use whisper_tensor::backends::ModelLoadedTensorCache;
use whisper_tensor::backends::eval_backend::EvalBackend;
use whisper_tensor::backends::ndarray_backend::NDArrayNumericTensor;
use whisper_tensor::compiler::CompilationSubject;
use whisper_tensor::graph::GlobalId;
use whisper_tensor::numeric_tensor::NumericTensor;
use whisper_tensor::super_graph::SuperGraphContext;
use whisper_tensor::super_graph::cache::{SuperGraphCache, SuperGraphTensorCache};
use whisper_tensor::super_graph::data::SuperGraphData;
use whisper_tensor::super_graph::observer::SuperGraphObserver;
use whisper_tensor::{DynRank, compiler};

#[derive(Debug)]
pub struct SchedulerReportSuperGraphNodeExecuted {
    pub attention: Option<u64>,
    pub start_instant: Instant,
    pub end_instant: Instant,
    pub path: Vec<GlobalId>,
    pub op_kind: String,
}

#[derive(Debug)]
pub struct SchedulerReportSuperGraphTensorAssigned {
    pub attention: Option<u64>,
    pub path: Vec<GlobalId>,
    pub value: NDArrayNumericTensor<DynRank>,
}

#[derive(Debug)]
pub struct SchedulerReportSuperGraphTensorAssignedAbbreviated {
    pub attention: Option<u64>,
    pub path: Vec<GlobalId>,
    pub value: AbbreviatedTensorValue,
}

#[derive(Debug)]
pub struct SchedulerReportSuperGraphProgress {
    pub attention: Option<u64>,
    pub path: Vec<GlobalId>,
    pub tier: i64,
    pub numerator: f64,
    pub denominator: f64,
}

#[derive(Debug)]
pub struct SchedulerReportSuperGraphLoadingWeight {
    pub attention: Option<u64>,
    pub path: Vec<GlobalId>,
    pub weight_name: Option<String>,
    pub instant: Instant,
}

#[allow(clippy::enum_variant_names)]
#[derive(Debug)]
pub enum SchedulerReport {
    SuperGraphNodeExecuted(SchedulerReportSuperGraphNodeExecuted),
    SuperGraphTensorAssignedFull(SchedulerReportSuperGraphTensorAssigned),
    SuperGraphTensorAssignedAbbreviated(SchedulerReportSuperGraphTensorAssignedAbbreviated),
    SuperGraphProgress(SchedulerReportSuperGraphProgress),
    SuperGraphLoadingWeight(SchedulerReportSuperGraphLoadingWeight),
}

impl SchedulerReport {
    pub fn get_attention_token(&self) -> Option<u64> {
        match self {
            SchedulerReport::SuperGraphNodeExecuted(report) => report.attention,
            SchedulerReport::SuperGraphTensorAssignedFull(report) => report.attention,
            SchedulerReport::SuperGraphTensorAssignedAbbreviated(report) => report.attention,
            SchedulerReport::SuperGraphProgress(report) => report.attention,
            SchedulerReport::SuperGraphLoadingWeight(report) => report.attention,
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

#[allow(clippy::large_enum_variant)]
pub enum SchedulerJob {
    SuperGraphRequest(
        (
            SuperGraphRequest,
            mpsc::Sender<SuperGraphResponse>,
            Option<SchedulerReporter>,
        ),
    ),
    CompileModelRequest {
        model_id: LoadedModelId,
    },
}

#[derive(Clone, Debug)]
pub struct ObserverSettingsRegistryEntry {
    pub version: u64,
    pub do_node_execute_report: bool,
    pub subscribed_tensors: HashSet<Vec<GlobalId>>,
    pub abbreviated_tensor_settings: Option<AbbreviatedTensorReportSettings>,
    pub abbreviated_tensor_subscribed_table: Option<HashSet<Vec<GlobalId>>>,
}

impl ObserverSettingsRegistryEntry {
    fn new(
        version: u64,
        do_node_execute_report: bool,
        abbreviated_tensor_settings: Option<AbbreviatedTensorReportSettings>,
        subscribed_tensors: Vec<Vec<GlobalId>>,
    ) -> Self {
        let subscribed_tensors = HashSet::from_iter(subscribed_tensors);
        let abbreviated_tensor_subscribed_table = abbreviated_tensor_settings
            .as_ref()
            .map(|settings| settings.subscribed_tensors.iter().cloned().collect());
        Self {
            version,
            do_node_execute_report,
            subscribed_tensors,
            abbreviated_tensor_settings,
            abbreviated_tensor_subscribed_table,
        }
    }
}

pub type ObserverSettingsRegistry = Arc<Mutex<HashMap<u64, ObserverSettingsRegistryEntry>>>;

pub fn ensure_observer_settings(
    observer_settings_registry: &ObserverSettingsRegistry,
    attention_token: Option<u64>,
    do_node_execute_report: bool,
    abbreviated_tensor_settings: Option<AbbreviatedTensorReportSettings>,
    subscribed_tensors: Vec<Vec<GlobalId>>,
) {
    let Some(attention_token) = attention_token else {
        return;
    };
    let _ = observer_settings_registry.lock().map(|mut settings| {
        settings.entry(attention_token).or_insert_with(|| {
            ObserverSettingsRegistryEntry::new(
                0,
                do_node_execute_report,
                abbreviated_tensor_settings,
                subscribed_tensors,
            )
        });
    });
}

pub fn update_observer_settings(
    observer_settings_registry: &ObserverSettingsRegistry,
    attention_token: u64,
    do_node_execute_report: bool,
    abbreviated_tensor_settings: Option<AbbreviatedTensorReportSettings>,
    subscribed_tensors: Vec<Vec<GlobalId>>,
) {
    let _ = observer_settings_registry.lock().map(|mut settings| {
        let version = settings
            .get(&attention_token)
            .map_or(0, |entry| entry.version.saturating_add(1));
        settings.insert(
            attention_token,
            ObserverSettingsRegistryEntry::new(
                version,
                do_node_execute_report,
                abbreviated_tensor_settings,
                subscribed_tensors,
            ),
        );
    });
}

pub fn remove_observer_settings(
    observer_settings_registry: &ObserverSettingsRegistry,
    attention_token: Option<u64>,
) {
    if let Some(attention_token) = attention_token {
        let _ = observer_settings_registry
            .lock()
            .map(|mut settings| settings.remove(&attention_token));
    }
}

struct LocalSuperGraphObserver {
    attention: Option<u64>,
    do_node_execute_report: bool,
    reporter: Option<SchedulerReporter>,
    subscribed_tensors: HashSet<Vec<GlobalId>>,
    abbreviated_tensor_settings: Option<AbbreviatedTensorReportSettings>,
    abbreviated_tensor_subscribed_table: Option<HashSet<Vec<GlobalId>>>,
    cancellation_registry: Option<Arc<Mutex<HashSet<u64>>>>,
    observer_settings_registry: Option<ObserverSettingsRegistry>,
    observer_settings_version: u64,
    cancellation_seen: bool,
}

impl LocalSuperGraphObserver {
    pub fn new(
        attention: Option<u64>,
        do_node_execute_report: bool,
        abbreviated_tensor_settings: Option<AbbreviatedTensorReportSettings>,
        reporter: Option<SchedulerReporter>,
        subscribed_tensors: HashSet<Vec<GlobalId>>,
        cancellation_registry: Option<Arc<Mutex<HashSet<u64>>>>,
        observer_settings_registry: Option<ObserverSettingsRegistry>,
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
            cancellation_registry,
            observer_settings_registry,
            observer_settings_version: 0,
            cancellation_seen: false,
        }
    }

    fn refresh_dynamic_settings(&mut self) {
        let Some(attention) = self.attention else {
            return;
        };
        let Some(registry) = &self.observer_settings_registry else {
            return;
        };
        let entry = registry
            .lock()
            .ok()
            .and_then(|settings| settings.get(&attention).cloned());
        if let Some(entry) = entry
            && entry.version != self.observer_settings_version
        {
            self.observer_settings_version = entry.version;
            self.do_node_execute_report = entry.do_node_execute_report;
            self.subscribed_tensors = entry.subscribed_tensors;
            self.abbreviated_tensor_settings = entry.abbreviated_tensor_settings;
            self.abbreviated_tensor_subscribed_table = entry.abbreviated_tensor_subscribed_table;
        }
    }
}

impl SuperGraphObserver for LocalSuperGraphObserver {
    fn on_node_executed(
        &mut self,
        path: &[GlobalId],
        op_kind: &str,
        start_instant: Instant,
        end_instant: Instant,
        _backend: &mut EvalBackend,
    ) {
        self.refresh_dynamic_settings();
        if let Some(reporter) = &mut self.reporter
            && self.do_node_execute_report
        {
            let report =
                SchedulerReport::SuperGraphNodeExecuted(SchedulerReportSuperGraphNodeExecuted {
                    attention: self.attention,
                    path: path.to_vec(),
                    op_kind: op_kind.to_string(),
                    start_instant,
                    end_instant,
                });
            reporter.push_report(report);
        }
    }

    fn on_tensor_assigned(
        &mut self,
        path: &[GlobalId],
        tensor: &NumericTensor<DynRank>,
        backend: &mut EvalBackend,
    ) {
        self.refresh_dynamic_settings();
        if let Some(reporter) = &mut self.reporter {
            if self.subscribed_tensors.contains(path) {
                let report = SchedulerReport::SuperGraphTensorAssignedFull(
                    SchedulerReportSuperGraphTensorAssigned {
                        attention: self.attention,
                        path: path.to_vec(),
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
                            path: path.to_vec(),
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

    fn on_progress(&mut self, path: &[GlobalId], tier: i64, numerator: f64, denominator: f64) {
        if let Some(reporter) = &mut self.reporter {
            let report = SchedulerReport::SuperGraphProgress(SchedulerReportSuperGraphProgress {
                attention: self.attention,
                path: path.to_vec(),
                tier,
                numerator,
                denominator,
            });
            reporter.push_report(report);
        }
    }

    fn on_loading_weight(&mut self, path: &[GlobalId], weight_name: Option<String>) {
        if let Some(reporter) = &mut self.reporter {
            let report =
                SchedulerReport::SuperGraphLoadingWeight(SchedulerReportSuperGraphLoadingWeight {
                    attention: self.attention,
                    path: path.to_vec(),
                    weight_name,
                    instant: Instant::now(),
                });
            reporter.push_report(report);
        }
    }

    fn should_cancel(&mut self) -> bool {
        if self.cancellation_seen {
            return true;
        }
        let Some(attention) = self.attention else {
            return false;
        };
        let Some(registry) = &self.cancellation_registry else {
            return false;
        };
        let should_cancel = registry
            .lock()
            .map(|set| set.contains(&attention))
            .unwrap_or(false);
        if should_cancel {
            self.cancellation_seen = true;
        }
        should_cancel
    }
}

fn cancel_response(attention_token: Option<u64>) -> SuperGraphResponse {
    SuperGraphResponse {
        attention_token,
        result: Err("Execution cancelled".to_string()),
    }
}

fn is_attention_cancelled(
    cancellation_registry: &Arc<Mutex<HashSet<u64>>>,
    attention_token: Option<u64>,
) -> bool {
    let Some(attention_token) = attention_token else {
        return false;
    };
    cancellation_registry
        .lock()
        .map(|set| set.contains(&attention_token))
        .unwrap_or(false)
}

fn clear_attention_cancellation(
    cancellation_registry: &Arc<Mutex<HashSet<u64>>>,
    attention_token: Option<u64>,
) {
    if let Some(attention_token) = attention_token {
        let _ = cancellation_registry
            .lock()
            .map(|mut set| set.remove(&attention_token));
    }
}

pub async fn scheduler(
    mut input: mpsc::Receiver<SchedulerJob>,
    model_server: Arc<ModelServer>,
    cancellation_registry: Arc<Mutex<HashSet<u64>>>,
    observer_settings_registry: ObserverSettingsRegistry,
) {
    #[cfg(feature = "vulkan")]
    let vulkan_runtime = {
        use whisper_tensor::backends::vulkan_backend::{VulkanContext, VulkanImmediateExecutor};
        let vulkan_context = VulkanContext::new().unwrap();
        Arc::new(Mutex::new(
            VulkanImmediateExecutor::new(vulkan_context).unwrap(),
        ))
    };
    let caches = Arc::new(Mutex::new(HashMap::new()));
    #[cfg(feature = "vulkan")]
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
                SchedulerJob::CompileModelRequest { model_id } => {
                    let model = model_server.get_model(model_id).await;
                    if let Some(model) = model {
                        let symbolic_graph = Arc::new(model.get_symbolic_graph().clone());
                        let subject = CompilationSubject::SymbolicGraph { symbolic_graph };
                        let program = compiler::build_program(subject);
                        model_server
                            .set_compiled_model(model_id, Arc::new(program))
                            .await;
                    }
                }
                SchedulerJob::SuperGraphRequest((req, resp_sender, reporter)) => {
                    ensure_observer_settings(
                        &observer_settings_registry,
                        req.attention_token,
                        req.do_node_execution_reports,
                        req.abbreviated_tensor_report_settings.clone(),
                        req.subscribed_tensors.clone(),
                    );
                    if is_attention_cancelled(&cancellation_registry, req.attention_token) {
                        let _ = resp_sender.send(cancel_response(req.attention_token)).await;
                        clear_attention_cancellation(&cancellation_registry, req.attention_token);
                        remove_observer_settings(&observer_settings_registry, req.attention_token);
                        continue;
                    }
                    // Collect links to needed models
                    let mut model_id_map = HashMap::new();
                    let mut compiled_models = HashMap::new();
                    let cancellation_registry_for_request = cancellation_registry.clone();
                    let observer_settings_registry_for_request = observer_settings_registry.clone();
                    let (models, symbolic_graph_models) = {
                        let mut models = HashMap::new();
                        let mut symbolic_graph_models = Vec::new();
                        for (link, &model_id) in &req.model_inputs {
                            let model = model_server.get_model(model_id).await;
                            let compiled_model = model_server.get_compiled_model(model_id).await;
                            if let Some(model) = model {
                                models.insert(*link, model.clone());
                                model_id_map.insert(model_id, model);
                                compiled_models.insert(model_id, compiled_model);
                            }
                        }
                        for model_id in req.symbolic_graph_ids {
                            symbolic_graph_models
                                .push(model_server.get_model(model_id).await.unwrap());
                        }
                        (models, symbolic_graph_models)
                    };
                    // Dispatch tight loop
                    let result = tokio::task::spawn_blocking(move || {
                        let mut ndarray_backend = EvalBackend::NDArray;
                        #[cfg(feature = "vulkan")]
                        let mut vulkan_runtime = vulkan_runtime.lock().unwrap();
                        #[cfg(feature = "vulkan")]
                        let mut vulkan_backend = EvalBackend::Vulkan(&mut vulkan_runtime);
                        let mut use_compiler = false;
                        let backend: Result<&mut EvalBackend, String> = match req.backend_mode {
                            SuperGraphRequestBackendMode::NDArray => Ok(&mut ndarray_backend),
                            SuperGraphRequestBackendMode::Vulkan => {
                                #[cfg(feature = "vulkan")]
                                let res = Ok(&mut vulkan_backend);
                                #[cfg(not(feature = "vulkan"))]
                                let res = Err("Vulkan feature not enabled!".to_string());
                                res
                            }
                            SuperGraphRequestBackendMode::Compiler => {
                                use_compiler = true;
                                Ok(&mut ndarray_backend)
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
                                for (link, clip) in req.audio_inputs {
                                    super_graph_data.audio_clips.insert(
                                        link,
                                        whisper_tensor::super_graph::data::SuperGraphAudioClip::new(
                                            NumericTensor::from(clip.samples),
                                            clip.sample_rate_hz,
                                        ),
                                    );
                                }

                                // Populate data with refs
                                for link in req.model_inputs.keys() {
                                    if let Some(model) = models.get(link) {
                                        super_graph_data
                                            .tensor_maps
                                            .insert(*link, model.get_tensor_store());
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
                                    Some(cancellation_registry_for_request.clone()),
                                    Some(observer_settings_registry_for_request.clone()),
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
                                                        res.caches.push((b.get_tensor_store(), x));
                                                    } else {
                                                        res.caches.push((
                                                            b.get_tensor_store(),
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
                                                        res.caches.push((b.get_tensor_store(), x));
                                                    } else {
                                                        res.caches.push((
                                                            b.get_tensor_store(),
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
                                    let compiled_models = {
                                        let mut ret = vec![];
                                        for (a, b) in &compiled_models {
                                            if let Some(x) = b {
                                                ret.push((
                                                    model_id_map.get(a).unwrap().as_ref(),
                                                    x.as_ref(),
                                                ));
                                            }
                                        }
                                        ret
                                    };
                                    let symbolic_graph_refs = symbolic_graph_models
                                        .iter()
                                        .map(|x| x.get_symbolic_graph())
                                        .collect();
                                    let mut context = SuperGraphContext {
                                        observer: &mut observer,
                                        eval_backend: backend,
                                        caches: cache,
                                        use_compiled_models: use_compiler,
                                        symbolic_graphs: symbolic_graph_refs,
                                        compiled_models: Some(compiled_models),
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
                                    images,
                                    audio_clips,
                                    strings,
                                    hashes,
                                    ..
                                } = res;

                                let mut tensor_outputs = tensors
                                    .iter()
                                    .map(|(k, v)| (*k, v.to_ndarray().unwrap()))
                                    .collect::<HashMap<_, _>>();
                                for (link, image) in images {
                                    tensor_outputs.insert(link, image.tensor.to_ndarray().unwrap());
                                }
                                for (link, clip) in audio_clips {
                                    tensor_outputs.insert(link, clip.samples.to_ndarray().unwrap());
                                }

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
                    clear_attention_cancellation(&cancellation_registry, req.attention_token);
                    remove_observer_settings(&observer_settings_registry, req.attention_token);

                    if let Err(e) = resp_sender.send(resp).await {
                        error!("Failed to send response: {e}");
                    }
                }
            }
        }
    }
}
