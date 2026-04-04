use std::sync::Arc;
use std::sync::atomic::AtomicU32;
use tokio::sync::{RwLock, watch};
use whisper_tensor::loader::{ConfigValues, Loader};
use whisper_tensor::model::Model;
use whisper_tensor::numeric_tensor::NumericTensor;
use whisper_tensor::pool::SystemPool;
use whisper_tensor::symbolic_graph::tensor_store::TensorStoreTensorId;
use whisper_tensor::tensor_rank::DynRank;

use crate::{
    CurrentInterfacesReportEntry, CurrentModelsAndInterfacesReport, CurrentModelsReportEntry,
    LoadedModelId, LoaderRegistryEntry, LoaderRegistryReport,
};

pub struct ModelData {
    pub model: Arc<Model>,
    pub model_id: LoadedModelId,
    pub model_name: String,
    pub model_compiled: bool,
}

pub struct ModelServer {
    models: RwLock<Vec<ModelData>>,
    interfaces: RwLock<Vec<CurrentInterfacesReportEntry>>,
    loaders: Vec<Box<dyn Loader>>,
    loader_registry_report: LoaderRegistryReport,
    next_model_id: AtomicU32,
    models_report_watch_sender: watch::Sender<CurrentModelsAndInterfacesReport>,
    models_report_watch_receiver: watch::Receiver<CurrentModelsAndInterfacesReport>,
}

impl ModelServer {
    pub fn new(loaders: Vec<Box<dyn Loader>>) -> Self {
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

    pub fn get_loader_registry_report(&self) -> &LoaderRegistryReport {
        &self.loader_registry_report
    }

    pub fn watch_models_report(&self) -> watch::Receiver<CurrentModelsAndInterfacesReport> {
        self.models_report_watch_receiver.clone()
    }

    pub async fn generate_new_model_report(&self) {
        let guard = self.models.read().await;
        let mut new_report = CurrentModelsAndInterfacesReport::default();

        for model in guard.iter() {
            new_report.models.push(CurrentModelsReportEntry {
                model_id: model.model_id,
                model_name: model.model_name.clone(),
                num_ops: model.model.get_symbolic_graph().get_operations().len() as u64,
                model_compiled: model.model_compiled,
            });
        }

        let ifaces = self.interfaces.read().await;
        for entry in ifaces.iter() {
            new_report.interfaces.push(entry.clone());
        }

        self.models_report_watch_sender.send(new_report).unwrap()
    }

    pub async fn run_loader(
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
                    model_compiled: false,
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

    pub async fn unload_model(&self, model_id: LoadedModelId) -> Result<(), anyhow::Error> {
        let mut guard = self.models.write().await;
        guard.retain(|model| model.model_id != model_id);
        drop(guard);
        let mut ifaces = self.interfaces.write().await;
        ifaces.retain(|entry| !entry.model_ids.contains(&model_id));
        drop(ifaces);
        self.generate_new_model_report().await;
        Ok(())
    }

    pub async fn get_model(&self, model_id: LoadedModelId) -> Option<Arc<Model>> {
        let guard = self.models.read().await;
        guard
            .iter()
            .find(|model| model.model_id == model_id)
            .map(|model| model.model.clone())
    }

    pub async fn with_model<T>(
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

    pub async fn get_stored_tensor_id(
        &self,
        model_id: LoadedModelId,
        stored_tensor_id: TensorStoreTensorId,
    ) -> Result<NumericTensor<'static, DynRank, SystemPool>, String> {
        let guard = self.models.read().await;
        if let Some(model) = guard.iter().find(|model| model.model_id == model_id) {
            model
                .model
                .get_tensor_store()
                .get_tensor(stored_tensor_id)
                .and_then(|x| x.to_pool_tensor(&SystemPool))
                .ok_or("Tensor not found in Tensor Store".to_string())
        } else {
            Err(format!("Model with id {model_id} not found"))
        }
    }
}

#[cfg(feature = "import")]
pub fn default_loaders() -> Vec<Box<dyn Loader>> {
    use whisper_tensor_import::loaders::{
        AllegroLoader, AutoLoader, CogVideoXLoader, F5TtsLoader, FluxLoader, GgufLoader,
        HunyuanVideoLoader, KokoroLoader, LtxVideoLoader, MochiLoader, OnnxLoader, PiperLoader,
        Rwkv7Loader, SD2Loader, SD15Loader, SD35Loader, SDXLLoader, TransformersLoader, WanLoader,
        WhisperLoader,
    };
    vec![
        Box::new(AutoLoader),
        Box::new(OnnxLoader),
        Box::new(TransformersLoader),
        Box::new(GgufLoader),
        Box::new(Rwkv7Loader),
        Box::new(WhisperLoader),
        Box::new(SD15Loader),
        Box::new(SD2Loader),
        Box::new(SDXLLoader),
        Box::new(SD35Loader),
        Box::new(FluxLoader),
        Box::new(CogVideoXLoader),
        Box::new(WanLoader),
        Box::new(LtxVideoLoader),
        Box::new(MochiLoader),
        Box::new(HunyuanVideoLoader),
        Box::new(AllegroLoader),
        Box::new(KokoroLoader),
        Box::new(PiperLoader),
        Box::new(F5TtsLoader),
    ]
}

#[cfg(not(feature = "import"))]
pub fn default_loaders() -> Vec<Box<dyn Loader>> {
    vec![]
}
