use super::{OnnxLoader, Rwkv7Loader, SD15Loader, TransformersLoader};
use crate::onnx_graph::weights::SafetensorsWeightManager;
use memmap2::Mmap;
use std::sync::Arc;
use whisper_tensor::loader::*;

/// Auto-detecting loader that probes the path and delegates to the right specific loader.
pub struct AutoLoader;

impl Loader for AutoLoader {
    fn name(&self) -> &str {
        "Auto"
    }

    fn description(&self) -> &str {
        "Automatically detect model format and load"
    }

    fn config_schema(&self) -> Vec<ConfigField> {
        vec![ConfigField {
            key: "path".to_string(),
            label: "Model Path".to_string(),
            description: "Path to a model file or directory".to_string(),
            field_type: ConfigFieldType::FilePath,
            required: true,
            default: None,
        }]
    }

    fn load(&self, config: ConfigValues) -> Result<LoaderOutput, LoaderError> {
        let path = require_path(&config, "path")?;

        if path.is_dir() {
            let config_path = path.join("config.json");
            if config_path.exists() {
                return TransformersLoader.load(config);
            }
            // Check for ONNX SD directory format (text_encoder/model.onnx, etc.)
            let sd_dir = path.join("text_encoder").join("model.onnx");
            if sd_dir.exists() {
                // SD ONNX directory — not yet a dedicated loader, fall through
            }
            return Err(LoaderError::CannotIdentify(path));
        }

        if let Some(ext) = path.extension() {
            match ext.to_str().unwrap_or("") {
                "onnx" => return OnnxLoader.load(config),
                "pth" => return Rwkv7Loader.load(config),
                "safetensors" => {
                    // Probe if it's an SD 1.5 checkpoint
                    if let Ok(file) = std::fs::File::open(&path)
                        && let Ok(mmap) = unsafe { Mmap::map(&file) }
                        && let Ok(wm) = SafetensorsWeightManager::new(vec![Arc::new(mmap)])
                        && crate::sd15::is_sd15_checkpoint(&wm)
                    {
                        return SD15Loader.load(config);
                    }
                    // Not SD — could be a single safetensors model, but we don't have
                    // a standalone safetensors loader yet. Fall through to error.
                }
                _ => {}
            }
        }

        Err(LoaderError::CannotIdentify(path))
    }
}
