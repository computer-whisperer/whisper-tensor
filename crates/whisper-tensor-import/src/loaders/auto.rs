use super::{
    F5TtsLoader, FluxLoader, GgufLoader, KokoroLoader, OnnxLoader, PiperLoader, Rwkv7Loader,
    SD2Loader, SD15Loader, SD35Loader, SDXLLoader, TransformersLoader, WhisperLoader,
};
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
            // Check for Kokoro TTS (has onnx/ subdirectory + tokenizer.json)
            if path.join("onnx").is_dir() && path.join("tokenizer.json").exists() {
                return KokoroLoader.load(config);
            }
            // Check for Piper TTS (directory with .onnx + .onnx.json)
            if crate::models::speech::piper::is_piper_model(&path) {
                return PiperLoader.load(config);
            }
            // Check for F5-TTS (3-part ONNX pipeline + vocab).
            if path.join("F5_Preprocess.onnx").exists()
                && path.join("F5_Transformer.onnx").exists()
                && path.join("F5_Decode.onnx").exists()
                && path.join("vocab.txt").exists()
            {
                return F5TtsLoader.load(config);
            }
            // Check for Whisper STT (HF directory with model + tokenizer).
            if path.join("model.safetensors").exists()
                && path.join("config.json").exists()
                && path.join("tokenizer.json").exists()
            {
                return WhisperLoader.load(config);
            }
            let config_path = path.join("config.json");
            if config_path.exists() {
                return TransformersLoader.load(config);
            }
            // Check for SD3.x ONNX pipeline directory.
            let has_onnx_component = |name: &str| {
                let dir = path.join(name);
                if !dir.is_dir() {
                    return false;
                }
                std::fs::read_dir(&dir).ok().is_some_and(|entries| {
                    entries
                        .filter_map(Result::ok)
                        .any(|entry| entry.path().extension().is_some_and(|ext| ext == "onnx"))
                })
            };
            if has_onnx_component("text_encoder")
                && has_onnx_component("text_encoder_2")
                && has_onnx_component("text_encoder_3")
                && has_onnx_component("transformer")
                && (has_onnx_component("vae_decoder") || has_onnx_component("vae"))
            {
                return SD35Loader.load(config);
            }
            let has_safetensors_component = |name: &str| {
                let dir = path.join(name);
                if !dir.is_dir() {
                    return false;
                }
                std::fs::read_dir(&dir).ok().is_some_and(|entries| {
                    entries.filter_map(Result::ok).any(|entry| {
                        entry
                            .path()
                            .extension()
                            .is_some_and(|ext| ext == "safetensors")
                    })
                })
            };
            if has_safetensors_component("text_encoder")
                && has_safetensors_component("text_encoder_2")
                && has_safetensors_component("text_encoder_3")
                && has_safetensors_component("transformer")
                && has_safetensors_component("vae")
            {
                return SD35Loader.load(config);
            }
            return Err(LoaderError::CannotIdentify(path));
        }

        if let Some(ext) = path.extension() {
            match ext.to_str().unwrap_or("") {
                "onnx" => {
                    // Check for Piper (.onnx.json alongside)
                    if crate::models::speech::piper::is_piper_model(&path) {
                        return PiperLoader.load(config);
                    }
                    return OnnxLoader.load(config);
                }
                "gguf" => return GgufLoader.load(config),
                "pth" => return Rwkv7Loader.load(config),
                "safetensors" => {
                    // Probe safetensors checkpoints for known formats
                    if let Ok(file) = std::fs::File::open(&path)
                        && let Ok(mmap) = unsafe { Mmap::map(&file) }
                        && let Ok(wm) = SafetensorsWeightManager::new(vec![Arc::new(mmap)])
                    {
                        if super::flux::is_flux_single_file_checkpoint(&wm) {
                            return FluxLoader.load(config);
                        }
                        if crate::models::diffusion::sd15::is_sd15_checkpoint(&wm) {
                            return SD15Loader.load(config);
                        }
                        if crate::models::diffusion::sd2::is_sd2_checkpoint(&wm) {
                            return SD2Loader.load(config);
                        }
                        if crate::models::diffusion::sd_xl::is_sdxl_checkpoint(&wm) {
                            return SDXLLoader.load(config);
                        }
                    }
                    // Not a recognized format. Fall through to error.
                }
                _ => {}
            }
        }

        Err(LoaderError::CannotIdentify(path))
    }
}
