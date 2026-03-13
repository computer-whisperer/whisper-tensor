use memmap2::Mmap;
use onnx_graph::WeightStorageStrategy;
use onnx_graph::weights::SafetensorsWeightManager;
use prost::DecodeError;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub mod deepseek_v2;
pub mod flux;
pub mod gguf;
pub mod llama3;
pub mod loaders;
pub mod onnx_graph;
pub mod phi3;
pub mod qwen2;
pub mod rwkv7;
pub mod sd15;
pub mod sd2;
pub mod sd_common;
pub mod sd_xl;
pub mod t5;
pub mod whisper;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Cannot identify model: {0}")]
    CannotIdentifyModel(PathBuf),
    #[error("Unknown model type: {0}")]
    UnknownModelType(String),
    #[error("Missing config entry: {0}")]
    MissingConfigEntryError(String),
    #[error("Config file read error: {0}")]
    ConfigFileReadError(#[from] std::io::Error),
    #[error("Config file parse error: {0}")]
    ConfigFileParseError(serde_json::Error),
    #[error("Model load error: {0}")]
    ModelLoadError(anyhow::Error),
    #[error("Model build error: {0}")]
    ModelBuildError(anyhow::Error),
    #[error("Model decode error: {0}")]
    ModelDecodeError(#[from] DecodeError),
    #[error("Unsupported configuration: {0} {1}")]
    UnsupportedConfigurationError(String, String),
}

pub fn identify_and_load(
    model_path: &Path,
    output_method: WeightStorageStrategy,
) -> Result<Vec<u8>, Error> {
    if model_path.is_dir() {
        let config_path = model_path.join("config.json");
        if config_path.exists() {
            Ok(load_transformers_format(model_path, output_method)?)
        } else {
            Err(Error::CannotIdentifyModel(model_path.to_path_buf()))
        }
    } else if let Some(ext) = model_path.extension() {
        if ext == "pth" {
            Ok(rwkv7::load_rwkv7_pth(model_path, output_method).map_err(Error::ModelBuildError)?)
        } else if ext == "onnx" {
            Ok(load_onnx_file(model_path)?)
        } else {
            Err(Error::CannotIdentifyModel(model_path.to_path_buf()))
        }
    } else {
        Err(Error::CannotIdentifyModel(model_path.to_path_buf()))
    }
}

pub fn load_onnx_file(model_path: &Path) -> Result<Vec<u8>, Error> {
    let mut onnx_data = Vec::new();
    File::open(model_path)?.read_to_end(&mut onnx_data)?;
    Ok(onnx_data)
}

pub fn load_transformers_format(
    model_path: &Path,
    output_method: WeightStorageStrategy,
) -> Result<Vec<u8>, Error> {
    println!(
        "Loading hf transformers weights from {}",
        model_path.display()
    );

    let config_path = model_path.join("config.json");
    let config_file = File::open(config_path)?;
    let config: serde_json::Value =
        serde_json::from_reader(config_file).map_err(Error::ConfigFileParseError)?;

    // Load all safetensors files present
    let safetensors_files = {
        // Get all .safetensors files in that dir
        let mut safetensors_files = vec![];
        for entry in std::fs::read_dir(model_path)? {
            let entry = entry.map_err(|x| Error::ModelLoadError(anyhow::Error::from(x)))?;
            let path = entry.path();
            if path.is_file() && path.extension().is_some_and(|ext| ext == "safetensors") {
                safetensors_files.push(path);
            }
        }
        safetensors_files
    };

    let safetensors_paths: Vec<std::path::PathBuf> = safetensors_files
        .iter()
        .map(|p| std::fs::canonicalize(p).unwrap_or_else(|_| p.clone()))
        .collect();

    let safetensors_mmaps = {
        let mut safetensors_mmaps = vec![];
        for safetensors_file in safetensors_files {
            let safetensors_file = File::open(safetensors_file)?;
            let mmap = unsafe { Mmap::map(&safetensors_file) }?;
            safetensors_mmaps.push(Arc::new(mmap));
        }
        safetensors_mmaps
    };

    let weight_manager =
        SafetensorsWeightManager::new_with_paths(safetensors_mmaps, safetensors_paths)
            .map_err(anyhow::Error::from)
            .map_err(Error::ModelLoadError)?;

    let model_type = config
        .get("model_type")
        .ok_or(Error::MissingConfigEntryError("model_type".to_string()))?
        .as_str()
        .ok_or(Error::MissingConfigEntryError("model_type".to_string()))?;
    Ok(match model_type {
        "llama" | "mistral" => {
            println!(
                "Loading as {}",
                if model_type == "mistral" {
                    "Mistral"
                } else {
                    "Llama3"
                }
            );
            let config = llama3::Llama3Config::from_huggingface_transformers_json(&config)?;
            llama3::load_llama3(weight_manager, config, output_method)
                .map_err(Error::ModelBuildError)?
        }
        "qwen2" | "qwen3" => {
            println!("Loading as {model_type}");
            let config = qwen2::Qwen2Config::from_huggingface_transformers_json(&config)?;
            qwen2::load_qwen2(weight_manager, config, output_method)
                .map_err(Error::ModelBuildError)?
        }
        "phi3" => {
            println!("Loading as Phi-3");
            let config = phi3::Phi3Config::from_huggingface_transformers_json(&config)?;
            phi3::load_phi3(weight_manager, config, output_method)
                .map_err(Error::ModelBuildError)?
        }
        "deepseek_v2" => {
            println!("Loading as DeepSeek-V2");
            let config =
                deepseek_v2::DeepseekV2Config::from_huggingface_transformers_json(&config)?;
            deepseek_v2::load_deepseek_v2(weight_manager, config, output_method)
                .map_err(Error::ModelBuildError)?
        }
        model_type => Err(Error::UnknownModelType(model_type.to_string()))?,
    })
}
