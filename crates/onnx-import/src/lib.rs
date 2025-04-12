use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use memmap2::Mmap;
use onnx_graph::weights::SafetensorsWeightManager;
use onnx_graph::WeightStorageStrategy;

pub mod rwkv7;
pub mod llama3;
mod llama4;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Cannot identify model: {0}")]
    CannotIdentifyModel(PathBuf),
    #[error("Unknown model type: {0}")]
    UnknownModelType(String),
    #[error("Missing config entry: {0}")]
    MissingConfigEntryError(String),
    #[error("Model load error: {0}")]
    ModelLoadError(#[from] anyhow::Error)
}

pub fn identify_and_load(model_path: &Path, output_method: WeightStorageStrategy) -> Result<Vec::<u8>, Error> {
    if model_path.is_dir() {
        let config_path = model_path.join("config.json");
        if config_path.exists() {
            Ok(load_transformers_format(model_path, output_method)?)
        }
        else {
            Err(Error::CannotIdentifyModel(model_path.to_path_buf()))
        }
    }
    else if let Some(ext) = model_path.extension() {
        if ext == "pth" {
            Ok(rwkv7::load_rwkv7_pth(model_path, output_method)?)
        }
        else {
            Err(Error::CannotIdentifyModel(model_path.to_path_buf()))
        }
    }
    else {
        Err(Error::CannotIdentifyModel(model_path.to_path_buf()))
    }
}

fn load_transformers_format(model_path: &Path, output_method: WeightStorageStrategy) -> Result<Vec::<u8>, anyhow::Error> {
    println!("Loading hf transformers weights from {}", model_path.display());
    
    let config_path = model_path.join("config.json");
    let config_file = File::open(config_path)?;
    let config: serde_json::Value = serde_json::from_reader(config_file)?;

    // Load all safetensors files present
    let safetensors_files = {
        // Get all .safetensors files in that dir
        let mut safetensors_files = vec![];
        for entry in std::fs::read_dir(model_path)? {
            let entry = entry.map_err(|x| anyhow::Error::from(x))?;
            let path = entry.path();
            if path.is_file() && path.extension().map_or(false, |ext| ext == "safetensors") {
                safetensors_files.push(path);
            }
        }
        safetensors_files
    };

    let safetensors_mmaps = {
        let mut safetensors_mmaps = vec![];
        for safetensors_file in safetensors_files {
            let safetensors_file = File::open(safetensors_file)?;
            let mmap = unsafe{ Mmap::map(&safetensors_file)}?;
            safetensors_mmaps.push(Arc::new(mmap));
        }
        safetensors_mmaps
    };
    
    let weight_manager = SafetensorsWeightManager::new(safetensors_mmaps).map_err(|x| anyhow::Error::from(x))?;

    let model_type = config.get("model_type").ok_or(Error::MissingConfigEntryError("model_type".to_string()))?.as_str().ok_or(Error::MissingConfigEntryError("model_type".to_string()))?;
    Ok(match model_type {
        "llama" => {
            println!("Loading as Llama3");
            let config = llama3::Llama3Config::from_huggingface_transformers_json(&config)?;
            llama3::load_llama3(weight_manager, config, output_method)?
        }
        "llama4" => {
            println!("Loading as Llama4");
            let config = llama4::Llama4Config::from_huggingface_transformers_json(&config)?;
            llama4::load_llama4(weight_manager, config, output_method)?
        }
        model_type => Err(Error::UnknownModelType(model_type.to_string()))?
    })
}