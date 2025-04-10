use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use memmap2::Mmap;
use onnx_graph::weights::SafetensorsWeightManager;
use crate::llama3::Llama3Config;

pub mod rwkv7;
pub mod llama3;

#[derive(Debug)]
pub enum Error {
    UnknownModel,
    ModelLoadError,
    OnnxGraphError(onnx_graph::Error),
    IoError(std::io::Error),
    ConfigParseError(serde_json::Error),
    SafeTensorError(safetensors::SafeTensorError),
    MissingConfigEntryError(String)
}

pub fn identify_and_load(model_path: &Path, bin_path: Option<&Path>) -> Result<Vec::<u8>, Error> {
    if model_path.is_dir() {
        let config_path = model_path.join("config.json");
        if config_path.exists() {
            load_transformers_format(model_path, bin_path)
        }
        else {
            Err(Error::UnknownModel)
        }
    }
    else if let Some(ext) = model_path.extension() {
        if ext == "pth" {
            rwkv7::load_rwkv7_pth(model_path, bin_path)
        }
        else {
            Err(Error::UnknownModel)
        }
    }
    else {
        Err(Error::UnknownModel)
    }
}

fn load_transformers_format(model_path: &Path, bin_path: Option<&Path>) -> Result<Vec::<u8>, Error> {
    println!("Loading hf transformers weights from {}", model_path.display());
    
    let config_path = model_path.join("config.json");
    let config_file = File::open(config_path).map_err(|x| Error::IoError(x))?;
    let config: serde_json::Value = serde_json::from_reader(config_file).map_err(|x| Error::ConfigParseError(x))?;

    // Load all safetensors files present
    let safetensors_files = {
        // Get all .safetensors files in that dir
        let mut safetensors_files = vec![];
        for entry in std::fs::read_dir(model_path).map_err(|x| Error::IoError(x))? {
            let entry = entry.map_err(|x| Error::IoError(x))?;
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
            let safetensors_file = File::open(safetensors_file).map_err(|x| Error::IoError(x))?;
            let mmap = unsafe{ Mmap::map(&safetensors_file)}.map_err(|x| Error::IoError(x))?;
            safetensors_mmaps.push(Arc::new(mmap));
        }
        safetensors_mmaps
    };

    /*
    let mut central_metadata: HashMap<String, String> = HashMap::new();

    let safetensors_metadata = {
        let mut out = vec![];
        for safetensors_mmap in &safetensors_mmaps {
            let metadata = SafeTensors::read_metadata(&safetensors_mmap).map_err(|x| Error::SafeTensorError(x))?;
            let data = metadata.1.metadata();
            if let Some(data) = data {
                for (k, v) in data {
                    central_metadata.insert(k.clone(), v.clone());
                }
            }
            out.push(metadata);
        }
        out
    };

    println!("Loaded {} files, here is the safetensors metadata:", safetensors_metadata.len());
    for (k, v) in central_metadata {
        println!("{}: {}", k, v);
    }
    */
    
    let weight_manager = SafetensorsWeightManager::new(safetensors_mmaps).map_err(|x| Error::OnnxGraphError(x))?;

    let model_type = config.get("model_type").ok_or(Error::UnknownModel)?.as_str().ok_or(Error::ModelLoadError)?;
    match model_type {
        "llama" => {
            println!("Loading as Llama3");
            let config = Llama3Config::from_huggingface_transformers_json(&config)?;
            llama3::load_llama3(weight_manager, config, bin_path).map_err(|x| Error::OnnxGraphError(x))
        }
        _ => Err(Error::UnknownModel)
    }


}