use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use memmap2::Mmap;
use onnx_graph::weights::SafetensorsWeightManager;
use onnx_graph::WeightStorageStrategy;

pub mod rwkv7;
pub mod llama3;
mod llama4;

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

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl core::error::Error for Error{
    fn cause(&self) -> Option<&dyn core::error::Error> {
        match self {
            Error::OnnxGraphError(e) => Some(e),
            Error::IoError(e) => Some(e),
            Error::ConfigParseError(e) => Some(e),
            Error::SafeTensorError(e) => Some(e),
            _ => None
        }
    }
}

pub fn identify_and_load(model_path: &Path, output_method: WeightStorageStrategy) -> Result<Vec::<u8>, Error> {
    if model_path.is_dir() {
        let config_path = model_path.join("config.json");
        if config_path.exists() {
            load_transformers_format(model_path, output_method)
        }
        else {
            Err(Error::UnknownModel)
        }
    }
    else if let Some(ext) = model_path.extension() {
        if ext == "pth" {
            rwkv7::load_rwkv7_pth(model_path, output_method)
        }
        else {
            Err(Error::UnknownModel)
        }
    }
    else {
        Err(Error::UnknownModel)
    }
}

fn load_transformers_format(model_path: &Path, output_method: WeightStorageStrategy) -> Result<Vec::<u8>, Error> {
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
    
    let weight_manager = SafetensorsWeightManager::new(safetensors_mmaps).map_err(|x| Error::OnnxGraphError(x))?;

    let model_type = config.get("model_type").ok_or(Error::UnknownModel)?.as_str().ok_or(Error::ModelLoadError)?;
    match model_type {
        "llama" => {
            println!("Loading as Llama3");
            let config = llama3::Llama3Config::from_huggingface_transformers_json(&config)?;
            llama3::load_llama3(weight_manager, config, output_method).map_err(|x| Error::OnnxGraphError(x))
        }
        "llama4" => {
            println!("Loading as Llama4");
            let config = llama4::Llama4Config::from_huggingface_transformers_json(&config)?;
            llama4::load_llama4(weight_manager, config, output_method).map_err(|x| Error::OnnxGraphError(x))
        }
        _ => Err(Error::UnknownModel)
    }


}