use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use memmap2::Mmap;
use prost::{DecodeError, Message};
use onnx_graph::onnx::{ModelProto, StringStringEntryProto};
use onnx_graph::weights::SafetensorsWeightManager;
use onnx_graph::{InputMetadata, ModelInputType, ModelMetadata, ModelOutputType, OutputMetadata, TokenizerInfo, WeightStorageStrategy};

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
    UnsupportedConfigurationError(String, String)
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize, strum_macros::EnumIter, strum_macros::Display)]
pub enum ModelTypeHint {
    GPT2,
    RWKV7
}

pub fn identify_and_load(model_path: &Path, output_method: WeightStorageStrategy, hint: Option<ModelTypeHint>) -> Result<Vec::<u8>, Error> {
    if model_path.is_dir() {
        let config_path = model_path.join("config.json");
        if config_path.exists() {
            Ok(load_transformers_format(model_path, output_method, hint)?)
        }
        else {
            Err(Error::CannotIdentifyModel(model_path.to_path_buf()))
        }
    }
    else if let Some(ext) = model_path.extension() {
        if ext == "pth" {
            if let Some(ModelTypeHint::RWKV7) = hint {
                Ok(rwkv7::load_rwkv7_pth(model_path, output_method).map_err(|x| Error::ModelBuildError(x))?)
            }
            else {
                Err(Error::CannotIdentifyModel(model_path.to_path_buf()))
            }
        }
        else if ext == "onnx" {
            Ok(load_onnx_file(model_path, hint)?)
        }
        else {
            Err(Error::CannotIdentifyModel(model_path.to_path_buf()))
        }
    }
    else {
        Err(Error::CannotIdentifyModel(model_path.to_path_buf()))
    }
}

fn load_onnx_file(model_path: &Path, hint: Option<ModelTypeHint>) -> Result<Vec::<u8>, Error> {
    let mut onnx_data = Vec::new();
    File::open(model_path).unwrap().read_to_end(&mut onnx_data)?;
    
    if let Some(hint) = hint {
        if let ModelTypeHint::GPT2 = hint {
            onnx_data = try_inject_onnx_metadata_for_simple_llm(onnx_data, TokenizerInfo::HFTokenizer("gpt2".to_string()))?;
        }
    }
    
    Ok(onnx_data)
}

fn try_inject_onnx_metadata_for_simple_llm(mut onnx_data: Vec<u8>, tokenizer_info: TokenizerInfo) -> Result<Vec::<u8>, Error> {
    let mut do_reencode = false;
    let mut model_proto = ModelProto::decode(onnx_data.as_slice())?;
    let metadata = ModelMetadata{
        tokenizer_infos: vec![tokenizer_info],
        max_token_batch: None
    };
    model_proto.metadata_props.push(StringStringEntryProto{
        key: "whisper_tensor_metadata".to_string(),
        value: serde_json::to_string(&metadata).unwrap()
    });
    do_reencode = true;
    if let Some(graph) = &mut model_proto.graph {
        let mut token_input = None;
        if graph.input.len() == 1 {
            token_input = Some(&mut graph.input[0]);
        }
        else {
            // Maybe try to find it with some heuristic?
        }
        if let Some(input) = token_input {
            let meta = InputMetadata{
                model_input_type: ModelInputType::TokenID(0)
            };
            input.metadata_props.push(StringStringEntryProto{
                key: "whisper_tensor_metadata".to_string(),
                value: serde_json::to_string(&meta).unwrap()});
            do_reencode = true;
        }

        let mut token_output = None;
        if graph.output.len() == 1 {
            token_output = Some(&mut graph.input[0]);
        }
        else if graph.output.len() > 0 {
            // Use the first one for now
            token_output = Some(&mut graph.output[0])
        } else {
            // Can't find
        }
        if let Some(output) = token_output {
            let meta = OutputMetadata{
                model_output_type: ModelOutputType::TokenID(0)
            };
            output.metadata_props.push(StringStringEntryProto{
                key: "whisper_tensor_metadata".to_string(),
                value: serde_json::to_string(&meta).unwrap()});
            do_reencode = true;
        }
    }

    // Re-serialize
    if do_reencode {
        onnx_data = model_proto.encode_to_vec();
    }
    Ok(onnx_data)
}

fn load_transformers_format(model_path: &Path, output_method: WeightStorageStrategy, _hint: Option<ModelTypeHint>) -> Result<Vec::<u8>, Error> {
    println!("Loading hf transformers weights from {}", model_path.display());
    
    let config_path = model_path.join("config.json");
    let config_file = File::open(config_path)?;
    let config: serde_json::Value = serde_json::from_reader(config_file).map_err(|x| Error::ConfigFileParseError(x))?;

    // Load all safetensors files present
    let safetensors_files = {
        // Get all .safetensors files in that dir
        let mut safetensors_files = vec![];
        for entry in std::fs::read_dir(model_path)? {
            let entry = entry.map_err(|x| Error::ModelLoadError(anyhow::Error::from(x)))?;
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
    
    let weight_manager = SafetensorsWeightManager::new(safetensors_mmaps).map_err(|x| anyhow::Error::from(x)).map_err(|x| Error::ModelLoadError(x))?;

    let model_type = config.get("model_type").ok_or(Error::MissingConfigEntryError("model_type".to_string()))?.as_str().ok_or(Error::MissingConfigEntryError("model_type".to_string()))?;
    Ok(match model_type {
        "llama" => {
            println!("Loading as Llama3");
            let config = llama3::Llama3Config::from_huggingface_transformers_json(&config)?;
            llama3::load_llama3(weight_manager, config, output_method).map_err(|x| Error::ModelBuildError(x))?
        }
        "llama4" => {
            println!("Loading as Llama4");
            let config = llama4::Llama4Config::from_huggingface_transformers_json(&config)?;
            llama4::load_llama4(weight_manager, config, output_method).map_err(|x| Error::ModelBuildError(x))?
        }
        model_type => Err(Error::UnknownModelType(model_type.to_string()))?
    })
}