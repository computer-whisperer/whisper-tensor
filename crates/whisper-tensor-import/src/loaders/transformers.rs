use super::{default_storage, onnx_bytes_to_output};
use whisper_tensor::loader::*;

/// Loader for HuggingFace Transformers format (directory with config.json + safetensors).
pub struct TransformersLoader;

impl Loader for TransformersLoader {
    fn name(&self) -> &str {
        "HuggingFace Transformers"
    }

    fn description(&self) -> &str {
        "Load a model from a HuggingFace Transformers directory (config.json + safetensors)"
    }

    fn config_schema(&self) -> Vec<ConfigField> {
        vec![ConfigField {
            key: "path".to_string(),
            label: "Model Directory".to_string(),
            description: "Path to directory containing config.json and .safetensors files"
                .to_string(),
            field_type: ConfigFieldType::FilePath,
            required: true,
            default: None,
        }]
    }

    fn load(&self, config: ConfigValues) -> Result<LoaderOutput, LoaderError> {
        let path = require_path(&config, "path")?;
        let storage = default_storage();

        let onnx_data = crate::load_transformers_format(&path, storage, None)
            .map_err(|e| LoaderError::LoadFailed(e.into()))?;

        let model_name = path
            .file_name()
            .unwrap_or_default()
            .to_str()
            .unwrap_or("model")
            .to_string();

        onnx_bytes_to_output(&onnx_data, &model_name, Some(&path))
    }
}
