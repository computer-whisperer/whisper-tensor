use super::onnx_bytes_to_output;
use whisper_tensor::loader::*;

/// Loader for raw ONNX model files.
pub struct OnnxLoader;

impl Loader for OnnxLoader {
    fn name(&self) -> &str {
        "ONNX"
    }

    fn description(&self) -> &str {
        "Load a raw ONNX model file"
    }

    fn config_schema(&self) -> Vec<ConfigField> {
        vec![
            ConfigField {
                key: "path".to_string(),
                label: "Model Path".to_string(),
                description: "Path to the .onnx file".to_string(),
                field_type: ConfigFieldType::FilePath,
                required: true,
                default: None,
            },
            ConfigField {
                key: "model_type".to_string(),
                label: "Model Type Hint".to_string(),
                description: "Optional hint for metadata injection (e.g. GPT2)".to_string(),
                field_type: ConfigFieldType::Enum {
                    options: vec!["None".to_string(), "GPT2".to_string()],
                },
                required: false,
                default: Some(ConfigValue::String("None".to_string())),
            },
        ]
    }

    fn load(&self, config: ConfigValues) -> Result<LoaderOutput, LoaderError> {
        let path = require_path(&config, "path")?;
        let hint_str = get_string(&config, "model_type")?.unwrap_or_default();
        let hint = match hint_str.as_str() {
            "GPT2" => Some(crate::ModelTypeHint::GPT2),
            _ => None,
        };

        let onnx_data =
            crate::load_onnx_file(&path, hint).map_err(|e| LoaderError::LoadFailed(e.into()))?;

        let model_name = path
            .file_stem()
            .unwrap_or_default()
            .to_str()
            .unwrap_or("model")
            .to_string();

        onnx_bytes_to_output(&onnx_data, &model_name, path.parent())
    }
}
