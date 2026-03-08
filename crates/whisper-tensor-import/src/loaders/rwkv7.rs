use super::{default_storage, onnx_bytes_to_output};
use whisper_tensor::loader::*;

/// Loader for RWKV7 .pth model files.
pub struct Rwkv7Loader;

impl Loader for Rwkv7Loader {
    fn name(&self) -> &str {
        "RWKV7"
    }

    fn description(&self) -> &str {
        "Load an RWKV7 language model from a .pth file"
    }

    fn config_schema(&self) -> Vec<ConfigField> {
        vec![ConfigField {
            key: "path".to_string(),
            label: "Model Path".to_string(),
            description: "Path to the RWKV7 .pth file".to_string(),
            field_type: ConfigFieldType::FilePath,
            required: true,
            default: None,
        }]
    }

    fn load(&self, config: ConfigValues) -> Result<LoaderOutput, LoaderError> {
        let path = require_path(&config, "path")?;
        let storage = default_storage();

        let onnx_data =
            crate::rwkv7::load_rwkv7_pth(&path, storage).map_err(LoaderError::LoadFailed)?;

        let model_name = path
            .file_stem()
            .unwrap_or_default()
            .to_str()
            .unwrap_or("rwkv7")
            .to_string();

        onnx_bytes_to_output(&onnx_data, &model_name, path.parent())
    }
}
