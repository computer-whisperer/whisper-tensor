use whisper_tensor::loader::*;

/// Loader for Piper VITS TTS models (ONNX format).
///
/// Expects either:
/// - A single `.onnx` file (with `.onnx.json` config alongside it), or
/// - A directory containing the `.onnx` and `.onnx.json` files.
pub struct PiperLoader;

impl Loader for PiperLoader {
    fn name(&self) -> &str {
        "Piper TTS"
    }

    fn description(&self) -> &str {
        "Load a Piper VITS text-to-speech model"
    }

    fn config_schema(&self) -> Vec<ConfigField> {
        vec![ConfigField {
            key: "path".to_string(),
            label: "Model Path".to_string(),
            description: "Path to the Piper .onnx file or directory containing it".to_string(),
            field_type: ConfigFieldType::FilePath,
            required: true,
            default: None,
        }]
    }

    fn load(&self, config: ConfigValues) -> Result<LoaderOutput, LoaderError> {
        let path = require_path(&config, "path")?;
        crate::models::speech::piper::load_piper(&path)
    }
}
