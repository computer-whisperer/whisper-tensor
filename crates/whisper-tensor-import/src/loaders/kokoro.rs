use whisper_tensor::loader::*;

/// Loader for Kokoro TTS models (ONNX format).
///
/// Expects a directory containing:
/// - `onnx/model.onnx` (or other quantized variants)
/// - `tokenizer.json` (phoneme tokenizer)
/// - `voices/` directory with `.bin` voice embedding files
pub struct KokoroLoader;

impl Loader for KokoroLoader {
    fn name(&self) -> &str {
        "Kokoro TTS"
    }

    fn description(&self) -> &str {
        "Load a Kokoro text-to-speech model"
    }

    fn config_schema(&self) -> Vec<ConfigField> {
        vec![
            ConfigField {
                key: "path".to_string(),
                label: "Model Directory".to_string(),
                description:
                    "Path to the Kokoro model directory (containing onnx/, tokenizer.json, voices/)"
                        .to_string(),
                field_type: ConfigFieldType::FilePath,
                required: true,
                default: None,
            },
            ConfigField {
                key: "variant".to_string(),
                label: "Model Variant".to_string(),
                description: "ONNX model variant to load".to_string(),
                field_type: ConfigFieldType::Enum {
                    options: vec![
                        "model".to_string(),
                        "model_fp16".to_string(),
                        "model_quantized".to_string(),
                        "model_q8f16".to_string(),
                    ],
                },
                required: false,
                default: Some(ConfigValue::String("model".to_string())),
            },
        ]
    }

    fn load(&self, config: ConfigValues) -> Result<LoaderOutput, LoaderError> {
        let dir = require_path(&config, "path")?;
        let variant = get_string(&config, "variant")?.unwrap_or_else(|| "model".to_string());
        crate::models::speech::kokoro::load_kokoro(&dir, &variant)
    }
}
