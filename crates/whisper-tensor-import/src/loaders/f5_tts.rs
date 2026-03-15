use whisper_tensor::loader::*;

/// Loader for F5-TTS models (3-model ONNX pipeline).
///
/// Expects a directory containing:
/// - `F5_Preprocess.onnx`
/// - `F5_Transformer.onnx`
/// - `F5_Decode.onnx`
/// - `vocab.txt` (character-level vocabulary, one token per line)
pub struct F5TtsLoader;

impl Loader for F5TtsLoader {
    fn name(&self) -> &str {
        "F5-TTS"
    }

    fn description(&self) -> &str {
        "Load an F5-TTS text-to-speech model (3-part ONNX pipeline)"
    }

    fn config_schema(&self) -> Vec<ConfigField> {
        vec![ConfigField {
            key: "path".to_string(),
            label: "Model Directory".to_string(),
            description: "Path to the F5-TTS directory (containing F5_Preprocess.onnx, F5_Transformer.onnx, F5_Decode.onnx, vocab.txt)".to_string(),
            field_type: ConfigFieldType::FilePath,
            required: true,
            default: None,
        }]
    }

    fn load(&self, config: ConfigValues) -> Result<LoaderOutput, LoaderError> {
        let dir = require_path(&config, "path")?;
        crate::models::speech::f5_tts::load_f5_tts(&dir)
    }
}
