use super::onnx_bytes_to_model;
use whisper_tensor::interfaces::{TTSInputConfig, TextToSpeechInterface};
use whisper_tensor::loader::*;
use whisper_tensor::metadata::TokenizerInfo;
use whisper_tensor::super_graph::SuperGraphBuilder;
use whisper_tensor::super_graph::links::SuperGraphLink;
use whisper_tensor::super_graph::nodes::{SuperGraphNode, SuperGraphNodeModelExecution};

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
                description: "Path to the Kokoro model directory (containing onnx/, tokenizer.json, voices/)".to_string(),
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
        let variant = get_string(&config, "variant")?
            .unwrap_or_else(|| "model".to_string());

        // Load the ONNX model
        let onnx_path = dir.join("onnx").join(format!("{variant}.onnx"));
        if !onnx_path.exists() {
            return Err(LoaderError::LoadFailed(anyhow::anyhow!(
                "ONNX model not found at {}",
                onnx_path.display()
            )));
        }
        let onnx_data =
            crate::load_onnx_file(&onnx_path).map_err(|e| LoaderError::LoadFailed(e.into()))?;

        let (_, mut output) = onnx_bytes_to_model(&onnx_data, "kokoro", Some(&dir))?;

        // Build the TTS interface
        let tokenizer_path = dir.join("tokenizer.json");
        let tokenizer = if tokenizer_path.exists() {
            TokenizerInfo::HFTokenizerLocal(tokenizer_path.to_string_lossy().to_string())
        } else {
            return Err(LoaderError::LoadFailed(anyhow::anyhow!(
                "tokenizer.json not found in {}",
                dir.display()
            )));
        };

        let mut rng = rand::rng();
        let interface = build_kokoro_supergraph(tokenizer, &mut rng);

        output.interfaces.push(LoadedInterface {
            name: "kokoro-TextToSpeech".to_string(),
            interface: interface.to_any(),
        });

        Ok(output)
    }
}

/// Build a SuperGraph for the Kokoro TTS model.
///
/// The graph is simple: three tensor inputs (input_ids, style, speed)
/// go directly into the model execution node, which produces the waveform output.
fn build_kokoro_supergraph(
    tokenizer: TokenizerInfo,
    rng: &mut impl rand::Rng,
) -> TextToSpeechInterface {
    let mut builder = SuperGraphBuilder::new();

    // External input links
    let input_ids_link = builder.new_tensor_link(rng);
    let style_link = builder.new_tensor_link(rng);
    let speed_link = builder.new_tensor_link(rng);
    let model_weights_link = builder.new_model_link(rng);
    let audio_output_link = builder.new_tensor_link(rng);

    // Model execution: wire named inputs/outputs to the ONNX graph
    let tensor_inputs = vec![
        (input_ids_link, "input_ids".to_string()),
        (style_link, "style".to_string()),
        (speed_link, "speed".to_string()),
    ];
    let tensor_outputs = vec![("waveform".to_string(), audio_output_link)];

    builder.add_node(
        SuperGraphNodeModelExecution::new(
            rng,
            model_weights_link,
            0, // model index
            tensor_inputs,
            tensor_outputs,
        )
        .to_any(),
    );

    // Build the SuperGraph
    let sg_inputs = vec![
        input_ids_link.to_any(),
        style_link.to_any(),
        speed_link.to_any(),
        model_weights_link.to_any(),
    ];
    let sg_outputs = vec![audio_output_link.to_any()];
    let super_graph = builder.build(rng, &sg_inputs, &sg_outputs);

    TextToSpeechInterface {
        super_graph,
        text_ids_link: input_ids_link,
        model_weights: vec![model_weights_link],
        audio_output_link,
        sample_rate: 24000,
        input_config: TTSInputConfig::Kokoro {
            style_link,
            speed_link,
            tokenizer,
        },
    }
}
