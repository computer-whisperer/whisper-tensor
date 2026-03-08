use super::onnx_bytes_to_model;
use whisper_tensor::dtype::DType;
use whisper_tensor::interfaces::TextInferenceTokensInLogitOutInterface;
use whisper_tensor::loader::*;
use whisper_tensor::metadata::TokenizerInfo;

/// Loader for raw ONNX model files.
///
/// For simple transformer models (e.g. GPT-2), set `tokenizer` to
/// the HuggingFace tokenizer name. The loader will build a
/// `TextInferenceTokensInLogitOutInterface` assuming the first
/// graph input is token IDs and the first output is logits.
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
                key: "tokenizer".to_string(),
                label: "Tokenizer".to_string(),
                description: "HuggingFace tokenizer name (e.g. 'gpt2'). If set, builds a text inference interface assuming first input=tokens, first output=logits.".to_string(),
                field_type: ConfigFieldType::String,
                required: false,
                default: None,
            },
        ]
    }

    fn load(&self, config: ConfigValues) -> Result<LoaderOutput, LoaderError> {
        let path = require_path(&config, "path")?;
        let tokenizer_name = get_string(&config, "tokenizer")?;

        let onnx_data =
            crate::load_onnx_file(&path).map_err(|e| LoaderError::LoadFailed(e.into()))?;

        let model_name = path
            .file_stem()
            .unwrap_or_default()
            .to_str()
            .unwrap_or("model")
            .to_string();

        let (model, mut output) = onnx_bytes_to_model(&onnx_data, &model_name, path.parent())?;

        // If a tokenizer is specified, build a simple transformer interface
        if let Some(tok_name) = tokenizer_name {
            let graph = model.get_symbolic_graph();
            let input_ids = graph.get_inputs();
            let output_ids = graph.get_outputs();

            let first_input = input_ids.first().and_then(|id| {
                let info = graph.get_tensor_info(*id)?;
                Some((*id, info))
            });
            let first_output = output_ids.first().and_then(|id| {
                let info = graph.get_tensor_info(*id)?;
                Some((*id, info))
            });

            if let (Some((_, input_info)), Some((_, output_info))) = (first_input, first_output) {
                let token_input_name = input_info.name().unwrap();
                let logit_output_name = output_info.name().unwrap();
                let input_dtype = input_info.dtype.unwrap_or(DType::I32);
                let input_rank = input_info.shape.as_ref().map_or(2, |s| s.len());
                let output_rank = output_info.shape.as_ref().map_or(3, |s| s.len());

                let mut rng = rand::rng();
                let interface = TextInferenceTokensInLogitOutInterface::build_simple_transformer(
                    TokenizerInfo::HFTokenizer(tok_name),
                    &token_input_name,
                    &logit_output_name,
                    input_dtype,
                    input_rank,
                    output_rank,
                    &mut rng,
                );
                output.interfaces.push(LoadedInterface {
                    name: format!("{model_name}-TextInference"),
                    interface: interface.to_any(),
                });
            }
        }

        Ok(output)
    }
}
