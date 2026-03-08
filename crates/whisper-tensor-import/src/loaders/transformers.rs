use super::{default_storage, onnx_bytes_to_model};
use whisper_tensor::interfaces::TextInferenceTokensInLogitOutInterface;
use whisper_tensor::loader::*;
use whisper_tensor::metadata::TokenizerInfo;

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

        let onnx_data = crate::load_transformers_format(&path, storage)
            .map_err(|e| LoaderError::LoadFailed(e.into()))?;

        let model_name = path
            .file_name()
            .unwrap_or_default()
            .to_str()
            .unwrap_or("model")
            .to_string();

        let (model, mut output) = onnx_bytes_to_model(&onnx_data, &model_name, Some(&path))?;

        // Detect architecture from config.json to determine tokenizer
        let config_path = path.join("config.json");
        let tokenizer_name = if config_path.exists() {
            // Use directory name as HF tokenizer ID
            path.to_str().unwrap_or(&model_name).to_string()
        } else {
            model_name.clone()
        };

        // Inspect model for state pairs (KV cache)
        let graph = model.get_symbolic_graph();
        let names_by_id = graph.get_tensors_by_name();

        // Llama-style KV cache: kv_cache_input_k_{i}/kv_cache_output_k_{i},
        // kv_cache_input_v_{i}/kv_cache_output_v_{i}
        let mut state_pairs = Vec::new();
        let mut layer = 0;
        loop {
            let mut found_any = false;
            for prefix in ["kv_cache_k", "kv_cache_v"] {
                let in_name = format!("{prefix}_input_{layer}");
                let out_name = format!("{prefix}_output_{layer}");
                if names_by_id.contains_key(&in_name) && names_by_id.contains_key(&out_name) {
                    state_pairs.push((in_name, out_name));
                    found_any = true;
                }
            }
            if !found_any {
                break;
            }
            layer += 1;
        }

        // Check for standard input/output names
        let token_input = if names_by_id.contains_key("input_ids") {
            "input_ids"
        } else {
            // Fall back to first input
            return Ok(output);
        };
        let logit_output = if names_by_id.contains_key("logits") {
            "logits"
        } else {
            return Ok(output);
        };

        let mut rng = rand::rng();
        if state_pairs.is_empty() {
            // Simple transformer (no KV cache state)
            let input_id = *names_by_id.get(token_input).unwrap();
            let input_info = graph.get_tensor_info(input_id).unwrap();
            let input_dtype = input_info
                .dtype
                .unwrap_or(whisper_tensor::dtype::DType::I32);
            let input_rank = input_info.shape.as_ref().map_or(2, |s| s.len());

            let output_id = *names_by_id.get(logit_output).unwrap();
            let output_info = graph.get_tensor_info(output_id).unwrap();
            let output_rank = output_info.shape.as_ref().map_or(3, |s| s.len());

            let interface = TextInferenceTokensInLogitOutInterface::build_simple_transformer(
                TokenizerInfo::HFTokenizer(tokenizer_name),
                token_input,
                logit_output,
                input_dtype,
                input_rank,
                output_rank,
                &mut rng,
            );
            output.interfaces.push(LoadedInterface {
                name: format!("{model_name}-TextInference"),
                interface: interface.to_any(),
            });
        } else {
            // RNN-style with KV cache
            let interface = TextInferenceTokensInLogitOutInterface::build_rnn(
                TokenizerInfo::HFTokenizer(tokenizer_name),
                token_input,
                logit_output,
                &state_pairs,
                graph,
                &mut rng,
            );
            output.interfaces.push(LoadedInterface {
                name: format!("{model_name}-TextInference"),
                interface: interface.to_any(),
            });
        }

        Ok(output)
    }
}
