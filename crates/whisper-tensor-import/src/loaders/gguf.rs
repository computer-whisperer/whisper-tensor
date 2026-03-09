use super::build_rnn_supergraph;
use std::sync::Arc;
use whisper_tensor::loader::*;
use whisper_tensor::metadata::TokenizerInfo;

/// Loader for GGUF quantized model files.
pub struct GgufLoader;

impl Loader for GgufLoader {
    fn name(&self) -> &str {
        "GGUF"
    }

    fn description(&self) -> &str {
        "Load a quantized model from a GGUF file (llama.cpp format)"
    }

    fn config_schema(&self) -> Vec<ConfigField> {
        vec![
            ConfigField {
                key: "path".to_string(),
                label: "Model Path".to_string(),
                description: "Path to the .gguf file".to_string(),
                field_type: ConfigFieldType::FilePath,
                required: true,
                default: None,
            },
            ConfigField {
                key: "tokenizer".to_string(),
                label: "Tokenizer".to_string(),
                description: "HuggingFace tokenizer name or path to tokenizer.json".to_string(),
                field_type: ConfigFieldType::String,
                required: false,
                default: None,
            },
        ]
    }

    fn load(&self, config: ConfigValues) -> Result<LoaderOutput, LoaderError> {
        let path = require_path(&config, "path")?;
        let tokenizer_name = get_string(&config, "tokenizer")?;

        // Parse GGUF once, detect architecture, dispatch to builder
        let gguf =
            crate::gguf::GgufFile::open(&path).map_err(|e| LoaderError::LoadFailed(e.into()))?;

        let arch = gguf
            .architecture()
            .ok_or_else(|| {
                LoaderError::LoadFailed(anyhow::anyhow!(
                    "GGUF file missing general.architecture metadata"
                ))
            })?
            .to_string();

        let info = match arch.as_str() {
            "llama" => crate::gguf::llama3::load_llama3_gguf(&gguf, &path)
                .map_err(LoaderError::LoadFailed)?,
            _ => {
                return Err(LoaderError::LoadFailed(anyhow::anyhow!(
                    "Unsupported GGUF architecture: {arch}"
                )));
            }
        };

        let model_name = path
            .file_stem()
            .unwrap_or_default()
            .to_str()
            .unwrap_or("model")
            .to_string();

        let model = Arc::new(info.model);
        let mut output = LoaderOutput {
            models: vec![LoadedModel {
                name: model_name.clone(),
                model: model.clone(),
            }],
            interfaces: vec![],
        };

        // Build TextInference interface if we have state pairs AND a tokenizer
        if !info.state_pairs.is_empty()
            && let Some(tokenizer_info) = resolve_tokenizer(&tokenizer_name, &path, &gguf)
        {
            let graph = model.get_symbolic_graph();
            let mut rng = rand::rng();
            let interface = build_rnn_supergraph(
                tokenizer_info,
                &info.token_input_name,
                &info.logit_output_name,
                &info.state_pairs,
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

/// Resolve tokenizer from config, adjacent files, or embedded GGUF metadata.
fn resolve_tokenizer(
    tokenizer_name: &Option<String>,
    gguf_path: &std::path::Path,
    gguf: &crate::gguf::GgufFile,
) -> Option<TokenizerInfo> {
    // Explicitly provided tokenizer
    if let Some(name) = tokenizer_name {
        let path = std::path::Path::new(name);
        if path.exists() {
            let abs = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
            return Some(TokenizerInfo::HFTokenizerLocal(
                abs.to_string_lossy().to_string(),
            ));
        }
        return Some(TokenizerInfo::HFTokenizer(name.clone()));
    }

    // Check for tokenizer.json next to the GGUF file
    if let Some(dir) = gguf_path.parent() {
        let local = dir.join("tokenizer.json");
        if local.exists() {
            let abs = std::fs::canonicalize(&local).unwrap_or(local);
            return Some(TokenizerInfo::HFTokenizerLocal(
                abs.to_string_lossy().to_string(),
            ));
        }
    }

    // Synthesize from GGUF embedded tokenizer metadata
    if let Some(json) = crate::gguf::tokenizer::synthesize_tokenizer_json(gguf) {
        return Some(TokenizerInfo::HFTokenizerJson(json));
    }

    None
}
