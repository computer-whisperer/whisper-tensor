use super::shared::{build_rnn_supergraph, default_storage, onnx_bytes_to_model};
use whisper_tensor::loader::*;
use whisper_tensor::metadata::TokenizerInfo;

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

        let onnx_data = crate::models::llm::rwkv7::load_rwkv7_pth(&path, storage)
            .map_err(LoaderError::LoadFailed)?;

        let model_name = path
            .file_stem()
            .unwrap_or_default()
            .to_str()
            .unwrap_or("rwkv7")
            .to_string();

        let (model, mut output) = onnx_bytes_to_model(&onnx_data, &model_name, path.parent())?;

        // Collect state pairs from RWKV7 naming convention
        let graph = model.get_symbolic_graph();
        let names_by_id = graph.get_tensors_by_name();

        let mut state_pairs = Vec::new();
        let mut layer = 0;
        loop {
            let mut found_any = false;
            for prefix in ["time_mixer_x", "channel_mixer_x", "vk_state"] {
                let in_name = format!("{prefix}_in_{layer}");
                let out_name = format!("{prefix}_out_{layer}");
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

        let mut rng = rand::rng();
        let interface = build_rnn_supergraph(
            TokenizerInfo::RWKVWorld,
            "token_input",
            "output",
            &state_pairs,
            graph,
            &mut rng,
        );
        output.interfaces.push(LoadedInterface {
            name: format!("{model_name}-TextInference"),
            interface: interface.to_any(),
        });

        Ok(output)
    }
}
