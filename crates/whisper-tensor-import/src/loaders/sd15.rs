use super::default_storage;
use std::sync::Arc;
use whisper_tensor::interfaces::StableDiffusionInterface;
use whisper_tensor::loader::*;
use whisper_tensor::metadata::TokenizerInfo;
use whisper_tensor::model::Model;

/// Loader for Stable Diffusion 1.5 checkpoints (.safetensors).
pub struct SD15Loader;

impl Loader for SD15Loader {
    fn name(&self) -> &str {
        "Stable Diffusion 1.5"
    }

    fn description(&self) -> &str {
        "Load a Stable Diffusion 1.5 checkpoint (.safetensors) as a multi-model pipeline"
    }

    fn config_schema(&self) -> Vec<ConfigField> {
        vec![ConfigField {
            key: "path".to_string(),
            label: "Checkpoint Path".to_string(),
            description: "Path to the SD 1.5 .safetensors checkpoint file".to_string(),
            field_type: ConfigFieldType::FilePath,
            required: true,
            default: None,
        }]
    }

    fn load(&self, config: ConfigValues) -> Result<LoaderOutput, LoaderError> {
        let path = require_path(&config, "path")?;
        let storage = default_storage();

        let (te_onnx, unet_onnx, vae_onnx) =
            crate::sd15::load_sd15_checkpoint(&path, storage).map_err(LoaderError::LoadFailed)?;

        let base_name = path
            .file_stem()
            .unwrap_or_default()
            .to_str()
            .unwrap_or("sd15")
            .to_string();
        let base_dir = path.parent();

        let mut models = Vec::new();
        for (suffix, onnx_data) in [
            ("text_encoder", te_onnx),
            ("unet", unet_onnx),
            ("vae_decoder", vae_onnx),
        ] {
            let mut rng = rand::rng();
            let model = Model::new_from_onnx(&onnx_data, &mut rng, base_dir)
                .map_err(|e| LoaderError::LoadFailed(e.into()))?;
            models.push(LoadedModel {
                name: format!("{base_name}-{suffix}"),
                model: Arc::new(model),
            });
        }

        let sd_interface = {
            let mut rng = rand::rng();
            StableDiffusionInterface::new(
                &mut rng,
                TokenizerInfo::HFTokenizer("openai/clip-vit-large-patch14".to_string()),
            )
        };

        let interfaces = vec![LoadedInterface {
            name: format!("{base_name}-StableDiffusion"),
            interface: sd_interface.to_any(),
        }];

        Ok(LoaderOutput { models, interfaces })
    }
}
