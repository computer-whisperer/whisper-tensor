use std::sync::Arc;
use whisper_tensor::loader::*;
use whisper_tensor::model::Model;

/// Loader for Stable Diffusion XL checkpoints (.safetensors).
pub struct SDXLLoader;

impl Loader for SDXLLoader {
    fn name(&self) -> &str {
        "Stable Diffusion XL"
    }

    fn description(&self) -> &str {
        "Load an SDXL checkpoint (.safetensors) as a multi-model pipeline"
    }

    fn config_schema(&self) -> Vec<ConfigField> {
        vec![ConfigField {
            key: "path".to_string(),
            label: "Checkpoint Path".to_string(),
            description: "Path to the SDXL .safetensors checkpoint file".to_string(),
            field_type: ConfigFieldType::FilePath,
            required: true,
            default: None,
        }]
    }

    fn load(&self, config: ConfigValues) -> Result<LoaderOutput, LoaderError> {
        let path = require_path(&config, "path")?;
        let storage = crate::onnx_graph::WeightStorageStrategy::OriginReference;

        let (te1_onnx, te2_onnx, unet_onnx, vae_onnx) =
            crate::sd_xl::load_sdxl_checkpoint(&path, storage)
                .map_err(LoaderError::LoadFailed)?;

        let base_name = path
            .file_stem()
            .unwrap_or_default()
            .to_str()
            .unwrap_or("sdxl")
            .to_string();
        let base_dir = path.parent();

        let mut models = Vec::new();
        for (suffix, onnx_data) in [
            ("text_encoder_1", te1_onnx),
            ("text_encoder_2", te2_onnx),
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

        // TODO: SDXL interface (dual text encoders, ADM conditioning, different VAE scale)
        let interfaces = vec![];

        Ok(LoaderOutput { models, interfaces })
    }
}
