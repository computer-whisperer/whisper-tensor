use std::sync::Arc;
use whisper_tensor::interfaces::ImageGenerationInterface;
use whisper_tensor::loader::*;
use whisper_tensor::metadata::TokenizerInfo;
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

        // Detect model dtype
        let model_dtype = {
            use crate::onnx_graph::weights::SafetensorsWeightManager;
            use memmap2::Mmap;
            let file = std::fs::File::open(&path).map_err(|e| LoaderError::LoadFailed(e.into()))?;
            let mmap =
                unsafe { Mmap::map(&file) }.map_err(|e| LoaderError::LoadFailed(e.into()))?;
            let wm = SafetensorsWeightManager::new(vec![Arc::new(mmap)])
                .map_err(|e| LoaderError::LoadFailed(e.into()))?;
            let import_dtype = crate::sd_common::detect_model_dtype(&wm);
            match import_dtype {
                crate::onnx_graph::tensor::DType::F16 => whisper_tensor::dtype::DType::F16,
                crate::onnx_graph::tensor::DType::BF16 => whisper_tensor::dtype::DType::BF16,
                crate::onnx_graph::tensor::DType::F32 => whisper_tensor::dtype::DType::F32,
                other => {
                    return Err(LoaderError::LoadFailed(anyhow::anyhow!(
                        "Unsupported model dtype: {:?}",
                        other
                    )));
                }
            }
        };

        let (te1_onnx, te2_onnx, unet_onnx, vae_onnx) =
            crate::sd_xl::load_sdxl_checkpoint(&path, storage).map_err(LoaderError::LoadFailed)?;

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

        let interface = {
            let mut rng = rand::rng();
            // Both SDXL text encoders use the same CLIP tokenizer
            ImageGenerationInterface::new_sdxl(
                &mut rng,
                TokenizerInfo::HFTokenizer("openai/clip-vit-large-patch14".to_string()),
                model_dtype,
            )
        };

        let interfaces = vec![LoadedInterface {
            name: format!("{base_name}-ImageGeneration"),
            interface: interface.to_any(),
        }];

        Ok(LoaderOutput { models, interfaces })
    }
}
