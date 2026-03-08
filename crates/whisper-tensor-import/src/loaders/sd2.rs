use std::sync::Arc;
use whisper_tensor::interfaces::StableDiffusionInterface;
use whisper_tensor::loader::*;
use whisper_tensor::metadata::TokenizerInfo;
use whisper_tensor::model::Model;

/// Loader for Stable Diffusion 2.x checkpoints (.safetensors).
pub struct SD2Loader;

impl Loader for SD2Loader {
    fn name(&self) -> &str {
        "Stable Diffusion 2"
    }

    fn description(&self) -> &str {
        "Load a Stable Diffusion 2.x checkpoint (.safetensors) as a multi-model pipeline"
    }

    fn config_schema(&self) -> Vec<ConfigField> {
        vec![ConfigField {
            key: "path".to_string(),
            label: "Checkpoint Path".to_string(),
            description: "Path to the SD 2.x .safetensors checkpoint file".to_string(),
            field_type: ConfigFieldType::FilePath,
            required: true,
            default: None,
        }]
    }

    fn load(&self, config: ConfigValues) -> Result<LoaderOutput, LoaderError> {
        let path = require_path(&config, "path")?;
        let storage = crate::onnx_graph::WeightStorageStrategy::OriginReference;

        // Detect model dtype for the interface (controls casting in the SuperGraph).
        let model_dtype = {
            use crate::onnx_graph::weights::SafetensorsWeightManager;
            use memmap2::Mmap;
            let file = std::fs::File::open(&path)
                .map_err(|e| LoaderError::LoadFailed(e.into()))?;
            let mmap = unsafe { Mmap::map(&file) }
                .map_err(|e| LoaderError::LoadFailed(e.into()))?;
            let wm = SafetensorsWeightManager::new(vec![Arc::new(mmap)])
                .map_err(|e| LoaderError::LoadFailed(e.into()))?;
            let import_dtype = crate::sd_common::detect_model_dtype(&wm);
            // Convert import crate DType -> whisper_tensor DType
            match import_dtype {
                crate::onnx_graph::tensor::DType::F16 => whisper_tensor::dtype::DType::F16,
                crate::onnx_graph::tensor::DType::BF16 => whisper_tensor::dtype::DType::BF16,
                crate::onnx_graph::tensor::DType::F32 => whisper_tensor::dtype::DType::F32,
                other => return Err(LoaderError::LoadFailed(
                    anyhow::anyhow!("Unsupported model dtype: {:?}", other),
                )),
            }
        };

        let (te_onnx, unet_onnx, vae_onnx) =
            crate::sd2::load_sd2_checkpoint(&path, storage).map_err(LoaderError::LoadFailed)?;

        let base_name = path
            .file_stem()
            .unwrap_or_default()
            .to_str()
            .unwrap_or("sd2")
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
            StableDiffusionInterface::new_with_dtype(
                &mut rng,
                TokenizerInfo::HFTokenizer("laion/CLIP-ViT-H-14-laion2B-s32B-b79K".to_string()),
                model_dtype,
            )
        };

        let interfaces = vec![LoadedInterface {
            name: format!("{base_name}-StableDiffusion"),
            interface: sd_interface.to_any(),
        }];

        Ok(LoaderOutput { models, interfaces })
    }
}
