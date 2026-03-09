use std::path::PathBuf;
use std::sync::Arc;
use whisper_tensor::interfaces::ImageGenerationInterface;
use whisper_tensor::loader::*;
use whisper_tensor::metadata::TokenizerInfo;
use whisper_tensor::model::Model;

/// Loader for Flux Schnell from separate model files (CLIP-L, T5-XXL, DiT, VAE).
pub struct FluxSchnellLoader;

impl Loader for FluxSchnellLoader {
    fn name(&self) -> &str {
        "Flux Schnell"
    }

    fn description(&self) -> &str {
        "Load Flux Schnell from separate CLIP-L, T5-XXL, DiT, and VAE safetensors files"
    }

    fn config_schema(&self) -> Vec<ConfigField> {
        vec![
            ConfigField {
                key: "dit_path".to_string(),
                label: "DiT Path".to_string(),
                description:
                    "Path to the Flux DiT .safetensors file (e.g. flux1-schnell.safetensors)"
                        .to_string(),
                field_type: ConfigFieldType::FilePath,
                required: true,
                default: None,
            },
            ConfigField {
                key: "vae_path".to_string(),
                label: "VAE Path".to_string(),
                description: "Path to the Flux VAE .safetensors file (e.g. ae.safetensors)"
                    .to_string(),
                field_type: ConfigFieldType::FilePath,
                required: true,
                default: None,
            },
            ConfigField {
                key: "clip_path".to_string(),
                label: "CLIP-L Path".to_string(),
                description: "Path to the CLIP-L .safetensors file (e.g. clip_l.safetensors)"
                    .to_string(),
                field_type: ConfigFieldType::FilePath,
                required: true,
                default: None,
            },
            ConfigField {
                key: "t5_path".to_string(),
                label: "T5-XXL Path".to_string(),
                description: "Path to the T5-XXL .safetensors file (e.g. t5xxl_fp16.safetensors)"
                    .to_string(),
                field_type: ConfigFieldType::FilePath,
                required: true,
                default: None,
            },
        ]
    }

    fn load(&self, config: ConfigValues) -> Result<LoaderOutput, LoaderError> {
        let dit_path = require_path(&config, "dit_path")?;
        let vae_path = require_path(&config, "vae_path")?;
        let clip_path = require_path(&config, "clip_path")?;
        let t5_path = require_path(&config, "t5_path")?;
        let storage = super::default_storage();

        // Detect DiT model dtype
        let model_dtype = {
            use crate::onnx_graph::weights::SafetensorsWeightManager;
            use memmap2::Mmap;
            let file =
                std::fs::File::open(&dit_path).map_err(|e| LoaderError::LoadFailed(e.into()))?;
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
                        "Unsupported DiT dtype: {:?}",
                        other
                    )));
                }
            }
        };

        // Build CLIP-L
        println!("Building CLIP-L encoder...");
        let clip_onnx = build_from_safetensors(&clip_path, |wm| {
            crate::flux::build_clip_l_pooled(wm, storage.clone(), Some(&clip_path))
        })?;

        // Build T5-XXL
        println!("Building T5-XXL encoder...");
        let t5_onnx = build_from_safetensors(&t5_path, |wm| {
            let config = crate::t5::T5Config::t5_xxl(256);
            crate::t5::load_t5_encoder_with_origin(wm, config, storage.clone(), Some(&t5_path))
        })?;

        // Build Flux DiT
        println!("Building Flux DiT...");
        let dit_onnx = build_from_safetensors(&dit_path, |wm| {
            let config = crate::flux::FluxConfig::schnell_1024(256);
            crate::flux::load_flux_dit_with_origin(wm, config, storage.clone(), Some(&dit_path))
        })?;

        // Build Flux VAE decoder
        println!("Building Flux VAE decoder...");
        let vae_onnx = build_from_safetensors(&vae_path, |wm| {
            crate::sd_common::build_flux_vae_decoder(
                wm,
                crate::onnx_graph::tensor::DType::F32,
                storage.clone(),
                &vae_path,
            )
        })?;

        // Load models
        let mut models = Vec::new();
        for (suffix, onnx_data, base_path) in [
            ("clip_l", clip_onnx, &clip_path),
            ("t5_xxl", t5_onnx, &t5_path),
            ("dit", dit_onnx, &dit_path),
            ("vae_decoder", vae_onnx, &vae_path),
        ] {
            let mut rng = rand::rng();
            let model = Model::new_from_onnx(&onnx_data, &mut rng, base_path.parent())
                .map_err(|e| LoaderError::LoadFailed(e.into()))?;
            models.push(LoadedModel {
                name: format!("flux-schnell-{suffix}"),
                model: Arc::new(model),
            });
        }

        let interface = {
            let mut rng = rand::rng();
            ImageGenerationInterface::new_flux_schnell(
                &mut rng,
                TokenizerInfo::HFTokenizer("openai/clip-vit-large-patch14".to_string()),
                TokenizerInfo::HFTokenizer("google/t5-v1_1-xxl".to_string()),
                model_dtype,
            )
        };

        let interfaces = vec![LoadedInterface {
            name: "flux-schnell-ImageGeneration".to_string(),
            interface: interface.to_any(),
        }];

        Ok(LoaderOutput { models, interfaces })
    }
}

/// Helper: open a safetensors file, mmap it, and call a builder function.
fn build_from_safetensors(
    path: &PathBuf,
    builder: impl FnOnce(
        crate::onnx_graph::weights::SafetensorsWeightManager,
    ) -> Result<Vec<u8>, anyhow::Error>,
) -> Result<Vec<u8>, LoaderError> {
    use crate::onnx_graph::weights::SafetensorsWeightManager;
    use memmap2::Mmap;

    let file = std::fs::File::open(path).map_err(|e| LoaderError::LoadFailed(e.into()))?;
    let mmap = unsafe { Mmap::map(&file) }.map_err(|e| LoaderError::LoadFailed(e.into()))?;
    let wm = SafetensorsWeightManager::new(vec![Arc::new(mmap)])
        .map_err(|e| LoaderError::LoadFailed(e.into()))?;
    builder(wm).map_err(LoaderError::LoadFailed)
}
