use std::sync::Arc;
use whisper_tensor::interfaces::ImageGenerationInterface;
use whisper_tensor::loader::*;
use whisper_tensor::metadata::TokenizerInfo;
use whisper_tensor::model::Model;

use crate::onnx_graph::weights::WeightManager;

/// Loader for Flux Schnell from a single ComfyUI-format FP8 checkpoint.
///
/// The checkpoint contains all components with prefixed tensor names:
/// - `model.diffusion_model.*` — DiT (F8E4M3, cast to BF16)
/// - `text_encoders.clip_l.transformer.*` — CLIP-L (F16)
/// - `text_encoders.t5xxl.transformer.*` — T5-XXL (F8E4M3, cast to BF16)
/// - `vae.*` — VAE encoder/decoder (F32)
pub struct FluxSchnellFP8Loader;

impl Loader for FluxSchnellFP8Loader {
    fn name(&self) -> &str {
        "Flux Schnell FP8"
    }

    fn description(&self) -> &str {
        "Load Flux Schnell from a single ComfyUI-format FP8 safetensors checkpoint"
    }

    fn config_schema(&self) -> Vec<ConfigField> {
        vec![ConfigField {
            key: "path".to_string(),
            label: "Checkpoint Path".to_string(),
            description:
                "Path to the Flux FP8 .safetensors file (e.g. flux1-schnell-fp8.safetensors)"
                    .to_string(),
            field_type: ConfigFieldType::FilePath,
            required: true,
            default: None,
        }]
    }

    fn load(&self, config: ConfigValues) -> Result<LoaderOutput, LoaderError> {
        use crate::onnx_graph::weights::SafetensorsWeightManager;
        use crate::sd_common::CastingWeightManager;
        use memmap2::Mmap;

        let path = require_path(&config, "path")?;
        let img_size = match config.get("img_size") {
            Some(ConfigValue::Integer(n)) => *n as usize,
            _ => 1024,
        };
        let storage = super::default_storage();

        // Mmap the single checkpoint file
        let file = std::fs::File::open(&path).map_err(|e| LoaderError::LoadFailed(e.into()))?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| LoaderError::LoadFailed(e.into()))?;
        let wm = SafetensorsWeightManager::new_with_paths(vec![Arc::new(mmap)], vec![path.clone()])
            .map_err(|e| LoaderError::LoadFailed(e.into()))?;

        // Detect DiT dtype — FP8 checkpoint stores weights as F8E4M3
        let dit_canary = "model.diffusion_model.double_blocks.0.img_attn.qkv.weight";
        let storage_dtype = crate::sd_common::detect_model_dtype_with_canary(&wm, dit_canary);
        println!("Detected DiT storage dtype: {:?}", storage_dtype);

        // Map storage dtype to compute dtype
        let compute_dtype = match storage_dtype {
            crate::onnx_graph::tensor::DType::F8E4M3 => whisper_tensor::dtype::DType::BF16,
            crate::onnx_graph::tensor::DType::BF16 => whisper_tensor::dtype::DType::BF16,
            crate::onnx_graph::tensor::DType::F16 => whisper_tensor::dtype::DType::F16,
            crate::onnx_graph::tensor::DType::F32 => whisper_tensor::dtype::DType::F32,
            other => {
                return Err(LoaderError::LoadFailed(anyhow::anyhow!(
                    "Unsupported DiT storage dtype: {:?}",
                    other
                )));
            }
        };

        let needs_cast = matches!(storage_dtype, crate::onnx_graph::tensor::DType::F8E4M3);

        // Build CLIP-L (F16 in the checkpoint — no FP8 cast needed)
        // Prefix: text_encoders.clip_l.transformer → builder expects text_model.* at root
        println!("Building CLIP-L encoder...");
        let clip_wm = wm
            .prefix("text_encoders")
            .prefix("clip_l")
            .prefix("transformer");
        let clip_onnx = crate::flux::build_clip_l_pooled(clip_wm, storage.clone(), Some(&path))
            .map_err(LoaderError::LoadFailed)?;

        // Build T5-XXL (F8E4M3 in the checkpoint — cast to F32 for compute)
        // T5 builder already wraps with CastingWeightManager(F32) internally,
        // but we need to first cast F8E4M3→BF16 so the inner cast sees a supported type.
        // Actually: T5 builder casts everything to F32 internally. F8E4M3→F32 should work
        // directly since Cast is a graph op resolved at inference time.
        println!("Building T5-XXL encoder...");
        let t5_wm = wm
            .prefix("text_encoders")
            .prefix("t5xxl")
            .prefix("transformer");
        let t5_onnx = {
            let config = crate::t5::T5Config::t5_xxl(256);
            crate::t5::load_t5_encoder_with_origin(t5_wm, config, storage.clone(), Some(&path))
                .map_err(LoaderError::LoadFailed)?
        };

        // Build Flux DiT (F8E4M3 in the checkpoint — cast to BF16)
        println!("Building Flux DiT...");
        let dit_wm = wm.prefix("model").prefix("diffusion_model");
        let dit_onnx = if needs_cast {
            let cast_wm = CastingWeightManager::new(dit_wm, crate::onnx_graph::tensor::DType::BF16);
            let config = crate::flux::FluxConfig::schnell(img_size, 256);
            crate::flux::load_flux_dit_with_origin(cast_wm, config, storage.clone(), Some(&path))
                .map_err(LoaderError::LoadFailed)?
        } else {
            let config = crate::flux::FluxConfig::schnell(img_size, 256);
            crate::flux::load_flux_dit_with_origin(dit_wm, config, storage.clone(), Some(&path))
                .map_err(LoaderError::LoadFailed)?
        };

        // Build Flux VAE decoder (F32 in the checkpoint — no cast needed)
        println!("Building Flux VAE decoder...");
        let vae_wm = wm.prefix("vae");
        let vae_onnx = crate::sd_common::build_flux_vae_decoder(
            vae_wm,
            crate::onnx_graph::tensor::DType::F32,
            storage.clone(),
            &path,
        )
        .map_err(LoaderError::LoadFailed)?;

        // Load all 4 models — all share the same origin file
        let mut models = Vec::new();
        let base_dir = path.parent();
        for (suffix, onnx_data) in [
            ("clip_l", clip_onnx),
            ("t5_xxl", t5_onnx),
            ("dit", dit_onnx),
            ("vae_decoder", vae_onnx),
        ] {
            let mut rng = rand::rng();
            let model = Model::new_from_onnx(&onnx_data, &mut rng, base_dir)
                .map_err(|e| LoaderError::LoadFailed(e.into()))?;
            models.push(LoadedModel {
                name: format!("flux-schnell-fp8-{suffix}"),
                model: Arc::new(model),
            });
        }

        let interface = {
            let mut rng = rand::rng();
            ImageGenerationInterface::new_flux_schnell(
                &mut rng,
                TokenizerInfo::HFTokenizer("openai/clip-vit-large-patch14".to_string()),
                TokenizerInfo::HFTokenizer("google-t5/t5-base".to_string()),
                compute_dtype,
            )
        };

        let interfaces = vec![LoadedInterface {
            name: "flux-schnell-ImageGeneration".to_string(),
            interface: interface.to_any(),
        }];

        Ok(LoaderOutput { models, interfaces })
    }
}
