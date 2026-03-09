use std::path::PathBuf;
use std::sync::Arc;
use whisper_tensor::interfaces::ImageGenerationInterface;
use whisper_tensor::loader::*;
use whisper_tensor::metadata::TokenizerInfo;
use whisper_tensor::model::Model;

use crate::onnx_graph::weights::{SafetensorsWeightManager, WeightManager};
use crate::sd_common::CastingWeightManager;

/// Unified Flux loader supporting both Schnell and Dev, in single-file or multi-file format.
///
/// **Single-file mode** (`path`): ComfyUI-format checkpoint with prefixed tensors
/// (`model.diffusion_model.*`, `text_encoders.clip_l.*`, `text_encoders.t5xxl.*`, `vae.*`).
/// Handles any weight dtype (F8E4M3, BF16, F16, F32).
///
/// **Multi-file mode** (`dit_path`, `vae_path`, `clip_path`, `t5_path`): Separate safetensors
/// files for each component.
///
/// Schnell vs Dev is auto-detected from the weights (presence of `guidance_in`).
pub struct FluxLoader;

impl Loader for FluxLoader {
    fn name(&self) -> &str {
        "Flux"
    }

    fn description(&self) -> &str {
        "Load Flux (Schnell or Dev) from a single checkpoint or separate safetensors files"
    }

    fn config_schema(&self) -> Vec<ConfigField> {
        vec![
            ConfigField {
                key: "path".to_string(),
                label: "Checkpoint Path".to_string(),
                description: "Single-file checkpoint (e.g. flux1-schnell-fp8.safetensors). \
                    Mutually exclusive with the per-component paths below."
                    .to_string(),
                field_type: ConfigFieldType::FilePath,
                required: false,
                default: None,
            },
            ConfigField {
                key: "dit_path".to_string(),
                label: "DiT Path".to_string(),
                description: "Path to the Flux DiT .safetensors file".to_string(),
                field_type: ConfigFieldType::FilePath,
                required: false,
                default: None,
            },
            ConfigField {
                key: "vae_path".to_string(),
                label: "VAE Path".to_string(),
                description: "Path to the Flux VAE .safetensors file (e.g. ae.safetensors)"
                    .to_string(),
                field_type: ConfigFieldType::FilePath,
                required: false,
                default: None,
            },
            ConfigField {
                key: "clip_path".to_string(),
                label: "CLIP-L Path".to_string(),
                description: "Path to the CLIP-L .safetensors file".to_string(),
                field_type: ConfigFieldType::FilePath,
                required: false,
                default: None,
            },
            ConfigField {
                key: "t5_path".to_string(),
                label: "T5-XXL Path".to_string(),
                description: "Path to the T5-XXL .safetensors file".to_string(),
                field_type: ConfigFieldType::FilePath,
                required: false,
                default: None,
            },
        ]
    }

    fn load(&self, config: ConfigValues) -> Result<LoaderOutput, LoaderError> {
        let img_size = match config.get("img_size") {
            Some(ConfigValue::Integer(n)) => *n as usize,
            _ => 1024,
        };

        if let Ok(path) = require_path(&config, "path") {
            load_single_file(path, img_size)
        } else {
            load_multi_file(&config, img_size)
        }
    }
}

/// Detect whether a safetensors file is a Flux single-file checkpoint.
///
/// Checks for the prefixed DiT canary tensor.
pub fn is_flux_single_file_checkpoint(wm: &SafetensorsWeightManager) -> bool {
    wm.get_tensor("model.diffusion_model.double_blocks.0.img_attn.qkv.weight")
        .is_ok()
}

// ============================================================================
// Single-file loading (ComfyUI format with prefixed tensors)
// ============================================================================

fn load_single_file(path: PathBuf, img_size: usize) -> Result<LoaderOutput, LoaderError> {
    use memmap2::Mmap;

    let storage = super::default_storage();

    let file = std::fs::File::open(&path).map_err(|e| LoaderError::LoadFailed(e.into()))?;
    let mmap = unsafe { Mmap::map(&file) }.map_err(|e| LoaderError::LoadFailed(e.into()))?;
    let wm = SafetensorsWeightManager::new_with_paths(vec![Arc::new(mmap)], vec![path.clone()])
        .map_err(|e| LoaderError::LoadFailed(e.into()))?;

    // Detect DiT storage dtype and compute dtype
    let dit_canary = "model.diffusion_model.double_blocks.0.img_attn.qkv.weight";
    let (compute_dtype, needs_dit_cast) = detect_compute_dtype(&wm, dit_canary)?;

    // Auto-detect Schnell vs Dev
    let has_guidance = wm
        .get_tensor("model.diffusion_model.guidance_in.in_layer.weight")
        .is_ok();
    let variant = if has_guidance { "dev" } else { "schnell" };
    println!("Detected Flux {variant} (guidance={has_guidance})");

    // Build CLIP-L (F16 — no FP8 cast needed)
    println!("Building CLIP-L encoder...");
    let clip_wm = wm
        .prefix("text_encoders")
        .prefix("clip_l")
        .prefix("transformer");
    let clip_onnx = crate::flux::build_clip_l_pooled(clip_wm, storage.clone(), Some(&path))
        .map_err(LoaderError::LoadFailed)?;

    // Build T5-XXL (T5 builder casts to F32 internally)
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

    // Build Flux DiT (cast F8E4M3→BF16 if needed)
    println!("Building Flux DiT...");
    let dit_wm = wm.prefix("model").prefix("diffusion_model");
    let flux_config = make_flux_config(img_size, has_guidance);
    let dit_onnx = if needs_dit_cast {
        let cast_wm = CastingWeightManager::new(dit_wm, crate::onnx_graph::tensor::DType::BF16);
        crate::flux::load_flux_dit_with_origin(cast_wm, flux_config, storage.clone(), Some(&path))
            .map_err(LoaderError::LoadFailed)?
    } else {
        crate::flux::load_flux_dit_with_origin(dit_wm, flux_config, storage.clone(), Some(&path))
            .map_err(LoaderError::LoadFailed)?
    };

    // Build Flux VAE decoder (F32 — no cast needed)
    println!("Building Flux VAE decoder...");
    let vae_wm = wm.prefix("vae");
    let vae_onnx = crate::sd_common::build_flux_vae_decoder(
        vae_wm,
        crate::onnx_graph::tensor::DType::F32,
        storage.clone(),
        &path,
    )
    .map_err(LoaderError::LoadFailed)?;

    assemble_output(
        &path,
        variant,
        compute_dtype,
        has_guidance,
        [
            ("clip_l", clip_onnx),
            ("t5_xxl", t5_onnx),
            ("dit", dit_onnx),
            ("vae_decoder", vae_onnx),
        ],
    )
}

// ============================================================================
// Multi-file loading (separate safetensors per component)
// ============================================================================

fn load_multi_file(config: &ConfigValues, img_size: usize) -> Result<LoaderOutput, LoaderError> {
    let dit_path = require_path(config, "dit_path")?;
    let vae_path = require_path(config, "vae_path")?;
    let clip_path = require_path(config, "clip_path")?;
    let t5_path = require_path(config, "t5_path")?;
    let storage = super::default_storage();

    // Detect DiT dtype and variant
    let (compute_dtype, needs_dit_cast, has_guidance) = {
        use memmap2::Mmap;
        let file = std::fs::File::open(&dit_path).map_err(|e| LoaderError::LoadFailed(e.into()))?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| LoaderError::LoadFailed(e.into()))?;
        let wm = SafetensorsWeightManager::new(vec![Arc::new(mmap)])
            .map_err(|e| LoaderError::LoadFailed(e.into()))?;
        let (compute_dtype, needs_cast) =
            detect_compute_dtype(&wm, "double_blocks.0.img_attn.qkv.weight")?;
        let has_guidance = wm.get_tensor("guidance_in.in_layer.weight").is_ok();
        (compute_dtype, needs_cast, has_guidance)
    };

    let variant = if has_guidance { "dev" } else { "schnell" };
    println!("Detected Flux {variant} (guidance={has_guidance})");

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
    let flux_config = make_flux_config(img_size, has_guidance);
    let dit_onnx = if needs_dit_cast {
        build_from_safetensors(&dit_path, |wm| {
            let cast_wm = CastingWeightManager::new(wm, crate::onnx_graph::tensor::DType::BF16);
            crate::flux::load_flux_dit_with_origin(
                cast_wm,
                flux_config,
                storage.clone(),
                Some(&dit_path),
            )
        })?
    } else {
        build_from_safetensors(&dit_path, |wm| {
            crate::flux::load_flux_dit_with_origin(
                wm,
                flux_config,
                storage.clone(),
                Some(&dit_path),
            )
        })?
    };

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

    // For multi-file, use dit_path as the reference for base_dir
    assemble_output(
        &dit_path,
        variant,
        compute_dtype,
        has_guidance,
        [
            ("clip_l", clip_onnx),
            ("t5_xxl", t5_onnx),
            ("dit", dit_onnx),
            ("vae_decoder", vae_onnx),
        ],
    )
}

// ============================================================================
// Shared helpers
// ============================================================================

/// Detect compute dtype from storage dtype. Returns (compute_dtype, needs_fp8_cast).
fn detect_compute_dtype(
    wm: &SafetensorsWeightManager,
    canary: &str,
) -> Result<(whisper_tensor::dtype::DType, bool), LoaderError> {
    let storage_dtype = crate::sd_common::detect_model_dtype_with_canary(wm, canary);
    println!("Detected DiT storage dtype: {storage_dtype:?}");
    match storage_dtype {
        crate::onnx_graph::tensor::DType::F8E4M3 => Ok((whisper_tensor::dtype::DType::BF16, true)),
        crate::onnx_graph::tensor::DType::BF16 => Ok((whisper_tensor::dtype::DType::BF16, false)),
        crate::onnx_graph::tensor::DType::F16 => Ok((whisper_tensor::dtype::DType::F16, false)),
        crate::onnx_graph::tensor::DType::F32 => Ok((whisper_tensor::dtype::DType::F32, false)),
        other => Err(LoaderError::LoadFailed(anyhow::anyhow!(
            "Unsupported DiT storage dtype: {other:?}"
        ))),
    }
}

fn make_flux_config(img_size: usize, has_guidance: bool) -> crate::flux::FluxConfig {
    if has_guidance {
        crate::flux::FluxConfig::dev(img_size, 256)
    } else {
        crate::flux::FluxConfig::schnell(img_size, 256)
    }
}

/// Build models and interface from the 4 ONNX blobs.
fn assemble_output(
    reference_path: &PathBuf,
    variant: &str,
    compute_dtype: whisper_tensor::dtype::DType,
    has_guidance: bool,
    components: [(&str, Vec<u8>); 4],
) -> Result<LoaderOutput, LoaderError> {
    let base_dir = reference_path.parent();
    let mut models = Vec::new();
    for (suffix, onnx_data) in &components {
        let mut rng = rand::rng();
        let model = Model::new_from_onnx(onnx_data, &mut rng, base_dir)
            .map_err(|e| LoaderError::LoadFailed(e.into()))?;
        models.push(LoadedModel {
            name: format!("flux-{variant}-{suffix}"),
            model: Arc::new(model),
        });
    }

    let interface = {
        let mut rng = rand::rng();
        ImageGenerationInterface::new_flux(
            &mut rng,
            TokenizerInfo::HFTokenizer("openai/clip-vit-large-patch14".to_string()),
            TokenizerInfo::HFTokenizer("google-t5/t5-base".to_string()),
            compute_dtype,
            has_guidance,
        )
    };

    let interfaces = vec![LoadedInterface {
        name: format!("flux-{variant}-ImageGeneration"),
        interface: interface.to_any(),
    }];

    Ok(LoaderOutput { models, interfaces })
}

/// Helper: open a safetensors file, mmap it, and call a builder function.
fn build_from_safetensors(
    path: &PathBuf,
    builder: impl FnOnce(SafetensorsWeightManager) -> Result<Vec<u8>, anyhow::Error>,
) -> Result<Vec<u8>, LoaderError> {
    use memmap2::Mmap;

    let file = std::fs::File::open(path).map_err(|e| LoaderError::LoadFailed(e.into()))?;
    let mmap = unsafe { Mmap::map(&file) }.map_err(|e| LoaderError::LoadFailed(e.into()))?;
    let wm = SafetensorsWeightManager::new(vec![Arc::new(mmap)])
        .map_err(|e| LoaderError::LoadFailed(e.into()))?;
    builder(wm).map_err(LoaderError::LoadFailed)
}
