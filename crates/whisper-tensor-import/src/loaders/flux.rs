use std::path::{Path, PathBuf};
use std::sync::Arc;
use whisper_tensor::loader::*;
use whisper_tensor::metadata::TokenizerInfo;
use whisper_tensor::model::Model;

use crate::models::diffusion::sd_common::CastingWeightManager;
use crate::onnx_graph::weights::{SafetensorsWeightManager, WeightManager};

/// Unified Flux loader supporting both Schnell and Dev, in single-file or multi-file format.
///
/// **Single-file mode** (`path`): ComfyUI-format checkpoint with prefixed tensors
/// (`model.diffusion_model.*`, `text_encoders.clip_l.*`, `text_encoders.t5xxl.*`, `vae.*`).
/// Handles any weight dtype (F8E4M3FN, BF16, F16, F32).
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

    let storage = super::shared::default_storage();

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
    let clip_onnx =
        crate::models::diffusion::flux::build_clip_l_pooled(clip_wm, storage.clone(), Some(&path))
            .map_err(LoaderError::LoadFailed)?;

    // Build T5-XXL (T5 builder casts to F32 internally)
    println!("Building T5-XXL encoder...");
    let t5_wm = wm
        .prefix("text_encoders")
        .prefix("t5xxl")
        .prefix("transformer");
    let t5_onnx = {
        let config = crate::models::diffusion::t5::T5Config::t5_xxl(256);
        crate::models::diffusion::t5::load_t5_encoder_with_origin(
            t5_wm,
            config,
            storage.clone(),
            Some(&path),
        )
        .map_err(LoaderError::LoadFailed)?
    };

    // Build Flux DiT (cast F8E4M3FN→BF16 if needed)
    println!("Building Flux DiT...");
    let dit_wm = wm.prefix("model").prefix("diffusion_model");
    let flux_config = make_flux_config(img_size, has_guidance);
    let dit_onnx = if needs_dit_cast {
        let cast_wm = CastingWeightManager::new(dit_wm, crate::onnx_graph::tensor::DType::BF16);
        crate::models::diffusion::flux::load_flux_dit_with_origin(
            cast_wm,
            flux_config,
            storage.clone(),
            Some(&path),
        )
        .map_err(LoaderError::LoadFailed)?
    } else {
        crate::models::diffusion::flux::load_flux_dit_with_origin(
            dit_wm,
            flux_config,
            storage.clone(),
            Some(&path),
        )
        .map_err(LoaderError::LoadFailed)?
    };

    // Build Flux VAE decoder (F32 — no cast needed)
    println!("Building Flux VAE decoder...");
    let vae_wm = wm.prefix("vae");
    let vae_onnx = crate::models::diffusion::sd_common::build_flux_vae_decoder(
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
    let storage = super::shared::default_storage();

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
        crate::models::diffusion::flux::build_clip_l_pooled(wm, storage.clone(), Some(&clip_path))
    })?;

    // Build T5-XXL
    println!("Building T5-XXL encoder...");
    let t5_onnx = build_from_safetensors(&t5_path, |wm| {
        let config = crate::models::diffusion::t5::T5Config::t5_xxl(256);
        crate::models::diffusion::t5::load_t5_encoder_with_origin(
            wm,
            config,
            storage.clone(),
            Some(&t5_path),
        )
    })?;

    // Build Flux DiT
    println!("Building Flux DiT...");
    let flux_config = make_flux_config(img_size, has_guidance);
    let dit_onnx = if needs_dit_cast {
        build_from_safetensors(&dit_path, |wm| {
            let cast_wm = CastingWeightManager::new(wm, crate::onnx_graph::tensor::DType::BF16);
            crate::models::diffusion::flux::load_flux_dit_with_origin(
                cast_wm,
                flux_config,
                storage.clone(),
                Some(&dit_path),
            )
        })?
    } else {
        build_from_safetensors(&dit_path, |wm| {
            crate::models::diffusion::flux::load_flux_dit_with_origin(
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
        crate::models::diffusion::sd_common::build_flux_vae_decoder(
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
    let storage_dtype =
        crate::models::diffusion::sd_common::detect_model_dtype_with_canary(wm, canary);
    println!("Detected DiT storage dtype: {storage_dtype:?}");
    match storage_dtype {
        crate::onnx_graph::tensor::DType::F8E4M3FN => {
            Ok((whisper_tensor::dtype::DType::BF16, true))
        }
        crate::onnx_graph::tensor::DType::BF16 => Ok((whisper_tensor::dtype::DType::BF16, false)),
        crate::onnx_graph::tensor::DType::F16 => Ok((whisper_tensor::dtype::DType::F16, false)),
        crate::onnx_graph::tensor::DType::F32 => Ok((whisper_tensor::dtype::DType::F32, false)),
        other => Err(LoaderError::LoadFailed(anyhow::anyhow!(
            "Unsupported DiT storage dtype: {other:?}"
        ))),
    }
}

fn make_flux_config(
    img_size: usize,
    has_guidance: bool,
) -> crate::models::diffusion::flux::FluxConfig {
    if has_guidance {
        crate::models::diffusion::flux::FluxConfig::dev(img_size, 256)
    } else {
        crate::models::diffusion::flux::FluxConfig::schnell(img_size, 256)
    }
}

/// Build models and interface from the 4 ONNX blobs.
fn assemble_output(
    reference_path: &Path,
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
        build_flux_interface(
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

/// Build interface for Flux (CLIP-L + T5-XXL + DiT + VAE, rectified flow).
///
/// Model weights order: [clip_l, t5_xxl, dit, vae_decoder]
///
/// When `has_guidance` is true (Flux Dev), the DiT expects a guidance input
/// and `guidance_scale_input` is populated. When false (Schnell), no guidance.
#[allow(clippy::too_many_arguments)]
fn build_flux_interface(
    rng: &mut impl rand::Rng,
    clip_tokenizer: TokenizerInfo,
    t5_tokenizer: TokenizerInfo,
    model_dtype: whisper_tensor::dtype::DType,
    has_guidance: bool,
) -> whisper_tensor::interfaces::ImageGenerationInterface {
    use super::shared::interface_helpers::{
        build_cast_node, build_eos_indices_node, build_flux_denoising_loop, build_flux_vae_decode,
    };
    use whisper_tensor::interfaces::{ImageGenerationInterface, SchedulerType};
    use whisper_tensor::super_graph::SuperGraphBuilder;
    use whisper_tensor::super_graph::nodes::{
        SuperGraphNode, SuperGraphNodeModelExecution, SuperGraphNodeTensorToImage,
        SuperGraphNodeTokenizerEncode, SuperGraphNodeTokenizerEncodeMode,
        SuperGraphNodeTokenizerLoad,
    };

    let mut builder = SuperGraphBuilder::new();

    // Create input links
    let positive_prompt_input = builder.new_string_link(rng);
    let initial_latent_input = builder.new_tensor_link(rng);
    let timesteps_input = builder.new_tensor_link(rng);
    let dt_input = builder.new_tensor_link(rng);
    let sigmas_input = builder.new_tensor_link(rng);
    let iteration_count_input = builder.new_tensor_link(rng);
    let guidance_scale_link = if has_guidance {
        Some(builder.new_tensor_link(rng))
    } else {
        None
    };
    let clip_weights = builder.new_model_link(rng);
    let t5_weights = builder.new_model_link(rng);
    let dit_weights = builder.new_model_link(rng);
    let vae_weights = builder.new_model_link(rng);
    builder.set_link_label(positive_prompt_input, "prompt_positive");
    builder.set_link_label(initial_latent_input, "latent_initial");
    builder.set_link_label(timesteps_input, "timesteps");
    builder.set_link_label(dt_input, "dt");
    builder.set_link_label(sigmas_input, "sigmas");
    builder.set_link_label(iteration_count_input, "iteration_count");
    if let Some(gl) = guidance_scale_link {
        builder.set_link_label(gl, "guidance_scale");
    }
    builder.set_link_label(clip_weights, "clip_l_weights");
    builder.set_link_label(t5_weights, "t5_weights");
    builder.set_link_label(dit_weights, "dit_weights");
    builder.set_link_label(vae_weights, "vae_decoder_weights");

    // Prompt tokenization inside the supergraph.
    let clip_tokenizer_link =
        SuperGraphNodeTokenizerLoad::new_and_add(&mut builder, clip_tokenizer, rng);
    let t5_tokenizer_link =
        SuperGraphNodeTokenizerLoad::new_and_add(&mut builder, t5_tokenizer, rng);
    let cond_ids_input = SuperGraphNodeTokenizerEncode::new_with_mode_and_add(
        &mut builder,
        clip_tokenizer_link,
        positive_prompt_input,
        SuperGraphNodeTokenizerEncodeMode::ClipStyle {
            seq_len: 77,
            bos: 49406,
            eos: 49407,
            pad: 0,
        },
        rng,
    );
    let t5_ids_input = SuperGraphNodeTokenizerEncode::new_with_mode_and_add(
        &mut builder,
        t5_tokenizer_link,
        positive_prompt_input,
        SuperGraphNodeTokenizerEncodeMode::RawPad {
            seq_len: 256,
            pad: 0,
        },
        rng,
    );

    // --- CLIP-L: input_ids + eos_indices -> pooled_output [1, 768] ---
    let clip_pooled_f32 = builder.new_tensor_link(rng);
    let clip_eos = build_eos_indices_node(&mut builder, rng, cond_ids_input);
    let mut clip_node = SuperGraphNodeModelExecution::new(
        rng,
        clip_weights,
        0,
        vec![
            (cond_ids_input, "input_ids".to_string()),
            (clip_eos, "eos_indices".to_string()),
        ],
        vec![("pooled_output".to_string(), clip_pooled_f32)],
    );
    clip_node.label = Some("clip_l_encode".to_string());
    builder.add_node(clip_node.to_any());
    let clip_pooled = build_cast_node(&mut builder, rng, clip_pooled_f32, model_dtype);

    // --- T5-XXL: input_ids -> hidden_states [1, seq, 4096] ---
    let t5_hidden_f32 = builder.new_tensor_link(rng);
    let mut t5_node = SuperGraphNodeModelExecution::new(
        rng,
        t5_weights,
        1,
        vec![(t5_ids_input, "input_ids".to_string())],
        vec![("hidden_states".to_string(), t5_hidden_f32)],
    );
    t5_node.label = Some("t5_encode".to_string());
    builder.add_node(t5_node.to_any());
    let t5_hidden = build_cast_node(&mut builder, rng, t5_hidden_f32, model_dtype);

    // --- Denoising loop (rectified flow) ---
    let final_latent = build_flux_denoising_loop(
        &mut builder,
        rng,
        dit_weights,
        clip_pooled,
        t5_hidden,
        initial_latent_input,
        timesteps_input,
        dt_input,
        sigmas_input,
        iteration_count_input,
        guidance_scale_link,
        model_dtype,
        2, // dit model index
    );

    // --- VAE decode ---
    // Flux VAE: latent / 0.3611 + 0.1159, then wrap tensor into Image
    let decoded_image_tensor =
        build_flux_vae_decode(&mut builder, rng, final_latent, vae_weights, 3);
    let image_output =
        SuperGraphNodeTensorToImage::new_and_add(&mut builder, decoded_image_tensor, rng);
    builder.set_link_label(image_output, "image_output");

    // Build outer graph
    let model_weights = vec![clip_weights, t5_weights, dit_weights, vae_weights];
    let mut input_links: Vec<_> = vec![
        positive_prompt_input.to_any(),
        initial_latent_input.to_any(),
        timesteps_input.to_any(),
        dt_input.to_any(),
        sigmas_input.to_any(),
        iteration_count_input.to_any(),
    ];
    if let Some(gl) = guidance_scale_link {
        input_links.push(gl.to_any());
    }
    input_links.extend([
        clip_weights.to_any(),
        t5_weights.to_any(),
        dit_weights.to_any(),
        vae_weights.to_any(),
    ]);
    let output_links: Vec<_> = vec![image_output.to_any()];
    let super_graph = builder.build(rng, &input_links, &output_links);

    ImageGenerationInterface {
        super_graph,
        positive_prompt_input,
        negative_prompt_input: None,
        initial_latent_input,
        timesteps_input,
        dt_input,
        sigmas_input,
        iteration_count_input,
        guidance_scale_input: guidance_scale_link,
        model_weights,
        image_output,
        scheduler: SchedulerType::RectifiedFlow,
        latent_channels: 16,
    }
}
