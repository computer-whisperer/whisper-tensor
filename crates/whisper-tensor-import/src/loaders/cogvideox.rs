use rand::Rng;
use std::path::PathBuf;
use std::sync::Arc;
use whisper_tensor::interfaces::{SchedulerType, VideoGenerationInterface};
use whisper_tensor::loader::*;
use whisper_tensor::metadata::TokenizerInfo;
use whisper_tensor::milli_graph::MilliOpGraph;
use whisper_tensor::milli_graph::ops::{Cast, Constant, SimpleBinary};
use whisper_tensor::model::Model;
use whisper_tensor::numeric_dtype::NumericDType;
use whisper_tensor::super_graph::links::{SuperGraphLinkDouble, SuperGraphLinkTriple};
use whisper_tensor::super_graph::nodes::{
    SuperGraphNode, SuperGraphNodeMilliOpGraph, SuperGraphNodeModelExecution,
    SuperGraphNodeReportProgress, SuperGraphNodeScan, SuperGraphNodeTensorToVideoClip,
    SuperGraphNodeTokenizerEncode, SuperGraphNodeTokenizerEncodeMode, SuperGraphNodeTokenizerLoad,
};
use whisper_tensor::super_graph::{SuperGraphBuilder, SuperGraphLink};

use super::shared::interface_helpers::{
    build_cast_node, build_input_prep, build_progress_init, build_step_increment, build_zeros_like,
};

use crate::onnx_graph::weights::SafetensorsWeightManager;

/// Loader for CogVideoX models in HuggingFace Diffusers format.
///
/// Expects a directory containing:
/// - `transformer/` — DiT safetensors + config.json
/// - `vae/` — 3D VAE safetensors + config.json
/// - `text_encoder/` — T5-XXL safetensors
/// - `scheduler/scheduler_config.json`
/// - `tokenizer/` — T5 tokenizer files
pub struct CogVideoXLoader;

impl Loader for CogVideoXLoader {
    fn name(&self) -> &str {
        "CogVideoX"
    }

    fn description(&self) -> &str {
        "Load CogVideoX text-to-video model from a HuggingFace Diffusers directory"
    }

    fn config_schema(&self) -> Vec<ConfigField> {
        vec![ConfigField {
            key: "path".to_string(),
            label: "Model Directory".to_string(),
            description: "Path to the CogVideoX model directory (e.g. THUDM/CogVideoX-2b)"
                .to_string(),
            field_type: ConfigFieldType::FilePath,
            required: true,
            default: None,
        }]
    }

    fn load(&self, config: ConfigValues) -> Result<LoaderOutput, LoaderError> {
        let base_path = require_path(&config, "path")?;
        load_cogvideox(base_path)
    }
}

fn load_cogvideox(base_path: PathBuf) -> Result<LoaderOutput, LoaderError> {
    use memmap2::Mmap;

    let storage = super::shared::default_storage();

    // Detect variant from transformer config
    let transformer_dir = base_path.join("transformer");
    let vae_dir = base_path.join("vae");
    let text_encoder_dir = base_path.join("text_encoder");

    // Read transformer config to detect 2B vs 5B
    let transformer_config_path = transformer_dir.join("config.json");
    let transformer_config_json: serde_json::Value = {
        let data = std::fs::read_to_string(&transformer_config_path)
            .map_err(|e| LoaderError::LoadFailed(e.into()))?;
        serde_json::from_str(&data).map_err(|e| LoaderError::LoadFailed(e.into()))?
    };
    let num_attention_heads = transformer_config_json["num_attention_heads"]
        .as_u64()
        .unwrap_or(30) as usize;
    let use_rope = transformer_config_json["use_rotary_positional_embeddings"]
        .as_bool()
        .unwrap_or(false);

    let sample_frames = transformer_config_json["sample_frames"]
        .as_u64()
        .unwrap_or(49) as usize;

    let (variant, dit_config, vae_config) = if num_attention_heads >= 48 || use_rope {
        if sample_frames > 49 {
            (
                "1.5-5b",
                crate::models::diffusion::cogvideox::CogVideoXTransformerConfig::cogvideox_1_5_5b(),
                crate::models::diffusion::cogvideox::CogVideoXVaeConfig::cogvideox_5b(),
            )
        } else {
            (
                "5b",
                crate::models::diffusion::cogvideox::CogVideoXTransformerConfig::cogvideox_5b(),
                crate::models::diffusion::cogvideox::CogVideoXVaeConfig::cogvideox_5b(),
            )
        }
    } else {
        (
            "2b",
            crate::models::diffusion::cogvideox::CogVideoXTransformerConfig::cogvideox_2b(),
            crate::models::diffusion::cogvideox::CogVideoXVaeConfig::cogvideox_2b(),
        )
    };
    println!("Detected CogVideoX-{variant}");

    // Load safetensors files from a directory (may be sharded)
    let load_safetensors_dir =
        |dir: &std::path::Path| -> Result<SafetensorsWeightManager, LoaderError> {
            let mut mmaps = Vec::new();
            let mut paths = Vec::new();
            // Collect all .safetensors files in the directory
            let mut entries: Vec<_> = std::fs::read_dir(dir)
                .map_err(|e| LoaderError::LoadFailed(e.into()))?
                .filter_map(|e| e.ok())
                .filter(|e| e.path().extension().is_some_and(|ext| ext == "safetensors"))
                .collect();
            entries.sort_by_key(|e| e.path());
            for entry in entries {
                let path = entry.path();
                let file =
                    std::fs::File::open(&path).map_err(|e| LoaderError::LoadFailed(e.into()))?;
                let mmap =
                    unsafe { Mmap::map(&file) }.map_err(|e| LoaderError::LoadFailed(e.into()))?;
                mmaps.push(Arc::new(mmap));
                paths.push(path);
            }
            if mmaps.is_empty() {
                return Err(LoaderError::LoadFailed(anyhow::anyhow!(
                    "No .safetensors files found in {}",
                    dir.display()
                )));
            }
            SafetensorsWeightManager::new_with_paths(mmaps, paths)
                .map_err(|e| LoaderError::LoadFailed(e.into()))
        };

    // Build T5-XXL encoder
    println!("Building T5-XXL encoder...");
    let t5_wm = load_safetensors_dir(&text_encoder_dir)?;
    let t5_config = crate::models::diffusion::t5::T5Config::t5_xxl(226);
    let t5_onnx = crate::models::diffusion::t5::load_t5_encoder(t5_wm, t5_config, storage.clone())
        .map_err(LoaderError::LoadFailed)?;

    // Build DiT transformer
    println!("Building CogVideoX DiT transformer...");
    let num_frames = dit_config.sample_frames;
    let dit_wm = load_safetensors_dir(&transformer_dir)?;
    let dit_onnx = crate::models::diffusion::cogvideox::load_cogvideox_transformer(
        dit_wm,
        dit_config,
        storage.clone(),
    )
    .map_err(LoaderError::LoadFailed)?;

    // Build VAE decoder
    println!("Building CogVideoX VAE decoder...");
    let vae_wm = load_safetensors_dir(&vae_dir)?;
    let vae_onnx = crate::models::diffusion::cogvideox::load_cogvideox_vae_decoder(
        vae_wm,
        vae_config,
        storage.clone(),
    )
    .map_err(LoaderError::LoadFailed)?;

    // Create Model objects
    let base_dir = Some(base_path.as_path());
    let mut models = Vec::new();
    for (suffix, onnx_data) in [
        ("t5_xxl", t5_onnx),
        ("dit", dit_onnx),
        ("vae_decoder", vae_onnx),
    ] {
        let mut rng = rand::rng();
        let model = Model::new_from_onnx(&onnx_data, &mut rng, base_dir)
            .map_err(|e| LoaderError::LoadFailed(e.into()))?;
        models.push(LoadedModel {
            name: format!("cogvideox-{variant}-{suffix}"),
            model: Arc::new(model),
        });
    }

    // Build VideoGenerationInterface
    let interface = {
        let mut rng = rand::rng();
        build_cogvideox_interface(
            &mut rng,
            TokenizerInfo::HFTokenizer("google-t5/t5-base".to_string()),
            NumericDType::BF16,
            num_frames,
        )
    };

    let interfaces = vec![LoadedInterface {
        name: format!("cogvideox-{variant}-VideoGeneration"),
        interface: interface.to_any(),
    }];

    Ok(LoaderOutput { models, interfaces })
}

// =============================================================================
// Interface builder
// =============================================================================

fn build_cogvideox_interface(
    rng: &mut impl Rng,
    t5_tokenizer: TokenizerInfo,
    model_dtype: NumericDType,
    num_frames: usize,
) -> VideoGenerationInterface {
    let mut builder = SuperGraphBuilder::new();

    let positive_prompt_input = builder.new_string_link(rng);
    let initial_latent_input = builder.new_tensor_link(rng);
    let timesteps_input = builder.new_tensor_link(rng);
    let dt_input = builder.new_tensor_link(rng);
    let sigmas_input = builder.new_tensor_link(rng);
    let iteration_count_input = builder.new_tensor_link(rng);
    let guidance_scale_input = builder.new_tensor_link(rng);
    let t5_weights = builder.new_model_link(rng);
    let dit_weights = builder.new_model_link(rng);
    let vae_weights = builder.new_model_link(rng);

    builder.set_link_label(positive_prompt_input, "prompt_positive");
    builder.set_link_label(initial_latent_input, "latent_initial");
    builder.set_link_label(timesteps_input, "timesteps");
    builder.set_link_label(dt_input, "dt");
    builder.set_link_label(sigmas_input, "sigmas");
    builder.set_link_label(iteration_count_input, "iteration_count");
    builder.set_link_label(guidance_scale_input, "guidance_scale");
    builder.set_link_label(t5_weights, "t5_weights");
    builder.set_link_label(dit_weights, "dit_weights");
    builder.set_link_label(vae_weights, "vae_decoder_weights");

    // T5 tokenization
    let t5_tokenizer_link =
        SuperGraphNodeTokenizerLoad::new_and_add(&mut builder, t5_tokenizer, rng);
    let t5_ids_input = SuperGraphNodeTokenizerEncode::new_with_mode_and_add(
        &mut builder,
        t5_tokenizer_link,
        positive_prompt_input,
        SuperGraphNodeTokenizerEncodeMode::RawPad {
            seq_len: 226,
            pad: 0,
        },
        rng,
    );

    // T5 encode
    let t5_hidden_f32 = builder.new_tensor_link(rng);
    let mut t5_node = SuperGraphNodeModelExecution::new(
        rng,
        t5_weights,
        0,
        vec![(t5_ids_input, "input_ids".to_string())],
        vec![("hidden_states".to_string(), t5_hidden_f32)],
    );
    t5_node.label = Some("t5_encode".to_string());
    builder.add_node(t5_node.to_any());
    let t5_hidden = build_cast_node(&mut builder, rng, t5_hidden_f32, model_dtype);

    // Denoising loop (v-prediction DDIM)
    let final_latent = build_cogvideox_denoising_loop(
        &mut builder,
        rng,
        dit_weights,
        t5_hidden,
        initial_latent_input,
        timesteps_input,
        dt_input,
        sigmas_input,
        iteration_count_input,
        guidance_scale_input,
        model_dtype,
        1,
    );

    // VAE decode
    let decoded_tensor = builder.new_tensor_link(rng);
    let mut vae_node = SuperGraphNodeModelExecution::new(
        rng,
        vae_weights,
        2,
        vec![(final_latent, "latent".to_string())],
        vec![("video_out".to_string(), decoded_tensor)],
    );
    vae_node.label = Some("vae_decode".to_string());
    builder.add_node(vae_node.to_any());

    let video_output =
        SuperGraphNodeTensorToVideoClip::new_and_add(&mut builder, decoded_tensor, 8.0, rng);
    builder.set_link_label(video_output, "video_output");

    let model_weights = vec![t5_weights, dit_weights, vae_weights];
    let input_links: Vec<_> = vec![
        positive_prompt_input.to_any(),
        initial_latent_input.to_any(),
        timesteps_input.to_any(),
        dt_input.to_any(),
        sigmas_input.to_any(),
        iteration_count_input.to_any(),
        guidance_scale_input.to_any(),
        t5_weights.to_any(),
        dit_weights.to_any(),
        vae_weights.to_any(),
    ];
    let output_links: Vec<_> = vec![video_output.to_any()];
    let super_graph = builder.build(rng, &input_links, &output_links);

    VideoGenerationInterface {
        super_graph,
        positive_prompt_input,
        negative_prompt_input: None,
        initial_latent_input,
        timesteps_input,
        dt_input,
        sigmas_input,
        iteration_count_input,
        guidance_scale_input: Some(guidance_scale_input),
        model_weights,
        video_output,
        scheduler: SchedulerType::DDIMVPrediction,
        latent_channels: 16,
        fps: 8.0,
        num_frames,
    }
}

// =============================================================================
// CogVideoX v-prediction DDIM denoising loop
// =============================================================================

#[allow(clippy::too_many_arguments)]
fn build_cogvideox_denoising_loop(
    builder: &mut SuperGraphBuilder,
    rng: &mut impl Rng,
    dit_weights: SuperGraphLink,
    t5_hidden: SuperGraphLink,
    initial_latent_input: SuperGraphLink,
    timesteps_input: SuperGraphLink,
    dt_input: SuperGraphLink,
    sigmas_input: SuperGraphLink,
    iteration_count_input: SuperGraphLink,
    guidance_scale_input: SuperGraphLink,
    model_dtype: NumericDType,
    dit_model_index: usize,
) -> SuperGraphLink {
    let outer_final_latent = builder.new_tensor_link(rng);
    let progress_tier_link = builder.new_tensor_link(rng);
    builder.set_link_label(outer_final_latent, "latent_final");
    builder.set_link_label(progress_tier_link, "progress_tier");
    build_progress_init(builder, rng, progress_tier_link, "cogvideo_progress_init");

    let mut inner_builder = SuperGraphBuilder::new();

    let inner_dit_weights = inner_builder.new_model_link(rng);
    let inner_t5_hidden = inner_builder.new_tensor_link(rng);
    let inner_guidance_scale = inner_builder.new_tensor_link(rng);
    let inner_latent_in = inner_builder.new_tensor_link(rng);
    let inner_latent_out = inner_builder.new_tensor_link(rng);
    let inner_timestep = inner_builder.new_tensor_link(rng);
    let inner_alpha_sigma = inner_builder.new_tensor_link(rng);
    let inner_alpha_sigma_prev = inner_builder.new_tensor_link(rng);
    let inner_progress_tier = inner_builder.new_tensor_link(rng);
    let inner_total_steps = inner_builder.new_tensor_link(rng);
    let inner_step_in = inner_builder.new_tensor_link(rng);
    let inner_step_out = inner_builder.new_tensor_link(rng);

    // Prep
    let cast_latent = inner_builder.new_tensor_link(rng);
    let cast_timestep = inner_builder.new_tensor_link(rng);
    build_input_prep(
        &mut inner_builder,
        rng,
        inner_latent_in,
        inner_timestep,
        cast_latent,
        cast_timestep,
        model_dtype,
        "cogvideo_input_prep",
    );

    // Zero text embeddings
    let zero_t5_hidden = inner_builder.new_tensor_link(rng);
    build_zeros_like(&mut inner_builder, rng, inner_t5_hidden, zero_t5_hidden);

    // DiT unconditional
    let uncond_output = inner_builder.new_tensor_link(rng);
    {
        let mut node = SuperGraphNodeModelExecution::new(
            rng,
            inner_dit_weights,
            dit_model_index,
            vec![
                (cast_latent, "hidden_states".to_string()),
                (zero_t5_hidden, "encoder_hidden_states".to_string()),
                (cast_timestep, "timestep".to_string()),
            ],
            vec![("out_sample".to_string(), uncond_output)],
        );
        node.label = Some("dit_unconditional".to_string());
        inner_builder.add_node(node.to_any());
    }

    // DiT conditional
    let cond_output = inner_builder.new_tensor_link(rng);
    {
        let mut node = SuperGraphNodeModelExecution::new(
            rng,
            inner_dit_weights,
            dit_model_index,
            vec![
                (cast_latent, "hidden_states".to_string()),
                (inner_t5_hidden, "encoder_hidden_states".to_string()),
                (cast_timestep, "timestep".to_string()),
            ],
            vec![("out_sample".to_string(), cond_output)],
        );
        node.label = Some("dit_conditional".to_string());
        inner_builder.add_node(node.to_any());
    }

    // CFG + v-prediction DDIM step
    {
        let (mut mg, input_map) = MilliOpGraph::new(
            [
                uncond_output.global_id(),
                cond_output.global_id(),
                inner_latent_in.global_id(),
                inner_guidance_scale.global_id(),
                inner_alpha_sigma.global_id(),
                inner_alpha_sigma_prev.global_id(),
            ],
            rng,
        );
        let uncond_in = *input_map.get(&uncond_output.global_id()).unwrap();
        let cond_in = *input_map.get(&cond_output.global_id()).unwrap();
        let lat_in = *input_map.get(&inner_latent_in.global_id()).unwrap();
        let gs_in = *input_map.get(&inner_guidance_scale.global_id()).unwrap();
        let as_in = *input_map.get(&inner_alpha_sigma.global_id()).unwrap();
        let as_prev_in = *input_map.get(&inner_alpha_sigma_prev.global_id()).unwrap();

        let uncond_f32 = Cast::push_new(&mut mg, uncond_in, NumericDType::F32, rng);
        let cond_f32 = Cast::push_new(&mut mg, cond_in, NumericDType::F32, rng);

        let diff = SimpleBinary::sub(&mut mg, cond_f32, uncond_f32, rng);
        let scaled = SimpleBinary::mul(&mut mg, diff, gs_in, rng);
        let v_pred = SimpleBinary::add(&mut mg, uncond_f32, scaled, rng);

        let idx_0 = Constant::from_vec(&mut mg, vec![0i64], rng);
        let idx_1 = Constant::from_vec(&mut mg, vec![1i64], rng);
        let alpha_t =
            whisper_tensor::milli_graph::ops::Gather::push_new(&mut mg, as_in, idx_0, 0, rng);
        let sigma_t =
            whisper_tensor::milli_graph::ops::Gather::push_new(&mut mg, as_in, idx_1, 0, rng);
        let alpha_prev =
            whisper_tensor::milli_graph::ops::Gather::push_new(&mut mg, as_prev_in, idx_0, 0, rng);
        let sigma_prev =
            whisper_tensor::milli_graph::ops::Gather::push_new(&mut mg, as_prev_in, idx_1, 0, rng);

        let ax = SimpleBinary::mul(&mut mg, alpha_t, lat_in, rng);
        let sv = SimpleBinary::mul(&mut mg, sigma_t, v_pred, rng);
        let x0_pred = SimpleBinary::sub(&mut mg, ax, sv, rng);

        let av = SimpleBinary::mul(&mut mg, alpha_t, v_pred, rng);
        let sx = SimpleBinary::mul(&mut mg, sigma_t, lat_in, rng);
        let eps_pred = SimpleBinary::add(&mut mg, av, sx, rng);

        let a_x0 = SimpleBinary::mul(&mut mg, alpha_prev, x0_pred, rng);
        let s_eps = SimpleBinary::mul(&mut mg, sigma_prev, eps_pred, rng);
        let latent_next = SimpleBinary::add(&mut mg, a_x0, s_eps, rng);

        mg.set_output_map(std::iter::once((latent_next, inner_latent_out.global_id())));
        let mut node = SuperGraphNodeMilliOpGraph::new(mg, rng);
        node.label = Some("cfg_ddim_vpred_step".to_string());
        inner_builder.add_node(node.to_any());
    }

    build_step_increment(&mut inner_builder, rng, inner_step_in, inner_step_out);

    let mut report = SuperGraphNodeReportProgress::new(
        inner_progress_tier,
        inner_step_out,
        inner_total_steps,
        rng,
    );
    report.label = Some("cogvideo_progress".to_string());
    inner_builder.add_node(report.to_any());

    let inner_inputs: Vec<_> = vec![
        inner_dit_weights.to_any(),
        inner_t5_hidden.to_any(),
        inner_guidance_scale.to_any(),
        inner_progress_tier.to_any(),
        inner_total_steps.to_any(),
        inner_latent_in.to_any(),
        inner_step_in.to_any(),
        inner_timestep.to_any(),
        inner_alpha_sigma.to_any(),
        inner_alpha_sigma_prev.to_any(),
    ];
    let inner_outputs: Vec<_> = vec![inner_latent_out.to_any(), inner_step_out.to_any()];
    let inner_graph = inner_builder.build(rng, &inner_inputs, &inner_outputs);

    let mut scan_node = SuperGraphNodeScan::new(
        inner_graph,
        iteration_count_input,
        vec![
            SuperGraphLinkDouble::new(dit_weights, inner_dit_weights),
            SuperGraphLinkDouble::new(t5_hidden, inner_t5_hidden),
            SuperGraphLinkDouble::new(guidance_scale_input, inner_guidance_scale),
            SuperGraphLinkDouble::new(progress_tier_link, inner_progress_tier),
            SuperGraphLinkDouble::new(iteration_count_input, inner_total_steps),
        ],
        vec![
            SuperGraphLinkTriple::new(initial_latent_input, inner_latent_in, inner_latent_out),
            SuperGraphLinkTriple::new(progress_tier_link, inner_step_in, inner_step_out),
        ],
        vec![
            (timesteps_input, inner_timestep, 0),
            (sigmas_input, inner_alpha_sigma, 0),
            (dt_input, inner_alpha_sigma_prev, 0),
        ],
        vec![],
        vec![SuperGraphLinkDouble::new(
            inner_latent_out,
            outer_final_latent,
        )],
        rng,
    );
    scan_node.label = Some("cogvideo_denoise_scan".to_string());
    builder.add_node(scan_node.to_any());

    outer_final_latent
}
