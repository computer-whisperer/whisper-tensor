use rand::Rng;
use std::path::PathBuf;
use std::sync::Arc;
use whisper_tensor::dtype::DType;
use whisper_tensor::interfaces::{SchedulerType, VideoGenerationInterface};
use whisper_tensor::loader::*;
use whisper_tensor::metadata::TokenizerInfo;
use whisper_tensor::milli_graph::MilliOpGraph;
use whisper_tensor::milli_graph::ops::Cast;
use whisper_tensor::model::Model;
use whisper_tensor::super_graph::links::{SuperGraphLinkDouble, SuperGraphLinkTriple};
use whisper_tensor::super_graph::nodes::{
    SuperGraphNode, SuperGraphNodeMilliOpGraph, SuperGraphNodeModelExecution,
    SuperGraphNodeReportProgress, SuperGraphNodeScan, SuperGraphNodeTensorToVideoClip,
    SuperGraphNodeTokenizerEncode, SuperGraphNodeTokenizerEncodeMode, SuperGraphNodeTokenizerLoad,
};
use whisper_tensor::super_graph::{SuperGraphBuilder, SuperGraphLink};

use super::shared::interface_helpers::{
    build_cast_node, build_eos_indices_node, build_progress_init, build_step_increment,
};

use crate::onnx_graph::weights::SafetensorsWeightManager;

/// Loader for HunyuanVideo models in HuggingFace Diffusers format.
///
/// Expects a directory containing:
/// - `transformer/` — DiT safetensors + config.json
/// - `vae/` — 3D VAE safetensors
/// - `text_encoder/` — LLaMA 3 safetensors
/// - `text_encoder_2/` — CLIP safetensors
pub struct HunyuanVideoLoader;

impl Loader for HunyuanVideoLoader {
    fn name(&self) -> &str {
        "HunyuanVideo"
    }

    fn description(&self) -> &str {
        "Load HunyuanVideo text-to-video model from a HuggingFace Diffusers directory"
    }

    fn config_schema(&self) -> Vec<ConfigField> {
        vec![ConfigField {
            key: "path".to_string(),
            label: "Model Directory".to_string(),
            description:
                "Path to the HunyuanVideo model directory (e.g. hunyuanvideo-community/HunyuanVideo)"
                    .to_string(),
            field_type: ConfigFieldType::FilePath,
            required: true,
            default: None,
        }]
    }

    fn load(&self, config: ConfigValues) -> Result<LoaderOutput, LoaderError> {
        let base_path = require_path(&config, "path")?;
        load_hunyuan_video(base_path)
    }
}

fn load_hunyuan_video(base_path: PathBuf) -> Result<LoaderOutput, LoaderError> {
    use memmap2::Mmap;

    let storage = super::shared::default_storage();

    let transformer_dir = base_path.join("transformer");
    let vae_dir = base_path.join("vae");
    let text_encoder_dir = base_path.join("text_encoder");
    let text_encoder_2_dir = base_path.join("text_encoder_2");

    println!("Detected HunyuanVideo");

    let dit_config =
        crate::models::diffusion::hunyuan_video::HunyuanVideoTransformerConfig::default_config();
    let vae_config =
        crate::models::diffusion::hunyuan_video::HunyuanVideoVaeConfig::default_config();

    let load_safetensors_dir =
        |dir: &std::path::Path| -> Result<SafetensorsWeightManager, LoaderError> {
            let mut mmaps = Vec::new();
            let mut paths = Vec::new();
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

    // Build LLaMA 3 encoder (primary text encoder — hidden states from layer -3)
    println!("Building LLaMA 3 text encoder...");
    let llama_wm = load_safetensors_dir(&text_encoder_dir)?;
    let llama_onnx = crate::models::diffusion::hunyuan_video::load_llama3_encoder(
        llama_wm,
        32,
        2,
        256,
        storage.clone(),
    )
    .map_err(LoaderError::LoadFailed)?;

    // Build CLIP encoder (secondary — pooled output only)
    println!("Building CLIP text encoder...");
    let clip_wm = load_safetensors_dir(&text_encoder_2_dir)?;
    let clip_config = crate::models::diffusion::sd_clip::ClipTextModelConfig {
        hidden_dim: 768,
        num_heads: 12,
        num_layers: 12,
        max_position: 77,
        layer_norm_eps: 1e-5,
        mlp_activation: crate::models::diffusion::sd_clip::ClipMlpActivation::QuickGelu,
        hidden_source: crate::models::diffusion::sd_clip::ClipHiddenStateSource::FinalLayerNorm,
        output_pooled_projection: true,
    };
    let clip_onnx = crate::models::diffusion::sd_clip::build_clip_text_model_with_projection(
        clip_wm,
        DType::F32,
        storage.clone(),
        &text_encoder_2_dir,
        "text_model",
        clip_config,
    )
    .map_err(LoaderError::LoadFailed)?;

    // Build DiT transformer
    println!("Building HunyuanVideo DiT transformer...");
    let dit_wm = load_safetensors_dir(&transformer_dir)?;
    let dit_onnx = crate::models::diffusion::hunyuan_video::load_hunyuan_video_transformer(
        dit_wm,
        dit_config,
        storage.clone(),
    )
    .map_err(LoaderError::LoadFailed)?;

    // Build VAE decoder
    println!("Building HunyuanVideo VAE decoder...");
    let vae_wm = load_safetensors_dir(&vae_dir)?;
    let vae_onnx = crate::models::diffusion::hunyuan_video::load_hunyuan_video_vae_decoder(
        vae_wm,
        vae_config,
        storage.clone(),
    )
    .map_err(LoaderError::LoadFailed)?;

    // Create Model objects
    let base_dir = Some(base_path.as_path());
    let mut models = Vec::new();
    for (suffix, onnx_data) in [
        ("llama3_encoder", llama_onnx),
        ("clip", clip_onnx),
        ("dit", dit_onnx),
        ("vae_decoder", vae_onnx),
    ] {
        let mut rng = rand::rng();
        let model = Model::new_from_onnx(&onnx_data, &mut rng, base_dir)
            .map_err(|e| LoaderError::LoadFailed(e.into()))?;
        models.push(LoadedModel {
            name: format!("hunyuan-video-{suffix}"),
            model: Arc::new(model),
        });
    }

    // Build VideoGenerationInterface
    let interface = {
        let mut rng = rand::rng();
        build_hunyuan_video_interface(&mut rng, DType::BF16)
    };

    let interfaces = vec![LoadedInterface {
        name: "hunyuan-video-VideoGeneration".to_string(),
        interface: interface.to_any(),
    }];

    Ok(LoaderOutput { models, interfaces })
}

// =============================================================================
// Interface builder
// =============================================================================

fn build_hunyuan_video_interface(
    rng: &mut impl Rng,
    model_dtype: DType,
) -> VideoGenerationInterface {
    let mut builder = SuperGraphBuilder::new();

    let positive_prompt_input = builder.new_string_link(rng);
    let initial_latent_input = builder.new_tensor_link(rng);
    let timesteps_input = builder.new_tensor_link(rng);
    let dt_input = builder.new_tensor_link(rng);
    let sigmas_input = builder.new_tensor_link(rng);
    let iteration_count_input = builder.new_tensor_link(rng);
    let guidance_scale_input = builder.new_tensor_link(rng);
    let llama_weights = builder.new_model_link(rng);
    let clip_weights = builder.new_model_link(rng);
    let dit_weights = builder.new_model_link(rng);
    let vae_weights = builder.new_model_link(rng);

    builder.set_link_label(positive_prompt_input, "prompt_positive");
    builder.set_link_label(initial_latent_input, "latent_initial");
    builder.set_link_label(timesteps_input, "timesteps");
    builder.set_link_label(dt_input, "dt");
    builder.set_link_label(sigmas_input, "sigmas");
    builder.set_link_label(iteration_count_input, "iteration_count");
    builder.set_link_label(guidance_scale_input, "guidance_scale");
    builder.set_link_label(llama_weights, "llama_weights");
    builder.set_link_label(clip_weights, "clip_weights");
    builder.set_link_label(dit_weights, "dit_weights");
    builder.set_link_label(vae_weights, "vae_decoder_weights");

    // LLaMA tokenization + encoding
    let llama_tokenizer = TokenizerInfo::HFTokenizer("llava-hf/llava-llama-3-8b-v1_1".to_string());
    let llama_tokenizer_link =
        SuperGraphNodeTokenizerLoad::new_and_add(&mut builder, llama_tokenizer, rng);
    let llama_ids = SuperGraphNodeTokenizerEncode::new_with_mode_and_add(
        &mut builder,
        llama_tokenizer_link,
        positive_prompt_input,
        SuperGraphNodeTokenizerEncodeMode::RawPad {
            seq_len: 256,
            pad: 0,
        },
        rng,
    );

    let llama_hidden_f32 = builder.new_tensor_link(rng);
    let mut llama_node = SuperGraphNodeModelExecution::new(
        rng,
        llama_weights,
        0,
        vec![(llama_ids, "input_ids".to_string())],
        vec![("hidden_states".to_string(), llama_hidden_f32)],
    );
    llama_node.label = Some("llama_encode".to_string());
    builder.add_node(llama_node.to_any());
    let llama_hidden = build_cast_node(&mut builder, rng, llama_hidden_f32, model_dtype);

    // CLIP tokenization + encoding
    let clip_tokenizer = TokenizerInfo::HFTokenizer("openai/clip-vit-large-patch14".to_string());
    let clip_tokenizer_link =
        SuperGraphNodeTokenizerLoad::new_and_add(&mut builder, clip_tokenizer, rng);
    let clip_ids = SuperGraphNodeTokenizerEncode::new_with_mode_and_add(
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
    let clip_eos_indices = build_eos_indices_node(&mut builder, rng, clip_ids);

    let clip_hidden_f32 = builder.new_tensor_link(rng);
    let clip_pooled_f32 = builder.new_tensor_link(rng);
    let mut clip_node = SuperGraphNodeModelExecution::new(
        rng,
        clip_weights,
        1,
        vec![
            (clip_ids, "input_ids".to_string()),
            (clip_eos_indices, "eos_indices".to_string()),
        ],
        vec![
            ("last_hidden_state".to_string(), clip_hidden_f32),
            ("pooled_output".to_string(), clip_pooled_f32),
        ],
    );
    clip_node.label = Some("clip_encode".to_string());
    builder.add_node(clip_node.to_any());
    let clip_pooled = build_cast_node(&mut builder, rng, clip_pooled_f32, model_dtype);

    // Denoising loop (embedded guidance, single pass — like Flux)
    let final_latent = build_hunyuan_denoising_loop(
        &mut builder,
        rng,
        dit_weights,
        llama_hidden,
        clip_pooled,
        initial_latent_input,
        timesteps_input,
        dt_input,
        iteration_count_input,
        guidance_scale_input,
        model_dtype,
        2,
    );

    // VAE decode
    let decoded_tensor = builder.new_tensor_link(rng);
    let mut vae_node = SuperGraphNodeModelExecution::new(
        rng,
        vae_weights,
        3,
        vec![(final_latent, "latent".to_string())],
        vec![("video_out".to_string(), decoded_tensor)],
    );
    vae_node.label = Some("vae_decode".to_string());
    builder.add_node(vae_node.to_any());

    let video_output =
        SuperGraphNodeTensorToVideoClip::new_and_add(&mut builder, decoded_tensor, 15.0, rng);
    builder.set_link_label(video_output, "video_output");

    let model_weights = vec![llama_weights, clip_weights, dit_weights, vae_weights];
    let input_links: Vec<_> = vec![
        positive_prompt_input.to_any(),
        initial_latent_input.to_any(),
        timesteps_input.to_any(),
        dt_input.to_any(),
        sigmas_input.to_any(),
        iteration_count_input.to_any(),
        guidance_scale_input.to_any(),
        llama_weights.to_any(),
        clip_weights.to_any(),
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
        scheduler: SchedulerType::RectifiedFlow,
        latent_channels: 16,
        fps: 15.0,
        num_frames: 49,
    }
}

// =============================================================================
// HunyuanVideo denoising loop (embedded guidance, single pass)
// =============================================================================

#[allow(clippy::too_many_arguments)]
fn build_hunyuan_denoising_loop(
    builder: &mut SuperGraphBuilder,
    rng: &mut impl Rng,
    dit_weights: SuperGraphLink,
    llama_hidden: SuperGraphLink,
    clip_pooled: SuperGraphLink,
    initial_latent_input: SuperGraphLink,
    timesteps_input: SuperGraphLink,
    dt_input: SuperGraphLink,
    iteration_count_input: SuperGraphLink,
    guidance_scale_input: SuperGraphLink,
    model_dtype: DType,
    dit_model_index: usize,
) -> SuperGraphLink {
    use whisper_tensor::backends::ndarray_backend::NDArrayNumericTensor;
    use whisper_tensor::milli_graph::ops::{Constant, SimpleBinary};

    let outer_final_latent = builder.new_tensor_link(rng);
    let progress_tier_link = builder.new_tensor_link(rng);
    builder.set_link_label(outer_final_latent, "latent_final");
    builder.set_link_label(progress_tier_link, "progress_tier");
    build_progress_init(builder, rng, progress_tier_link, "hunyuan_progress_init");

    let mut inner_builder = SuperGraphBuilder::new();

    let inner_dit_weights = inner_builder.new_model_link(rng);
    let inner_llama_hidden = inner_builder.new_tensor_link(rng);
    let inner_clip_pooled = inner_builder.new_tensor_link(rng);
    let inner_guidance = inner_builder.new_tensor_link(rng);
    let inner_latent_in = inner_builder.new_tensor_link(rng);
    let inner_latent_out = inner_builder.new_tensor_link(rng);
    let inner_timestep = inner_builder.new_tensor_link(rng);
    let inner_dt = inner_builder.new_tensor_link(rng);
    let inner_progress_tier = inner_builder.new_tensor_link(rng);
    let inner_total_steps = inner_builder.new_tensor_link(rng);
    let inner_step_in = inner_builder.new_tensor_link(rng);
    let inner_step_out = inner_builder.new_tensor_link(rng);

    // Input prep: cast latent, reshape timestep
    let cast_latent = inner_builder.new_tensor_link(rng);
    let cast_timestep = inner_builder.new_tensor_link(rng);
    super::shared::interface_helpers::build_input_prep(
        &mut inner_builder,
        rng,
        inner_latent_in,
        inner_timestep,
        cast_latent,
        cast_timestep,
        model_dtype,
        "hunyuan_input_prep",
    );

    // Scale guidance: guidance_scale * 1000.0, reshape to [1]
    let cast_guidance = inner_builder.new_tensor_link(rng);
    {
        let (mut mg, input_map) =
            MilliOpGraph::new(std::iter::once(inner_guidance.global_id()), rng);
        let g_in = *input_map.get(&inner_guidance.global_id()).unwrap();
        let scale = Constant::push_new(
            &mut mg,
            NDArrayNumericTensor::from_vec_shape(vec![1000.0f32], &vec![1]).unwrap(),
            rng,
        );
        let scaled = SimpleBinary::mul(&mut mg, g_in, scale, rng);
        let shape = Constant::push_new(
            &mut mg,
            NDArrayNumericTensor::from_vec_shape(vec![1i64], &vec![1]).unwrap(),
            rng,
        );
        let reshaped =
            whisper_tensor::milli_graph::ops::Reshape::push_new(&mut mg, scaled, shape, false, rng);
        mg.set_output_map(std::iter::once((reshaped, cast_guidance.global_id())));
        let mut node = whisper_tensor::super_graph::nodes::SuperGraphNodeMilliOpGraph::new(mg, rng);
        node.label = Some("guidance_scale_prep".to_string());
        inner_builder.add_node(node.to_any());
    }

    // DiT single forward pass (embedded guidance, not CFG)
    let dit_output = inner_builder.new_tensor_link(rng);
    {
        let mut node = SuperGraphNodeModelExecution::new(
            rng,
            inner_dit_weights,
            dit_model_index,
            vec![
                (cast_latent, "hidden_states".to_string()),
                (inner_llama_hidden, "encoder_hidden_states".to_string()),
                (cast_timestep, "timestep".to_string()),
                (inner_clip_pooled, "pooled_projections".to_string()),
                (cast_guidance, "guidance".to_string()),
            ],
            vec![("out_sample".to_string(), dit_output)],
        );
        node.label = Some("dit_forward".to_string());
        inner_builder.add_node(node.to_any());
    }

    // Euler step: latent_next = latent + velocity * dt
    {
        let (mut mg, input_map) = MilliOpGraph::new(
            [
                dit_output.global_id(),
                inner_latent_in.global_id(),
                inner_dt.global_id(),
            ],
            rng,
        );
        let velocity = *input_map.get(&dit_output.global_id()).unwrap();
        let lat_in = *input_map.get(&inner_latent_in.global_id()).unwrap();
        let dt_in = *input_map.get(&inner_dt.global_id()).unwrap();

        let vel_f32 = Cast::push_new(&mut mg, velocity, DType::F32, rng);
        let step = SimpleBinary::mul(&mut mg, vel_f32, dt_in, rng);
        let latent_next = SimpleBinary::add(&mut mg, lat_in, step, rng);

        mg.set_output_map(std::iter::once((latent_next, inner_latent_out.global_id())));
        let mut node = whisper_tensor::super_graph::nodes::SuperGraphNodeMilliOpGraph::new(mg, rng);
        node.label = Some("euler_step".to_string());
        inner_builder.add_node(node.to_any());
    }

    build_step_increment(&mut inner_builder, rng, inner_step_in, inner_step_out);

    let mut report = SuperGraphNodeReportProgress::new(
        inner_progress_tier,
        inner_step_out,
        inner_total_steps,
        rng,
    );
    report.label = Some("hunyuan_progress".to_string());
    inner_builder.add_node(report.to_any());

    let inner_inputs: Vec<_> = vec![
        inner_dit_weights.to_any(),
        inner_llama_hidden.to_any(),
        inner_clip_pooled.to_any(),
        inner_guidance.to_any(),
        inner_progress_tier.to_any(),
        inner_total_steps.to_any(),
        inner_latent_in.to_any(),
        inner_step_in.to_any(),
        inner_timestep.to_any(),
        inner_dt.to_any(),
    ];
    let inner_outputs: Vec<_> = vec![inner_latent_out.to_any(), inner_step_out.to_any()];
    let inner_graph = inner_builder.build(rng, &inner_inputs, &inner_outputs);

    let mut scan_node = SuperGraphNodeScan::new(
        inner_graph,
        iteration_count_input,
        vec![
            SuperGraphLinkDouble::new(dit_weights, inner_dit_weights),
            SuperGraphLinkDouble::new(llama_hidden, inner_llama_hidden),
            SuperGraphLinkDouble::new(clip_pooled, inner_clip_pooled),
            SuperGraphLinkDouble::new(guidance_scale_input, inner_guidance),
            SuperGraphLinkDouble::new(progress_tier_link, inner_progress_tier),
            SuperGraphLinkDouble::new(iteration_count_input, inner_total_steps),
        ],
        vec![
            SuperGraphLinkTriple::new(initial_latent_input, inner_latent_in, inner_latent_out),
            SuperGraphLinkTriple::new(progress_tier_link, inner_step_in, inner_step_out),
        ],
        vec![
            (timesteps_input, inner_timestep, 0),
            (dt_input, inner_dt, 0),
        ],
        vec![],
        vec![SuperGraphLinkDouble::new(
            inner_latent_out,
            outer_final_latent,
        )],
        rng,
    );
    scan_node.label = Some("hunyuan_denoise_scan".to_string());
    builder.add_node(scan_node.to_any());

    outer_final_latent
}
