mod diffusers;
mod io_infer;
mod path;

use self::diffusers::load_diffusers_sd3_pipeline;
use self::io_infer::{
    infer_clip_io, infer_latent_channels, infer_model_dtype, infer_t5_io, infer_t5_seq_len,
    infer_transformer_io, infer_vae_io,
};
use self::path::{
    is_sd3_diffusers_safetensors_dir, load_onnx_model, resolve_component_onnx,
    resolve_component_onnx_any,
};
use whisper_tensor::dtype::DType;
use whisper_tensor::loader::*;
use whisper_tensor::metadata::TokenizerInfo;

/// Loader for Stable Diffusion 3.5 ONNX pipelines.
///
/// Expected directory layout:
/// - text_encoder/model.onnx
/// - text_encoder_2/model.onnx
/// - text_encoder_3/model.onnx
/// - transformer/model.onnx
/// - vae_decoder/model.onnx
pub struct SD35Loader;

impl Loader for SD35Loader {
    fn name(&self) -> &str {
        "Stable Diffusion 3.5"
    }

    fn description(&self) -> &str {
        "Load SD3.5 from official diffusers safetensors or ONNX pipeline directories"
    }

    fn config_schema(&self) -> Vec<ConfigField> {
        vec![
            ConfigField {
                key: "path".to_string(),
                label: "Pipeline Path".to_string(),
                description:
                    "Path to an SD3.5 directory (official diffusers safetensors or ONNX pipeline)"
                        .to_string(),
                field_type: ConfigFieldType::FilePath,
                required: true,
                default: None,
            },
            ConfigField {
                key: "clip_tokenizer".to_string(),
                label: "CLIP Tokenizer".to_string(),
                description:
                    "HF tokenizer for CLIP encoders (default: openai/clip-vit-large-patch14)"
                        .to_string(),
                field_type: ConfigFieldType::String,
                required: false,
                default: Some(ConfigValue::String(
                    "openai/clip-vit-large-patch14".to_string(),
                )),
            },
            ConfigField {
                key: "t5_tokenizer".to_string(),
                label: "T5 Tokenizer".to_string(),
                description: "HF tokenizer for T5 encoder (default: google-t5/t5-base)".to_string(),
                field_type: ConfigFieldType::String,
                required: false,
                default: Some(ConfigValue::String("google-t5/t5-base".to_string())),
            },
            ConfigField {
                key: "t5_seq_len".to_string(),
                label: "T5 Sequence Length".to_string(),
                description: "Prompt token length for T5 encoder (default: inferred or 256)"
                    .to_string(),
                field_type: ConfigFieldType::Integer {
                    min: Some(1),
                    max: Some(1024),
                },
                required: false,
                default: Some(ConfigValue::Integer(256)),
            },
            ConfigField {
                key: "vae_scale_factor".to_string(),
                label: "VAE Scale Factor".to_string(),
                description: "Latent scale factor before VAE decode (default: 1.5305 for SD3.x)"
                    .to_string(),
                field_type: ConfigFieldType::Float {
                    min: Some(0.0001),
                    max: None,
                },
                required: false,
                default: Some(ConfigValue::Float(1.5305)),
            },
            ConfigField {
                key: "vae_shift_factor".to_string(),
                label: "VAE Shift Factor".to_string(),
                description: "Latent shift factor before VAE decode (default: 0.0609 for SD3.x)"
                    .to_string(),
                field_type: ConfigFieldType::Float {
                    min: None,
                    max: None,
                },
                required: false,
                default: Some(ConfigValue::Float(0.0609)),
            },
        ]
    }

    fn load(&self, config: ConfigValues) -> Result<LoaderOutput, LoaderError> {
        let path = require_path(&config, "path")?;
        if !path.is_dir() {
            return Err(LoaderError::LoadFailed(anyhow::anyhow!(
                "SD3.5 loader expects a directory, got: {}",
                path.display()
            )));
        }

        let clip_tokenizer = get_string(&config, "clip_tokenizer")?
            .unwrap_or_else(|| "openai/clip-vit-large-patch14".to_string());
        let t5_tokenizer =
            get_string(&config, "t5_tokenizer")?.unwrap_or_else(|| "google-t5/t5-base".to_string());
        let t5_seq_len_override = match config.get("t5_seq_len") {
            Some(ConfigValue::Integer(v)) => Some(*v as usize),
            Some(_) => {
                return Err(LoaderError::InvalidValue {
                    field: "t5_seq_len".to_string(),
                    reason: "expected integer".to_string(),
                });
            }
            None => None,
        };
        let vae_scale_factor_override = match config.get("vae_scale_factor") {
            Some(ConfigValue::Float(v)) => Some(*v as f32),
            Some(ConfigValue::Integer(v)) => Some(*v as f32),
            Some(_) => {
                return Err(LoaderError::InvalidValue {
                    field: "vae_scale_factor".to_string(),
                    reason: "expected float".to_string(),
                });
            }
            None => None,
        };
        let vae_shift_factor_override = match config.get("vae_shift_factor") {
            Some(ConfigValue::Float(v)) => Some(*v as f32),
            Some(ConfigValue::Integer(v)) => Some(*v as f32),
            Some(_) => {
                return Err(LoaderError::InvalidValue {
                    field: "vae_shift_factor".to_string(),
                    reason: "expected float".to_string(),
                });
            }
            None => None,
        };

        if is_sd3_diffusers_safetensors_dir(&path) {
            return load_diffusers_sd3_pipeline(
                &path,
                clip_tokenizer,
                t5_tokenizer,
                t5_seq_len_override,
                vae_scale_factor_override,
                vae_shift_factor_override,
            );
        }

        let vae_scale_factor = vae_scale_factor_override.unwrap_or(1.5305);
        let vae_shift_factor = vae_shift_factor_override.unwrap_or(0.0609);

        let clip_l_path = resolve_component_onnx(&path, "text_encoder")?;
        let clip_g_path = resolve_component_onnx(&path, "text_encoder_2")?;
        let t5_path = resolve_component_onnx(&path, "text_encoder_3")?;
        let transformer_path = resolve_component_onnx(&path, "transformer")?;
        let vae_path = resolve_component_onnx_any(&path, &["vae_decoder", "vae"])?;

        let clip_l_model = load_onnx_model(&clip_l_path)?;
        let clip_g_model = load_onnx_model(&clip_g_path)?;
        let t5_model = load_onnx_model(&t5_path)?;
        let transformer_model = load_onnx_model(&transformer_path)?;
        let vae_model = load_onnx_model(&vae_path)?;

        let clip_l_io = infer_clip_io(clip_l_model.get_symbolic_graph(), "text_encoder")?;
        let clip_g_io = infer_clip_io(clip_g_model.get_symbolic_graph(), "text_encoder_2")?;
        let t5_io = infer_t5_io(t5_model.get_symbolic_graph(), "text_encoder_3")?;
        let transformer_io =
            infer_transformer_io(transformer_model.get_symbolic_graph(), "transformer")?;
        let vae_io = infer_vae_io(vae_model.get_symbolic_graph(), "vae_decoder")?;

        let model_dtype =
            infer_model_dtype(transformer_model.get_symbolic_graph(), &transformer_io)
                .unwrap_or(DType::F16);
        let latent_channels =
            infer_latent_channels(transformer_model.get_symbolic_graph(), &transformer_io)
                .unwrap_or(16);
        let inferred_t5_seq_len = infer_t5_seq_len(t5_model.get_symbolic_graph(), &t5_io);
        let t5_seq_len = t5_seq_len_override.or(inferred_t5_seq_len).unwrap_or(256);

        let base_name = path
            .file_name()
            .and_then(|x| x.to_str())
            .unwrap_or("sd35")
            .to_string();

        let models = vec![
            LoadedModel {
                name: format!("{base_name}-clip_l"),
                model: clip_l_model.clone(),
            },
            LoadedModel {
                name: format!("{base_name}-clip_g"),
                model: clip_g_model.clone(),
            },
            LoadedModel {
                name: format!("{base_name}-t5"),
                model: t5_model.clone(),
            },
            LoadedModel {
                name: format!("{base_name}-transformer"),
                model: transformer_model.clone(),
            },
            LoadedModel {
                name: format!("{base_name}-vae_decoder"),
                model: vae_model.clone(),
            },
        ];

        let interface = {
            let mut rng = rand::rng();
            build_sd3_interface(
                &mut rng,
                TokenizerInfo::HFTokenizer(clip_tokenizer),
                TokenizerInfo::HFTokenizer(t5_tokenizer),
                model_dtype,
                t5_seq_len,
                &clip_l_io.input,
                clip_l_io.eos_input.as_deref(),
                &clip_l_io.hidden_output,
                &clip_l_io.pooled_output,
                &clip_g_io.input,
                clip_g_io.eos_input.as_deref(),
                &clip_g_io.hidden_output,
                &clip_g_io.pooled_output,
                &t5_io.input,
                &t5_io.hidden_output,
                &transformer_io.latent_input,
                &transformer_io.timestep_input,
                &transformer_io.context_input,
                &transformer_io.pooled_input,
                &transformer_io.output,
                &vae_io.input,
                &vae_io.output,
                vae_scale_factor,
                vae_shift_factor,
                latent_channels,
            )
        };

        let interfaces = vec![LoadedInterface {
            name: format!("{base_name}-ImageGeneration"),
            interface: interface.to_any(),
        }];

        Ok(LoaderOutput { models, interfaces })
    }
}

/// Build interface for SD3/SD3.5 ONNX pipelines.
///
/// Model weights order: [clip_l, clip_g, t5_xxl, transformer, vae_decoder]
///
/// This function accepts IO tensor names to support different ONNX export variants.
#[allow(clippy::too_many_arguments)]
pub(super) fn build_sd3_interface(
    rng: &mut impl rand::Rng,
    clip_tokenizer: TokenizerInfo,
    t5_tokenizer: TokenizerInfo,
    model_dtype: DType,
    t5_sequence_length: usize,
    clip_l_input_name: &str,
    clip_l_eos_input_name: Option<&str>,
    clip_l_hidden_output_name: &str,
    clip_l_pooled_output_name: &str,
    clip_g_input_name: &str,
    clip_g_eos_input_name: Option<&str>,
    clip_g_hidden_output_name: &str,
    clip_g_pooled_output_name: &str,
    t5_input_name: &str,
    t5_hidden_output_name: &str,
    transformer_latent_input_name: &str,
    transformer_timestep_input_name: &str,
    transformer_context_input_name: &str,
    transformer_pooled_input_name: &str,
    transformer_output_name: &str,
    vae_input_name: &str,
    vae_output_name: &str,
    vae_scale_factor: f32,
    vae_shift_factor: f32,
    latent_channels: usize,
) -> whisper_tensor::interfaces::ImageGenerationInterface {
    use super::shared::interface_helpers::{
        build_cast_node, build_eos_indices_node, build_sd3_denoising_loop,
        build_vae_decode_with_shift,
    };
    use whisper_tensor::backends::ndarray_backend::NDArrayNumericTensor;
    use whisper_tensor::interfaces::{ImageGenerationInterface, SchedulerType};
    use whisper_tensor::milli_graph::MilliOpGraph;
    use whisper_tensor::milli_graph::ops::{Cast, Concat as MilliConcat, Constant, Pad, PadMode};
    use whisper_tensor::super_graph::SuperGraphBuilder;
    use whisper_tensor::super_graph::nodes::{
        SuperGraphNode, SuperGraphNodeMilliOpGraph, SuperGraphNodeModelExecution,
        SuperGraphNodeTensorToImage, SuperGraphNodeTokenizerEncode,
        SuperGraphNodeTokenizerEncodeMode, SuperGraphNodeTokenizerLoad,
    };

    let mut builder = SuperGraphBuilder::new();

    // Inputs
    let positive_prompt_input = builder.new_string_link(rng);
    let negative_prompt_input = builder.new_string_link(rng);
    let initial_latent_input = builder.new_tensor_link(rng);
    let timesteps_input = builder.new_tensor_link(rng);
    let dt_input = builder.new_tensor_link(rng);
    let sigmas_input = builder.new_tensor_link(rng);
    let iteration_count_input = builder.new_tensor_link(rng);
    let guidance_scale_input = builder.new_tensor_link(rng);
    let clip_l_weights = builder.new_model_link(rng);
    let clip_g_weights = builder.new_model_link(rng);
    let t5_weights = builder.new_model_link(rng);
    let transformer_weights = builder.new_model_link(rng);
    let vae_weights = builder.new_model_link(rng);
    builder.set_link_label(positive_prompt_input, "prompt_positive");
    builder.set_link_label(negative_prompt_input, "prompt_negative");
    builder.set_link_label(initial_latent_input, "latent_initial");
    builder.set_link_label(timesteps_input, "timesteps");
    builder.set_link_label(dt_input, "dt");
    builder.set_link_label(sigmas_input, "sigmas");
    builder.set_link_label(iteration_count_input, "iteration_count");
    builder.set_link_label(guidance_scale_input, "guidance_scale");
    builder.set_link_label(clip_l_weights, "clip_l_weights");
    builder.set_link_label(clip_g_weights, "clip_g_weights");
    builder.set_link_label(t5_weights, "t5_weights");
    builder.set_link_label(transformer_weights, "transformer_weights");
    builder.set_link_label(vae_weights, "vae_decoder_weights");

    // Tokenizers
    let clip_tokenizer_link =
        SuperGraphNodeTokenizerLoad::new_and_add(&mut builder, clip_tokenizer, rng);
    let t5_tokenizer_link =
        SuperGraphNodeTokenizerLoad::new_and_add(&mut builder, t5_tokenizer, rng);

    let clip_ids_pos = SuperGraphNodeTokenizerEncode::new_with_mode_and_add(
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
    let clip_ids_neg = SuperGraphNodeTokenizerEncode::new_with_mode_and_add(
        &mut builder,
        clip_tokenizer_link,
        negative_prompt_input,
        SuperGraphNodeTokenizerEncodeMode::ClipStyle {
            seq_len: 77,
            bos: 49406,
            eos: 49407,
            pad: 0,
        },
        rng,
    );
    let t5_ids_pos = SuperGraphNodeTokenizerEncode::new_with_mode_and_add(
        &mut builder,
        t5_tokenizer_link,
        positive_prompt_input,
        SuperGraphNodeTokenizerEncodeMode::RawPad {
            seq_len: t5_sequence_length,
            pad: 0,
        },
        rng,
    );
    let t5_ids_neg = SuperGraphNodeTokenizerEncode::new_with_mode_and_add(
        &mut builder,
        t5_tokenizer_link,
        negative_prompt_input,
        SuperGraphNodeTokenizerEncodeMode::RawPad {
            seq_len: t5_sequence_length,
            pad: 0,
        },
        rng,
    );

    let clip_eos_pos = build_eos_indices_node(&mut builder, rng, clip_ids_pos);
    let clip_eos_neg = build_eos_indices_node(&mut builder, rng, clip_ids_neg);

    // Conditional encoders
    let clip_l_hidden_pos = builder.new_tensor_link(rng);
    let clip_l_pooled_pos = builder.new_tensor_link(rng);
    let mut clip_l_cond_inputs = vec![(clip_ids_pos, clip_l_input_name.to_string())];
    if let Some(eos_name) = clip_l_eos_input_name {
        clip_l_cond_inputs.push((clip_eos_pos, eos_name.to_string()));
    }
    let mut clip_l_cond = SuperGraphNodeModelExecution::new(
        rng,
        clip_l_weights,
        0,
        clip_l_cond_inputs,
        vec![
            (clip_l_hidden_output_name.to_string(), clip_l_hidden_pos),
            (clip_l_pooled_output_name.to_string(), clip_l_pooled_pos),
        ],
    );
    clip_l_cond.label = Some("clip_l_conditional".to_string());
    builder.add_node(clip_l_cond.to_any());

    let clip_g_hidden_pos = builder.new_tensor_link(rng);
    let clip_g_pooled_pos = builder.new_tensor_link(rng);
    let mut clip_g_cond_inputs = vec![(clip_ids_pos, clip_g_input_name.to_string())];
    if let Some(eos_name) = clip_g_eos_input_name {
        clip_g_cond_inputs.push((clip_eos_pos, eos_name.to_string()));
    }
    let mut clip_g_cond = SuperGraphNodeModelExecution::new(
        rng,
        clip_g_weights,
        1,
        clip_g_cond_inputs,
        vec![
            (clip_g_hidden_output_name.to_string(), clip_g_hidden_pos),
            (clip_g_pooled_output_name.to_string(), clip_g_pooled_pos),
        ],
    );
    clip_g_cond.label = Some("clip_g_conditional".to_string());
    builder.add_node(clip_g_cond.to_any());

    let t5_hidden_pos = builder.new_tensor_link(rng);
    let mut t5_cond = SuperGraphNodeModelExecution::new(
        rng,
        t5_weights,
        2,
        vec![(t5_ids_pos, t5_input_name.to_string())],
        vec![(t5_hidden_output_name.to_string(), t5_hidden_pos)],
    );
    t5_cond.label = Some("t5_conditional".to_string());
    builder.add_node(t5_cond.to_any());

    // Unconditional encoders
    let clip_l_hidden_neg = builder.new_tensor_link(rng);
    let clip_l_pooled_neg = builder.new_tensor_link(rng);
    let mut clip_l_uncond_inputs = vec![(clip_ids_neg, clip_l_input_name.to_string())];
    if let Some(eos_name) = clip_l_eos_input_name {
        clip_l_uncond_inputs.push((clip_eos_neg, eos_name.to_string()));
    }
    let mut clip_l_uncond = SuperGraphNodeModelExecution::new(
        rng,
        clip_l_weights,
        0,
        clip_l_uncond_inputs,
        vec![
            (clip_l_hidden_output_name.to_string(), clip_l_hidden_neg),
            (clip_l_pooled_output_name.to_string(), clip_l_pooled_neg),
        ],
    );
    clip_l_uncond.label = Some("clip_l_unconditional".to_string());
    builder.add_node(clip_l_uncond.to_any());

    let clip_g_hidden_neg = builder.new_tensor_link(rng);
    let clip_g_pooled_neg = builder.new_tensor_link(rng);
    let mut clip_g_uncond_inputs = vec![(clip_ids_neg, clip_g_input_name.to_string())];
    if let Some(eos_name) = clip_g_eos_input_name {
        clip_g_uncond_inputs.push((clip_eos_neg, eos_name.to_string()));
    }
    let mut clip_g_uncond = SuperGraphNodeModelExecution::new(
        rng,
        clip_g_weights,
        1,
        clip_g_uncond_inputs,
        vec![
            (clip_g_hidden_output_name.to_string(), clip_g_hidden_neg),
            (clip_g_pooled_output_name.to_string(), clip_g_pooled_neg),
        ],
    );
    clip_g_uncond.label = Some("clip_g_unconditional".to_string());
    builder.add_node(clip_g_uncond.to_any());

    let t5_hidden_neg = builder.new_tensor_link(rng);
    let mut t5_uncond = SuperGraphNodeModelExecution::new(
        rng,
        t5_weights,
        2,
        vec![(t5_ids_neg, t5_input_name.to_string())],
        vec![(t5_hidden_output_name.to_string(), t5_hidden_neg)],
    );
    t5_uncond.label = Some("t5_unconditional".to_string());
    builder.add_node(t5_uncond.to_any());

    // Build conditional context + pooled projections.
    let cond_context = builder.new_tensor_link(rng);
    let cond_pooled = builder.new_tensor_link(rng);
    {
        let (mut mg, input_map) = MilliOpGraph::new(
            [
                clip_l_hidden_pos.global_id(),
                clip_g_hidden_pos.global_id(),
                t5_hidden_pos.global_id(),
                clip_l_pooled_pos.global_id(),
                clip_g_pooled_pos.global_id(),
            ],
            rng,
        );
        let l_hidden = *input_map.get(&clip_l_hidden_pos.global_id()).unwrap();
        let g_hidden = *input_map.get(&clip_g_hidden_pos.global_id()).unwrap();
        let t5_hidden = *input_map.get(&t5_hidden_pos.global_id()).unwrap();
        let l_pooled = *input_map.get(&clip_l_pooled_pos.global_id()).unwrap();
        let g_pooled = *input_map.get(&clip_g_pooled_pos.global_id()).unwrap();

        // [1,77,768] + [1,77,1280] -> [1,77,2048]
        let clip_hidden = MilliConcat::push_new(&mut mg, vec![l_hidden, g_hidden], -1, rng);
        // Pad feature dim to 4096, then append T5 along sequence dim.
        let pads = Constant::push_new(
            &mut mg,
            NDArrayNumericTensor::from_vec_shape(vec![0i64, 0, 0, 0, 0, 2048], &vec![6]).unwrap(),
            rng,
        );
        let clip_hidden_padded = Pad::push_new(
            &mut mg,
            clip_hidden,
            pads,
            None,
            None,
            PadMode::Constant,
            rng,
        );
        let combined_context =
            MilliConcat::push_new(&mut mg, vec![clip_hidden_padded, t5_hidden], -2, rng);
        let combined_context = Cast::push_new(&mut mg, combined_context, model_dtype, rng);

        // [1,768] + [1,1280] -> [1,2048]
        let pooled = MilliConcat::push_new(&mut mg, vec![l_pooled, g_pooled], -1, rng);
        let pooled = Cast::push_new(&mut mg, pooled, model_dtype, rng);

        mg.set_output_map([
            (combined_context, cond_context.global_id()),
            (pooled, cond_pooled.global_id()),
        ]);
        let mut node = SuperGraphNodeMilliOpGraph::new(mg, rng);
        node.label = Some("sd3_conditioning_conditional_assemble".to_string());
        builder.add_node(node.to_any());
    }

    // Build unconditional context + pooled projections.
    let uncond_context = builder.new_tensor_link(rng);
    let uncond_pooled = builder.new_tensor_link(rng);
    {
        let (mut mg, input_map) = MilliOpGraph::new(
            [
                clip_l_hidden_neg.global_id(),
                clip_g_hidden_neg.global_id(),
                t5_hidden_neg.global_id(),
                clip_l_pooled_neg.global_id(),
                clip_g_pooled_neg.global_id(),
            ],
            rng,
        );
        let l_hidden = *input_map.get(&clip_l_hidden_neg.global_id()).unwrap();
        let g_hidden = *input_map.get(&clip_g_hidden_neg.global_id()).unwrap();
        let t5_hidden = *input_map.get(&t5_hidden_neg.global_id()).unwrap();
        let l_pooled = *input_map.get(&clip_l_pooled_neg.global_id()).unwrap();
        let g_pooled = *input_map.get(&clip_g_pooled_neg.global_id()).unwrap();

        let clip_hidden = MilliConcat::push_new(&mut mg, vec![l_hidden, g_hidden], -1, rng);
        let pads = Constant::push_new(
            &mut mg,
            NDArrayNumericTensor::from_vec_shape(vec![0i64, 0, 0, 0, 0, 2048], &vec![6]).unwrap(),
            rng,
        );
        let clip_hidden_padded = Pad::push_new(
            &mut mg,
            clip_hidden,
            pads,
            None,
            None,
            PadMode::Constant,
            rng,
        );
        let combined_context =
            MilliConcat::push_new(&mut mg, vec![clip_hidden_padded, t5_hidden], -2, rng);
        let combined_context = Cast::push_new(&mut mg, combined_context, model_dtype, rng);

        let pooled = MilliConcat::push_new(&mut mg, vec![l_pooled, g_pooled], -1, rng);
        let pooled = Cast::push_new(&mut mg, pooled, model_dtype, rng);

        mg.set_output_map([
            (combined_context, uncond_context.global_id()),
            (pooled, uncond_pooled.global_id()),
        ]);
        let mut node = SuperGraphNodeMilliOpGraph::new(mg, rng);
        node.label = Some("sd3_conditioning_unconditional_assemble".to_string());
        builder.add_node(node.to_any());
    }

    let final_latent = build_sd3_denoising_loop(
        &mut builder,
        rng,
        transformer_weights,
        cond_context,
        uncond_context,
        cond_pooled,
        uncond_pooled,
        guidance_scale_input,
        initial_latent_input,
        timesteps_input,
        dt_input,
        sigmas_input,
        iteration_count_input,
        model_dtype,
        3,
        transformer_latent_input_name,
        transformer_timestep_input_name,
        transformer_context_input_name,
        transformer_pooled_input_name,
        transformer_output_name,
    );

    let decoded_image_tensor = build_vae_decode_with_shift(
        &mut builder,
        rng,
        final_latent,
        vae_weights,
        4,
        vae_scale_factor,
        vae_shift_factor,
        model_dtype,
        vae_input_name,
        vae_output_name,
    );
    let image_output =
        SuperGraphNodeTensorToImage::new_and_add(&mut builder, decoded_image_tensor, rng);
    builder.set_link_label(image_output, "image_output");

    let model_weights = vec![
        clip_l_weights,
        clip_g_weights,
        t5_weights,
        transformer_weights,
        vae_weights,
    ];
    let input_links: Vec<_> = vec![
        positive_prompt_input.to_any(),
        negative_prompt_input.to_any(),
        initial_latent_input.to_any(),
        timesteps_input.to_any(),
        dt_input.to_any(),
        sigmas_input.to_any(),
        iteration_count_input.to_any(),
        guidance_scale_input.to_any(),
        clip_l_weights.to_any(),
        clip_g_weights.to_any(),
        t5_weights.to_any(),
        transformer_weights.to_any(),
        vae_weights.to_any(),
    ];
    let output_links: Vec<_> = vec![image_output.to_any()];
    let super_graph = builder.build(rng, &input_links, &output_links);

    ImageGenerationInterface {
        super_graph,
        positive_prompt_input,
        negative_prompt_input: Some(negative_prompt_input),
        initial_latent_input,
        timesteps_input,
        dt_input,
        sigmas_input,
        iteration_count_input,
        guidance_scale_input: Some(guidance_scale_input),
        model_weights,
        image_output,
        scheduler: SchedulerType::RectifiedFlow,
        latent_channels,
    }
}
