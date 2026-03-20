use std::sync::Arc;
use whisper_tensor::loader::*;
use whisper_tensor::metadata::TokenizerInfo;
use whisper_tensor::model::Model;

use super::shared::interface_helpers::{
    build_cast_node, build_denoising_loop, build_eos_indices_node, build_vae_decode,
};

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
            let import_dtype = crate::models::diffusion::sd_common::detect_model_dtype(&wm);
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
            crate::models::diffusion::sd_xl::load_sdxl_checkpoint(&path, storage)
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

        let interface = {
            let mut rng = rand::rng();
            // Both SDXL text encoders use the same CLIP tokenizer
            build_sdxl_interface(
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

/// Build interface for SDXL (dual text encoders + ADM conditioning + CFG).
///
/// Model weights order: [text_encoder_1, text_encoder_2, unet, vae_decoder]
fn build_sdxl_interface(
    rng: &mut impl rand::Rng,
    tokenizer: TokenizerInfo,
    model_dtype: whisper_tensor::dtype::DType,
) -> whisper_tensor::interfaces::ImageGenerationInterface {
    use whisper_tensor::backends::ndarray_backend::NDArrayNumericTensor;
    use whisper_tensor::interfaces::{ImageGenerationInterface, SchedulerType};
    use whisper_tensor::milli_graph::MilliOpGraph;
    use whisper_tensor::milli_graph::ops::{Cast, Concat as MilliConcat, Constant, Pad, PadMode};
    use whisper_tensor::super_graph::nodes::{
        SuperGraphNode, SuperGraphNodeMilliOpGraph, SuperGraphNodeModelExecution,
        SuperGraphNodeTensorToImage, SuperGraphNodeTokenizerEncode,
        SuperGraphNodeTokenizerEncodeMode, SuperGraphNodeTokenizerLoad,
    };
    use whisper_tensor::super_graph::SuperGraphBuilder;

    let mut builder = SuperGraphBuilder::new();

    // Create input links
    let positive_prompt_input = builder.new_string_link(rng);
    let negative_prompt_input = builder.new_string_link(rng);
    let initial_latent_input = builder.new_tensor_link(rng);
    let timesteps_input = builder.new_tensor_link(rng);
    let dt_input = builder.new_tensor_link(rng);
    let sigmas_input = builder.new_tensor_link(rng);
    let iteration_count_input = builder.new_tensor_link(rng);
    let guidance_scale_input = builder.new_tensor_link(rng);
    let te1_weights = builder.new_model_link(rng);
    let te2_weights = builder.new_model_link(rng);
    let unet_weights = builder.new_model_link(rng);
    let vae_weights = builder.new_model_link(rng);
    builder.set_link_label(positive_prompt_input, "prompt_positive");
    builder.set_link_label(negative_prompt_input, "prompt_negative");
    builder.set_link_label(initial_latent_input, "latent_initial");
    builder.set_link_label(timesteps_input, "timesteps");
    builder.set_link_label(dt_input, "dt");
    builder.set_link_label(sigmas_input, "sigmas");
    builder.set_link_label(iteration_count_input, "iteration_count");
    builder.set_link_label(guidance_scale_input, "guidance_scale");
    builder.set_link_label(te1_weights, "text_encoder_1_weights");
    builder.set_link_label(te2_weights, "text_encoder_2_weights");
    builder.set_link_label(unet_weights, "unet_weights");
    builder.set_link_label(vae_weights, "vae_decoder_weights");

    // Prompt tokenization inside the supergraph.
    let tokenizer_link =
        SuperGraphNodeTokenizerLoad::new_and_add(&mut builder, tokenizer, rng);
    let cond_ids_input = SuperGraphNodeTokenizerEncode::new_with_mode_and_add(
        &mut builder,
        tokenizer_link,
        positive_prompt_input,
        SuperGraphNodeTokenizerEncodeMode::ClipStyle {
            seq_len: 77,
            bos: 49406,
            eos: 49407,
            pad: 0,
        },
        rng,
    );
    let negative_cond_ids_input = SuperGraphNodeTokenizerEncode::new_with_mode_and_add(
        &mut builder,
        tokenizer_link,
        negative_prompt_input,
        SuperGraphNodeTokenizerEncodeMode::ClipStyle {
            seq_len: 77,
            bos: 49406,
            eos: 49407,
            pad: 0,
        },
        rng,
    );

    // --- Conditional path ---

    // TE1 conditional: cond_ids -> hidden1 [1, 77, 768] (F32)
    let cond_hidden1_f32 = builder.new_tensor_link(rng);
    let mut te1_cond = SuperGraphNodeModelExecution::new(
        rng,
        te1_weights,
        0,
        vec![(cond_ids_input, "input_ids".to_string())],
        vec![("last_hidden_state".to_string(), cond_hidden1_f32)],
    );
    te1_cond.label = Some("text_encoder_1_conditional".to_string());
    builder.add_node(te1_cond.to_any());

    // Compute eos_indices for conditional
    let cond_eos = build_eos_indices_node(&mut builder, rng, cond_ids_input);

    // TE2 conditional: cond_ids + eos -> penultimate [1, 77, 1280] + pooled [1, 1280] (F32)
    let cond_penult2_f32 = builder.new_tensor_link(rng);
    let cond_pooled_f32 = builder.new_tensor_link(rng);
    let mut te2_cond = SuperGraphNodeModelExecution::new(
        rng,
        te2_weights,
        1,
        vec![
            (cond_ids_input, "input_ids".to_string()),
            (cond_eos, "eos_indices".to_string()),
        ],
        vec![
            ("penultimate_hidden_state".to_string(), cond_penult2_f32),
            ("pooled_output".to_string(), cond_pooled_f32),
        ],
    );
    te2_cond.label = Some("text_encoder_2_conditional".to_string());
    builder.add_node(te2_cond.to_any());

    // Concat hidden1 + penult2 -> context [1, 77, 2048], pad pooled -> y [1, 2816]
    // Then cast both to model_dtype
    let cond_context = builder.new_tensor_link(rng);
    let cond_y = builder.new_tensor_link(rng);
    {
        let (mut mg, input_map) = MilliOpGraph::new(
            [
                cond_hidden1_f32.global_id(),
                cond_penult2_f32.global_id(),
                cond_pooled_f32.global_id(),
            ],
            rng,
        );
        let h1 = *input_map.get(&cond_hidden1_f32.global_id()).unwrap();
        let p2 = *input_map.get(&cond_penult2_f32.global_id()).unwrap();
        let pooled = *input_map.get(&cond_pooled_f32.global_id()).unwrap();

        // Concat along last dim: [1,77,768] + [1,77,1280] -> [1,77,2048]
        let ctx = MilliConcat::push_new(&mut mg, vec![h1, p2], -1, rng);
        let ctx_cast = Cast::push_new(&mut mg, ctx, model_dtype, rng);

        // Pad pooled [1,1280] -> [1,2816] (add 1536 zeros on right of dim 1)
        let pads = Constant::push_new(
            &mut mg,
            NDArrayNumericTensor::from_vec_shape(vec![0i64, 0, 0, 1536], &vec![4]).unwrap(),
            rng,
        );
        let padded = Pad::push_new(&mut mg, pooled, pads, None, None, PadMode::Constant, rng);
        let y_cast = Cast::push_new(&mut mg, padded, model_dtype, rng);

        mg.set_output_map([
            (ctx_cast, cond_context.global_id()),
            (y_cast, cond_y.global_id()),
        ]);
        let mut node = SuperGraphNodeMilliOpGraph::new(mg, rng);
        node.label = Some("conditioning_conditional_assemble".to_string());
        builder.add_node(node.to_any());
    }

    // --- Unconditional path ---

    // TE1 unconditional
    let uncond_hidden1_f32 = builder.new_tensor_link(rng);
    let mut te1_uncond = SuperGraphNodeModelExecution::new(
        rng,
        te1_weights,
        0,
        vec![(negative_cond_ids_input, "input_ids".to_string())],
        vec![("last_hidden_state".to_string(), uncond_hidden1_f32)],
    );
    te1_uncond.label = Some("text_encoder_1_unconditional".to_string());
    builder.add_node(te1_uncond.to_any());

    // Compute eos_indices for unconditional
    let uncond_eos = build_eos_indices_node(&mut builder, rng, negative_cond_ids_input);

    // TE2 unconditional
    let uncond_penult2_f32 = builder.new_tensor_link(rng);
    let uncond_pooled_f32 = builder.new_tensor_link(rng);
    let mut te2_uncond = SuperGraphNodeModelExecution::new(
        rng,
        te2_weights,
        1,
        vec![
            (negative_cond_ids_input, "input_ids".to_string()),
            (uncond_eos, "eos_indices".to_string()),
        ],
        vec![
            ("penultimate_hidden_state".to_string(), uncond_penult2_f32),
            ("pooled_output".to_string(), uncond_pooled_f32),
        ],
    );
    te2_uncond.label = Some("text_encoder_2_unconditional".to_string());
    builder.add_node(te2_uncond.to_any());

    // Concat + pad for unconditional
    let uncond_context = builder.new_tensor_link(rng);
    let uncond_y = builder.new_tensor_link(rng);
    {
        let (mut mg, input_map) = MilliOpGraph::new(
            [
                uncond_hidden1_f32.global_id(),
                uncond_penult2_f32.global_id(),
                uncond_pooled_f32.global_id(),
            ],
            rng,
        );
        let h1 = *input_map.get(&uncond_hidden1_f32.global_id()).unwrap();
        let p2 = *input_map.get(&uncond_penult2_f32.global_id()).unwrap();
        let pooled = *input_map.get(&uncond_pooled_f32.global_id()).unwrap();

        let ctx = MilliConcat::push_new(&mut mg, vec![h1, p2], -1, rng);
        let ctx_cast = Cast::push_new(&mut mg, ctx, model_dtype, rng);

        let pads = Constant::push_new(
            &mut mg,
            NDArrayNumericTensor::from_vec_shape(vec![0i64, 0, 0, 1536], &vec![4]).unwrap(),
            rng,
        );
        let padded = Pad::push_new(&mut mg, pooled, pads, None, None, PadMode::Constant, rng);
        let y_cast = Cast::push_new(&mut mg, padded, model_dtype, rng);

        mg.set_output_map([
            (ctx_cast, uncond_context.global_id()),
            (y_cast, uncond_y.global_id()),
        ]);
        let mut node = SuperGraphNodeMilliOpGraph::new(mg, rng);
        node.label = Some("conditioning_unconditional_assemble".to_string());
        builder.add_node(node.to_any());
    }

    // --- Denoising loop ---
    let final_latent = build_denoising_loop(
        &mut builder,
        rng,
        unet_weights,
        cond_context,
        uncond_context,
        Some(cond_y), // ADM conditioning
        Some(uncond_y),
        guidance_scale_input,
        initial_latent_input,
        timesteps_input,
        dt_input,
        sigmas_input,
        iteration_count_input,
        model_dtype,
        2, // unet model index (te1=0, te2=1, unet=2)
    );

    // VAE decode (SDXL uses 0.13025 scale factor) + wrap tensor into Image
    let decoded_image_tensor = build_vae_decode(
        &mut builder,
        rng,
        final_latent,
        vae_weights,
        3,
        0.13025,
        model_dtype,
    );
    let image_output =
        SuperGraphNodeTensorToImage::new_and_add(&mut builder, decoded_image_tensor, rng);
    builder.set_link_label(image_output, "image_output");

    // Build outer graph
    let model_weights = vec![te1_weights, te2_weights, unet_weights, vae_weights];
    let input_links: Vec<_> = vec![
        positive_prompt_input.to_any(),
        negative_prompt_input.to_any(),
        initial_latent_input.to_any(),
        timesteps_input.to_any(),
        dt_input.to_any(),
        sigmas_input.to_any(),
        iteration_count_input.to_any(),
        guidance_scale_input.to_any(),
        te1_weights.to_any(),
        te2_weights.to_any(),
        unet_weights.to_any(),
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
        scheduler: SchedulerType::EulerDiscrete,
        latent_channels: 4,
    }
}
