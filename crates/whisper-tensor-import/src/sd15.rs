use crate::onnx_graph::Error;
use crate::onnx_graph::WeightStorageStrategy;
use crate::onnx_graph::operators::{Add, Softmax, Transpose};
use crate::onnx_graph::pytorch::{cast, conv2d, div_scalar, group_norm, layer_norm, linear, silu};
use crate::onnx_graph::tensor::{DType, Dimension, InputTensor, Shape, Tensor};
use crate::onnx_graph::weights::{SafetensorsWeightManager, WeightManager};
use crate::sd_common::{self, downsample, resnet_block, spatial_transformer, upsample};
use memmap2::Mmap;
use prost::Message;
use std::path::Path;
use std::sync::Arc;

// SD 1.5 UNet config
const MODEL_CHANNELS: usize = 320;
const CHANNEL_MULT: [usize; 4] = [1, 2, 4, 4];
const NUM_RES_BLOCKS: usize = 2;
const NUM_HEADS: usize = 8;
const CONTEXT_DIM: usize = 768;
const TRANSFORMER_DEPTH: usize = 1;

// CLIP text encoder config
const CLIP_HIDDEN_DIM: usize = 768;
const CLIP_NUM_HEADS: usize = 12;
const CLIP_NUM_LAYERS: usize = 12;
const CLIP_MAX_POSITION: usize = 77;

/// ONNX model bytes for the three SD 1.5 sub-models.
pub type Sd15Models = (Vec<u8>, Vec<u8>, Vec<u8>);

/// Load a standard SD 1.5 checkpoint (.safetensors) and produce three ONNX models.
/// Returns (text_encoder_onnx, unet_onnx, vae_decoder_onnx).
pub fn load_sd15_checkpoint(
    checkpoint_path: &Path,
    output_method: WeightStorageStrategy,
) -> Result<Sd15Models, anyhow::Error> {
    let file = std::fs::File::open(checkpoint_path)?;
    let mmap = unsafe { Mmap::map(&file) }?;
    let weight_manager = SafetensorsWeightManager::new(vec![Arc::new(mmap)])?;

    let model_dtype = sd_common::detect_model_dtype(&weight_manager);
    println!("Detected model dtype: {:?}", model_dtype);

    let origin_path =
        std::fs::canonicalize(checkpoint_path).unwrap_or_else(|_| checkpoint_path.to_path_buf());

    println!("Building SD 1.5 text encoder...");
    let text_encoder = build_text_encoder(
        &weight_manager,
        model_dtype,
        output_method.clone(),
        &origin_path,
    )?;

    println!("Building SD 1.5 UNet...");
    let unet = build_unet(
        &weight_manager,
        model_dtype,
        output_method.clone(),
        &origin_path,
    )?;

    println!("Building SD 1.5 VAE decoder...");
    let vae_decoder =
        sd_common::build_vae_decoder(&weight_manager, model_dtype, output_method, &origin_path)?;

    Ok((text_encoder, unet, vae_decoder))
}

/// Detect if a safetensors file is an SD 1.5 checkpoint by checking for key prefixes.
pub fn is_sd15_checkpoint(weight_manager: &impl WeightManager) -> bool {
    weight_manager
        .get_tensor("model.diffusion_model.input_blocks.0.0.weight")
        .is_ok()
        && weight_manager
            .get_tensor("cond_stage_model.transformer.text_model.embeddings.token_embedding.weight")
            .is_ok()
        && weight_manager
            .get_tensor("first_stage_model.decoder.conv_in.weight")
            .is_ok()
}

// ============================================================================
// CLIP Text Encoder (SD 1.5 specific — HuggingFace CLIP format)
// ============================================================================

pub fn build_text_encoder(
    weight_manager: &impl WeightManager,
    model_dtype: DType,
    output_method: WeightStorageStrategy,
    origin_path: &Path,
) -> Result<Vec<u8>, anyhow::Error> {
    crate::sd_clip::build_clip_text_model_with_projection(
        weight_manager.prefix("cond_stage_model.transformer.text_model"),
        model_dtype,
        output_method,
        origin_path,
        "",
        crate::sd_clip::ClipTextModelConfig {
            hidden_dim: CLIP_HIDDEN_DIM,
            num_heads: CLIP_NUM_HEADS,
            num_layers: CLIP_NUM_LAYERS,
            max_position: CLIP_MAX_POSITION,
            layer_norm_eps: 1e-5,
            mlp_activation: crate::sd_clip::ClipMlpActivation::QuickGelu,
            hidden_source: crate::sd_clip::ClipHiddenStateSource::FinalLayerNorm,
            output_pooled_projection: false,
        },
    )
}

pub(crate) fn clip_encoder_layer(
    wm: &impl WeightManager,
    input: Arc<dyn Tensor>,
    causal_mask: Arc<dyn Tensor>,
) -> Result<Arc<dyn Tensor>, Error> {
    let normed = layer_norm(&wm.prefix("layer_norm1"), input.clone(), 1e-5)?;
    let attn_out = clip_attention(&wm.prefix("self_attn"), normed, causal_mask)?;
    let x = Add::new(None, input, attn_out)?;

    let normed = layer_norm(&wm.prefix("layer_norm2"), x.clone(), 1e-5)?;
    let mlp_out = clip_mlp(&wm.prefix("mlp"), normed)?;
    Ok(Add::new(None, x, mlp_out)?)
}

pub(crate) fn clip_attention(
    wm: &impl WeightManager,
    input: Arc<dyn Tensor>,
    causal_mask: Arc<dyn Tensor>,
) -> Result<Arc<dyn Tensor>, Error> {
    use crate::onnx_graph::operators::MatMul;

    let head_dim = CLIP_HIDDEN_DIM / CLIP_NUM_HEADS;

    let q = linear(&wm.prefix("q_proj"), input.clone())?;
    let k = linear(&wm.prefix("k_proj"), input.clone())?;
    let v = linear(&wm.prefix("v_proj"), input)?;

    let q = Transpose::new(
        None,
        crate::onnx_graph::pytorch::reshape(
            q,
            vec![0, -1, CLIP_NUM_HEADS as i64, head_dim as i64],
        )?,
        Some(vec![0, 2, 1, 3]),
    );
    let k = Transpose::new(
        None,
        crate::onnx_graph::pytorch::reshape(
            k,
            vec![0, -1, CLIP_NUM_HEADS as i64, head_dim as i64],
        )?,
        Some(vec![0, 2, 1, 3]),
    );
    let v = Transpose::new(
        None,
        crate::onnx_graph::pytorch::reshape(
            v,
            vec![0, -1, CLIP_NUM_HEADS as i64, head_dim as i64],
        )?,
        Some(vec![0, 2, 1, 3]),
    );

    let kt = Transpose::new(None, k, Some(vec![0, 1, 3, 2]));
    let scores = MatMul::new(None, q, kt)?;
    let scores = div_scalar(scores, (head_dim as f32).sqrt())?;

    // Apply causal mask
    let scores = Add::new(None, scores, causal_mask)?;

    let attn_weights = Softmax::new(None, scores, Some(-1));
    let attn_output = MatMul::new(None, attn_weights, v)?;

    let attn_output = Transpose::new(None, attn_output, Some(vec![0, 2, 1, 3]));
    let attn_output =
        crate::onnx_graph::pytorch::reshape(attn_output, vec![0, -1, CLIP_HIDDEN_DIM as i64])?;

    linear(&wm.prefix("out_proj"), attn_output)
}

pub(crate) fn clip_mlp(
    wm: &impl WeightManager,
    input: Arc<dyn Tensor>,
) -> Result<Arc<dyn Tensor>, Error> {
    let x = linear(&wm.prefix("fc1"), input)?;
    let x = crate::onnx_graph::pytorch::quick_gelu(x)?;
    linear(&wm.prefix("fc2"), x)
}

// ============================================================================
// UNet (SD 1.5)
// ============================================================================

pub fn build_unet(
    weight_manager: &impl WeightManager,
    model_dtype: DType,
    output_method: WeightStorageStrategy,
    origin_path: &Path,
) -> Result<Vec<u8>, anyhow::Error> {
    // UNet runs in model_dtype (no CastingWeightManager — weights stay in native dtype).
    // Only the timestep sinusoidal embedding is computed in F32 for precision.
    let wm = weight_manager.prefix("model.diffusion_model");

    let batch_dim = Dimension::new(Some(1), Some("batch".to_string()), None);
    let h_dim = Dimension::new(None, Some("height".to_string()), None);
    let w_dim = Dimension::new(None, Some("width".to_string()), None);

    let sample_input = InputTensor::new(
        "sample".to_string(),
        model_dtype,
        Shape::new(vec![
            batch_dim.clone(),
            Dimension::new(Some(4), None, None),
            h_dim,
            w_dim,
        ]),
    );
    let timestep_input = InputTensor::new(
        "timestep".to_string(),
        model_dtype,
        Shape::new(vec![Dimension::new(Some(1), None, None)]),
    );
    let context_input = InputTensor::new(
        "encoder_hidden_states".to_string(),
        model_dtype,
        Shape::new(vec![
            batch_dim,
            Dimension::new(Some(CLIP_MAX_POSITION), None, None),
            Dimension::new(Some(CONTEXT_DIM), None, None),
        ]),
    );

    let sample: Arc<dyn Tensor> = sample_input.clone();
    let context: Arc<dyn Tensor> = context_input.clone();
    // Timestep sinusoidal computed in F32, then cast to model_dtype before MLP
    let t_emb = sd_common::timestep_embedding(
        &wm,
        cast(timestep_input.clone(), DType::F32),
        MODEL_CHANNELS,
        model_dtype,
    )?;

    let channels = CHANNEL_MULT.map(|m| m * MODEL_CHANNELS);

    // input_blocks.0: initial conv
    let h = conv2d(&wm.prefix("input_blocks.0.0"), sample, 3, 1, 1)?;
    let mut skip_connections: Vec<Arc<dyn Tensor>> = vec![h.clone()];

    let mut h = h;
    let mut input_block_idx = 1;
    let mut current_ch = MODEL_CHANNELS;

    for (level, &ch) in channels.iter().enumerate() {
        for _block in 0..NUM_RES_BLOCKS {
            h = resnet_block(
                &wm.prefix(&format!("input_blocks.{input_block_idx}.0")),
                h,
                t_emb.clone(),
                current_ch,
                ch,
            )?;
            current_ch = ch;

            if level < 3 {
                h = spatial_transformer(
                    &wm.prefix(&format!("input_blocks.{input_block_idx}.1")),
                    h,
                    context.clone(),
                    ch,
                    NUM_HEADS,
                    TRANSFORMER_DEPTH,
                    false, // SD 1.5 uses conv proj_in/proj_out
                )?;
            }

            skip_connections.push(h.clone());
            input_block_idx += 1;
        }

        if level < 3 {
            h = downsample(&wm.prefix(&format!("input_blocks.{input_block_idx}.0")), h)?;
            skip_connections.push(h.clone());
            input_block_idx += 1;
        }
    }

    // Middle block
    h = resnet_block(
        &wm.prefix("middle_block.0"),
        h,
        t_emb.clone(),
        current_ch,
        current_ch,
    )?;
    h = spatial_transformer(
        &wm.prefix("middle_block.1"),
        h,
        context.clone(),
        current_ch,
        NUM_HEADS,
        TRANSFORMER_DEPTH,
        false,
    )?;
    h = resnet_block(
        &wm.prefix("middle_block.2"),
        h,
        t_emb.clone(),
        current_ch,
        current_ch,
    )?;

    // Output blocks
    let mut output_block_idx = 0;
    for level in (0..4).rev() {
        let ch = channels[level];
        for block in 0..(NUM_RES_BLOCKS + 1) {
            let skip = skip_connections.pop().unwrap();
            h = crate::onnx_graph::operators::Concat::new(None, vec![h, skip], 1)?;

            let concat_ch = h.shape()[1].resolve()?;

            h = resnet_block(
                &wm.prefix(&format!("output_blocks.{output_block_idx}.0")),
                h,
                t_emb.clone(),
                concat_ch,
                ch,
            )?;

            if level < 3 {
                h = spatial_transformer(
                    &wm.prefix(&format!("output_blocks.{output_block_idx}.1")),
                    h,
                    context.clone(),
                    ch,
                    NUM_HEADS,
                    TRANSFORMER_DEPTH,
                    false,
                )?;
            }

            if level > 0 && block == NUM_RES_BLOCKS {
                let upsample_idx = if level < 3 { 2 } else { 1 };
                h = upsample(
                    &wm.prefix(&format!("output_blocks.{output_block_idx}.{upsample_idx}")),
                    h,
                )?;
            }

            output_block_idx += 1;
        }
    }

    // Output
    h = group_norm(&wm.prefix("out.0"), h, 1e-5, 32)?;
    h = silu(h)?;
    h = conv2d(&wm.prefix("out.2"), h, 3, 1, 1)?;

    let input_tensors: Vec<Arc<dyn Tensor>> = vec![sample_input, timestep_input, context_input];
    let output_tensors: Vec<(String, Arc<dyn Tensor>)> = vec![("out_sample".to_string(), h)];

    let onnx_model = crate::onnx_graph::build_proto_with_origin_path(
        &input_tensors,
        &output_tensors,
        output_method,
        Some(origin_path),
    )?;
    Ok(onnx_model.encode_to_vec())
}
