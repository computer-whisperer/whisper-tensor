use crate::onnx_graph::Error;
use crate::onnx_graph::WeightStorageStrategy;
use crate::onnx_graph::operators::{Add, Gather, MatMul, Softmax, Transpose};
use crate::onnx_graph::pytorch::{cast, conv2d, gelu, group_norm, layer_norm, linear, silu};
use crate::onnx_graph::tensor::{
    DType, Dimension, InputTensor, InputTensorInitialized, Shape, Tensor, TensorData,
};
use crate::onnx_graph::weights::{SafetensorsWeightManager, WeightManager};
use crate::sd_common::{
    self, CastingWeightManager, build_causal_mask, downsample, resnet_block, reshape_symbolic,
    slice_axis, spatial_transformer, upsample,
};
use memmap2::Mmap;
use prost::Message;
use std::path::Path;
use std::sync::Arc;

// SD 2.x UNet config
const MODEL_CHANNELS: usize = 320;
const CHANNEL_MULT: [usize; 4] = [1, 2, 4, 4];
const NUM_RES_BLOCKS: usize = 2;
const NUM_HEAD_CHANNELS: usize = 64; // SD 2 uses per-channel head dim instead of fixed head count
const CONTEXT_DIM: usize = 1024;
const TRANSFORMER_DEPTH: usize = 1;

// OpenCLIP text encoder config (ViT-H/14)
const OPENCLIP_HIDDEN_DIM: usize = 1024;
const OPENCLIP_NUM_HEADS: usize = 16;
const OPENCLIP_NUM_LAYERS: usize = 24;
const OPENCLIP_MAX_POSITION: usize = 77;

/// ONNX model bytes for the three SD 2.x sub-models.
pub type Sd2Models = (Vec<u8>, Vec<u8>, Vec<u8>);

/// Load a standard SD 2.x checkpoint (.safetensors) and produce three ONNX models.
/// Returns (text_encoder_onnx, unet_onnx, vae_decoder_onnx).
pub fn load_sd2_checkpoint(
    checkpoint_path: &Path,
    output_method: WeightStorageStrategy,
) -> Result<Sd2Models, anyhow::Error> {
    let file = std::fs::File::open(checkpoint_path)?;
    let mmap = unsafe { Mmap::map(&file) }?;
    let weight_manager = SafetensorsWeightManager::new(vec![Arc::new(mmap)])?;

    let model_dtype = sd_common::detect_model_dtype(&weight_manager);
    println!("Detected model dtype: {:?}", model_dtype);

    let origin_path =
        std::fs::canonicalize(checkpoint_path).unwrap_or_else(|_| checkpoint_path.to_path_buf());

    println!("Building SD 2 text encoder (OpenCLIP ViT-H/14)...");
    let text_encoder = build_text_encoder(
        &weight_manager,
        model_dtype,
        output_method.clone(),
        &origin_path,
    )?;

    println!("Building SD 2 UNet...");
    let unet = build_unet(
        &weight_manager,
        model_dtype,
        output_method.clone(),
        &origin_path,
    )?;

    println!("Building SD 2 VAE decoder...");
    let vae_decoder =
        sd_common::build_vae_decoder(&weight_manager, model_dtype, output_method, &origin_path)?;

    Ok((text_encoder, unet, vae_decoder))
}

/// Detect if a safetensors file is an SD 2.x checkpoint.
/// Distinguished from SD 1.5 by the OpenCLIP text encoder weight prefix.
pub fn is_sd2_checkpoint(weight_manager: &impl WeightManager) -> bool {
    weight_manager
        .get_tensor("model.diffusion_model.input_blocks.0.0.weight")
        .is_ok()
        && weight_manager
            .get_tensor("cond_stage_model.model.transformer.resblocks.0.attn.in_proj_weight")
            .is_ok()
        && weight_manager
            .get_tensor("first_stage_model.decoder.conv_in.weight")
            .is_ok()
}

// ============================================================================
// OpenCLIP Text Encoder (ViT-H/14 — SD 2.x specific)
// ============================================================================

/// Build the OpenCLIP ViT-H/14 text encoder used by SD 2.x.
///
/// Key differences from SD 1.5's CLIP encoder:
/// - Combined QKV projection (in_proj_weight/bias) instead of separate q/k/v_proj
/// - Standard GELU instead of quick_gelu
/// - OpenCLIP weight naming (resblocks, ln_1/ln_2, c_fc/c_proj)
/// - Outputs penultimate hidden state (before the last transformer layer)
pub fn build_text_encoder(
    weight_manager: &impl WeightManager,
    model_dtype: DType,
    output_method: WeightStorageStrategy,
    origin_path: &Path,
) -> Result<Vec<u8>, anyhow::Error> {
    // Text encoder always computes in f32 for precision.
    let wm = CastingWeightManager::new(
        weight_manager.prefix("cond_stage_model.model"),
        DType::F32,
    );
    let _ = model_dtype;

    let batch_dim = Dimension::new(Some(1), Some("batch_size".to_string()), None);
    let seq_dim = Dimension::new(
        Some(OPENCLIP_MAX_POSITION),
        Some("seq_len".to_string()),
        None,
    );
    let input_shape = Shape::new(vec![batch_dim.clone(), seq_dim.clone()]);

    let input_ids = InputTensor::new("input_ids".to_string(), DType::I32, input_shape);

    // Token embedding
    let token_emb = Gather::new(
        Some("token_embedding".to_string()),
        wm.get_tensor("token_embedding.weight")?,
        input_ids.clone(),
        0,
    )?;

    // Position embedding (precomputed [77, 1024])
    let pos_emb = wm.get_tensor("positional_embedding")?;
    let x = Add::new(Some("pos_embed".to_string()), token_emb, pos_emb)?;

    // Build causal mask
    let mask_data = build_causal_mask(OPENCLIP_MAX_POSITION);
    let causal_mask = InputTensorInitialized::new(
        "causal_mask".to_string(),
        TensorData::new(
            mask_data.into(),
            Shape::new(vec![
                Dimension::new(Some(1), None, None),
                Dimension::new(Some(1), None, None),
                Dimension::new(Some(OPENCLIP_MAX_POSITION), None, None),
                Dimension::new(Some(OPENCLIP_MAX_POSITION), None, None),
            ]),
        )?,
    );

    // Transformer layers — output penultimate hidden state (skip last layer)
    let mut hidden: Arc<dyn Tensor> = x;
    for i in 0..(OPENCLIP_NUM_LAYERS - 1) {
        let layer_wm = wm.prefix(&format!("transformer.resblocks.{i}"));
        hidden = openclip_encoder_layer(&layer_wm, hidden, causal_mask.clone())?;
    }

    // No final layer norm — SD 2.x uses the penultimate hidden state directly.

    let input_tensors: Vec<Arc<dyn Tensor>> = vec![input_ids];
    let output_tensors: Vec<(String, Arc<dyn Tensor>)> =
        vec![("last_hidden_state".to_string(), hidden)];

    let onnx_model = crate::onnx_graph::build_proto_with_origin_path(
        &input_tensors,
        &output_tensors,
        output_method,
        Some(origin_path),
    )?;
    Ok(onnx_model.encode_to_vec())
}

fn openclip_encoder_layer(
    wm: &impl WeightManager,
    input: Arc<dyn Tensor>,
    causal_mask: Arc<dyn Tensor>,
) -> Result<Arc<dyn Tensor>, Error> {
    // Pre-norm self-attention
    let normed = layer_norm(&wm.prefix("ln_1"), input.clone(), 1e-5)?;
    let attn_out = openclip_attention(&wm.prefix("attn"), normed, causal_mask)?;
    let x = Add::new(None, input, attn_out)?;

    // Pre-norm MLP
    let normed = layer_norm(&wm.prefix("ln_2"), x.clone(), 1e-5)?;
    let mlp_out = openclip_mlp(&wm.prefix("mlp"), normed)?;
    Ok(Add::new(None, x, mlp_out)?)
}

/// OpenCLIP attention uses a combined in_proj_weight [3*hidden, hidden] for Q/K/V.
fn openclip_attention(
    wm: &impl WeightManager,
    input: Arc<dyn Tensor>,
    causal_mask: Arc<dyn Tensor>,
) -> Result<Arc<dyn Tensor>, Error> {
    let head_dim = OPENCLIP_HIDDEN_DIM / OPENCLIP_NUM_HEADS;

    // Combined QKV projection: in_proj_weight [3*1024, 1024], in_proj_bias [3*1024]
    let in_proj_weight = wm.get_tensor("in_proj_weight")?;
    let in_proj_bias = wm.get_tensor("in_proj_bias")?;

    // Slice weight and bias into Q, K, V parts
    let q_weight = slice_axis(in_proj_weight.clone(), 0, 0, OPENCLIP_HIDDEN_DIM as i64)?;
    let k_weight = slice_axis(
        in_proj_weight.clone(),
        0,
        OPENCLIP_HIDDEN_DIM as i64,
        (OPENCLIP_HIDDEN_DIM * 2) as i64,
    )?;
    let v_weight = slice_axis(
        in_proj_weight,
        0,
        (OPENCLIP_HIDDEN_DIM * 2) as i64,
        (OPENCLIP_HIDDEN_DIM * 3) as i64,
    )?;

    let q_bias = slice_axis(in_proj_bias.clone(), 0, 0, OPENCLIP_HIDDEN_DIM as i64)?;
    let k_bias = slice_axis(
        in_proj_bias.clone(),
        0,
        OPENCLIP_HIDDEN_DIM as i64,
        (OPENCLIP_HIDDEN_DIM * 2) as i64,
    )?;
    let v_bias = slice_axis(
        in_proj_bias,
        0,
        (OPENCLIP_HIDDEN_DIM * 2) as i64,
        (OPENCLIP_HIDDEN_DIM * 3) as i64,
    )?;

    // Compute Q, K, V: input @ weight^T + bias
    let q = matmul_with_transposed_weight(input.clone(), q_weight, Some(q_bias))?;
    let k = matmul_with_transposed_weight(input.clone(), k_weight, Some(k_bias))?;
    let v = matmul_with_transposed_weight(input, v_weight, Some(v_bias))?;

    // Reshape to multi-head: [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
    let q = Transpose::new(
        None,
        reshape_symbolic(q, vec![0, -1, OPENCLIP_NUM_HEADS as i64, head_dim as i64])?,
        Some(vec![0, 2, 1, 3]),
    );
    let k = Transpose::new(
        None,
        reshape_symbolic(k, vec![0, -1, OPENCLIP_NUM_HEADS as i64, head_dim as i64])?,
        Some(vec![0, 2, 1, 3]),
    );
    let v = Transpose::new(
        None,
        reshape_symbolic(v, vec![0, -1, OPENCLIP_NUM_HEADS as i64, head_dim as i64])?,
        Some(vec![0, 2, 1, 3]),
    );

    // Attention: softmax(Q @ K^T / sqrt(head_dim) + mask) @ V
    let kt = Transpose::new(None, k, Some(vec![0, 1, 3, 2]));
    let scores = MatMul::new(None, q, kt)?;
    let scores = crate::onnx_graph::pytorch::div_scalar(scores, (head_dim as f32).sqrt())?;
    let scores = Add::new(None, scores, causal_mask)?;

    let attn_weights = Softmax::new(None, scores, Some(-1));
    let attn_output = MatMul::new(None, attn_weights, v)?;

    // Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, hidden]
    let attn_output = Transpose::new(None, attn_output, Some(vec![0, 2, 1, 3]));
    let attn_output =
        reshape_symbolic(attn_output, vec![0, -1, OPENCLIP_HIDDEN_DIM as i64])?;

    // Output projection
    linear(&wm.prefix("out_proj"), attn_output)
}

/// Matrix multiply input by transposed weight, with optional bias.
/// Computes: input @ weight^T + bias
fn matmul_with_transposed_weight(
    input: Arc<dyn Tensor>,
    weight: Arc<dyn Tensor>,
    bias: Option<Arc<dyn Tensor>>,
) -> Result<Arc<dyn Tensor>, Error> {
    let weight_t = Transpose::new(None, weight, Some(vec![1, 0]));
    let result = MatMul::new(None, input, weight_t)?;
    if let Some(bias) = bias {
        Ok(Add::new(None, result, bias)?)
    } else {
        Ok(result)
    }
}

fn openclip_mlp(
    wm: &impl WeightManager,
    input: Arc<dyn Tensor>,
) -> Result<Arc<dyn Tensor>, Error> {
    let x = linear(&wm.prefix("c_fc"), input)?;
    let x = gelu(x)?;
    linear(&wm.prefix("c_proj"), x)
}

// ============================================================================
// UNet (SD 2.x)
// ============================================================================

/// Build the SD 2.x UNet.
///
/// Key differences from SD 1.5:
/// - Context dim is 1024 (OpenCLIP ViT-H/14) instead of 768
/// - Number of attention heads is computed from NUM_HEAD_CHANNELS (64) per channel level
/// - Spatial transformer uses linear proj_in/proj_out instead of conv2d(1x1)
pub fn build_unet(
    weight_manager: &impl WeightManager,
    model_dtype: DType,
    output_method: WeightStorageStrategy,
    origin_path: &Path,
) -> Result<Vec<u8>, anyhow::Error> {
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
            Dimension::new(Some(OPENCLIP_MAX_POSITION), None, None),
            Dimension::new(Some(CONTEXT_DIM), None, None),
        ]),
    );

    let sample: Arc<dyn Tensor> = sample_input.clone();
    let context: Arc<dyn Tensor> = context_input.clone();
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

            // Attention at levels 0, 1, 2 (not 3)
            if level < 3 {
                let num_heads = ch / NUM_HEAD_CHANNELS;
                h = spatial_transformer(
                    &wm.prefix(&format!("input_blocks.{input_block_idx}.1")),
                    h,
                    context.clone(),
                    ch,
                    num_heads,
                    TRANSFORMER_DEPTH,
                    true, // SD 2 uses linear proj_in/proj_out
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
    let mid_heads = current_ch / NUM_HEAD_CHANNELS;
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
        mid_heads,
        TRANSFORMER_DEPTH,
        true,
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
                let num_heads = ch / NUM_HEAD_CHANNELS;
                h = spatial_transformer(
                    &wm.prefix(&format!("output_blocks.{output_block_idx}.1")),
                    h,
                    context.clone(),
                    ch,
                    num_heads,
                    TRANSFORMER_DEPTH,
                    true,
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
