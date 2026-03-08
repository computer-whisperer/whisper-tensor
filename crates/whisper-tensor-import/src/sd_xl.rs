use crate::onnx_graph::Error;
use crate::onnx_graph::WeightStorageStrategy;
use crate::onnx_graph::operators::{Add, Concat, Gather, MatMul, Softmax, Transpose};
use crate::onnx_graph::pytorch::{
    cast, conv2d, div_scalar, gelu, group_norm, layer_norm, linear, silu,
};
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

// SDXL UNet config
const MODEL_CHANNELS: usize = 320;
const CHANNEL_MULT: [usize; 3] = [1, 2, 4]; // 3 levels (not 4)
const NUM_RES_BLOCKS: usize = 2;
const NUM_HEAD_CHANNELS: usize = 64;
const CONTEXT_DIM: usize = 2048;
const TRANSFORMER_DEPTH: [usize; 3] = [0, 2, 10]; // Variable depth per level
const ADM_IN_CHANNELS: usize = 2816;

// CLIP ViT-L/14 text encoder config (TE1)
const CLIP_HIDDEN_DIM: usize = 768;
const CLIP_NUM_HEADS: usize = 12;
const CLIP_NUM_LAYERS: usize = 12;
const CLIP_MAX_POSITION: usize = 77;

// OpenCLIP ViT-bigG/14 text encoder config (TE2)
const OPENCLIP_HIDDEN_DIM: usize = 1280;
const OPENCLIP_NUM_HEADS: usize = 20;
const OPENCLIP_NUM_LAYERS: usize = 32;
const OPENCLIP_MAX_POSITION: usize = 77;

/// ONNX model bytes for the four SDXL sub-models.
pub type SdXlModels = (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>);

/// Load an SDXL checkpoint (.safetensors) and produce four ONNX models.
/// Returns (text_encoder_1_onnx, text_encoder_2_onnx, unet_onnx, vae_decoder_onnx).
pub fn load_sdxl_checkpoint(
    checkpoint_path: &Path,
    output_method: WeightStorageStrategy,
) -> Result<SdXlModels, anyhow::Error> {
    let file = std::fs::File::open(checkpoint_path)?;
    let mmap = unsafe { Mmap::map(&file) }?;
    let weight_manager = SafetensorsWeightManager::new(vec![Arc::new(mmap)])?;

    let model_dtype = sd_common::detect_model_dtype(&weight_manager);
    println!("Detected model dtype: {:?}", model_dtype);

    let origin_path =
        std::fs::canonicalize(checkpoint_path).unwrap_or_else(|_| checkpoint_path.to_path_buf());

    println!("Building SDXL text encoder 1 (CLIP ViT-L/14)...");
    let te1 = build_text_encoder_1(
        &weight_manager,
        model_dtype,
        output_method.clone(),
        &origin_path,
    )?;

    println!("Building SDXL text encoder 2 (OpenCLIP ViT-bigG/14)...");
    let te2 = build_text_encoder_2(
        &weight_manager,
        model_dtype,
        output_method.clone(),
        &origin_path,
    )?;

    println!("Building SDXL UNet...");
    let unet = build_unet(
        &weight_manager,
        model_dtype,
        output_method.clone(),
        &origin_path,
    )?;

    println!("Building SDXL VAE decoder...");
    let vae_decoder =
        sd_common::build_vae_decoder(&weight_manager, model_dtype, output_method, &origin_path)?;

    Ok((te1, te2, unet, vae_decoder))
}

/// Detect if a safetensors file is an SDXL checkpoint.
pub fn is_sdxl_checkpoint(weight_manager: &impl WeightManager) -> bool {
    // SDXL has dual text encoders under conditioner.embedders.{0,1}
    weight_manager
        .get_tensor("model.diffusion_model.input_blocks.0.0.weight")
        .is_ok()
        && weight_manager
            .get_tensor("conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight")
            .is_ok()
        && weight_manager
            .get_tensor("conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj_weight")
            .is_ok()
}

// ============================================================================
// Text Encoder 1: CLIP ViT-L/14 (same architecture as SD 1.5, different prefix)
// ============================================================================

/// Build CLIP ViT-L/14 text encoder (TE1).
///
/// Same architecture as SD 1.5 CLIP but:
/// - Weight prefix: `conditioner.embedders.0.transformer.text_model`
/// - Outputs penultimate hidden state (after layer 10, skipping layer 11 + final_layer_norm)
pub fn build_text_encoder_1(
    weight_manager: &impl WeightManager,
    model_dtype: DType,
    output_method: WeightStorageStrategy,
    origin_path: &Path,
) -> Result<Vec<u8>, anyhow::Error> {
    let wm = CastingWeightManager::new(
        weight_manager.prefix("conditioner.embedders.0.transformer.text_model"),
        DType::F32,
    );
    let emb_wm = wm.prefix("embeddings");
    let _ = model_dtype;

    let batch_dim = Dimension::new(Some(1), Some("batch_size".to_string()), None);
    let seq_dim = Dimension::new(Some(CLIP_MAX_POSITION), Some("seq_len".to_string()), None);
    let input_shape = Shape::new(vec![batch_dim.clone(), seq_dim.clone()]);

    let input_ids = InputTensor::new("input_ids".to_string(), DType::I32, input_shape);

    // Token embedding
    let token_emb = Gather::new(
        Some("token_embedding".to_string()),
        emb_wm.get_tensor("token_embedding.weight")?,
        input_ids.clone(),
        0,
    )?;

    // Position embedding
    let pos_emb = emb_wm.get_tensor("position_embedding.weight")?;
    let x = Add::new(Some("pos_embed".to_string()), token_emb, pos_emb)?;

    // Build causal mask
    let mask_data = build_causal_mask(CLIP_MAX_POSITION);
    let causal_mask = InputTensorInitialized::new(
        "causal_mask".to_string(),
        TensorData::new(
            mask_data.into(),
            Shape::new(vec![
                Dimension::new(Some(1), None, None),
                Dimension::new(Some(1), None, None),
                Dimension::new(Some(CLIP_MAX_POSITION), None, None),
                Dimension::new(Some(CLIP_MAX_POSITION), None, None),
            ]),
        )?,
    );

    // Run layers 0..10 (11 layers), skip layer 11 = penultimate hidden state
    let mut hidden: Arc<dyn Tensor> = x;
    for i in 0..(CLIP_NUM_LAYERS - 1) {
        let layer_wm = wm.prefix(&format!("encoder.layers.{i}"));
        hidden = clip_encoder_layer(&layer_wm, hidden, causal_mask.clone())?;
    }

    // No final_layer_norm — output penultimate hidden state directly.

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

fn clip_encoder_layer(
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

fn clip_attention(
    wm: &impl WeightManager,
    input: Arc<dyn Tensor>,
    causal_mask: Arc<dyn Tensor>,
) -> Result<Arc<dyn Tensor>, Error> {
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
    let scores = Add::new(None, scores, causal_mask)?;

    let attn_weights = Softmax::new(None, scores, Some(-1));
    let attn_output = MatMul::new(None, attn_weights, v)?;

    let attn_output = Transpose::new(None, attn_output, Some(vec![0, 2, 1, 3]));
    let attn_output =
        crate::onnx_graph::pytorch::reshape(attn_output, vec![0, -1, CLIP_HIDDEN_DIM as i64])?;

    linear(&wm.prefix("out_proj"), attn_output)
}

fn clip_mlp(
    wm: &impl WeightManager,
    input: Arc<dyn Tensor>,
) -> Result<Arc<dyn Tensor>, Error> {
    let x = linear(&wm.prefix("fc1"), input)?;
    let x = crate::onnx_graph::pytorch::quick_gelu(x)?;
    linear(&wm.prefix("fc2"), x)
}

// ============================================================================
// Text Encoder 2: OpenCLIP ViT-bigG/14
// ============================================================================

/// Build OpenCLIP ViT-bigG/14 text encoder (TE2).
///
/// Outputs both:
/// - `penultimate_hidden_state` [1, 77, 1280]: after layer 30 (for UNet cross-attention)
/// - `pooled_output` [1, 1280]: from EOS token of final layer through text_projection
///
/// Takes `eos_indices` [1] I64 as extra input: the position of the EOS token in each sequence.
pub fn build_text_encoder_2(
    weight_manager: &impl WeightManager,
    model_dtype: DType,
    output_method: WeightStorageStrategy,
    origin_path: &Path,
) -> Result<Vec<u8>, anyhow::Error> {
    let wm = CastingWeightManager::new(
        weight_manager.prefix("conditioner.embedders.1.model"),
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
    let eos_indices = InputTensor::new(
        "eos_indices".to_string(),
        DType::I64,
        Shape::new(vec![batch_dim.clone()]),
    );

    // Token embedding
    let token_emb = Gather::new(
        Some("token_embedding".to_string()),
        wm.get_tensor("token_embedding.weight")?,
        input_ids.clone(),
        0,
    )?;

    // Position embedding
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

    // Run layers 0..30 → penultimate hidden state
    let mut hidden: Arc<dyn Tensor> = x;
    for i in 0..(OPENCLIP_NUM_LAYERS - 1) {
        let layer_wm = wm.prefix(&format!("transformer.resblocks.{i}"));
        hidden = openclip_encoder_layer(&layer_wm, hidden, causal_mask.clone())?;
    }
    let penultimate_hidden = hidden.clone();

    // Run last layer (31) for pooled output
    {
        let layer_wm = wm.prefix(&format!(
            "transformer.resblocks.{}",
            OPENCLIP_NUM_LAYERS - 1
        ));
        hidden = openclip_encoder_layer(&layer_wm, hidden, causal_mask)?;
    }

    // Apply ln_final
    let final_normed = layer_norm(&wm.prefix("ln_final"), hidden, 1e-5)?;

    // Extract EOS token: Gather along seq dim (axis=1) using eos_indices
    // eos_indices: [batch] → reshape to [batch, 1] for Gather
    let eos_idx_2d = crate::onnx_graph::pytorch::unsqueeze(eos_indices.clone(), 1)?;
    // Gather: [batch, 77, 1280] gather with [batch, 1] at axis=1 → [batch, 1, 1280]
    let eos_token = Gather::new(None, final_normed, eos_idx_2d, 1)?;
    // Squeeze to [batch, 1280]
    let eos_token = crate::onnx_graph::pytorch::squeeze(eos_token, 1)?;

    // Project through text_projection: [batch, 1280] @ [1280, 1280] → [batch, 1280]
    let text_projection = wm.get_tensor("text_projection")?;
    let pooled_output = MatMul::new(None, eos_token, text_projection)?;

    let input_tensors: Vec<Arc<dyn Tensor>> = vec![input_ids, eos_indices];
    let output_tensors: Vec<(String, Arc<dyn Tensor>)> = vec![
        ("penultimate_hidden_state".to_string(), penultimate_hidden),
        ("pooled_output".to_string(), pooled_output),
    ];

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
    let normed = layer_norm(&wm.prefix("ln_1"), input.clone(), 1e-5)?;
    let attn_out = openclip_attention(&wm.prefix("attn"), normed, causal_mask)?;
    let x = Add::new(None, input, attn_out)?;

    let normed = layer_norm(&wm.prefix("ln_2"), x.clone(), 1e-5)?;
    let mlp_out = openclip_mlp(&wm.prefix("mlp"), normed)?;
    Ok(Add::new(None, x, mlp_out)?)
}

fn openclip_attention(
    wm: &impl WeightManager,
    input: Arc<dyn Tensor>,
    causal_mask: Arc<dyn Tensor>,
) -> Result<Arc<dyn Tensor>, Error> {
    let head_dim = OPENCLIP_HIDDEN_DIM / OPENCLIP_NUM_HEADS;

    // Combined QKV: in_proj_weight [3*1280, 1280], in_proj_bias [3*1280]
    let in_proj_weight = wm.get_tensor("in_proj_weight")?;
    let in_proj_bias = wm.get_tensor("in_proj_bias")?;

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

    let q = matmul_with_transposed_weight(input.clone(), q_weight, Some(q_bias))?;
    let k = matmul_with_transposed_weight(input.clone(), k_weight, Some(k_bias))?;
    let v = matmul_with_transposed_weight(input, v_weight, Some(v_bias))?;

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

    let kt = Transpose::new(None, k, Some(vec![0, 1, 3, 2]));
    let scores = MatMul::new(None, q, kt)?;
    let scores = div_scalar(scores, (head_dim as f32).sqrt())?;
    let scores = Add::new(None, scores, causal_mask)?;

    let attn_weights = Softmax::new(None, scores, Some(-1));
    let attn_output = MatMul::new(None, attn_weights, v)?;

    let attn_output = Transpose::new(None, attn_output, Some(vec![0, 2, 1, 3]));
    let attn_output =
        reshape_symbolic(attn_output, vec![0, -1, OPENCLIP_HIDDEN_DIM as i64])?;

    linear(&wm.prefix("out_proj"), attn_output)
}

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
// UNet (SDXL)
// ============================================================================

/// Build the SDXL UNet.
///
/// Key differences from SD 2:
/// - 3 levels (channel_mult=[1,2,4]) instead of 4
/// - Variable transformer depth: [0, 2, 10] (no attention at level 0)
/// - ADM conditioning via label_emb (processes pooled text embed + micro-conditioning)
/// - Additional input: y [batch, 2816]
pub fn build_unet(
    weight_manager: &impl WeightManager,
    model_dtype: DType,
    output_method: WeightStorageStrategy,
    origin_path: &Path,
) -> Result<Vec<u8>, anyhow::Error> {
    let wm = CastingWeightManager::new(weight_manager.prefix("model.diffusion_model"), DType::F32);

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
        DType::F32,
        Shape::new(vec![
            batch_dim.clone(),
            Dimension::new(Some(CLIP_MAX_POSITION), None, None),
            Dimension::new(Some(CONTEXT_DIM), None, None),
        ]),
    );
    // ADM conditioning input
    let y_input = InputTensor::new(
        "y".to_string(),
        DType::F32,
        Shape::new(vec![
            batch_dim,
            Dimension::new(Some(ADM_IN_CHANNELS), None, None),
        ]),
    );

    let sample = cast(sample_input.clone(), DType::F32);
    let context = context_input.clone();
    let y: Arc<dyn Tensor> = y_input.clone();

    // Timestep embedding
    let t_emb =
        sd_common::timestep_embedding(&wm, cast(timestep_input.clone(), DType::F32), MODEL_CHANNELS)?;

    // Label embedding (ADM): y → Linear → SiLU → Linear → add to t_emb
    let y_emb = linear(&wm.prefix("label_emb.0.0"), y)?;
    let y_emb = silu(y_emb)?;
    let y_emb = linear(&wm.prefix("label_emb.0.2"), y_emb)?;
    let t_emb = Add::new(None, t_emb, y_emb)?;

    let channels = CHANNEL_MULT.map(|m| m * MODEL_CHANNELS);
    // channels = [320, 640, 1280]

    // input_blocks.0: initial conv
    let h = conv2d(&wm.prefix("input_blocks.0.0"), sample, 3, 1, 1)?;
    let mut skip_connections: Vec<Arc<dyn Tensor>> = vec![h.clone()];

    let mut h = h;
    let mut input_block_idx = 1;
    let mut current_ch = MODEL_CHANNELS;

    for (level, &ch) in channels.iter().enumerate() {
        let td = TRANSFORMER_DEPTH[level];

        for _block in 0..NUM_RES_BLOCKS {
            h = resnet_block(
                &wm.prefix(&format!("input_blocks.{input_block_idx}.0")),
                h,
                t_emb.clone(),
                current_ch,
                ch,
            )?;
            current_ch = ch;

            // Attention only at levels with transformer_depth > 0
            if td > 0 {
                let num_heads = ch / NUM_HEAD_CHANNELS;
                h = spatial_transformer(
                    &wm.prefix(&format!("input_blocks.{input_block_idx}.1")),
                    h,
                    context.clone(),
                    ch,
                    num_heads,
                    td,
                    true, // SDXL uses linear proj_in/proj_out
                )?;
            }

            skip_connections.push(h.clone());
            input_block_idx += 1;
        }

        // Downsample (except at last level)
        if level < channels.len() - 1 {
            h = downsample(&wm.prefix(&format!("input_blocks.{input_block_idx}.0")), h)?;
            skip_connections.push(h.clone());
            input_block_idx += 1;
        }
    }

    // Middle block — uses the deepest level's transformer depth (10)
    let mid_td = TRANSFORMER_DEPTH[channels.len() - 1];
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
        mid_td,
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
    for level in (0..channels.len()).rev() {
        let ch = channels[level];
        let td = TRANSFORMER_DEPTH[level];

        for block in 0..(NUM_RES_BLOCKS + 1) {
            let skip = skip_connections.pop().unwrap();
            h = Concat::new(None, vec![h, skip], 1)?;

            let concat_ch = h.shape()[1].resolve()?;

            h = resnet_block(
                &wm.prefix(&format!("output_blocks.{output_block_idx}.0")),
                h,
                t_emb.clone(),
                concat_ch,
                ch,
            )?;

            if td > 0 {
                let num_heads = ch / NUM_HEAD_CHANNELS;
                h = spatial_transformer(
                    &wm.prefix(&format!("output_blocks.{output_block_idx}.1")),
                    h,
                    context.clone(),
                    ch,
                    num_heads,
                    td,
                    true,
                )?;
            }

            // Upsample at end of each level (except level 0), at last block
            if level > 0 && block == NUM_RES_BLOCKS {
                let upsample_idx = if td > 0 { 2 } else { 1 };
                h = upsample(
                    &wm.prefix(&format!("output_blocks.{output_block_idx}.{upsample_idx}")),
                    h,
                )?;
            }

            output_block_idx += 1;
        }
    }

    // Output: GroupNorm -> SiLU -> Conv
    h = group_norm(&wm.prefix("out.0"), h, 1e-5, 32)?;
    h = silu(h)?;
    h = conv2d(&wm.prefix("out.2"), h, 3, 1, 1)?;

    let output = cast(h, model_dtype);

    let input_tensors: Vec<Arc<dyn Tensor>> =
        vec![sample_input, timestep_input, context_input, y_input];
    let output_tensors: Vec<(String, Arc<dyn Tensor>)> = vec![("out_sample".to_string(), output)];

    let onnx_model = crate::onnx_graph::build_proto_with_origin_path(
        &input_tensors,
        &output_tensors,
        output_method,
        Some(origin_path),
    )?;
    Ok(onnx_model.encode_to_vec())
}
