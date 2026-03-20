use crate::models::diffusion::sd_common::{
    CastingWeightManager, adaln_modulate, cos_op, layer_norm_bare, ones_constant, sin_op,
    slice_axis, split_chunks,
};
use crate::onnx_graph::Error;
use crate::onnx_graph::WeightStorageStrategy;
use crate::onnx_graph::operators::{
    Add, Concat, Constant, Conv, LayerNormalization, MatMul, Mul, Resize, RotaryEmbedding, Softmax,
    Transpose,
};
use crate::onnx_graph::pytorch::{
    cast, div_scalar, gelu_pytorch_tanh, linear, reshape, rms_norm, silu, unsqueeze,
};
use crate::onnx_graph::tensor::{
    DType, Dimension, InputTensor, InputTensorInitialized, Shape, Tensor, TensorData,
    TensorDataValue,
};
use crate::onnx_graph::weights::WeightManager;
use prost::Message;
use std::path::Path;
use std::sync::Arc;

// =============================================================================
// Config
// =============================================================================

#[derive(Clone, Debug)]
pub struct WanTransformerConfig {
    pub dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub ffn_dim: usize,
    pub freq_dim: usize,
    pub in_dim: usize,
    pub out_dim: usize,
    pub text_dim: usize,
    pub text_len: usize,
    pub patch_size: [usize; 3],
    pub rope_max_seq_len: usize,
    pub eps: f32,
}

impl WanTransformerConfig {
    pub fn head_dim(&self) -> usize {
        self.dim / self.num_heads
    }

    pub fn wan_1_3b() -> Self {
        Self {
            dim: 1536,
            num_heads: 12,
            num_layers: 30,
            ffn_dim: 8960,
            freq_dim: 256,
            in_dim: 16,
            out_dim: 16,
            text_dim: 4096,
            text_len: 512,
            patch_size: [1, 2, 2],
            rope_max_seq_len: 1024,
            eps: 1e-6,
        }
    }

    pub fn wan_14b() -> Self {
        Self {
            dim: 5120,
            num_heads: 40,
            num_layers: 40,
            ffn_dim: 13824,
            freq_dim: 256,
            in_dim: 16,
            out_dim: 16,
            text_dim: 4096,
            text_len: 512,
            patch_size: [1, 2, 2],
            rope_max_seq_len: 1024,
            eps: 1e-6,
        }
    }
}

#[derive(Clone, Debug)]
pub struct WanVaeConfig {
    pub base_dim: usize,
    pub z_dim: usize,
    pub dim_mult: Vec<usize>,
    pub num_res_blocks: usize,
    pub temporal_downsample: Vec<bool>,
    pub in_channels: usize,
    pub out_channels: usize,
    pub eps: f32,
}

impl WanVaeConfig {
    pub fn default_config() -> Self {
        Self {
            base_dim: 96,
            z_dim: 16,
            dim_mult: vec![1, 2, 4, 4],
            num_res_blocks: 2,
            temporal_downsample: vec![false, true, true],
            in_channels: 3,
            out_channels: 3,
            eps: 1e-6,
        }
    }
}

// Shared helpers (slice_axis, split_chunks, ones_constant, layer_norm_bare, adaln_modulate)
// are in sd_common.rs

// =============================================================================
// 3D RoPE for Wan2.1
// =============================================================================

/// Precompute 3D factored RoPE cos/sin caches.
///
/// Wan splits head_dim=128 as: temporal=44, height=42, width=42.
/// Returns (cos_cache, sin_cache) each [video_seq_len, head_dim/2].
fn precompute_wan_3d_rope(
    config: &WanTransformerConfig,
    latent_frames: usize,
    patch_h: usize,
    patch_w: usize,
) -> (Vec<f32>, Vec<f32>) {
    let head_dim = config.head_dim();
    let half_head = head_dim / 2; // 64

    // Wan dimension split: h_dim = w_dim = 2 * (head_dim / 6), t_dim = head_dim - h_dim - w_dim
    let h_dim = 2 * (head_dim / 6); // 42
    let w_dim = h_dim; // 42
    let t_dim = head_dim - h_dim - w_dim; // 44
    let axes_dim = [t_dim, h_dim, w_dim];
    let half_dims = [t_dim / 2, h_dim / 2, w_dim / 2]; // [22, 21, 21]

    let theta = 10000.0f64;

    let inv_freqs: Vec<Vec<f64>> = axes_dim
        .iter()
        .map(|&axis_dim| {
            (0..axis_dim / 2)
                .map(|i| 1.0 / theta.powf(2.0 * i as f64 / axis_dim as f64))
                .collect()
        })
        .collect();

    let video_seq = latent_frames * patch_h * patch_w;
    let mut cos_cache = vec![0.0f32; video_seq * half_head];
    let mut sin_cache = vec![0.0f32; video_seq * half_head];

    for vid_idx in 0..video_seq {
        let t = (vid_idx / (patch_h * patch_w)) as f64;
        let spatial_idx = vid_idx % (patch_h * patch_w);
        let y = (spatial_idx / patch_w) as f64;
        let x = (spatial_idx % patch_w) as f64;

        let positions = [t, y, x];
        let mut offset = 0;
        for (axis, &pos_val) in positions.iter().enumerate() {
            for (i, &freq) in inv_freqs[axis].iter().enumerate() {
                let angle = pos_val * freq;
                cos_cache[vid_idx * half_head + offset + i] = angle.cos() as f32;
                sin_cache[vid_idx * half_head + offset + i] = angle.sin() as f32;
            }
            offset += half_dims[axis];
        }
    }

    (cos_cache, sin_cache)
}

// =============================================================================
// Wan2.1 Transformer Components
// =============================================================================

/// Wan timestep + text condition embedding.
///
/// Returns (temb, timestep_proj, text_embeds):
/// - temb: [B, dim] — raw time embedding for output norm
/// - timestep_proj: [B, 1, 6, dim] — per-block modulation
/// - text_embeds: [B, seq, dim] — projected text
fn wan_condition_embedding(
    wm: &impl WeightManager,
    timestep: Arc<dyn Tensor>,
    encoder_hidden_states: Arc<dyn Tensor>,
    config: &WanTransformerConfig,
    model_dtype: DType,
) -> Result<(Arc<dyn Tensor>, Arc<dyn Tensor>, Arc<dyn Tensor>), Error> {
    // Timestep sinusoidal embedding
    let timestep = cast(timestep, DType::F32);
    let half_dim = config.freq_dim / 2;
    let freqs: Vec<f32> = (0..half_dim)
        .map(|i| (-(10000.0f32.ln()) * i as f32 / half_dim as f32).exp())
        .collect();
    let freq_tensor = InputTensorInitialized::new(
        "wan_timestep_freqs".to_string(),
        TensorData::new(
            freqs.into(),
            Shape::new(vec![
                Dimension::new(Some(1), None, None),
                Dimension::new(Some(half_dim), None, None),
            ]),
        )?,
    );
    let args = Mul::new(None, timestep, freq_tensor)?;
    let cos_part = cos_op(args.clone())?;
    let sin_part = sin_op(args)?;
    let t_emb = Concat::new(None, vec![cos_part, sin_part], -1)?;
    let t_emb = cast(t_emb, model_dtype);

    // TimestepEmbedding: Linear -> SiLU -> Linear
    let temb = linear(
        &wm.prefix("condition_embedder.time_embedder.linear_1"),
        t_emb,
    )?;
    let temb = silu(temb)?;
    let temb = linear(
        &wm.prefix("condition_embedder.time_embedder.linear_2"),
        temb,
    )?;

    // Save temb for output norm (before 6x projection)
    let temb_out = temb.clone();

    // Project to 6*dim for modulation: Linear(dim -> 6*dim), reshape to [B, 6, dim]
    let timestep_proj = linear(&wm.prefix("condition_embedder.time_proj"), temb)?;
    let timestep_proj = reshape(timestep_proj, vec![0, 6, config.dim as i64])?;
    let timestep_proj = unsqueeze(timestep_proj, 1)?; // [B, 1, 6, dim]

    // Text projection: Linear -> GELU_tanh -> Linear
    let text = linear(
        &wm.prefix("condition_embedder.text_embedder.linear_1"),
        encoder_hidden_states,
    )?;
    let text = gelu_pytorch_tanh(text)?;
    let text = linear(
        &wm.prefix("condition_embedder.text_embedder.linear_2"),
        text,
    )?;

    Ok((temb_out, timestep_proj, text))
}

/// Wan self-attention with RoPE + RMSNorm on Q/K.
fn wan_self_attention(
    wm: &impl WeightManager,
    hidden_states: Arc<dyn Tensor>,
    config: &WanTransformerConfig,
    rope_cos: Arc<dyn Tensor>,
    rope_sin: Arc<dyn Tensor>,
) -> Result<Arc<dyn Tensor>, Error> {
    let nh = config.num_heads as i64;
    let hd = config.head_dim() as i64;

    let q = linear(&wm.prefix("to_q"), hidden_states.clone())?;
    let k = linear(&wm.prefix("to_k"), hidden_states.clone())?;
    let v = linear(&wm.prefix("to_v"), hidden_states)?;

    // Reshape: [B, seq, D] -> [B, seq, nh, hd] -> [B, nh, seq, hd]
    let q = Transpose::new(
        None,
        reshape(q, vec![0, 0, nh, hd])?,
        Some(vec![0, 2, 1, 3]),
    );
    let k = Transpose::new(
        None,
        reshape(k, vec![0, 0, nh, hd])?,
        Some(vec![0, 2, 1, 3]),
    );
    let v = Transpose::new(
        None,
        reshape(v, vec![0, 0, nh, hd])?,
        Some(vec![0, 2, 1, 3]),
    );

    // RMSNorm on Q/K (learned affine)
    let q = rms_norm(&wm.prefix("norm_q"), q, Some(config.eps))?;
    let k = rms_norm(&wm.prefix("norm_k"), k, Some(config.eps))?;

    // Apply 3D RoPE (interleaved=1, like Flux)
    let q = RotaryEmbedding::new(
        None,
        q,
        rope_cos.clone(),
        rope_sin.clone(),
        None,
        Some(1),
        None,
        None,
    )?;
    let k = RotaryEmbedding::new(None, k, rope_cos, rope_sin, None, Some(1), None, None)?;

    // Scaled dot-product attention
    let scores = MatMul::new(None, q, Transpose::new(None, k, Some(vec![0, 1, 3, 2])))?;
    let scores = div_scalar(scores, (config.head_dim() as f32).sqrt())?;
    let attn = Softmax::new(None, scores, Some(-1));
    let attn_out = MatMul::new(None, attn, v)?;

    // Reshape: [B, nh, seq, hd] -> [B, seq, D]
    let attn_out = Transpose::new(None, attn_out, Some(vec![0, 2, 1, 3]));
    let attn_out = reshape(attn_out, vec![0, 0, config.dim as i64])?;

    // Output projection
    linear(&wm.prefix("to_out.0"), attn_out)
}

/// Wan cross-attention (text → video). No RoPE, no gating.
fn wan_cross_attention(
    wm: &impl WeightManager,
    hidden_states: Arc<dyn Tensor>,
    encoder_hidden_states: Arc<dyn Tensor>,
    config: &WanTransformerConfig,
) -> Result<Arc<dyn Tensor>, Error> {
    let nh = config.num_heads as i64;
    let hd = config.head_dim() as i64;

    let q = linear(&wm.prefix("to_q"), hidden_states)?;
    let k = linear(&wm.prefix("to_k"), encoder_hidden_states.clone())?;
    let v = linear(&wm.prefix("to_v"), encoder_hidden_states)?;

    let q = Transpose::new(
        None,
        reshape(q, vec![0, 0, nh, hd])?,
        Some(vec![0, 2, 1, 3]),
    );
    let k = Transpose::new(
        None,
        reshape(k, vec![0, 0, nh, hd])?,
        Some(vec![0, 2, 1, 3]),
    );
    let v = Transpose::new(
        None,
        reshape(v, vec![0, 0, nh, hd])?,
        Some(vec![0, 2, 1, 3]),
    );

    // RMSNorm on Q/K
    let q = rms_norm(&wm.prefix("norm_q"), q, Some(config.eps))?;
    let k = rms_norm(&wm.prefix("norm_k"), k, Some(config.eps))?;

    // No RoPE for cross-attention

    let scores = MatMul::new(None, q, Transpose::new(None, k, Some(vec![0, 1, 3, 2])))?;
    let scores = div_scalar(scores, (config.head_dim() as f32).sqrt())?;
    let attn = Softmax::new(None, scores, Some(-1));
    let attn_out = MatMul::new(None, attn, v)?;

    let attn_out = Transpose::new(None, attn_out, Some(vec![0, 2, 1, 3]));
    let attn_out = reshape(attn_out, vec![0, 0, config.dim as i64])?;

    linear(&wm.prefix("to_out.0"), attn_out)
}

/// Wan transformer block.
///
/// 1. Modulation from scale_shift_table + timestep_proj -> 6 values
/// 2. Self-attention with AdaLN + RoPE + gating
/// 3. Cross-attention with LayerNorm (no gating)
/// 4. Feed-forward with AdaLN + gating
fn wan_block(
    wm: &impl WeightManager,
    hidden_states: Arc<dyn Tensor>,
    encoder_hidden_states: Arc<dyn Tensor>,
    timestep_proj: Arc<dyn Tensor>,
    config: &WanTransformerConfig,
    rope_cos: Arc<dyn Tensor>,
    rope_sin: Arc<dyn Tensor>,
) -> Result<Arc<dyn Tensor>, Error> {
    let dim = config.dim;
    let eps = config.eps;

    // Modulation: scale_shift_table [1, 6, dim] + timestep_proj [B, 1, 6, dim]
    // -> [B, 1, 6, dim] -> squeeze -> [B, 6, dim] -> chunk(6)
    let sst = wm.get_tensor("scale_shift_table")?;
    let modulation = Add::new(None, sst, timestep_proj)?;
    // modulation is [B, 1, 6, dim], squeeze the seq dim
    let modulation = reshape(modulation, vec![0, 6, dim as i64])?;
    let chunks = split_chunks(modulation, dim, 6)?;
    let (shift_msa, scale_msa, gate_msa) =
        (chunks[0].clone(), chunks[1].clone(), chunks[2].clone());
    let (c_shift, c_scale, c_gate) = (chunks[3].clone(), chunks[4].clone(), chunks[5].clone());

    // Unsqueeze modulation values for broadcasting: [B, dim] -> [B, 1, dim]
    let shift_msa = unsqueeze(shift_msa, 1)?;
    let scale_msa = unsqueeze(scale_msa, 1)?;
    let gate_msa = unsqueeze(gate_msa, 1)?;
    let c_shift = unsqueeze(c_shift, 1)?;
    let c_scale = unsqueeze(c_scale, 1)?;
    let c_gate = unsqueeze(c_gate, 1)?;

    // 1. Self-attention with AdaLN
    let normed = adaln_modulate(hidden_states.clone(), shift_msa, scale_msa, dim, eps)?;
    let attn_out = wan_self_attention(&wm.prefix("attn1"), normed, config, rope_cos, rope_sin)?;
    let hidden_states = Add::new(None, hidden_states, Mul::new(None, gate_msa, attn_out)?)?;

    // 2. Cross-attention with LayerNorm (cross_attn_norm=true -> learned LN)
    let normed_cross =
        crate::onnx_graph::pytorch::layer_norm(&wm.prefix("norm2"), hidden_states.clone(), eps)?;
    let cross_out = wan_cross_attention(
        &wm.prefix("attn2"),
        normed_cross,
        encoder_hidden_states,
        config,
    )?;
    // No gate on cross-attention
    let hidden_states = Add::new(None, hidden_states, cross_out)?;

    // 3. Feed-forward with AdaLN
    let normed_ff = adaln_modulate(hidden_states.clone(), c_shift, c_scale, dim, eps)?;
    let ff = linear(&wm.prefix("ffn.net.0.proj"), normed_ff)?;
    let ff = gelu_pytorch_tanh(ff)?;
    let ff = linear(&wm.prefix("ffn.net.2"), ff)?;
    let hidden_states = Add::new(None, hidden_states, Mul::new(None, c_gate, ff)?)?;

    Ok(hidden_states as Arc<dyn Tensor>)
}

/// Wan unpatchify: [B, F*pH*pW, pT*pH_s*pW_s*C] -> [B, C, F*pT, H, W]
fn wan_unpatchify(
    input: Arc<dyn Tensor>,
    config: &WanTransformerConfig,
    latent_frames: usize,
    patch_h: usize,
    patch_w: usize,
) -> Result<Arc<dyn Tensor>, Error> {
    let f = latent_frames as i64;
    let ph = patch_h as i64;
    let pw = patch_w as i64;
    let [pt, ps_h, ps_w] = [
        config.patch_size[0] as i64,
        config.patch_size[1] as i64,
        config.patch_size[2] as i64,
    ];
    let c = config.out_dim as i64;

    // [B, F*pH*pW, pt*ps_h*ps_w*C]
    let x = reshape(input, vec![0, f, ph, pw, pt, ps_h, ps_w, c])?;
    // -> [B, C, F, pt, pH, ps_h, pW, ps_w]
    let x = Transpose::new(None, x, Some(vec![0, 7, 1, 4, 2, 5, 3, 6]));
    // -> [B, C, F*pt, pH*ps_h, pW*ps_w]
    reshape(x, vec![0, c, f * pt, ph * ps_h, pw * ps_w]).map(|x| x as Arc<dyn Tensor>)
}

// =============================================================================
// Main Transformer Builder
// =============================================================================

pub fn load_wan_transformer(
    weight_manager: impl WeightManager,
    config: WanTransformerConfig,
    output_method: WeightStorageStrategy,
) -> Result<Vec<u8>, anyhow::Error> {
    load_wan_transformer_with_origin(weight_manager, config, output_method, None)
}

pub fn load_wan_transformer_with_origin(
    weight_manager: impl WeightManager,
    config: WanTransformerConfig,
    output_method: WeightStorageStrategy,
    origin_path: Option<&Path>,
) -> Result<Vec<u8>, anyhow::Error> {
    let model_dtype = weight_manager
        .get_tensor("blocks.0.attn1.to_q.weight")
        .map(|t| t.dtype())
        .unwrap_or(DType::BF16);
    let wm = CastingWeightManager::new(weight_manager, model_dtype);

    let batch_dim = Dimension::new(Some(1), Some("batch".to_string()), None);

    // For 1.3B defaults: 81 frames -> 21 latent frames, 480x832 -> 60x104 latent
    let latent_frames = 21usize;
    let latent_h = 60usize;
    let latent_w = 104usize;
    let patch_h = latent_h / config.patch_size[1];
    let patch_w = latent_w / config.patch_size[2];
    let video_seq = latent_frames * patch_h * patch_w;

    let latent_input = InputTensor::new(
        "hidden_states".to_string(),
        model_dtype,
        Shape::new(vec![
            batch_dim.clone(),
            Dimension::new(Some(config.in_dim), None, None),
            Dimension::new(Some(latent_frames), None, None),
            Dimension::new(Some(latent_h), None, None),
            Dimension::new(Some(latent_w), None, None),
        ]),
    );
    let encoder_hidden_states_input = InputTensor::new(
        "encoder_hidden_states".to_string(),
        model_dtype,
        Shape::new(vec![
            batch_dim.clone(),
            Dimension::new(Some(config.text_len), None, None),
            Dimension::new(Some(config.text_dim), None, None),
        ]),
    );
    let timestep_input = InputTensor::new(
        "timestep".to_string(),
        model_dtype,
        Shape::new(vec![batch_dim.clone()]),
    );

    let input_tensors: Vec<Arc<dyn Tensor>> = vec![
        latent_input.clone(),
        encoder_hidden_states_input.clone(),
        timestep_input.clone(),
    ];

    // 1. Condition embedding (timestep + text)
    let (temb, timestep_proj, text_embeds) = wan_condition_embedding(
        &wm,
        timestep_input,
        encoder_hidden_states_input,
        &config,
        model_dtype,
    )?;

    // 2. Patch embedding: Conv3d(in_dim, dim, kernel=(1,2,2), stride=(1,2,2))
    let patch_weight = wm.get_tensor("patch_embedding.weight")?;
    let patch_bias = wm.get_tensor("patch_embedding.bias").ok();
    let patched = Conv::new(
        Some("patch_embedding".to_string()),
        latent_input,
        patch_weight,
        patch_bias,
        vec![
            config.patch_size[0] as i64,
            config.patch_size[1] as i64,
            config.patch_size[2] as i64,
        ],
        vec![
            config.patch_size[0] as i64,
            config.patch_size[1] as i64,
            config.patch_size[2] as i64,
        ],
        vec![0, 0, 0, 0, 0, 0],
        vec![1, 1, 1],
        1,
    )?;
    // [B, dim, F, pH, pW] -> [B, F*pH*pW, dim]
    let hidden_states = reshape(patched, vec![0, config.dim as i64, -1])?;
    let hidden_states = Transpose::new(None, hidden_states, Some(vec![0, 2, 1]));

    // 3. Precompute 3D RoPE
    let (cos_vals, sin_vals) = precompute_wan_3d_rope(&config, latent_frames, patch_h, patch_w);
    let half_head = config.head_dim() / 2;
    let rope_shape = Shape::new(vec![
        Dimension::new(Some(video_seq), None, None),
        Dimension::new(Some(half_head), None, None),
    ]);
    let rope_cos: Arc<dyn Tensor> = InputTensorInitialized::new(
        "rope_cos_cache".to_string(),
        TensorData::new(TensorDataValue::F32(cos_vals), rope_shape.clone())?,
    );
    let rope_sin: Arc<dyn Tensor> = InputTensorInitialized::new(
        "rope_sin_cache".to_string(),
        TensorData::new(TensorDataValue::F32(sin_vals), rope_shape)?,
    );
    let rope_cos = cast(rope_cos, model_dtype);
    let rope_sin = cast(rope_sin, model_dtype);

    // 4. Transformer blocks
    println!(
        "Building Wan2.1 transformer: {} blocks, dim={}...",
        config.num_layers, config.dim
    );
    let mut hidden_states: Arc<dyn Tensor> = hidden_states;
    for i in 0..config.num_layers {
        let block_wm = wm.prefix(&format!("blocks.{i}"));
        hidden_states = wan_block(
            &block_wm,
            hidden_states,
            text_embeds.clone(),
            timestep_proj.clone(),
            &config,
            rope_cos.clone(),
            rope_sin.clone(),
        )?;
        if (i + 1) % 6 == 0 {
            println!("  Block {}/{}", i + 1, config.num_layers);
        }
    }

    // 5. Final norm + modulation + projection
    // scale_shift_table [1, 2, dim] + temb [B, dim] -> [B, 2, dim] -> chunk(2)
    let final_sst = wm.get_tensor("scale_shift_table")?;
    let temb_unsq = unsqueeze(temb, 1)?; // [B, 1, dim]
    let final_mod = Add::new(None, final_sst, temb_unsq)?;
    let shift_out = slice_axis(final_mod.clone(), 1, 0, 1)?;
    let scale_out = slice_axis(final_mod, 1, 1, 2)?;
    let hidden_states =
        adaln_modulate(hidden_states, shift_out, scale_out, config.dim, config.eps)?;
    let hidden_states = linear(&wm.prefix("proj_out"), hidden_states)?;

    // 6. Unpatchify
    let output = wan_unpatchify(hidden_states, &config, latent_frames, patch_h, patch_w)?;

    let output_tensors: Vec<(String, Arc<dyn Tensor>)> = vec![("out_sample".to_string(), output)];

    println!("Built Wan2.1 transformer graph, exporting...");
    let onnx_model = if let Some(origin) = origin_path {
        crate::onnx_graph::build_proto_with_origin_path(
            &input_tensors,
            &output_tensors,
            output_method,
            Some(origin),
        )?
    } else {
        crate::onnx_graph::build_proto(&input_tensors, &output_tensors, output_method)?
    };
    Ok(onnx_model.encode_to_vec())
}

// =============================================================================
// Wan2.1 VAE Decoder
// =============================================================================

/// WanCausalConv3d: temporal-causal Conv3d.
/// Pad temporal: (2*t_pad, 0) — double on past, zero on future.
/// Pad spatial: symmetric.
fn wan_causal_conv3d(
    wm: &impl WeightManager,
    input: Arc<dyn Tensor>,
    kernel_t: i64,
    kernel_s: i64,
    stride_t: i64,
    stride_s: i64,
) -> Result<Arc<dyn Tensor>, Error> {
    let weight = wm.get_tensor("weight")?;
    let bias = wm.get_tensor("bias").ok();

    // Causal temporal padding: repeat first frame
    let padded = if kernel_t > 1 {
        let time_pad = kernel_t - 1;
        let first_frame = slice_axis(input.clone(), 2, 0, 1)?;
        let mut parts = Vec::with_capacity(time_pad as usize + 1);
        for _ in 0..(time_pad * stride_t) {
            parts.push(first_frame.clone());
        }
        parts.push(input);
        Concat::new(None, parts, 2)? as Arc<dyn Tensor>
    } else {
        input
    };

    let spatial_pad = (kernel_s - 1) / 2;
    let conv = Conv::new(
        wm.get_prefix().map(|x| x.to_string()),
        padded,
        weight,
        bias,
        vec![kernel_t, kernel_s, kernel_s],
        vec![stride_t, stride_s, stride_s],
        vec![0, spatial_pad, spatial_pad, 0, spatial_pad, spatial_pad],
        vec![1, 1, 1],
        1,
    )?;
    Ok(conv)
}

/// RMSNorm for 5D tensors [B, C, T, H, W].
/// Wan VAE uses RMSNorm over channel dim (axis 1).
fn wan_rms_norm_5d(
    wm: &impl WeightManager,
    input: Arc<dyn Tensor>,
    channels: usize,
    eps: f32,
) -> Result<Arc<dyn Tensor>, Error> {
    // Transpose C to last so RMSNorm (axis=-1) operates on channels
    let x = Transpose::new(None, input, Some(vec![0, 2, 3, 4, 1])); // [B, T, H, W, C]
    let x = rms_norm(wm, x, Some(eps))?;
    let x = Transpose::new(None, x, Some(vec![0, 4, 1, 2, 3])); // [B, C, T, H, W]
    Ok(x as Arc<dyn Tensor>)
}

/// Wan VAE ResNet block with RMSNorm.
fn wan_vae_resnet(
    wm: &impl WeightManager,
    input: Arc<dyn Tensor>,
    in_channels: usize,
    out_channels: usize,
    eps: f32,
) -> Result<Arc<dyn Tensor>, Error> {
    // RMSNorm -> SiLU -> CausalConv3d(k=3)
    let h = wan_rms_norm_5d(&wm.prefix("norm1"), input.clone(), in_channels, eps)?;
    let h = silu(h)?;
    let h = wan_causal_conv3d(&wm.prefix("conv1"), h, 3, 3, 1, 1)?;

    // RMSNorm -> SiLU -> CausalConv3d(k=3)
    let h = wan_rms_norm_5d(&wm.prefix("norm2"), h, out_channels, eps)?;
    let h = silu(h)?;
    let h = wan_causal_conv3d(&wm.prefix("conv2"), h, 3, 3, 1, 1)?;

    // Skip connection
    let residual = if in_channels != out_channels {
        wan_causal_conv3d(&wm.prefix("shortcut"), input, 1, 1, 1, 1)?
    } else {
        input
    };

    Ok(Add::new(None, residual, h)?)
}

/// Spatial-only attention with known spatial dims.
fn wan_vae_spatial_attention(
    wm: &impl WeightManager,
    input: Arc<dyn Tensor>,
    channels: usize,
    t: usize,
    h: usize,
    w: usize,
    eps: f32,
) -> Result<Arc<dyn Tensor>, Error> {
    let normed = wan_rms_norm_5d(&wm.prefix("norm"), input.clone(), channels, eps)?;

    // [B, C, T, H, W] -> [B*T, H*W, C]
    let x = Transpose::new(None, normed, Some(vec![0, 2, 1, 3, 4])); // [B, T, C, H, W]
    let x = reshape(x, vec![-1, channels as i64, h as i64, w as i64])?; // [B*T, C, H, W]
    let x = reshape(x, vec![0, channels as i64, -1])?; // [B*T, C, H*W]
    let x = Transpose::new(None, x, Some(vec![0, 2, 1])); // [B*T, H*W, C]

    let q = linear(&wm.prefix("to_q"), x.clone())?;
    let k = linear(&wm.prefix("to_k"), x.clone())?;
    let v = linear(&wm.prefix("to_v"), x)?;

    let q = unsqueeze(q, 1)?;
    let k = unsqueeze(k, 1)?;
    let v = unsqueeze(v, 1)?;

    let scores = MatMul::new(None, q, Transpose::new(None, k, Some(vec![0, 1, 3, 2])))?;
    let scores = div_scalar(scores, (channels as f32).sqrt())?;
    let attn = Softmax::new(None, scores, Some(-1));
    let out = MatMul::new(None, attn, v)?;
    let out = reshape(out, vec![0, -1, channels as i64])?; // [B*T, H*W, C]
    let out = linear(&wm.prefix("to_out.0"), out)?;

    // [B*T, H*W, C] -> [B*T, C, H, W] -> [B, T, C, H, W] -> [B, C, T, H, W]
    let out = Transpose::new(None, out, Some(vec![0, 2, 1])); // [B*T, C, H*W]
    let out = reshape(out, vec![0, channels as i64, h as i64, w as i64])?;
    let out = reshape(out, vec![-1, t as i64, channels as i64, h as i64, w as i64])?;
    let out = Transpose::new(None, out, Some(vec![0, 2, 1, 3, 4])); // [B, C, T, H, W]

    Ok(Add::new(None, input, out)?)
}

/// Upsample: nearest-neighbor 3D resize + Conv2d per frame.
fn wan_vae_upsample(
    wm: &impl WeightManager,
    input: Arc<dyn Tensor>,
    cur_t: usize,
    cur_h: usize,
    cur_w: usize,
    target_t: usize,
    target_h: usize,
    target_w: usize,
) -> Result<Arc<dyn Tensor>, Error> {
    // 3D nearest-neighbor resize
    let scale_t = target_t as f32 / cur_t as f32;
    let scale_h = target_h as f32 / cur_h as f32;
    let scale_w = target_w as f32 / cur_w as f32;
    let scales = Constant::new(
        None,
        TensorData::new(
            vec![1.0f32, 1.0, scale_t, scale_h, scale_w].into(),
            Shape::from(&[5usize][..]),
        )?,
    );
    let output_dims = vec![
        input.shape()[0].clone(),
        input.shape()[1].clone(),
        Dimension::new(Some(target_t), None, None),
        Dimension::new(Some(target_h), None, None),
        Dimension::new(Some(target_w), None, None),
    ];
    let x = Resize::new_with_scales(
        None,
        input,
        scales,
        "nearest".to_string(),
        Shape::new(output_dims),
    )?;

    // Conv3d 3x3x3 after upsample (Wan uses CausalConv3d k=3)
    wan_causal_conv3d(&wm.prefix("conv"), x, 3, 3, 1, 1)
}

/// Build the Wan VAE decoder.
pub fn load_wan_vae_decoder(
    weight_manager: impl WeightManager,
    config: WanVaeConfig,
    output_method: WeightStorageStrategy,
) -> Result<Vec<u8>, anyhow::Error> {
    load_wan_vae_decoder_with_origin(weight_manager, config, output_method, None)
}

pub fn load_wan_vae_decoder_with_origin(
    weight_manager: impl WeightManager,
    config: WanVaeConfig,
    output_method: WeightStorageStrategy,
    origin_path: Option<&Path>,
) -> Result<Vec<u8>, anyhow::Error> {
    let model_dtype = weight_manager
        .get_tensor("decoder.conv_in.weight")
        .map(|t| t.dtype())
        .unwrap_or(DType::F32);
    let wm = CastingWeightManager::new(weight_manager, model_dtype);
    let dec = wm.prefix("decoder");
    let eps = config.eps;

    // Channel progression (reversed for decoder): [96, 192, 384, 384] -> [384, 384, 192, 96]
    let channels: Vec<usize> = config
        .dim_mult
        .iter()
        .map(|&m| config.base_dim * m)
        .collect();
    let rev_channels: Vec<usize> = channels.iter().copied().rev().collect();
    let num_stages = rev_channels.len();

    // Temporal decompression flags (reversed): [false, true, true] -> [true, true, false]
    let temporal_upsample: Vec<bool> = config.temporal_downsample.iter().copied().rev().collect();

    // Starting dims (for default 81 frames, 480x832):
    // After encoder: T=21, H=60, W=104
    let mut cur_t = 21usize;
    let mut cur_h = 60usize;
    let mut cur_w = 104usize;

    let batch_dim = Dimension::new(Some(1), Some("batch".to_string()), None);
    let latent_input = InputTensor::new(
        "latent".to_string(),
        model_dtype,
        Shape::new(vec![
            batch_dim.clone(),
            Dimension::new(Some(config.z_dim), None, None),
            Dimension::new(Some(cur_t), None, None),
            Dimension::new(Some(cur_h), None, None),
            Dimension::new(Some(cur_w), None, None),
        ]),
    );
    let input_tensors: Vec<Arc<dyn Tensor>> = vec![latent_input.clone()];

    // conv_in: CausalConv3d(z_dim -> last_channel, k=3)
    let last_ch = *rev_channels.first().unwrap(); // 384
    let mut x = wan_causal_conv3d(&dec.prefix("conv_in"), latent_input, 3, 3, 1, 1)?;

    // Mid block: ResBlock -> Attention -> ResBlock
    x = wan_vae_resnet(&dec.prefix("mid_block.0"), x, last_ch, last_ch, eps)?;
    x = wan_vae_spatial_attention(
        &dec.prefix("mid_block.1"),
        x,
        last_ch,
        cur_t,
        cur_h,
        cur_w,
        eps,
    )?;
    x = wan_vae_resnet(&dec.prefix("mid_block.2"), x, last_ch, last_ch, eps)?;

    println!("Building Wan2.1 VAE decoder: {} stages...", num_stages);

    // Decoder stages
    let mut current_ch = last_ch;
    for stage in 0..num_stages {
        let out_ch = rev_channels[stage];
        let has_upsample = stage < num_stages - 1;
        let do_temporal =
            has_upsample && stage < temporal_upsample.len() && temporal_upsample[stage];

        let num_res = config.num_res_blocks;
        for r in 0..num_res {
            let in_ch = if r == 0 { current_ch } else { out_ch };
            x = wan_vae_resnet(
                &dec.prefix(&format!("decoder_blocks.{stage}.{r}")),
                x,
                in_ch,
                out_ch,
                eps,
            )?;
        }
        current_ch = out_ch;

        if has_upsample {
            let next_h = cur_h * 2;
            let next_w = cur_w * 2;
            let next_t = if do_temporal {
                (cur_t - 1) * 2 + 1
            } else {
                cur_t
            };

            x = wan_vae_upsample(
                &dec.prefix(&format!("decoder_blocks.{stage}.{num_res}")),
                x,
                cur_t,
                cur_h,
                cur_w,
                next_t,
                next_h,
                next_w,
            )?;
            cur_t = next_t;
            cur_h = next_h;
            cur_w = next_w;
        }

        println!(
            "  Stage {}/{}: {}ch, [T={}, H={}, W={}]",
            stage + 1,
            num_stages,
            out_ch,
            cur_t,
            cur_h,
            cur_w
        );
    }

    // Final: RMSNorm -> SiLU -> CausalConv3d(channels -> out_channels, k=3)
    x = wan_rms_norm_5d(&dec.prefix("norm_out"), x, current_ch, eps)?;
    x = silu(x)?;
    let output = wan_causal_conv3d(&dec.prefix("conv_out"), x, 3, 3, 1, 1)?;

    let output_tensors: Vec<(String, Arc<dyn Tensor>)> = vec![("video_out".to_string(), output)];

    println!("Built Wan2.1 VAE decoder graph, exporting...");
    let onnx_model = if let Some(origin) = origin_path {
        crate::onnx_graph::build_proto_with_origin_path(
            &input_tensors,
            &output_tensors,
            output_method,
            Some(origin),
        )?
    } else {
        crate::onnx_graph::build_proto(&input_tensors, &output_tensors, output_method)?
    };
    Ok(onnx_model.encode_to_vec())
}
