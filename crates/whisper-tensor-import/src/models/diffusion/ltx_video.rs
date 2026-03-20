use crate::models::diffusion::sd_common::{
    CastingWeightManager, adaln_modulate, cos_op, layer_norm_bare, ones_constant, sin_op,
    slice_axis, split_chunks,
};
use crate::onnx_graph::Error;
use crate::onnx_graph::WeightStorageStrategy;
use crate::onnx_graph::operators::{
    Add, Concat, Constant, MatMul, Mul, Resize, RotaryEmbedding, Softmax, Transpose,
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
pub struct LtxVideoTransformerConfig {
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub attention_head_dim: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub caption_channels: usize,
    pub cross_attention_dim: usize,
    pub activation_fn: &'static str,
    pub norm_eps: f32,
    pub attention_bias: bool,
    pub ffn_mult: usize,
}

impl LtxVideoTransformerConfig {
    pub fn inner_dim(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }

    pub fn ffn_inner_dim(&self) -> usize {
        self.inner_dim() * self.ffn_mult
    }

    pub fn ltxv_2b() -> Self {
        Self {
            num_layers: 28,
            num_attention_heads: 32,
            attention_head_dim: 64,
            in_channels: 128,
            out_channels: 128,
            caption_channels: 4096,
            cross_attention_dim: 2048,
            activation_fn: "gelu-approximate",
            norm_eps: 1e-6,
            attention_bias: true,
            ffn_mult: 4,
        }
    }

    pub fn ltxv_13b() -> Self {
        Self {
            num_layers: 48,
            num_attention_heads: 32,
            attention_head_dim: 128,
            in_channels: 128,
            out_channels: 128,
            caption_channels: 4096,
            cross_attention_dim: 4096,
            activation_fn: "gelu-approximate",
            norm_eps: 1e-6,
            attention_bias: true,
            ffn_mult: 4,
        }
    }
}

#[derive(Clone, Debug)]
pub struct LtxVideoVaeConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub latent_channels: usize,
    pub block_out_channels: Vec<usize>,
    pub layers_per_block: Vec<usize>,
    pub patch_size: usize,
    pub patch_size_t: usize,
    pub spatio_temporal_scaling: Vec<bool>,
    pub norm_eps: f32,
}

impl LtxVideoVaeConfig {
    pub fn default_config() -> Self {
        Self {
            in_channels: 3,
            out_channels: 3,
            latent_channels: 128,
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: vec![4, 3, 3, 3, 4],
            patch_size: 4,
            patch_size_t: 1,
            spatio_temporal_scaling: vec![true, true, true, false],
            norm_eps: 1e-6,
        }
    }

    /// Spatial compression ratio: patch_size * 2^(num spatial downsamples).
    pub fn spatial_compression(&self) -> usize {
        let spatial_stages = self.spatio_temporal_scaling.iter().filter(|&&s| s).count();
        self.patch_size * (1 << spatial_stages)
    }

    /// Temporal compression ratio: patch_size_t * 2^(num temporal downsamples).
    pub fn temporal_compression(&self) -> usize {
        let temporal_stages = self.spatio_temporal_scaling.iter().filter(|&&s| s).count();
        self.patch_size_t * (1 << temporal_stages)
    }
}

// =============================================================================
// 3D RoPE for LTX-Video
// =============================================================================

/// Precompute 3D factored RoPE cos/sin caches for LTX-Video.
///
/// LTX-Video splits head_dim into 3 equal parts for temporal, height, width.
/// Frequencies are logarithmically spaced in the range [1, theta].
/// Grid coordinates are fractional, normalized by base dimensions.
/// Returns (cos_cache, sin_cache) each [video_seq_len, head_dim].
fn precompute_ltxv_3d_rope(
    config: &LtxVideoTransformerConfig,
    latent_frames: usize,
    latent_h: usize,
    latent_w: usize,
) -> (Vec<f32>, Vec<f32>) {
    let head_dim = config.attention_head_dim;

    // LTX-Video splits head_dim into 3 equal parts
    let dim_per_axis = head_dim / 3;
    let remainder = head_dim - 3 * dim_per_axis;
    // temporal gets any remainder
    let t_dim = dim_per_axis + remainder;
    let h_dim = dim_per_axis;
    let w_dim = dim_per_axis;
    let axes_dim = [t_dim, h_dim, w_dim];

    let theta = 10000.0f64;
    let base_num_frames = 20.0f64;
    let base_height = 2048.0f64;
    let base_width = 2048.0f64;

    // Compute inverse frequencies for each axis
    // Frequencies are logarithmically spaced: theta^linspace(log_theta(1.0), log_theta(theta), n)
    // then scaled by pi/2
    let inv_freqs: Vec<Vec<f64>> = axes_dim
        .iter()
        .map(|&axis_dim| {
            let n = axis_dim / 2;
            (0..n)
                .map(|i| {
                    let t = if n > 1 {
                        i as f64 / (n - 1) as f64
                    } else {
                        0.0
                    };
                    // linspace from 0 to log_theta(theta) = 1.0
                    // theta^t gives frequencies from 1.0 to theta
                    // then multiply by pi/2
                    let freq = theta.powf(t) * std::f64::consts::FRAC_PI_2;
                    1.0 / freq
                })
                .collect()
        })
        .collect();

    let video_seq = latent_frames * latent_h * latent_w;
    // Output size: video_seq * head_dim (cos/sin are repeat_interleaved for paired dims)
    let mut cos_cache = vec![0.0f32; video_seq * head_dim];
    let mut sin_cache = vec![0.0f32; video_seq * head_dim];

    for vid_idx in 0..video_seq {
        let t_idx = vid_idx / (latent_h * latent_w);
        let spatial_idx = vid_idx % (latent_h * latent_w);
        let h_idx = spatial_idx / latent_w;
        let w_idx = spatial_idx % latent_w;

        // Fractional coordinates normalized by base dimensions
        let t_coord = t_idx as f64 / base_num_frames;
        let h_coord = h_idx as f64 / base_height;
        let w_coord = w_idx as f64 / base_width;

        let positions = [t_coord, h_coord, w_coord];
        let mut offset = 0;
        for (axis, &pos_val) in positions.iter().enumerate() {
            let half = axes_dim[axis] / 2;
            for i in 0..half {
                let angle = pos_val * inv_freqs[axis][i];
                let cos_val = angle.cos() as f32;
                let sin_val = angle.sin() as f32;
                // repeat_interleave(2): each frequency gets two consecutive slots
                cos_cache[vid_idx * head_dim + offset + 2 * i] = cos_val;
                cos_cache[vid_idx * head_dim + offset + 2 * i + 1] = cos_val;
                sin_cache[vid_idx * head_dim + offset + 2 * i] = sin_val;
                sin_cache[vid_idx * head_dim + offset + 2 * i + 1] = sin_val;
            }
            offset += axes_dim[axis];
        }
    }

    (cos_cache, sin_cache)
}

// =============================================================================
// LTX-Video Transformer Components
// =============================================================================

/// LTX-Video timestep + caption embedding.
///
/// Returns (temb_6x, embedded_timestep):
/// - temb_6x: [B, 1, 6 * inner_dim] — global modulation for all blocks
/// - embedded_timestep: [B, 1, inner_dim] — raw embedding for output norm
fn ltxv_timestep_embedding(
    wm: &impl WeightManager,
    timestep: Arc<dyn Tensor>,
    config: &LtxVideoTransformerConfig,
    model_dtype: DType,
) -> Result<(Arc<dyn Tensor>, Arc<dyn Tensor>), Error> {
    let inner_dim = config.inner_dim();

    // Sinusoidal embedding (256 channels, flip_sin_to_cos=true)
    let timestep = cast(timestep, DType::F32);
    let half_dim = 128; // 256 / 2
    let freqs: Vec<f32> = (0..half_dim)
        .map(|i| (-(10000.0f32.ln()) * i as f32 / half_dim as f32).exp())
        .collect();
    let freq_tensor = InputTensorInitialized::new(
        "ltxv_timestep_freqs".to_string(),
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
    // flip_sin_to_cos: cos first, then sin
    let t_emb = Concat::new(None, vec![cos_part, sin_part], -1)?;
    let t_emb = cast(t_emb, model_dtype);

    // TimestepEmbedding: Linear(256 -> inner_dim) -> SiLU -> Linear(inner_dim -> inner_dim)
    let temb = linear(
        &wm.prefix("adaln_single.emb.timestep_embedder.linear_1"),
        t_emb,
    )?;
    let temb = silu(temb)?;
    let temb = linear(
        &wm.prefix("adaln_single.emb.timestep_embedder.linear_2"),
        temb,
    )?;

    // Save embedded_timestep before 6x projection
    let embedded_timestep = unsqueeze(temb.clone(), 1)?; // [B, 1, inner_dim]

    // Project to 6*inner_dim: SiLU -> Linear(inner_dim -> 6*inner_dim)
    let temb = silu(temb)?;
    let temb_6x = linear(&wm.prefix("adaln_single.linear"), temb)?;
    let temb_6x = unsqueeze(temb_6x, 1)?; // [B, 1, 6*inner_dim]

    Ok((temb_6x, embedded_timestep))
}

/// LTX-Video caption projection: T5-XXL 4096-dim -> inner_dim.
fn ltxv_caption_projection(
    wm: &impl WeightManager,
    encoder_hidden_states: Arc<dyn Tensor>,
) -> Result<Arc<dyn Tensor>, Error> {
    // PixArtAlphaTextProjection: Linear -> GELU(tanh) -> Linear
    let x = linear(
        &wm.prefix("caption_projection.linear_1"),
        encoder_hidden_states,
    )?;
    let x = gelu_pytorch_tanh(x)?;
    linear(&wm.prefix("caption_projection.linear_2"), x)
}

/// LTX-Video self-attention with RoPE and QK RMSNorm across heads.
fn ltxv_self_attention(
    wm: &impl WeightManager,
    hidden_states: Arc<dyn Tensor>,
    config: &LtxVideoTransformerConfig,
    rope_cos: Arc<dyn Tensor>,
    rope_sin: Arc<dyn Tensor>,
) -> Result<Arc<dyn Tensor>, Error> {
    let nh = config.num_attention_heads as i64;
    let hd = config.attention_head_dim as i64;
    let inner_dim = config.inner_dim() as i64;

    let q = linear(&wm.prefix("attn1.to_q"), hidden_states.clone())?;
    let k = linear(&wm.prefix("attn1.to_k"), hidden_states.clone())?;
    let v = linear(&wm.prefix("attn1.to_v"), hidden_states)?;

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

    // QK RMSNorm across heads:
    // Reshape [B, nh, seq, hd] -> [B, seq, nh*hd], apply RMSNorm, reshape back
    let q_merged = Transpose::new(None, q, Some(vec![0, 2, 1, 3])); // [B, seq, nh, hd]
    let q_merged = reshape(q_merged, vec![0, 0, inner_dim])?; // [B, seq, D]
    let q_normed = rms_norm(&wm.prefix("attn1.norm_q"), q_merged, Some(config.norm_eps))?;
    let q = Transpose::new(
        None,
        reshape(q_normed, vec![0, 0, nh, hd])?,
        Some(vec![0, 2, 1, 3]),
    ); // [B, nh, seq, hd]

    let k_merged = Transpose::new(None, k, Some(vec![0, 2, 1, 3]));
    let k_merged = reshape(k_merged, vec![0, 0, inner_dim])?;
    let k_normed = rms_norm(&wm.prefix("attn1.norm_k"), k_merged, Some(config.norm_eps))?;
    let k = Transpose::new(
        None,
        reshape(k_normed, vec![0, 0, nh, hd])?,
        Some(vec![0, 2, 1, 3]),
    );

    // Apply 3D RoPE (interleaved=1 for repeat_interleave-style cos/sin)
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
    let scores = div_scalar(scores, (config.attention_head_dim as f32).sqrt())?;
    let attn = Softmax::new(None, scores, Some(-1));
    let attn_out = MatMul::new(None, attn, v)?;

    // Reshape: [B, nh, seq, hd] -> [B, seq, D]
    let attn_out = Transpose::new(None, attn_out, Some(vec![0, 2, 1, 3]));
    let attn_out = reshape(attn_out, vec![0, 0, inner_dim])?;

    // Output projection
    linear(&wm.prefix("attn1.to_out.0"), attn_out)
}

/// LTX-Video cross-attention (text → video). No RoPE, no QK norm.
fn ltxv_cross_attention(
    wm: &impl WeightManager,
    hidden_states: Arc<dyn Tensor>,
    encoder_hidden_states: Arc<dyn Tensor>,
    config: &LtxVideoTransformerConfig,
) -> Result<Arc<dyn Tensor>, Error> {
    let nh = config.num_attention_heads as i64;
    let hd = config.attention_head_dim as i64;
    let inner_dim = config.inner_dim() as i64;

    let q = linear(&wm.prefix("attn2.to_q"), hidden_states)?;
    let k = linear(&wm.prefix("attn2.to_k"), encoder_hidden_states.clone())?;
    let v = linear(&wm.prefix("attn2.to_v"), encoder_hidden_states)?;

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

    // No QK norm for cross-attention, no RoPE

    let scores = MatMul::new(None, q, Transpose::new(None, k, Some(vec![0, 1, 3, 2])))?;
    let scores = div_scalar(scores, (config.attention_head_dim as f32).sqrt())?;
    let attn = Softmax::new(None, scores, Some(-1));
    let attn_out = MatMul::new(None, attn, v)?;

    let attn_out = Transpose::new(None, attn_out, Some(vec![0, 2, 1, 3]));
    let attn_out = reshape(attn_out, vec![0, 0, inner_dim])?;

    linear(&wm.prefix("attn2.to_out.0"), attn_out)
}

/// LTX-Video transformer block.
///
/// 1. AdaLN modulation from scale_shift_table + temb -> 6 values
/// 2. Self-attention with RoPE + gating
/// 3. Cross-attention (ungated residual)
/// 4. Feed-forward with gating
fn ltxv_block(
    wm: &impl WeightManager,
    hidden_states: Arc<dyn Tensor>,
    encoder_hidden_states: Arc<dyn Tensor>,
    temb_6x: Arc<dyn Tensor>,
    config: &LtxVideoTransformerConfig,
    rope_cos: Arc<dyn Tensor>,
    rope_sin: Arc<dyn Tensor>,
) -> Result<Arc<dyn Tensor>, Error> {
    let inner_dim = config.inner_dim();
    let eps = config.norm_eps;

    // Modulation: scale_shift_table [6, inner_dim] -> [1, 1, 6*inner_dim]
    // + temb_6x [B, 1, 6*inner_dim] -> [B, 1, 6*inner_dim]
    // -> split into 6 chunks of inner_dim
    let sst = wm.get_tensor("scale_shift_table")?;
    // Reshape [6, inner_dim] -> [1, 1, 6*inner_dim]
    let sst = reshape(sst, vec![1, 1, (6 * inner_dim) as i64])?;
    let modulation = Add::new(None, sst, temb_6x)?;
    let chunks = split_chunks(modulation, inner_dim, 6)?;
    let (shift_msa, scale_msa, gate_msa) =
        (chunks[0].clone(), chunks[1].clone(), chunks[2].clone());
    let (shift_mlp, scale_mlp, gate_mlp) =
        (chunks[3].clone(), chunks[4].clone(), chunks[5].clone());

    // 1. Self-attention with AdaLN
    let normed = adaln_modulate(hidden_states.clone(), shift_msa, scale_msa, inner_dim, eps)?;
    let attn_out = ltxv_self_attention(wm, normed, config, rope_cos, rope_sin)?;
    let hidden_states = Add::new(None, hidden_states, Mul::new(None, gate_msa, attn_out)?)?;

    // 2. Cross-attention (ungated residual)
    let cross_out = ltxv_cross_attention(wm, hidden_states.clone(), encoder_hidden_states, config)?;
    let hidden_states = Add::new(None, hidden_states, cross_out)?;

    // 3. Feed-forward with AdaLN + gating
    let normed_ff = adaln_modulate(hidden_states.clone(), shift_mlp, scale_mlp, inner_dim, eps)?;
    let ff = linear(&wm.prefix("ff.net.0.proj"), normed_ff)?;
    let ff = gelu_pytorch_tanh(ff)?;
    let ff = linear(&wm.prefix("ff.net.2"), ff)?;
    let hidden_states = Add::new(None, hidden_states, Mul::new(None, gate_mlp, ff)?)?;

    Ok(hidden_states as Arc<dyn Tensor>)
}

// =============================================================================
// Main Transformer Builder
// =============================================================================

pub fn load_ltxv_transformer(
    weight_manager: impl WeightManager,
    config: LtxVideoTransformerConfig,
    output_method: WeightStorageStrategy,
) -> Result<Vec<u8>, anyhow::Error> {
    load_ltxv_transformer_with_origin(weight_manager, config, output_method, None)
}

pub fn load_ltxv_transformer_with_origin(
    weight_manager: impl WeightManager,
    config: LtxVideoTransformerConfig,
    output_method: WeightStorageStrategy,
    origin_path: Option<&Path>,
) -> Result<Vec<u8>, anyhow::Error> {
    let model_dtype = weight_manager
        .get_tensor("transformer_blocks.0.attn1.to_q.weight")
        .map(|t| t.dtype())
        .unwrap_or(DType::BF16);
    let wm = CastingWeightManager::new(weight_manager, model_dtype);

    let inner_dim = config.inner_dim();
    let batch_dim = Dimension::new(Some(1), Some("batch".to_string()), None);

    // For default 704x512, 161 frames:
    // Spatial: 704/32 = 22, 512/32 = 16
    // Temporal: (161-1)/8 + 1 = 21
    let latent_frames = 21usize;
    let latent_h = 16usize;
    let latent_w = 22usize;
    let video_seq = latent_frames * latent_h * latent_w; // 7392

    let latent_input = InputTensor::new(
        "hidden_states".to_string(),
        model_dtype,
        Shape::new(vec![
            batch_dim.clone(),
            Dimension::new(Some(config.in_channels), None, None),
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
            Dimension::new(Some(128), None, None), // T5 seq_len
            Dimension::new(Some(config.caption_channels), None, None),
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

    // 1. Timestep embedding (produces global modulation + embedded_timestep for output)
    let (temb_6x, embedded_timestep) =
        ltxv_timestep_embedding(&wm, timestep_input, &config, model_dtype)?;

    // 2. Caption projection: T5 4096-dim -> inner_dim
    let text_embeds = ltxv_caption_projection(&wm, encoder_hidden_states_input)?;

    // 3. Project latent: Linear(in_channels -> inner_dim)
    // No patchification in transformer (patch_size=1)
    // [B, C, T, H, W] -> [B, C, T*H*W] -> [B, T*H*W, C] -> Linear -> [B, T*H*W, inner_dim]
    let hidden_states = reshape(latent_input, vec![0, config.in_channels as i64, -1])?;
    let hidden_states = Transpose::new(None, hidden_states, Some(vec![0, 2, 1]));
    let hidden_states = linear(&wm.prefix("proj_in"), hidden_states)?;

    // 4. Precompute 3D RoPE
    let (cos_vals, sin_vals) = precompute_ltxv_3d_rope(&config, latent_frames, latent_h, latent_w);
    let rope_shape = Shape::new(vec![
        Dimension::new(Some(video_seq), None, None),
        Dimension::new(Some(config.attention_head_dim), None, None),
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

    // 5. Transformer blocks
    println!(
        "Building LTX-Video transformer: {} blocks, inner_dim={}...",
        config.num_layers, inner_dim
    );
    let mut hidden_states: Arc<dyn Tensor> = hidden_states;
    for i in 0..config.num_layers {
        let block_wm = wm.prefix(&format!("transformer_blocks.{i}"));
        hidden_states = ltxv_block(
            &block_wm,
            hidden_states,
            text_embeds.clone(),
            temb_6x.clone(),
            &config,
            rope_cos.clone(),
            rope_sin.clone(),
        )?;
        if (i + 1) % 7 == 0 {
            println!("  Block {}/{}", i + 1, config.num_layers);
        }
    }

    // 6. Final output: LayerNorm + AdaLN modulation from scale_shift_table + proj_out
    // scale_shift_table [2, inner_dim] + embedded_timestep [B, 1, inner_dim]
    let final_sst = wm.get_tensor("scale_shift_table")?;
    // Reshape [2, inner_dim] -> [1, 2, inner_dim]
    let final_sst = reshape(final_sst, vec![1, 2, inner_dim as i64])?;
    // embedded_timestep is [B, 1, inner_dim] -> [B, 2, inner_dim] by expanding
    // Actually: add sst to embedded_timestep repeated along dim 1
    // The diffusers code does: shift, scale = (self.scale_shift_table[None] + embedded_timestep).chunk(2, dim=1)
    // So sst is [1, 2, inner_dim], embedded_timestep is [B, 1, inner_dim], they broadcast to [B, 2, inner_dim]
    let final_mod = Add::new(None, final_sst, embedded_timestep)?;
    let shift_out = slice_axis(final_mod.clone(), 1, 0, 1)?;
    let scale_out = slice_axis(final_mod, 1, 1, 2)?;
    let hidden_states = adaln_modulate(
        hidden_states,
        shift_out,
        scale_out,
        inner_dim,
        config.norm_eps,
    )?;
    let hidden_states = linear(&wm.prefix("proj_out"), hidden_states)?;

    // 7. Reshape back to [B, out_channels, T, H, W]
    let output = Transpose::new(None, hidden_states, Some(vec![0, 2, 1])); // [B, C, seq]
    let output = reshape(
        output,
        vec![
            0,
            config.out_channels as i64,
            latent_frames as i64,
            latent_h as i64,
            latent_w as i64,
        ],
    )?;

    let output_tensors: Vec<(String, Arc<dyn Tensor>)> = vec![("out_sample".to_string(), output)];

    println!("Built LTX-Video transformer graph, exporting...");
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
// LTX-Video VAE Decoder
// =============================================================================

/// RMSNorm for 5D tensors [B, C, T, H, W].
/// Normalizes over channel dim (axis 1).
fn ltxv_rms_norm_5d(
    wm: &impl WeightManager,
    input: Arc<dyn Tensor>,
    eps: f32,
) -> Result<Arc<dyn Tensor>, Error> {
    let x = Transpose::new(None, input, Some(vec![0, 2, 3, 4, 1])); // [B, T, H, W, C]
    let x = rms_norm(wm, x, Some(eps))?;
    let x = Transpose::new(None, x, Some(vec![0, 4, 1, 2, 3])); // [B, C, T, H, W]
    Ok(x as Arc<dyn Tensor>)
}

/// Standard (non-causal) Conv3d for LTX-Video decoder.
fn ltxv_conv3d(
    wm: &impl WeightManager,
    input: Arc<dyn Tensor>,
    kernel_t: i64,
    kernel_s: i64,
    stride_t: i64,
    stride_s: i64,
) -> Result<Arc<dyn Tensor>, Error> {
    use crate::onnx_graph::operators::Conv;

    let weight = wm.get_tensor("weight")?;
    let bias = wm.get_tensor("bias").ok();

    let temporal_pad = (kernel_t - 1) / 2;
    let spatial_pad = (kernel_s - 1) / 2;
    let conv = Conv::new(
        wm.get_prefix().map(|x| x.to_string()),
        input,
        weight,
        bias,
        vec![kernel_t, kernel_s, kernel_s],
        vec![stride_t, stride_s, stride_s],
        vec![
            temporal_pad,
            spatial_pad,
            spatial_pad,
            temporal_pad,
            spatial_pad,
            spatial_pad,
        ],
        vec![1, 1, 1],
        1,
    )?;
    Ok(conv)
}

/// LTX-Video VAE ResNet block with RMSNorm.
fn ltxv_vae_resnet(
    wm: &impl WeightManager,
    input: Arc<dyn Tensor>,
    in_channels: usize,
    out_channels: usize,
    eps: f32,
) -> Result<Arc<dyn Tensor>, Error> {
    // RMSNorm -> SiLU -> Conv3d(k=3)
    let h = ltxv_rms_norm_5d(&wm.prefix("norm1"), input.clone(), eps)?;
    let h = silu(h)?;
    let h = ltxv_conv3d(&wm.prefix("conv1"), h, 3, 3, 1, 1)?;

    // RMSNorm -> SiLU -> Conv3d(k=3)
    let h = ltxv_rms_norm_5d(&wm.prefix("norm2"), h, eps)?;
    let h = silu(h)?;
    let h = ltxv_conv3d(&wm.prefix("conv2"), h, 3, 3, 1, 1)?;

    // Skip connection
    let residual = if in_channels != out_channels {
        ltxv_conv3d(&wm.prefix("conv_shortcut"), input, 1, 1, 1, 1)?
    } else {
        input
    };

    Ok(Add::new(None, residual, h)?)
}

/// Spatiotemporal 2x upsample: nearest-neighbor 3D resize + Conv3d.
fn ltxv_vae_upsample(
    wm: &impl WeightManager,
    input: Arc<dyn Tensor>,
    cur_t: usize,
    cur_h: usize,
    cur_w: usize,
) -> Result<Arc<dyn Tensor>, Error> {
    let target_t = cur_t * 2;
    let target_h = cur_h * 2;
    let target_w = cur_w * 2;

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

    ltxv_conv3d(&wm.prefix("conv"), x, 3, 3, 1, 1)
}

/// Build the LTX-Video VAE decoder.
///
/// Decoder is non-causal (uses symmetric Conv3d padding).
/// Structure: conv_in -> mid_block (ResNets) -> up_blocks (ResNets + upsample) -> conv_out -> depatchify.
pub fn load_ltxv_vae_decoder(
    weight_manager: impl WeightManager,
    config: LtxVideoVaeConfig,
    output_method: WeightStorageStrategy,
) -> Result<Vec<u8>, anyhow::Error> {
    load_ltxv_vae_decoder_with_origin(weight_manager, config, output_method, None)
}

pub fn load_ltxv_vae_decoder_with_origin(
    weight_manager: impl WeightManager,
    config: LtxVideoVaeConfig,
    output_method: WeightStorageStrategy,
    origin_path: Option<&Path>,
) -> Result<Vec<u8>, anyhow::Error> {
    let model_dtype = weight_manager
        .get_tensor("decoder.conv_in.weight")
        .map(|t| t.dtype())
        .unwrap_or(DType::F32);
    let wm = CastingWeightManager::new(weight_manager, model_dtype);
    let dec = wm.prefix("decoder");
    let eps = config.norm_eps;

    // Channel progression reversed for decoder:
    // block_out_channels = [128, 256, 512, 512]
    // Decoder goes: 512 -> 512 -> 256 -> 128
    let rev_channels: Vec<usize> = config.block_out_channels.iter().copied().rev().collect();
    let num_stages = rev_channels.len(); // 4

    // spatio_temporal_scaling reversed for decoder: [true, true, true, false] -> [false, true, true, true]
    let upsample_flags: Vec<bool> = config
        .spatio_temporal_scaling
        .iter()
        .copied()
        .rev()
        .collect();

    // layers_per_block = [4, 3, 3, 3, 4]
    // mid = layers_per_block[last] = 4
    // up blocks: layers_per_block[0..4] reversed = [3, 3, 3, 4] (reversed to match decoder order)
    let mid_layers = *config.layers_per_block.last().unwrap();
    let up_layers: Vec<usize> = config.layers_per_block[..config.layers_per_block.len() - 1]
        .iter()
        .copied()
        .rev()
        .collect();

    // Starting dims (for default 704x512, 161 frames):
    // After encoder: T=21, H=16, W=22
    let mut cur_t = 21usize;
    let mut cur_h = 16usize;
    let mut cur_w = 22usize;

    let batch_dim = Dimension::new(Some(1), Some("batch".to_string()), None);
    let latent_input = InputTensor::new(
        "latent".to_string(),
        model_dtype,
        Shape::new(vec![
            batch_dim.clone(),
            Dimension::new(Some(config.latent_channels), None, None),
            Dimension::new(Some(cur_t), None, None),
            Dimension::new(Some(cur_h), None, None),
            Dimension::new(Some(cur_w), None, None),
        ]),
    );
    let input_tensors: Vec<Arc<dyn Tensor>> = vec![latent_input.clone()];

    // conv_in: Conv3d(latent_channels -> last_channel, k=3)
    let last_ch = *rev_channels.first().unwrap(); // 512
    let mut x = ltxv_conv3d(&dec.prefix("conv_in"), latent_input, 3, 3, 1, 1)?;

    // Mid block: multiple ResNet blocks at last_ch
    for r in 0..mid_layers {
        x = ltxv_vae_resnet(
            &dec.prefix(&format!("mid_block.resnets.{r}")),
            x,
            last_ch,
            last_ch,
            eps,
        )?;
    }

    println!(
        "Building LTX-Video VAE decoder: {} up stages...",
        num_stages
    );

    // Up blocks
    let mut current_ch = last_ch;
    for stage in 0..num_stages {
        let out_ch = rev_channels[stage];
        let num_res = up_layers[stage];
        let do_upsample = upsample_flags[stage];

        for r in 0..num_res {
            let in_ch = if r == 0 { current_ch } else { out_ch };
            x = ltxv_vae_resnet(
                &dec.prefix(&format!("up_blocks.{stage}.resnets.{r}")),
                x,
                in_ch,
                out_ch,
                eps,
            )?;
        }
        current_ch = out_ch;

        if do_upsample {
            x = ltxv_vae_upsample(
                &dec.prefix(&format!("up_blocks.{stage}.upsamplers.0")),
                x,
                cur_t,
                cur_h,
                cur_w,
            )?;
            cur_t *= 2;
            cur_h *= 2;
            cur_w *= 2;
        }

        println!(
            "  Up stage {}/{}: {}ch, [T={}, H={}, W={}]",
            stage + 1,
            num_stages,
            out_ch,
            cur_t,
            cur_h,
            cur_w
        );
    }

    // Final: RMSNorm -> SiLU -> Conv3d -> depatchify
    x = ltxv_rms_norm_5d(&dec.prefix("norm_out"), x, eps)?;
    x = silu(x)?;
    // conv_out produces patch_size^2 * out_channels = 48 channels
    x = ltxv_conv3d(&dec.prefix("conv_out"), x, 3, 3, 1, 1)?;

    // Depatchify: [B, patch_channels, T, H/ps, W/ps] -> [B, out_channels, T, H, W]
    // patch_channels = 3 * 4 * 4 = 48
    let ps = config.patch_size as i64;
    let c = config.out_channels as i64;
    // [B, 48, T, H', W'] -> [B, 3, 4, 4, T, H', W'] -> [B, 3, T, H'*4, W'*4]
    let output = reshape(
        x,
        vec![0, c, ps, ps, cur_t as i64, cur_h as i64, cur_w as i64],
    )?;
    // -> [B, C, T, H', ps, W', ps]
    let output = Transpose::new(None, output, Some(vec![0, 1, 4, 2, 5, 3, 6]));
    let output = reshape(
        output,
        vec![0, c, cur_t as i64, cur_h as i64 * ps, cur_w as i64 * ps],
    )?;

    let output_tensors: Vec<(String, Arc<dyn Tensor>)> = vec![("video_out".to_string(), output)];

    println!("Built LTX-Video VAE decoder graph, exporting...");
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
