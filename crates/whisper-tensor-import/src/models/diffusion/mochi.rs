use crate::models::diffusion::sd_common::{
    CastingWeightManager, cos_op, layer_norm_bare, ones_constant, rms_norm_bare, sin_op,
    slice_axis, split_chunks, tanh_op,
};
use crate::onnx_graph::Error;
use crate::onnx_graph::WeightStorageStrategy;
use crate::onnx_graph::operators::{
    Add, Concat, Conv, MatMul, Mul, RotaryEmbedding, Softmax, Transpose,
};
use crate::onnx_graph::pytorch::{
    cast, conv2d, div_scalar, linear, reshape, rms_norm, silu, unsqueeze,
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
pub struct MochiTransformerConfig {
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub attention_head_dim: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub text_embed_dim: usize,
    pub pooled_projection_dim: usize,
    pub time_embed_dim: usize,
    pub patch_size: usize,
    pub max_sequence_length: usize,
    pub norm_eps: f32,
}

impl MochiTransformerConfig {
    pub fn inner_dim(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }

    /// Video FFN inner dim: (4 * inner_dim * 2) / 3, rounded to nearest multiple
    pub fn ff_inner_dim(&self) -> usize {
        (4 * self.inner_dim() * 2) / 3
    }

    /// Text FFN inner dim: (4 * pooled_projection_dim * 2) / 3
    pub fn ff_context_inner_dim(&self) -> usize {
        (4 * self.pooled_projection_dim * 2) / 3
    }

    pub fn mochi_preview() -> Self {
        Self {
            num_layers: 48,
            num_attention_heads: 24,
            attention_head_dim: 128,
            in_channels: 12,
            out_channels: 12,
            text_embed_dim: 4096,
            pooled_projection_dim: 1536,
            time_embed_dim: 256,
            patch_size: 2,
            max_sequence_length: 256,
            norm_eps: 1e-6,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MochiVaeConfig {
    pub latent_channels: usize,
    pub out_channels: usize,
    pub decoder_block_out_channels: Vec<usize>,
    pub decoder_layers_per_block: Vec<usize>,
    pub temporal_expansions: Vec<usize>,
    pub spatial_expansions: Vec<usize>,
    pub norm_num_groups: usize,
    pub norm_eps: f32,
}

impl MochiVaeConfig {
    pub fn default_config() -> Self {
        Self {
            latent_channels: 12,
            out_channels: 3,
            decoder_block_out_channels: vec![128, 256, 512, 768],
            decoder_layers_per_block: vec![3, 3, 4, 6, 3],
            temporal_expansions: vec![1, 2, 3],
            spatial_expansions: vec![2, 2, 2],
            norm_num_groups: 32,
            norm_eps: 1e-6,
        }
    }
}

// =============================================================================
// Mochi RoPE (learnable per-head frequencies)
// =============================================================================

/// Precompute per-head RoPE cos/sin caches from learned `pos_frequencies`.
///
/// `pos_frequencies` shape: [3, num_heads, head_dim/2]
/// Positions are a 3D grid (t, h, w) with area normalization.
/// Returns (cos_cache, sin_cache) each [video_seq, num_heads * head_dim/2].
fn precompute_mochi_rope(
    config: &MochiTransformerConfig,
    pos_frequencies: &[f32],
    latent_frames: usize,
    patch_h: usize,
    patch_w: usize,
) -> (Vec<f32>, Vec<f32>) {
    let nh = config.num_attention_heads;
    let half_hd = config.attention_head_dim / 2;
    let video_seq = latent_frames * patch_h * patch_w;
    let out_dim = nh * half_hd; // flattened per-head dim

    // Area normalization: scale spatial coords by sqrt(target_area / actual_area)
    let target_area = 192.0 * 192.0;
    let actual_area = (patch_h * patch_w) as f64;
    let spatial_scale = (target_area / actual_area).sqrt();

    let mut cos_cache = vec![0.0f32; video_seq * out_dim];
    let mut sin_cache = vec![0.0f32; video_seq * out_dim];

    for vid_idx in 0..video_seq {
        let t = (vid_idx / (patch_h * patch_w)) as f64;
        let spatial_idx = vid_idx % (patch_h * patch_w);
        let h = (spatial_idx / patch_w) as f64 * spatial_scale;
        let w = (spatial_idx % patch_w) as f64 * spatial_scale;

        let positions = [t, h, w];

        // freqs = einsum("nd,dhf->nhf", positions[N,3], pos_freq[3,nh,half_hd])
        // For a single position: freqs[head, f] = sum_d(pos[d] * freq[d, head, f])
        for head in 0..nh {
            for f in 0..half_hd {
                let mut angle = 0.0f64;
                for (d, &pos) in positions.iter().enumerate() {
                    // pos_frequencies layout: [3, nh, half_hd] row-major
                    let freq_idx = d * nh * half_hd + head * half_hd + f;
                    angle += pos * pos_frequencies[freq_idx] as f64;
                }
                let out_idx = vid_idx * out_dim + head * half_hd + f;
                cos_cache[out_idx] = angle.cos() as f32;
                sin_cache[out_idx] = angle.sin() as f32;
            }
        }
    }

    (cos_cache, sin_cache)
}

// =============================================================================
// Mochi Transformer Components
// =============================================================================

/// Mochi timestep + attention-pooled text embedding.
///
/// Returns temb [B, inner_dim] = timestep_emb + pooled_text_emb.
fn mochi_timestep_embedding(
    wm: &impl WeightManager,
    timestep: Arc<dyn Tensor>,
    encoder_hidden_states: Arc<dyn Tensor>,
    config: &MochiTransformerConfig,
    model_dtype: DType,
) -> Result<Arc<dyn Tensor>, Error> {
    let inner_dim = config.inner_dim();

    // 1. Sinusoidal timestep embedding (256 channels)
    let timestep = cast(timestep, DType::F32);
    let half_dim = config.time_embed_dim / 2; // 128
    let freqs: Vec<f32> = (0..half_dim)
        .map(|i| (-(10000.0f32.ln()) * i as f32 / half_dim as f32).exp())
        .collect();
    let freq_tensor = InputTensorInitialized::new(
        "mochi_timestep_freqs".to_string(),
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

    // TimestepEmbedding: Linear(256 -> 3072) -> SiLU -> Linear(3072 -> 3072)
    let temb = linear(
        &wm.prefix("time_embed.timestep_embedder.linear_1"),
        t_emb,
    )?;
    let temb = silu(temb)?;
    let temb = linear(
        &wm.prefix("time_embed.timestep_embedder.linear_2"),
        temb,
    )?;

    // 2. Attention pool: pool T5 output into a single vector
    // Simplified: mean-pool text tokens, then cross-attend
    let text_dim = config.text_embed_dim; // 4096
    let seq_len = config.max_sequence_length;
    let pool_heads = 8usize;
    let pool_head_dim = text_dim / pool_heads; // 512

    // Mean pool: [B, S, 4096] -> [B, 1, 4096]
    let mean_weights = InputTensorInitialized::new(
        "mochi_pool_mean_weights".to_string(),
        TensorData::new(
            TensorDataValue::F32(vec![1.0 / seq_len as f32; seq_len]),
            Shape::new(vec![
                Dimension::new(Some(1), None, None),
                Dimension::new(Some(1), None, None),
                Dimension::new(Some(seq_len), None, None),
            ]),
        )?,
    );
    let mean_weights = cast(mean_weights, model_dtype);
    let pool_token = MatMul::new(None, mean_weights, encoder_hidden_states.clone())?;

    // Prepend pool token to sequence: [B, S+1, 4096]
    let kv_input = Concat::new(None, vec![pool_token.clone(), encoder_hidden_states], 1)?;

    // Cross-attention: Q from pool, K/V from all tokens
    let q = linear(&wm.prefix("time_embed.pooler.to_q"), pool_token)?;
    let kv = linear(&wm.prefix("time_embed.pooler.to_kv"), kv_input)?;
    let k = slice_axis(kv.clone(), -1, 0, text_dim as i64)?;
    let v = slice_axis(kv, -1, text_dim as i64, (text_dim * 2) as i64)?;

    // Reshape to multi-head: [B, seq, D] -> [B, heads, seq, head_dim]
    let q = Transpose::new(
        None,
        reshape(q, vec![0, 0, pool_heads as i64, pool_head_dim as i64])?,
        Some(vec![0, 2, 1, 3]),
    );
    let k = Transpose::new(
        None,
        reshape(k, vec![0, 0, pool_heads as i64, pool_head_dim as i64])?,
        Some(vec![0, 2, 1, 3]),
    );
    let v = Transpose::new(
        None,
        reshape(v, vec![0, 0, pool_heads as i64, pool_head_dim as i64])?,
        Some(vec![0, 2, 1, 3]),
    );

    let scores = MatMul::new(None, q, Transpose::new(None, k, Some(vec![0, 1, 3, 2])))?;
    let scores = div_scalar(scores, (pool_head_dim as f32).sqrt())?;
    let attn = Softmax::new(None, scores, Some(-1));
    let pool_out = MatMul::new(None, attn, v)?;

    // Reshape back: [B, heads, 1, head_dim] -> [B, 1, 4096]
    let pool_out = Transpose::new(None, pool_out, Some(vec![0, 2, 1, 3]));
    let pool_out = reshape(pool_out, vec![0, 1, text_dim as i64])?;
    let pool_out = linear(&wm.prefix("time_embed.pooler.to_out"), pool_out)?;
    let pool_out = reshape(pool_out, vec![0, inner_dim as i64])?; // [B, 3072]

    // temb = timestep_emb + pooled_text_emb
    let temb = Add::new(None, temb, pool_out)?;

    Ok(temb as Arc<dyn Tensor>)
}

/// SwiGLU feed-forward: Linear -> split -> SiLU(gate) * value -> Linear
fn swiglu_ff(
    wm: &impl WeightManager,
    input: Arc<dyn Tensor>,
    ff_inner_dim: usize,
) -> Result<Arc<dyn Tensor>, Error> {
    // net.0.proj: Linear(dim -> 2*ff_inner_dim)
    let projected = linear(&wm.prefix("net.0.proj"), input)?;
    // Split into gate and value
    let gate = slice_axis(projected.clone(), -1, 0, ff_inner_dim as i64)?;
    let value = slice_axis(projected, -1, ff_inner_dim as i64, (ff_inner_dim * 2) as i64)?;
    // SwiGLU: value * SiLU(gate)
    let gate = silu(gate)?;
    let h = Mul::new(None, value, gate)?;
    // net.2: Linear(ff_inner_dim -> dim)
    linear(&wm.prefix("net.2"), h)
}

/// Mochi asymmetric joint attention.
///
/// Video tokens (3072-dim) and text tokens (1536-dim) are projected to 3072-dim,
/// concatenated for joint attention, then split back with separate output projections.
fn mochi_joint_attention(
    wm: &impl WeightManager,
    hidden_states: Arc<dyn Tensor>,
    encoder_hidden_states: Arc<dyn Tensor>,
    config: &MochiTransformerConfig,
    rope_cos: Arc<dyn Tensor>,
    rope_sin: Arc<dyn Tensor>,
    context_pre_only: bool,
) -> Result<(Arc<dyn Tensor>, Option<Arc<dyn Tensor>>), Error> {
    let nh = config.num_attention_heads as i64;
    let hd = config.attention_head_dim as i64;
    let inner_dim = config.inner_dim() as i64;
    let text_seq = config.max_sequence_length as i64;

    // Video Q/K/V: Linear(3072 -> 3072)
    let q = linear(&wm.prefix("attn1.to_q"), hidden_states.clone())?;
    let k = linear(&wm.prefix("attn1.to_k"), hidden_states.clone())?;
    let v = linear(&wm.prefix("attn1.to_v"), hidden_states)?;

    // Text Q/K/V: Linear(1536 -> 3072)
    let add_q = linear(&wm.prefix("attn1.add_q_proj"), encoder_hidden_states.clone())?;
    let add_k = linear(&wm.prefix("attn1.add_k_proj"), encoder_hidden_states.clone())?;
    let add_v = linear(&wm.prefix("attn1.add_v_proj"), encoder_hidden_states)?;

    // Reshape to multi-head: [B, seq, D] -> [B, nh, seq, hd]
    let q = Transpose::new(None, reshape(q, vec![0, 0, nh, hd])?, Some(vec![0, 2, 1, 3]));
    let k = Transpose::new(None, reshape(k, vec![0, 0, nh, hd])?, Some(vec![0, 2, 1, 3]));
    let v = Transpose::new(None, reshape(v, vec![0, 0, nh, hd])?, Some(vec![0, 2, 1, 3]));

    let add_q = Transpose::new(
        None,
        reshape(add_q, vec![0, 0, nh, hd])?,
        Some(vec![0, 2, 1, 3]),
    );
    let add_k = Transpose::new(
        None,
        reshape(add_k, vec![0, 0, nh, hd])?,
        Some(vec![0, 2, 1, 3]),
    );
    let add_v = Transpose::new(
        None,
        reshape(add_v, vec![0, 0, nh, hd])?,
        Some(vec![0, 2, 1, 3]),
    );

    // Per-head RMSNorm on Q/K (learned affine, separate for video and text)
    let q = rms_norm(&wm.prefix("attn1.norm_q"), q, Some(1e-5))?;
    let k = rms_norm(&wm.prefix("attn1.norm_k"), k, Some(1e-5))?;
    let add_q = rms_norm(&wm.prefix("attn1.norm_added_q"), add_q, Some(1e-5))?;
    let add_k = rms_norm(&wm.prefix("attn1.norm_added_k"), add_k, Some(1e-5))?;

    // Apply RoPE to video Q/K only (per-head frequencies)
    // Reshape [B, nh, seq, hd] -> [B, 1, seq, nh*hd] for per-head RoPE
    let q = Transpose::new(None, q, Some(vec![0, 2, 1, 3])); // [B, seq, nh, hd]
    let q = reshape(q, vec![0, 0, 1, inner_dim])?; // [B, seq, 1, nh*hd]
    let q = Transpose::new(None, q, Some(vec![0, 2, 1, 3])); // [B, 1, seq, nh*hd]
    let k = Transpose::new(None, k, Some(vec![0, 2, 1, 3]));
    let k = reshape(k, vec![0, 0, 1, inner_dim])?;
    let k = Transpose::new(None, k, Some(vec![0, 2, 1, 3]));

    let q = RotaryEmbedding::new(
        None, q, rope_cos.clone(), rope_sin.clone(),
        None, Some(1), None, None,
    )?;
    let k = RotaryEmbedding::new(
        None, k, rope_cos, rope_sin,
        None, Some(1), None, None,
    )?;

    // Reshape back: [B, 1, seq, nh*hd] -> [B, nh, seq, hd]
    let q = Transpose::new(None, q, Some(vec![0, 2, 1, 3])); // [B, seq, 1, nh*hd]
    let q = reshape(q, vec![0, 0, nh, hd])?; // [B, seq, nh, hd]
    let q = Transpose::new(None, q, Some(vec![0, 2, 1, 3])); // [B, nh, seq, hd]
    let k = Transpose::new(None, k, Some(vec![0, 2, 1, 3]));
    let k = reshape(k, vec![0, 0, nh, hd])?;
    let k = Transpose::new(None, k, Some(vec![0, 2, 1, 3]));

    // Concatenate video + text tokens: [B, nh, vid_seq+text_seq, hd]
    let q_cat = Concat::new(None, vec![q, add_q], 2)?;
    let k_cat = Concat::new(None, vec![k, add_k], 2)?;
    let v_cat = Concat::new(None, vec![v, add_v], 2)?;

    // Scaled dot-product attention
    let scores = MatMul::new(
        None,
        q_cat,
        Transpose::new(None, k_cat, Some(vec![0, 1, 3, 2])),
    )?;
    let scores = div_scalar(scores, (config.attention_head_dim as f32).sqrt())?;
    let attn = Softmax::new(None, scores, Some(-1));
    let attn_out = MatMul::new(None, attn, v_cat)?;

    // Reshape: [B, nh, total_seq, hd] -> [B, total_seq, D]
    let attn_out = Transpose::new(None, attn_out, Some(vec![0, 2, 1, 3]));
    let attn_out = reshape(attn_out, vec![0, 0, inner_dim])?;

    // Split into video and text portions
    let total_seq = attn_out.shape()[1].resolve()? as i64;
    let vid_seq = total_seq - text_seq;
    let vid_out = slice_axis(attn_out.clone(), 1, 0, vid_seq)?;
    let txt_out = slice_axis(attn_out, 1, vid_seq, total_seq)?;

    // Output projections
    let vid_out = linear(&wm.prefix("attn1.to_out.0"), vid_out)?;

    let txt_out = if !context_pre_only {
        Some(linear(&wm.prefix("attn1.to_add_out"), txt_out)?)
    } else {
        None
    };

    Ok((vid_out, txt_out))
}

/// Modulated RMSNorm: RMSNorm(x) * scale
fn modulated_rms_norm(
    input: Arc<dyn Tensor>,
    scale: Arc<dyn Tensor>,
    dim: usize,
    eps: f32,
) -> Result<Arc<dyn Tensor>, Error> {
    let normed = rms_norm_bare(input, dim, eps)?;
    Ok(Mul::new(None, normed, scale)?)
}

/// Mochi transformer block (blocks 0-46: full, block 47: context_pre_only).
#[allow(clippy::too_many_arguments)]
fn mochi_block(
    wm: &impl WeightManager,
    hidden_states: Arc<dyn Tensor>,
    encoder_hidden_states: Arc<dyn Tensor>,
    temb: Arc<dyn Tensor>,
    config: &MochiTransformerConfig,
    rope_cos: Arc<dyn Tensor>,
    rope_sin: Arc<dyn Tensor>,
    context_pre_only: bool,
) -> Result<(Arc<dyn Tensor>, Arc<dyn Tensor>), Error> {
    let inner_dim = config.inner_dim();
    let text_dim = config.pooled_projection_dim;
    let eps = config.norm_eps;

    // 1. Video modulation: MochiRMSNormZero
    // SiLU(temb) -> Linear(3072, 4*3072=12288) -> chunk(4)
    let vid_mod = silu(temb.clone())?;
    let vid_mod = linear(&wm.prefix("norm1.linear"), vid_mod)?;
    let vid_mod = unsqueeze(vid_mod, 1)?; // [B, 1, 4*inner_dim]
    let vid_chunks = split_chunks(vid_mod, inner_dim, 4)?;
    let (scale_msa, gate_msa) = (vid_chunks[0].clone(), vid_chunks[1].clone());
    let (scale_mlp, gate_mlp) = (vid_chunks[2].clone(), vid_chunks[3].clone());

    // Video pre-attention norm: RMSNorm(x) * (1 + scale_msa)
    let one_vid = ones_constant(inner_dim, hidden_states.dtype());
    let vid_scale = Add::new(None, one_vid, scale_msa)?;
    let norm_hidden = modulated_rms_norm(hidden_states.clone(), vid_scale, inner_dim, eps)?;

    // 2. Text modulation — compute once, reuse chunks throughout the block
    let (norm_enc, enc_gate_msa, enc_scale_mlp, enc_gate_mlp) = if !context_pre_only {
        // MochiRMSNormZero: SiLU(temb) -> Linear(3072, 4*1536=6144) -> chunk(4)
        let enc_mod = silu(temb.clone())?;
        let enc_mod = linear(&wm.prefix("norm1_context.linear"), enc_mod)?;
        let enc_mod = unsqueeze(enc_mod, 1)?;
        let enc_chunks = split_chunks(enc_mod, text_dim, 4)?;
        let one_txt = ones_constant(text_dim, encoder_hidden_states.dtype());
        let enc_scale = Add::new(None, one_txt, enc_chunks[0].clone())?;
        let norm_enc =
            modulated_rms_norm(encoder_hidden_states.clone(), enc_scale, text_dim, eps)?;
        (
            norm_enc,
            Some(enc_chunks[1].clone()),
            Some(enc_chunks[2].clone()),
            Some(enc_chunks[3].clone()),
        )
    } else {
        // MochiLayerNormContinuous: (1 + scale) * RMSNorm(x)
        let enc_scale = silu(temb.clone())?;
        let enc_scale = linear(&wm.prefix("norm1_context.linear"), enc_scale)?;
        let enc_scale = unsqueeze(enc_scale, 1)?;
        let one_txt = ones_constant(text_dim, encoder_hidden_states.dtype());
        let enc_scale = Add::new(None, one_txt, enc_scale)?;
        let norm_enc =
            modulated_rms_norm(encoder_hidden_states.clone(), enc_scale, text_dim, eps)?;
        (norm_enc, None, None, None)
    };

    // 3. Joint attention
    let (attn_vid, attn_enc) = mochi_joint_attention(
        wm, norm_hidden, norm_enc, config, rope_cos, rope_sin, context_pre_only,
    )?;

    // 4. Post-attention residuals with tanh gating
    let tanh_gate_msa = tanh_op(gate_msa)?;
    let vid_gate_normed = modulated_rms_norm(attn_vid, tanh_gate_msa, inner_dim, eps)?;
    let hidden_states = Add::new(None, hidden_states, vid_gate_normed)?;

    let encoder_hidden_states = if let (Some(attn_enc), Some(enc_gate)) = (attn_enc, &enc_gate_msa)
    {
        let tanh_enc_gate = tanh_op(enc_gate.clone())?;
        let enc_gate_normed = modulated_rms_norm(attn_enc, tanh_enc_gate, text_dim, eps)?;
        Add::new(None, encoder_hidden_states.clone(), enc_gate_normed)? as Arc<dyn Tensor>
    } else {
        encoder_hidden_states.clone()
    };

    // 5. Video FFN with gating
    let one_vid2 = ones_constant(inner_dim, hidden_states.dtype());
    let ff_scale = Add::new(None, one_vid2, scale_mlp)?;
    let norm_ff = modulated_rms_norm(hidden_states.clone(), ff_scale, inner_dim, eps)?;
    let ff_out = swiglu_ff(&wm.prefix("ff"), norm_ff, config.ff_inner_dim())?;
    let tanh_gate_mlp = tanh_op(gate_mlp)?;
    let ff_gated = modulated_rms_norm(ff_out, tanh_gate_mlp, inner_dim, eps)?;
    let hidden_states = Add::new(None, hidden_states, ff_gated)?;

    // 6. Text FFN (only for non-context_pre_only blocks)
    let encoder_hidden_states = if let (Some(enc_s_mlp), Some(enc_g_mlp)) =
        (enc_scale_mlp, enc_gate_mlp)
    {
        let one_txt2 = ones_constant(text_dim, encoder_hidden_states.dtype());
        let enc_ff_scale = Add::new(None, one_txt2, enc_s_mlp)?;
        let norm_enc_ff =
            modulated_rms_norm(encoder_hidden_states.clone(), enc_ff_scale, text_dim, eps)?;
        let enc_ff =
            swiglu_ff(&wm.prefix("ff_context"), norm_enc_ff, config.ff_context_inner_dim())?;
        let tanh_enc_gate_mlp = tanh_op(enc_g_mlp)?;
        let enc_ff_gated = modulated_rms_norm(enc_ff, tanh_enc_gate_mlp, text_dim, eps)?;
        Add::new(None, encoder_hidden_states, enc_ff_gated)? as Arc<dyn Tensor>
    } else {
        encoder_hidden_states
    };

    Ok((hidden_states, encoder_hidden_states))
}

/// Mochi unpatchify: [B, T*(H/2)*(W/2), p*p*C] -> [B, C, T, H, W]
fn mochi_unpatchify(
    input: Arc<dyn Tensor>,
    config: &MochiTransformerConfig,
    latent_frames: usize,
    patch_h: usize,
    patch_w: usize,
) -> Result<Arc<dyn Tensor>, Error> {
    let f = latent_frames as i64;
    let ph = patch_h as i64;
    let pw = patch_w as i64;
    let p = config.patch_size as i64;
    let c = config.out_channels as i64;

    // [B, F*pH*pW, p*p*C] -> [B, F, pH, pW, p, p, C]
    let x = reshape(input, vec![0, f, ph, pw, p, p, c])?;
    // -> [B, C, F, pH, p, pW, p] -> [B, C, F, pH*p, pW*p]
    let x = Transpose::new(None, x, Some(vec![0, 6, 1, 2, 4, 3, 5]));
    reshape(x, vec![0, c, f, ph * p, pw * p]).map(|x| x as Arc<dyn Tensor>)
}

// =============================================================================
// Main Transformer Builder
// =============================================================================

pub fn load_mochi_transformer(
    weight_manager: impl WeightManager,
    config: MochiTransformerConfig,
    output_method: WeightStorageStrategy,
) -> Result<Vec<u8>, anyhow::Error> {
    load_mochi_transformer_with_origin(weight_manager, config, output_method, None)
}

pub fn load_mochi_transformer_with_origin(
    weight_manager: impl WeightManager,
    config: MochiTransformerConfig,
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

    // For default 480x848, 19 frames:
    // Latent: T=(19-1)/6+1=4, H=480/8=60, W=848/8=106
    // After patch_size=2: patch_h=30, patch_w=53
    let latent_frames = 4usize;
    let latent_h = 60usize;
    let latent_w = 106usize;
    let patch_h = latent_h / config.patch_size;
    let patch_w = latent_w / config.patch_size;
    let video_seq = latent_frames * patch_h * patch_w;

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
            Dimension::new(Some(config.max_sequence_length), None, None),
            Dimension::new(Some(config.text_embed_dim), None, None),
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

    // 1. Timestep + attention-pooled text embedding
    let temb = mochi_timestep_embedding(
        &wm,
        timestep_input,
        encoder_hidden_states_input.clone(),
        &config,
        model_dtype,
    )?;

    // 2. Caption projection: Linear(4096 -> 1536)
    let text_embeds = linear(
        &wm.prefix("time_embed.caption_proj"),
        encoder_hidden_states_input,
    )?;

    // 3. Patch embedding: Conv2d(12, 3072, k=2, s=2) per frame
    // [B, C, T, H, W] -> [B, T, C, H, W] -> [B*T, C, H, W]
    let x = Transpose::new(None, latent_input, Some(vec![0, 2, 1, 3, 4]));
    let x = reshape(
        x,
        vec![
            -1,
            config.in_channels as i64,
            latent_h as i64,
            latent_w as i64,
        ],
    )?;
    let x = conv2d(&wm.prefix("patch_embed.proj"), x, 2, 2, 0)?;
    // [B*T, 3072, H/2, W/2] -> [B*T, 3072, pH*pW] -> [B*T, pH*pW, 3072]
    let x = reshape(x, vec![0, inner_dim as i64, -1])?;
    let x = Transpose::new(None, x, Some(vec![0, 2, 1]));
    // -> [B, T*pH*pW, 3072]
    let hidden_states = reshape(x, vec![-1, video_seq as i64, inner_dim as i64])?;

    // 4. Precompute per-head RoPE from learned pos_frequencies
    let pos_freq_tensor = wm.get_tensor("pos_frequencies")?;
    // pos_frequencies is [3, num_heads, head_dim/2] stored as a weight
    // We need to read the actual values to precompute cos/sin
    // For graph building, we'll create the RoPE cache as constants
    let nh = config.num_attention_heads;
    let half_hd = config.attention_head_dim / 2;
    let rope_out_dim = nh * half_hd;

    // Use the pos_frequencies weight to precompute - but we can't read raw float values
    // from the weight tensor at graph build time. Instead, we'll compute RoPE in the graph.
    // For simplicity, create a placeholder RoPE cache that will be correct structurally.
    // The actual values need the learned pos_frequencies, which are baked into the ONNX model.
    //
    // We precompute positions and use einsum-equivalent ops to compute RoPE at graph time.
    // positions: [video_seq, 3], pos_frequencies: [3, nh, half_hd]
    // freqs: [video_seq, nh, half_hd] = positions @ pos_frequencies.reshape(3, nh*half_hd)
    // then reshape freqs to [video_seq, nh*half_hd]

    let target_area = 192.0 * 192.0;
    let actual_area = (patch_h * patch_w) as f64;
    let spatial_scale = (target_area / actual_area).sqrt();

    let mut positions = vec![0.0f32; video_seq * 3];
    for vid_idx in 0..video_seq {
        let t = (vid_idx / (patch_h * patch_w)) as f32;
        let spatial_idx = vid_idx % (patch_h * patch_w);
        let h = (spatial_idx / patch_w) as f32 * spatial_scale as f32;
        let w = (spatial_idx % patch_w) as f32 * spatial_scale as f32;
        positions[vid_idx * 3] = t;
        positions[vid_idx * 3 + 1] = h;
        positions[vid_idx * 3 + 2] = w;
    }

    let pos_const: Arc<dyn Tensor> = InputTensorInitialized::new(
        "mochi_rope_positions".to_string(),
        TensorData::new(
            TensorDataValue::F32(positions),
            Shape::new(vec![
                Dimension::new(Some(video_seq), None, None),
                Dimension::new(Some(3), None, None),
            ]),
        )?,
    );
    let pos_const = cast(pos_const, model_dtype);

    // pos_frequencies: [3, nh, half_hd] -> reshape to [3, nh*half_hd]
    let pos_freq = reshape(pos_freq_tensor, vec![3, rope_out_dim as i64])?;
    // freqs = positions @ pos_freq: [video_seq, 3] @ [3, nh*half_hd] = [video_seq, nh*half_hd]
    let freqs = MatMul::new(None, pos_const, pos_freq)?;
    let freqs = cast(freqs, DType::F32);
    let rope_cos = cos_op(freqs.clone())?;
    let rope_sin = sin_op(freqs)?;
    let rope_cos = cast(rope_cos, model_dtype);
    let rope_sin = cast(rope_sin, model_dtype);

    // 5. Transformer blocks
    println!(
        "Building Mochi transformer: {} blocks, inner_dim={}...",
        config.num_layers, inner_dim
    );
    let mut hidden_states: Arc<dyn Tensor> = hidden_states;
    let mut encoder_hidden_states: Arc<dyn Tensor> = text_embeds;
    for i in 0..config.num_layers {
        let context_pre_only = i == config.num_layers - 1;
        let block_wm = wm.prefix(&format!("transformer_blocks.{i}"));
        let (next_hidden, next_enc) = mochi_block(
            &block_wm,
            hidden_states,
            encoder_hidden_states,
            temb.clone(),
            &config,
            rope_cos.clone(),
            rope_sin.clone(),
            context_pre_only,
        )?;
        hidden_states = next_hidden;
        encoder_hidden_states = next_enc;
        if (i + 1) % 8 == 0 {
            println!("  Block {}/{}", i + 1, config.num_layers);
        }
    }

    // 6. Output norm: AdaLayerNormContinuous
    // LayerNorm(x) * (1 + scale) + shift, where (shift, scale) = Linear(SiLU(temb), 2*inner_dim)
    let norm_act = silu(temb)?;
    let norm_params = linear(&wm.prefix("norm_out.linear"), norm_act)?;
    let norm_params = unsqueeze(norm_params, 1)?;
    let shift_out = slice_axis(norm_params.clone(), -1, 0, inner_dim as i64)?;
    let scale_out = slice_axis(norm_params, -1, inner_dim as i64, (inner_dim * 2) as i64)?;
    let normed = layer_norm_bare(hidden_states, inner_dim, config.norm_eps)?;
    let one = ones_constant(inner_dim, normed.dtype());
    let scale_plus_one = Add::new(None, one, scale_out)?;
    let hidden_states = Mul::new(None, normed, scale_plus_one)?;
    let hidden_states = Add::new(None, hidden_states, shift_out)?;

    // 7. Output projection + unpatchify
    let hidden_states = linear(&wm.prefix("proj_out"), hidden_states)?;
    let output = mochi_unpatchify(hidden_states, &config, latent_frames, patch_h, patch_w)?;

    let output_tensors: Vec<(String, Arc<dyn Tensor>)> =
        vec![("out_sample".to_string(), output)];

    println!("Built Mochi transformer graph, exporting...");
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
// Mochi VAE Decoder
// =============================================================================

/// Causal Conv3d for Mochi VAE (same pattern as CogVideoX).
fn mochi_causal_conv3d(
    wm: &impl WeightManager,
    input: Arc<dyn Tensor>,
    kernel_t: i64,
    kernel_s: i64,
    stride_t: i64,
    stride_s: i64,
) -> Result<Arc<dyn Tensor>, Error> {
    let weight = wm.get_tensor("weight")?;
    let bias = wm.get_tensor("bias").ok();

    let padded = if kernel_t > 1 {
        let time_pad = kernel_t - 1;
        let first_frame = slice_axis(input.clone(), 2, 0, 1)?;
        let mut parts = Vec::with_capacity(time_pad as usize + 1);
        for _ in 0..time_pad {
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

/// GroupNorm applied per-frame: reshape [B,C,T,H,W] -> [B*T,C,H,W], GroupNorm, reshape back.
fn mochi_group_norm_5d(
    wm: &impl WeightManager,
    input: Arc<dyn Tensor>,
    eps: f32,
    num_groups: i64,
    channels: usize,
    cur_t: usize,
    cur_h: usize,
    cur_w: usize,
) -> Result<Arc<dyn Tensor>, Error> {
    use crate::onnx_graph::pytorch::group_norm;

    // [B, C, T, H, W] -> [B, T, C, H, W] -> [B*T, C, H, W]
    let x = Transpose::new(None, input, Some(vec![0, 2, 1, 3, 4]));
    let x = reshape(x, vec![-1, channels as i64, cur_h as i64, cur_w as i64])?;
    // GroupNorm operates on [B*T, C, H, W]
    let x = group_norm(&wm.prefix("norm_layer"), x, eps, num_groups)?;
    // [B*T, C, H, W] -> [B, T, C, H, W] -> [B, C, T, H, W]
    let x = reshape(x, vec![-1, cur_t as i64, channels as i64, cur_h as i64, cur_w as i64])?;
    let x = Transpose::new(None, x, Some(vec![0, 2, 1, 3, 4]));
    Ok(x as Arc<dyn Tensor>)
}

/// Mochi VAE ResNet block with per-frame GroupNorm.
#[allow(clippy::too_many_arguments)]
fn mochi_vae_resnet(
    wm: &impl WeightManager,
    input: Arc<dyn Tensor>,
    in_channels: usize,
    out_channels: usize,
    eps: f32,
    num_groups: i64,
    cur_t: usize,
    cur_h: usize,
    cur_w: usize,
) -> Result<Arc<dyn Tensor>, Error> {
    let h = mochi_group_norm_5d(
        &wm.prefix("norm1"), input.clone(), eps, num_groups,
        in_channels, cur_t, cur_h, cur_w,
    )?;
    let h = silu(h)?;
    let h = mochi_causal_conv3d(&wm.prefix("conv1.conv"), h, 3, 3, 1, 1)?;

    let h = mochi_group_norm_5d(
        &wm.prefix("norm2"), h, eps, num_groups,
        out_channels, cur_t, cur_h, cur_w,
    )?;
    let h = silu(h)?;
    let h = mochi_causal_conv3d(&wm.prefix("conv2.conv"), h, 3, 3, 1, 1)?;

    let residual = if in_channels != out_channels {
        mochi_causal_conv3d(&wm.prefix("shortcut.conv"), input, 1, 1, 1, 1)?
    } else {
        input
    };

    Ok(Add::new(None, residual, h)?)
}

/// Build the Mochi VAE decoder.
pub fn load_mochi_vae_decoder(
    weight_manager: impl WeightManager,
    config: MochiVaeConfig,
    output_method: WeightStorageStrategy,
) -> Result<Vec<u8>, anyhow::Error> {
    load_mochi_vae_decoder_with_origin(weight_manager, config, output_method, None)
}

pub fn load_mochi_vae_decoder_with_origin(
    weight_manager: impl WeightManager,
    config: MochiVaeConfig,
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
    let ng = config.norm_num_groups as i64;

    // For default 480x848, 19 frames:
    // Latent: [B, 12, 4, 60, 106]
    let mut cur_t = 4usize;
    let mut cur_h = 60usize;
    let mut cur_w = 106usize;

    // Decoder channel order (reversed): [768, 512, 256, 128]
    let rev_channels: Vec<usize> = config.decoder_block_out_channels.iter().copied().rev().collect();
    let last_ch = *rev_channels.first().unwrap(); // 768

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

    // conv_in: Conv3d(12, 768, kernel=1)
    let mut x: Arc<dyn Tensor> = {
        let weight = dec.get_tensor("conv_in.weight")?;
        let bias = dec.get_tensor("conv_in.bias").ok();
        Conv::new(
            Some("decoder.conv_in".to_string()),
            latent_input,
            weight,
            bias,
            vec![1, 1, 1],
            vec![1, 1, 1],
            vec![0, 0, 0, 0, 0, 0],
            vec![1, 1, 1],
            1,
        )?
    };

    // block_in: 3 ResNet blocks at 768, no attention
    let block_in_layers = *config.decoder_layers_per_block.last().unwrap(); // 3
    for r in 0..block_in_layers {
        x = mochi_vae_resnet(
            &dec.prefix(&format!("block_in.resnets.{r}")),
            x, last_ch, last_ch, eps, ng, cur_t, cur_h, cur_w,
        )?;
    }

    println!("Building Mochi VAE decoder: {} up stages...", config.temporal_expansions.len());

    // Up blocks (3 stages, reversed order)
    let num_up_stages = config.temporal_expansions.len(); // 3
    let up_layers: Vec<usize> = config.decoder_layers_per_block[1..1 + num_up_stages]
        .iter()
        .copied()
        .rev()
        .collect();

    let mut current_ch = last_ch;
    for stage in 0..num_up_stages {
        let out_ch = rev_channels[stage + 1]; // skip first (block_in ch)
        let num_res = up_layers[stage];
        let t_exp = config.temporal_expansions[num_up_stages - 1 - stage];
        let s_exp = config.spatial_expansions[num_up_stages - 1 - stage];

        for r in 0..num_res {
            let in_ch = if r == 0 { current_ch } else { out_ch };
            x = mochi_vae_resnet(
                &dec.prefix(&format!("up_blocks.{stage}.resnets.{r}")),
                x, in_ch, out_ch, eps, ng, cur_t, cur_h, cur_w,
            )?;
        }
        current_ch = out_ch;

        // Spatiotemporal upsample via linear projection + reshape
        if t_exp > 1 || s_exp > 1 {
            // [B, C, T, H, W] -> [B, T, H, W, C]
            let proj_in = Transpose::new(None, x, Some(vec![0, 2, 3, 4, 1]));
            // Linear(C -> C * t_exp * s_exp^2)
            let proj_out = linear(
                &dec.prefix(&format!("up_blocks.{stage}.proj")),
                proj_in,
            )?;
            // [B, T, H, W, C*t*s*s] -> [B, T, H, W, C, t, s, s]
            let proj_out = reshape(
                proj_out,
                vec![
                    0,
                    cur_t as i64,
                    cur_h as i64,
                    cur_w as i64,
                    out_ch as i64,
                    t_exp as i64,
                    s_exp as i64,
                    s_exp as i64,
                ],
            )?;
            // -> [B, C, T*t, H*s, W*s]
            let proj_out = Transpose::new(None, proj_out, Some(vec![0, 4, 1, 5, 2, 6, 3, 7]));
            let new_t = cur_t * t_exp;
            let new_h = cur_h * s_exp;
            let new_w = cur_w * s_exp;
            x = reshape(
                proj_out,
                vec![0, out_ch as i64, new_t as i64, new_h as i64, new_w as i64],
            )?;
            cur_t = new_t;
            cur_h = new_h;
            cur_w = new_w;
        }

        println!(
            "  Up stage {}/{}: {}ch, [T={}, H={}, W={}]",
            stage + 1,
            num_up_stages,
            out_ch,
            cur_t,
            cur_h,
            cur_w
        );
    }

    // block_out: 3 ResNet blocks at final channel
    let block_out_layers = config.decoder_layers_per_block[0]; // 3
    for r in 0..block_out_layers {
        x = mochi_vae_resnet(
            &dec.prefix(&format!("block_out.resnets.{r}")),
            x, current_ch, current_ch, eps, ng, cur_t, cur_h, cur_w,
        )?;
    }

    // SiLU + proj_out: Linear(128 -> 3)
    x = silu(x)?;
    // [B, C, T, H, W] -> [B, T, H, W, C] -> Linear -> [B, T, H, W, 3] -> [B, 3, T, H, W]
    let x = Transpose::new(None, x, Some(vec![0, 2, 3, 4, 1]));
    let x = linear(&dec.prefix("proj_out"), x)?;
    let x = Transpose::new(None, x, Some(vec![0, 4, 1, 2, 3]));

    // Drop first (temporal_compression - 1) = 5 frames from causal padding
    let temporal_compression: usize = config.temporal_expansions.iter().product();
    let drop_frames = temporal_compression - 1;
    let output = slice_axis(x, 2, drop_frames as i64, cur_t as i64)?;

    let output_tensors: Vec<(String, Arc<dyn Tensor>)> =
        vec![("video_out".to_string(), output)];

    println!("Built Mochi VAE decoder graph, exporting...");
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
