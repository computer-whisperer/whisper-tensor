use crate::models::diffusion::sd_common::{
    CastingWeightManager, adaln_modulate, cos_op, layer_norm_bare, ones_constant, sin_op,
    slice_axis, split_chunks,
};
use crate::onnx_graph::Error;
use crate::onnx_graph::WeightStorageStrategy;
use crate::onnx_graph::operators::{
    Add, Concat, Conv, Gather, MatMul, Mul, RotaryEmbedding, Softmax, Transpose,
};
use crate::onnx_graph::pytorch::{
    cast, div_scalar, gelu_pytorch_tanh, layer_norm, linear, reshape, rms_norm, silu, unsqueeze,
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
pub struct HunyuanVideoTransformerConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub num_attention_heads: usize,
    pub attention_head_dim: usize,
    pub num_layers: usize,        // dual-stream blocks
    pub num_single_layers: usize, // single-stream blocks
    pub num_refiner_layers: usize,
    pub mlp_ratio: f32,
    pub patch_size: usize,
    pub patch_size_t: usize,
    pub text_embed_dim: usize,
    pub pooled_projection_dim: usize,
    pub guidance_embeds: bool,
    pub rope_theta: f64,
    pub rope_axes_dim: [usize; 3],
    pub norm_eps: f32,
}

impl HunyuanVideoTransformerConfig {
    pub fn inner_dim(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }

    pub fn mlp_dim(&self) -> usize {
        (self.inner_dim() as f32 * self.mlp_ratio) as usize
    }

    pub fn default_config() -> Self {
        Self {
            in_channels: 16,
            out_channels: 16,
            num_attention_heads: 24,
            attention_head_dim: 128,
            num_layers: 20,
            num_single_layers: 40,
            num_refiner_layers: 2,
            mlp_ratio: 4.0,
            patch_size: 2,
            patch_size_t: 1,
            text_embed_dim: 4096,
            pooled_projection_dim: 768,
            guidance_embeds: true,
            rope_theta: 256.0,
            rope_axes_dim: [16, 56, 56],
            norm_eps: 1e-6,
        }
    }
}

#[derive(Clone, Debug)]
pub struct HunyuanVideoVaeConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub latent_channels: usize,
    pub block_out_channels: Vec<usize>,
    pub layers_per_block: usize,
    pub norm_num_groups: usize,
    pub temporal_compression_ratio: usize,
    pub spatial_compression_ratio: usize,
    pub norm_eps: f32,
    pub scaling_factor: f32,
}

impl HunyuanVideoVaeConfig {
    pub fn default_config() -> Self {
        Self {
            in_channels: 3,
            out_channels: 3,
            latent_channels: 16,
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            norm_num_groups: 32,
            temporal_compression_ratio: 4,
            spatial_compression_ratio: 8,
            norm_eps: 1e-6,
            scaling_factor: 0.476986,
        }
    }
}

// =============================================================================
// 3D RoPE for HunyuanVideo (theta=256, axes_dim=[16,56,56])
// =============================================================================

/// Precompute 3D factored RoPE cos/sin caches.
///
/// axes_dim = [16, 56, 56] splits head_dim=128. Each axis gets its own
/// set of frequencies with theta=256.0. Returns (cos, sin) each [video_seq, head_dim].
fn precompute_hunyuan_3d_rope(
    config: &HunyuanVideoTransformerConfig,
    latent_frames: usize,
    patch_h: usize,
    patch_w: usize,
) -> (Vec<f32>, Vec<f32>) {
    let head_dim = config.attention_head_dim;
    let axes_dim = config.rope_axes_dim;
    let theta = config.rope_theta;

    let video_seq = latent_frames * patch_h * patch_w;
    let mut cos_cache = vec![0.0f32; video_seq * head_dim];
    let mut sin_cache = vec![0.0f32; video_seq * head_dim];

    // For each axis, compute inverse frequencies
    let inv_freqs: Vec<Vec<f64>> = axes_dim
        .iter()
        .map(|&axis_dim| {
            (0..axis_dim / 2)
                .map(|i| 1.0 / theta.powf(2.0 * i as f64 / axis_dim as f64))
                .collect()
        })
        .collect();

    for vid_idx in 0..video_seq {
        let t = (vid_idx / (patch_h * patch_w)) as f64;
        let spatial_idx = vid_idx % (patch_h * patch_w);
        let h = (spatial_idx / patch_w) as f64;
        let w = (spatial_idx % patch_w) as f64;

        let positions = [t, h, w];
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
// HunyuanVideo Transformer Components
// =============================================================================

/// Timestep + pooled text + guidance embedding.
/// Returns temb [B, inner_dim].
fn hunyuan_condition_embedding(
    wm: &impl WeightManager,
    timestep: Arc<dyn Tensor>,
    pooled_projections: Arc<dyn Tensor>,
    guidance: Option<Arc<dyn Tensor>>,
    config: &HunyuanVideoTransformerConfig,
    model_dtype: DType,
) -> Result<Arc<dyn Tensor>, Error> {
    let inner_dim = config.inner_dim();

    // Sinusoidal timestep embedding (256 channels)
    let timestep = cast(timestep, DType::F32);
    let half_dim = 128usize;
    let freqs: Vec<f32> = (0..half_dim)
        .map(|i| (-(10000.0f32.ln()) * i as f32 / half_dim as f32).exp())
        .collect();
    let freq_tensor = InputTensorInitialized::new(
        "hunyuan_timestep_freqs".to_string(),
        TensorData::new(
            freqs.clone().into(),
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
        &wm.prefix("time_text_embed.timestep_embedder.linear_1"),
        t_emb,
    )?;
    let temb = silu(temb)?;
    let temb = linear(
        &wm.prefix("time_text_embed.timestep_embedder.linear_2"),
        temb,
    )?;

    // Pooled text embedding: Linear(768 -> 3072) -> SiLU -> Linear(3072 -> 3072)
    let pooled = linear(
        &wm.prefix("time_text_embed.text_embedder.linear_1"),
        pooled_projections,
    )?;
    let pooled = silu(pooled)?;
    let pooled = linear(
        &wm.prefix("time_text_embed.text_embedder.linear_2"),
        pooled,
    )?;

    let mut temb: Arc<dyn Tensor> = Add::new(None, temb, pooled)?;

    // Guidance embedding (optional): guidance_scale * 1000 -> sinusoidal -> MLP
    if let Some(guidance) = guidance {
        let guidance = cast(guidance, DType::F32);
        let g_freq_tensor = InputTensorInitialized::new(
            "hunyuan_guidance_freqs".to_string(),
            TensorData::new(
                freqs.into(),
                Shape::new(vec![
                    Dimension::new(Some(1), None, None),
                    Dimension::new(Some(half_dim), None, None),
                ]),
            )?,
        );
        let g_args = Mul::new(None, guidance, g_freq_tensor)?;
        let g_cos = cos_op(g_args.clone())?;
        let g_sin = sin_op(g_args)?;
        let g_emb = Concat::new(None, vec![g_cos, g_sin], -1)?;
        let g_emb = cast(g_emb, model_dtype);

        let g = linear(
            &wm.prefix("time_text_embed.guidance_embedder.linear_1"),
            g_emb,
        )?;
        let g = silu(g)?;
        let g = linear(
            &wm.prefix("time_text_embed.guidance_embedder.linear_2"),
            g,
        )?;
        temb = Add::new(None, temb, g)?;
    }

    Ok(temb)
}

/// Context embedder: project LLaMA output + 2 refiner blocks.
/// Input: [B, 256, 4096] -> Output: [B, 256, 3072]
fn hunyuan_context_embedder(
    wm: &impl WeightManager,
    encoder_hidden_states: Arc<dyn Tensor>,
    timestep: Arc<dyn Tensor>,
    config: &HunyuanVideoTransformerConfig,
    model_dtype: DType,
) -> Result<Arc<dyn Tensor>, Error> {
    let inner_dim = config.inner_dim();
    let ce = wm.prefix("context_embedder");

    // Pool text tokens (mean) for refiner temb
    let text_seq = config.max_text_seq_len() as i64;
    let mean_weights = InputTensorInitialized::new(
        "hunyuan_ctx_pool_weights".to_string(),
        TensorData::new(
            TensorDataValue::F32(vec![1.0 / text_seq as f32; text_seq as usize]),
            Shape::new(vec![
                Dimension::new(Some(1), None, None),
                Dimension::new(Some(1), None, None),
                Dimension::new(Some(text_seq as usize), None, None),
            ]),
        )?,
    );
    let mean_weights = cast(mean_weights, model_dtype);
    let pooled = MatMul::new(None, mean_weights, encoder_hidden_states.clone())?;
    let pooled = reshape(pooled, vec![0, config.text_embed_dim as i64])?; // [B, 4096]

    // Refiner timestep + pooled conditioning
    let timestep = cast(timestep, DType::F32);
    let half_dim = 128usize;
    let freqs: Vec<f32> = (0..half_dim)
        .map(|i| (-(10000.0f32.ln()) * i as f32 / half_dim as f32).exp())
        .collect();
    let freq_tensor = InputTensorInitialized::new(
        "hunyuan_refiner_freqs".to_string(),
        TensorData::new(
            freqs.into(),
            Shape::new(vec![
                Dimension::new(Some(1), None, None),
                Dimension::new(Some(half_dim), None, None),
            ]),
        )?,
    );
    let args = Mul::new(None, timestep, freq_tensor)?;
    let cos_p = cos_op(args.clone())?;
    let sin_p = sin_op(args)?;
    let t_emb = Concat::new(None, vec![cos_p, sin_p], -1)?;
    let t_emb = cast(t_emb, model_dtype);
    let ref_temb = linear(&ce.prefix("t_embedder.timestep_embedder.linear_1"), t_emb)?;
    let ref_temb = silu(ref_temb)?;
    let ref_temb = linear(&ce.prefix("t_embedder.timestep_embedder.linear_2"), ref_temb)?;

    let ref_pooled = linear(&ce.prefix("t_embedder.pooler.linear_1"), pooled)?;
    let ref_pooled = silu(ref_pooled)?;
    let ref_pooled = linear(&ce.prefix("t_embedder.pooler.linear_2"), ref_pooled)?;
    let ref_temb: Arc<dyn Tensor> = Add::new(None, ref_temb, ref_pooled)?;

    // Project: Linear(4096 -> 3072)
    let mut hidden = linear(&ce.prefix("x_embedder"), encoder_hidden_states)?;

    // 2 refiner blocks
    for i in 0..config.num_refiner_layers {
        let blk = ce.prefix(&format!("refiner_blocks.{i}"));

        // AdaNorm for self-attention: SiLU(temb) -> Linear(3072, 6144) -> chunk(2) -> (gate_msa, gate_mlp)
        let ada = silu(ref_temb.clone())?;
        let ada = linear(&blk.prefix("norm1.linear"), ada)?;
        let ada = unsqueeze(ada, 1)?;
        let chunks = split_chunks(ada, inner_dim, 2)?;
        let gate_msa = chunks[0].clone();
        let gate_mlp = chunks[1].clone();

        // Self-attention
        let normed = layer_norm(&blk.prefix("norm1.norm"), hidden.clone(), config.norm_eps)?;
        let attn = hunyuan_refiner_attention(&blk, normed, config)?;
        hidden = Add::new(None, hidden.clone(), Mul::new(None, gate_msa, attn)?)?;

        // FFN (GELU-tanh activation, matching diffusers' FeedForward)
        let normed_ff = layer_norm(&blk.prefix("norm2.norm"), hidden.clone(), config.norm_eps)?;
        let ff = linear(&blk.prefix("ff.net.0.proj"), normed_ff)?;
        let ff = gelu_pytorch_tanh(ff)?;
        let ff = linear(&blk.prefix("ff.net.2"), ff)?;
        hidden = Add::new(None, hidden, Mul::new(None, gate_mlp, ff)?)?;
    }

    Ok(hidden)
}

/// Simple self-attention for refiner blocks (no RoPE).
fn hunyuan_refiner_attention(
    wm: &impl WeightManager,
    input: Arc<dyn Tensor>,
    config: &HunyuanVideoTransformerConfig,
) -> Result<Arc<dyn Tensor>, Error> {
    let nh = config.num_attention_heads as i64;
    let hd = config.attention_head_dim as i64;
    let inner_dim = config.inner_dim() as i64;

    let q = linear(&wm.prefix("attn.to_q"), input.clone())?;
    let k = linear(&wm.prefix("attn.to_k"), input.clone())?;
    let v = linear(&wm.prefix("attn.to_v"), input)?;

    let q = Transpose::new(None, reshape(q, vec![0, 0, nh, hd])?, Some(vec![0, 2, 1, 3]));
    let k = Transpose::new(None, reshape(k, vec![0, 0, nh, hd])?, Some(vec![0, 2, 1, 3]));
    let v = Transpose::new(None, reshape(v, vec![0, 0, nh, hd])?, Some(vec![0, 2, 1, 3]));

    let scores = MatMul::new(None, q, Transpose::new(None, k, Some(vec![0, 1, 3, 2])))?;
    let scores = div_scalar(scores, (config.attention_head_dim as f32).sqrt())?;
    let attn = Softmax::new(None, scores, Some(-1));
    let out = MatMul::new(None, attn, v)?;

    let out = Transpose::new(None, out, Some(vec![0, 2, 1, 3]));
    let out = reshape(out, vec![0, 0, inner_dim])?;
    linear(&wm.prefix("attn.to_out.0"), out)
}

impl HunyuanVideoTransformerConfig {
    fn max_text_seq_len(&self) -> usize {
        256
    }
}

/// Dual-stream block: separate video/text modulation, joint attention, separate FFN.
#[allow(clippy::too_many_arguments)]
fn hunyuan_dual_stream_block(
    wm: &impl WeightManager,
    hidden_states: Arc<dyn Tensor>,
    encoder_hidden_states: Arc<dyn Tensor>,
    temb: Arc<dyn Tensor>,
    config: &HunyuanVideoTransformerConfig,
    rope_cos: Arc<dyn Tensor>,
    rope_sin: Arc<dyn Tensor>,
) -> Result<(Arc<dyn Tensor>, Arc<dyn Tensor>), Error> {
    let inner_dim = config.inner_dim();
    let eps = config.norm_eps;

    // Video AdaLayerNormZero: SiLU(temb) -> Linear(3072, 6*3072=18432) -> chunk(6)
    let vid_mod = silu(temb.clone())?;
    let vid_mod = linear(&wm.prefix("norm1.linear"), vid_mod)?;
    let vid_mod = unsqueeze(vid_mod, 1)?;
    let vid_chunks = split_chunks(vid_mod, inner_dim, 6)?;
    let (shift_msa, scale_msa, gate_msa) =
        (vid_chunks[0].clone(), vid_chunks[1].clone(), vid_chunks[2].clone());
    let (shift_mlp, scale_mlp, gate_mlp) =
        (vid_chunks[3].clone(), vid_chunks[4].clone(), vid_chunks[5].clone());

    // Text AdaLayerNormZero: same structure
    let enc_mod = silu(temb.clone())?;
    let enc_mod = linear(&wm.prefix("norm1_context.linear"), enc_mod)?;
    let enc_mod = unsqueeze(enc_mod, 1)?;
    let enc_chunks = split_chunks(enc_mod, inner_dim, 6)?;
    let (c_shift_msa, c_scale_msa, c_gate_msa) =
        (enc_chunks[0].clone(), enc_chunks[1].clone(), enc_chunks[2].clone());
    let (c_shift_mlp, c_scale_mlp, c_gate_mlp) =
        (enc_chunks[3].clone(), enc_chunks[4].clone(), enc_chunks[5].clone());

    // Video pre-attention norm
    let norm_hidden = adaln_modulate(
        hidden_states.clone(), shift_msa, scale_msa, inner_dim, eps,
    )?;
    // Text pre-attention norm
    let norm_enc = adaln_modulate(
        encoder_hidden_states.clone(), c_shift_msa, c_scale_msa, inner_dim, eps,
    )?;

    // Joint attention with separate projections
    let (attn_vid, attn_enc) = hunyuan_dual_attention(
        wm, norm_hidden, norm_enc, config, rope_cos, rope_sin,
    )?;

    // Residual with gating
    let hidden_states = Add::new(
        None, hidden_states, Mul::new(None, gate_msa, attn_vid)?,
    )?;
    let encoder_hidden_states = Add::new(
        None, encoder_hidden_states, Mul::new(None, c_gate_msa, attn_enc)?,
    )?;

    // Video FFN: norm2 (bare LayerNorm, no affine) -> AdaLN modulate -> GELU FF -> gated residual
    let norm_h = layer_norm_bare(hidden_states.clone(), inner_dim, eps)?;
    let one_vid = ones_constant(inner_dim, norm_h.dtype());
    let ff_scale = Add::new(None, one_vid, scale_mlp)?;
    let norm_h = Mul::new(None, norm_h, ff_scale)?;
    let norm_h = Add::new(None, norm_h, shift_mlp)?;
    let ff_vid = linear(&wm.prefix("ff.net.0.proj"), norm_h)?;
    let ff_vid = gelu_pytorch_tanh(ff_vid)?;
    let ff_vid = linear(&wm.prefix("ff.net.2"), ff_vid)?;
    let hidden_states = Add::new(None, hidden_states, Mul::new(None, gate_mlp, ff_vid)?)?;

    // Text FFN: same pattern
    let norm_e = layer_norm_bare(encoder_hidden_states.clone(), inner_dim, eps)?;
    let one_enc = ones_constant(inner_dim, norm_e.dtype());
    let enc_ff_scale = Add::new(None, one_enc, c_scale_mlp)?;
    let norm_e = Mul::new(None, norm_e, enc_ff_scale)?;
    let norm_e = Add::new(None, norm_e, c_shift_mlp)?;
    let ff_enc = linear(&wm.prefix("ff_context.net.0.proj"), norm_e)?;
    let ff_enc = gelu_pytorch_tanh(ff_enc)?;
    let ff_enc = linear(&wm.prefix("ff_context.net.2"), ff_enc)?;
    let encoder_hidden_states = Add::new(
        None, encoder_hidden_states, Mul::new(None, c_gate_mlp, ff_enc)?,
    )?;

    Ok((hidden_states, encoder_hidden_states))
}

/// Dual-stream joint attention with separate video/text Q/K/V projections.
fn hunyuan_dual_attention(
    wm: &impl WeightManager,
    norm_hidden: Arc<dyn Tensor>,
    norm_enc: Arc<dyn Tensor>,
    config: &HunyuanVideoTransformerConfig,
    rope_cos: Arc<dyn Tensor>,
    rope_sin: Arc<dyn Tensor>,
) -> Result<(Arc<dyn Tensor>, Arc<dyn Tensor>), Error> {
    let nh = config.num_attention_heads as i64;
    let hd = config.attention_head_dim as i64;
    let inner_dim = config.inner_dim() as i64;
    let text_seq = config.max_text_seq_len() as i64;

    // Video Q/K/V
    let q = linear(&wm.prefix("attn.to_q"), norm_hidden.clone())?;
    let k = linear(&wm.prefix("attn.to_k"), norm_hidden.clone())?;
    let v = linear(&wm.prefix("attn.to_v"), norm_hidden)?;

    // Text Q/K/V (separate projections)
    let eq = linear(&wm.prefix("attn.add_q_proj"), norm_enc.clone())?;
    let ek = linear(&wm.prefix("attn.add_k_proj"), norm_enc.clone())?;
    let ev = linear(&wm.prefix("attn.add_v_proj"), norm_enc)?;

    // Reshape to multi-head
    let q = Transpose::new(None, reshape(q, vec![0, 0, nh, hd])?, Some(vec![0, 2, 1, 3]));
    let k = Transpose::new(None, reshape(k, vec![0, 0, nh, hd])?, Some(vec![0, 2, 1, 3]));
    let v = Transpose::new(None, reshape(v, vec![0, 0, nh, hd])?, Some(vec![0, 2, 1, 3]));
    let eq = Transpose::new(None, reshape(eq, vec![0, 0, nh, hd])?, Some(vec![0, 2, 1, 3]));
    let ek = Transpose::new(None, reshape(ek, vec![0, 0, nh, hd])?, Some(vec![0, 2, 1, 3]));
    let ev = Transpose::new(None, reshape(ev, vec![0, 0, nh, hd])?, Some(vec![0, 2, 1, 3]));

    // Per-head QK RMSNorm (learned affine)
    let q = rms_norm(&wm.prefix("attn.norm_q"), q, Some(config.norm_eps))?;
    let k = rms_norm(&wm.prefix("attn.norm_k"), k, Some(config.norm_eps))?;
    let eq = rms_norm(&wm.prefix("attn.norm_added_q"), eq, Some(config.norm_eps))?;
    let ek = rms_norm(&wm.prefix("attn.norm_added_k"), ek, Some(config.norm_eps))?;

    // RoPE on video Q/K only (interleaved=1 for repeat_interleave cos/sin)
    let q = RotaryEmbedding::new(
        None, q, rope_cos.clone(), rope_sin.clone(), None, Some(1), None, None,
    )?;
    let k = RotaryEmbedding::new(
        None, k, rope_cos, rope_sin, None, Some(1), None, None,
    )?;

    // Concatenate video + text: [B, nh, vid+txt, hd]
    let q_cat = Concat::new(None, vec![q, eq], 2)?;
    let k_cat = Concat::new(None, vec![k, ek], 2)?;
    let v_cat = Concat::new(None, vec![v, ev], 2)?;

    // Scaled dot-product attention
    let scores = MatMul::new(
        None, q_cat, Transpose::new(None, k_cat, Some(vec![0, 1, 3, 2])),
    )?;
    let scores = div_scalar(scores, (config.attention_head_dim as f32).sqrt())?;
    let attn = Softmax::new(None, scores, Some(-1));
    let out = MatMul::new(None, attn, v_cat)?;

    // Reshape and split
    let out = Transpose::new(None, out, Some(vec![0, 2, 1, 3]));
    let out = reshape(out, vec![0, 0, inner_dim])?;
    let total_seq = out.shape()[1].resolve()? as i64;
    let vid_seq = total_seq - text_seq;
    let vid_out = slice_axis(out.clone(), 1, 0, vid_seq)?;
    let txt_out = slice_axis(out, 1, vid_seq, total_seq)?;

    let vid_out = linear(&wm.prefix("attn.to_out.0"), vid_out)?;
    let txt_out = linear(&wm.prefix("attn.to_add_out"), txt_out)?;

    Ok((vid_out, txt_out))
}

/// Single-stream block: concatenated video+text, parallel attention + MLP.
#[allow(clippy::too_many_arguments)]
fn hunyuan_single_stream_block(
    wm: &impl WeightManager,
    hidden_states: Arc<dyn Tensor>,
    encoder_hidden_states: Arc<dyn Tensor>,
    temb: Arc<dyn Tensor>,
    config: &HunyuanVideoTransformerConfig,
    rope_cos: Arc<dyn Tensor>,
    rope_sin: Arc<dyn Tensor>,
) -> Result<(Arc<dyn Tensor>, Arc<dyn Tensor>), Error> {
    let inner_dim = config.inner_dim();
    let text_seq = config.max_text_seq_len() as i64;

    // Concatenate video + text
    let combined = Concat::new(None, vec![hidden_states.clone(), encoder_hidden_states.clone()], 1)?;
    let residual = combined.clone();

    // AdaLayerNormZeroSingle: SiLU(temb) -> Linear(3072, 3*3072=9216) -> chunk(3)
    let mod_params = silu(temb)?;
    let mod_params = linear(&wm.prefix("norm.linear"), mod_params)?;
    let mod_params = unsqueeze(mod_params, 1)?;
    let chunks = split_chunks(mod_params, inner_dim, 3)?;
    let (shift_msa, scale_msa, gate) =
        (chunks[0].clone(), chunks[1].clone(), chunks[2].clone());

    // Norm + modulate
    let normed = adaln_modulate(combined, shift_msa, scale_msa, inner_dim, config.norm_eps)?;

    // Parallel MLP path: proj_mlp -> GELU
    let mlp_hidden = linear(&wm.prefix("proj_mlp"), normed.clone())?;
    let mlp_hidden = gelu_pytorch_tanh(mlp_hidden)?;

    // Attention path: split video/text for RoPE, then joint attention
    let total_seq = normed.shape()[1].resolve()? as i64;
    let vid_seq = total_seq - text_seq;
    let norm_vid = slice_axis(normed.clone(), 1, 0, vid_seq)?;
    let norm_txt = slice_axis(normed, 1, vid_seq, total_seq)?;

    let (attn_vid, attn_txt) = hunyuan_single_attention(
        wm, norm_vid, norm_txt, config, rope_cos, rope_sin,
    )?;
    let attn_out = Concat::new(None, vec![attn_vid, attn_txt], 1)?;

    // Combine: cat([attn_out, mlp_hidden], dim=2) -> proj_out -> gate -> residual
    let combined_out = Concat::new(None, vec![attn_out, mlp_hidden], 2)?;
    let projected = linear(&wm.prefix("proj_out"), combined_out)?;
    let gated = Mul::new(None, gate, projected)?;
    let result = Add::new(None, residual, gated)?;

    // Split back into video and text
    let vid_out = slice_axis(result.clone(), 1, 0, vid_seq)?;
    let txt_out = slice_axis(result, 1, vid_seq, total_seq)?;

    Ok((vid_out, txt_out))
}

/// Single-stream attention: unified Q/K/V, RoPE on video portion only.
fn hunyuan_single_attention(
    wm: &impl WeightManager,
    norm_vid: Arc<dyn Tensor>,
    norm_txt: Arc<dyn Tensor>,
    config: &HunyuanVideoTransformerConfig,
    rope_cos: Arc<dyn Tensor>,
    rope_sin: Arc<dyn Tensor>,
) -> Result<(Arc<dyn Tensor>, Arc<dyn Tensor>), Error> {
    let nh = config.num_attention_heads as i64;
    let hd = config.attention_head_dim as i64;
    let text_seq = config.max_text_seq_len() as i64;

    // Concatenate video + text, then unified Q/K/V projections
    let combined = Concat::new(None, vec![norm_vid, norm_txt], 1)?;
    let q = linear(&wm.prefix("attn.to_q"), combined.clone())?;
    let k = linear(&wm.prefix("attn.to_k"), combined.clone())?;
    let v = linear(&wm.prefix("attn.to_v"), combined)?;

    let q = Transpose::new(None, reshape(q, vec![0, 0, nh, hd])?, Some(vec![0, 2, 1, 3]));
    let k = Transpose::new(None, reshape(k, vec![0, 0, nh, hd])?, Some(vec![0, 2, 1, 3]));
    let v = Transpose::new(None, reshape(v, vec![0, 0, nh, hd])?, Some(vec![0, 2, 1, 3]));

    // QK RMSNorm
    let q = rms_norm(&wm.prefix("attn.norm_q"), q, Some(config.norm_eps))?;
    let k = rms_norm(&wm.prefix("attn.norm_k"), k, Some(config.norm_eps))?;

    // RoPE on video portion only: split Q/K, apply RoPE to video, rejoin
    let total_seq = q.shape()[2].resolve()? as i64;
    let vid_seq = total_seq - text_seq;

    let q_vid = slice_axis(q.clone(), 2, 0, vid_seq)?;
    let q_txt = slice_axis(q, 2, vid_seq, total_seq)?;
    let k_vid = slice_axis(k.clone(), 2, 0, vid_seq)?;
    let k_txt = slice_axis(k, 2, vid_seq, total_seq)?;

    let q_vid = RotaryEmbedding::new(
        None, q_vid, rope_cos.clone(), rope_sin.clone(), None, Some(1), None, None,
    )?;
    let k_vid = RotaryEmbedding::new(
        None, k_vid, rope_cos, rope_sin, None, Some(1), None, None,
    )?;

    let q = Concat::new(None, vec![q_vid, q_txt], 2)?;
    let k = Concat::new(None, vec![k_vid, k_txt], 2)?;

    // Scaled dot-product attention
    let scores = MatMul::new(
        None, q, Transpose::new(None, k, Some(vec![0, 1, 3, 2])),
    )?;
    let scores = div_scalar(scores, (config.attention_head_dim as f32).sqrt())?;
    let attn = Softmax::new(None, scores, Some(-1));
    let out = MatMul::new(None, attn, v)?;

    let out = Transpose::new(None, out, Some(vec![0, 2, 1, 3]));
    let inner_dim = config.inner_dim() as i64;
    let out = reshape(out, vec![0, 0, inner_dim])?;

    // Split output (no to_out projection in single-stream)
    let vid_out = slice_axis(out.clone(), 1, 0, vid_seq)?;
    let txt_out = slice_axis(out, 1, vid_seq, total_seq)?;

    Ok((vid_out, txt_out))
}

/// Unpatchify: [B, T*pH*pW, pT*pH_s*pW_s*C] -> [B, C, T*pT, H, W]
fn hunyuan_unpatchify(
    input: Arc<dyn Tensor>,
    config: &HunyuanVideoTransformerConfig,
    latent_frames: usize,
    patch_h: usize,
    patch_w: usize,
) -> Result<Arc<dyn Tensor>, Error> {
    let f = latent_frames as i64;
    let ph = patch_h as i64;
    let pw = patch_w as i64;
    let pt = config.patch_size_t as i64;
    let ps = config.patch_size as i64;
    let c = config.out_channels as i64;

    // [B, F*pH*pW, pt*ps*ps*C] -> [B, F, pH, pW, pt, ps, ps, C]
    let x = reshape(input, vec![0, f, ph, pw, pt, ps, ps, c])?;
    // -> [B, C, F, pt, pH, ps, pW, ps] -> [B, C, F*pt, pH*ps, pW*ps]
    let x = Transpose::new(None, x, Some(vec![0, 7, 1, 4, 2, 5, 3, 6]));
    reshape(x, vec![0, c, f * pt, ph * ps, pw * ps]).map(|x| x as Arc<dyn Tensor>)
}

// =============================================================================
// Main Transformer Builder
// =============================================================================

pub fn load_hunyuan_video_transformer(
    weight_manager: impl WeightManager,
    config: HunyuanVideoTransformerConfig,
    output_method: WeightStorageStrategy,
) -> Result<Vec<u8>, anyhow::Error> {
    load_hunyuan_video_transformer_with_origin(weight_manager, config, output_method, None)
}

pub fn load_hunyuan_video_transformer_with_origin(
    weight_manager: impl WeightManager,
    config: HunyuanVideoTransformerConfig,
    output_method: WeightStorageStrategy,
    origin_path: Option<&Path>,
) -> Result<Vec<u8>, anyhow::Error> {
    let model_dtype = weight_manager
        .get_tensor("transformer_blocks.0.attn.to_q.weight")
        .map(|t| t.dtype())
        .unwrap_or(DType::BF16);
    let wm = CastingWeightManager::new(weight_manager, model_dtype);

    let inner_dim = config.inner_dim();
    let batch_dim = Dimension::new(Some(1), Some("batch".to_string()), None);

    // Default: 49 frames, 544x960
    // Latent: [B, 16, 13, 68, 120]
    // After patch (1,2,2): T=13, H=34, W=60
    let latent_frames = 13usize;
    let latent_h = 68usize;
    let latent_w = 120usize;
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
            Dimension::new(Some(config.max_text_seq_len()), None, None),
            Dimension::new(Some(config.text_embed_dim), None, None),
        ]),
    );
    let timestep_input = InputTensor::new(
        "timestep".to_string(),
        model_dtype,
        Shape::new(vec![batch_dim.clone()]),
    );
    let pooled_projections_input = InputTensor::new(
        "pooled_projections".to_string(),
        model_dtype,
        Shape::new(vec![
            batch_dim.clone(),
            Dimension::new(Some(config.pooled_projection_dim), None, None),
        ]),
    );
    let guidance_input = if config.guidance_embeds {
        Some(InputTensor::new(
            "guidance".to_string(),
            model_dtype,
            Shape::new(vec![batch_dim.clone()]),
        ))
    } else {
        None
    };

    let mut input_tensors: Vec<Arc<dyn Tensor>> = vec![
        latent_input.clone(),
        encoder_hidden_states_input.clone(),
        timestep_input.clone(),
        pooled_projections_input.clone(),
    ];
    if let Some(ref g) = guidance_input {
        input_tensors.push(g.clone());
    }

    // 1. Condition embedding
    let temb = hunyuan_condition_embedding(
        &wm, timestep_input.clone(), pooled_projections_input,
        guidance_input.map(|g| g as Arc<dyn Tensor>), &config, model_dtype,
    )?;

    // 2. Context embedder (LLaMA output -> refiner -> 3072-dim)
    let text_embeds = hunyuan_context_embedder(
        &wm, encoder_hidden_states_input, timestep_input, &config, model_dtype,
    )?;

    // 3. Patch embed: Conv3d(16, 3072, k=(1,2,2), s=(1,2,2))
    let patch_weight = wm.get_tensor("x_embedder.weight")?;
    let patch_bias = wm.get_tensor("x_embedder.bias").ok();
    let patched = Conv::new(
        Some("x_embedder".to_string()),
        latent_input,
        patch_weight,
        patch_bias,
        vec![config.patch_size_t as i64, config.patch_size as i64, config.patch_size as i64],
        vec![config.patch_size_t as i64, config.patch_size as i64, config.patch_size as i64],
        vec![0, 0, 0, 0, 0, 0],
        vec![1, 1, 1],
        1,
    )?;
    // [B, 3072, T, pH, pW] -> [B, T*pH*pW, 3072]
    let hidden_states = reshape(patched, vec![0, inner_dim as i64, -1])?;
    let hidden_states = Transpose::new(None, hidden_states, Some(vec![0, 2, 1]));

    // 4. Precompute 3D RoPE
    let (cos_vals, sin_vals) =
        precompute_hunyuan_3d_rope(&config, latent_frames, patch_h, patch_w);
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

    // 5. Dual-stream blocks
    println!(
        "Building HunyuanVideo transformer: {} dual + {} single blocks...",
        config.num_layers, config.num_single_layers
    );
    let mut hidden_states: Arc<dyn Tensor> = hidden_states;
    let mut encoder_hidden_states: Arc<dyn Tensor> = text_embeds;

    for i in 0..config.num_layers {
        let block_wm = wm.prefix(&format!("transformer_blocks.{i}"));
        let (next_h, next_e) = hunyuan_dual_stream_block(
            &block_wm, hidden_states, encoder_hidden_states,
            temb.clone(), &config, rope_cos.clone(), rope_sin.clone(),
        )?;
        hidden_states = next_h;
        encoder_hidden_states = next_e;
        if (i + 1) % 5 == 0 {
            println!("  Dual block {}/{}", i + 1, config.num_layers);
        }
    }

    // 6. Single-stream blocks
    for i in 0..config.num_single_layers {
        let block_wm = wm.prefix(&format!("single_transformer_blocks.{i}"));
        let (next_h, next_e) = hunyuan_single_stream_block(
            &block_wm, hidden_states, encoder_hidden_states,
            temb.clone(), &config, rope_cos.clone(), rope_sin.clone(),
        )?;
        hidden_states = next_h;
        encoder_hidden_states = next_e;
        if (i + 1) % 10 == 0 {
            println!("  Single block {}/{}", i + 1, config.num_single_layers);
        }
    }

    // 7. Output norm: AdaLayerNormContinuous
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

    // 8. Output projection + unpatchify
    let hidden_states = linear(&wm.prefix("proj_out"), hidden_states)?;
    let output = hunyuan_unpatchify(hidden_states, &config, latent_frames, patch_h, patch_w)?;

    let output_tensors: Vec<(String, Arc<dyn Tensor>)> =
        vec![("out_sample".to_string(), output)];

    println!("Built HunyuanVideo transformer graph, exporting...");
    let onnx_model = if let Some(origin) = origin_path {
        crate::onnx_graph::build_proto_with_origin_path(
            &input_tensors, &output_tensors, output_method, Some(origin),
        )?
    } else {
        crate::onnx_graph::build_proto(&input_tensors, &output_tensors, output_method)?
    };
    Ok(onnx_model.encode_to_vec())
}

// =============================================================================
// LLaMA 3 Encoder (simplified, no KV cache — for text encoding)
// =============================================================================

/// Build a LLaMA 3 encoder that outputs hidden states from layer -3.
/// This is a simplified version without KV cache for use as a text encoder.
pub fn load_llama3_encoder(
    weight_manager: impl WeightManager,
    num_layers: usize,
    num_skip_layers: usize, // skip last N layers (HunyuanVideo uses 2)
    max_seq_len: usize,
    output_method: WeightStorageStrategy,
) -> Result<Vec<u8>, anyhow::Error> {
    load_llama3_encoder_with_origin(
        weight_manager, num_layers, num_skip_layers, max_seq_len, output_method, None,
    )
}

pub fn load_llama3_encoder_with_origin(
    weight_manager: impl WeightManager,
    num_layers: usize,
    num_skip_layers: usize,
    max_seq_len: usize,
    output_method: WeightStorageStrategy,
    origin_path: Option<&Path>,
) -> Result<Vec<u8>, anyhow::Error> {
    let model_wm = weight_manager.prefix("model");

    let model_dtype = model_wm
        .get_tensor("layers.0.self_attn.q_proj.weight")
        .map(|t| t.dtype())
        .unwrap_or(DType::BF16);
    let wm = CastingWeightManager::new(model_wm, model_dtype);

    let batch_dim = Dimension::new(Some(1), Some("batch".to_string()), None);
    let seq_dim = Dimension::new(Some(max_seq_len), None, None);

    let token_input = InputTensor::new(
        "input_ids".to_string(),
        DType::I32,
        Shape::new(vec![batch_dim.clone(), seq_dim]),
    );

    let x = Gather::new(
        Some("embed_tokens".to_string()),
        wm.get_tensor("embed_tokens.weight")?,
        token_input.clone(),
        0,
    )?;

    let model_dim = x.shape().dims.last().unwrap().resolve()?;
    let num_heads = 32usize; // LLaMA 3 8B
    let num_kv_heads = 8usize;
    let head_dim = model_dim / num_heads;
    let rope_theta = 500000.0f64;

    // Precompute RoPE
    let half_head_dim = head_dim / 2;
    let (cos_values, sin_values) = {
        let inv_freq: Vec<f64> = (0..half_head_dim)
            .map(|i| 1.0 / rope_theta.powf(i as f64 * 2.0 / head_dim as f64))
            .collect();
        let mut cos_vals = vec![0.0f32; max_seq_len * half_head_dim];
        let mut sin_vals = vec![0.0f32; max_seq_len * half_head_dim];
        for pos in 0..max_seq_len {
            for (j, &freq) in inv_freq.iter().enumerate() {
                let angle = pos as f64 * freq;
                cos_vals[pos * half_head_dim + j] = angle.cos() as f32;
                sin_vals[pos * half_head_dim + j] = angle.sin() as f32;
            }
        }
        (cos_vals, sin_vals)
    };
    let rope_shape = Shape::new(vec![
        Dimension::new(Some(max_seq_len), None, None),
        Dimension::new(Some(half_head_dim), None, None),
    ]);
    let rope_cos: Arc<dyn Tensor> = InputTensorInitialized::new(
        "llama_rope_cos".to_string(),
        TensorData::new(TensorDataValue::F32(cos_values), rope_shape.clone())?,
    );
    let rope_sin: Arc<dyn Tensor> = InputTensorInitialized::new(
        "llama_rope_sin".to_string(),
        TensorData::new(TensorDataValue::F32(sin_values), rope_shape)?,
    );
    let rope_cos = cast(rope_cos, model_dtype);
    let rope_sin = cast(rope_sin, model_dtype);

    let active_layers = num_layers - num_skip_layers;
    println!("Building LLaMA 3 encoder: {} of {} layers...", active_layers, num_layers);

    let mut h: Arc<dyn Tensor> = x;
    for i in 0..active_layers {
        let lw = wm.prefix(&format!("layers.{i}"));

        // RMSNorm -> Self-attention
        let normed = rms_norm(&lw.prefix("input_layernorm"), h.clone(), None)?;

        let q = linear(&lw.prefix("self_attn.q_proj"), normed.clone())?;
        let k = linear(&lw.prefix("self_attn.k_proj"), normed.clone())?;
        let v = linear(&lw.prefix("self_attn.v_proj"), normed)?;

        let q = Transpose::new(
            None,
            reshape(q, vec![0, 0, num_heads as i64, head_dim as i64])?,
            Some(vec![0, 2, 1, 3]),
        );
        let k = Transpose::new(
            None,
            reshape(k, vec![0, 0, num_kv_heads as i64, head_dim as i64])?,
            Some(vec![0, 2, 1, 3]),
        );
        let v = Transpose::new(
            None,
            reshape(v, vec![0, 0, num_kv_heads as i64, head_dim as i64])?,
            Some(vec![0, 2, 1, 3]),
        );

        // RoPE (non-interleaved, half-split)
        let q = RotaryEmbedding::new(
            None, q, rope_cos.clone(), rope_sin.clone(), None, None, None, None,
        )?;
        let k = RotaryEmbedding::new(
            None, k, rope_cos.clone(), rope_sin.clone(), None, None, None, None,
        )?;

        // GQA: repeat KV heads
        let (k, v): (Arc<dyn Tensor>, Arc<dyn Tensor>) = if num_kv_heads != num_heads {
            let n_rep = num_heads / num_kv_heads;
            let expand_kv = |x: Arc<dyn Tensor>| -> Result<Arc<dyn Tensor>, Error> {
                let x = unsqueeze(x, 2)?;
                let x: Arc<dyn Tensor> = Concat::new(None, vec![x.clone(); n_rep], 2)?;
                reshape(x, vec![0, num_heads as i64, -1, head_dim as i64])
            };
            (expand_kv(k)?, expand_kv(v)?)
        } else {
            (k as Arc<dyn Tensor>, v as Arc<dyn Tensor>)
        };

        let scores = MatMul::new(
            None, q, Transpose::new(None, k, Some(vec![0, 1, 3, 2])),
        )?;
        let scores = div_scalar(scores, (head_dim as f32).sqrt())?;
        let attn = Softmax::new(None, scores, Some(-1));
        let out = MatMul::new(None, attn, v)?;
        let out = Transpose::new(None, out, Some(vec![0, 2, 1, 3]));
        let out = reshape(out, vec![0, 0, -1])?;
        let attn_out = linear(&lw.prefix("self_attn.o_proj"), out)?;
        h = Add::new(None, h, attn_out)?;

        // RMSNorm -> SwiGLU MLP
        let normed = rms_norm(&lw.prefix("post_attention_layernorm"), h.clone(), None)?;
        let gate = linear(&lw.prefix("mlp.gate_proj"), normed.clone())?;
        let gate = silu(gate)?;
        let up = linear(&lw.prefix("mlp.up_proj"), normed)?;
        let ff = Mul::new(None, gate, up)?;
        let ff = linear(&lw.prefix("mlp.down_proj"), ff)?;
        h = Add::new(None, h, ff)?;

        if (i + 1) % 10 == 0 {
            println!("  Layer {}/{}", i + 1, active_layers);
        }
    }

    // Final RMSNorm (applied before output, matching diffusers' model.norm)
    let h = rms_norm(&wm.prefix("norm"), h, None)?;

    let input_tensors: Vec<Arc<dyn Tensor>> = vec![token_input];
    let output_tensors: Vec<(String, Arc<dyn Tensor>)> =
        vec![("hidden_states".to_string(), h as Arc<dyn Tensor>)];

    println!("Built LLaMA 3 encoder graph, exporting...");
    let onnx_model = if let Some(origin) = origin_path {
        crate::onnx_graph::build_proto_with_origin_path(
            &input_tensors, &output_tensors, output_method, Some(origin),
        )?
    } else {
        crate::onnx_graph::build_proto(&input_tensors, &output_tensors, output_method)?
    };
    Ok(onnx_model.encode_to_vec())
}

// =============================================================================
// HunyuanVideo VAE Decoder
// =============================================================================

/// Causal Conv3d for HunyuanVideo VAE.
fn hunyuan_causal_conv3d(
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
    Conv::new(
        wm.get_prefix().map(|x| x.to_string()),
        padded, weight, bias,
        vec![kernel_t, kernel_s, kernel_s],
        vec![stride_t, stride_s, stride_s],
        vec![0, spatial_pad, spatial_pad, 0, spatial_pad, spatial_pad],
        vec![1, 1, 1],
        1,
    )
    .map(|x| x as Arc<dyn Tensor>)
}

/// VAE ResNet block with GroupNorm.
fn hunyuan_vae_resnet(
    wm: &impl WeightManager,
    input: Arc<dyn Tensor>,
    in_channels: usize,
    out_channels: usize,
    eps: f32,
    num_groups: i64,
) -> Result<Arc<dyn Tensor>, Error> {
    use crate::onnx_graph::pytorch::group_norm;

    let h = group_norm(&wm.prefix("norm1"), input.clone(), eps, num_groups)?;
    let h = silu(h)?;
    let h = hunyuan_causal_conv3d(&wm.prefix("conv1"), h, 3, 3, 1, 1)?;

    let h = group_norm(&wm.prefix("norm2"), h, eps, num_groups)?;
    let h = silu(h)?;
    let h = hunyuan_causal_conv3d(&wm.prefix("conv2"), h, 3, 3, 1, 1)?;

    let residual = if in_channels != out_channels {
        hunyuan_causal_conv3d(&wm.prefix("conv_shortcut"), input, 1, 1, 1, 1)?
    } else {
        input
    };

    Ok(Add::new(None, residual, h)?)
}

/// Spatial-only attention for VAE mid block (per-frame, using Linear Q/K/V).
fn hunyuan_vae_attention(
    wm: &impl WeightManager,
    input: Arc<dyn Tensor>,
    channels: usize,
    num_groups: i64,
    eps: f32,
    cur_t: usize,
    cur_h: usize,
    cur_w: usize,
) -> Result<Arc<dyn Tensor>, Error> {
    use crate::onnx_graph::pytorch::group_norm;

    let residual = input.clone();
    let h = group_norm(&wm.prefix("group_norm"), input, eps, num_groups)?;

    // [B, C, T, H, W] -> [B, T, C, H, W] -> [B*T, C, H, W] -> [B*T, C, H*W] -> [B*T, H*W, C]
    let x = Transpose::new(None, h, Some(vec![0, 2, 1, 3, 4]));
    let x = reshape(x, vec![-1, channels as i64, cur_h as i64, cur_w as i64])?;
    let x = reshape(x, vec![0, channels as i64, -1])?;
    let x = Transpose::new(None, x, Some(vec![0, 2, 1])); // [B*T, H*W, C]

    // Linear Q/K/V (matches diffusers' nn.Linear storage)
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

    // [B*T, H*W, C] -> [B*T, C, H*W] -> [B*T, C, H, W] -> [B, T, C, H, W] -> [B, C, T, H, W]
    let out = Transpose::new(None, out, Some(vec![0, 2, 1]));
    let out = reshape(out, vec![0, channels as i64, cur_h as i64, cur_w as i64])?;
    let out = reshape(out, vec![-1, cur_t as i64, channels as i64, cur_h as i64, cur_w as i64])?;
    let out = Transpose::new(None, out, Some(vec![0, 2, 1, 3, 4]));

    Ok(Add::new(None, residual, out)?)
}

/// Nearest-neighbor 3D upsample + CausalConv3d.
fn hunyuan_vae_upsample(
    wm: &impl WeightManager,
    input: Arc<dyn Tensor>,
    cur_t: usize,
    cur_h: usize,
    cur_w: usize,
    temporal: bool,
) -> Result<Arc<dyn Tensor>, Error> {
    use crate::onnx_graph::operators::Resize;

    let target_t = if temporal { (cur_t - 1) * 2 + 1 } else { cur_t };
    let target_h = cur_h * 2;
    let target_w = cur_w * 2;

    let scale_t = target_t as f32 / cur_t as f32;
    let scales = crate::onnx_graph::operators::Constant::new(
        None,
        TensorData::new(
            vec![1.0f32, 1.0, scale_t, 2.0, 2.0].into(),
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
        None, input, scales, "nearest".to_string(), Shape::new(output_dims),
    )?;

    hunyuan_causal_conv3d(&wm.prefix("conv"), x, 3, 3, 1, 1)
}

/// Build the HunyuanVideo VAE decoder.
pub fn load_hunyuan_video_vae_decoder(
    weight_manager: impl WeightManager,
    config: HunyuanVideoVaeConfig,
    output_method: WeightStorageStrategy,
) -> Result<Vec<u8>, anyhow::Error> {
    load_hunyuan_video_vae_decoder_with_origin(weight_manager, config, output_method, None)
}

pub fn load_hunyuan_video_vae_decoder_with_origin(
    weight_manager: impl WeightManager,
    config: HunyuanVideoVaeConfig,
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

    // For default 49 frames, 544x960:
    // Latent: [B, 16, 13, 68, 120]
    let mut cur_t = 13usize;
    let mut cur_h = 68usize;
    let mut cur_w = 120usize;

    let rev_channels: Vec<usize> = config.block_out_channels.iter().copied().rev().collect();
    let last_ch = *rev_channels.first().unwrap(); // 512
    let num_stages = rev_channels.len(); // 4

    // Temporal upsample flags: blocks 2,3 have temporal upsample (to get 4x total)
    // block 0 (512): no upsample
    // block 1 (512->512): spatial only
    // block 2 (256): spatial + temporal
    // block 3 (128): spatial + temporal
    // Actually: encoder does temporal downsample at blocks 2,3. Decoder reverses:
    // stage 0 (rev of block 3): 512, upsample spatial=1 temporal=2
    // stage 1 (rev of block 2): 512, upsample spatial=2 temporal=2
    // stage 2 (rev of block 1): 256, upsample spatial=2 temporal=1
    // stage 3 (rev of block 0): 128, upsample spatial=2 temporal=1
    let temporal_upsample = [true, true, false, false]; // reversed encoder downsamples

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

    // conv_in
    let mut x = hunyuan_causal_conv3d(&dec.prefix("conv_in"), latent_input, 3, 3, 1, 1)?;

    // Mid block: ResNet -> Attention -> ResNet
    x = hunyuan_vae_resnet(&dec.prefix("mid_block.resnets.0"), x, last_ch, last_ch, eps, ng)?;
    x = hunyuan_vae_attention(
        &dec.prefix("mid_block.attentions.0"), x, last_ch, ng, eps, cur_t, cur_h, cur_w,
    )?;
    x = hunyuan_vae_resnet(&dec.prefix("mid_block.resnets.1"), x, last_ch, last_ch, eps, ng)?;

    println!("Building HunyuanVideo VAE decoder: {} up stages...", num_stages);

    // Up blocks (reversed)
    let mut current_ch = last_ch;
    for stage in 0..num_stages {
        let out_ch = rev_channels[stage];
        let has_upsample = stage < num_stages - 1;
        let do_temporal = has_upsample && temporal_upsample[stage];

        for r in 0..(config.layers_per_block + 1) {
            let in_ch = if r == 0 { current_ch } else { out_ch };
            x = hunyuan_vae_resnet(
                &dec.prefix(&format!("up_blocks.{stage}.resnets.{r}")),
                x, in_ch, out_ch, eps, ng,
            )?;
        }
        current_ch = out_ch;

        if has_upsample {
            x = hunyuan_vae_upsample(
                &dec.prefix(&format!("up_blocks.{stage}.upsamplers.0")),
                x, cur_t, cur_h, cur_w, do_temporal,
            )?;
            if do_temporal {
                cur_t = (cur_t - 1) * 2 + 1;
            }
            cur_h *= 2;
            cur_w *= 2;
        }

        println!(
            "  Up stage {}/{}: {}ch, [T={}, H={}, W={}]",
            stage + 1, num_stages, out_ch, cur_t, cur_h, cur_w
        );
    }

    // Final: GroupNorm -> SiLU -> CausalConv3d
    {
        use crate::onnx_graph::pytorch::group_norm;
        x = group_norm(&dec.prefix("conv_norm_out"), x, eps, ng)?;
    }
    x = silu(x)?;
    let output = hunyuan_causal_conv3d(&dec.prefix("conv_out"), x, 3, 3, 1, 1)?;

    let output_tensors: Vec<(String, Arc<dyn Tensor>)> =
        vec![("video_out".to_string(), output)];

    println!("Built HunyuanVideo VAE decoder graph, exporting...");
    let onnx_model = if let Some(origin) = origin_path {
        crate::onnx_graph::build_proto_with_origin_path(
            &input_tensors, &output_tensors, output_method, Some(origin),
        )?
    } else {
        crate::onnx_graph::build_proto(&input_tensors, &output_tensors, output_method)?
    };
    Ok(onnx_model.encode_to_vec())
}
