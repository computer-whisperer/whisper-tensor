use crate::onnx_graph::WeightStorageStrategy;
use crate::onnx_graph::operators::{
    Add, Concat, Constant, Gather, LayerNormalization, MatMul, Mul, RMSNormalization,
    RotaryEmbedding, Slice, Softmax, Transpose,
};
use crate::onnx_graph::pytorch::{
    cast, div_scalar, gelu, layer_norm, linear, reshape, silu, unsqueeze,
};
use crate::onnx_graph::tensor::{
    DType, Dimension, InputTensor, InputTensorInitialized, Shape, Tensor, TensorData,
    TensorDataValue,
};
use crate::onnx_graph::weights::WeightManager;
use crate::sd_common::{CastingWeightManager, build_causal_mask, cos_op, sin_op};
use prost::Message;

type TensorPair = (Arc<dyn Tensor>, Arc<dyn Tensor>);
use std::sync::Arc;

pub struct FluxConfig {
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub mlp_ratio: f64,
    pub num_double_blocks: usize,
    pub num_single_blocks: usize,
    pub latent_channels: usize,
    pub patch_size: usize,
    pub t5_dim: usize,
    pub clip_pooled_dim: usize,
    pub timestep_dim: usize,
    pub theta: f64,
    pub axes_dim: [usize; 3],
    /// Image resolution (pixels).
    pub img_height: usize,
    pub img_width: usize,
    /// Maximum T5 text sequence length.
    pub max_txt_seq_len: usize,
    /// Whether the model has a guidance embedding (Dev=true, Schnell=false).
    pub has_guidance: bool,
}

impl FluxConfig {
    fn base(img_size: usize, max_txt_seq_len: usize, has_guidance: bool) -> Self {
        Self {
            hidden_dim: 3072,
            num_heads: 24,
            head_dim: 128,
            mlp_ratio: 4.0,
            num_double_blocks: 19,
            num_single_blocks: 38,
            latent_channels: 16,
            patch_size: 2,
            t5_dim: 4096,
            clip_pooled_dim: 768,
            timestep_dim: 256,
            theta: 10_000.0,
            axes_dim: [16, 56, 56],
            img_height: img_size,
            img_width: img_size,
            max_txt_seq_len,
            has_guidance,
        }
    }

    pub fn schnell(img_size: usize, max_txt_seq_len: usize) -> Self {
        Self::base(img_size, max_txt_seq_len, false)
    }

    pub fn dev(img_size: usize, max_txt_seq_len: usize) -> Self {
        Self::base(img_size, max_txt_seq_len, true)
    }

    pub fn schnell_1024(max_txt_seq_len: usize) -> Self {
        Self::schnell(1024, max_txt_seq_len)
    }

    fn latent_h(&self) -> usize {
        self.img_height / 8
    }
    fn latent_w(&self) -> usize {
        self.img_width / 8
    }
    fn patch_h(&self) -> usize {
        self.latent_h() / self.patch_size
    }
    fn patch_w(&self) -> usize {
        self.latent_w() / self.patch_size
    }
    fn img_seq_len(&self) -> usize {
        self.patch_h() * self.patch_w()
    }
    fn total_seq_len(&self) -> usize {
        self.max_txt_seq_len + self.img_seq_len()
    }
    fn mlp_dim(&self) -> usize {
        (self.hidden_dim as f64 * self.mlp_ratio) as usize
    }
    fn patch_dim(&self) -> usize {
        self.latent_channels * self.patch_size * self.patch_size
    }
}

// ============================================================================
// RoPE precomputation
// ============================================================================

/// Precompute 2D axial RoPE cos/sin caches for the combined txt+img sequence.
///
/// Returns (cos_cache, sin_cache) each of shape [total_seq_len, head_dim/2].
/// Text tokens (all-zero positions) get cos=1, sin=0 (identity rotation).
/// Image tokens get 2D spatial encoding from (y, x) patch coordinates.
fn precompute_flux_rope(config: &FluxConfig) -> (Vec<f32>, Vec<f32>) {
    let total_seq = config.total_seq_len();
    let half_head = config.head_dim / 2;
    let mut cos_cache = vec![0.0f32; total_seq * half_head];
    let mut sin_cache = vec![0.0f32; total_seq * half_head];

    // Compute per-axis frequencies
    let axes_dim = config.axes_dim;
    let theta = config.theta;

    // axes_dim = [16, 56, 56] → half dims per axis: [8, 28, 28] = 64 total
    let half_dims: [usize; 3] = [axes_dim[0] / 2, axes_dim[1] / 2, axes_dim[2] / 2];

    // Precompute inverse frequencies for each axis
    let mut inv_freqs: Vec<Vec<f64>> = Vec::new();
    for &axis_dim in &axes_dim {
        let freqs: Vec<f64> = (0..axis_dim / 2)
            .map(|i| 1.0 / theta.powf(2.0 * i as f64 / axis_dim as f64))
            .collect();
        inv_freqs.push(freqs);
    }

    for pos in 0..total_seq {
        let (id_pos, y_pos, x_pos) = if pos < config.max_txt_seq_len {
            // Text tokens: all-zero positions
            (0.0f64, 0.0, 0.0)
        } else {
            // Image tokens: 2D grid position
            let img_idx = pos - config.max_txt_seq_len;
            let y = (img_idx / config.patch_w()) as f64;
            let x = (img_idx % config.patch_w()) as f64;
            (0.0, y, x)
        };

        let positions = [id_pos, y_pos, x_pos];
        let mut offset = 0;
        for (axis, &pos_val) in positions.iter().enumerate() {
            for (i, &freq) in inv_freqs[axis].iter().enumerate() {
                let angle = pos_val * freq;
                cos_cache[pos * half_head + offset + i] = angle.cos() as f32;
                sin_cache[pos * half_head + offset + i] = angle.sin() as f32;
            }
            offset += half_dims[axis];
        }
    }

    (cos_cache, sin_cache)
}

// ============================================================================
// Helpers
// ============================================================================

/// Slice a tensor along a single axis (handles negative axis).
fn slice_axis(
    input: Arc<dyn Tensor>,
    axis: i64,
    start: i64,
    end: i64,
) -> Result<Arc<dyn Tensor>, crate::onnx_graph::Error> {
    // Normalize negative axis
    let resolved_axis = if axis < 0 {
        input.rank() as i64 + axis
    } else {
        axis
    };
    let shape1 = Shape::new(vec![Dimension::new(Some(1), None, None)]);
    let starts = Constant::new(None, TensorData::new(vec![start].into(), shape1.clone())?);
    let ends = Constant::new(None, TensorData::new(vec![end].into(), shape1.clone())?);
    let axes = Constant::new(None, TensorData::new(vec![resolved_axis].into(), shape1)?);
    Ok(Slice::new(None, input, starts, ends, Some(axes), None)?)
}

/// Create a constant of ones with given size and dtype.
fn ones_constant(size: usize, dtype: DType) -> Arc<dyn Tensor> {
    let shape = Shape::new(vec![Dimension::new(Some(size), None, None)]);
    match dtype {
        DType::F32 => Constant::new(
            None,
            TensorData::new(TensorDataValue::F32(vec![1.0; size]), shape).unwrap(),
        ),
        DType::BF16 => Constant::new(
            None,
            TensorData::new(TensorDataValue::BF16(vec![half::bf16::ONE; size]), shape).unwrap(),
        ),
        DType::F16 => Constant::new(
            None,
            TensorData::new(TensorDataValue::F16(vec![half::f16::ONE; size]), shape).unwrap(),
        ),
        _ => panic!("unsupported dtype for ones_constant: {:?}", dtype),
    }
}

/// LayerNorm without learned affine parameters (elementwise_affine=False).
fn layer_norm_bare(
    input: Arc<dyn Tensor>,
    hidden_dim: usize,
    epsilon: f32,
) -> Result<Arc<LayerNormalization>, crate::onnx_graph::Error> {
    let scale = ones_constant(hidden_dim, input.dtype());
    LayerNormalization::new(None, input, scale, None, -1, epsilon, 1)
}

/// AdaLN modulation: (1 + scale) * LayerNorm(x) + shift
fn adaln_modulate(
    input: Arc<dyn Tensor>,
    shift: Arc<dyn Tensor>,
    scale: Arc<dyn Tensor>,
    hidden_dim: usize,
) -> Result<Arc<dyn Tensor>, crate::onnx_graph::Error> {
    let normed = layer_norm_bare(input, hidden_dim, 1e-6)?;
    let one = ones_constant(hidden_dim, normed.dtype());
    let scale_plus_one = Add::new(None, one, scale)?;
    let scaled = Mul::new(None, normed, scale_plus_one)?;
    Ok(Add::new(None, scaled, shift)?)
}

/// Apply QK-norm (RMSNorm with learned scale on the head_dim dimension).
fn qk_norm(
    q: Arc<dyn Tensor>,
    k: Arc<dyn Tensor>,
    q_scale: Arc<dyn Tensor>,
    k_scale: Arc<dyn Tensor>,
) -> Result<TensorPair, crate::onnx_graph::Error> {
    let q = RMSNormalization::new(None, q, q_scale, None, -1)?;
    let k = RMSNormalization::new(None, k, k_scale, None, -1)?;
    Ok((q, k))
}

/// Patchify latent: [B, C, H, W] → [B, H/p * W/p, C*p*p]
fn patchify(
    input: Arc<dyn Tensor>,
    config: &FluxConfig,
) -> Result<Arc<dyn Tensor>, crate::onnx_graph::Error> {
    let c = config.latent_channels as i64;
    let ph = config.patch_h() as i64;
    let pw = config.patch_w() as i64;
    let p = config.patch_size as i64;

    // [B, C, H, W] → [B, C, H/p, p, W/p, p]
    let x = reshape(input, vec![0, c, ph, p, pw, p])?;
    // → [B, H/p, W/p, C, p, p]
    let x = Transpose::new(None, x, Some(vec![0, 2, 4, 1, 3, 5]));
    // → [B, H/p * W/p, C*p*p]
    Ok(reshape(x, vec![0, -1, config.patch_dim() as i64])?)
}

/// Unpatchify: [B, H/p * W/p, C*p*p] → [B, C, H, W]
fn unpatchify(
    input: Arc<dyn Tensor>,
    config: &FluxConfig,
) -> Result<Arc<dyn Tensor>, crate::onnx_graph::Error> {
    let c = config.latent_channels as i64;
    let ph = config.patch_h() as i64;
    let pw = config.patch_w() as i64;
    let p = config.patch_size as i64;

    // [B, num_patches, patch_dim] → [B, H/p, W/p, C, p, p]
    let x = reshape(input, vec![0, ph, pw, c, p, p])?;
    // → [B, C, H/p, p, W/p, p]
    let x = Transpose::new(None, x, Some(vec![0, 3, 1, 4, 2, 5]));
    // → [B, C, H, W]
    let h = config.latent_h() as i64;
    let w = config.latent_w() as i64;
    Ok(reshape(x, vec![0, c, h, w])?)
}

/// Flux timestep embedding: t → sinusoidal(t * 1000, dim=256) → MLP(256→3072→3072)
fn flux_timestep_embedding(
    wm: &impl WeightManager,
    timestep: Arc<dyn Tensor>,
    config: &FluxConfig,
    model_dtype: DType,
) -> Result<Arc<dyn Tensor>, crate::onnx_graph::Error> {
    let half_dim = config.timestep_dim / 2;

    // Scale timestep by 1000 (Flux convention)
    let time_factor = Constant::new(
        None,
        TensorData::new(
            TensorDataValue::F32(vec![1000.0]),
            Shape::new(vec![Dimension::new(Some(1), None, None)]),
        )?,
    );
    let scaled_t = Mul::new(None, timestep, time_factor)?;

    // Precompute frequencies
    let freqs: Vec<f32> = (0..half_dim)
        .map(|i| (-(10000.0f32.ln()) * i as f32 / half_dim as f32).exp())
        .collect();
    let freq_tensor = InputTensorInitialized::new(
        "timestep_freqs".to_string(),
        TensorData::new(
            freqs.into(),
            Shape::new(vec![
                Dimension::new(Some(1), None, None),
                Dimension::new(Some(half_dim), None, None),
            ]),
        )?,
    );

    let args = Mul::new(None, scaled_t, freq_tensor)?;
    let cos_part = cos_op(args.clone())?;
    let sin_part = sin_op(args)?;
    let emb = Concat::new(None, vec![cos_part, sin_part], -1)?;

    // Cast to model_dtype, then MLP
    let emb = cast(emb, model_dtype);
    let emb = linear(&wm.prefix("in_layer"), emb)?;
    let emb = silu(emb)?;
    let emb = linear(&wm.prefix("out_layer"), emb)?;
    Ok(emb)
}

// ============================================================================
// Double stream block
// ============================================================================

/// One Flux double-stream block: separate img/txt streams with joint attention.
#[allow(clippy::too_many_arguments)]
fn flux_double_block(
    wm: &impl WeightManager,
    img: Arc<dyn Tensor>,
    txt: Arc<dyn Tensor>,
    vec_cond: Arc<dyn Tensor>,
    cos_cache: Arc<dyn Tensor>,
    sin_cache: Arc<dyn Tensor>,
    config: &FluxConfig,
) -> Result<TensorPair, crate::onnx_graph::Error> {
    let h = config.hidden_dim;
    let nh = config.num_heads as i64;
    let hd = config.head_dim as i64;

    // --- Modulation ---
    let vec_act = silu(vec_cond.clone())?;
    let img_mod_out = linear(&wm.prefix("img_mod.lin"), vec_act.clone())?;
    let txt_mod_out = linear(&wm.prefix("txt_mod.lin"), vec_act)?;

    // Unsqueeze to [B, 1, 6*hidden] for broadcasting with [B, seq, hidden]
    let img_mod_out = unsqueeze(img_mod_out, 1)?;
    let txt_mod_out = unsqueeze(txt_mod_out, 1)?;

    // Split into 6 modulation params each: shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp
    let img_mods: Vec<Arc<dyn Tensor>> = (0..6)
        .map(|i| {
            slice_axis(
                img_mod_out.clone(),
                -1,
                (i * h) as i64,
                ((i + 1) * h) as i64,
            )
        })
        .collect::<Result<_, _>>()?;
    let txt_mods: Vec<Arc<dyn Tensor>> = (0..6)
        .map(|i| {
            slice_axis(
                txt_mod_out.clone(),
                -1,
                (i * h) as i64,
                ((i + 1) * h) as i64,
            )
        })
        .collect::<Result<_, _>>()?;

    // --- Self-attention ---
    // AdaLN modulate
    let img_modulated = adaln_modulate(img.clone(), img_mods[0].clone(), img_mods[1].clone(), h)?;
    let txt_modulated = adaln_modulate(txt.clone(), txt_mods[0].clone(), txt_mods[1].clone(), h)?;

    // QKV projections
    let img_qkv = linear(&wm.prefix("img_attn.qkv"), img_modulated)?;
    let txt_qkv = linear(&wm.prefix("txt_attn.qkv"), txt_modulated)?;

    // Split Q, K, V (each 3072 from 9216)
    let img_q = slice_axis(img_qkv.clone(), -1, 0, h as i64)?;
    let img_k = slice_axis(img_qkv.clone(), -1, h as i64, 2 * h as i64)?;
    let img_v = slice_axis(img_qkv, -1, 2 * h as i64, 3 * h as i64)?;
    let txt_q = slice_axis(txt_qkv.clone(), -1, 0, h as i64)?;
    let txt_k = slice_axis(txt_qkv.clone(), -1, h as i64, 2 * h as i64)?;
    let txt_v = slice_axis(txt_qkv, -1, 2 * h as i64, 3 * h as i64)?;

    // Reshape to [B, seq, num_heads, head_dim] and transpose to [B, num_heads, seq, head_dim]
    let img_q = Transpose::new(
        None,
        reshape(img_q, vec![0, 0, nh, hd])?,
        Some(vec![0, 2, 1, 3]),
    );
    let img_k = Transpose::new(
        None,
        reshape(img_k, vec![0, 0, nh, hd])?,
        Some(vec![0, 2, 1, 3]),
    );
    let img_v = Transpose::new(
        None,
        reshape(img_v, vec![0, 0, nh, hd])?,
        Some(vec![0, 2, 1, 3]),
    );
    let txt_q = Transpose::new(
        None,
        reshape(txt_q, vec![0, 0, nh, hd])?,
        Some(vec![0, 2, 1, 3]),
    );
    let txt_k = Transpose::new(
        None,
        reshape(txt_k, vec![0, 0, nh, hd])?,
        Some(vec![0, 2, 1, 3]),
    );
    let txt_v = Transpose::new(
        None,
        reshape(txt_v, vec![0, 0, nh, hd])?,
        Some(vec![0, 2, 1, 3]),
    );

    // QK-norm
    let img_qn_scale = wm.get_tensor("img_attn.norm.query_norm.scale")?;
    let img_kn_scale = wm.get_tensor("img_attn.norm.key_norm.scale")?;
    let txt_qn_scale = wm.get_tensor("txt_attn.norm.query_norm.scale")?;
    let txt_kn_scale = wm.get_tensor("txt_attn.norm.key_norm.scale")?;
    let (img_q, img_k) = qk_norm(img_q, img_k, img_qn_scale, img_kn_scale)?;
    let (txt_q, txt_k) = qk_norm(txt_q, txt_k, txt_qn_scale, txt_kn_scale)?;

    // Concatenate for joint attention: [B, H, txt_seq+img_seq, D]
    let q = Concat::new(None, vec![txt_q, img_q], 2)?;
    let k = Concat::new(None, vec![txt_k, img_k], 2)?;
    let v = Concat::new(None, vec![txt_v, img_v], 2)?;

    // Apply RoPE (interleaved=1 for Flux)
    let q = RotaryEmbedding::new(
        None,
        q,
        cos_cache.clone(),
        sin_cache.clone(),
        None,
        Some(1), // interleaved
        None,
        None,
    )?;
    let k = RotaryEmbedding::new(
        None,
        k,
        cos_cache.clone(),
        sin_cache.clone(),
        None,
        Some(1),
        None,
        None,
    )?;

    // Scaled dot-product attention
    let scale = (config.head_dim as f32).sqrt();
    let scores = MatMul::new(None, q, Transpose::new(None, k, Some(vec![0, 1, 3, 2])))?;
    let scores = div_scalar(scores, scale)?;
    let attn_weights = Softmax::new(None, scores, Some(-1));
    let attn_out = MatMul::new(None, attn_weights, v)?;

    // Split back: txt [B, H, txt_seq, D], img [B, H, img_seq, D]
    let txt_seq = config.max_txt_seq_len as i64;
    let txt_attn = slice_axis(attn_out.clone(), 2, 0, txt_seq)?;
    let img_attn = slice_axis(attn_out, 2, txt_seq, txt_seq + config.img_seq_len() as i64)?;

    // Reshape back to [B, seq, hidden]
    let img_attn = reshape(
        Transpose::new(None, img_attn, Some(vec![0, 2, 1, 3])),
        vec![0, 0, h as i64],
    )?;
    let txt_attn = reshape(
        Transpose::new(None, txt_attn, Some(vec![0, 2, 1, 3])),
        vec![0, 0, h as i64],
    )?;

    // Output projections
    let img_attn = linear(&wm.prefix("img_attn.proj"), img_attn)?;
    let txt_attn = linear(&wm.prefix("txt_attn.proj"), txt_attn)?;

    // Gate + residual for attention
    let img = Add::new(None, img, Mul::new(None, img_mods[2].clone(), img_attn)?)?;
    let txt = Add::new(None, txt, Mul::new(None, txt_mods[2].clone(), txt_attn)?)?;

    // --- MLP ---
    let img_mlp_in = adaln_modulate(img.clone(), img_mods[3].clone(), img_mods[4].clone(), h)?;
    let txt_mlp_in = adaln_modulate(txt.clone(), txt_mods[3].clone(), txt_mods[4].clone(), h)?;

    // MLP: linear → GELU → linear
    let img_mlp = linear(&wm.prefix("img_mlp.0"), img_mlp_in)?;
    let img_mlp = gelu(img_mlp)?;
    let img_mlp = linear(&wm.prefix("img_mlp.2"), img_mlp)?;
    let txt_mlp = linear(&wm.prefix("txt_mlp.0"), txt_mlp_in)?;
    let txt_mlp = gelu(txt_mlp)?;
    let txt_mlp = linear(&wm.prefix("txt_mlp.2"), txt_mlp)?;

    // Gate + residual for MLP
    let img = Add::new(None, img, Mul::new(None, img_mods[5].clone(), img_mlp)?)?;
    let txt = Add::new(None, txt, Mul::new(None, txt_mods[5].clone(), txt_mlp)?)?;

    Ok((img, txt))
}

// ============================================================================
// Single stream block
// ============================================================================

/// One Flux single-stream block: combined img+txt sequence.
fn flux_single_block(
    wm: &impl WeightManager,
    x: Arc<dyn Tensor>,
    vec_cond: Arc<dyn Tensor>,
    cos_cache: Arc<dyn Tensor>,
    sin_cache: Arc<dyn Tensor>,
    config: &FluxConfig,
) -> Result<Arc<dyn Tensor>, crate::onnx_graph::Error> {
    let h = config.hidden_dim;
    let nh = config.num_heads as i64;
    let hd = config.head_dim as i64;
    let mlp_dim = config.mlp_dim();

    // Modulation: 3 params (shift, scale, gate)
    let vec_act = silu(vec_cond)?;
    let mod_out = linear(&wm.prefix("modulation.lin"), vec_act)?;
    let mod_out = unsqueeze(mod_out, 1)?; // [B, 1, 3*hidden]
    let shift = slice_axis(mod_out.clone(), -1, 0, h as i64)?;
    let scale = slice_axis(mod_out.clone(), -1, h as i64, 2 * h as i64)?;
    let gate = slice_axis(mod_out, -1, 2 * h as i64, 3 * h as i64)?;

    // AdaLN
    let x_mod = adaln_modulate(x.clone(), shift, scale, h)?;

    // Fused linear1: QKV + MLP input → [B, seq, 9216 + 12288] = [B, seq, 21504]
    let fused = linear(&wm.prefix("linear1"), x_mod)?;
    let qkv_dim = 3 * h;
    let qkv = slice_axis(fused.clone(), -1, 0, qkv_dim as i64)?;
    let mlp_in = slice_axis(fused, -1, qkv_dim as i64, (qkv_dim + mlp_dim) as i64)?;

    // Split Q, K, V
    let q = slice_axis(qkv.clone(), -1, 0, h as i64)?;
    let k = slice_axis(qkv.clone(), -1, h as i64, 2 * h as i64)?;
    let v = slice_axis(qkv, -1, 2 * h as i64, 3 * h as i64)?;

    // Reshape to [B, seq, H, D] → [B, H, seq, D]
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

    // QK-norm
    let q_scale = wm.get_tensor("norm.query_norm.scale")?;
    let k_scale = wm.get_tensor("norm.key_norm.scale")?;
    let (q, k) = qk_norm(q, k, q_scale, k_scale)?;

    // RoPE
    let q = RotaryEmbedding::new(
        None,
        q,
        cos_cache.clone(),
        sin_cache.clone(),
        None,
        Some(1),
        None,
        None,
    )?;
    let k = RotaryEmbedding::new(None, k, cos_cache, sin_cache, None, Some(1), None, None)?;

    // Attention
    let scale_val = (config.head_dim as f32).sqrt();
    let scores = MatMul::new(None, q, Transpose::new(None, k, Some(vec![0, 1, 3, 2])))?;
    let scores = div_scalar(scores, scale_val)?;
    let attn_weights = Softmax::new(None, scores, Some(-1));
    let attn_out = MatMul::new(None, attn_weights, v)?;

    // Reshape attention output back to [B, seq, hidden]
    let attn_out = reshape(
        Transpose::new(None, attn_out, Some(vec![0, 2, 1, 3])),
        vec![0, 0, h as i64],
    )?;

    // GELU on MLP
    let mlp_act = gelu(mlp_in)?;

    // Fused linear2: concat(attn_out, mlp_act) → hidden
    let fused_out = Concat::new(None, vec![attn_out, mlp_act], -1)?;
    let output = linear(&wm.prefix("linear2"), fused_out)?;

    // Gate + residual
    Ok(Add::new(None, x, Mul::new(None, gate, output)?)?)
}

// ============================================================================
// Main builder
// ============================================================================

pub fn load_flux_dit(
    weight_manager: impl WeightManager,
    config: FluxConfig,
    output_method: WeightStorageStrategy,
) -> Result<Vec<u8>, anyhow::Error> {
    load_flux_dit_with_origin(weight_manager, config, output_method, None)
}

pub fn load_flux_dit_with_origin(
    weight_manager: impl WeightManager,
    config: FluxConfig,
    output_method: WeightStorageStrategy,
    origin_path: Option<&std::path::Path>,
) -> Result<Vec<u8>, anyhow::Error> {
    let model_dtype = weight_manager
        .get_tensor("img_in.weight")
        .map(|t| t.dtype())
        .unwrap_or(DType::BF16);

    let batch_dim = Dimension::new(Some(1), Some("batch".to_string()), None);

    // --- Inputs ---
    let latent_input = InputTensor::new(
        "latent_sample".to_string(),
        model_dtype,
        Shape::new(vec![
            batch_dim.clone(),
            Dimension::new(Some(config.latent_channels), None, None),
            Dimension::new(Some(config.latent_h()), None, None),
            Dimension::new(Some(config.latent_w()), None, None),
        ]),
    );
    let timestep_input = InputTensor::new(
        "timestep".to_string(),
        DType::F32,
        Shape::new(vec![batch_dim.clone(), Dimension::new(Some(1), None, None)]),
    );
    let clip_pooled_input = InputTensor::new(
        "clip_pooled".to_string(),
        model_dtype,
        Shape::new(vec![
            batch_dim.clone(),
            Dimension::new(Some(config.clip_pooled_dim), None, None),
        ]),
    );
    let t5_hidden_input = InputTensor::new(
        "t5_hidden_states".to_string(),
        model_dtype,
        Shape::new(vec![
            batch_dim.clone(),
            Dimension::new(Some(config.max_txt_seq_len), None, None),
            Dimension::new(Some(config.t5_dim), None, None),
        ]),
    );

    // Optional guidance input (Flux Dev only)
    let guidance_input = if config.has_guidance {
        Some(InputTensor::new(
            "guidance".to_string(),
            DType::F32,
            Shape::new(vec![batch_dim.clone(), Dimension::new(Some(1), None, None)]),
        ))
    } else {
        None
    };

    let mut input_tensors: Vec<Arc<dyn Tensor>> = vec![
        latent_input.clone(),
        timestep_input.clone(),
        clip_pooled_input.clone(),
        t5_hidden_input.clone(),
    ];
    if let Some(ref gi) = guidance_input {
        input_tensors.push(gi.clone());
    }

    // --- Precompute RoPE ---
    let (cos_vals, sin_vals) = precompute_flux_rope(&config);
    let rope_shape = Shape::new(vec![
        Dimension::new(Some(1), None, None),
        Dimension::new(Some(config.total_seq_len()), None, None),
        Dimension::new(Some(config.head_dim / 2), None, None),
    ]);
    let cos_cache: Arc<dyn Tensor> = InputTensorInitialized::new(
        "rope_cos_cache".to_string(),
        TensorData::new(TensorDataValue::F32(cos_vals), rope_shape.clone())?,
    );
    let sin_cache: Arc<dyn Tensor> = InputTensorInitialized::new(
        "rope_sin_cache".to_string(),
        TensorData::new(TensorDataValue::F32(sin_vals), rope_shape)?,
    );
    // Cast RoPE caches to model dtype
    let cos_cache = cast(cos_cache, model_dtype);
    let sin_cache = cast(sin_cache, model_dtype);

    // --- Embeddings ---
    // Patchify latent and project
    let img = patchify(latent_input.clone(), &config)?;
    let img = linear(&weight_manager.prefix("img_in"), img)?;

    // Project T5 hidden states
    let txt = linear(&weight_manager.prefix("txt_in"), t5_hidden_input.clone())?;

    // Timestep embedding
    let t_emb = flux_timestep_embedding(
        &weight_manager.prefix("time_in"),
        timestep_input.clone(),
        &config,
        model_dtype,
    )?;

    // CLIP pooled embedding
    let vec_emb = linear(
        &weight_manager.prefix("vector_in.in_layer"),
        clip_pooled_input.clone(),
    )?;
    let vec_emb = silu(vec_emb)?;
    let vec_emb = linear(&weight_manager.prefix("vector_in.out_layer"), vec_emb)?;

    // Guidance embedding (Dev only)
    let guidance_emb: Option<Arc<dyn Tensor>> = if let Some(ref gi) = guidance_input {
        Some(flux_timestep_embedding(
            &weight_manager.prefix("guidance_in"),
            gi.clone(),
            &config,
            model_dtype,
        )?)
    } else {
        None
    };

    // Conditioning vector
    let mut vec_cond: Arc<dyn Tensor> = Add::new(None, t_emb, vec_emb)?;
    if let Some(g_emb) = guidance_emb {
        vec_cond = Add::new(None, vec_cond, g_emb)?;
    }

    println!(
        "Building Flux DiT: {} double + {} single blocks...",
        config.num_double_blocks, config.num_single_blocks
    );

    // --- Double blocks ---
    let mut img: Arc<dyn Tensor> = img;
    let mut txt: Arc<dyn Tensor> = txt;
    for i in 0..config.num_double_blocks {
        let block_wm = weight_manager.prefix(&format!("double_blocks.{i}"));
        let result = flux_double_block(
            &block_wm,
            img,
            txt,
            vec_cond.clone(),
            cos_cache.clone(),
            sin_cache.clone(),
            &config,
        )?;
        img = result.0;
        txt = result.1;
        if (i + 1) % 5 == 0 {
            println!("  Double block {}/{}", i + 1, config.num_double_blocks);
        }
    }

    // --- Concatenate txt + img for single blocks ---
    let mut x: Arc<dyn Tensor> = Concat::new(None, vec![txt, img], 1)?;

    // --- Single blocks ---
    for i in 0..config.num_single_blocks {
        let block_wm = weight_manager.prefix(&format!("single_blocks.{i}"));
        x = flux_single_block(
            &block_wm,
            x,
            vec_cond.clone(),
            cos_cache.clone(),
            sin_cache.clone(),
            &config,
        )?;
        if (i + 1) % 10 == 0 {
            println!("  Single block {}/{}", i + 1, config.num_single_blocks);
        }
    }

    // --- Extract image portion ---
    let txt_seq = config.max_txt_seq_len as i64;
    let total_seq = config.total_seq_len() as i64;
    let img_out = slice_axis(x, 1, txt_seq, total_seq)?;

    // --- Final layer ---
    let final_wm = weight_manager.prefix("final_layer");
    let vec_act = silu(vec_cond)?;
    let final_mod = linear(&final_wm.prefix("adaLN_modulation.1"), vec_act)?;
    let final_mod = unsqueeze(final_mod, 1)?; // [B, 1, 2*hidden]
    let final_shift = slice_axis(final_mod.clone(), -1, 0, config.hidden_dim as i64)?;
    let final_scale = slice_axis(
        final_mod,
        -1,
        config.hidden_dim as i64,
        2 * config.hidden_dim as i64,
    )?;
    let img_out = adaln_modulate(img_out, final_shift, final_scale, config.hidden_dim)?;
    let img_out = linear(&final_wm.prefix("linear"), img_out)?;

    // --- Unpatchify ---
    let output = unpatchify(img_out, &config)?;

    let output_tensors: Vec<(String, Arc<dyn Tensor>)> = vec![("out_sample".to_string(), output)];

    println!("Built Flux DiT graph, exporting...");
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

// ============================================================================
// CLIP-L encoder for Flux (outputs pooled representation)
// ============================================================================

const CLIP_NUM_LAYERS: usize = 12;
const CLIP_MAX_POSITION: usize = 77;

/// Build a CLIP-L text encoder that outputs the pooled representation.
///
/// The standalone CLIP-L file has weights with prefix `text_model.`.
/// This outputs `pooled_output` [1, 768] — the final-layer-normed hidden state
/// at the EOS token position.
///
/// Inputs: `input_ids` [1, 77] I32, `eos_indices` [1] I64
/// Outputs: `pooled_output` [1, 768]
pub fn build_clip_l_pooled(
    weight_manager: impl WeightManager,
    output_method: WeightStorageStrategy,
    origin_path: Option<&std::path::Path>,
) -> Result<Vec<u8>, anyhow::Error> {
    let wm = CastingWeightManager::new(weight_manager.prefix("text_model"), DType::F32);
    let emb_wm = wm.prefix("embeddings");

    let batch_dim = Dimension::new(Some(1), Some("batch_size".to_string()), None);
    let seq_dim = Dimension::new(Some(CLIP_MAX_POSITION), Some("seq_len".to_string()), None);
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

    // Transformer layers
    let mut hidden: Arc<dyn Tensor> = x;
    for i in 0..CLIP_NUM_LAYERS {
        let layer_wm = wm.prefix(&format!("encoder.layers.{i}"));
        hidden = crate::sd15::clip_encoder_layer(&layer_wm, hidden, causal_mask.clone())?;
    }

    // Final layer norm
    let final_normed = layer_norm(&wm.prefix("final_layer_norm"), hidden, 1e-5)?;

    // Extract EOS token: Gather along seq dim (axis=1) using eos_indices
    let eos_idx_2d = unsqueeze(eos_indices.clone(), 1)?;
    let eos_token = Gather::new(None, final_normed, eos_idx_2d, 1)?;
    let pooled_output = reshape(eos_token, vec![0, -1])?;

    let input_tensors: Vec<Arc<dyn Tensor>> = vec![input_ids, eos_indices];
    let output_tensors: Vec<(String, Arc<dyn Tensor>)> =
        vec![("pooled_output".to_string(), pooled_output)];

    println!("Built CLIP-L pooled encoder, exporting...");
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
