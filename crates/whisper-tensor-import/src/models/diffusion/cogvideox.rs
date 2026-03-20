use crate::models::diffusion::sd_common::{
    self, CastingWeightManager, adaln_modulate, cos_op, layer_norm_bare, ones_constant, sin_op,
    slice_axis, split_chunks,
};
use crate::onnx_graph::Error;
use crate::onnx_graph::WeightStorageStrategy;
use crate::onnx_graph::operators::{
    Add, Concat, Constant, Conv, LayerNormalization, MatMul, Mul, Resize, RotaryEmbedding,
    Softmax, Transpose,
};
use crate::onnx_graph::pytorch::{
    cast, conv2d, div_scalar, gelu_pytorch_tanh, group_norm, linear, reshape, silu, unsqueeze,
};
use crate::onnx_graph::tensor::{
    DType, Dimension, InputTensor, InputTensorInitialized, Shape, Tensor, TensorData,
    TensorDataValue,
};
use crate::onnx_graph::weights::WeightManager;
use prost::Message;
use std::path::Path;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct CogVideoXTransformerConfig {
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub attention_head_dim: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub text_embed_dim: usize,
    pub time_embed_dim: usize,
    pub patch_size: usize,
    pub sample_frames: usize,
    pub sample_height: usize,
    pub sample_width: usize,
    pub max_text_seq_length: usize,
    pub use_rotary_positional_embeddings: bool,
    pub norm_eps: f32,
}

impl CogVideoXTransformerConfig {
    pub fn inner_dim(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }

    pub fn latent_frames(&self) -> usize {
        // Temporal compression ratio of 4x: (frames - 1) / 4 + 1
        (self.sample_frames - 1) / 4 + 1
    }

    pub fn patch_h(&self) -> usize {
        self.sample_height / self.patch_size
    }

    pub fn patch_w(&self) -> usize {
        self.sample_width / self.patch_size
    }

    pub fn video_seq_len(&self) -> usize {
        self.latent_frames() * self.patch_h() * self.patch_w()
    }

    pub fn cogvideox_2b() -> Self {
        Self {
            num_layers: 30,
            num_attention_heads: 30,
            attention_head_dim: 64,
            in_channels: 16,
            out_channels: 16,
            text_embed_dim: 4096,
            time_embed_dim: 512,
            patch_size: 2,
            sample_frames: 49,
            sample_height: 60,
            sample_width: 90,
            max_text_seq_length: 226,
            use_rotary_positional_embeddings: false,
            norm_eps: 1e-5,
        }
    }

    pub fn cogvideox_5b() -> Self {
        Self {
            num_layers: 42,
            num_attention_heads: 48,
            attention_head_dim: 64,
            in_channels: 16,
            out_channels: 16,
            text_embed_dim: 4096,
            time_embed_dim: 512,
            patch_size: 2,
            sample_frames: 49,
            sample_height: 60,
            sample_width: 90,
            max_text_seq_length: 226,
            use_rotary_positional_embeddings: true,
            norm_eps: 1e-5,
        }
    }

    /// CogVideoX1.5-5B: same architecture as 5B, longer videos, higher resolution.
    /// Uses pure 3D RoPE (no learnable positional embeddings).
    /// Supports 81 frames (10 sec) at up to 1360x768.
    pub fn cogvideox_1_5_5b() -> Self {
        Self {
            num_layers: 42,
            num_attention_heads: 48,
            attention_head_dim: 64,
            in_channels: 16,
            out_channels: 16,
            text_embed_dim: 4096,
            time_embed_dim: 512,
            patch_size: 2,
            sample_frames: 81,
            sample_height: 96,  // 768 / 8
            sample_width: 170,  // 1360 / 8
            max_text_seq_length: 226,
            use_rotary_positional_embeddings: true,
            norm_eps: 1e-5,
        }
    }
}

// Shared helpers (slice_axis, split_chunks, ones_constant, layer_norm_bare, adaln_modulate)
// are in sd_common.rs

// --- CogVideoX 3D RoPE ---

/// Precompute 3D axial RoPE cos/sin caches for the combined text+video sequence.
///
/// Returns (cos_cache, sin_cache) each of shape [total_seq_len, head_dim/2].
///
/// CogVideoX 5B splits head_dim=64 as:
///   temporal: 16 dims (1/4), height: 24 dims (3/8), width: 24 dims (3/8)
/// Half-dims for cos/sin: [8, 12, 12] = 32 total.
///
/// Text tokens get cos=1, sin=0 (identity rotation).
/// Video tokens get 3D positional encoding from (t, y, x) coordinates.
fn precompute_cogvideox_3d_rope(config: &CogVideoXTransformerConfig) -> (Vec<f32>, Vec<f32>) {
    let text_seq = config.max_text_seq_length;
    let video_seq = config.video_seq_len();
    let total_seq = text_seq + video_seq;
    let head_dim = config.attention_head_dim;
    let half_head = head_dim / 2; // 32

    // Axis dim allocation: temporal=head_dim/4, height=3*head_dim/8, width=3*head_dim/8
    let temporal_dim = head_dim / 4; // 16
    let spatial_dim = 3 * head_dim / 8; // 24
    let axes_dim = [temporal_dim, spatial_dim, spatial_dim]; // [16, 24, 24]
    let half_dims = [axes_dim[0] / 2, axes_dim[1] / 2, axes_dim[2] / 2]; // [8, 12, 12]

    let theta = 10000.0f64;

    // Precompute inverse frequencies for each axis
    let inv_freqs: Vec<Vec<f64>> = axes_dim
        .iter()
        .map(|&axis_dim| {
            (0..axis_dim / 2)
                .map(|i| 1.0 / theta.powf(2.0 * i as f64 / axis_dim as f64))
                .collect()
        })
        .collect();

    let _latent_frames = config.latent_frames();
    let patch_h = config.patch_h();
    let patch_w = config.patch_w();

    let mut cos_cache = vec![0.0f32; total_seq * half_head];
    let mut sin_cache = vec![0.0f32; total_seq * half_head];

    for pos in 0..total_seq {
        let (t_pos, y_pos, x_pos) = if pos < text_seq {
            // Text tokens: identity (all zeros -> cos=1, sin=0)
            (0.0f64, 0.0, 0.0)
        } else {
            // Video tokens: 3D grid position
            let vid_idx = pos - text_seq;
            let t = (vid_idx / (patch_h * patch_w)) as f64;
            let spatial_idx = vid_idx % (patch_h * patch_w);
            let y = (spatial_idx / patch_w) as f64;
            let x = (spatial_idx % patch_w) as f64;
            (t, y, x)
        };

        let positions = [t_pos, y_pos, x_pos];
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

// --- CogVideoX Timestep Embedding ---

fn cogvideox_timestep_embedding(
    wm: &impl WeightManager,
    timestep: Arc<dyn Tensor>,
    model_dtype: DType,
    inner_dim: usize,
) -> Result<Arc<dyn Tensor>, Error> {
    // Sinusoidal embedding: same pattern as SD3
    let timestep = cast(timestep, DType::F32);
    let half_dim = inner_dim / 2;
    let freqs: Vec<f32> = (0..half_dim)
        .map(|i| (-(10000.0f32.ln()) * i as f32 / half_dim as f32).exp())
        .collect();
    let freq_tensor = InputTensorInitialized::new(
        "cogvideox_timestep_freqs".to_string(),
        TensorData::new(
            freqs.into(),
            Shape::new(vec![
                Dimension::new(Some(1), None, None),
                Dimension::new(Some(half_dim), None, None),
            ]),
        )?,
    );
    let args = Mul::new(None, timestep, freq_tensor)?;
    let sin_part = sin_op(args.clone())?;
    let cos_part = cos_op(args)?;
    // CogVideoX uses flip_sin_to_cos=True, so cos first
    let emb = Concat::new(None, vec![cos_part, sin_part], -1)?;
    let emb = cast(emb, model_dtype);

    // TimestepEmbedding: Linear(inner_dim -> time_embed_dim) -> SiLU -> Linear(time_embed_dim -> time_embed_dim)
    let emb = linear(&wm.prefix("time_embedding.linear_1"), emb)?;
    let emb = silu(emb)?;
    linear(&wm.prefix("time_embedding.linear_2"), emb)
}

// --- CogVideoX Patch Embedding ---

fn cogvideox_patch_embed(
    wm: &impl WeightManager,
    hidden_states: Arc<dyn Tensor>,
    encoder_hidden_states: Arc<dyn Tensor>,
    config: &CogVideoXTransformerConfig,
) -> Result<(Arc<dyn Tensor>, Arc<dyn Tensor>), Error> {
    let inner_dim = config.inner_dim();

    // Project text embeddings: Linear(text_embed_dim -> inner_dim)
    let text_embeds = linear(&wm.prefix("patch_embed.text_proj"), encoder_hidden_states)?;

    // Patch-embed video: Conv2d(in_channels, inner_dim, kernel=patch_size, stride=patch_size)
    // Input: [B, latent_frames, C, H, W] -> reshape to [B*latent_frames, C, H, W]
    let latent_frames = config.latent_frames();
    let x = reshape(
        hidden_states,
        vec![
            -1,
            config.in_channels as i64,
            config.sample_height as i64,
            config.sample_width as i64,
        ],
    )?;
    let x = conv2d(
        &wm.prefix("patch_embed.proj"),
        x,
        config.patch_size as i64,
        config.patch_size as i64,
        0,
    )?;
    // Output: [B*F, inner_dim, H/p, W/p] -> [B, F, inner_dim, H/p * W/p] -> [B, F*H/p*W/p, inner_dim]
    let patch_h = config.patch_h();
    let patch_w = config.patch_w();
    let x = reshape(
        x,
        vec![
            -1,
            latent_frames as i64,
            inner_dim as i64,
            (patch_h * patch_w) as i64,
        ],
    )?;
    let x = Transpose::new(None, x, Some(vec![0, 1, 3, 2])); // [B, F, H/p*W/p, inner_dim]
    let video_embeds = reshape(x, vec![-1, config.video_seq_len() as i64, inner_dim as i64])?;

    Ok((text_embeds, video_embeds))
}

// --- CogVideoX Joint Attention ---

fn cogvideox_attention(
    wm: &impl WeightManager,
    hidden_states: Arc<dyn Tensor>,
    encoder_hidden_states: Arc<dyn Tensor>,
    config: &CogVideoXTransformerConfig,
    rope_cos: Option<Arc<dyn Tensor>>,
    rope_sin: Option<Arc<dyn Tensor>>,
) -> Result<(Arc<dyn Tensor>, Arc<dyn Tensor>), Error> {
    let nh = config.num_attention_heads as i64;
    let hd = config.attention_head_dim as i64;
    let text_seq = config.max_text_seq_length as i64;

    // Concatenate text + video tokens for joint attention
    let combined = Concat::new(None, vec![encoder_hidden_states, hidden_states], 1)?;

    // Q, K, V projections (single set operating on combined sequence)
    let q = linear(&wm.prefix("attn.to_q"), combined.clone())?;
    let k = linear(&wm.prefix("attn.to_k"), combined.clone())?;
    let v = linear(&wm.prefix("attn.to_v"), combined)?;

    // Reshape to multi-head: [B, seq, D] -> [B, seq, nh, hd] -> [B, nh, seq, hd]
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

    // QK normalization (LayerNorm per head, applied to the head_dim axis)
    let q: Arc<dyn Tensor> = layer_norm_bare(q, config.attention_head_dim, 1e-6)?;
    let k: Arc<dyn Tensor> = layer_norm_bare(k, config.attention_head_dim, 1e-6)?;

    // 3D RoPE for 5B variant
    let (q, k) = if let (Some(cos_cache), Some(sin_cache)) = (rope_cos, rope_sin) {
        // Apply RoPE to full sequence (text tokens have cos=1/sin=0 = identity)
        // interleaved=0: CogVideoX uses non-interleaved (half-split) rotation
        let q = RotaryEmbedding::new(
            None,
            q,
            cos_cache.clone(),
            sin_cache.clone(),
            None,
            Some(0),
            None,
            None,
        )? as Arc<dyn Tensor>;
        let k = RotaryEmbedding::new(None, k, cos_cache, sin_cache, None, Some(0), None, None)?
            as Arc<dyn Tensor>;
        (q, k)
    } else {
        (q, k)
    };

    // Scaled dot-product attention
    let scores = MatMul::new(None, q, Transpose::new(None, k, Some(vec![0, 1, 3, 2])))?;
    let scores = div_scalar(scores, (config.attention_head_dim as f32).sqrt())?;
    let attn = Softmax::new(None, scores, Some(-1));
    let attn_out = MatMul::new(None, attn, v)?;

    // Reshape back: [B, nh, seq, hd] -> [B, seq, D]
    let attn_out = Transpose::new(None, attn_out, Some(vec![0, 2, 1, 3]));
    let inner_dim = config.inner_dim() as i64;
    let attn_out = reshape(attn_out, vec![0, 0, inner_dim])?;

    // Output projection
    let attn_out = linear(&wm.prefix("attn.to_out.0"), attn_out)?;

    // Split back into text and video portions
    let total_len = attn_out.shape()[1].resolve()? as i64;
    let enc_out = slice_axis(attn_out.clone(), 1, 0, text_seq)?;
    let vid_out = slice_axis(attn_out, 1, text_seq, total_len)?;

    Ok((vid_out, enc_out))
}

// --- CogVideoX Feed-Forward ---

fn cogvideox_feed_forward(
    wm: &impl WeightManager,
    hidden_states: Arc<dyn Tensor>,
    encoder_hidden_states: Arc<dyn Tensor>,
) -> Result<(Arc<dyn Tensor>, Arc<dyn Tensor>), Error> {
    let text_seq = encoder_hidden_states.shape()[1].resolve()? as i64;

    // Joint FF: concatenate, project, split
    let combined = Concat::new(None, vec![encoder_hidden_states, hidden_states], 1)?;
    let x = linear(&wm.prefix("ff.net.0.proj"), combined)?;
    let x = gelu_pytorch_tanh(x)?;
    let x = linear(&wm.prefix("ff.net.2"), x)?;

    let total_len = x.shape()[1].resolve()? as i64;
    let enc_ff = slice_axis(x.clone(), 1, 0, text_seq)?;
    let vid_ff = slice_axis(x, 1, text_seq, total_len)?;

    Ok((vid_ff, enc_ff))
}

// --- CogVideoX Transformer Block ---

fn cogvideox_block(
    wm: &impl WeightManager,
    hidden_states: Arc<dyn Tensor>,
    encoder_hidden_states: Arc<dyn Tensor>,
    emb: Arc<dyn Tensor>,
    config: &CogVideoXTransformerConfig,
    rope_cos: Option<Arc<dyn Tensor>>,
    rope_sin: Option<Arc<dyn Tensor>>,
) -> Result<(Arc<dyn Tensor>, Arc<dyn Tensor>), Error> {
    let inner_dim = config.inner_dim();
    let eps = config.norm_eps;

    // CogVideoXLayerNormZero for attention
    // SiLU(emb) -> Linear(time_embed_dim -> 6 * inner_dim) -> chunk(6)
    let norm_act = silu(emb.clone())?;
    let norm_params = linear(&wm.prefix("norm1.linear"), norm_act)?;
    let norm_params = unsqueeze(norm_params, 1)?;
    let chunks = split_chunks(norm_params, inner_dim, 6)?;
    let (shift_h, scale_h, gate_h) = (chunks[0].clone(), chunks[1].clone(), chunks[2].clone());
    let (enc_shift, enc_scale, enc_gate) =
        (chunks[3].clone(), chunks[4].clone(), chunks[5].clone());

    // Apply AdaLN to both streams
    let norm_hidden = adaln_modulate(hidden_states.clone(), shift_h, scale_h, inner_dim, eps)?;
    let norm_enc = adaln_modulate(
        encoder_hidden_states.clone(),
        enc_shift,
        enc_scale,
        inner_dim,
        eps,
    )?;

    // Joint attention
    let (attn_vid, attn_enc) =
        cogvideox_attention(wm, norm_hidden, norm_enc, config, rope_cos, rope_sin)?;

    // Residual with gating
    let hidden_states = Add::new(
        None,
        hidden_states.clone(),
        Mul::new(None, gate_h, attn_vid)?,
    )?;
    let encoder_hidden_states = Add::new(
        None,
        encoder_hidden_states.clone(),
        Mul::new(None, enc_gate, attn_enc)?,
    )?;

    // CogVideoXLayerNormZero for feed-forward
    let norm_act2 = silu(emb)?;
    let norm_params2 = linear(&wm.prefix("norm2.linear"), norm_act2)?;
    let norm_params2 = unsqueeze(norm_params2, 1)?;
    let chunks2 = split_chunks(norm_params2, inner_dim, 6)?;
    let (shift_ff_h, scale_ff_h, gate_ff_h) =
        (chunks2[0].clone(), chunks2[1].clone(), chunks2[2].clone());
    let (enc_shift_ff, enc_scale_ff, enc_gate_ff) =
        (chunks2[3].clone(), chunks2[4].clone(), chunks2[5].clone());

    let norm_hidden_ff = adaln_modulate(
        hidden_states.clone(),
        shift_ff_h,
        scale_ff_h,
        inner_dim,
        eps,
    )?;
    let norm_enc_ff = adaln_modulate(
        encoder_hidden_states.clone(),
        enc_shift_ff,
        enc_scale_ff,
        inner_dim,
        eps,
    )?;

    // Joint feed-forward
    let (ff_vid, ff_enc) = cogvideox_feed_forward(wm, norm_hidden_ff, norm_enc_ff)?;

    // Residual with gating
    let hidden_states = Add::new(None, hidden_states, Mul::new(None, gate_ff_h, ff_vid)?)?;
    let encoder_hidden_states = Add::new(
        None,
        encoder_hidden_states,
        Mul::new(None, enc_gate_ff, ff_enc)?,
    )?;

    Ok((hidden_states, encoder_hidden_states))
}

// --- CogVideoX Unpatchify ---

fn cogvideox_unpatchify(
    input: Arc<dyn Tensor>,
    config: &CogVideoXTransformerConfig,
) -> Result<Arc<dyn Tensor>, Error> {
    let latent_frames = config.latent_frames() as i64;
    let patch_h = config.patch_h() as i64;
    let patch_w = config.patch_w() as i64;
    let p = config.patch_size as i64;
    let c = config.out_channels as i64;

    // [B, F*pH*pW, p*p*C] -> [B, F, pH, pW, C, p, p]
    let x = reshape(input, vec![0, latent_frames, patch_h, patch_w, c, p, p])?;
    // -> [B, F, C, pH, p, pW, p]
    let x = Transpose::new(None, x, Some(vec![0, 1, 4, 2, 5, 3, 6]));
    // -> [B, F, C, pH*p, pW*p] = [B, F, C, H, W]
    reshape(
        x,
        vec![
            0,
            latent_frames,
            c,
            config.sample_height as i64,
            config.sample_width as i64,
        ],
    )
    .map(|x| x as Arc<dyn Tensor>)
}

// --- Main Graph Builder ---

pub fn load_cogvideox_transformer(
    weight_manager: impl WeightManager,
    config: CogVideoXTransformerConfig,
    output_method: WeightStorageStrategy,
) -> Result<Vec<u8>, anyhow::Error> {
    load_cogvideox_transformer_with_origin(weight_manager, config, output_method, None)
}

pub fn load_cogvideox_transformer_with_origin(
    weight_manager: impl WeightManager,
    config: CogVideoXTransformerConfig,
    output_method: WeightStorageStrategy,
    origin_path: Option<&Path>,
) -> Result<Vec<u8>, anyhow::Error> {
    let inner_dim = config.inner_dim();
    let model_dtype = weight_manager
        .get_tensor("transformer_blocks.0.attn.to_q.weight")
        .map(|t| t.dtype())
        .unwrap_or(DType::BF16);
    let wm = CastingWeightManager::new(weight_manager, model_dtype);

    let batch_dim = Dimension::new(Some(1), Some("batch".to_string()), None);

    // Inputs
    let latent_input = InputTensor::new(
        "hidden_states".to_string(),
        model_dtype,
        Shape::new(vec![
            batch_dim.clone(),
            Dimension::new(Some(config.latent_frames()), None, None),
            Dimension::new(Some(config.in_channels), None, None),
            Dimension::new(Some(config.sample_height), None, None),
            Dimension::new(Some(config.sample_width), None, None),
        ]),
    );
    let encoder_hidden_states_input = InputTensor::new(
        "encoder_hidden_states".to_string(),
        model_dtype,
        Shape::new(vec![
            batch_dim.clone(),
            Dimension::new(Some(config.max_text_seq_length), None, None),
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

    // 1. Timestep embedding
    let emb = cogvideox_timestep_embedding(&wm, timestep_input, model_dtype, inner_dim)?;

    // 2. Patch embedding
    let (mut encoder_hidden_states, mut hidden_states) =
        cogvideox_patch_embed(&wm, latent_input, encoder_hidden_states_input, &config)?;

    // 3. Add positional embedding (2B: sinusoidal, added to concatenated text+video)
    if !config.use_rotary_positional_embeddings {
        // Load precomputed positional embedding
        let pos_embed = wm.get_tensor("patch_embed.pos_embedding")?;
        let text_pos = slice_axis(pos_embed.clone(), 1, 0, config.max_text_seq_length as i64)?;
        let video_pos = slice_axis(
            pos_embed,
            1,
            config.max_text_seq_length as i64,
            (config.max_text_seq_length + config.video_seq_len()) as i64,
        )?;
        encoder_hidden_states = Add::new(None, encoder_hidden_states, text_pos)?;
        hidden_states = Add::new(None, hidden_states, video_pos)?;
    }

    // 3b. Precompute 3D RoPE caches (5B only)
    let (rope_cos, rope_sin) = if config.use_rotary_positional_embeddings {
        let (cos_vals, sin_vals) = precompute_cogvideox_3d_rope(&config);
        let total_seq = config.max_text_seq_length + config.video_seq_len();
        let half_head = config.attention_head_dim / 2;
        let rope_shape = Shape::new(vec![
            Dimension::new(Some(total_seq), None, None),
            Dimension::new(Some(half_head), None, None),
        ]);
        let cos_cache: Arc<dyn Tensor> = InputTensorInitialized::new(
            "rope_cos_cache".to_string(),
            TensorData::new(TensorDataValue::F32(cos_vals), rope_shape.clone())?,
        );
        let sin_cache: Arc<dyn Tensor> = InputTensorInitialized::new(
            "rope_sin_cache".to_string(),
            TensorData::new(TensorDataValue::F32(sin_vals), rope_shape)?,
        );
        // Cast to model dtype
        let cos_cache = cast(cos_cache, model_dtype);
        let sin_cache = cast(sin_cache, model_dtype);
        (Some(cos_cache), Some(sin_cache))
    } else {
        (None, None)
    };

    // 4. Transformer blocks
    println!(
        "Building CogVideoX transformer: {} blocks, inner_dim={}...",
        config.num_layers, inner_dim
    );
    for i in 0..config.num_layers {
        let block_wm = wm.prefix(&format!("transformer_blocks.{i}"));
        let (next_hidden, next_enc) = cogvideox_block(
            &block_wm,
            hidden_states,
            encoder_hidden_states,
            emb.clone(),
            &config,
            rope_cos.clone(),
            rope_sin.clone(),
        )?;
        hidden_states = next_hidden;
        encoder_hidden_states = next_enc;
        if (i + 1) % 6 == 0 {
            println!("  Block {}/{}", i + 1, config.num_layers);
        }
    }

    // 5. Final norm + AdaLN + projection
    // norm_final: LayerNorm
    let hidden_states = layer_norm_bare(hidden_states, inner_dim, config.norm_eps)?;
    // AdaLayerNorm: SiLU(emb) -> Linear(time_embed_dim -> 2*inner_dim) -> chunk(2) -> shift, scale
    let norm_act = silu(emb)?;
    let norm_params = linear(&wm.prefix("norm_out.linear"), norm_act)?;
    let norm_params = unsqueeze(norm_params, 1)?;
    let shift = slice_axis(norm_params.clone(), -1, 0, inner_dim as i64)?;
    let scale = slice_axis(norm_params, -1, inner_dim as i64, (inner_dim * 2) as i64)?;
    let one = ones_constant(inner_dim, hidden_states.dtype());
    let scale_plus_one = Add::new(None, one, scale)?;
    let hidden_states = Mul::new(None, hidden_states, scale_plus_one)?;
    let hidden_states = Add::new(None, hidden_states, shift)?;

    // proj_out: Linear(inner_dim -> patch_size^2 * out_channels)
    let hidden_states = linear(&wm.prefix("proj_out"), hidden_states)?;

    // 6. Unpatchify
    let output = cogvideox_unpatchify(hidden_states, &config)?;

    let output_tensors: Vec<(String, Arc<dyn Tensor>)> = vec![("out_sample".to_string(), output)];

    println!("Built CogVideoX transformer graph, exporting...");
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
// CogVideoX 3D VAE Decoder
// =============================================================================

#[derive(Clone, Debug)]
pub struct CogVideoXVaeConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub latent_channels: usize,
    pub block_out_channels: Vec<usize>,
    pub layers_per_block: usize,
    pub norm_num_groups: usize,
    pub temporal_compression_ratio: usize,
    pub norm_eps: f32,
    pub scaling_factor: f32,
}

impl CogVideoXVaeConfig {
    pub fn cogvideox_2b() -> Self {
        Self {
            in_channels: 3,
            out_channels: 3,
            latent_channels: 16,
            block_out_channels: vec![128, 256, 256, 512],
            layers_per_block: 3,
            norm_num_groups: 32,
            temporal_compression_ratio: 4,
            norm_eps: 1e-6,
            scaling_factor: 1.15258426,
        }
    }

    pub fn cogvideox_5b() -> Self {
        let mut c = Self::cogvideox_2b();
        c.scaling_factor = 0.7;
        c
    }
}

// --- VAE Helpers ---

/// CausalConv3d: pad temporal dim by repeating the first frame, then Conv3d.
/// For kernel_t=3: prepend first frame twice, zero spatial padding of 1.
fn causal_conv3d(
    wm: &impl WeightManager,
    input: Arc<dyn Tensor>,
    kernel_t: i64,
    kernel_s: i64,
    stride_t: i64,
) -> Result<Arc<dyn Tensor>, Error> {
    let weight = wm.get_tensor("weight")?;
    let bias = wm.get_tensor("bias").ok();

    // Causal temporal padding: repeat first frame (kernel_t - 1) times
    let padded = if kernel_t > 1 {
        let time_pad = kernel_t - 1;
        // Slice first frame: [B, C, 1, H, W]
        let first_frame = slice_axis(input.clone(), 2, 0, 1)?;
        // Expand to [B, C, time_pad, H, W] by concatenating copies
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
        vec![stride_t, 1, 1],
        vec![0, spatial_pad, spatial_pad, 0, spatial_pad, spatial_pad],
        vec![1, 1, 1],
        1,
    )?;
    Ok(conv)
}

/// 1x1x1 Conv3d (no padding needed).
fn conv3d_1x1(wm: &impl WeightManager, input: Arc<dyn Tensor>) -> Result<Arc<dyn Tensor>, Error> {
    let weight = wm.get_tensor("weight")?;
    let bias = wm.get_tensor("bias").ok();
    let conv = Conv::new(
        wm.get_prefix().map(|x| x.to_string()),
        input,
        weight,
        bias,
        vec![1, 1, 1],
        vec![1, 1, 1],
        vec![0, 0, 0, 0, 0, 0],
        vec![1, 1, 1],
        1,
    )?;
    Ok(conv)
}

/// Nearest-neighbor 3D resize to target dimensions [T, H, W].
/// Input shape: [B, C, T_in, H_in, W_in]
fn resize_nearest_3d(
    input: Arc<dyn Tensor>,
    cur_t: usize,
    cur_h: usize,
    cur_w: usize,
    target_t: usize,
    target_h: usize,
    target_w: usize,
) -> Result<Arc<dyn Tensor>, Error> {
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
    Resize::new_with_scales(
        None,
        input,
        scales,
        "nearest".to_string(),
        Shape::new(output_dims),
    )
    .map(|x| x as Arc<dyn Tensor>)
}

/// SpatialNorm3D: GroupNorm conditioned on interpolated latent input.
/// output = GroupNorm(x) * conv_y(interpolate(zq)) + conv_b(interpolate(zq))
fn spatial_norm_3d(
    wm: &impl WeightManager,
    input: Arc<dyn Tensor>,
    zq: Arc<dyn Tensor>,
    num_groups: i64,
    epsilon: f32,
    zq_t: usize,
    zq_h: usize,
    zq_w: usize,
    target_t: usize,
    target_h: usize,
    target_w: usize,
) -> Result<Arc<dyn Tensor>, Error> {
    let normed = group_norm(&wm.prefix("norm_layer"), input, epsilon, num_groups)?;

    // Interpolate zq to match feature spatial dims
    let zq_interp = resize_nearest_3d(zq, zq_t, zq_h, zq_w, target_t, target_h, target_w)?;

    // conv_y and conv_b are CausalConv3d with kernel=1 (no causal padding needed)
    let scale = conv3d_1x1(&wm.prefix("conv_y"), zq_interp.clone())?;
    let bias = conv3d_1x1(&wm.prefix("conv_b"), zq_interp)?;

    let out = Mul::new(None, normed, scale)?;
    Ok(Add::new(None, out, bias)?)
}

/// VAE ResNet block with SpatialNorm3D.
fn vae_resnet_block(
    wm: &impl WeightManager,
    input: Arc<dyn Tensor>,
    zq: Arc<dyn Tensor>,
    in_channels: usize,
    out_channels: usize,
    num_groups: i64,
    eps: f32,
    zq_t: usize,
    zq_h: usize,
    zq_w: usize,
    target_t: usize,
    target_h: usize,
    target_w: usize,
) -> Result<Arc<dyn Tensor>, Error> {
    // SpatialNorm3D -> SiLU -> CausalConv3d(k=3)
    let h = spatial_norm_3d(
        &wm.prefix("norm1"),
        input.clone(),
        zq.clone(),
        num_groups,
        eps,
        zq_t,
        zq_h,
        zq_w,
        target_t,
        target_h,
        target_w,
    )?;
    let h = silu(h)?;
    let h = causal_conv3d(&wm.prefix("conv1"), h, 3, 3, 1)?;

    // SpatialNorm3D -> SiLU -> CausalConv3d(k=3)
    let h = spatial_norm_3d(
        &wm.prefix("norm2"),
        h,
        zq.clone(),
        num_groups,
        eps,
        zq_t,
        zq_h,
        zq_w,
        target_t,
        target_h,
        target_w,
    )?;
    let h = silu(h)?;
    let h = causal_conv3d(&wm.prefix("conv2"), h, 3, 3, 1)?;

    // Skip connection
    let residual = if in_channels != out_channels {
        conv3d_1x1(&wm.prefix("conv_shortcut"), input)?
    } else {
        input
    };

    Ok(Add::new(None, residual, h)?)
}

/// Upsample: spatial-only or spatial+temporal.
/// Spatial: F.interpolate(scale=2) per spatial dims + Conv2d(k=3,s=1,pad=1)
/// Temporal: also doubles the temporal dim.
fn vae_upsample(
    wm: &impl WeightManager,
    input: Arc<dyn Tensor>,
    _compress_time: bool,
    cur_t: usize,
    cur_h: usize,
    cur_w: usize,
    target_t: usize,
    target_h: usize,
    target_w: usize,
) -> Result<Arc<dyn Tensor>, Error> {
    // 3D nearest-neighbor resize to target dims
    let x = resize_nearest_3d(input, cur_t, cur_h, cur_w, target_t, target_h, target_w)?;

    // Apply Conv2d per-frame: reshape [B,C,T,H,W] -> [B*T,C,H,W], conv, reshape back
    let x = Transpose::new(None, x, Some(vec![0, 2, 1, 3, 4])); // [B,T,C,H,W]
    let x = reshape(x, vec![-1, 0, target_h as i64, target_w as i64])?; // [B*T, C, H, W]
    let x = conv2d(&wm.prefix("conv"), x, 3, 1, 1)?;
    let c_out = x.shape()[1].resolve()? as i64;
    let x = reshape(
        x,
        vec![-1, target_t as i64, c_out, target_h as i64, target_w as i64],
    )?;
    let x = Transpose::new(None, x, Some(vec![0, 2, 1, 3, 4])); // [B,C,T,H,W]

    Ok(x as Arc<dyn Tensor>)
}

/// Build the VAE decoder graph.
pub fn load_cogvideox_vae_decoder(
    weight_manager: impl WeightManager,
    config: CogVideoXVaeConfig,
    output_method: WeightStorageStrategy,
) -> Result<Vec<u8>, anyhow::Error> {
    load_cogvideox_vae_decoder_with_origin(weight_manager, config, output_method, None)
}

pub fn load_cogvideox_vae_decoder_with_origin(
    weight_manager: impl WeightManager,
    config: CogVideoXVaeConfig,
    output_method: WeightStorageStrategy,
    origin_path: Option<&Path>,
) -> Result<Vec<u8>, anyhow::Error> {
    let model_dtype = weight_manager
        .get_tensor("decoder.conv_in.weight")
        .map(|t| t.dtype())
        .unwrap_or(DType::F32);
    let wm = CastingWeightManager::new(weight_manager, model_dtype);
    let dec = wm.prefix("decoder");
    let ng = config.norm_num_groups as i64;
    let eps = config.norm_eps;

    // Reverse block channels for decoder: [128, 256, 256, 512] -> [512, 256, 256, 128]
    let rev_channels: Vec<usize> = config.block_out_channels.iter().copied().rev().collect();
    let num_up_blocks = rev_channels.len();

    // Which up blocks decompress time: last two (indices 2 and 3 for 4-block config)
    // In the encoder, blocks 0 and 1 compress time. In the decoder (reversed), blocks 2 and 3 decompress.
    // More precisely: compress_time[encoder_i] -> decompress at decoder block (num_blocks - 1 - i)
    let compress_time_flags: Vec<bool> = (0..num_up_blocks)
        .map(|i| {
            // Encoder block i compresses time if i < temporal_compression_ratio.log2() blocks
            // For ratio=4: blocks 0,1 compress. Decoder reverses: blocks 2,3 decompress.
            let encoder_idx = num_up_blocks - 1 - i;
            encoder_idx < 2 // First 2 encoder blocks compress time
        })
        .collect();

    // Compute spatial/temporal sizes at each decoder stage
    // Start from latent: T=13, H=60, W=90 (for 49 frames, 480x720)
    // We need the target sizes after each upsample.
    let base_h = 60usize; // sample_height in latent space
    let base_w = 90usize;
    let base_t = 13usize; // latent_frames

    // Track current spatial dims through the decoder
    let mut cur_t = base_t;
    let mut cur_h = base_h;
    let mut cur_w = base_w;

    let batch_dim = Dimension::new(Some(1), Some("batch".to_string()), None);
    let latent_input = InputTensor::new(
        "latent".to_string(),
        model_dtype,
        Shape::new(vec![
            batch_dim.clone(),
            Dimension::new(Some(config.latent_channels), None, None),
            Dimension::new(Some(base_t), None, None),
            Dimension::new(Some(base_h), None, None),
            Dimension::new(Some(base_w), None, None),
        ]),
    );
    let input_tensors: Vec<Arc<dyn Tensor>> = vec![latent_input.clone()];

    // conv_in: CausalConv3d(latent_channels -> last_block_channels, k=3)
    let last_block_ch = *rev_channels.first().unwrap(); // 512
    let mut x = causal_conv3d(&dec.prefix("conv_in"), latent_input.clone(), 3, 3, 1)?;

    // Mid block: 2 ResNet blocks with SpatialNorm3D
    let mid_ch = last_block_ch;
    for i in 0..2 {
        x = vae_resnet_block(
            &dec.prefix(&format!("mid_block.resnets.{i}")),
            x,
            latent_input.clone(),
            mid_ch,
            mid_ch,
            ng,
            eps,
            base_t,
            base_h,
            base_w,
            cur_t,
            cur_h,
            cur_w,
        )?;
    }

    println!(
        "Building CogVideoX VAE decoder: {} up blocks...",
        num_up_blocks
    );

    // Up blocks
    let mut current_channels = last_block_ch;
    for block_idx in 0..num_up_blocks {
        let out_ch = rev_channels[block_idx];
        let has_upsample = block_idx < num_up_blocks - 1; // Last block has no upsample
        let decompress_time = compress_time_flags[block_idx];

        // Decoder uses layers_per_block + 1 resnets per up block
        let num_resnets = config.layers_per_block + 1;
        for resnet_idx in 0..num_resnets {
            let in_ch = if resnet_idx == 0 {
                current_channels
            } else {
                out_ch
            };
            x = vae_resnet_block(
                &dec.prefix(&format!("up_blocks.{block_idx}.resnets.{resnet_idx}")),
                x,
                latent_input.clone(),
                in_ch,
                out_ch,
                ng,
                eps,
                base_t,
                base_h,
                base_w,
                cur_t,
                cur_h,
                cur_w,
            )?;
        }
        current_channels = out_ch;

        if has_upsample {
            let next_h = cur_h * 2;
            let next_w = cur_w * 2;
            let next_t = if decompress_time {
                // Temporal decompression: first frame stays, rest doubles
                // 13 -> 25: 1 + (13-1)*2 = 25
                (cur_t - 1) * 2 + 1
            } else {
                cur_t
            };

            x = vae_upsample(
                &dec.prefix(&format!("up_blocks.{block_idx}.upsamplers.0")),
                x,
                decompress_time,
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
            "  Up block {}/{}: {}ch, [T={}, H={}, W={}]",
            block_idx + 1,
            num_up_blocks,
            out_ch,
            cur_t,
            cur_h,
            cur_w
        );
    }

    // Final: SpatialNorm3D -> SiLU -> CausalConv3d(channels -> out_channels, k=3)
    let x = spatial_norm_3d(
        &dec.prefix("conv_norm_out"),
        x,
        latent_input.clone(),
        ng,
        eps,
        base_t,
        base_h,
        base_w,
        cur_t,
        cur_h,
        cur_w,
    )?;
    let x = silu(x)?;
    let output = causal_conv3d(&dec.prefix("conv_out"), x, 3, 3, 1)?;

    let output_tensors: Vec<(String, Arc<dyn Tensor>)> = vec![("video_out".to_string(), output)];

    println!("Built CogVideoX VAE decoder graph, exporting...");
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
