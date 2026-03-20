use crate::models::diffusion::sd_common::{
    CastingWeightManager, adaln_modulate, cos_op, sin_op, slice_axis, split_chunks,
};
use crate::onnx_graph::Error;
use crate::onnx_graph::WeightStorageStrategy;
use crate::onnx_graph::operators::{
    Add, Concat, Conv, MatMul, Mul, Resize, RotaryEmbedding, Softmax, Transpose,
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

// =============================================================================
// Config
// =============================================================================

#[derive(Clone, Debug)]
pub struct AllegroTransformerConfig {
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub attention_head_dim: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub caption_channels: usize,
    pub patch_size: usize,
    pub patch_size_t: usize,
    pub sample_height: usize,
    pub sample_width: usize,
    pub sample_frames: usize,
    pub interpolation_scale_h: f64,
    pub interpolation_scale_w: f64,
    pub interpolation_scale_t: f64,
    pub norm_eps: f32,
}

impl AllegroTransformerConfig {
    pub fn inner_dim(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }

    pub fn ffn_inner_dim(&self) -> usize {
        self.inner_dim() * 4
    }

    /// Dimensions per RoPE axis (equal thirds of head_dim).
    pub fn rope_dim_per_axis(&self) -> usize {
        self.attention_head_dim / 3
    }

    pub fn allegro_2_8b() -> Self {
        Self {
            num_layers: 32,
            num_attention_heads: 24,
            attention_head_dim: 96,
            in_channels: 4,
            out_channels: 4,
            caption_channels: 4096,
            patch_size: 2,
            patch_size_t: 1,
            sample_height: 90,
            sample_width: 160,
            sample_frames: 22,
            interpolation_scale_h: 2.0,
            interpolation_scale_w: 2.0,
            interpolation_scale_t: 2.2,
            norm_eps: 1e-6,
        }
    }
}

#[derive(Clone, Debug)]
pub struct AllegroVaeConfig {
    pub latent_channels: usize,
    pub out_channels: usize,
    pub block_out_channels: Vec<usize>,
    pub layers_per_block: usize,
    pub norm_num_groups: usize,
    pub temporal_compression_ratio: usize,
    pub temporal_upsample_blocks: Vec<bool>,
    pub norm_eps: f32,
    pub scaling_factor: f32,
}

impl AllegroVaeConfig {
    pub fn default_config() -> Self {
        Self {
            latent_channels: 4,
            out_channels: 3,
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            norm_num_groups: 32,
            temporal_compression_ratio: 4,
            temporal_upsample_blocks: vec![false, true, true, false],
            norm_eps: 1e-6,
            scaling_factor: 0.13,
        }
    }
}

// =============================================================================
// 3D RoPE for Allegro (per-axis, non-interleaved)
// =============================================================================

/// Precompute per-axis 3D RoPE cos/sin caches.
///
/// Allegro splits head_dim=96 into 3 equal chunks of 32. Each chunk uses
/// 16 frequencies with theta=10000, non-interleaved rotation.
/// Returns 3 pairs of (cos, sin) each [video_seq, 16].
fn precompute_allegro_3d_rope(
    config: &AllegroTransformerConfig,
    latent_frames: usize,
    patch_h: usize,
    patch_w: usize,
) -> Vec<(Vec<f32>, Vec<f32>)> {
    let rope_dim = config.rope_dim_per_axis(); // 32
    let half_rope_dim = rope_dim / 2; // 16
    let theta = 10000.0f64;
    let video_seq = latent_frames * patch_h * patch_w;

    let inv_freqs: Vec<f64> = (0..half_rope_dim)
        .map(|i| 1.0 / theta.powf(2.0 * i as f64 / rope_dim as f64))
        .collect();

    let axes = [
        (latent_frames, config.interpolation_scale_t),
        (patch_h, config.interpolation_scale_h),
        (patch_w, config.interpolation_scale_w),
    ];

    let mut result = Vec::with_capacity(3);

    for (axis_idx, &(_axis_len, scale)) in axes.iter().enumerate() {
        let mut cos_cache = vec![0.0f32; video_seq * half_rope_dim];
        let mut sin_cache = vec![0.0f32; video_seq * half_rope_dim];

        for vid_idx in 0..video_seq {
            let t = vid_idx / (patch_h * patch_w);
            let spatial_idx = vid_idx % (patch_h * patch_w);
            let h = spatial_idx / patch_w;
            let w = spatial_idx % patch_w;

            // Positions are divided by interpolation scale for resolution normalization
            let pos = match axis_idx {
                0 => t as f64 / scale,
                1 => h as f64 / scale,
                2 => w as f64 / scale,
                _ => unreachable!(),
            };

            for (i, &freq) in inv_freqs.iter().enumerate() {
                let angle = pos * freq;
                cos_cache[vid_idx * half_rope_dim + i] = angle.cos() as f32;
                sin_cache[vid_idx * half_rope_dim + i] = angle.sin() as f32;
            }
        }

        result.push((cos_cache, sin_cache));
    }

    result
}

// =============================================================================
// Allegro Transformer Components
// =============================================================================

/// Allegro timestep embedding (AdaLN-single).
///
/// Returns (temb_6x, embedded_timestep):
/// - temb_6x: [B, 1, 6 * inner_dim] — global modulation for all blocks
/// - embedded_timestep: [B, 1, inner_dim] — raw embedding for output norm
#[allow(clippy::type_complexity)]
fn allegro_timestep_embedding(
    wm: &impl WeightManager,
    timestep: Arc<dyn Tensor>,
    config: &AllegroTransformerConfig,
    model_dtype: DType,
) -> Result<(Arc<dyn Tensor>, Arc<dyn Tensor>), Error> {
    let _inner_dim = config.inner_dim();

    // Sinusoidal embedding (256 channels)
    let timestep = cast(timestep, DType::F32);
    let half_dim = 128usize;
    let freqs: Vec<f32> = (0..half_dim)
        .map(|i| (-(10000.0f32.ln()) * i as f32 / half_dim as f32).exp())
        .collect();
    let freq_tensor = InputTensorInitialized::new(
        "allegro_timestep_freqs".to_string(),
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

    let embedded_timestep = unsqueeze(temb.clone(), 1)?; // [B, 1, inner_dim]

    // Project to 6*inner_dim: SiLU -> Linear(inner_dim -> 6*inner_dim)
    let temb = silu(temb)?;
    let temb_6x = linear(&wm.prefix("adaln_single.linear"), temb)?;
    let temb_6x = unsqueeze(temb_6x, 1)?; // [B, 1, 6*inner_dim]

    Ok((temb_6x, embedded_timestep))
}

/// Allegro caption projection: T5-XXL 4096-dim -> inner_dim.
fn allegro_caption_projection(
    wm: &impl WeightManager,
    encoder_hidden_states: Arc<dyn Tensor>,
) -> Result<Arc<dyn Tensor>, Error> {
    let x = linear(
        &wm.prefix("caption_projection.linear_1"),
        encoder_hidden_states,
    )?;
    let x = gelu_pytorch_tanh(x)?;
    linear(&wm.prefix("caption_projection.linear_2"), x)
}

/// Allegro self-attention with per-axis 3D RoPE.
///
/// RoPE is applied by splitting Q/K into 3 chunks of 32 dims, applying
/// non-interleaved rotation per axis, then concatenating.
#[allow(clippy::type_complexity, clippy::needless_range_loop)]
fn allegro_self_attention(
    wm: &impl WeightManager,
    hidden_states: Arc<dyn Tensor>,
    config: &AllegroTransformerConfig,
    rope_caches: &[(Arc<dyn Tensor>, Arc<dyn Tensor>)],
) -> Result<Arc<dyn Tensor>, Error> {
    let nh = config.num_attention_heads as i64;
    let hd = config.attention_head_dim as i64;
    let inner_dim = config.inner_dim() as i64;
    let rope_dim = config.rope_dim_per_axis() as i64; // 32

    let q = linear(&wm.prefix("attn1.to_q"), hidden_states.clone())?;
    let k = linear(&wm.prefix("attn1.to_k"), hidden_states.clone())?;
    let v = linear(&wm.prefix("attn1.to_v"), hidden_states)?;

    // [B, seq, D] -> [B, nh, seq, hd]
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

    // Per-axis RoPE: split head_dim=96 into 3 chunks of 32, apply RoPE to each
    let mut q_parts = Vec::with_capacity(3);
    let mut k_parts = Vec::with_capacity(3);
    for axis in 0..3 {
        let start = (axis as i64) * rope_dim;
        let end = start + rope_dim;
        let q_chunk = slice_axis(q.clone(), 3, start, end)?;
        let k_chunk = slice_axis(k.clone(), 3, start, end)?;

        let (ref cos, ref sin) = rope_caches[axis];
        // Non-interleaved rotation (interleaved=None/0)
        let q_rotated = RotaryEmbedding::new(
            None,
            q_chunk,
            cos.clone(),
            sin.clone(),
            None,
            None,
            None,
            None,
        )?;
        let k_rotated = RotaryEmbedding::new(
            None,
            k_chunk,
            cos.clone(),
            sin.clone(),
            None,
            None,
            None,
            None,
        )?;
        q_parts.push(q_rotated as Arc<dyn Tensor>);
        k_parts.push(k_rotated as Arc<dyn Tensor>);
    }
    let q = Concat::new(None, q_parts, 3)?;
    let k = Concat::new(None, k_parts, 3)?;

    // Scaled dot-product attention
    let scores = MatMul::new(None, q, Transpose::new(None, k, Some(vec![0, 1, 3, 2])))?;
    let scores = div_scalar(scores, (config.attention_head_dim as f32).sqrt())?;
    let attn = Softmax::new(None, scores, Some(-1));
    let attn_out = MatMul::new(None, attn, v)?;

    let attn_out = Transpose::new(None, attn_out, Some(vec![0, 2, 1, 3]));
    let attn_out = reshape(attn_out, vec![0, 0, inner_dim])?;
    linear(&wm.prefix("attn1.to_out.0"), attn_out)
}

/// Allegro cross-attention (text -> video). No RoPE, no pre-norm on query.
fn allegro_cross_attention(
    wm: &impl WeightManager,
    hidden_states: Arc<dyn Tensor>,
    encoder_hidden_states: Arc<dyn Tensor>,
    config: &AllegroTransformerConfig,
) -> Result<Arc<dyn Tensor>, Error> {
    let nh = config.num_attention_heads as i64;
    let hd = config.attention_head_dim as i64;
    let inner_dim = config.inner_dim() as i64;

    // No pre-normalization on hidden_states for cross-attention
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

    let scores = MatMul::new(None, q, Transpose::new(None, k, Some(vec![0, 1, 3, 2])))?;
    let scores = div_scalar(scores, (config.attention_head_dim as f32).sqrt())?;
    let attn = Softmax::new(None, scores, Some(-1));
    let attn_out = MatMul::new(None, attn, v)?;

    let attn_out = Transpose::new(None, attn_out, Some(vec![0, 2, 1, 3]));
    let attn_out = reshape(attn_out, vec![0, 0, inner_dim])?;
    linear(&wm.prefix("attn2.to_out.0"), attn_out)
}

/// Allegro transformer block.
///
/// 1. AdaLN modulation from per-block scale_shift_table + global temb
/// 2. Self-attention with per-axis 3D RoPE + gating
/// 3. Cross-attention (ungated residual, no query pre-norm)
/// 4. FFN with AdaLN + gating
#[allow(clippy::type_complexity)]
fn allegro_block(
    wm: &impl WeightManager,
    hidden_states: Arc<dyn Tensor>,
    encoder_hidden_states: Arc<dyn Tensor>,
    temb_6x: Arc<dyn Tensor>,
    config: &AllegroTransformerConfig,
    rope_caches: &[(Arc<dyn Tensor>, Arc<dyn Tensor>)],
) -> Result<Arc<dyn Tensor>, Error> {
    let inner_dim = config.inner_dim();
    let eps = config.norm_eps;

    // Modulation: scale_shift_table [6, inner_dim] + temb_6x [B, 1, 6*inner_dim]
    let sst = wm.get_tensor("scale_shift_table")?;
    let sst = reshape(sst, vec![1, 1, (6 * inner_dim) as i64])?;
    let modulation = Add::new(None, sst, temb_6x)?;
    let chunks = split_chunks(modulation, inner_dim, 6)?;
    let (shift_msa, scale_msa, gate_msa) =
        (chunks[0].clone(), chunks[1].clone(), chunks[2].clone());
    let (shift_mlp, scale_mlp, gate_mlp) =
        (chunks[3].clone(), chunks[4].clone(), chunks[5].clone());

    // 1. Self-attention with AdaLN + RoPE + gating
    let normed = adaln_modulate(hidden_states.clone(), shift_msa, scale_msa, inner_dim, eps)?;
    let attn_out = allegro_self_attention(wm, normed, config, rope_caches)?;
    let hidden_states = Add::new(None, hidden_states, Mul::new(None, gate_msa, attn_out)?)?;

    // 2. Cross-attention (ungated residual, no pre-norm on query)
    let cross_out =
        allegro_cross_attention(wm, hidden_states.clone(), encoder_hidden_states, config)?;
    let hidden_states = Add::new(None, hidden_states, cross_out)?;

    // 3. FFN with AdaLN + gating (uses norm2, not norm3)
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

pub fn load_allegro_transformer(
    weight_manager: impl WeightManager,
    config: AllegroTransformerConfig,
    output_method: WeightStorageStrategy,
) -> Result<Vec<u8>, anyhow::Error> {
    load_allegro_transformer_with_origin(weight_manager, config, output_method, None)
}

pub fn load_allegro_transformer_with_origin(
    weight_manager: impl WeightManager,
    config: AllegroTransformerConfig,
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

    // Default: 88 frames, 720x1280
    // Latent: [B, 4, 22, 90, 160] (temporal_compression=4, spatial=8)
    let latent_frames = config.sample_frames;
    let latent_h = config.sample_height;
    let latent_w = config.sample_width;
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
            Dimension::new(Some(512), None, None), // T5 max seq_len
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

    // 1. Timestep embedding
    let (temb_6x, embedded_timestep) =
        allegro_timestep_embedding(&wm, timestep_input, &config, model_dtype)?;

    // 2. Caption projection
    let text_embeds = allegro_caption_projection(&wm, encoder_hidden_states_input)?;

    // 3. Patch embedding: Conv2d(4, inner_dim, k=2, s=2) per frame
    // [B, C, F, H, W] -> [B, F, C, H, W] -> [B*F, C, H, W]
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
    let x = conv2d(&wm.prefix("pos_embed.proj"), x, 2, 2, 0)?;
    // [B*F, inner_dim, pH, pW] -> [B*F, inner_dim, pH*pW] -> [B*F, pH*pW, inner_dim]
    let x = reshape(x, vec![0, inner_dim as i64, -1])?;
    let x = Transpose::new(None, x, Some(vec![0, 2, 1]));
    // -> [B, F*pH*pW, inner_dim]
    let hidden_states = reshape(x, vec![-1, video_seq as i64, inner_dim as i64])?;

    // 4. Precompute per-axis 3D RoPE caches
    let rope_data = precompute_allegro_3d_rope(&config, latent_frames, patch_h, patch_w);
    let half_rope_dim = config.rope_dim_per_axis() / 2;
    let mut rope_caches: Vec<(Arc<dyn Tensor>, Arc<dyn Tensor>)> = Vec::with_capacity(3);
    for (axis, (cos_vals, sin_vals)) in rope_data.into_iter().enumerate() {
        let rope_shape = Shape::new(vec![
            Dimension::new(Some(video_seq), None, None),
            Dimension::new(Some(half_rope_dim), None, None),
        ]);
        let cos: Arc<dyn Tensor> = InputTensorInitialized::new(
            format!("allegro_rope_cos_axis{axis}"),
            TensorData::new(TensorDataValue::F32(cos_vals), rope_shape.clone())?,
        );
        let sin: Arc<dyn Tensor> = InputTensorInitialized::new(
            format!("allegro_rope_sin_axis{axis}"),
            TensorData::new(TensorDataValue::F32(sin_vals), rope_shape)?,
        );
        rope_caches.push((cast(cos, model_dtype), cast(sin, model_dtype)));
    }

    // 5. Transformer blocks
    println!(
        "Building Allegro transformer: {} blocks, inner_dim={}...",
        config.num_layers, inner_dim
    );
    let mut hidden_states: Arc<dyn Tensor> = hidden_states;
    for i in 0..config.num_layers {
        let block_wm = wm.prefix(&format!("transformer_blocks.{i}"));
        hidden_states = allegro_block(
            &block_wm,
            hidden_states,
            text_embeds.clone(),
            temb_6x.clone(),
            &config,
            &rope_caches,
        )?;
        if (i + 1) % 8 == 0 {
            println!("  Block {}/{}", i + 1, config.num_layers);
        }
    }

    // 6. Output norm: scale_shift_table [2, inner_dim] + embedded_timestep
    let final_sst = wm.get_tensor("scale_shift_table")?;
    let final_sst = reshape(final_sst, vec![1, 2, inner_dim as i64])?;
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

    // 7. Unpatchify: [B, F*pH*pW, p*p*C] -> [B, C, F, H, W]
    let f = latent_frames as i64;
    let ph = patch_h as i64;
    let pw = patch_w as i64;
    let p = config.patch_size as i64;
    let c = config.out_channels as i64;
    let x = reshape(hidden_states, vec![0, f, ph, pw, 1, p, p, c])?;
    let x = Transpose::new(None, x, Some(vec![0, 7, 1, 4, 2, 5, 3, 6]));
    let output = reshape(x, vec![0, c, f, ph * p, pw * p])?;

    let output_tensors: Vec<(String, Arc<dyn Tensor>)> = vec![("out_sample".to_string(), output)];

    println!("Built Allegro transformer graph, exporting...");
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
// Allegro VAE Decoder
// =============================================================================

/// Allegro VAE uses a mixed 2D spatial + temporal architecture.
/// Spatial processing uses standard 2D convolutions per frame.
/// Temporal processing uses separate temporal convolution layers.
pub fn load_allegro_vae_decoder(
    weight_manager: impl WeightManager,
    config: AllegroVaeConfig,
    output_method: WeightStorageStrategy,
) -> Result<Vec<u8>, anyhow::Error> {
    load_allegro_vae_decoder_with_origin(weight_manager, config, output_method, None)
}

#[allow(clippy::needless_range_loop)]
pub fn load_allegro_vae_decoder_with_origin(
    weight_manager: impl WeightManager,
    config: AllegroVaeConfig,
    output_method: WeightStorageStrategy,
    origin_path: Option<&Path>,
) -> Result<Vec<u8>, anyhow::Error> {
    let model_dtype = weight_manager
        .get_tensor("decoder.conv_in.weight")
        .map(|t| t.dtype())
        .unwrap_or(DType::F32);
    let wm = CastingWeightManager::new(weight_manager, model_dtype);
    let eps = config.norm_eps;
    let ng = config.norm_num_groups as i64;

    // post_quant_conv: Conv2d(4, 4, 1)
    // Decoder processes per-frame spatially with temporal conv layers interspersed

    // For default 88 frames, 720x1280:
    // Latent: [B, 4, 22, 90, 160]
    let mut cur_t = 22usize;
    let mut cur_h = 90usize;
    let mut cur_w = 160usize;

    let rev_channels: Vec<usize> = config.block_out_channels.iter().copied().rev().collect();
    let last_ch = *rev_channels.first().unwrap(); // 512
    let num_stages = rev_channels.len(); // 4

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

    // post_quant_conv: Conv2d(4, 4, 1) applied per frame
    // [B, C, T, H, W] -> [B, T, C, H, W] -> [B*T, C, H, W]
    let x = Transpose::new(None, latent_input, Some(vec![0, 2, 1, 3, 4]));
    let x = reshape(
        x,
        vec![
            -1,
            config.latent_channels as i64,
            cur_h as i64,
            cur_w as i64,
        ],
    )?;
    let x = conv2d(&wm.prefix("post_quant_conv"), x, 1, 1, 0)?;
    // -> [B, T, C, H, W] -> [B, C, T, H, W]
    let x = reshape(
        x,
        vec![
            -1,
            cur_t as i64,
            config.latent_channels as i64,
            cur_h as i64,
            cur_w as i64,
        ],
    )?;
    let mut x: Arc<dyn Tensor> = Transpose::new(None, x, Some(vec![0, 2, 1, 3, 4]));

    // conv_in: Conv2d(4, 512, 3) per frame
    x = allegro_per_frame_conv2d(
        &wm.prefix("decoder.conv_in"),
        x,
        cur_t,
        cur_h,
        cur_w,
        3,
        1,
        1,
    )?;

    // Mid block: ResBlock -> Attention -> ResBlock (spatial only, per frame)
    x = allegro_vae_resnet_per_frame(
        &wm.prefix("decoder.mid_block.resnets.0"),
        x,
        last_ch,
        last_ch,
        eps,
        ng,
        cur_t,
        cur_h,
        cur_w,
    )?;
    // Attention: spatial-only per frame
    x = allegro_vae_spatial_attention(
        &wm.prefix("decoder.mid_block.attentions.0"),
        x,
        last_ch,
        ng,
        eps,
        cur_t,
        cur_h,
        cur_w,
    )?;
    x = allegro_vae_resnet_per_frame(
        &wm.prefix("decoder.mid_block.resnets.1"),
        x,
        last_ch,
        last_ch,
        eps,
        ng,
        cur_t,
        cur_h,
        cur_w,
    )?;

    println!("Building Allegro VAE decoder: {} up stages...", num_stages);

    // Up blocks
    let mut current_ch = last_ch;
    for stage in 0..num_stages {
        let out_ch = rev_channels[stage];
        let has_spatial_upsample = stage < num_stages - 1;
        let has_temporal_upsample = config.temporal_upsample_blocks[stage];

        for r in 0..(config.layers_per_block + 1) {
            let in_ch = if r == 0 { current_ch } else { out_ch };
            x = allegro_vae_resnet_per_frame(
                &wm.prefix(&format!("decoder.up_blocks.{stage}.resnets.{r}")),
                x,
                in_ch,
                out_ch,
                eps,
                ng,
                cur_t,
                cur_h,
                cur_w,
            )?;
        }
        current_ch = out_ch;

        if has_spatial_upsample {
            // Nearest-neighbor 2x spatial upsample + conv per frame
            x = allegro_vae_spatial_upsample(
                &wm.prefix(&format!("decoder.up_blocks.{stage}.upsamplers.0")),
                x,
                cur_t,
                cur_h,
                cur_w,
            )?;
            cur_h *= 2;
            cur_w *= 2;
        }

        if has_temporal_upsample {
            // Temporal upsample: simple 2x (Allegro uses non-causal temporal compression)
            x = allegro_vae_temporal_upsample(
                &wm.prefix(&format!("decoder.up_blocks.{stage}.temp_upsamplers.0")),
                x,
                out_ch,
                cur_t,
                cur_h,
                cur_w,
            )?;
            cur_t *= 2;
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

    // Final: GroupNorm -> SiLU -> Conv2d per frame
    x = allegro_per_frame_group_norm(
        &wm.prefix("decoder.conv_norm_out"),
        x,
        eps,
        ng,
        current_ch,
        cur_t,
        cur_h,
        cur_w,
    )?;
    x = silu(x)?;
    x = allegro_per_frame_conv2d(
        &wm.prefix("decoder.conv_out"),
        x,
        cur_t,
        cur_h,
        cur_w,
        3,
        1,
        1,
    )?;

    let output_tensors: Vec<(String, Arc<dyn Tensor>)> = vec![("video_out".to_string(), x)];

    println!("Built Allegro VAE decoder graph, exporting...");
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

/// Apply Conv2d per frame: [B, C, T, H, W] -> per-frame -> [B, C_out, T, H_out, W_out]
#[allow(clippy::too_many_arguments)]
fn allegro_per_frame_conv2d(
    wm: &impl WeightManager,
    input: Arc<dyn Tensor>,
    cur_t: usize,
    cur_h: usize,
    cur_w: usize,
    kernel: i64,
    stride: i64,
    padding: i64,
) -> Result<Arc<dyn Tensor>, Error> {
    // [B, C, T, H, W] -> [B, T, C, H, W] -> [B*T, C, H, W]
    let x = Transpose::new(None, input, Some(vec![0, 2, 1, 3, 4]));
    let channels = x.shape()[2].resolve()?;
    let x = reshape(x, vec![-1, channels as i64, cur_h as i64, cur_w as i64])?;
    let x = conv2d(wm, x, kernel, stride, padding)?;
    let out_ch = x.shape()[1].resolve()?;
    let out_h = x.shape()[2].resolve()?;
    let out_w = x.shape()[3].resolve()?;
    // [B*T, C_out, H_out, W_out] -> [B, T, C_out, H_out, W_out] -> [B, C_out, T, H_out, W_out]
    let x = reshape(
        x,
        vec![-1, cur_t as i64, out_ch as i64, out_h as i64, out_w as i64],
    )?;
    Ok(Transpose::new(None, x, Some(vec![0, 2, 1, 3, 4])) as Arc<dyn Tensor>)
}

/// GroupNorm per frame.
#[allow(clippy::too_many_arguments)]
fn allegro_per_frame_group_norm(
    wm: &impl WeightManager,
    input: Arc<dyn Tensor>,
    eps: f32,
    num_groups: i64,
    channels: usize,
    cur_t: usize,
    cur_h: usize,
    cur_w: usize,
) -> Result<Arc<dyn Tensor>, Error> {
    let x = Transpose::new(None, input, Some(vec![0, 2, 1, 3, 4]));
    let x = reshape(x, vec![-1, channels as i64, cur_h as i64, cur_w as i64])?;
    let x = group_norm(wm, x, eps, num_groups)?;
    let x = reshape(
        x,
        vec![
            -1,
            cur_t as i64,
            channels as i64,
            cur_h as i64,
            cur_w as i64,
        ],
    )?;
    Ok(Transpose::new(None, x, Some(vec![0, 2, 1, 3, 4])) as Arc<dyn Tensor>)
}

/// VAE ResNet block applied per frame.
#[allow(clippy::too_many_arguments)]
fn allegro_vae_resnet_per_frame(
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
    let h = allegro_per_frame_group_norm(
        &wm.prefix("norm1"),
        input.clone(),
        eps,
        num_groups,
        in_channels,
        cur_t,
        cur_h,
        cur_w,
    )?;
    let h = silu(h)?;
    let h = allegro_per_frame_conv2d(&wm.prefix("conv1"), h, cur_t, cur_h, cur_w, 3, 1, 1)?;

    let h = allegro_per_frame_group_norm(
        &wm.prefix("norm2"),
        h,
        eps,
        num_groups,
        out_channels,
        cur_t,
        cur_h,
        cur_w,
    )?;
    let h = silu(h)?;
    let h = allegro_per_frame_conv2d(&wm.prefix("conv2"), h, cur_t, cur_h, cur_w, 3, 1, 1)?;

    let residual = if in_channels != out_channels {
        allegro_per_frame_conv2d(
            &wm.prefix("conv_shortcut"),
            input,
            cur_t,
            cur_h,
            cur_w,
            1,
            1,
            0,
        )?
    } else {
        input
    };

    Ok(Add::new(None, residual, h)?)
}

/// Spatial-only attention per frame.
#[allow(clippy::too_many_arguments)]
fn allegro_vae_spatial_attention(
    wm: &impl WeightManager,
    input: Arc<dyn Tensor>,
    channels: usize,
    num_groups: i64,
    eps: f32,
    cur_t: usize,
    cur_h: usize,
    cur_w: usize,
) -> Result<Arc<dyn Tensor>, Error> {
    let residual = input.clone();
    let h = allegro_per_frame_group_norm(
        &wm.prefix("group_norm"),
        input,
        eps,
        num_groups,
        channels,
        cur_t,
        cur_h,
        cur_w,
    )?;

    // [B, C, T, H, W] -> [B, T, C, H, W] -> [B*T, C, H, W] -> [B*T, C, H*W] -> [B*T, H*W, C]
    let x = Transpose::new(None, h, Some(vec![0, 2, 1, 3, 4]));
    let x = reshape(x, vec![-1, channels as i64, cur_h as i64, cur_w as i64])?;
    let x = reshape(x, vec![0, channels as i64, -1])?;
    let x = Transpose::new(None, x, Some(vec![0, 2, 1]));

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
    let out = reshape(out, vec![0, -1, channels as i64])?;
    let out = linear(&wm.prefix("to_out.0"), out)?;

    // [B*T, H*W, C] -> [B*T, C, H*W] -> [B*T, C, H, W] -> [B, T, C, H, W] -> [B, C, T, H, W]
    let out = Transpose::new(None, out, Some(vec![0, 2, 1]));
    let out = reshape(out, vec![0, channels as i64, cur_h as i64, cur_w as i64])?;
    let out = reshape(
        out,
        vec![
            -1,
            cur_t as i64,
            channels as i64,
            cur_h as i64,
            cur_w as i64,
        ],
    )?;
    let out = Transpose::new(None, out, Some(vec![0, 2, 1, 3, 4]));

    Ok(Add::new(None, residual, out)?)
}

/// Spatial 2x upsample per frame: nearest-neighbor resize + Conv2d.
fn allegro_vae_spatial_upsample(
    wm: &impl WeightManager,
    input: Arc<dyn Tensor>,
    cur_t: usize,
    cur_h: usize,
    cur_w: usize,
) -> Result<Arc<dyn Tensor>, Error> {
    // [B, C, T, H, W] -> per-frame 2x upsample + conv
    let channels = input.shape()[1].resolve()?;
    let x = Transpose::new(None, input, Some(vec![0, 2, 1, 3, 4]));
    let x = reshape(x, vec![-1, channels as i64, cur_h as i64, cur_w as i64])?;
    let x = crate::onnx_graph::pytorch::upsample_nearest_2x(x)?;
    let x = conv2d(&wm.prefix("conv"), x, 3, 1, 1)?;
    let out_h = cur_h * 2;
    let out_w = cur_w * 2;
    let x = reshape(
        x,
        vec![
            -1,
            cur_t as i64,
            channels as i64,
            out_h as i64,
            out_w as i64,
        ],
    )?;
    Ok(Transpose::new(None, x, Some(vec![0, 2, 1, 3, 4])) as Arc<dyn Tensor>)
}

/// Temporal 2x upsample: nearest-neighbor temporal resize + temporal conv.
fn allegro_vae_temporal_upsample(
    wm: &impl WeightManager,
    input: Arc<dyn Tensor>,
    _channels: usize,
    cur_t: usize,
    cur_h: usize,
    cur_w: usize,
) -> Result<Arc<dyn Tensor>, Error> {
    let target_t = cur_t * 2;
    let scale_t = target_t as f32 / cur_t as f32;

    let scales = crate::onnx_graph::operators::Constant::new(
        None,
        TensorData::new(
            vec![1.0f32, 1.0, scale_t, 1.0, 1.0].into(),
            Shape::from(&[5usize][..]),
        )?,
    );
    let output_dims = vec![
        input.shape()[0].clone(),
        input.shape()[1].clone(),
        Dimension::new(Some(target_t), None, None),
        Dimension::new(Some(cur_h), None, None),
        Dimension::new(Some(cur_w), None, None),
    ];
    let x = Resize::new_with_scales(
        None,
        input,
        scales,
        "nearest".to_string(),
        Shape::new(output_dims),
    )?;

    // Temporal conv: Conv3d with temporal kernel
    let weight = wm.get_tensor("conv.weight")?;
    let bias = wm.get_tensor("conv.bias").ok();
    let conv = Conv::new(
        wm.get_prefix().map(|p| p.to_string()),
        x,
        weight,
        bias,
        vec![3, 1, 1],
        vec![1, 1, 1],
        vec![1, 0, 0, 1, 0, 0],
        vec![1, 1, 1],
        1,
    )?;
    Ok(conv as Arc<dyn Tensor>)
}
