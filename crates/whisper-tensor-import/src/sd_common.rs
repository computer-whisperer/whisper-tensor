//! Shared building blocks for Stable Diffusion model graph builders (SD 1.5, SD 2.x).

use crate::onnx_graph::Error;
use crate::onnx_graph::WeightStorageStrategy;
use crate::onnx_graph::operators::{
    Add, Concat, Constant, Conv, MatMul, Mul, Reshape, ShapeOp, Softmax, Transpose,
};
use crate::onnx_graph::pytorch::{
    cast, conv2d, div_scalar, gelu, group_norm, layer_norm, linear, reshape, silu,
    upsample_nearest_2x,
};
use crate::onnx_graph::tensor::{
    DType, Dimension, InputTensor, InputTensorInitialized, Shape, Tensor, TensorData,
};
use crate::onnx_graph::weights::WeightManager;
use prost::Message;
use std::path::Path;
use std::sync::Arc;

// ============================================================================
// CastingWeightManager
// ============================================================================

/// Weight manager wrapper that casts all retrieved tensors to a target dtype.
pub struct CastingWeightManager<T: WeightManager> {
    inner: T,
    target_dtype: DType,
}

impl<T: WeightManager> CastingWeightManager<T> {
    pub fn new(inner: T, target_dtype: DType) -> Self {
        Self {
            inner,
            target_dtype,
        }
    }
}

impl<T: WeightManager> WeightManager for CastingWeightManager<T> {
    fn prefix(&self, name: &str) -> Self {
        Self {
            inner: self.inner.prefix(name),
            target_dtype: self.target_dtype,
        }
    }

    fn get_tensor(&self, name: &str) -> Result<Arc<dyn Tensor>, Error> {
        let tensor = self.inner.get_tensor(name)?;
        Ok(cast(tensor, self.target_dtype))
    }

    fn get_prefix_tail(&self) -> Option<&str> {
        self.inner.get_prefix_tail()
    }

    fn get_prefix(&self) -> Option<&str> {
        self.inner.get_prefix()
    }

    fn get_tensor_names(&self) -> Vec<String> {
        self.inner.get_tensor_names()
    }
}

// ============================================================================
// Dtype detection
// ============================================================================

/// Detect the storage dtype of an SD checkpoint by checking a canary weight.
pub fn detect_model_dtype(weight_manager: &impl WeightManager) -> DType {
    detect_model_dtype_with_canary(
        weight_manager,
        "model.diffusion_model.input_blocks.0.0.weight",
    )
}

/// Detect model dtype using a specific canary weight name.
pub fn detect_model_dtype_with_canary(
    weight_manager: &impl WeightManager,
    canary_name: &str,
) -> DType {
    let canary = weight_manager
        .get_tensor(canary_name)
        .unwrap_or_else(|e| {
            panic!("cannot detect model dtype: canary weight '{canary_name}' not found: {e}")
        });
    canary.dtype()
}

// ============================================================================
// Symbolic reshape
// ============================================================================

/// Reshape that supports symbolic (unresolved) dimensions.
///
/// `dims` uses the same convention as ONNX Reshape: 0 = copy from input, -1 = infer.
pub fn reshape_symbolic(input: Arc<dyn Tensor>, dims: Vec<i64>) -> Result<Arc<dyn Tensor>, Error> {
    let shape_const = Constant::new(
        None,
        TensorData::new(
            dims.clone().into(),
            Shape::new(vec![Dimension::new(Some(dims.len()), None, None)]),
        )?,
    );

    let mut output_dims = Vec::new();
    for (i, &d) in dims.iter().enumerate() {
        if d == 0 {
            output_dims.push(input.shape()[i].clone());
        } else if d == -1 {
            output_dims.push(Dimension::new(None, Some("inferred".to_string()), None));
        } else {
            output_dims.push(Dimension::new(Some(d as usize), None, None));
        }
    }
    let output_shape = Shape::new(output_dims);

    Ok(Reshape::new_with_forced_output(
        None,
        input,
        shape_const,
        output_shape,
    )?)
}

// ============================================================================
// Unary ops (Cos, Sin)
// ============================================================================

struct UnaryOp {
    name: Option<String>,
    input: Arc<dyn Tensor>,
    op_type: &'static str,
}

impl crate::onnx_graph::node::Node for UnaryOp {
    fn get_input_tensors(&self) -> Vec<&dyn Tensor> {
        vec![self.input.as_ref()]
    }
    fn get_output_tensors(&self) -> Vec<&dyn Tensor> {
        vec![self]
    }
    fn get_name(&self) -> Option<&str> {
        self.name.as_deref()
    }
    fn get_onnx_type(&self) -> &str {
        self.op_type
    }
    fn get_onnx_attributes(&self) -> Vec<crate::onnx_graph::onnx::AttributeProto> {
        vec![]
    }
}

impl crate::onnx_graph::node::SingleOutputNode for UnaryOp {
    fn get_output_shape(&self) -> &Shape {
        self.input.shape()
    }
    fn get_output_dtype(&self) -> DType {
        self.input.dtype()
    }
}

#[allow(clippy::arc_with_non_send_sync)]
pub fn cos_op(input: Arc<dyn Tensor>) -> Result<Arc<dyn Tensor>, Error> {
    Ok(Arc::new(UnaryOp {
        name: None,
        input,
        op_type: "Cos",
    }))
}

#[allow(clippy::arc_with_non_send_sync)]
pub fn sin_op(input: Arc<dyn Tensor>) -> Result<Arc<dyn Tensor>, Error> {
    Ok(Arc::new(UnaryOp {
        name: None,
        input,
        op_type: "Sin",
    }))
}

// ============================================================================
// Causal mask
// ============================================================================

pub fn build_causal_mask(size: usize) -> Vec<f32> {
    let mut mask = vec![0.0f32; size * size];
    for i in 0..size {
        for j in 0..size {
            if j > i {
                mask[i * size + j] = -1e9;
            }
        }
    }
    mask
}

// ============================================================================
// Timestep embedding
// ============================================================================

/// Build timestep embedding: sinusoidal basis (F32) → cast to model_dtype → MLP.
///
/// The sinusoidal computation runs in F32 for precision, then casts to model_dtype
/// before the Linear layers (whose weights are in model_dtype). The `timestep` input
/// should be F32.
pub fn timestep_embedding(
    wm: &impl WeightManager,
    timestep: Arc<dyn Tensor>,
    model_channels: usize,
    model_dtype: DType,
) -> Result<Arc<dyn Tensor>, Error> {
    let half_dim = model_channels / 2;
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

    // Sinusoidal embedding in F32
    let args = Mul::new(None, timestep, freq_tensor)?;
    let cos_part = cos_op(args.clone())?;
    let sin_part = sin_op(args)?;
    let emb = Concat::new(None, vec![cos_part, sin_part], -1)?;

    // Cast to model_dtype before MLP (weights are in model_dtype)
    let emb = cast(emb, model_dtype);

    let emb = linear(&wm.prefix("time_embed.0"), emb)?;
    let emb = silu(emb)?;
    let emb = linear(&wm.prefix("time_embed.2"), emb)?;
    Ok(emb)
}

// ============================================================================
// UNet building blocks
// ============================================================================

pub fn resnet_block(
    wm: &impl WeightManager,
    input: Arc<dyn Tensor>,
    temb: Arc<dyn Tensor>,
    in_channels: usize,
    out_channels: usize,
) -> Result<Arc<dyn Tensor>, Error> {
    // in_layers: GroupNorm -> SiLU -> Conv2d(3x3)
    let h = group_norm(&wm.prefix("in_layers.0"), input.clone(), 1e-5, 32)?;
    let h = silu(h)?;
    let h = conv2d(&wm.prefix("in_layers.2"), h, 3, 1, 1)?;

    // emb_layers: SiLU -> Linear
    let emb = silu(temb)?;
    let emb = linear(&wm.prefix("emb_layers.1"), emb)?;

    // Add timestep embedding (unsqueeze emb to [N, C, 1, 1] for broadcasting)
    let emb = reshape(emb, vec![0, -1, 1, 1])?;
    let h = Add::new(None, h, emb)?;

    // out_layers: GroupNorm -> SiLU -> Conv2d(3x3)
    let h = group_norm(&wm.prefix("out_layers.0"), h, 1e-5, 32)?;
    let h = silu(h)?;
    let h = conv2d(&wm.prefix("out_layers.3"), h, 3, 1, 1)?;

    // Skip connection
    let input = if in_channels != out_channels {
        conv2d(&wm.prefix("skip_connection"), input, 1, 1, 0)?
    } else {
        input
    };

    Ok(Add::new(None, input, h)?)
}

pub fn spatial_transformer(
    wm: &impl WeightManager,
    input: Arc<dyn Tensor>,
    context: Arc<dyn Tensor>,
    channels: usize,
    num_heads: usize,
    transformer_depth: usize,
    use_linear_proj: bool,
) -> Result<Arc<dyn Tensor>, Error> {
    let residual = input.clone();

    // GroupNorm
    let h = group_norm(&wm.prefix("norm"), input, 1e-5, 32)?;

    if use_linear_proj {
        // SD 2 style: reshape NCHW -> [N, H*W, C], then linear proj_in
        let nchw_shape = ShapeOp::new(None, h.clone(), None, None)?;
        let nchw_output_shape = h.shape().clone();
        let c = h.shape()[1].resolve()?;
        let h = reshape_symbolic(h, vec![0, c as i64, -1])?;
        let h = Transpose::new(None, h, Some(vec![0, 2, 1]));
        let h = linear(&wm.prefix("proj_in"), h)?;

        // Transformer blocks
        let mut h: Arc<dyn Tensor> = h;
        for tb in 0..transformer_depth {
            let tb_wm = wm.prefix(&format!("transformer_blocks.{tb}"));
            h = basic_transformer_block(&tb_wm, h, context.clone(), channels, num_heads)?;
        }

        let h = linear(&wm.prefix("proj_out"), h)?;
        // Reshape back [N, H*W, C] -> [N, C, H*W] -> [N, C, H, W]
        let h = Transpose::new(None, h, Some(vec![0, 2, 1]));
        let h = Reshape::new_with_forced_output(None, h, nchw_shape, nchw_output_shape)?;

        Ok(Add::new(None, residual, h)?)
    } else {
        // SD 1.5 style: conv2d(1x1) proj_in, then reshape
        let h = conv2d(&wm.prefix("proj_in"), h, 1, 1, 0)?;

        let nchw_shape = ShapeOp::new(None, h.clone(), None, None)?;
        let nchw_output_shape = h.shape().clone();
        let c = h.shape()[1].resolve()?;
        let h = reshape_symbolic(h, vec![0, c as i64, -1])?;
        let h = Transpose::new(None, h, Some(vec![0, 2, 1]));

        // Transformer blocks
        let mut h: Arc<dyn Tensor> = h;
        for tb in 0..transformer_depth {
            let tb_wm = wm.prefix(&format!("transformer_blocks.{tb}"));
            h = basic_transformer_block(&tb_wm, h, context.clone(), channels, num_heads)?;
        }

        // Reshape back
        let h = Transpose::new(None, h, Some(vec![0, 2, 1]));
        let h = Reshape::new_with_forced_output(None, h, nchw_shape, nchw_output_shape)?;
        let h = conv2d(&wm.prefix("proj_out"), h, 1, 1, 0)?;

        Ok(Add::new(None, residual, h)?)
    }
}

fn basic_transformer_block(
    wm: &impl WeightManager,
    input: Arc<dyn Tensor>,
    context: Arc<dyn Tensor>,
    channels: usize,
    num_heads: usize,
) -> Result<Arc<dyn Tensor>, Error> {
    let head_dim = channels / num_heads;

    // Self-attention
    let normed = layer_norm(&wm.prefix("norm1"), input.clone(), 1e-5)?;
    let attn1 = cross_attention(
        &wm.prefix("attn1"),
        normed.clone(),
        normed,
        num_heads,
        head_dim,
    )?;
    let h = Add::new(None, input, attn1)?;

    // Cross-attention
    let normed = layer_norm(&wm.prefix("norm2"), h.clone(), 1e-5)?;
    let attn2 = cross_attention(&wm.prefix("attn2"), normed, context, num_heads, head_dim)?;
    let h = Add::new(None, h, attn2)?;

    // Feed-forward (GEGLU)
    let normed = layer_norm(&wm.prefix("norm3"), h.clone(), 1e-5)?;
    let ff = feed_forward(&wm.prefix("ff"), normed, channels)?;
    Ok(Add::new(None, h, ff)?)
}

pub fn cross_attention(
    wm: &impl WeightManager,
    query: Arc<dyn Tensor>,
    context: Arc<dyn Tensor>,
    num_heads: usize,
    head_dim: usize,
) -> Result<Arc<dyn Tensor>, Error> {
    let q = linear_no_bias(&wm.prefix("to_q"), query)?;
    let k = linear_no_bias(&wm.prefix("to_k"), context.clone())?;
    let v = linear_no_bias(&wm.prefix("to_v"), context)?;

    // Reshape to multi-head
    let q = Transpose::new(
        None,
        reshape_symbolic(q, vec![0, -1, num_heads as i64, head_dim as i64])?,
        Some(vec![0, 2, 1, 3]),
    );
    let k = Transpose::new(
        None,
        reshape_symbolic(k, vec![0, -1, num_heads as i64, head_dim as i64])?,
        Some(vec![0, 2, 1, 3]),
    );
    let v = Transpose::new(
        None,
        reshape_symbolic(v, vec![0, -1, num_heads as i64, head_dim as i64])?,
        Some(vec![0, 2, 1, 3]),
    );

    // Attention
    let kt = Transpose::new(None, k, Some(vec![0, 1, 3, 2]));
    let scores = MatMul::new(None, q, kt)?;
    let scores = div_scalar(scores, (head_dim as f32).sqrt())?;
    let weights = Softmax::new(None, scores, Some(-1));
    let output = MatMul::new(None, weights, v)?;

    // Reshape back
    let output = Transpose::new(None, output, Some(vec![0, 2, 1, 3]));
    let hidden_dim = num_heads * head_dim;
    let output = reshape_symbolic(output, vec![0, -1, hidden_dim as i64])?;

    // Output projection (has bias)
    linear(&wm.prefix("to_out.0"), output)
}

pub fn linear_no_bias(
    wm: &impl WeightManager,
    input: Arc<dyn Tensor>,
) -> Result<Arc<dyn Tensor>, Error> {
    use crate::onnx_graph::operators;
    use crate::onnx_graph::pytorch::transpose;

    let weight = wm.get_tensor("weight")?;
    let weight = transpose(weight);
    let input_rank = input.rank();
    let input = crate::onnx_graph::pytorch::unsqueeze(input, (input_rank as i64) - 1)?;
    let mat_out = operators::MatMul::new(wm.get_prefix().map(|x| x.to_string()), input, weight)?;
    Ok(crate::onnx_graph::pytorch::squeeze(
        mat_out,
        (input_rank as i64) - 1,
    )?)
}

pub fn feed_forward(
    wm: &impl WeightManager,
    input: Arc<dyn Tensor>,
    channels: usize,
) -> Result<Arc<dyn Tensor>, Error> {
    // GEGLU: ff.net.0.proj produces 2x inner_dim, split into gate and value
    let inner_dim = channels * 4;
    let projected = linear(&wm.prefix("net.0.proj"), input)?;
    let value = slice_axis(projected.clone(), -1, 0, inner_dim as i64)?;
    let gate = slice_axis(projected, -1, inner_dim as i64, (inner_dim * 2) as i64)?;
    let gate = gelu(gate)?;
    let h = Mul::new(None, value, gate)?;
    linear(&wm.prefix("net.2"), h)
}

pub fn slice_axis(
    input: Arc<dyn Tensor>,
    axis: i64,
    start: i64,
    end: i64,
) -> Result<Arc<dyn Tensor>, Error> {
    use crate::onnx_graph::operators::{Constant, Slice};
    let resolved_axis = if axis < 0 {
        input.rank() as i64 + axis
    } else {
        axis
    };
    let const_shape = Shape::new(vec![Dimension::new(Some(1), None, None)]);
    let starts = Constant::new(
        None,
        TensorData::new(vec![start].into(), const_shape.clone())?,
    );
    let ends = Constant::new(
        None,
        TensorData::new(vec![end].into(), const_shape.clone())?,
    );
    let axes = Constant::new(
        None,
        TensorData::new(vec![resolved_axis].into(), const_shape)?,
    );
    Ok(Slice::new(None, input, starts, ends, Some(axes), None)?)
}

pub fn downsample(
    wm: &impl WeightManager,
    input: Arc<dyn Tensor>,
) -> Result<Arc<dyn Tensor>, Error> {
    let weight = wm.get_tensor("op.weight")?;
    let bias = wm.get_tensor("op.bias").ok();
    Conv::new(
        wm.get_prefix().map(|x| x.to_string()),
        input,
        weight,
        bias,
        vec![3, 3],
        vec![2, 2],
        vec![1, 1, 1, 1],
        vec![1, 1],
        1,
    )
    .map(|x| x as Arc<dyn Tensor>)
}

pub fn upsample(wm: &impl WeightManager, input: Arc<dyn Tensor>) -> Result<Arc<dyn Tensor>, Error> {
    let h = upsample_nearest_2x(input)?;
    conv2d(&wm.prefix("conv"), h, 3, 1, 1)
}

// ============================================================================
// VAE Decoder (shared between SD 1.5, SD 2.x, SDXL, and Flux)
// ============================================================================

const VAE_NUM_RES_BLOCKS: usize = 2;

/// Build a VAE decoder for SD 1.5/2.x/SDXL (4-channel latent, with post_quant_conv,
/// weights under `first_stage_model.*`).
pub fn build_vae_decoder(
    weight_manager: &impl WeightManager,
    model_dtype: DType,
    output_method: WeightStorageStrategy,
    origin_path: &Path,
) -> Result<Vec<u8>, anyhow::Error> {
    let wm = CastingWeightManager::new(weight_manager.prefix("first_stage_model"), DType::F32);
    build_vae_decoder_core(&wm, model_dtype, output_method, origin_path, 4, true)
}

/// Build a VAE decoder for Flux (16-channel latent, no post_quant_conv,
/// weights at root level `decoder.*`).
pub fn build_flux_vae_decoder(
    weight_manager: impl WeightManager,
    model_dtype: DType,
    output_method: WeightStorageStrategy,
    origin_path: &Path,
) -> Result<Vec<u8>, anyhow::Error> {
    let wm = CastingWeightManager::new(weight_manager, DType::F32);
    build_vae_decoder_core(&wm, model_dtype, output_method, origin_path, 16, false)
}

fn build_vae_decoder_core(
    wm: &impl WeightManager,
    model_dtype: DType,
    output_method: WeightStorageStrategy,
    origin_path: &Path,
    latent_channels: usize,
    has_post_quant_conv: bool,
) -> Result<Vec<u8>, anyhow::Error> {
    let batch_dim = Dimension::new(Some(1), Some("batch".to_string()), None);
    let h_dim = Dimension::new(None, Some("height".to_string()), None);
    let w_dim = Dimension::new(None, Some("width".to_string()), None);

    let latent_input = InputTensor::new(
        "latent_sample".to_string(),
        model_dtype,
        Shape::new(vec![
            batch_dim,
            Dimension::new(Some(latent_channels), None, None),
            h_dim,
            w_dim,
        ]),
    );

    let mut h: Arc<dyn Tensor> = cast(latent_input.clone(), DType::F32);

    if has_post_quant_conv {
        h = conv2d(&wm.prefix("post_quant_conv"), h, 1, 1, 0)?;
    }

    // Decoder
    let dec_wm = wm.prefix("decoder");
    h = conv2d(&dec_wm.prefix("conv_in"), h, 3, 1, 1)?;

    // mid block
    h = vae_resnet_block(&dec_wm.prefix("mid.block_1"), h)?;
    h = vae_attention(&dec_wm.prefix("mid.attn_1"), h)?;
    h = vae_resnet_block(&dec_wm.prefix("mid.block_2"), h)?;

    // up blocks (level 3, 2, 1, 0)
    for level in (0..4).rev() {
        for block in 0..(VAE_NUM_RES_BLOCKS + 1) {
            h = vae_resnet_block(&dec_wm.prefix(&format!("up.{level}.block.{block}")), h)?;
        }
        if level > 0 {
            h = vae_upsample(&dec_wm.prefix(&format!("up.{level}.upsample")), h)?;
        }
    }

    // norm_out + silu + conv_out
    h = group_norm(&dec_wm.prefix("norm_out"), h, 1e-6, 32)?;
    h = silu(h)?;
    h = conv2d(&dec_wm.prefix("conv_out"), h, 3, 1, 1)?;

    let input_tensors: Vec<Arc<dyn Tensor>> = vec![latent_input];
    let output_tensors: Vec<(String, Arc<dyn Tensor>)> = vec![("sample".to_string(), h)];

    let onnx_model = crate::onnx_graph::build_proto_with_origin_path(
        &input_tensors,
        &output_tensors,
        output_method,
        Some(origin_path),
    )?;
    Ok(onnx_model.encode_to_vec())
}

fn vae_resnet_block(
    wm: &impl WeightManager,
    input: Arc<dyn Tensor>,
) -> Result<Arc<dyn Tensor>, Error> {
    let in_ch = input.shape()[1].resolve()?;

    let h = group_norm(&wm.prefix("norm1"), input.clone(), 1e-6, 32)?;
    let h = silu(h)?;
    let h = conv2d(&wm.prefix("conv1"), h, 3, 1, 1)?;

    let out_ch = h.shape()[1].resolve()?;

    let h = group_norm(&wm.prefix("norm2"), h, 1e-6, 32)?;
    let h = silu(h)?;
    let h = conv2d(&wm.prefix("conv2"), h, 3, 1, 1)?;

    let input = if in_ch != out_ch {
        conv2d(&wm.prefix("nin_shortcut"), input, 1, 1, 0)?
    } else {
        input
    };

    Ok(Add::new(None, input, h)?)
}

fn vae_attention(
    wm: &impl WeightManager,
    input: Arc<dyn Tensor>,
) -> Result<Arc<dyn Tensor>, Error> {
    let residual = input.clone();
    let channels = input.shape()[1].resolve()?;

    let h = group_norm(&wm.prefix("norm"), input, 1e-6, 32)?;

    let q = conv2d(&wm.prefix("q"), h.clone(), 1, 1, 0)?;
    let k = conv2d(&wm.prefix("k"), h.clone(), 1, 1, 0)?;
    let v = conv2d(&wm.prefix("v"), h, 1, 1, 0)?;

    let q = reshape_symbolic(q, vec![0, channels as i64, -1])?;
    let q = Transpose::new(None, q, Some(vec![0, 2, 1]));
    let k = reshape_symbolic(k, vec![0, channels as i64, -1])?;
    let v = reshape_symbolic(v, vec![0, channels as i64, -1])?;
    let v = Transpose::new(None, v, Some(vec![0, 2, 1]));

    let scores = MatMul::new(None, q, k)?;
    let scores = div_scalar(scores, (channels as f32).sqrt())?;
    let weights = Softmax::new(None, scores, Some(-1));
    let output = MatMul::new(None, weights, v)?;

    let output = Transpose::new(None, output, Some(vec![0, 2, 1]));
    let nchw_shape = ShapeOp::new(None, residual.clone(), None, None)?;
    let nchw_output_shape = residual.shape().clone();
    let output = Reshape::new_with_forced_output(None, output, nchw_shape, nchw_output_shape)?;

    let output = conv2d(&wm.prefix("proj_out"), output, 1, 1, 0)?;

    Ok(Add::new(None, residual, output)?)
}

fn vae_upsample(wm: &impl WeightManager, input: Arc<dyn Tensor>) -> Result<Arc<dyn Tensor>, Error> {
    let h = upsample_nearest_2x(input)?;
    conv2d(&wm.prefix("conv"), h, 3, 1, 1)
}
