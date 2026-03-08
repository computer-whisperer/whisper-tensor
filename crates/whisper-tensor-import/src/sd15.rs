use crate::onnx_graph::WeightStorageStrategy;
use crate::onnx_graph::operators::{
    Add, Concat, Constant, Conv, Gather, MatMul, Mul, Reshape, ShapeOp, Softmax, Transpose,
};
use crate::onnx_graph::pytorch::{
    cast, conv2d, div_scalar, gelu, group_norm, layer_norm, linear, reshape, silu,
    upsample_nearest_2x,
};
use crate::onnx_graph::tensor::{
    DType, Dimension, InputTensor, InputTensorInitialized, Shape, Tensor, TensorData,
};
use crate::onnx_graph::weights::{SafetensorsWeightManager, WeightManager};
use crate::onnx_graph::Error;
use memmap2::Mmap;
use prost::Message;
use std::path::Path;
use std::sync::Arc;

/// Weight manager wrapper that casts all retrieved tensors to a target dtype.
/// Used by the SD 1.5 builder to explicitly convert f16/bf16 weights to f32
/// for sections that compute in f32.
struct CastingWeightManager<T: WeightManager> {
    inner: T,
    target_dtype: DType,
}

impl<T: WeightManager> CastingWeightManager<T> {
    fn new(inner: T, target_dtype: DType) -> Self {
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

/// Detect the storage dtype of the model by checking a canary weight.
fn detect_model_dtype(weight_manager: &impl WeightManager) -> DType {
    // Use a UNet conv weight as canary — always present in SD 1.5 checkpoints.
    let canary = weight_manager
        .get_tensor("model.diffusion_model.input_blocks.0.0.weight")
        .expect("cannot detect model dtype: canary weight not found");
    canary.dtype()
}

/// ONNX model bytes for the three SD 1.5 sub-models.
pub type Sd15Models = (Vec<u8>, Vec<u8>, Vec<u8>);

/// Load a standard SD 1.5 checkpoint (.safetensors) and produce three ONNX models.
/// Returns (text_encoder_onnx, unet_onnx, vae_decoder_onnx).
pub fn load_sd15_checkpoint(
    checkpoint_path: &Path,
    output_method: WeightStorageStrategy,
) -> Result<Sd15Models, anyhow::Error> {
    let file = std::fs::File::open(checkpoint_path)?;
    let mmap = unsafe { Mmap::map(&file) }?;
    let weight_manager = SafetensorsWeightManager::new(vec![Arc::new(mmap)])?;

    let model_dtype = detect_model_dtype(&weight_manager);
    println!("Detected model dtype: {:?}", model_dtype);

    // Canonicalize the checkpoint path so external references resolve correctly.
    let origin_path = std::fs::canonicalize(checkpoint_path)
        .unwrap_or_else(|_| checkpoint_path.to_path_buf());

    println!("Building SD 1.5 text encoder...");
    let text_encoder = build_text_encoder(
        &weight_manager, model_dtype, output_method.clone(), &origin_path,
    )?;

    println!("Building SD 1.5 UNet...");
    let unet = build_unet(
        &weight_manager, model_dtype, output_method.clone(), &origin_path,
    )?;

    println!("Building SD 1.5 VAE decoder...");
    let vae_decoder = build_vae_decoder(
        &weight_manager, model_dtype, output_method, &origin_path,
    )?;

    Ok((text_encoder, unet, vae_decoder))
}

/// Detect if a safetensors file is an SD 1.5 checkpoint by checking for key prefixes.
pub fn is_sd15_checkpoint(weight_manager: &impl WeightManager) -> bool {
    weight_manager
        .get_tensor("model.diffusion_model.input_blocks.0.0.weight")
        .is_ok()
        && weight_manager
            .get_tensor("cond_stage_model.transformer.text_model.embeddings.token_embedding.weight")
            .is_ok()
        && weight_manager
            .get_tensor("first_stage_model.decoder.conv_in.weight")
            .is_ok()
}

/// Reshape that supports symbolic (unresolved) dimensions.
///
/// `dims` uses the same convention as ONNX Reshape: 0 = copy from input, -1 = infer.
/// Positive values are taken literally. The output shape is constructed symbolically
/// so that downstream operators can propagate symbolic dims.
fn reshape_symbolic(
    input: Arc<dyn Tensor>,
    dims: Vec<i64>,
) -> Result<Arc<dyn Tensor>, crate::onnx_graph::Error> {
    // Build the constant shape tensor (for ONNX)
    let shape_const = Constant::new(
        None,
        TensorData::new(
            dims.clone().into(),
            Shape::new(vec![Dimension::new(Some(dims.len()), None, None)]),
        )?,
    );

    // Build the output shape symbolically
    let mut output_dims = Vec::new();
    for (i, &d) in dims.iter().enumerate() {
        if d == 0 {
            output_dims.push(input.shape()[i].clone());
        } else if d == -1 {
            // Inferred dim — symbolic, name-only
            output_dims.push(Dimension::new(None, Some("inferred".to_string()), None));
        } else {
            output_dims.push(Dimension::new(Some(d as usize), None, None));
        }
    }
    let output_shape = Shape::new(output_dims);

    Ok(Reshape::new_with_forced_output(None, input, shape_const, output_shape)?)
}

// SD 1.5 UNet config
const MODEL_CHANNELS: usize = 320;
const CHANNEL_MULT: [usize; 4] = [1, 2, 4, 4];
const NUM_RES_BLOCKS: usize = 2;
const NUM_HEADS: usize = 8;
const CONTEXT_DIM: usize = 768;
const TRANSFORMER_DEPTH: usize = 1;

// CLIP text encoder config
const CLIP_HIDDEN_DIM: usize = 768;
const CLIP_NUM_HEADS: usize = 12;
const CLIP_NUM_LAYERS: usize = 12;
const CLIP_MAX_POSITION: usize = 77;

// VAE config
const VAE_NUM_RES_BLOCKS: usize = 2;

// ============================================================================
// CLIP Text Encoder
// ============================================================================

pub fn build_text_encoder(
    weight_manager: &impl WeightManager,
    model_dtype: DType,
    output_method: WeightStorageStrategy,
    origin_path: &Path,
) -> Result<Vec<u8>, anyhow::Error> {
    // Text encoder always computes in f32 for precision.
    let wm = CastingWeightManager::new(
        weight_manager.prefix("cond_stage_model.transformer.text_model"),
        DType::F32,
    );
    let emb_wm = wm.prefix("embeddings");
    let _ = model_dtype; // text encoder is always f32

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

    // Position embedding (precomputed [77, 768], just add)
    let pos_emb = emb_wm.get_tensor("position_embedding.weight")?;
    let x = Add::new(Some("pos_embed".to_string()), token_emb, pos_emb)?;

    // Build causal mask once
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
        hidden = clip_encoder_layer(&layer_wm, hidden, causal_mask.clone())?;
    }

    // Final layer norm
    let output = layer_norm(&wm.prefix("final_layer_norm"), hidden, 1e-5)?;

    let input_tensors: Vec<Arc<dyn Tensor>> = vec![input_ids];
    let output_tensors: Vec<(String, Arc<dyn Tensor>)> =
        vec![("last_hidden_state".to_string(), output)];

    let onnx_model =
        crate::onnx_graph::build_proto_with_origin_path(
            &input_tensors, &output_tensors, output_method, Some(origin_path),
        )?;
    Ok(onnx_model.encode_to_vec())
}

fn clip_encoder_layer(
    wm: &impl WeightManager,
    input: Arc<dyn Tensor>,
    causal_mask: Arc<dyn Tensor>,
) -> Result<Arc<dyn Tensor>, crate::onnx_graph::Error> {
    // Pre-norm self-attention
    let normed = layer_norm(&wm.prefix("layer_norm1"), input.clone(), 1e-5)?;
    let attn_out = clip_attention(&wm.prefix("self_attn"), normed, causal_mask)?;
    let x = Add::new(None, input, attn_out)?;

    // Pre-norm MLP
    let normed = layer_norm(&wm.prefix("layer_norm2"), x.clone(), 1e-5)?;
    let mlp_out = clip_mlp(&wm.prefix("mlp"), normed)?;
    Ok(Add::new(None, x, mlp_out)?)
}

fn clip_attention(
    wm: &impl WeightManager,
    input: Arc<dyn Tensor>,
    causal_mask: Arc<dyn Tensor>,
) -> Result<Arc<dyn Tensor>, crate::onnx_graph::Error> {
    let head_dim = CLIP_HIDDEN_DIM / CLIP_NUM_HEADS;

    let q = linear(&wm.prefix("q_proj"), input.clone())?;
    let k = linear(&wm.prefix("k_proj"), input.clone())?;
    let v = linear(&wm.prefix("v_proj"), input)?;

    // Reshape to multi-head: [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
    let q = Transpose::new(
        None,
        reshape(q, vec![0, -1, CLIP_NUM_HEADS as i64, head_dim as i64])?,
        Some(vec![0, 2, 1, 3]),
    );
    let k = Transpose::new(
        None,
        reshape(k, vec![0, -1, CLIP_NUM_HEADS as i64, head_dim as i64])?,
        Some(vec![0, 2, 1, 3]),
    );
    let v = Transpose::new(
        None,
        reshape(v, vec![0, -1, CLIP_NUM_HEADS as i64, head_dim as i64])?,
        Some(vec![0, 2, 1, 3]),
    );

    // Attention: softmax(Q @ K^T / sqrt(head_dim)) @ V
    let kt = Transpose::new(None, k, Some(vec![0, 1, 3, 2]));
    let scores = MatMul::new(None, q, kt)?;
    let scores = div_scalar(scores, (head_dim as f32).sqrt())?;

    // Apply causal mask
    let scores = Add::new(None, scores, causal_mask)?;

    let attn_weights = Softmax::new(None, scores, Some(-1));
    let attn_output = MatMul::new(None, attn_weights, v)?;

    // Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, hidden]
    let attn_output = Transpose::new(None, attn_output, Some(vec![0, 2, 1, 3]));
    let attn_output = reshape(attn_output, vec![0, -1, CLIP_HIDDEN_DIM as i64])?;

    // Output projection
    linear(&wm.prefix("out_proj"), attn_output)
}

fn build_causal_mask(size: usize) -> Vec<f32> {
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

fn clip_mlp(
    wm: &impl WeightManager,
    input: Arc<dyn Tensor>,
) -> Result<Arc<dyn Tensor>, crate::onnx_graph::Error> {
    let x = linear(&wm.prefix("fc1"), input)?;
    // CLIP uses quick_gelu
    let x = crate::onnx_graph::pytorch::quick_gelu(x)?;
    linear(&wm.prefix("fc2"), x)
}

// ============================================================================
// UNet
// ============================================================================

pub fn build_unet(
    weight_manager: &impl WeightManager,
    model_dtype: DType,
    output_method: WeightStorageStrategy,
    origin_path: &Path,
) -> Result<Vec<u8>, anyhow::Error> {
    // UNet computes in f32; cast weights from model_dtype as needed.
    let wm = CastingWeightManager::new(
        weight_manager.prefix("model.diffusion_model"),
        DType::F32,
    );

    let batch_dim = Dimension::new(Some(1), Some("batch".to_string()), None);
    let h_dim = Dimension::new(None, Some("height".to_string()), None);
    let w_dim = Dimension::new(None, Some("width".to_string()), None);

    // I/O uses model_dtype (f16 or f32 depending on checkpoint).
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
            batch_dim,
            Dimension::new(Some(CLIP_MAX_POSITION), None, None),
            Dimension::new(Some(CONTEXT_DIM), None, None),
        ]),
    );

    // Cast inputs to F32 for computation
    let sample = cast(sample_input.clone(), DType::F32);
    let context = context_input.clone();

    // Timestep embedding
    let t_emb = timestep_embedding(&wm, cast(timestep_input.clone(), DType::F32))?;

    // Input blocks
    let channels = CHANNEL_MULT.map(|m| m * MODEL_CHANNELS);
    // channels = [320, 640, 1280, 1280]

    // input_blocks.0: initial conv
    let h = conv2d(&wm.prefix("input_blocks.0.0"), sample, 3, 1, 1)?;

    // Track all skip connections for the decoder
    let mut skip_connections: Vec<Arc<dyn Tensor>> = vec![h.clone()];

    // Build input blocks
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
                h = spatial_transformer(
                    &wm.prefix(&format!("input_blocks.{input_block_idx}.1")),
                    h,
                    context.clone(),
                    ch,
                )?;
            }

            skip_connections.push(h.clone());
            input_block_idx += 1;
        }

        // Downsample (except at last level)
        if level < 3 {
            h = downsample(&wm.prefix(&format!("input_blocks.{input_block_idx}.0")), h)?;
            skip_connections.push(h.clone());
            input_block_idx += 1;
        }
    }

    // Middle block
    h = resnet_block(
        &wm.prefix("middle_block.0"),
        h,
        t_emb.clone(),
        current_ch,
        current_ch,
    )?;
    h = spatial_transformer(&wm.prefix("middle_block.1"), h, context.clone(), current_ch)?;
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
            // Concatenate skip connection
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

            // Attention at levels 0, 1, 2 (not 3)
            if level < 3 {
                h = spatial_transformer(
                    &wm.prefix(&format!("output_blocks.{output_block_idx}.1")),
                    h,
                    context.clone(),
                    ch,
                )?;
            }

            // Upsample at end of each level (except level 0), at last block
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

    // Output: GroupNorm -> SiLU -> Conv
    h = group_norm(&wm.prefix("out.0"), h, 1e-5, 32)?;
    h = silu(h)?;
    h = conv2d(&wm.prefix("out.2"), h, 3, 1, 1)?;

    // Cast output back to model dtype
    let output = cast(h, model_dtype);

    let input_tensors: Vec<Arc<dyn Tensor>> = vec![sample_input, timestep_input, context_input];
    let output_tensors: Vec<(String, Arc<dyn Tensor>)> = vec![("out_sample".to_string(), output)];

    let onnx_model =
        crate::onnx_graph::build_proto_with_origin_path(
            &input_tensors, &output_tensors, output_method, Some(origin_path),
        )?;
    Ok(onnx_model.encode_to_vec())
}

fn timestep_embedding(
    wm: &impl WeightManager,
    timestep: Arc<dyn Tensor>,
) -> Result<Arc<dyn Tensor>, crate::onnx_graph::Error> {
    // Sinusoidal timestep embedding -> MLP
    // The sinusoidal embedding is computed at runtime in the original code,
    // but here we need to build it as a graph operation.
    // timestep: [1] -> embedding: [1, model_channels] -> MLP -> [1, time_embed_dim]

    // Build sinusoidal embedding as graph ops
    let half_dim = MODEL_CHANNELS / 2;
    // freq[i] = exp(-ln(10000) * i / half_dim)
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

    // timestep * freqs -> [1, half_dim]
    let args = Mul::new(None, timestep, freq_tensor)?;

    // cos and sin
    let cos_part = cos_op(args.clone())?;
    let sin_part = sin_op(args)?;

    // Concat [cos, sin] -> [1, model_channels]
    let emb = Concat::new(None, vec![cos_part, sin_part], -1)?;

    // MLP: linear -> silu -> linear
    let emb = linear(&wm.prefix("time_embed.0"), emb)?;
    let emb = silu(emb)?;
    let emb = linear(&wm.prefix("time_embed.2"), emb)?;
    Ok(emb)
}

#[allow(clippy::arc_with_non_send_sync)]
fn cos_op(input: Arc<dyn Tensor>) -> Result<Arc<dyn Tensor>, crate::onnx_graph::Error> {
    Ok(Arc::new(UnaryOp {
        name: None,
        input,
        op_type: "Cos",
    }))
}

#[allow(clippy::arc_with_non_send_sync)]
fn sin_op(input: Arc<dyn Tensor>) -> Result<Arc<dyn Tensor>, crate::onnx_graph::Error> {
    Ok(Arc::new(UnaryOp {
        name: None,
        input,
        op_type: "Sin",
    }))
}

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

fn resnet_block(
    wm: &impl WeightManager,
    input: Arc<dyn Tensor>,
    temb: Arc<dyn Tensor>,
    in_channels: usize,
    out_channels: usize,
) -> Result<Arc<dyn Tensor>, crate::onnx_graph::Error> {
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

fn spatial_transformer(
    wm: &impl WeightManager,
    input: Arc<dyn Tensor>,
    context: Arc<dyn Tensor>,
    channels: usize,
) -> Result<Arc<dyn Tensor>, crate::onnx_graph::Error> {
    let residual = input.clone();

    // GroupNorm
    let h = group_norm(&wm.prefix("norm"), input, 1e-5, 32)?;

    // proj_in is a 1x1 conv
    let h = conv2d(&wm.prefix("proj_in"), h, 1, 1, 0)?;

    // Capture NCHW shape for restoring later (supports symbolic spatial dims)
    let nchw_shape = ShapeOp::new(None, h.clone(), None, None)?;
    let nchw_output_shape = h.shape().clone();

    // Reshape NCHW -> [N, C, H*W]
    let c = h.shape()[1].resolve()?; // channel dim is always concrete
    let h = reshape_symbolic(h, vec![0, c as i64, -1])?;
    // Transpose to [N, H*W, C]
    let h = Transpose::new(None, h, Some(vec![0, 2, 1]));

    // Transformer blocks
    let mut h: Arc<dyn Tensor> = h;
    for tb in 0..TRANSFORMER_DEPTH {
        let tb_wm = wm.prefix(&format!("transformer_blocks.{tb}"));
        h = basic_transformer_block(&tb_wm, h, context.clone(), channels)?;
    }

    // Transpose back [N, H*W, C] -> [N, C, H*W]
    let h = Transpose::new(None, h, Some(vec![0, 2, 1]));
    // Reshape to [N, C, H, W] using the captured NCHW shape
    let h = Reshape::new_with_forced_output(None, h, nchw_shape, nchw_output_shape)?;

    // proj_out (1x1 conv)
    let h = conv2d(&wm.prefix("proj_out"), h, 1, 1, 0)?;

    // Residual
    Ok(Add::new(None, residual, h)?)
}

fn basic_transformer_block(
    wm: &impl WeightManager,
    input: Arc<dyn Tensor>,
    context: Arc<dyn Tensor>,
    channels: usize,
) -> Result<Arc<dyn Tensor>, crate::onnx_graph::Error> {
    let num_heads = NUM_HEADS;
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

fn cross_attention(
    wm: &impl WeightManager,
    query: Arc<dyn Tensor>,
    context: Arc<dyn Tensor>,
    num_heads: usize,
    head_dim: usize,
) -> Result<Arc<dyn Tensor>, crate::onnx_graph::Error> {
    // Q from query, K and V from context
    // Note: SD 1.5 cross-attention uses no bias on Q/K/V projections
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

fn linear_no_bias(
    wm: &impl WeightManager,
    input: Arc<dyn Tensor>,
) -> Result<Arc<dyn Tensor>, crate::onnx_graph::Error> {
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

fn feed_forward(
    wm: &impl WeightManager,
    input: Arc<dyn Tensor>,
    channels: usize,
) -> Result<Arc<dyn Tensor>, crate::onnx_graph::Error> {
    // GEGLU: ff.net.0.proj produces 2x inner_dim, split into gate and value
    let inner_dim = channels * 4;
    let projected = linear(&wm.prefix("net.0.proj"), input)?;
    // GEGLU: first half passes through, second half gets GELU (the "gate")
    let value = slice_axis(projected.clone(), -1, 0, inner_dim as i64)?;
    let gate = slice_axis(projected, -1, inner_dim as i64, (inner_dim * 2) as i64)?;
    let gate = gelu(gate)?;
    let h = Mul::new(None, value, gate)?;
    // Output linear
    linear(&wm.prefix("net.2"), h)
}

fn slice_axis(
    input: Arc<dyn Tensor>,
    axis: i64,
    start: i64,
    end: i64,
) -> Result<Arc<dyn Tensor>, crate::onnx_graph::Error> {
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

fn downsample(
    wm: &impl WeightManager,
    input: Arc<dyn Tensor>,
) -> Result<Arc<dyn Tensor>, crate::onnx_graph::Error> {
    // op: Conv2d(channels, channels, 3, stride=2, padding=1)
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

fn upsample(
    wm: &impl WeightManager,
    input: Arc<dyn Tensor>,
) -> Result<Arc<dyn Tensor>, crate::onnx_graph::Error> {
    // Nearest-neighbor 2x upscale then conv
    let h = upsample_nearest_2x(input)?;
    conv2d(&wm.prefix("conv"), h, 3, 1, 1)
}

// ============================================================================
// VAE Decoder
// ============================================================================

pub fn build_vae_decoder(
    weight_manager: &impl WeightManager,
    model_dtype: DType,
    output_method: WeightStorageStrategy,
    origin_path: &Path,
) -> Result<Vec<u8>, anyhow::Error> {
    // VAE computes in f32 for precision; cast weights from model_dtype.
    let wm = CastingWeightManager::new(
        weight_manager.prefix("first_stage_model"),
        DType::F32,
    );

    let batch_dim = Dimension::new(Some(1), Some("batch".to_string()), None);
    let h_dim = Dimension::new(None, Some("height".to_string()), None);
    let w_dim = Dimension::new(None, Some("width".to_string()), None);

    // Input uses model_dtype
    let latent_input = InputTensor::new(
        "latent_sample".to_string(),
        model_dtype,
        Shape::new(vec![
            batch_dim,
            Dimension::new(Some(4), None, None),
            h_dim,
            w_dim,
        ]),
    );

    let h = cast(latent_input.clone(), DType::F32);

    // post_quant_conv (1x1)
    let h = conv2d(&wm.prefix("post_quant_conv"), h, 1, 1, 0)?;

    // Decoder
    let dec_wm = wm.prefix("decoder");

    // conv_in
    let h = conv2d(&dec_wm.prefix("conv_in"), h, 3, 1, 1)?;

    // mid block
    let h = vae_resnet_block(&dec_wm.prefix("mid.block_1"), h)?;
    let h = vae_attention(&dec_wm.prefix("mid.attn_1"), h)?;
    let h = vae_resnet_block(&dec_wm.prefix("mid.block_2"), h)?;

    // up blocks (level 3, 2, 1, 0 — from deepest to shallowest)
    // channel structure: level 3=512, level 2=512, level 1=256, level 0=128
    let mut h = h;

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

    let onnx_model =
        crate::onnx_graph::build_proto_with_origin_path(
            &input_tensors, &output_tensors, output_method, Some(origin_path),
        )?;
    Ok(onnx_model.encode_to_vec())
}

fn vae_resnet_block(
    wm: &impl WeightManager,
    input: Arc<dyn Tensor>,
) -> Result<Arc<dyn Tensor>, crate::onnx_graph::Error> {
    let in_ch = input.shape()[1].resolve()?;

    // norm1 -> silu -> conv1
    let h = group_norm(&wm.prefix("norm1"), input.clone(), 1e-6, 32)?;
    let h = silu(h)?;
    let h = conv2d(&wm.prefix("conv1"), h, 3, 1, 1)?;

    let out_ch = h.shape()[1].resolve()?;

    // norm2 -> silu -> conv2
    let h = group_norm(&wm.prefix("norm2"), h, 1e-6, 32)?;
    let h = silu(h)?;
    let h = conv2d(&wm.prefix("conv2"), h, 3, 1, 1)?;

    // Skip connection
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
) -> Result<Arc<dyn Tensor>, crate::onnx_graph::Error> {
    let residual = input.clone();
    let channels = input.shape()[1].resolve()?;

    // GroupNorm
    let h = group_norm(&wm.prefix("norm"), input, 1e-6, 32)?;

    // Q, K, V via 1x1 conv
    let q = conv2d(&wm.prefix("q"), h.clone(), 1, 1, 0)?;
    let k = conv2d(&wm.prefix("k"), h.clone(), 1, 1, 0)?;
    let v = conv2d(&wm.prefix("v"), h, 1, 1, 0)?;

    // Reshape [N, C, H, W] -> [N, C, H*W] -> [N, H*W, C]
    let q = reshape_symbolic(q, vec![0, channels as i64, -1])?;
    let q = Transpose::new(None, q, Some(vec![0, 2, 1]));
    let k = reshape_symbolic(k, vec![0, channels as i64, -1])?;
    // k stays as [N, C, H*W] for Q @ K^T
    let v = reshape_symbolic(v, vec![0, channels as i64, -1])?;
    let v = Transpose::new(None, v, Some(vec![0, 2, 1]));

    // Attention: softmax(Q @ K / sqrt(C)) @ V
    let scores = MatMul::new(None, q, k)?;
    let scores = div_scalar(scores, (channels as f32).sqrt())?;
    let weights = Softmax::new(None, scores, Some(-1));
    let output = MatMul::new(None, weights, v)?;

    // Reshape back: [N, H*W, C] -> [N, C, H*W] -> [N, C, H, W]
    let output = Transpose::new(None, output, Some(vec![0, 2, 1]));
    // Use ShapeOp on the residual to get the original NCHW shape dynamically
    let nchw_shape = ShapeOp::new(None, residual.clone(), None, None)?;
    let nchw_output_shape = residual.shape().clone();
    let output = Reshape::new_with_forced_output(None, output, nchw_shape, nchw_output_shape)?;

    // proj_out (1x1 conv)
    let output = conv2d(&wm.prefix("proj_out"), output, 1, 1, 0)?;

    Ok(Add::new(None, residual, output)?)
}

fn vae_upsample(
    wm: &impl WeightManager,
    input: Arc<dyn Tensor>,
) -> Result<Arc<dyn Tensor>, crate::onnx_graph::Error> {
    let h = upsample_nearest_2x(input)?;
    conv2d(&wm.prefix("conv"), h, 3, 1, 1)
}
