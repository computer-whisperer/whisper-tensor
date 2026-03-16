use crate::models::diffusion::sd_common::{
    self, CastingWeightManager, cos_op, sin_op, slice_axis as slice_axis_common,
};
use crate::onnx_graph::Error;
use crate::onnx_graph::WeightStorageStrategy;
use crate::onnx_graph::operators::{
    Add, Concat, Constant, LayerNormalization, MatMul, Mul, RMSNormalization, Slice, Softmax,
    Transpose,
};
use crate::onnx_graph::pytorch::{
    cast, conv2d, div_scalar, gelu_pytorch_tanh, linear, reshape, silu,
};
use crate::onnx_graph::tensor::{
    DType, Dimension, InputTensor, InputTensorInitialized, Shape, Tensor, TensorData,
    TensorDataValue,
};
use crate::onnx_graph::weights::{SafetensorsWeightManager, WeightManager};
use prost::Message;
use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;

type TensorPair = (Arc<dyn Tensor>, Arc<dyn Tensor>);
type TensorOptionalContextPair = (Arc<dyn Tensor>, Option<Arc<dyn Tensor>>);

#[derive(Clone, Debug)]
pub struct Sd3TransformerConfig {
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub hidden_dim: usize,
    pub latent_channels: usize,
    pub out_channels: usize,
    pub patch_size: usize,
    pub sample_size: usize,
    pub pos_embed_max_size: usize,
    pub joint_attention_dim: usize,
    pub pooled_projection_dim: usize,
    pub context_seq_len: usize,
    pub dual_attention_layers: HashSet<usize>,
}

impl Sd3TransformerConfig {
    fn patch_h(&self) -> usize {
        self.sample_size / self.patch_size
    }

    fn patch_w(&self) -> usize {
        self.sample_size / self.patch_size
    }

    fn patch_seq_len(&self) -> usize {
        self.patch_h() * self.patch_w()
    }
}

fn slice_axis(
    input: Arc<dyn Tensor>,
    axis: i64,
    start: i64,
    end: i64,
) -> Result<Arc<dyn Tensor>, Error> {
    let resolved_axis = if axis < 0 {
        input.rank() as i64 + axis
    } else {
        axis
    };
    let shape = Shape::new(vec![Dimension::new(Some(1), None, None)]);
    let starts = Constant::new(None, TensorData::new(vec![start].into(), shape.clone())?);
    let ends = Constant::new(None, TensorData::new(vec![end].into(), shape.clone())?);
    let axes = Constant::new(None, TensorData::new(vec![resolved_axis].into(), shape)?);
    Ok(Slice::new(None, input, starts, ends, Some(axes), None)?)
}

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

fn layer_norm_bare(
    input: Arc<dyn Tensor>,
    hidden_dim: usize,
    epsilon: f32,
) -> Result<Arc<LayerNormalization>, Error> {
    let scale = ones_constant(hidden_dim, input.dtype());
    LayerNormalization::new(None, input, scale, None, -1, epsilon, 1)
}

fn adaln_modulate(
    input: Arc<dyn Tensor>,
    shift: Arc<dyn Tensor>,
    scale: Arc<dyn Tensor>,
    hidden_dim: usize,
) -> Result<Arc<dyn Tensor>, Error> {
    let normed = layer_norm_bare(input, hidden_dim, 1e-6)?;
    let one = ones_constant(hidden_dim, normed.dtype());
    let scale_plus_one = Add::new(None, one, scale)?;
    let scaled = Mul::new(None, normed, scale_plus_one)?;
    Ok(Add::new(None, scaled, shift)?)
}

fn sd3_feed_forward(
    wm: &impl WeightManager,
    input: Arc<dyn Tensor>,
) -> Result<Arc<dyn Tensor>, Error> {
    let x = linear(&wm.prefix("net.0.proj"), input)?;
    let x = gelu_pytorch_tanh(x)?;
    linear(&wm.prefix("net.2"), x)
}

fn split_chunks(
    input: Arc<dyn Tensor>,
    hidden_dim: usize,
    num_chunks: usize,
) -> Result<Vec<Arc<dyn Tensor>>, Error> {
    let mut out = Vec::with_capacity(num_chunks);
    for i in 0..num_chunks {
        out.push(slice_axis(
            input.clone(),
            -1,
            (i * hidden_dim) as i64,
            ((i + 1) * hidden_dim) as i64,
        )?);
    }
    Ok(out)
}

fn sd3_timestep_embedding(
    wm: &impl WeightManager,
    timestep: Arc<dyn Tensor>,
    model_dtype: DType,
    embedding_dim: usize,
) -> Result<Arc<dyn Tensor>, Error> {
    let timestep = cast(timestep, DType::F32);
    let half_dim = embedding_dim / 2;
    let freqs: Vec<f32> = (0..half_dim)
        .map(|i| (-(10000.0f32.ln()) * i as f32 / half_dim as f32).exp())
        .collect();
    let freq_tensor = InputTensorInitialized::new(
        "sd3_timestep_freqs".to_string(),
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
    let emb = Concat::new(None, vec![cos_part, sin_part], -1)?;
    let emb = cast(emb, model_dtype);
    let emb = linear(&wm.prefix("linear_1"), emb)?;
    let emb = silu(emb)?;
    linear(&wm.prefix("linear_2"), emb)
}

fn sd3_joint_attention(
    wm: &impl WeightManager,
    hidden: Arc<dyn Tensor>,
    context: Arc<dyn Tensor>,
    config: &Sd3TransformerConfig,
) -> Result<TensorOptionalContextPair, Error> {
    let nh = config.num_heads as i64;
    let hd = config.head_dim as i64;

    let q_img = linear(&wm.prefix("to_q"), hidden.clone())?;
    let k_img = linear(&wm.prefix("to_k"), hidden.clone())?;
    let v_img = linear(&wm.prefix("to_v"), hidden)?;

    let q_ctx = linear(&wm.prefix("add_q_proj"), context.clone())?;
    let k_ctx = linear(&wm.prefix("add_k_proj"), context.clone())?;
    let v_ctx = linear(&wm.prefix("add_v_proj"), context)?;

    let q_img = Transpose::new(
        None,
        reshape(q_img, vec![0, 0, nh, hd])?,
        Some(vec![0, 2, 1, 3]),
    );
    let k_img = Transpose::new(
        None,
        reshape(k_img, vec![0, 0, nh, hd])?,
        Some(vec![0, 2, 1, 3]),
    );
    let v_img = Transpose::new(
        None,
        reshape(v_img, vec![0, 0, nh, hd])?,
        Some(vec![0, 2, 1, 3]),
    );
    let q_ctx = Transpose::new(
        None,
        reshape(q_ctx, vec![0, 0, nh, hd])?,
        Some(vec![0, 2, 1, 3]),
    );
    let k_ctx = Transpose::new(
        None,
        reshape(k_ctx, vec![0, 0, nh, hd])?,
        Some(vec![0, 2, 1, 3]),
    );
    let v_ctx = Transpose::new(
        None,
        reshape(v_ctx, vec![0, 0, nh, hd])?,
        Some(vec![0, 2, 1, 3]),
    );

    let q_img =
        RMSNormalization::new(None, q_img, wm.get_tensor("norm_q.weight")?, Some(1e-6), -1)?;
    let k_img =
        RMSNormalization::new(None, k_img, wm.get_tensor("norm_k.weight")?, Some(1e-6), -1)?;
    let q_ctx = RMSNormalization::new(
        None,
        q_ctx,
        wm.get_tensor("norm_added_q.weight")?,
        Some(1e-6),
        -1,
    )?;
    let k_ctx = RMSNormalization::new(
        None,
        k_ctx,
        wm.get_tensor("norm_added_k.weight")?,
        Some(1e-6),
        -1,
    )?;

    // Concatenate context first, then image tokens.
    let q = Concat::new(None, vec![q_ctx, q_img], 2)?;
    let k = Concat::new(None, vec![k_ctx, k_img], 2)?;
    let v = Concat::new(None, vec![v_ctx, v_img], 2)?;

    let scores = MatMul::new(None, q, Transpose::new(None, k, Some(vec![0, 1, 3, 2])))?;
    let scores = div_scalar(scores, (config.head_dim as f32).sqrt())?;
    let attn = Softmax::new(None, scores, Some(-1));
    let attn_out = MatMul::new(None, attn, v)?;
    let attn_out = Transpose::new(None, attn_out, Some(vec![0, 2, 1, 3]));
    let attn_out = reshape(attn_out, vec![0, 0, config.hidden_dim as i64])?;

    let ctx_len = config.context_seq_len as i64;
    let total_len = attn_out.shape()[1].resolve()? as i64;
    let ctx_out = slice_axis(attn_out.clone(), 1, 0, ctx_len)?;
    let img_out = slice_axis(attn_out, 1, ctx_len, total_len)?;

    let img_out = linear(&wm.prefix("to_out.0"), img_out)?;
    let ctx_out = if wm.prefix("to_add_out").get_tensor("weight").is_ok() {
        Some(linear(&wm.prefix("to_add_out"), ctx_out)?)
    } else {
        None
    };

    Ok((img_out, ctx_out))
}

fn sd3_self_attention(
    wm: &impl WeightManager,
    hidden: Arc<dyn Tensor>,
    config: &Sd3TransformerConfig,
) -> Result<Arc<dyn Tensor>, Error> {
    let nh = config.num_heads as i64;
    let hd = config.head_dim as i64;

    let q = linear(&wm.prefix("to_q"), hidden.clone())?;
    let k = linear(&wm.prefix("to_k"), hidden.clone())?;
    let v = linear(&wm.prefix("to_v"), hidden)?;

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

    let q = RMSNormalization::new(None, q, wm.get_tensor("norm_q.weight")?, Some(1e-6), -1)?;
    let k = RMSNormalization::new(None, k, wm.get_tensor("norm_k.weight")?, Some(1e-6), -1)?;

    let scores = MatMul::new(None, q, Transpose::new(None, k, Some(vec![0, 1, 3, 2])))?;
    let scores = div_scalar(scores, (config.head_dim as f32).sqrt())?;
    let attn = Softmax::new(None, scores, Some(-1));
    let out = MatMul::new(None, attn, v)?;
    let out = Transpose::new(None, out, Some(vec![0, 2, 1, 3]));
    let out = reshape(out, vec![0, 0, config.hidden_dim as i64])?;
    linear(&wm.prefix("to_out.0"), out)
}

fn sd3_joint_block(
    wm: &impl WeightManager,
    hidden: Arc<dyn Tensor>,
    context: Arc<dyn Tensor>,
    vec_cond: Arc<dyn Tensor>,
    config: &Sd3TransformerConfig,
    layer_idx: usize,
) -> Result<TensorPair, Error> {
    let hidden_dim = config.hidden_dim;
    let has_attn2 = config.dual_attention_layers.contains(&layer_idx)
        && wm.prefix("attn2").get_tensor("to_q.weight").is_ok();

    // SD3 AdaLN blocks apply SiLU(vec_cond) before each modulation projection.
    let vec_act = silu(vec_cond)?;

    let norm1_params = linear(&wm.prefix("norm1.linear"), vec_act.clone())?;
    let norm1_params = crate::onnx_graph::pytorch::unsqueeze(norm1_params, 1)?;
    let norm1_chunks_total = norm1_params.shape()[2].resolve()?;
    let norm1_chunks = norm1_chunks_total / hidden_dim;
    let mods = split_chunks(norm1_params, hidden_dim, norm1_chunks)?;

    let shift_msa = mods[0].clone();
    let scale_msa = mods[1].clone();
    let gate_msa = mods[2].clone();
    let shift_mlp = mods[3].clone();
    let scale_mlp = mods[4].clone();
    let gate_mlp = mods[5].clone();

    let (shift_msa2, scale_msa2, gate_msa2) = if has_attn2 && norm1_chunks >= 9 {
        (
            Some(mods[6].clone()),
            Some(mods[7].clone()),
            Some(mods[8].clone()),
        )
    } else {
        (None, None, None)
    };

    let norm1_ctx_params = linear(&wm.prefix("norm1_context.linear"), vec_act)?;
    let norm1_ctx_params = crate::onnx_graph::pytorch::unsqueeze(norm1_ctx_params, 1)?;
    let norm1_ctx_chunks_total = norm1_ctx_params.shape()[2].resolve()?;
    let norm1_ctx_chunks = norm1_ctx_chunks_total / hidden_dim;
    let ctx_mods = split_chunks(norm1_ctx_params, hidden_dim, norm1_ctx_chunks)?;
    let ctx_shift_msa = ctx_mods[0].clone();
    let ctx_scale_msa = ctx_mods[1].clone();
    let ctx_gate_msa = if norm1_ctx_chunks >= 3 {
        Some(ctx_mods[2].clone())
    } else {
        None
    };
    let ctx_shift_mlp = if norm1_ctx_chunks >= 4 {
        Some(ctx_mods[3].clone())
    } else {
        None
    };
    let ctx_scale_mlp = if norm1_ctx_chunks >= 5 {
        Some(ctx_mods[4].clone())
    } else {
        None
    };
    let ctx_gate_mlp = if norm1_ctx_chunks >= 6 {
        Some(ctx_mods[5].clone())
    } else {
        None
    };

    let hidden_msa = adaln_modulate(hidden.clone(), shift_msa, scale_msa, hidden_dim)?;
    let context_msa = adaln_modulate(context.clone(), ctx_shift_msa, ctx_scale_msa, hidden_dim)?;
    let (attn_hidden, attn_context) =
        sd3_joint_attention(&wm.prefix("attn"), hidden_msa, context_msa, config)?;

    let mut hidden_out = Add::new(None, hidden.clone(), Mul::new(None, gate_msa, attn_hidden)?)?;
    if let (Some(shift2), Some(scale2), Some(gate2)) = (shift_msa2, scale_msa2, gate_msa2) {
        let hidden_attn2_in = adaln_modulate(hidden, shift2, scale2, hidden_dim)?;
        let attn2_out = sd3_self_attention(&wm.prefix("attn2"), hidden_attn2_in, config)?;
        hidden_out = Add::new(None, hidden_out, Mul::new(None, gate2, attn2_out)?)?;
    }

    let hidden_mlp_in = adaln_modulate(hidden_out.clone(), shift_mlp, scale_mlp, hidden_dim)?;
    let hidden_mlp_out = sd3_feed_forward(&wm.prefix("ff"), hidden_mlp_in)?;
    hidden_out = Add::new(None, hidden_out, Mul::new(None, gate_mlp, hidden_mlp_out)?)?;

    let mut context_out = context;
    if let (Some(ctx_attn), Some(gate_ctx_attn)) = (attn_context, ctx_gate_msa) {
        context_out = Add::new(None, context_out, Mul::new(None, gate_ctx_attn, ctx_attn)?)?;
    }
    if let (Some(ctx_shift), Some(ctx_scale), Some(ctx_gate)) =
        (ctx_shift_mlp, ctx_scale_mlp, ctx_gate_mlp)
    {
        let context_mlp_in = adaln_modulate(context_out.clone(), ctx_shift, ctx_scale, hidden_dim)?;
        let context_mlp_out = sd3_feed_forward(&wm.prefix("ff_context"), context_mlp_in)?;
        context_out = Add::new(
            None,
            context_out,
            Mul::new(None, ctx_gate, context_mlp_out)?,
        )?;
    }

    Ok((hidden_out, context_out))
}

fn patch_embed_with_pos(
    wm: &impl WeightManager,
    latent: Arc<dyn Tensor>,
    config: &Sd3TransformerConfig,
) -> Result<Arc<dyn Tensor>, Error> {
    // Conv patch projection: [B, C, H, W] -> [B, hidden, H/p, W/p]
    let x = conv2d(
        &wm.prefix("pos_embed.proj"),
        latent,
        config.patch_size as i64,
        config.patch_size as i64,
        0,
    )?;
    // -> [B, seq, hidden]
    let x = reshape(x, vec![0, config.hidden_dim as i64, -1])?;
    let x = Transpose::new(None, x, Some(vec![0, 2, 1]));

    // Position embedding is stored at max size and center-cropped to sample_size/patch_size.
    let pos = wm.get_tensor("pos_embed.pos_embed")?;
    let pos = reshape(
        pos,
        vec![
            1,
            config.pos_embed_max_size as i64,
            config.pos_embed_max_size as i64,
            config.hidden_dim as i64,
        ],
    )?;
    let patch_h = config.patch_h();
    let patch_w = config.patch_w();
    let top = ((config.pos_embed_max_size - patch_h) / 2) as i64;
    let left = ((config.pos_embed_max_size - patch_w) / 2) as i64;
    let pos = slice_axis_common(pos, 1, top, top + patch_h as i64)?;
    let pos = slice_axis_common(pos, 2, left, left + patch_w as i64)?;
    let pos = reshape(
        pos,
        vec![1, config.patch_seq_len() as i64, config.hidden_dim as i64],
    )?;

    Ok(Add::new(None, x, pos)?)
}

fn unpatchify(
    input: Arc<dyn Tensor>,
    config: &Sd3TransformerConfig,
) -> Result<Arc<dyn Tensor>, Error> {
    let patch_h = config.patch_h() as i64;
    let patch_w = config.patch_w() as i64;
    let p = config.patch_size as i64;
    let c = config.out_channels as i64;

    let x = reshape(input, vec![0, patch_h, patch_w, c, p, p])?;
    let x = Transpose::new(None, x, Some(vec![0, 3, 1, 4, 2, 5]));
    reshape(
        x,
        vec![0, c, config.sample_size as i64, config.sample_size as i64],
    )
    .map(|x| x as Arc<dyn Tensor>)
}

pub fn load_sd3_transformer(
    weight_manager: impl WeightManager,
    config: Sd3TransformerConfig,
    output_method: WeightStorageStrategy,
) -> Result<Vec<u8>, anyhow::Error> {
    load_sd3_transformer_with_origin(weight_manager, config, output_method, None)
}

pub fn load_sd3_transformer_with_origin(
    weight_manager: impl WeightManager,
    config: Sd3TransformerConfig,
    output_method: WeightStorageStrategy,
    origin_path: Option<&Path>,
) -> Result<Vec<u8>, anyhow::Error> {
    let model_dtype = weight_manager
        .get_tensor("transformer_blocks.0.attn.to_q.weight")
        .map(|t| t.dtype())
        .unwrap_or(DType::F16);
    let wm = CastingWeightManager::new(weight_manager, model_dtype);

    let batch_dim = Dimension::new(Some(1), Some("batch".to_string()), None);
    let latent_input = InputTensor::new(
        "latent_sample".to_string(),
        model_dtype,
        Shape::new(vec![
            batch_dim.clone(),
            Dimension::new(Some(config.latent_channels), None, None),
            Dimension::new(Some(config.sample_size), None, None),
            Dimension::new(Some(config.sample_size), None, None),
        ]),
    );
    let timestep_input = InputTensor::new(
        "timestep".to_string(),
        model_dtype,
        Shape::new(vec![batch_dim.clone()]),
    );
    let context_input = InputTensor::new(
        "encoder_hidden_states".to_string(),
        model_dtype,
        Shape::new(vec![
            batch_dim.clone(),
            Dimension::new(Some(config.context_seq_len), None, None),
            Dimension::new(Some(config.joint_attention_dim), None, None),
        ]),
    );
    let pooled_input = InputTensor::new(
        "pooled_projections".to_string(),
        model_dtype,
        Shape::new(vec![
            batch_dim.clone(),
            Dimension::new(Some(config.pooled_projection_dim), None, None),
        ]),
    );

    let input_tensors: Vec<Arc<dyn Tensor>> = vec![
        latent_input.clone(),
        timestep_input.clone(),
        context_input.clone(),
        pooled_input.clone(),
    ];

    // Text context projection into transformer hidden dim.
    let mut context = linear(&wm.prefix("context_embedder"), context_input)?;
    // Image patch embedding + learned 2D positional embedding.
    let mut hidden = patch_embed_with_pos(&wm, latent_input, &config)?;

    let t_emb = sd3_timestep_embedding(
        &wm.prefix("time_text_embed.timestep_embedder"),
        timestep_input,
        model_dtype,
        256,
    )?;
    let pooled_emb = linear(
        &wm.prefix("time_text_embed.text_embedder.linear_1"),
        pooled_input,
    )?;
    let pooled_emb = silu(pooled_emb)?;
    let pooled_emb = linear(
        &wm.prefix("time_text_embed.text_embedder.linear_2"),
        pooled_emb,
    )?;
    let vec_cond = Add::new(None, t_emb, pooled_emb)?;

    println!("Building SD3 transformer: {} blocks...", config.num_layers);
    for i in 0..config.num_layers {
        let block_wm = wm.prefix(&format!("transformer_blocks.{i}"));
        let (next_hidden, next_context) =
            sd3_joint_block(&block_wm, hidden, context, vec_cond.clone(), &config, i)?;
        hidden = next_hidden;
        context = next_context;
        if (i + 1) % 6 == 0 {
            println!("  Transformer block {}/{}", i + 1, config.num_layers);
        }
    }

    // Final AdaLN + projection back to latent patches.
    let norm_params = linear(&wm.prefix("norm_out.linear"), silu(vec_cond)?)?;
    let norm_params = crate::onnx_graph::pytorch::unsqueeze(norm_params, 1)?;
    let shift = slice_axis(norm_params.clone(), -1, 0, config.hidden_dim as i64)?;
    let scale = slice_axis(
        norm_params,
        -1,
        config.hidden_dim as i64,
        (config.hidden_dim * 2) as i64,
    )?;
    let hidden = adaln_modulate(hidden, shift, scale, config.hidden_dim)?;
    let hidden = linear(&wm.prefix("proj_out"), hidden)?;

    let output = unpatchify(hidden, &config)?;
    let output_tensors: Vec<(String, Arc<dyn Tensor>)> = vec![("out_sample".to_string(), output)];

    println!("Built SD3 transformer graph, exporting...");
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

#[derive(Clone)]
struct Sd3VaeRemapWeightManager {
    root: Arc<SafetensorsWeightManager>,
    prefix_tail: Option<String>,
    prefix: Option<String>,
}

impl Sd3VaeRemapWeightManager {
    fn new(root: SafetensorsWeightManager) -> Self {
        Self {
            root: Arc::new(root),
            prefix_tail: None,
            prefix: None,
        }
    }
}

impl WeightManager for Sd3VaeRemapWeightManager {
    fn prefix(&self, name: &str) -> Self {
        let prefix = Some(if let Some(prefix) = &self.prefix {
            format!("{prefix}.{name}")
        } else {
            name.to_string()
        });
        Self {
            root: self.root.clone(),
            prefix_tail: Some(name.to_string()),
            prefix,
        }
    }

    fn get_tensor(&self, name: &str) -> Result<Arc<dyn Tensor>, Error> {
        let full_name = if let Some(prefix) = &self.prefix {
            format!("{prefix}.{name}")
        } else {
            name.to_string()
        };
        let mapped_name = map_sd3_vae_weight_name(&full_name);
        self.root.get_tensor(&mapped_name)
    }

    fn get_prefix_tail(&self) -> Option<&str> {
        self.prefix_tail.as_deref()
    }

    fn get_prefix(&self) -> Option<&str> {
        self.prefix.as_deref()
    }

    fn get_tensor_names(&self) -> Vec<String> {
        self.root.get_tensor_names()
    }
}

fn map_sd3_vae_weight_name(name: &str) -> String {
    if let Some(rest) = name.strip_prefix("decoder.mid.block_1.") {
        return format!("decoder.mid_block.resnets.0.{}", map_resnet_tail(rest));
    }
    if let Some(rest) = name.strip_prefix("decoder.mid.block_2.") {
        return format!("decoder.mid_block.resnets.1.{}", map_resnet_tail(rest));
    }
    if let Some(rest) = name.strip_prefix("decoder.mid.attn_1.") {
        if let Some(tail) = rest.strip_prefix("norm.") {
            return format!("decoder.mid_block.attentions.0.group_norm.{tail}");
        }
        if let Some(tail) = rest.strip_prefix("q.") {
            return format!("decoder.mid_block.attentions.0.to_q.{tail}");
        }
        if let Some(tail) = rest.strip_prefix("k.") {
            return format!("decoder.mid_block.attentions.0.to_k.{tail}");
        }
        if let Some(tail) = rest.strip_prefix("v.") {
            return format!("decoder.mid_block.attentions.0.to_v.{tail}");
        }
        if let Some(tail) = rest.strip_prefix("proj_out.") {
            return format!("decoder.mid_block.attentions.0.to_out.0.{tail}");
        }
    }
    if let Some(rest) = name.strip_prefix("decoder.norm_out.") {
        return format!("decoder.conv_norm_out.{rest}");
    }
    if let Some(rest) = name.strip_prefix("decoder.up.") {
        let parts: Vec<&str> = rest.split('.').collect();
        if parts.len() >= 4
            && let Ok(level) = parts[0].parse::<usize>()
        {
            let block_level = 3usize.saturating_sub(level);
            if parts[1] == "block"
                && parts.len() >= 5
                && let Ok(block) = parts[2].parse::<usize>()
            {
                let tail = map_resnet_tail(&parts[3..].join("."));
                return format!("decoder.up_blocks.{block_level}.resnets.{block}.{tail}");
            }
            if parts[1] == "upsample" && parts[2] == "conv" {
                let tail = parts[3..].join(".");
                return format!("decoder.up_blocks.{block_level}.upsamplers.0.conv.{tail}");
            }
        }
    }
    name.to_string()
}

fn map_resnet_tail(tail: &str) -> String {
    if let Some(rest) = tail.strip_prefix("nin_shortcut.") {
        format!("conv_shortcut.{rest}")
    } else {
        tail.to_string()
    }
}

pub fn build_sd3_vae_decoder(
    weight_manager: SafetensorsWeightManager,
    model_dtype: DType,
    output_method: WeightStorageStrategy,
    origin_path: &Path,
) -> Result<Vec<u8>, anyhow::Error> {
    let remapped = Sd3VaeRemapWeightManager::new(weight_manager);
    sd_common::build_flux_vae_decoder(remapped, model_dtype, output_method, origin_path)
}
