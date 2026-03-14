use crate::Error;
use crate::onnx_graph::WeightStorageStrategy;
use crate::onnx_graph::operators::{
    Add, Concat, Constant, Gather, MatMul, Mul, RMSNormalization, Reshape, RotaryEmbedding,
    ShapeOp, Softmax, Tanh, Transpose,
};
use crate::onnx_graph::pytorch::{
    cast, div_scalar, gelu, gelu_pytorch_tanh, linear, reshape, transpose, unsqueeze,
};
use crate::onnx_graph::tensor::{
    DType, Dimension, InputTensor, InputTensorInitialized, Shape, Tensor, TensorData,
    TensorDataValue,
};
use crate::onnx_graph::weights::WeightManager;
use prost::Message;
use std::sync::Arc;

pub struct Gemma2Config {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: Option<usize>,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
    pub rms_norm_eps: f32,
    pub hidden_activation: String,
    pub query_pre_attn_scalar: f32,
    pub attn_logit_softcapping: Option<f32>,
    pub final_logit_softcapping: Option<f32>,
}

impl Gemma2Config {
    pub fn from_huggingface_transformers_json(config: &serde_json::Value) -> Result<Self, Error> {
        fn get_int(config: &serde_json::Value, key: &str) -> Result<i64, Error> {
            config
                .get(key)
                .ok_or(Error::MissingConfigEntryError(key.to_string()))?
                .as_i64()
                .ok_or(Error::MissingConfigEntryError(key.to_string()))
        }

        let hidden_size = get_int(config, "hidden_size")? as usize;
        let num_hidden_layers = get_int(config, "num_hidden_layers")? as usize;
        let num_attention_heads = get_int(config, "num_attention_heads")? as usize;
        let num_key_value_heads = get_int(config, "num_key_value_heads")? as usize;
        let head_dim = config
            .get("head_dim")
            .and_then(|v| v.as_i64())
            .map(|v| v as usize);
        let rope_theta = config
            .get("rope_theta")
            .and_then(|v| v.as_f64())
            .unwrap_or(10000.0);
        let max_position_embeddings = config
            .get("max_position_embeddings")
            .and_then(|v| v.as_i64())
            .unwrap_or(8192) as usize;
        let tie_word_embeddings = config
            .get("tie_word_embeddings")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let rms_norm_eps = config
            .get("rms_norm_eps")
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-6) as f32;
        let hidden_activation = config
            .get("hidden_activation")
            .or_else(|| config.get("hidden_act"))
            .and_then(|v| v.as_str())
            .unwrap_or("gelu_pytorch_tanh")
            .to_string();
        let query_pre_attn_scalar = config
            .get("query_pre_attn_scalar")
            .and_then(|v| v.as_f64())
            .unwrap_or((head_dim.unwrap_or(hidden_size / num_attention_heads)) as f64)
            as f32;
        let attn_logit_softcapping = config
            .get("attn_logit_softcapping")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32);
        let final_logit_softcapping = config
            .get("final_logit_softcapping")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32);

        Ok(Self {
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rope_theta,
            max_position_embeddings,
            tie_word_embeddings,
            rms_norm_eps,
            hidden_activation,
            query_pre_attn_scalar,
            attn_logit_softcapping,
            final_logit_softcapping,
        })
    }
}

fn gemma_rms_norm(
    weight_manager: &impl WeightManager,
    input: Arc<dyn Tensor>,
    epsilon: f32,
) -> Result<Arc<dyn Tensor>, crate::onnx_graph::Error> {
    let weight = weight_manager.get_tensor("weight")?;
    let one: Arc<dyn Tensor> =
        Constant::new(None, TensorData::fill(Shape::from(&[1usize][..]), 1.0f32)?);
    let one = cast(one, weight.dtype());
    let scale = Add::new(None, weight, one)?;
    let out = RMSNormalization::new(
        weight_manager.get_prefix().map(|x| x.to_string()),
        input,
        scale,
        Some(epsilon),
        -1,
    )?;
    Ok(out as Arc<dyn Tensor>)
}

pub fn load_gemma2(
    weight_manager: impl WeightManager,
    config: Gemma2Config,
    output_method: WeightStorageStrategy,
) -> Result<Vec<u8>, anyhow::Error> {
    let model_weight_manager = weight_manager.prefix("model");

    let mut input_tensors: Vec<Arc<dyn Tensor>> = vec![];
    let mut output_tensors: Vec<(String, Arc<dyn Tensor>)> = vec![];

    let batch_dimension = Dimension::new(
        Some(1),
        Some("batch_size".to_string()),
        Some("DATA_BATCH".to_string()),
    );
    let sequence_dimension = Dimension::new(None, Some("seq_len".to_string()), None);

    let input_shape = Shape::new(vec![batch_dimension.clone(), sequence_dimension.clone()]);

    let token_input = InputTensor::new("input_ids".to_string(), DType::I32, input_shape);
    input_tensors.push(token_input.clone());

    let embed_weight = model_weight_manager.get_tensor("embed_tokens.weight")?;
    let x = Gather::new(
        Some("embed_tokens".to_string()),
        embed_weight.clone(),
        token_input.clone(),
        0,
    )?;
    let embed_scale: Arc<dyn Tensor> = Constant::new(
        Some("embed_scale".to_string()),
        TensorData::fill(
            Shape::from(&[1usize][..]),
            (config.hidden_size as f32).sqrt(),
        )?,
    );
    let embed_scale = cast(embed_scale, x.dtype());
    let x = Mul::new(Some("scale_embeddings".to_string()), x, embed_scale)?;

    let kv_cache_seq_dim = Dimension::new(None, Some("kv_cache_sequence".to_string()), None);

    let head_dim = config
        .head_dim
        .unwrap_or(config.hidden_size / config.num_attention_heads);
    let kv_cache_input_type = x.dtype();
    let kv_cache_input_shape = Shape::new(vec![
        batch_dimension,
        Dimension::new(Some(config.num_key_value_heads), None, None),
        kv_cache_seq_dim,
        Dimension::new(Some(head_dim), None, None),
    ]);

    // Precompute RoPE frequency tables and embed as constants.
    let half_head_dim = head_dim / 2;
    let max_len = config.max_position_embeddings;
    let (cos_values, sin_values) = {
        let inv_freq: Vec<f64> = (0..half_head_dim)
            .map(|i| 1.0 / config.rope_theta.powf(i as f64 * 2.0 / head_dim as f64))
            .collect();
        let mut cos_vals = Vec::with_capacity(max_len * half_head_dim);
        let mut sin_vals = Vec::with_capacity(max_len * half_head_dim);
        for pos in 0..max_len {
            for &freq in &inv_freq {
                let angle = pos as f64 * freq;
                cos_vals.push(half::bf16::from_f64(angle.cos()));
                sin_vals.push(half::bf16::from_f64(angle.sin()));
            }
        }
        (cos_vals, sin_vals)
    };
    let cos_sin_cache_shape = Shape::new(vec![
        Dimension::new(Some(max_len), None, None),
        Dimension::new(Some(half_head_dim), None, None),
    ]);
    let cos_cache = InputTensorInitialized::new(
        "cos_cache".to_string(),
        TensorData::new(
            TensorDataValue::BF16(cos_values),
            cos_sin_cache_shape.clone(),
        )?,
    );
    let sin_cache = InputTensorInitialized::new(
        "sin_cache".to_string(),
        TensorData::new(TensorDataValue::BF16(sin_values), cos_sin_cache_shape)?,
    );

    let mut layer_output: Arc<dyn Tensor> = x;
    for i in 0..config.num_hidden_layers {
        let layer_weight_manager = model_weight_manager.prefix(&format!("layers.{i}"));
        let layer_input = layer_output.clone();
        let att_norm = gemma_rms_norm(
            &layer_weight_manager.prefix("input_layernorm"),
            layer_input.clone(),
            config.rms_norm_eps,
        )?;

        // Multi-head Attention
        let q = linear(
            &layer_weight_manager.prefix("self_attn.q_proj"),
            att_norm.clone(),
        )?;
        let k = linear(
            &layer_weight_manager.prefix("self_attn.k_proj"),
            att_norm.clone(),
        )?;
        let v = linear(&layer_weight_manager.prefix("self_attn.v_proj"), att_norm)?;

        let q = Transpose::new(
            None,
            reshape(
                q,
                vec![0, 0, config.num_attention_heads as i64, head_dim as i64],
            )?,
            Some(vec![0, 2, 1, 3]),
        );
        let k = Transpose::new(
            None,
            reshape(
                k,
                vec![0, 0, config.num_key_value_heads as i64, head_dim as i64],
            )?,
            Some(vec![0, 2, 1, 3]),
        );
        let v = Transpose::new(
            None,
            reshape(
                v,
                vec![0, 0, config.num_key_value_heads as i64, head_dim as i64],
            )?,
            Some(vec![0, 2, 1, 3]),
        );

        let kv_cache_input_k = InputTensor::new(
            format!("kv_cache_input_k_{i}"),
            kv_cache_input_type,
            kv_cache_input_shape.clone(),
        );

        let kv_cache_input_v = InputTensor::new(
            format!("kv_cache_input_v_{i}"),
            kv_cache_input_type,
            kv_cache_input_shape.clone(),
        );

        // RoPE position from KV cache sequence length.
        let rope_pos = ShapeOp::new(None, kv_cache_input_k.clone(), Some(2), Some(3))?;
        let cos_at_pos = Gather::new(None, cos_cache.clone(), rope_pos.clone(), 0)?;
        let sin_at_pos = Gather::new(None, sin_cache.clone(), rope_pos, 0)?;
        let cos_at_pos = reshape(cos_at_pos, vec![1, 1, half_head_dim as i64])?;
        let sin_at_pos = reshape(sin_at_pos, vec![1, 1, half_head_dim as i64])?;

        let q = RotaryEmbedding::new(
            None,
            q,
            cos_at_pos.clone(),
            sin_at_pos.clone(),
            None,
            None,
            None,
            None,
        )?;
        let k = RotaryEmbedding::new(None, k, cos_at_pos, sin_at_pos, None, None, None, None)?;

        let new_seq_len_dim = Dimension::new(None, Some("new_seq_len".to_string()), None);
        let new_shape = Shape::new(vec![
            k.shape()[0].clone(),
            k.shape()[1].clone(),
            new_seq_len_dim,
            k.shape()[3].clone(),
        ]);
        let k = Concat::new_with_output_shape(
            None,
            vec![kv_cache_input_k.clone(), k],
            2,
            new_shape.clone(),
        )?;
        let v =
            Concat::new_with_output_shape(None, vec![kv_cache_input_v.clone(), v], 2, new_shape)?;

        input_tensors.push(kv_cache_input_k);
        output_tensors.push((format!("kv_cache_output_k_{i}"), k.clone()));

        input_tensors.push(kv_cache_input_v);
        output_tensors.push((format!("kv_cache_output_v_{i}"), v.clone()));

        let (k, v): (Arc<dyn Tensor>, Arc<dyn Tensor>) = if config.num_key_value_heads
            == config.num_attention_heads
        {
            (k, v)
        } else {
            let repeat_kv =
                |x: Arc<dyn Tensor>| -> Result<Arc<dyn Tensor>, crate::onnx_graph::Error> {
                    let n_rep = config.num_attention_heads / config.num_key_value_heads;
                    let seq_dim = x.shape()[2].clone();
                    let x = unsqueeze(x, 2)?;
                    let x: Arc<dyn Tensor> = Concat::new(None, vec![x.clone(); n_rep], 2)?;
                    let target_dims =
                        vec![0i64, config.num_attention_heads as i64, -1, head_dim as i64];
                    let shape_const = Constant::new(
                        None,
                        TensorData::new(target_dims.into(), Shape::from(&[4usize][..]))?,
                    );
                    let output_shape = Shape::new(vec![
                        x.shape()[0].clone(),
                        Dimension::new(Some(config.num_attention_heads), None, None),
                        seq_dim,
                        Dimension::new(Some(head_dim), None, None),
                    ]);
                    let x = Reshape::new_with_forced_output(None, x, shape_const, output_shape)?;
                    Ok(x as Arc<dyn Tensor>)
                };

            (repeat_kv(k)?, repeat_kv(v)?)
        };

        let scores = MatMul::new(None, q, transpose(k.clone()))?;
        let mut scores: Arc<dyn Tensor> = div_scalar(scores, config.query_pre_attn_scalar.sqrt())?;
        if let Some(softcap) = config.attn_logit_softcapping {
            if softcap > 0.0 {
                let s = div_scalar(scores, softcap)?;
                let s = Tanh::new(None, s);
                let scale: Arc<dyn Tensor> =
                    Constant::new(None, TensorData::fill(Shape::from(&[1usize][..]), softcap)?);
                let scale = cast(scale, s.dtype());
                scores = Mul::new(None, s, scale)?;
            }
        }

        let scores = Softmax::new(None, scores, Some(3));
        let output = MatMul::new(None, scores, v)?;
        let output = Transpose::new(None, output, Some(vec![0, 2, 1, 3]));
        let output = reshape(output, vec![0, 0, -1])?;

        let hidden_layer = linear(&layer_weight_manager.prefix("self_attn.o_proj"), output)?;
        let hidden_layer = gemma_rms_norm(
            &layer_weight_manager.prefix("post_attention_layernorm"),
            hidden_layer,
            config.rms_norm_eps,
        )?;
        let attention_output = Add::new(None, layer_input, hidden_layer)?;

        let ffn_norm = gemma_rms_norm(
            &layer_weight_manager.prefix("pre_feedforward_layernorm"),
            attention_output.clone(),
            config.rms_norm_eps,
        )?;

        // Gemma2 FFN uses GeGLU with gelu_pytorch_tanh.
        let x = linear(
            &layer_weight_manager.prefix("mlp.gate_proj"),
            ffn_norm.clone(),
        )?;
        let x = match config.hidden_activation.as_str() {
            "gelu_pytorch_tanh" => gelu_pytorch_tanh(x)?,
            "gelu" => gelu(x)?,
            act => Err(Error::UnsupportedConfigurationError(
                "hidden_activation".to_string(),
                act.to_string(),
            ))?,
        };
        let x2 = linear(&layer_weight_manager.prefix("mlp.up_proj"), ffn_norm)?;
        let hidden_layer = Mul::new(None, x, x2)?;
        let hidden_layer = linear(&layer_weight_manager.prefix("mlp.down_proj"), hidden_layer)?;
        let hidden_layer = gemma_rms_norm(
            &layer_weight_manager.prefix("post_feedforward_layernorm"),
            hidden_layer,
            config.rms_norm_eps,
        )?;

        layer_output = Add::new(None, attention_output, hidden_layer)?;
    }

    let h = gemma_rms_norm(
        &model_weight_manager.prefix("norm"),
        layer_output,
        config.rms_norm_eps,
    )?;

    // Prefer explicit lm_head when present; fall back to tied embeddings.
    let mut out: Arc<dyn Tensor> = if !config.tie_word_embeddings
        && weight_manager
            .prefix("lm_head")
            .get_tensor("weight")
            .is_ok()
    {
        linear(&weight_manager.prefix("lm_head"), h)?
    } else {
        let weight = transpose(embed_weight);
        let h_rank = h.rank();
        let h_unsqueezed = unsqueeze(h, (h_rank as i64) - 1)?;
        MatMul::new(Some("lm_head".to_string()), h_unsqueezed, weight)?
    };
    if let Some(softcap) = config.final_logit_softcapping {
        if softcap > 0.0 {
            let o = div_scalar(out, softcap)?;
            let o = Tanh::new(None, o);
            let scale: Arc<dyn Tensor> =
                Constant::new(None, TensorData::fill(Shape::from(&[1usize][..]), softcap)?);
            let scale = cast(scale, o.dtype());
            out = Mul::new(None, o, scale)?;
        }
    }
    output_tensors.push(("logits".to_string(), out));

    println!("Built graph, exporting...");
    let onnx_model =
        crate::onnx_graph::build_proto(&input_tensors, &output_tensors, output_method)?;
    Ok(onnx_model.encode_to_vec())
}
