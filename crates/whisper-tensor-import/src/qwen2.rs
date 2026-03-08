use crate::Error;
use crate::onnx_graph::WeightStorageStrategy;
use crate::onnx_graph::operators::{
    Add, Concat, Constant, Gather, MatMul, Mul, Reshape, RotaryEmbedding, ShapeOp, Softmax,
    Transpose,
};
use crate::onnx_graph::pytorch::{div_scalar, linear, reshape, rms_norm, silu, transpose, unsqueeze};
use crate::onnx_graph::tensor::{
    DType, Dimension, InputTensor, InputTensorInitialized, Shape, Tensor, TensorData,
    TensorDataValue,
};
use crate::onnx_graph::weights::WeightManager;
use prost::Message;
use std::sync::Arc;

pub struct Qwen2Config {
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
}

impl Qwen2Config {
    pub fn from_huggingface_transformers_json(config: &serde_json::Value) -> Result<Self, Error> {
        fn get_int(config: &serde_json::Value, key: &str) -> Result<i64, Error> {
            config
                .get(key)
                .ok_or(Error::MissingConfigEntryError(key.to_string()))?
                .as_i64()
                .ok_or(Error::MissingConfigEntryError(key.to_string()))
        }

        let num_hidden_layers = get_int(config, "num_hidden_layers")? as usize;
        let num_attention_heads = get_int(config, "num_attention_heads")? as usize;
        let num_key_value_heads = get_int(config, "num_key_value_heads")? as usize;
        let rope_theta = config
            .get("rope_theta")
            .and_then(|v| v.as_f64())
            .unwrap_or(1000000.0);
        let max_position_embeddings = config
            .get("max_position_embeddings")
            .and_then(|v| v.as_i64())
            .unwrap_or(32768) as usize;
        let tie_word_embeddings = config
            .get("tie_word_embeddings")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        Ok(Self {
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            rope_theta,
            max_position_embeddings,
            tie_word_embeddings,
        })
    }
}

pub fn load_qwen2(
    weight_manager: impl WeightManager,
    config: Qwen2Config,
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

    let kv_cache_seq_dim = Dimension::new(None, Some("kv_cache_sequence".to_string()), None);

    let model_dim = x.shape().dims.last().unwrap().resolve()?;
    let head_dim = model_dim / config.num_attention_heads;
    let kv_cache_input_type = x.dtype();
    let kv_cache_input_shape = Shape::new(vec![
        batch_dimension,
        Dimension::new(Some(config.num_key_value_heads), None, None),
        kv_cache_seq_dim,
        Dimension::new(Some(head_dim), None, None),
    ]);

    // Precompute RoPE frequency tables.
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
        let att_norm = rms_norm(
            &layer_weight_manager.prefix("input_layernorm"),
            layer_input.clone(),
            None,
        )?;

        // Multi-head Attention (QKV projections have bias in Qwen2, handled by linear())
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

        let (k, v): (Arc<dyn Tensor>, Arc<dyn Tensor>) =
            if config.num_key_value_heads == config.num_attention_heads {
                (k, v)
            } else {
                let repeat_kv =
                    |x: Arc<dyn Tensor>| -> Result<Arc<dyn Tensor>, crate::onnx_graph::Error> {
                        let n_rep = config.num_attention_heads / config.num_key_value_heads;
                        let seq_dim = x.shape()[2].clone();
                        let x = unsqueeze(x, 2)?;
                        let x: Arc<dyn Tensor> =
                            Concat::new(None, vec![x.clone(); n_rep], 2)?;
                        let target_dims = vec![
                            0i64,
                            config.num_attention_heads as i64,
                            -1,
                            head_dim as i64,
                        ];
                        let shape_const = Constant::new(
                            None,
                            TensorData::new(
                                target_dims.into(),
                                Shape::from(&[4usize][..]),
                            )?,
                        );
                        let output_shape = Shape::new(vec![
                            x.shape()[0].clone(),
                            Dimension::new(Some(config.num_attention_heads), None, None),
                            seq_dim,
                            Dimension::new(Some(head_dim), None, None),
                        ]);
                        let x = Reshape::new_with_forced_output(
                            None, x, shape_const, output_shape,
                        )?;
                        Ok(x as Arc<dyn Tensor>)
                    };

                (repeat_kv(k)?, repeat_kv(v)?)
            };

        let scores = MatMul::new(None, q, transpose(k.clone()))?;
        let scores = div_scalar(scores, half::bf16::from_f32((head_dim as f32).sqrt()))?;

        let scores = Softmax::new(None, scores, Some(3));
        let output = MatMul::new(None, scores, v)?;
        let output = Transpose::new(None, output, Some(vec![0, 2, 1, 3]));
        let output = reshape(output, vec![0, 0, -1])?;

        let hidden_layer = linear(&layer_weight_manager.prefix("self_attn.o_proj"), output)?;

        let attention_output = Add::new(None, layer_input, hidden_layer)?;
        let ffn_norm = rms_norm(
            &layer_weight_manager.prefix("post_attention_layernorm"),
            attention_output.clone(),
            None,
        )?;

        // SwiGLU Feed-Forward
        let x = linear(
            &layer_weight_manager.prefix("mlp.gate_proj"),
            ffn_norm.clone(),
        )?;
        let x = silu(x)?;
        let x2 = linear(
            &layer_weight_manager.prefix("mlp.up_proj"),
            ffn_norm.clone(),
        )?;
        let hidden_layer = Mul::new(None, x, x2)?;
        let hidden_layer = linear(&layer_weight_manager.prefix("mlp.down_proj"), hidden_layer)?;

        layer_output = Add::new(None, attention_output, hidden_layer)?;
    }

    let h = rms_norm(&model_weight_manager.prefix("norm"), layer_output, None)?;

    // Qwen2 can tie word embeddings: reuse embed_tokens.weight as the output projection
    let out = if config.tie_word_embeddings {
        let weight = transpose(embed_weight);
        let h_rank = h.rank();
        let h_unsqueezed = unsqueeze(h, (h_rank as i64) - 1)?;
        MatMul::new(Some("lm_head".to_string()), h_unsqueezed, weight)?
    } else {
        linear(&weight_manager.prefix("lm_head"), h)?
    };
    output_tensors.push(("logits".to_string(), out));

    println!("Built graph, exporting...");
    let onnx_model =
        crate::onnx_graph::build_proto(&input_tensors, &output_tensors, output_method)?;
    Ok(onnx_model.encode_to_vec())
}
