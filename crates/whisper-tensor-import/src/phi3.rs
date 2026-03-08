use crate::Error;
use crate::onnx_graph::WeightStorageStrategy;
use crate::onnx_graph::operators::{
    Add, Concat, Constant, Gather, MatMul, Mul, Reshape, RotaryEmbedding, ShapeOp, Slice, Softmax,
    Transpose,
};
use crate::onnx_graph::pytorch::{div_scalar, reshape, rms_norm, silu, squeeze, transpose, unsqueeze};
use crate::onnx_graph::tensor::{
    DType, Dimension, InputTensor, InputTensorInitialized, Shape, Tensor, TensorData,
    TensorDataValue,
};
use crate::onnx_graph::weights::WeightManager;
use prost::Message;
use std::sync::Arc;

pub struct Phi3Config {
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
}

impl Phi3Config {
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
        let hidden_size = get_int(config, "hidden_size")? as usize;
        let intermediate_size = get_int(config, "intermediate_size")? as usize;
        let rope_theta = config
            .get("rope_theta")
            .and_then(|v| v.as_f64())
            .unwrap_or(10000.0);
        let max_position_embeddings = config
            .get("max_position_embeddings")
            .and_then(|v| v.as_i64())
            .unwrap_or(4096) as usize;
        let tie_word_embeddings = config
            .get("tie_word_embeddings")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        Ok(Self {
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            hidden_size,
            intermediate_size,
            rope_theta,
            max_position_embeddings,
            tie_word_embeddings,
        })
    }
}

/// Slice a tensor along a single axis.
fn slice_axis(
    input: Arc<dyn Tensor>,
    axis: i64,
    start: i64,
    end: i64,
) -> Result<Arc<Slice>, crate::onnx_graph::Error> {
    // Normalize negative axis
    let axis = if axis < 0 {
        input.rank() as i64 + axis
    } else {
        axis
    };
    let const_shape = Shape::new(vec![Dimension::new(Some(1), None, None)]);
    let starts = Constant::new(None, TensorData::new(vec![start].into(), const_shape.clone())?);
    let ends = Constant::new(None, TensorData::new(vec![end].into(), const_shape.clone())?);
    let axes = Constant::new(None, TensorData::new(vec![axis].into(), const_shape)?);
    Slice::new(None, input, starts, ends, Some(axes), None)
}

pub fn load_phi3(
    weight_manager: impl WeightManager,
    config: Phi3Config,
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

    let head_dim = config.hidden_size / config.num_attention_heads;
    let q_dim = config.num_attention_heads * head_dim;
    let kv_dim = config.num_key_value_heads * head_dim;
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

        // Fused QKV projection: qkv_proj.weight is [q_dim + 2*kv_dim, hidden_size]
        let qkv_weight = layer_weight_manager
            .prefix("self_attn.qkv_proj")
            .get_tensor("weight")?;
        let qkv_weight_t = transpose(qkv_weight);
        let att_norm_unsq = unsqueeze(att_norm.clone(), (att_norm.rank() as i64) - 1)?;
        let qkv = MatMul::new(
            Some(format!("layers.{i}.self_attn.qkv_proj")),
            att_norm_unsq,
            qkv_weight_t,
        )?;
        let qkv = squeeze(qkv, (att_norm.rank() as i64) - 1)?;

        // Split Q, K, V from fused output along last axis
        let q: Arc<dyn Tensor> = slice_axis(qkv.clone(), -1, 0, q_dim as i64)?;
        let k: Arc<dyn Tensor> =
            slice_axis(qkv.clone(), -1, q_dim as i64, (q_dim + kv_dim) as i64)?;
        let v: Arc<dyn Tensor> =
            slice_axis(qkv, -1, (q_dim + kv_dim) as i64, (q_dim + 2 * kv_dim) as i64)?;

        // Reshape and transpose to [batch, heads, seq, head_dim]
        let q: Arc<dyn Tensor> = Transpose::new(
            None,
            reshape(
                q,
                vec![0, 0, config.num_attention_heads as i64, head_dim as i64],
            )?,
            Some(vec![0, 2, 1, 3]),
        );
        let k: Arc<dyn Tensor> = Transpose::new(
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

        // KV cache
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

        // RoPE
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

        // Concat with KV cache
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

        // GQA repeat if needed
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

        // Attention scores
        let scores = MatMul::new(None, q, transpose(k.clone()))?;
        let scores = div_scalar(scores, half::bf16::from_f32((head_dim as f32).sqrt()))?;

        let scores = Softmax::new(None, scores, Some(3));
        let output = MatMul::new(None, scores, v)?;
        let output = Transpose::new(None, output, Some(vec![0, 2, 1, 3]));
        let output = reshape(output, vec![0, 0, -1])?;

        // Output projection
        let hidden_layer = crate::onnx_graph::pytorch::linear(
            &layer_weight_manager.prefix("self_attn.o_proj"),
            output,
        )?;

        let attention_output = Add::new(None, layer_input, hidden_layer)?;

        // MLP with fused gate_up_proj
        let ffn_norm = rms_norm(
            &layer_weight_manager.prefix("post_attention_layernorm"),
            attention_output.clone(),
            None,
        )?;

        // gate_up_proj.weight is [2*intermediate_size, hidden_size]
        let gate_up_weight = layer_weight_manager
            .prefix("mlp.gate_up_proj")
            .get_tensor("weight")?;
        let gate_up_weight_t = transpose(gate_up_weight);
        let ffn_norm_unsq = unsqueeze(ffn_norm.clone(), (ffn_norm.rank() as i64) - 1)?;
        let gate_up = MatMul::new(
            Some(format!("layers.{i}.mlp.gate_up_proj")),
            ffn_norm_unsq,
            gate_up_weight_t,
        )?;
        let gate_up = squeeze(gate_up, (ffn_norm.rank() as i64) - 1)?;

        // Split gate and up
        let gate: Arc<dyn Tensor> =
            slice_axis(gate_up.clone(), -1, 0, config.intermediate_size as i64)?;
        let up: Arc<dyn Tensor> = slice_axis(
            gate_up,
            -1,
            config.intermediate_size as i64,
            (2 * config.intermediate_size) as i64,
        )?;

        let gate = silu(gate)?;
        let hidden_layer = Mul::new(None, gate, up)?;
        let hidden_layer = crate::onnx_graph::pytorch::linear(
            &layer_weight_manager.prefix("mlp.down_proj"),
            hidden_layer,
        )?;

        layer_output = Add::new(None, attention_output, hidden_layer)?;
    }

    let h = rms_norm(&model_weight_manager.prefix("norm"), layer_output, None)?;

    let out = if config.tie_word_embeddings {
        let weight = transpose(embed_weight);
        let h_rank = h.rank();
        let h_unsqueezed = unsqueeze(h, (h_rank as i64) - 1)?;
        MatMul::new(Some("lm_head".to_string()), h_unsqueezed, weight)?
    } else {
        crate::onnx_graph::pytorch::linear(&weight_manager.prefix("lm_head"), h)?
    };
    output_tensors.push(("logits".to_string(), out));

    println!("Built graph, exporting...");
    let onnx_model =
        crate::onnx_graph::build_proto(&input_tensors, &output_tensors, output_method)?;
    Ok(onnx_model.encode_to_vec())
}
