use crate::Error;
use crate::onnx_graph::WeightStorageStrategy;
use crate::onnx_graph::operators::{
    Add, Concat, Constant, Gather, MatMul, Mul, RotaryEmbedding, ShapeOp, Slice, Softmax,
    Transpose,
};
use crate::onnx_graph::pytorch::{
    div_scalar, linear, reshape, rms_norm, silu, transpose, unsqueeze,
};
use crate::onnx_graph::tensor::{
    DType, Dimension, InputTensor, InputTensorInitialized, Shape, Tensor, TensorData,
    TensorDataValue,
};
use crate::onnx_graph::weights::WeightManager;
use prost::Message;
use std::sync::Arc;

pub struct DeepseekV2Config {
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub hidden_size: usize,
    pub kv_lora_rank: usize,
    pub q_lora_rank: Option<usize>,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub n_shared_experts: usize,
    pub first_k_dense_replace: usize,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
}

impl DeepseekV2Config {
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
        let hidden_size = get_int(config, "hidden_size")? as usize;
        let kv_lora_rank = get_int(config, "kv_lora_rank")? as usize;
        let q_lora_rank = config
            .get("q_lora_rank")
            .and_then(|v| v.as_i64())
            .map(|v| v as usize);
        let qk_nope_head_dim = get_int(config, "qk_nope_head_dim")? as usize;
        let qk_rope_head_dim = get_int(config, "qk_rope_head_dim")? as usize;
        let v_head_dim = get_int(config, "v_head_dim")? as usize;
        let intermediate_size = get_int(config, "intermediate_size")? as usize;
        let moe_intermediate_size = config
            .get("moe_intermediate_size")
            .and_then(|v| v.as_i64())
            .unwrap_or(0) as usize;
        let n_shared_experts = config
            .get("n_shared_experts")
            .and_then(|v| v.as_i64())
            .unwrap_or(0) as usize;
        let first_k_dense_replace = config
            .get("first_k_dense_replace")
            .and_then(|v| v.as_i64())
            .unwrap_or(0) as usize;
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
            hidden_size,
            kv_lora_rank,
            q_lora_rank,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
            intermediate_size,
            moe_intermediate_size,
            n_shared_experts,
            first_k_dense_replace,
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
    let const_shape = Shape::new(vec![Dimension::new(Some(1), None, None)]);
    let start_const = Constant::new(None, TensorData::new(vec![start].into(), const_shape.clone())?);
    let end_const = Constant::new(None, TensorData::new(vec![end].into(), const_shape.clone())?);
    let axis_const = Constant::new(None, TensorData::new(vec![axis].into(), const_shape)?);
    Slice::new(None, input, start_const, end_const, Some(axis_const), None)
}

pub fn load_deepseek_v2(
    weight_manager: impl WeightManager,
    config: DeepseekV2Config,
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

    let qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim;
    let kv_cache_input_type = x.dtype();

    // K cache: [B, nH, S, qk_nope_head_dim + qk_rope_head_dim]
    let k_cache_shape = Shape::new(vec![
        batch_dimension.clone(),
        Dimension::new(Some(config.num_attention_heads), None, None),
        kv_cache_seq_dim.clone(),
        Dimension::new(Some(qk_head_dim), None, None),
    ]);
    // V cache: [B, nH, S, v_head_dim]
    let v_cache_shape = Shape::new(vec![
        batch_dimension.clone(),
        Dimension::new(Some(config.num_attention_heads), None, None),
        kv_cache_seq_dim,
        Dimension::new(Some(config.v_head_dim), None, None),
    ]);

    // Precompute RoPE frequency tables for the rope portion only.
    // cos/sin shape: [max_len, qk_rope_head_dim / 2]
    let half_rope_dim = config.qk_rope_head_dim / 2;
    // Cap at original_max_position_embeddings for standard RoPE (YaRN not yet implemented)
    let max_len = config.max_position_embeddings.min(4096);
    let (cos_values, sin_values) = {
        let inv_freq: Vec<f64> = (0..half_rope_dim)
            .map(|i| {
                1.0 / config
                    .rope_theta
                    .powf(i as f64 * 2.0 / config.qk_rope_head_dim as f64)
            })
            .collect();
        let mut cos_vals = Vec::with_capacity(max_len * half_rope_dim);
        let mut sin_vals = Vec::with_capacity(max_len * half_rope_dim);
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
        Dimension::new(Some(half_rope_dim), None, None),
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

        // === MLA Attention ===

        // Q path
        let q: Arc<dyn Tensor> = if let Some(_q_lora_rank) = config.q_lora_rank {
            // V3-style: q_a_proj → layernorm → q_b_proj
            let q_compressed = linear(
                &layer_weight_manager.prefix("self_attn.q_a_proj"),
                att_norm.clone(),
            )?;
            let q_compressed = rms_norm(
                &layer_weight_manager.prefix("self_attn.q_a_layernorm"),
                q_compressed,
                None,
            )?;
            linear(
                &layer_weight_manager.prefix("self_attn.q_b_proj"),
                q_compressed,
            )?
        } else {
            // V2-Lite: direct q_proj
            linear(
                &layer_weight_manager.prefix("self_attn.q_proj"),
                att_norm.clone(),
            )?
        };

        // q: [B, S, nH * qk_head_dim] → [B, nH, S, qk_head_dim]
        let q = Transpose::new(
            None,
            reshape(
                q,
                vec![
                    0,
                    0,
                    config.num_attention_heads as i64,
                    qk_head_dim as i64,
                ],
            )?,
            Some(vec![0, 2, 1, 3]),
        );
        // Split Q into nope and rope portions
        let q_nope: Arc<dyn Tensor> =
            slice_axis(q.clone(), 3, 0, config.qk_nope_head_dim as i64)?;
        let q_rope: Arc<dyn Tensor> = slice_axis(q, 3, config.qk_nope_head_dim as i64, qk_head_dim as i64)?;

        // KV path: compress then decompress
        // kv_a_proj_with_mqa: [B, S, hidden] → [B, S, kv_lora_rank + qk_rope_head_dim]
        let kv_a = linear(
            &layer_weight_manager.prefix("self_attn.kv_a_proj_with_mqa"),
            att_norm,
        )?;
        // Split into compressed KV latent and shared rope key
        let c_kv: Arc<dyn Tensor> = slice_axis(kv_a.clone(), 2, 0, config.kv_lora_rank as i64)?;
        let k_rope_shared: Arc<dyn Tensor> = slice_axis(
            kv_a,
            2,
            config.kv_lora_rank as i64,
            (config.kv_lora_rank + config.qk_rope_head_dim) as i64,
        )?;

        // RMSNorm on the compressed KV latent
        let c_kv = rms_norm(
            &layer_weight_manager.prefix("self_attn.kv_a_layernorm"),
            c_kv,
            None,
        )?;

        // Decompress: kv_b_proj: [B, S, kv_lora_rank] → [B, S, nH * (qk_nope_head_dim + v_head_dim)]
        let kv_b = linear(
            &layer_weight_manager.prefix("self_attn.kv_b_proj"),
            c_kv,
        )?;
        let nope_plus_v = config.qk_nope_head_dim + config.v_head_dim;
        // kv_b: [B, S, nH * (nope + v)] → [B, nH, S, nope + v]
        let kv_b = Transpose::new(
            None,
            reshape(
                kv_b,
                vec![
                    0,
                    0,
                    config.num_attention_heads as i64,
                    nope_plus_v as i64,
                ],
            )?,
            Some(vec![0, 2, 1, 3]),
        );
        let k_nope: Arc<dyn Tensor> =
            slice_axis(kv_b.clone(), 3, 0, config.qk_nope_head_dim as i64)?;
        let v: Arc<dyn Tensor> = slice_axis(
            kv_b,
            3,
            config.qk_nope_head_dim as i64,
            nope_plus_v as i64,
        )?;

        // k_rope_shared: [B, S, rope_dim] → [B, 1, S, rope_dim]
        let k_rope_shared = unsqueeze(k_rope_shared, 1)?;

        // KV cache inputs
        let kv_cache_input_k = InputTensor::new(
            format!("kv_cache_input_k_{i}"),
            kv_cache_input_type,
            k_cache_shape.clone(),
        );
        let kv_cache_input_v = InputTensor::new(
            format!("kv_cache_input_v_{i}"),
            kv_cache_input_type,
            v_cache_shape.clone(),
        );

        // RoPE position from KV cache sequence length
        let rope_pos = ShapeOp::new(None, kv_cache_input_k.clone(), Some(2), Some(3))?;
        let cos_at_pos = Gather::new(None, cos_cache.clone(), rope_pos.clone(), 0)?;
        let sin_at_pos = Gather::new(None, sin_cache.clone(), rope_pos, 0)?;
        let cos_at_pos = reshape(cos_at_pos, vec![1, 1, half_rope_dim as i64])?;
        let sin_at_pos = reshape(sin_at_pos, vec![1, 1, half_rope_dim as i64])?;

        // Apply RoPE to q_rope [B, nH, S, rope_dim] and k_rope [B, 1, S, rope_dim]
        let q_rope = RotaryEmbedding::new(
            None,
            q_rope,
            cos_at_pos.clone(),
            sin_at_pos.clone(),
            None,
            None,
            None,
            None,
        )?;
        let k_rope = RotaryEmbedding::new(
            None,
            k_rope_shared,
            cos_at_pos,
            sin_at_pos,
            None,
            None,
            None,
            None,
        )?;

        // Broadcast k_rope from [B, 1, S, rope_dim] to [B, nH, S, rope_dim]
        let k_rope: Arc<dyn Tensor> =
            Concat::new(None, vec![k_rope.clone(); config.num_attention_heads], 1)?;

        // Assemble full Q and K
        // Q = concat(q_nope, q_rope) → [B, nH, S, qk_head_dim]
        let q: Arc<dyn Tensor> = Concat::new(None, vec![q_nope, q_rope], 3)?;
        // K = concat(k_nope, k_rope) → [B, nH, S, qk_head_dim]
        let k: Arc<dyn Tensor> = Concat::new(None, vec![k_nope, k_rope], 3)?;

        // Append to KV cache
        let new_seq_len_dim = Dimension::new(None, Some("new_seq_len".to_string()), None);
        let k_new_shape = Shape::new(vec![
            k.shape()[0].clone(),
            k.shape()[1].clone(),
            new_seq_len_dim.clone(),
            k.shape()[3].clone(),
        ]);
        let v_new_shape = Shape::new(vec![
            v.shape()[0].clone(),
            v.shape()[1].clone(),
            new_seq_len_dim,
            v.shape()[3].clone(),
        ]);
        let k = Concat::new_with_output_shape(
            None,
            vec![kv_cache_input_k.clone(), k],
            2,
            k_new_shape,
        )?;
        let v = Concat::new_with_output_shape(
            None,
            vec![kv_cache_input_v.clone(), v],
            2,
            v_new_shape,
        )?;

        input_tensors.push(kv_cache_input_k);
        output_tensors.push((format!("kv_cache_output_k_{i}"), k.clone()));
        input_tensors.push(kv_cache_input_v);
        output_tensors.push((format!("kv_cache_output_v_{i}"), v.clone()));

        // Attention scores
        let scores = MatMul::new(None, q, transpose(k))?;
        let scores = div_scalar(scores, (qk_head_dim as f32).sqrt())?;
        let scores = Softmax::new(None, scores, Some(3));

        // Attention output: [B, nH, S, v_head_dim]
        let output = MatMul::new(None, scores, v)?;
        let output = Transpose::new(None, output, Some(vec![0, 2, 1, 3]));
        let output = reshape(
            output,
            vec![0, 0, (config.num_attention_heads * config.v_head_dim) as i64],
        )?;
        let hidden_layer = linear(&layer_weight_manager.prefix("self_attn.o_proj"), output)?;

        let attention_output = Add::new(None, layer_input, hidden_layer)?;

        // === Feed-Forward ===
        let ffn_norm = rms_norm(
            &layer_weight_manager.prefix("post_attention_layernorm"),
            attention_output.clone(),
            None,
        )?;

        let is_dense_layer = i < config.first_k_dense_replace;
        let ffn_output: Arc<dyn Tensor> = if is_dense_layer {
            // Dense MLP: gate_proj, up_proj, down_proj
            let gate = linear(
                &layer_weight_manager.prefix("mlp.gate_proj"),
                ffn_norm.clone(),
            )?;
            let gate = silu(gate)?;
            let up = linear(
                &layer_weight_manager.prefix("mlp.up_proj"),
                ffn_norm,
            )?;
            let hidden = Mul::new(None, gate, up)?;
            linear(&layer_weight_manager.prefix("mlp.down_proj"), hidden)?
        } else {
            // MoE layer — shared experts only (routed experts TODO)
            let shared_wm = layer_weight_manager.prefix("mlp.shared_experts");
            let gate = linear(&shared_wm.prefix("gate_proj"), ffn_norm.clone())?;
            let gate = silu(gate)?;
            let up = linear(&shared_wm.prefix("up_proj"), ffn_norm)?;
            let hidden = Mul::new(None, gate, up)?;
            linear(&shared_wm.prefix("down_proj"), hidden)?
        };

        layer_output = Add::new(None, attention_output, ffn_output)?;
    }

    let h = rms_norm(&model_weight_manager.prefix("norm"), layer_output, None)?;

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
