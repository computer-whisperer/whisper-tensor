use crate::onnx_graph::WeightStorageStrategy;
use crate::onnx_graph::operators::{Add, Gather, MatMul, Mul, Softmax, Transpose};
use crate::onnx_graph::pytorch::{gelu, linear, reshape, rms_norm, transpose, unsqueeze};
use crate::onnx_graph::tensor::{
    DType, Dimension, InputTensor, InputTensorInitialized, Shape, Tensor, TensorData,
    TensorDataValue,
};
use crate::onnx_graph::weights::WeightManager;
use crate::sd_common::CastingWeightManager;
use prost::Message;
use std::sync::Arc;

pub struct T5Config {
    pub num_layers: usize,
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub ff_dim: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub rms_norm_epsilon: f32,
    pub relative_attention_num_buckets: usize,
    pub relative_attention_max_distance: usize,
}

impl T5Config {
    /// T5-XXL configuration as used in Flux.
    pub fn t5_xxl(max_seq_len: usize) -> Self {
        Self {
            num_layers: 24,
            hidden_dim: 4096,
            num_heads: 64,
            ff_dim: 10240,
            vocab_size: 32128,
            max_seq_len,
            rms_norm_epsilon: 1e-6,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
        }
    }
}

/// Compute the T5 relative position bias bucket indices.
///
/// Returns a [seq_len, seq_len] table of bucket indices.
fn compute_relative_position_buckets(
    seq_len: usize,
    num_buckets: usize,
    max_distance: usize,
) -> Vec<i32> {
    let mut buckets = vec![0i32; seq_len * seq_len];
    let half_buckets = num_buckets / 2;
    let max_exact = half_buckets / 2;
    let log_ratio = (max_distance as f64 / max_exact as f64).ln();

    for query_pos in 0..seq_len {
        for key_pos in 0..seq_len {
            let relative_position = key_pos as i64 - query_pos as i64;
            let is_negative = relative_position < 0;
            let abs_pos = relative_position.unsigned_abs() as usize;

            // Bidirectional: first half_buckets for positive, second for negative
            let mut bucket = if is_negative { half_buckets } else { 0 };

            if abs_pos < max_exact {
                bucket += abs_pos;
            } else {
                let val = max_exact as f64
                    + ((abs_pos as f64 / max_exact as f64).ln() / log_ratio)
                        * (half_buckets - max_exact) as f64;
                bucket += val.min((half_buckets - 1) as f64) as usize;
            }

            buckets[query_pos * seq_len + key_pos] = bucket as i32;
        }
    }
    buckets
}

pub fn load_t5_encoder(
    weight_manager: impl WeightManager,
    config: T5Config,
    output_method: WeightStorageStrategy,
) -> Result<Vec<u8>, anyhow::Error> {
    load_t5_encoder_with_origin(weight_manager, config, output_method, None)
}

pub fn load_t5_encoder_with_origin(
    weight_manager: impl WeightManager,
    config: T5Config,
    output_method: WeightStorageStrategy,
    origin_path: Option<&std::path::Path>,
) -> Result<Vec<u8>, anyhow::Error> {
    // T5-XXL must compute in F32 — F16 overflows after ~10 layers due to
    // unscaled attention and large hidden dim (4096) accumulations.
    let weight_manager = CastingWeightManager::new(weight_manager, DType::F32);

    let encoder_wm = weight_manager.prefix("encoder");
    let head_dim = config.hidden_dim / config.num_heads;

    let mut input_tensors: Vec<Arc<dyn Tensor>> = vec![];
    let mut output_tensors: Vec<(String, Arc<dyn Tensor>)> = vec![];

    let batch_dimension = Dimension::new(
        Some(1),
        Some("batch_size".to_string()),
        Some("DATA_BATCH".to_string()),
    );
    let sequence_dimension =
        Dimension::new(Some(config.max_seq_len), Some("seq_len".to_string()), None);

    let input_shape = Shape::new(vec![batch_dimension.clone(), sequence_dimension.clone()]);
    let token_input = InputTensor::new("input_ids".to_string(), DType::I32, input_shape);
    input_tensors.push(token_input.clone());

    // Token embedding (T5 uses shared.weight = encoder.embed_tokens.weight)
    // Try shared.weight first, fall back to encoder.embed_tokens.weight
    let embed_weight = weight_manager
        .get_tensor("shared.weight")
        .or_else(|_| encoder_wm.get_tensor("embed_tokens.weight"))?;

    let x = Gather::new(
        Some("embed_tokens".to_string()),
        embed_weight,
        token_input.clone(),
        0,
    )?;

    // Precompute relative position bias table.
    // bucket_indices: [max_seq, max_seq] I32
    // bias_weight: [num_buckets, num_heads] from block 0
    // Gather bucket_indices from bias_weight → [max_seq, max_seq, num_heads]
    // Transpose to [num_heads, max_seq, max_seq], unsqueeze batch → [1, num_heads, max_seq, max_seq]
    let position_bias: Arc<dyn Tensor> = {
        let bias_weight = encoder_wm
            .get_tensor("block.0.layer.0.SelfAttention.relative_attention_bias.weight")?;

        let bucket_indices = compute_relative_position_buckets(
            config.max_seq_len,
            config.relative_attention_num_buckets,
            config.relative_attention_max_distance,
        );
        let bucket_shape = Shape::new(vec![
            Dimension::new(Some(config.max_seq_len), None, None),
            Dimension::new(Some(config.max_seq_len), None, None),
        ]);
        let bucket_data = TensorData::new(TensorDataValue::I32(bucket_indices), bucket_shape)?;
        let bucket_const = InputTensorInitialized::new("position_buckets".to_string(), bucket_data);

        // Gather: [num_buckets, num_heads] with [max_seq, max_seq] indices at axis=0
        // → [max_seq, max_seq, num_heads]
        let bias = Gather::new(None, bias_weight, bucket_const, 0)?;
        // Transpose to [num_heads, max_seq, max_seq]
        let bias = Transpose::new(None, bias, Some(vec![2, 0, 1]));
        // Unsqueeze to [1, num_heads, max_seq, max_seq] for broadcasting with attention scores
        unsqueeze(bias, 0)?
    };

    let mut layer_output: Arc<dyn Tensor> = x;
    for i in 0..config.num_layers {
        let block_wm = encoder_wm.prefix(&format!("block.{i}"));
        let block_input = layer_output.clone();

        // --- Self-attention ---
        let attn_wm = block_wm.prefix("layer.0");
        let normed = rms_norm(
            &attn_wm.prefix("layer_norm"),
            block_input.clone(),
            Some(config.rms_norm_epsilon),
        )?;

        let q = linear(&attn_wm.prefix("SelfAttention.q"), normed.clone())?;
        let k = linear(&attn_wm.prefix("SelfAttention.k"), normed.clone())?;
        let v = linear(&attn_wm.prefix("SelfAttention.v"), normed)?;

        // Reshape to [batch, seq, num_heads, head_dim] then transpose to [batch, num_heads, seq, head_dim]
        let q = Transpose::new(
            None,
            reshape(q, vec![0, 0, config.num_heads as i64, head_dim as i64])?,
            Some(vec![0, 2, 1, 3]),
        );
        let k = Transpose::new(
            None,
            reshape(k, vec![0, 0, config.num_heads as i64, head_dim as i64])?,
            Some(vec![0, 2, 1, 3]),
        );
        let v = Transpose::new(
            None,
            reshape(v, vec![0, 0, config.num_heads as i64, head_dim as i64])?,
            Some(vec![0, 2, 1, 3]),
        );

        // T5 attention: Q @ K^T + position_bias (no 1/sqrt(d) scaling)
        let scores = MatMul::new(None, q, transpose(k))?;
        let scores = Add::new(None, scores, position_bias.clone())?;
        let scores = Softmax::new(None, scores, Some(-1));

        let attn_output = MatMul::new(None, scores, v)?;
        // Transpose back to [batch, seq, num_heads, head_dim] then reshape to [batch, seq, hidden]
        let attn_output = Transpose::new(None, attn_output, Some(vec![0, 2, 1, 3]));
        let attn_output = reshape(attn_output, vec![0, 0, config.hidden_dim as i64])?;

        // Output projection
        let attn_output = linear(&attn_wm.prefix("SelfAttention.o"), attn_output)?;

        // Residual
        let after_attn = Add::new(None, block_input, attn_output)?;

        // --- Gated FFN ---
        let ffn_wm = block_wm.prefix("layer.1");
        let normed = rms_norm(
            &ffn_wm.prefix("layer_norm"),
            after_attn.clone(),
            Some(config.rms_norm_epsilon),
        )?;

        // GeGLU: gate = gelu(wi_0(x)), value = wi_1(x), hidden = gate * value
        let gate = linear(&ffn_wm.prefix("DenseReluDense.wi_0"), normed.clone())?;
        let gate = gelu(gate)?;
        let value = linear(&ffn_wm.prefix("DenseReluDense.wi_1"), normed)?;
        let hidden = Mul::new(None, gate, value)?;
        let ffn_output = linear(&ffn_wm.prefix("DenseReluDense.wo"), hidden)?;

        // Residual
        layer_output = Add::new(None, after_attn, ffn_output)?;
    }

    // Final layer norm
    let hidden_states = rms_norm(
        &encoder_wm.prefix("final_layer_norm"),
        layer_output,
        Some(config.rms_norm_epsilon),
    )?;

    output_tensors.push(("hidden_states".to_string(), hidden_states));

    println!("Built T5 encoder graph, exporting...");
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
