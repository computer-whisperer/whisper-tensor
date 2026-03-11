use crate::onnx_graph::Error;
use crate::onnx_graph::WeightStorageStrategy;
use crate::onnx_graph::operators::{Add, Gather, MatMul, ShapeOp, Softmax, Transpose};
use crate::onnx_graph::pytorch::{conv1d, gelu, layer_norm, linear, reshape};
use crate::onnx_graph::tensor::{DType, Dimension, InputTensor, Shape, Tensor};
use crate::onnx_graph::weights::WeightManager;
use prost::Message;
use std::path::Path;
use std::sync::Arc;

/// Whisper model configuration (parsed from config.json).
#[derive(Clone)]
pub struct WhisperConfig {
    pub d_model: usize,
    pub encoder_layers: usize,
    pub encoder_attention_heads: usize,
    pub encoder_ffn_dim: usize,
    pub decoder_layers: usize,
    pub decoder_attention_heads: usize,
    pub decoder_ffn_dim: usize,
    pub num_mel_bins: usize,
    pub max_source_positions: usize,
    pub max_target_positions: usize,
    pub vocab_size: usize,
}

/// Build the Whisper encoder as ONNX model bytes.
///
/// Input: `input_features` [1, num_mel_bins, seq_len] (mel spectrogram)
/// Output: `last_hidden_state` [1, seq_len/2, d_model]
pub fn build_encoder(
    wm: &impl WeightManager,
    config: &WhisperConfig,
    output_method: WeightStorageStrategy,
    origin_path: &Path,
) -> Result<Vec<u8>, anyhow::Error> {
    let enc_wm = wm.prefix("model.encoder");

    // Input: mel spectrogram [1, 80, 3000]
    let mel_dim = Dimension::new(Some(config.num_mel_bins), None, None);
    let seq_dim = Dimension::new(None, Some("source_len".to_string()), None);
    let batch_dim = Dimension::new(Some(1), Some("batch".to_string()), None);
    let input_shape = Shape::new(vec![batch_dim.clone(), mel_dim, seq_dim]);
    let input_features = InputTensor::new("input_features".to_string(), DType::F32, input_shape);

    // Conv1d front-end: conv1(80→d_model, k=3, s=1, p=1) + GELU + conv2(d_model→d_model, k=3, s=2, p=1) + GELU
    let x = conv1d(&enc_wm.prefix("conv1"), input_features.clone(), 3, 1, 1)?;
    let x = gelu(x)?;
    let x = conv1d(&enc_wm.prefix("conv2"), x, 3, 2, 1)?;
    let x = gelu(x)?;

    // Transpose from [B, C, T] to [B, T, C] for transformer
    let x = Transpose::new(None, x, Some(vec![0, 2, 1]));

    // Add sinusoidal position embeddings
    let pos_emb = enc_wm.get_tensor("embed_positions.weight")?;
    let x: Arc<dyn Tensor> = Add::new(Some("pos_embed".to_string()), x, pos_emb)?;

    // Encoder transformer layers
    let mut hidden = x;
    for i in 0..config.encoder_layers {
        let layer_wm = enc_wm.prefix(&format!("layers.{i}"));
        hidden = whisper_encoder_layer(&layer_wm, hidden, config)?;
    }

    // Final layer norm
    let output: Arc<dyn Tensor> = layer_norm(&enc_wm.prefix("layer_norm"), hidden, 1e-5)?;

    let input_tensors: Vec<Arc<dyn Tensor>> = vec![input_features];
    let output_tensors = vec![("last_hidden_state".to_string(), output)];

    let onnx_model = crate::onnx_graph::build_proto_with_origin_path(
        &input_tensors,
        &output_tensors,
        output_method,
        Some(origin_path),
    )?;
    Ok(onnx_model.encode_to_vec())
}

/// Build the Whisper decoder as ONNX model bytes.
///
/// Inputs:
/// - `decoder_input_ids` [1, tgt_len] (token IDs)
/// - `encoder_hidden_states` [1, src_len, d_model]
/// - KV cache state inputs (for autoregressive inference)
///
/// Outputs:
/// - `logits` [1, tgt_len, vocab_size]
/// - KV cache state outputs
pub fn build_decoder(
    wm: &impl WeightManager,
    config: &WhisperConfig,
    output_method: WeightStorageStrategy,
    origin_path: &Path,
) -> Result<Vec<u8>, anyhow::Error> {
    let dec_wm = wm.prefix("model.decoder");
    let head_dim = config.d_model / config.decoder_attention_heads;

    // Inputs
    let batch_dim = Dimension::new(Some(1), Some("batch".to_string()), None);
    let tgt_dim = Dimension::new(None, Some("tgt_len".to_string()), None);
    let src_dim = Dimension::new(None, Some("src_len".to_string()), None);
    let hidden_dim = Dimension::new(Some(config.d_model), None, None);

    let decoder_input_ids = InputTensor::new(
        "decoder_input_ids".to_string(),
        DType::I64,
        Shape::new(vec![batch_dim.clone(), tgt_dim.clone()]),
    );
    let encoder_hidden_states = InputTensor::new(
        "encoder_hidden_states".to_string(),
        DType::F32,
        Shape::new(vec![batch_dim.clone(), src_dim, hidden_dim]),
    );

    // Create KV cache state inputs/outputs
    let mut state_inputs: Vec<Arc<dyn Tensor>> = Vec::new();
    let mut state_outputs: Vec<(String, Arc<dyn Tensor>)> = Vec::new();

    // Create first layer's self_k_cache early to derive position from cache length
    let self_k_cache_0 = InputTensor::new(
        "self_k_cache_0".to_string(),
        DType::F32,
        Shape::new(vec![
            batch_dim.clone(),
            Dimension::new(Some(config.decoder_attention_heads), None, None),
            Dimension::new(None, Some("past_self_0".to_string()), None),
            Dimension::new(Some(head_dim), None, None),
        ]),
    );

    // Token + position embeddings
    let embed_weight = dec_wm.get_tensor("embed_tokens.weight")?;
    let token_emb = Gather::new(
        Some("token_embed".to_string()),
        embed_weight.clone(),
        decoder_input_ids.clone(),
        0,
    )?;

    // Derive position from self-attention KV cache length:
    // Shape(cache)[2:3] → [1] containing past_len
    // Gather(pos_emb_weight, past_len, axis=0) → [1, d_model]
    let cache_seq_len = ShapeOp::new(None, self_k_cache_0.clone(), Some(2), Some(3))?;
    let pos_emb_weight = dec_wm.get_tensor("embed_positions.weight")?;
    let pos_emb = Gather::new(Some("dec_pos_gather".to_string()), pos_emb_weight, cache_seq_len, 0)?;
    let x: Arc<dyn Tensor> = Add::new(Some("dec_pos_embed".to_string()), token_emb, pos_emb)?;

    // Decoder transformer layers
    let mut hidden = x;
    for i in 0..config.decoder_layers {
        let layer_wm = dec_wm.prefix(&format!("layers.{i}"));

        // Self-attention KV cache (reuse self_k_cache_0 for layer 0)
        let self_k_in: Arc<dyn Tensor> = if i == 0 {
            self_k_cache_0.clone()
        } else {
            InputTensor::new(
                format!("self_k_cache_{i}"),
                DType::F32,
                Shape::new(vec![
                    batch_dim.clone(),
                    Dimension::new(Some(config.decoder_attention_heads), None, None),
                    Dimension::new(None, Some(format!("past_self_{i}")), None),
                    Dimension::new(Some(head_dim), None, None),
                ]),
            )
        };
        let self_v_in = InputTensor::new(
            format!("self_v_cache_{i}"),
            DType::F32,
            Shape::new(vec![
                batch_dim.clone(),
                Dimension::new(Some(config.decoder_attention_heads), None, None),
                Dimension::new(None, Some(format!("past_self_{i}")), None),
                Dimension::new(Some(head_dim), None, None),
            ]),
        );

        state_inputs.push(self_k_in.clone());
        state_inputs.push(self_v_in.clone());

        let (layer_out, self_k_out, self_v_out) =
            whisper_decoder_layer(
                &layer_wm,
                hidden,
                encoder_hidden_states.clone(),
                self_k_in,
                self_v_in,
                config,
            )?;

        hidden = layer_out;

        state_outputs.push((format!("self_k_cache_out_{i}"), self_k_out));
        state_outputs.push((format!("self_v_cache_out_{i}"), self_v_out));
    }

    // Final layer norm
    let hidden: Arc<dyn Tensor> = layer_norm(&dec_wm.prefix("layer_norm"), hidden, 1e-5)?;

    // Output projection (tied to embed_tokens.weight — reuse the same tensor)
    let logits = linear_with_weight(hidden, embed_weight)?;

    // Assemble inputs/outputs
    let mut input_tensors: Vec<Arc<dyn Tensor>> = vec![decoder_input_ids, encoder_hidden_states];
    input_tensors.extend(state_inputs);

    let mut output_tensors = vec![("logits".to_string(), logits)];
    output_tensors.extend(state_outputs);

    let onnx_model = crate::onnx_graph::build_proto_with_origin_path(
        &input_tensors,
        &output_tensors,
        output_method,
        Some(origin_path),
    )?;
    Ok(onnx_model.encode_to_vec())
}

// ============================================================================
// Encoder layer
// ============================================================================

fn whisper_encoder_layer(
    wm: &impl WeightManager,
    input: Arc<dyn Tensor>,
    config: &WhisperConfig,
) -> Result<Arc<dyn Tensor>, Error> {
    // Pre-norm self-attention
    let normed: Arc<dyn Tensor> =
        layer_norm(&wm.prefix("self_attn_layer_norm"), input.clone(), 1e-5)?;
    let attn_out = self_attention(&wm.prefix("self_attn"), normed, config)?;
    let x: Arc<dyn Tensor> = Add::new(None, input, attn_out)?;

    // Pre-norm FFN
    let normed: Arc<dyn Tensor> = layer_norm(&wm.prefix("final_layer_norm"), x.clone(), 1e-5)?;
    let ffn_out = whisper_ffn(wm, normed)?;
    Ok(Add::new(None, x, ffn_out)?)
}

// ============================================================================
// Decoder layer
// ============================================================================

#[allow(clippy::type_complexity)]
fn whisper_decoder_layer(
    wm: &impl WeightManager,
    input: Arc<dyn Tensor>,
    encoder_hidden: Arc<dyn Tensor>,
    self_k_cache: Arc<dyn Tensor>,
    self_v_cache: Arc<dyn Tensor>,
    config: &WhisperConfig,
) -> Result<(Arc<dyn Tensor>, Arc<dyn Tensor>, Arc<dyn Tensor>), Error> {
    // Self-attention with KV cache
    let normed: Arc<dyn Tensor> =
        layer_norm(&wm.prefix("self_attn_layer_norm"), input.clone(), 1e-5)?;
    let (attn_out, self_k_out, self_v_out) =
        self_attention_with_cache(&wm.prefix("self_attn"), normed, self_k_cache, self_v_cache, config)?;
    let x: Arc<dyn Tensor> = Add::new(None, input, attn_out)?;

    // Cross-attention (no cache — K/V from encoder are constant)
    let normed: Arc<dyn Tensor> =
        layer_norm(&wm.prefix("encoder_attn_layer_norm"), x.clone(), 1e-5)?;
    let cross_out = cross_attention(
        &wm.prefix("encoder_attn"),
        normed,
        encoder_hidden,
        config,
    )?;
    let x: Arc<dyn Tensor> = Add::new(None, x, cross_out)?;

    // FFN
    let normed: Arc<dyn Tensor> = layer_norm(&wm.prefix("final_layer_norm"), x.clone(), 1e-5)?;
    let ffn_out = whisper_ffn(wm, normed)?;
    let x: Arc<dyn Tensor> = Add::new(None, x, ffn_out)?;

    Ok((x, self_k_out, self_v_out))
}

// ============================================================================
// Attention
// ============================================================================

/// Self-attention without KV cache (encoder).
fn self_attention(
    wm: &impl WeightManager,
    input: Arc<dyn Tensor>,
    config: &WhisperConfig,
) -> Result<Arc<dyn Tensor>, Error> {
    let num_heads = config.encoder_attention_heads;
    let head_dim = config.d_model / num_heads;

    let q = linear(&wm.prefix("q_proj"), input.clone())?;
    let k = linear(&wm.prefix("k_proj"), input.clone())?;
    let v = linear(&wm.prefix("v_proj"), input)?;

    let attn_out = scaled_dot_product_attention(q, k, v, num_heads, head_dim, None)?;
    linear(&wm.prefix("out_proj"), attn_out)
}

/// Self-attention with KV cache append (decoder).
#[allow(clippy::type_complexity)]
fn self_attention_with_cache(
    wm: &impl WeightManager,
    input: Arc<dyn Tensor>,
    k_cache: Arc<dyn Tensor>,
    v_cache: Arc<dyn Tensor>,
    config: &WhisperConfig,
) -> Result<(Arc<dyn Tensor>, Arc<dyn Tensor>, Arc<dyn Tensor>), Error> {
    let num_heads = config.decoder_attention_heads;
    let head_dim = config.d_model / num_heads;

    let q = linear(&wm.prefix("q_proj"), input.clone())?;
    let k_new = linear(&wm.prefix("k_proj"), input.clone())?;
    let v_new = linear(&wm.prefix("v_proj"), input)?;

    // Reshape new K, V to [B, num_heads, new_len, head_dim]
    let k_new = reshape_heads(k_new, num_heads, head_dim)?;
    let v_new = reshape_heads(v_new, num_heads, head_dim)?;

    // Concat with cache: [B, num_heads, past+new, head_dim]
    let k_full = crate::onnx_graph::operators::Concat::new(
        None,
        vec![k_cache, k_new.clone()],
        2,
    )?;
    let v_full = crate::onnx_graph::operators::Concat::new(
        None,
        vec![v_cache, v_new.clone()],
        2,
    )?;

    // Q: reshape to heads
    let q = reshape_heads(q, num_heads, head_dim)?;

    let attn_out = sdpa_from_heads(q, k_full.clone(), v_full.clone(), num_heads, head_dim, None)?;
    let output = linear(&wm.prefix("out_proj"), attn_out)?;

    Ok((output, k_full, v_full))
}

/// Cross-attention (decoder attending to encoder).
/// K/V are always computed fresh from encoder_hidden_states — no cache needed.
fn cross_attention(
    wm: &impl WeightManager,
    input: Arc<dyn Tensor>,
    encoder_hidden: Arc<dyn Tensor>,
    config: &WhisperConfig,
) -> Result<Arc<dyn Tensor>, Error> {
    let num_heads = config.decoder_attention_heads;
    let head_dim = config.d_model / num_heads;

    let q = linear(&wm.prefix("q_proj"), input)?;
    let k = linear(&wm.prefix("k_proj"), encoder_hidden.clone())?;
    let v = linear(&wm.prefix("v_proj"), encoder_hidden)?;

    let attn_out = scaled_dot_product_attention(q, k, v, num_heads, head_dim, None)?;
    linear(&wm.prefix("out_proj"), attn_out)
}

// ============================================================================
// Helpers
// ============================================================================

/// Reshape [B, T, D] → [B, num_heads, T, head_dim] for multi-head attention.
fn reshape_heads(
    x: Arc<dyn Tensor>,
    num_heads: usize,
    head_dim: usize,
) -> Result<Arc<dyn Tensor>, Error> {
    let reshaped = reshape(x, vec![0, -1, num_heads as i64, head_dim as i64])?;
    Ok(Transpose::new(None, reshaped, Some(vec![0, 2, 1, 3])))
}

/// Scaled dot-product attention from raw Q, K, V (not yet reshaped to heads).
fn scaled_dot_product_attention(
    q: Arc<dyn Tensor>,
    k: Arc<dyn Tensor>,
    v: Arc<dyn Tensor>,
    num_heads: usize,
    head_dim: usize,
    mask: Option<Arc<dyn Tensor>>,
) -> Result<Arc<dyn Tensor>, Error> {
    let q = reshape_heads(q, num_heads, head_dim)?;
    let k = reshape_heads(k, num_heads, head_dim)?;
    let v = reshape_heads(v, num_heads, head_dim)?;
    sdpa_from_heads(q, k, v, num_heads, head_dim, mask)
}

/// SDPA from already-reshaped heads [B, num_heads, T, head_dim].
fn sdpa_from_heads(
    q: Arc<dyn Tensor>,
    k: Arc<dyn Tensor>,
    v: Arc<dyn Tensor>,
    num_heads: usize,
    head_dim: usize,
    mask: Option<Arc<dyn Tensor>>,
) -> Result<Arc<dyn Tensor>, Error> {
    use crate::onnx_graph::pytorch::div_scalar;

    let kt = Transpose::new(None, k, Some(vec![0, 1, 3, 2]));
    let scores = MatMul::new(None, q, kt)?;
    let scores = div_scalar(scores, (head_dim as f32).sqrt())?;

    let scores: Arc<dyn Tensor> = if let Some(mask) = mask {
        Add::new(None, scores, mask)?
    } else {
        scores
    };

    let attn_weights = Softmax::new(None, scores, Some(-1));
    let attn_output = MatMul::new(None, attn_weights, v)?;

    let attn_output = Transpose::new(None, attn_output, Some(vec![0, 2, 1, 3]));
    // Reshape back to [B, T, D]
    let d_model = head_dim * num_heads;
    Ok(reshape(attn_output, vec![0, -1, d_model as i64])?)
}

/// FFN: fc1 + gelu + fc2
fn whisper_ffn(
    wm: &impl WeightManager,
    input: Arc<dyn Tensor>,
) -> Result<Arc<dyn Tensor>, Error> {
    let x = linear(&wm.prefix("fc1"), input)?;
    let x = gelu(x)?;
    linear(&wm.prefix("fc2"), x)
}

/// Linear with explicit weight tensor (for tied embeddings).
fn linear_with_weight(
    input: Arc<dyn Tensor>,
    weight: Arc<dyn Tensor>,
) -> Result<Arc<dyn Tensor>, Error> {
    use crate::onnx_graph::pytorch::transpose;
    let wt = transpose(weight);
    Ok(MatMul::new(None, input, wt)?)
}
