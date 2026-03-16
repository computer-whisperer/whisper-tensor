use crate::models::diffusion::sd_common::{self, CastingWeightManager};
use crate::onnx_graph::Error;
use crate::onnx_graph::WeightStorageStrategy;
use crate::onnx_graph::operators::{Add, Gather, MatMul, Softmax, Transpose};
use crate::onnx_graph::pytorch::{gelu, layer_norm, linear, quick_gelu, reshape, unsqueeze};
use crate::onnx_graph::tensor::{
    DType, Dimension, InputTensor, InputTensorInitialized, Shape, Tensor, TensorData,
};
use crate::onnx_graph::weights::WeightManager;
use prost::Message;
use std::path::Path;
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
pub enum ClipMlpActivation {
    QuickGelu,
    Gelu,
}

#[derive(Clone, Copy, Debug)]
pub enum ClipHiddenStateSource {
    Penultimate,
    FinalPreLayerNorm,
    FinalLayerNorm,
}

#[derive(Clone, Copy, Debug)]
pub struct ClipTextModelConfig {
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub max_position: usize,
    pub layer_norm_eps: f32,
    pub mlp_activation: ClipMlpActivation,
    pub hidden_source: ClipHiddenStateSource,
    pub output_pooled_projection: bool,
}

pub fn build_clip_text_model_with_projection(
    weight_manager: impl WeightManager,
    model_dtype: DType,
    output_method: WeightStorageStrategy,
    origin_path: &Path,
    weight_prefix: &str,
    config: ClipTextModelConfig,
) -> Result<Vec<u8>, anyhow::Error> {
    let wm = if weight_prefix.is_empty() {
        CastingWeightManager::new(weight_manager, DType::F32)
    } else {
        CastingWeightManager::new(weight_manager.prefix(weight_prefix), DType::F32)
    };
    let emb_wm = wm.prefix("embeddings");
    let _ = model_dtype;

    let batch_dim = Dimension::new(Some(1), Some("batch_size".to_string()), None);
    let seq_dim = Dimension::new(Some(config.max_position), Some("seq_len".to_string()), None);
    let input_shape = Shape::new(vec![batch_dim.clone(), seq_dim.clone()]);

    let input_ids = InputTensor::new("input_ids".to_string(), DType::I32, input_shape);

    let eos_indices = if config.output_pooled_projection {
        Some(InputTensor::new(
            "eos_indices".to_string(),
            DType::I64,
            Shape::new(vec![batch_dim.clone()]),
        ))
    } else {
        None
    };

    // Token + position embedding.
    let token_emb = Gather::new(
        Some("token_embedding".to_string()),
        emb_wm.get_tensor("token_embedding.weight")?,
        input_ids.clone(),
        0,
    )?;
    let pos_emb = emb_wm.get_tensor("position_embedding.weight")?;
    let x = Add::new(Some("pos_embed".to_string()), token_emb, pos_emb)?;

    // Causal mask.
    let mask_data = sd_common::build_causal_mask(config.max_position);
    let causal_mask = InputTensorInitialized::new(
        "causal_mask".to_string(),
        TensorData::new(
            mask_data.into(),
            Shape::new(vec![
                Dimension::new(Some(1), None, None),
                Dimension::new(Some(1), None, None),
                Dimension::new(Some(config.max_position), None, None),
                Dimension::new(Some(config.max_position), None, None),
            ]),
        )?,
    );

    // Transformer stack.
    let mut hidden: Arc<dyn Tensor> = x;
    let mut penultimate: Option<Arc<dyn Tensor>> = None;
    for i in 0..config.num_layers {
        if i + 1 == config.num_layers {
            penultimate = Some(hidden.clone());
        }
        let layer_wm = wm.prefix(&format!("encoder.layers.{i}"));
        hidden = clip_encoder_layer(
            &layer_wm,
            hidden,
            causal_mask.clone(),
            config.hidden_dim,
            config.num_heads,
            config.mlp_activation,
        )?;
    }

    let final_pre_ln = hidden.clone();
    let final_normed = layer_norm(
        &wm.prefix("final_layer_norm"),
        hidden,
        config.layer_norm_eps,
    )?;

    let hidden_state = match config.hidden_source {
        ClipHiddenStateSource::Penultimate => {
            penultimate.ok_or_else(|| anyhow::anyhow!("CLIP has no penultimate hidden state"))?
        }
        ClipHiddenStateSource::FinalPreLayerNorm => final_pre_ln,
        ClipHiddenStateSource::FinalLayerNorm => final_normed.clone(),
    };

    let mut input_tensors: Vec<Arc<dyn Tensor>> = vec![input_ids];
    let mut output_tensors: Vec<(String, Arc<dyn Tensor>)> =
        vec![("last_hidden_state".to_string(), hidden_state)];

    if let Some(eos_indices) = eos_indices {
        let eos_2d = unsqueeze(eos_indices.clone(), 1)?;
        let eos_token = Gather::new(None, final_normed, eos_2d, 1)?;
        let pooled = reshape(eos_token, vec![0, -1])?;
        let pooled = sd_common::linear_no_bias(&wm.prefix("text_projection"), pooled)?;
        input_tensors.push(eos_indices);
        output_tensors.push(("pooled_output".to_string(), pooled));
    }

    let onnx_model = crate::onnx_graph::build_proto_with_origin_path(
        &input_tensors,
        &output_tensors,
        output_method,
        Some(origin_path),
    )?;
    Ok(onnx_model.encode_to_vec())
}

fn clip_encoder_layer(
    wm: &impl WeightManager,
    input: Arc<dyn Tensor>,
    causal_mask: Arc<dyn Tensor>,
    hidden_dim: usize,
    num_heads: usize,
    mlp_activation: ClipMlpActivation,
) -> Result<Arc<dyn Tensor>, Error> {
    let normed = layer_norm(&wm.prefix("layer_norm1"), input.clone(), 1e-5)?;
    let attn_out = clip_attention(
        &wm.prefix("self_attn"),
        normed,
        causal_mask,
        hidden_dim,
        num_heads,
    )?;
    let x = Add::new(None, input, attn_out)?;

    let normed = layer_norm(&wm.prefix("layer_norm2"), x.clone(), 1e-5)?;
    let mlp_out = clip_mlp(&wm.prefix("mlp"), normed, mlp_activation)?;
    Ok(Add::new(None, x, mlp_out)?)
}

fn clip_attention(
    wm: &impl WeightManager,
    input: Arc<dyn Tensor>,
    causal_mask: Arc<dyn Tensor>,
    hidden_dim: usize,
    num_heads: usize,
) -> Result<Arc<dyn Tensor>, Error> {
    let head_dim = hidden_dim / num_heads;

    let q = linear(&wm.prefix("q_proj"), input.clone())?;
    let k = linear(&wm.prefix("k_proj"), input.clone())?;
    let v = linear(&wm.prefix("v_proj"), input)?;

    let q = Transpose::new(
        None,
        reshape(q, vec![0, -1, num_heads as i64, head_dim as i64])?,
        Some(vec![0, 2, 1, 3]),
    );
    let k = Transpose::new(
        None,
        reshape(k, vec![0, -1, num_heads as i64, head_dim as i64])?,
        Some(vec![0, 2, 1, 3]),
    );
    let v = Transpose::new(
        None,
        reshape(v, vec![0, -1, num_heads as i64, head_dim as i64])?,
        Some(vec![0, 2, 1, 3]),
    );

    let kt = Transpose::new(None, k, Some(vec![0, 1, 3, 2]));
    let scores = MatMul::new(None, q, kt)?;
    let scores = crate::onnx_graph::pytorch::div_scalar(scores, (head_dim as f32).sqrt())?;
    let scores = Add::new(None, scores, causal_mask)?;

    let attn_weights = Softmax::new(None, scores, Some(-1));
    let attn_output = MatMul::new(None, attn_weights, v)?;
    let attn_output = Transpose::new(None, attn_output, Some(vec![0, 2, 1, 3]));
    let attn_output = reshape(attn_output, vec![0, -1, hidden_dim as i64])?;

    linear(&wm.prefix("out_proj"), attn_output)
}

fn clip_mlp(
    wm: &impl WeightManager,
    input: Arc<dyn Tensor>,
    activation: ClipMlpActivation,
) -> Result<Arc<dyn Tensor>, Error> {
    let x = linear(&wm.prefix("fc1"), input)?;
    let x = match activation {
        ClipMlpActivation::QuickGelu => quick_gelu(x)?,
        ClipMlpActivation::Gelu => gelu(x)?,
    };
    linear(&wm.prefix("fc2"), x)
}
