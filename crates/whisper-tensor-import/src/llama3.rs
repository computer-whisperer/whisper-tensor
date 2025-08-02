use crate::Error;
use crate::onnx_graph::operators::{
    Add, Concat, Gather, MatMul, Mul, RotaryEmbedding, ShapeOp, Softmax, Transpose,
};
use crate::onnx_graph::pytorch::{div_scalar, linear, reshape, rms_norm, silu, transpose};
use crate::onnx_graph::tensor::{DType, Dimension, InputTensor, Shape, Tensor};
use crate::onnx_graph::weights::WeightManager;
use crate::onnx_graph::{
    InputMetadata, ModelInputType, ModelMetadata, ModelOutputType, OutputMetadata, TokenizerInfo,
    WeightStorageStrategy,
};
use prost::Message;
use std::sync::Arc;

pub struct Llama3Config {
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
}

impl Llama3Config {
    pub fn from_huggingface_transformers_json(config: &serde_json::Value) -> Result<Self, Error> {
        //println!("Config: {:?}", config);
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
        Ok(Self {
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
        })
    }
}

pub fn load_llama3(
    weight_manager: impl WeightManager,
    config: Llama3Config,
    output_method: WeightStorageStrategy,
) -> Result<Vec<u8>, anyhow::Error> {
    let model_weight_manager = weight_manager.prefix("model");

    let mut input_tensors: Vec<(Arc<dyn Tensor>, Option<InputMetadata>)> = vec![];
    let mut output_tensors: Vec<(String, Arc<dyn Tensor>, Option<OutputMetadata>)> = vec![];

    let batch_dimension = Dimension::new(
        Some(1),
        Some("batch_size".to_string()),
        Some("DATA_BATCH".to_string()),
    );
    let sequence_dimension = Dimension::new(Some(1), Some("seq_len".to_string()), None);

    let input_shape = Shape::new(vec![
        batch_dimension.clone(),
        sequence_dimension.clone(),
        Dimension::new(Some(1), None, None),
    ]);

    let token_input = InputTensor::new("input_ids".to_string(), DType::I32, input_shape);
    input_tensors.push((
        token_input.clone(),
        Some(InputMetadata {
            model_input_type: ModelInputType::TokenID(0),
        }),
    ));

    let x = Gather::new(
        Some("embed_tokens".to_string()),
        model_weight_manager.get_tensor("embed_tokens.weight")?,
        token_input.clone(),
        0,
    )?;

    let kv_cache_seq_dim = Dimension::new(None, Some("kv_cache_sequence".to_string()), None);

    let model_dim = x.shape().dims[2].resolve()?;
    let head_dim = model_dim / config.num_attention_heads;
    let kv_cache_input_type = x.dtype();
    let kv_cache_input_shape = Shape::new(vec![
        batch_dimension,
        Dimension::new(Some(config.num_key_value_heads), None, None),
        kv_cache_seq_dim,
        Dimension::new(Some(head_dim), None, None),
    ]);

    let cos_sin_cache_shape = Shape::new(vec![
        Dimension::new(Some(8192), Some("RoPE max_len".to_string()), None),
        Dimension::new(Some(model_dim / config.num_attention_heads), None, None),
    ]);
    let sin_cache = InputTensor::new(
        "sin_cache".to_string(),
        kv_cache_input_type,
        cos_sin_cache_shape.clone(),
    );
    let cos_cache = InputTensor::new(
        "cos_cache".to_string(),
        kv_cache_input_type,
        cos_sin_cache_shape,
    );

    let mut next_io_id = 0;

    let mut layer_output: Arc<dyn Tensor> = x;
    for i in 0..config.num_hidden_layers {
        let layer_weight_manager = model_weight_manager.prefix(&format!("layers.{}", i));
        let layer_input = layer_output.clone();
        let att_norm = rms_norm(
            &layer_weight_manager.prefix("input_layernorm"),
            layer_input.clone(),
            None,
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
            format!("kv_cache_input_k_{}", i),
            kv_cache_input_type,
            kv_cache_input_shape.clone(),
        );

        let kv_cache_input_v = InputTensor::new(
            format!("kv_cache_input_v_{}", i),
            kv_cache_input_type,
            kv_cache_input_shape.clone(),
        );

        let rope_i = ShapeOp::new(None, kv_cache_input_k.clone(), Some(0), Some(2))?;

        let q = RotaryEmbedding::new(
            None,
            q,
            cos_cache.clone(),
            sin_cache.clone(),
            Some(rope_i.clone()),
            None,
            None,
            None,
        )?;
        let k = RotaryEmbedding::new(
            None,
            k,
            cos_cache.clone(),
            sin_cache.clone(),
            Some(rope_i),
            None,
            None,
            None,
        )?;

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

        input_tensors.push((
            kv_cache_input_k,
            Some(InputMetadata {
                model_input_type: ModelInputType::PreviousInternal(next_io_id),
            }),
        ));
        output_tensors.push((
            format!("kv_cache_output_k_{}", i),
            k.clone(),
            Some(OutputMetadata {
                model_output_type: ModelOutputType::NextInternal(next_io_id),
            }),
        ));
        next_io_id += 1;

        input_tensors.push((
            kv_cache_input_v,
            Some(InputMetadata {
                model_input_type: ModelInputType::PreviousInternal(next_io_id),
            }),
        ));
        output_tensors.push((
            format!("kv_cache_output_v_{}", i),
            v.clone(),
            Some(OutputMetadata {
                model_output_type: ModelOutputType::NextInternal(next_io_id),
            }),
        ));
        next_io_id += 1;

        let (k, v): (Arc<dyn Tensor>, Arc<dyn Tensor>) =
            if config.num_key_value_heads == config.num_attention_heads {
                (k, v)
            } else {
                let repeat_kv =
                    |x: Arc<dyn Tensor>| -> Result<Arc<dyn Tensor>, crate::onnx_graph::Error> {
                        let n_rep = config.num_attention_heads / config.num_key_value_heads;
                        let x = Concat::new(None, vec![x.clone(); n_rep], 1)?;
                        Ok(x)
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

        // FeedForward
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
    let out = linear(&weight_manager.prefix("lm_head"), h)?;
    output_tensors.push((
        "logits".to_string(),
        out,
        Some(OutputMetadata {
            model_output_type: ModelOutputType::TokenID(0),
        }),
    ));

    println!("Built graph, exporting...");
    let model_metadata = ModelMetadata {
        tokenizer_infos: vec![TokenizerInfo::HFTokenizer("meta/llama3".to_string())],
        max_token_batch: Some(1),
    };
    let onnx_model = crate::onnx_graph::build_proto(
        &input_tensors,
        &output_tensors,
        output_method,
        Some(model_metadata),
    )?;
    Ok(onnx_model.encode_to_vec())
}
