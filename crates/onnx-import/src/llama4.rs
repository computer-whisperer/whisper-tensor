use std::sync::Arc;
use prost::Message;
use onnx_graph::operators::{Add, Gather, Mul, TopK};
use onnx_graph::pytorch::{layer_norm, linear, reshape, rms_norm, silu, topk, transpose};
use onnx_graph::tensor::{DType, Dimension, InputTensor, Shape, Tensor};
use onnx_graph::weights::WeightManager;
use onnx_graph::WeightStorageStrategy;
use crate::Error;

enum HiddenAct {
    Silu
}

impl HiddenAct {
    pub fn from_str(s: &str) -> Result<Self, Error> {
        match s {
            "silu" => Ok(HiddenAct::Silu),
            s => Err(Error::UnsupportedConfigurationError("activation".to_string(), s.to_string()))
        }
    }
    
    pub fn apply(&self, input: Arc<dyn Tensor>) -> Result<Arc<dyn Tensor>, onnx_graph::Error> {
        match self {
            HiddenAct::Silu => silu(input)
        }
    }
}

pub struct Llama4Config {
    num_hidden_layers: usize,
    num_attention_heads: usize,
    moe_layers: Vec<usize>,
    num_key_value_heads: usize,
    rms_norm_eps: f32,
    num_experts_per_tok: usize,
    hidden_act: HiddenAct
}

impl Llama4Config {
    pub fn from_huggingface_transformers_json(config: &serde_json::Value) -> Result<Self, Error> {
        
        fn get_int(config: &serde_json::Value, key: &str) -> Result<i64, Error> {
            config.get(key)
                .ok_or(Error::MissingConfigEntryError(key.to_string()))?
                .as_i64().ok_or(Error::MissingConfigEntryError(key.to_string()))
        }

        fn get_float(config: &serde_json::Value, key: &str) -> Result<f64, Error> {
            config.get(key)
                .ok_or(Error::MissingConfigEntryError(key.to_string()))?
                .as_f64().ok_or(Error::MissingConfigEntryError(key.to_string()))
        }

        fn get_string<'a>(config: &'a serde_json::Value, key: &str) -> Result<&'a str, Error> {
            config.get(key)
                .ok_or(Error::MissingConfigEntryError(key.to_string()))?
                .as_str().ok_or(Error::MissingConfigEntryError(key.to_string()))
        }

        fn get_value<'a>(config: &'a serde_json::Value, key: &str) -> Result<&'a serde_json::Value, Error> {
            config.get(key)
                .ok_or(Error::MissingConfigEntryError(key.to_string()))
        }
        let text_config = get_value(config, "text_config")?;

        let num_hidden_layers = get_int(text_config, "num_hidden_layers")? as usize;
        let num_attention_heads = get_int(text_config, "num_attention_heads")? as usize;
        let num_key_value_heads = get_int(text_config, "num_key_value_heads")? as usize;
        let rms_norm_eps = get_float(text_config, "rms_norm_eps")? as f32;
        
        let mut moe_layers = vec![];
        let moe_layer_step = get_int(text_config, "interleave_moe_layer_step")?;
        let mut i = (moe_layer_step-1).max(0) as usize;
        while i < num_hidden_layers {
            moe_layers.push(i);
            i += moe_layer_step as usize;
        }
        
        let hidden_act = HiddenAct::from_str(get_string(text_config, "hidden_act")?)?;
        
        let num_experts_per_tok = get_int(text_config, "num_experts_per_tok")? as usize;
        if num_experts_per_tok > 1 {
            Err(Error::UnsupportedConfigurationError("num_experts_per_tok".to_string(), num_experts_per_tok.to_string()))?;
        }
        
        Ok(Self {
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            num_experts_per_tok,
            rms_norm_eps,
            moe_layers,
            hidden_act
        })
    }
}

pub fn load_llama4(weight_manager: impl WeightManager, config: Llama4Config, output_method: WeightStorageStrategy) -> Result<Vec<u8>, anyhow::Error> {
    weight_manager.print_weight_list();
    let weight_manager = weight_manager.prefix("language_model");
    let model_weight_manager = weight_manager.prefix("model");

    let mut input_tensors: Vec<Arc<dyn Tensor>> = vec![];
    let mut output_tensors: Vec<(String, Arc<dyn Tensor>)> = vec![];

    let batch_dimension = Dimension::new(Some(1), Some("batch_size".to_string()), Some("DATA_BATCH".to_string()));
    let sequence_dimension = Dimension::new(Some(1), Some("seq_len".to_string()), None);

    let input_shape = Shape::new(vec![
        batch_dimension.clone(),
        sequence_dimension.clone(),
        Dimension::new(Some(1), None, None)
    ]);

    let token_input = InputTensor::new("input_ids".to_string(), DType::I32, input_shape);
    input_tensors.push(token_input.clone());

    let x = Gather::new(Some("embed_tokens".to_string()), model_weight_manager.get_tensor("embed_tokens.weight")?, token_input.clone(), 0)?;

    let model_dim = x.shape().dims[2].resolve()?;
    
    let mut layer_output: Arc<dyn Tensor> = x;
    for i in 0..config.num_hidden_layers {
        let layer_weight_manager = model_weight_manager.prefix(&format!("layers.{i}"));
        let layer_input = layer_output.clone();
        
        let residual = layer_input.clone();
        let hidden_state = layer_norm(&layer_weight_manager.prefix("input_layernorm"), layer_input, 1e-5)?;
        
        // Attention block
        //todo!();
        
        let attention_output = Add::new(None, residual, hidden_state)?;
        let residual = attention_output.clone();
        
        let hidden_state = layer_norm(&layer_weight_manager.prefix("post_attention_layernorm"), attention_output, 1e-5)?;
        
        // Feed forward
        
        let hidden_state: Arc<dyn Tensor> = if config.moe_layers.contains(&i) {
            // MOE layer
            let hidden_state = reshape(hidden_state.clone(), vec![-1, model_dim as i64])?;
            let router_logits = linear(&layer_weight_manager.prefix("feed_forward.router"), hidden_state.clone())?;
            let router_logits = transpose(router_logits);
            let (_values, _selected_experts) = topk(router_logits, config.num_experts_per_tok as i64, 1)?;
            
            
            todo!();

            reshape(hidden_state, vec![1, 1, model_dim as i64])?
        } else {
            // MLP layer
            let x = linear(&layer_weight_manager.prefix("feed_forward.gate_proj"), hidden_state.clone())?;
            let x = config.hidden_act.apply(x)?;
            let x2 = linear(&layer_weight_manager.prefix("feed_forward.up_proj"), hidden_state.clone())?;
            let hidden_state = Mul::new(None, x, x2)?;
            linear(&layer_weight_manager.prefix("feed_forward.down_proj"), hidden_state)?
        };
        
        let hidden_state = Add::new(None, residual, hidden_state)?;
        layer_output = hidden_state;
    }
    
    let h = rms_norm(&model_weight_manager.prefix("norm"), layer_output, Some(config.rms_norm_eps))?;
    let out = linear(&weight_manager.prefix("lm_head"), h)?;
    output_tensors.push(("logits".to_string(), out));

    println!("Built graph, exporting...");
    let onnx_model = onnx_graph::build_proto(&input_tensors, &output_tensors, output_method)?;

    Ok(onnx_model.encode_to_vec())
}
