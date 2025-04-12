use std::path::Path;
use std::sync::Arc;
use candle_core::pickle::PthTensors;
use prost::Message;
use onnx_graph::operators::{Add, Cast, Constant, Exp, Gather, LpNormalization, MatMul, Mul, Neg, Sigmoid, Softplus, Sub, Tanh, Relu};
use onnx_graph::pytorch::{cast, cumsum, group_norm, layer_norm, linear, reshape, squeeze, transpose};
use onnx_graph::tensor::{DType, Dimension, InputTensor, Shape, Tensor, TensorData, TensorDataValue};
use onnx_graph::weights::{WeightManager};
use onnx_graph::WeightStorageStrategy;
use crate::Error;

fn lerp(a: Arc<dyn Tensor>, b: Arc<dyn Tensor>, t: Arc<dyn Tensor>) -> Result<Arc<dyn Tensor>, onnx_graph::Error> {
    let x = Sub::new(None, b, a.clone())?;
    let x = Mul::new(None, x, t)?;
    Ok(Add::new(None, a, x)?)
}

fn lora_forward(a: Arc<dyn Tensor>, b: Arc<dyn Tensor>, c: Option<Arc<dyn Tensor>>, x: Arc<dyn Tensor>) -> Result<Arc<dyn Tensor>, onnx_graph::Error>{
    let x1 = MatMul::new(None, x, a)?;
    let x2 = MatMul::new(None, x1, b)?;
    Ok(if let Some(c) = c {
        Add::new(None, x2, c)?
    } else {
        x2
    })
}

fn lora_forward_sigmoid(a: Arc<dyn Tensor>, b: Arc<dyn Tensor>, c: Option<Arc<dyn Tensor>>, x: Arc<dyn Tensor>) -> Result<Arc<dyn Tensor>, onnx_graph::Error>{
    let x1 = MatMul::new(None, x, a)?;
    let x2 = MatMul::new(None, Sigmoid::new(None, x1), b)?;
    Ok(if let Some(c) = c {
        Add::new(None, x2, c)?
    } else {
        x2
    })
}

fn lora_forward_tanh(a: Arc<dyn Tensor>, b: Arc<dyn Tensor>, c: Option<Arc<dyn Tensor>>, x: Arc<dyn Tensor>) -> Result<Arc<dyn Tensor>, onnx_graph::Error>{
    let x1 = MatMul::new(None, x, a)?;
    let x2 = MatMul::new(None, Tanh::new(None, x1), b)?;
    Ok(if let Some(c) = c {
        Add::new(None, x2, c)?
    } else {
        x2
    })
}

pub fn add_scalar<T>(x: Arc<dyn Tensor>, scalar: T) -> Result<Arc<dyn Tensor>, onnx_graph::Error>
where
    T: Copy,
    TensorDataValue: From<Vec<T>>
{
    let constant = Constant::new(None, TensorData::fill(x.shape().clone(), scalar)?);
    Ok(Add::new(None, x, constant)?)
}

pub fn load_rwkv7_pth(pth_path: &Path, output_method: WeightStorageStrategy) -> Result<Vec<u8>, Error> {
    println!("Attempting to load RWKV7 from {}", pth_path.display());
    let tensors = Arc::new(PthTensors::new(pth_path, None).unwrap());

    // Survey tensor names
    let tensor_infos = tensors.tensor_infos();
    let mut layer_count = 0;
    for (name, _info) in tensor_infos {
        let name_split = name.split(".").collect::<Vec<&str>>();
        if name_split[0] == "blocks" {
            layer_count = layer_count.max(name_split[1].parse::<usize>().unwrap());
        }
    }
    
    let weight_manager = onnx_graph::weights::PthWeightManager::new(tensors.clone());

    load_rwkv7(weight_manager, layer_count, output_method).map_err(|e| Error::OnnxGraphError(e))
}

pub fn load_rwkv7(weight_manager: impl WeightManager, layer_count: usize, output_method: WeightStorageStrategy) -> Result<Vec<u8>, onnx_graph::Error> {

    let mut input_tensors = vec![];
    let mut output_tensors: Vec<(String, Arc<dyn Tensor>)> = vec![];

    let batch_dimension = Dimension::new(Some(1), Some("batch_size".to_string()), Some("DATA_BATCH".to_string()));
    let sequence_dimension = Dimension::new(Some(1), Some("seq_len".to_string()), None);

    let input_shape = Shape::new(vec![
        batch_dimension.clone(),
        sequence_dimension.clone(),
        Dimension::new(Some(1), None, None)
    ]);
    let token_input = InputTensor::new("token_input".to_string(), DType::I32, input_shape);
    input_tensors.push(token_input.clone());

    let x = Gather::new(Some("emb".to_string()), weight_manager.get_tensor("emb.weight")?, token_input.clone(), 0)?;

    let hidden_dim = x.shape().dim(-1).clone();

    let n_heads = weight_manager.get_tensor("blocks.0.att.r_k")?.shape().resolve()?[0];

    let k_dimension = Dimension::new(Some(hidden_dim.resolve()?/n_heads), Some("k_dim".to_string()), None);
    let v_dimension = Dimension::new(Some(hidden_dim.resolve()?/n_heads), Some("v_dim".to_string()), None);
    let n_heads_dimension = Dimension::new(Some(n_heads), Some("n_heads".to_string()), None);

    let mut x: Arc<dyn Tensor> = layer_norm(
        &weight_manager.prefix("blocks.0.ln0"),
        x,
        1e-5,
    )?;

    let mut v0: Option<Arc<dyn Tensor>> = None;

    for layer_id in 0..layer_count {
        let block_weight_manager = weight_manager.prefix(&format!("blocks.{layer_id}"));

        let after_ln1 = layer_norm(
            &block_weight_manager.prefix("ln1"),
            x, 1e-5)?;

        let time_mixer_x_in = InputTensor::new(format!("time_mixer_x_in_{layer_id}"), DType::BF16, after_ln1.shape().clone());
        input_tensors.push(time_mixer_x_in.clone());

        output_tensors.push((format!("time_mixer_x_out_{layer_id}"), after_ln1.clone()));

        let dx_prev = Sub::new(None, time_mixer_x_in, after_ln1.clone())?;

        let receptance = Add::new(None, after_ln1.clone(), Mul::new(None, dx_prev.clone(), block_weight_manager.get_tensor("att.x_r")?)?)?;
        let decay =      Add::new(None, after_ln1.clone(), Mul::new(None, dx_prev.clone(), block_weight_manager.get_tensor("att.x_w")?)?)?;
        let key =        Add::new(None, after_ln1.clone(), Mul::new(None, dx_prev.clone(), block_weight_manager.get_tensor("att.x_k")?)?)?;
        let value =      Add::new(None, after_ln1.clone(), Mul::new(None, dx_prev.clone(), block_weight_manager.get_tensor("att.x_v")?)?)?;
        let iclr =       Add::new(None, after_ln1.clone(), Mul::new(None, dx_prev.clone(), block_weight_manager.get_tensor("att.x_a")?)?)?;
        let gate =       Add::new(None, after_ln1.clone(), Mul::new(None, dx_prev.clone(), block_weight_manager.get_tensor("att.x_g")?)?)?;

        let receptance = linear(&block_weight_manager.prefix("att.receptance"), receptance)?;
        let key = linear(&block_weight_manager.prefix("att.key"), key)?;
        let value = linear(&block_weight_manager.prefix("att.value"), value)?;

        let (new_v0, value) = if let Some(v0) = v0 {
            let v0_mix = lora_forward(
                block_weight_manager.get_tensor("att.v1")?,
                block_weight_manager.get_tensor("att.v2")?,
                Some(block_weight_manager.get_tensor("att.v0")?),
                value.clone()
            )?;
            let v_lerp = lerp(value, v0.clone(), Sigmoid::new(None, v0_mix))?;
            (v0, v_lerp)
        } else {
            (value.clone(), value)
        };
        v0 = Some(new_v0);

        let gate = lora_forward_sigmoid(
            block_weight_manager.get_tensor("att.g1")?,
            block_weight_manager.get_tensor("att.g2")?,
            None,
            gate
        )?;

        let log_neg_log_of_decay = lora_forward_tanh(
            block_weight_manager.get_tensor("att.w1")?,
            block_weight_manager.get_tensor("att.w2")?,
            Some(block_weight_manager.get_tensor("att.w0")?),
            decay
        )?;
        let log_neg_log_of_decay = add_scalar(Neg::new(None, Softplus::new(None, Neg::new(None, log_neg_log_of_decay))), half::bf16::from_f32(-0.5))?;

        let log_of_decay = Neg::new(None, Exp::new(None, Cast::new(None, log_neg_log_of_decay, DType::F32)));
        let decay = Exp::new(None, log_of_decay);

        let deformed_key = Mul::new(None, key.clone(), block_weight_manager.get_tensor("att.k_k")?)?;
        let deformed_key = reshape(deformed_key, vec![0, 0, n_heads as i64, -1])?;
        let deformed_key = LpNormalization::new(None, deformed_key, 2, -1);

        let vk_state_in = InputTensor::new(format!("vk_state_in_{layer_id}"), DType::F32, Shape::new(vec![
            batch_dimension.clone(),
            n_heads_dimension.clone(),
            k_dimension.clone(),
            v_dimension.clone()
        ]));
        input_tensors.push(vk_state_in.clone());



        // Token-by-token loop (only one for now)
        let vk_state = vk_state_in;
        let (out, vk_state_out) = {
            let r = cast(reshape(receptance.clone(), vec![0, n_heads as i64, -1, 1])?, DType::F32);
            let k = cast(reshape(key.clone(), vec![0, n_heads as i64, -1, 1])?, DType::F32);
            let v = cast(reshape(value.clone(), vec![0, n_heads as i64, -1, 1])?, DType::F32);
            let decay = reshape(decay, vec![0, n_heads as i64, -1, 1])?;
            let iclr = cast(reshape(iclr, vec![0, n_heads as i64, -1, 1])?, DType::F32);
            let deformed_key = cast(reshape(deformed_key, vec![0, n_heads as i64, -1, 1])?, DType::F32);


            let t_decay = transpose(decay);
            let temp = transpose(Mul::new(None, iclr, deformed_key.clone())?);
            let temp2 = MatMul::new(None, vk_state.clone(), deformed_key.clone())?;
            let vk_state = Sub::new(None, Mul::new(None, vk_state.clone(), t_decay)?, MatMul::new(None, temp2, temp)?)?;

            let vk_state = Add::new(None, vk_state, MatMul::new(None, v, transpose(k))?)?;

            let out = MatMul::new(None, vk_state.clone(), r)?;

            let out = cast(squeeze(out, 3)?, DType::BF16);
            (out, vk_state)
        };
        output_tensors.push((format!("vk_state_out_{layer_id}"), vk_state_out));

        let value = reshape(value, vec![0, 0, n_heads as i64, -1])?;

        // Reshape to standard
        let out = reshape(out, vec![1, -1])?;

        let out = group_norm(&block_weight_manager.prefix("att.ln_x"), out, 1e-5, n_heads as i64)?;

        let out = reshape(out, vec![1, 1, -1])?;

        let bonus = reshape(Mul::new(None, receptance, key)? , vec![0, 0, n_heads as i64, -1])?;
        let bonus = Mul::new(None, bonus, block_weight_manager.get_tensor("att.r_k")?)?;
        let bonus = Mul::new(None, cumsum(bonus, 3)?, value)?;
        let bonus = reshape(bonus, vec![1, 1, -1])?;
        let out = Mul::new(None, Add::new(None, bonus, out)?, gate)?;
        let hidden_state = linear(&block_weight_manager.prefix("att.output"), out)?;
        // Done with time mixer

        let after_time_mix = Add::new(None, after_ln1, hidden_state)?;
        let after_ln2 = layer_norm(&block_weight_manager.prefix("ln2"), after_time_mix.clone(), 1e-5)?;

        // Channel mixing
        let channel_mixer_x_in = InputTensor::new(format!("channel_mixer_x_in_{layer_id}"), DType::BF16, after_ln2.shape().clone());
        input_tensors.push(channel_mixer_x_in.clone());
        output_tensors.push((format!("channel_mixer_x_out_{layer_id}"), after_ln2.clone()));
        let hidden_state = lerp(after_ln2.clone(), channel_mixer_x_in, block_weight_manager.get_tensor("ffn.x_k")?)?;
        let hidden_state = linear(&block_weight_manager.prefix("ffn.key"), hidden_state)?;
        let hidden_state = Relu::new(None, hidden_state)?;
        let hidden_state = Mul::new(None, hidden_state.clone(), hidden_state)?;
        let out = linear(&block_weight_manager.prefix("ffn.value"), hidden_state)?;

        let out = Add::new(None, after_time_mix, out)?;

        x = out;
    }

    let ln_out = layer_norm(
        &weight_manager.prefix("ln_out"),
        x,
        1e-5,
    )?;

    let output = linear(&weight_manager.prefix("head"), ln_out)?;

    output_tensors.push(("output".to_string(), output));

    println!("Built graph, exporting...");
    let onnx_model = onnx_graph::build_proto(&input_tensors, &output_tensors, output_method)?;

    Ok(onnx_model.encode_to_vec())
}