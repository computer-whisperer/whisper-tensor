use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use candle_core::NdArray;
use candle_core::pickle::PthTensors;
use prost::Message;
use onnx_graph::operators::{Add, Cast, Constant, Exp, Gather, LpNormalization, MatMul, Mul, Neg, Sigmoid, Softplus, Sub, Tanh};
use onnx_graph::tensor::{DType, Dimension, InputTensor, Shape, Tensor, TensorData};
use onnx_graph::weights::{WeightManager};

fn lerp(a: Arc<dyn Tensor>, b: Arc<dyn Tensor>, t: Arc<dyn Tensor>) -> Arc<dyn Tensor> {
    let x = Sub::new(None, b, a.clone()).unwrap();
    let x = Mul::new(None, x, t).unwrap();
    Add::new(None, a, x).unwrap()
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

pub fn add_scalar(x: Arc<dyn Tensor>, scalar: f32) -> Result<Arc<dyn Tensor>, onnx_graph::Error> {
    let constant = Constant::new(None, TensorData::fill(x.shape().clone(), scalar)?);
    Ok(Add::new(None, x, constant)?)
}

pub fn translate(pth_path: &Path, mut onnx_output: impl Write, data_output: &Path) -> Result<(), onnx_graph::Error> {
    let tensors = Arc::new(PthTensors::new(pth_path, None).unwrap());
    println!("Got tensors!");

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

    let input_shape = Shape::new(vec![
        Dimension::new(None, Some("batch_size".to_string()), Some("DATA_BATCH".to_string())),
        Dimension::new(Some(1), Some("seq_len".to_string()), None),
        Dimension::new(Some(1), None, None)
    ]);
    let token_input = InputTensor::new("token_input".to_string(), onnx_graph::tensor::DType::I32, input_shape);

    let x = Gather::new(Some("emb".to_string()), weight_manager.get_tensor("emb.weight")?, token_input.clone(), 0)?;

    let hidden_dim = x.shape().dim(-1).clone();

    let mut x: Arc<dyn Tensor> = onnx_graph::pytorch::layer_norm(
        &weight_manager.prefix("blocks.0.ln0"),
        x,
        1e-5,
        )?;

    let mut v0: Option<Arc<dyn Tensor>> = None;

    for layer_id in 0..layer_count {
        let block_weight_manager = weight_manager.prefix(&format!("blocks.{}", layer_id));

        let after_ln = onnx_graph::pytorch::layer_norm(
            &block_weight_manager.prefix("ln1"),
            x, 1e-5)?;

        let time_mixer_x_in = InputTensor::new("token_input".to_string(), onnx_graph::tensor::DType::U16, Shape::new(vec![hidden_dim.clone()]));

        let dxprev = Sub::new(None, time_mixer_x_in, after_ln.clone())?;

        let receptance = Add::new(None, after_ln.clone(), Mul::new(None, dxprev.clone(), block_weight_manager.get_tensor("att.x_r")?)?)?;
        let decay =      Add::new(None, after_ln.clone(), Mul::new(None, dxprev.clone(), block_weight_manager.get_tensor("att.x_w")?)?)?;
        let key =        Add::new(None, after_ln.clone(), Mul::new(None, dxprev.clone(), block_weight_manager.get_tensor("att.x_k")?)?)?;
        let value =      Add::new(None, after_ln.clone(), Mul::new(None, dxprev.clone(), block_weight_manager.get_tensor("att.x_v")?)?)?;
        let iclr =       Add::new(None, after_ln.clone(), Mul::new(None, dxprev.clone(), block_weight_manager.get_tensor("att.x_a")?)?)?;
        let gate =       Add::new(None, after_ln.clone(), Mul::new(None, dxprev.clone(), block_weight_manager.get_tensor("att.x_g")?)?)?;

        let receptance = onnx_graph::pytorch::linear(&block_weight_manager.prefix("att.receptance"), receptance)?;
        let key = onnx_graph::pytorch::linear(&block_weight_manager.prefix("att.key"), key)?;
        let value = onnx_graph::pytorch::linear(&block_weight_manager.prefix("att.value"), value)?;

        let (new_v0, value) = if let Some(v0) = v0 {
            let v0_mix = lora_forward(
                block_weight_manager.get_tensor("att.v1")?,
                block_weight_manager.get_tensor("att.v2")?,
                Some(block_weight_manager.get_tensor("att.v0")?),
                value.clone()
            )?;
            let v_lerp = lerp(value, v0.clone(), Sigmoid::new(None, v0_mix));
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

        let log_neglog_of_decay = lora_forward_tanh(
            block_weight_manager.get_tensor("att.w1")?,
            block_weight_manager.get_tensor("att.w2")?,
            Some(block_weight_manager.get_tensor("att.w0")?),
            decay
        )?;
        let log_neglog_of_decay = add_scalar(Neg::new(None, Softplus::new(None, Neg::new(None, log_neglog_of_decay))), -0.5)?;

        let log_of_decay = Neg::new(None, Exp::new(None, Cast::new(None, log_neglog_of_decay, DType::F32)));
        let decay = Exp::new(None, log_of_decay);

        let deformed_key = Mul::new(None, key.clone(), block_weight_manager.get_tensor("att.k_k")?)?;
        let deformed_key = LpNormalization::new(None, deformed_key, 2, -1);


        x = dxprev;
    }

    let ln_out = onnx_graph::pytorch::layer_norm(
        &weight_manager.prefix("ln_out"),
        x,
        1e-5,
    )?;

    let output = onnx_graph::pytorch::linear(&weight_manager.prefix("head"), ln_out)?;

    let onnx_model = onnx_graph::build_proto(&[token_input], &[("output", output)], data_output)?;

    onnx_output.write_all(&onnx_model.encode_to_vec()).unwrap();
    Ok(())
}