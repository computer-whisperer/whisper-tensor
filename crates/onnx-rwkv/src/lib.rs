use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use candle_core::pickle::PthTensors;
use prost::Message;
use onnx_graph::tensor::InputTensor;
use onnx_graph::weights::PthTensor;

pub fn translate(pth_path: &Path, mut onnx_output: impl Write, data_output: impl Write) {
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
    
    let find_pth_tensor = |name: &str| -> Arc<PthTensor> {
        PthTensor::new(tensor_infos.get(name).expect(&format!("Couldn't find {name}")).clone(), tensors.clone()).unwrap()
    };
    
    let token_input = InputTensor::new(onnx_graph::tensor::DType::U16, vec![1]);
    
    let x = onnx_graph::operators::Gather::new(find_pth_tensor("emb.weight"), token_input.clone(), 0).unwrap();
    
    let mut x = onnx_graph::operators::LayerNormalization::new(
        x,
        find_pth_tensor("blocks.0.ln0.weight"),
        Some(find_pth_tensor("blocks.0.ln0.bias")),
        -1,
        1e-5,
        1
        ).unwrap();
    
    /*
    for layer_id in 0..layer_count {
        
    }*/
    
    let ln_out = onnx_graph::operators::LayerNormalization::new(
        x,
        find_pth_tensor("ln_out.weight"),
        Some(find_pth_tensor("ln_out.bias")),
        -1,
        1e-5,
        1
    ).unwrap();
    

    let onnx_model = onnx_graph::build_proto(&[("input", token_input)], &[("output", ln_out)]);

    onnx_output.write_all(&onnx_model.encode_to_vec()).unwrap();
}