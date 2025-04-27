use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use onnx_graph::WeightStorageStrategy;
use onnx_import::identify_and_load;
use whisper_tensor::{RuntimeModel, Backend};
use whisper_tensor::numeric_tensor::{NumericTensor};
use whisper_tensor::sampler::{GreedySampler, Sampler};

fn main() {
    tracing_subscriber::fmt::init();

    let input_path = Path::new("gpt2-10.onnx");
    let mut onnx_data = Vec::new();
    File::open(input_path).unwrap().read_to_end(&mut onnx_data).unwrap();

    let runtime = RuntimeModel::load_onnx(&onnx_data, Backend::ONNXReference).unwrap();
    let input = NumericTensor::from_vec(vec![0u32], vec![1, 1, 1]);
    let ret = runtime.run(HashMap::from([("input1".to_string(), input)])).unwrap();
    let output = ret.get("output1").unwrap();
    println!("logits: {}", output);
    let chosen_token = GreedySampler{}.sample(output).unwrap();
    println!("chosen token: {}", chosen_token);
}