use std::fs::File;
use std::io::Write;
use std::path::Path;
use whisper_tensor_import::identify_and_load;
use whisper_tensor_import::onnx_graph::WeightStorageStrategy;

fn main() {
    //let file_in = Path::new("/ceph/public/neural_models/llms/Llama-4-Scout-17B-16E-Instruct");
    let file_in = Path::new("/ceph/public/neural_models/llms/Llama-4-Maverick-17B-128E-Instruct");
    let onnx_out = Path::new("out.onnx");
    let bin_out = Path::new("out.bin");
    let onnx_data = identify_and_load(file_in, WeightStorageStrategy::None, None).unwrap();
    File::create(onnx_out)
        .unwrap()
        .write_all(&onnx_data)
        .unwrap();
}
