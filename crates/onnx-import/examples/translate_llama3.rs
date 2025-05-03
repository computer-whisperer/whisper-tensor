use std::fs::File;
use std::io::Write;
use std::path::Path;
use onnx_graph::WeightStorageStrategy;
use onnx_import::identify_and_load;

fn main() {
    let file_in = Path::new("/mnt/secondary/neural_networks/llms/Llama-3.1-8B-Instruct");
    //let file_in = Path::new("/ceph/public/neural_models/llms/Meta-Llama-3.1-70B");
    let onnx_out = Path::new("out.onnx");
    let bin_out = Path::new("out.bin");
    let onnx_data = identify_and_load(file_in, WeightStorageStrategy::BinFile(bin_out.to_path_buf()), None).unwrap();
    File::create(onnx_out).unwrap().write_all(&onnx_data).unwrap();
}