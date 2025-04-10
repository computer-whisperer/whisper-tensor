use std::fs::File;
use std::io::Write;
use std::path::Path;
use onnx_import::identify_and_load;

fn main() {
    let file_in = Path::new("/mnt/secondary/neural_networks/llms/Llama-3.1-8B-Instruct");
    let onnx_out = Path::new("out.onnx");
    let bin_out = Path::new("out.bin");
    let onnx_data = identify_and_load(file_in, Some(bin_out)).unwrap();
    File::create(onnx_out).unwrap().write_all(&onnx_data).unwrap();
}