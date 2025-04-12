use std::fs::File;
use std::io::Write;
use std::path::Path;
use onnx_graph::WeightStorageStrategy;

fn main() {
    //let file_in = Path::new("/mnt/secondary/temp-latest-training-models/RWKV7-G1-1.5B-50%trained-20250330-ctx4k.pth");
    let file_in = Path::new("/mnt/secondary/rwkv7-g1/rwkv7-g1-0.1b-20250307-ctx4096.pth");
    let onnx_out = Path::new("out.onnx");
    let bin_out = Path::new("out.bin");
    let onnx_data = onnx_import::identify_and_load(file_in, WeightStorageStrategy::BinFile(bin_out.to_path_buf())).unwrap();
    File::create(onnx_out).unwrap().write_all(&onnx_data).unwrap();
}