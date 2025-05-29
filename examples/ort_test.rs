use std::fs::File;
use std::io::Write;
use std::path::{Path};
use ort::session::{builder::GraphOptimizationLevel, Session};
use whisper_tensor_import::{identify_and_load, ModelTypeHint};
use whisper_tensor_import::onnx_graph::WeightStorageStrategy;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    let input_path = Path::new("/mnt/secondary/rwkv7-g1/rwkv7-g1-0.1b-20250307-ctx4096.pth");
    let onnx_out = Path::new("out.onnx");
    let bin_out = Path::new("out.bin");
    let onnx_data = identify_and_load(input_path, WeightStorageStrategy::BinFile(bin_out.to_path_buf()), Some(ModelTypeHint::RWKV7)).unwrap();
    File::create(onnx_out).unwrap().write_all(&onnx_data).unwrap();
    let model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_memory(&onnx_data)?;
    
    Ok(())
}