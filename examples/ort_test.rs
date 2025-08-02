use ort::session::{Session, builder::GraphOptimizationLevel};
use std::fs::File;
use std::io::Write;
use std::path::Path;
use whisper_tensor_import::onnx_graph::WeightStorageStrategy;
use whisper_tensor_import::{ModelTypeHint, identify_and_load};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    let input_path = Path::new("/mnt/secondary/rwkv7-g1/rwkv7-g1-0.1b-20250307-ctx4096.pth");
    let onnx_out = Path::new("out.onnx");
    let bin_out = Path::new("out.bin");
    let onnx_data = identify_and_load(
        input_path,
        WeightStorageStrategy::BinFile(bin_out.to_path_buf()),
        Some(ModelTypeHint::RWKV7),
    )
    .unwrap();
    File::create(onnx_out)
        .unwrap()
        .write_all(&onnx_data)
        .unwrap();
    let _model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_memory(&onnx_data)?;

    Ok(())
}
