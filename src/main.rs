use std::path::Path;
use onnx_graph::WeightStorageStrategy;
use onnx_import::identify_and_load;
use onnx_spirv::{build};

fn main() {

    //VulkanContext::new().unwrap();
    
    let input_path = Path::new("/mnt/secondary/rwkv7-g1/rwkv7-g1-0.1b-20250307-ctx4096.pth");
    
    let onnx_data = identify_and_load(input_path, WeightStorageStrategy::EmbeddedData).unwrap();
    build(&onnx_data).unwrap();
}