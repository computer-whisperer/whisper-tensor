use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use onnx_graph::WeightStorageStrategy;
use onnx_import::identify_and_load;
use whisper_tensor::{RuntimeModel, RuntimeBackend, RuntimeEnvironment};
use whisper_tensor::numeric_tensor::{NumericTensor};

fn main() {
    tracing_subscriber::fmt::init();

    let input_path = Path::new("/mnt/secondary/rwkv7-g1/rwkv7-g1-0.1b-20250307-ctx4096.pth");
    //let input_path = Path::new("/ceph-fuse/public/neural_models/llms/rwkv7-g1/rwkv7-g1-0.1b-20250307-ctx4096.pth");
    let onnx_out = Path::new("out.onnx");
    let bin_out = Path::new("out.bin");
    let onnx_data = identify_and_load(input_path, WeightStorageStrategy::BinFile(bin_out.to_path_buf())).unwrap();
    File::create(onnx_out).unwrap().write_all(&onnx_data).unwrap();

    let mut runtime = RuntimeModel::load_onnx(&onnx_data, RuntimeBackend::ORT, Default::default()).unwrap();

    let input = NumericTensor::from_vec_shape(vec![0.0], vec![1, 1, 1]).unwrap();
    runtime.run(HashMap::from([("tensor_input".to_string(), input)])).unwrap();
}