use std::fs::File;
use std::io::Write;
use std::path::Path;
use whisper_tensor_import::identify_and_load;
use whisper_tensor_import::onnx_graph::WeightStorageStrategy;

fn main() {
    let file_in = Path::new("/ceph/public/neural_models/llms/Qwen2-0.5B");
    let onnx_out = Path::new("out.onnx");
    let bin_out = Path::new("out.bin");
    let onnx_data = identify_and_load(
        file_in,
        WeightStorageStrategy::BinFile(bin_out.to_path_buf()),
    )
    .unwrap();
    File::create(onnx_out)
        .unwrap()
        .write_all(&onnx_data)
        .unwrap();
}
