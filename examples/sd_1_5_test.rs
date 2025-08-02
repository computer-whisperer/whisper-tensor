use std::fs::File;
use std::io::Write;
use std::path::Path;
use whisper_tensor::model::Model;
use whisper_tensor_import::identify_and_load;
use whisper_tensor_import::onnx_graph::WeightStorageStrategy;

fn main() {
    tracing_subscriber::fmt::init();

    //let input_path = Path::new("/mnt/secondary/neural_networks/v1-5-pruned-emaonly.safetensors");
    //let input_path = Path::new("/mnt/secondary/neural_networks/stable-diffusion-1.5-onnx-fp16/unet/model.onnx");
    let input_path = Path::new(
        "/mnt/secondary/neural_networks/stable-diffusion-1.5-onnx-fp16/text_encoder/model.onnx",
    );
    let onnx_out = Path::new("out.onnx");
    let _bin_out = Path::new("out.bin");
    let onnx_data =
        identify_and_load(input_path, WeightStorageStrategy::EmbeddedData, None).unwrap();
    File::create(onnx_out)
        .unwrap()
        .write_all(&onnx_data)
        .unwrap();
    let _runtime = Model::new_from_onnx(&onnx_data).unwrap();
}
