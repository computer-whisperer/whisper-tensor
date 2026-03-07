use std::path::Path;
use whisper_tensor::model::Model;
use whisper_tensor_import::identify_and_load;
use whisper_tensor_import::onnx_graph::WeightStorageStrategy;

const SD_BASE: &str = "/mnt/secondary/neural_networks/stable-diffusion-1.5-onnx-fp16";

fn try_load(name: &str, subpath: &str) {
    let input_path = Path::new(SD_BASE).join(subpath).join("model.onnx");
    println!("=== Loading {name} from {} ===", input_path.display());

    let onnx_data = match identify_and_load(&input_path, WeightStorageStrategy::EmbeddedData, None)
    {
        Ok(data) => {
            println!("  Import OK ({} bytes)", data.len());
            data
        }
        Err(e) => {
            println!("  Import FAILED: {e}");
            return;
        }
    };

    let mut rng = rand::rng();
    match Model::new_from_onnx(&onnx_data, &mut rng) {
        Ok(_runtime) => println!("  Model load OK"),
        Err(e) => println!("  Model load FAILED: {e}"),
    }
    println!();
}

fn main() {
    tracing_subscriber::fmt::init();

    try_load("text_encoder", "text_encoder");
    try_load("unet", "unet");
    try_load("vae_decoder", "vae_decoder");
    try_load("vae_encoder", "vae_encoder");
}
