use std::path::Path;

pub mod rwkv7;

pub fn identify_and_translate(model_path: &Path, bin_path: Option<&Path>) -> Result<Vec::<u8>, onnx_graph::Error> {
    // TODO: identify model type
    rwkv7::translate(model_path, bin_path)
}