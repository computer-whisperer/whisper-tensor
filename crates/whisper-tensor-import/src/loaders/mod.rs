mod auto;
mod onnx;
mod rwkv7;
mod sd15;
mod transformers;

pub use self::auto::AutoLoader;
pub use self::onnx::OnnxLoader;
pub use self::rwkv7::Rwkv7Loader;
pub use self::sd15::SD15Loader;
pub use self::transformers::TransformersLoader;

use crate::onnx_graph::WeightStorageStrategy;
use std::path::Path;
use std::sync::Arc;
use whisper_tensor::interfaces::get_automatic_interfaces_from_model;
use whisper_tensor::loader::{LoadedInterface, LoadedModel, LoaderError, LoaderOutput};
use whisper_tensor::model::Model;

/// Helper: load ONNX bytes into a Model + automatically detect interfaces.
/// Returns a single named model and any auto-detected interfaces.
fn onnx_bytes_to_output(
    onnx_data: &[u8],
    model_name: &str,
    base_dir: Option<&Path>,
) -> Result<LoaderOutput, LoaderError> {
    let mut rng = rand::rng();
    let model = Model::new_from_onnx(onnx_data, &mut rng, base_dir)
        .map_err(|e| LoaderError::LoadFailed(e.into()))?;

    let interfaces: Vec<LoadedInterface> = get_automatic_interfaces_from_model(&model)
        .into_iter()
        .map(|iface| LoadedInterface {
            name: format!("{}-{}", model_name, iface.name()),
            interface: iface,
        })
        .collect();

    Ok(LoaderOutput {
        models: vec![LoadedModel {
            name: model_name.to_string(),
            model: Arc::new(model),
        }],
        interfaces,
    })
}

/// Get the default weight storage strategy for loaders.
fn default_storage() -> WeightStorageStrategy {
    WeightStorageStrategy::EmbeddedData
}
