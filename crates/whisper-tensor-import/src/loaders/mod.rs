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
use whisper_tensor::loader::{LoadedModel, LoaderError, LoaderOutput};
use whisper_tensor::model::Model;

/// Helper: load ONNX bytes into a Model (no interface detection).
/// The calling loader is responsible for building any interfaces.
fn onnx_bytes_to_model(
    onnx_data: &[u8],
    model_name: &str,
    base_dir: Option<&Path>,
) -> Result<(Arc<Model>, LoaderOutput), LoaderError> {
    let mut rng = rand::rng();
    let model = Model::new_from_onnx(onnx_data, &mut rng, base_dir)
        .map_err(|e| LoaderError::LoadFailed(e.into()))?;
    let model = Arc::new(model);

    let output = LoaderOutput {
        models: vec![LoadedModel {
            name: model_name.to_string(),
            model: model.clone(),
        }],
        interfaces: vec![],
    };

    Ok((model, output))
}

/// Get the default weight storage strategy for loaders.
fn default_storage() -> WeightStorageStrategy {
    WeightStorageStrategy::OriginReference
}
