use std::path::Path;
use std::sync::Arc;
use whisper_tensor::loader::{LoadedModel, LoaderError, LoaderOutput};
use whisper_tensor::model::Model;

pub mod f5_tts;
pub mod kokoro;
pub mod piper;
pub mod whisper;

/// Helper: load ONNX bytes into a Model (no interface detection).
/// Speech model modules use this to assemble LoaderOutput.
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
