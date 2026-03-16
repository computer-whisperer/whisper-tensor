use std::path::{Path, PathBuf};
use std::sync::Arc;
use whisper_tensor::loader::LoaderError;
use whisper_tensor::model::Model;

pub(super) fn load_onnx_model(path: &Path) -> Result<Arc<Model>, LoaderError> {
    if path
        .file_name()
        .and_then(|x| x.to_str())
        .is_some_and(|name| name.contains(".int8.onnx"))
    {
        return Err(LoaderError::LoadFailed(anyhow::anyhow!(
            "INT8 SD3.5 ONNX checkpoints are not supported yet (requires DequantizeLinear support). \
             Please use FP16/FP32 ONNX exports instead: {}",
            path.display()
        )));
    }

    let onnx_data = crate::load_onnx_file(path).map_err(|e| LoaderError::LoadFailed(e.into()))?;
    let mut rng = rand::rng();
    let model = Model::new_from_onnx(&onnx_data, &mut rng, path.parent())
        .map_err(|e| LoaderError::LoadFailed(e.into()))?;
    Ok(Arc::new(model))
}

pub(super) fn is_sd3_diffusers_safetensors_dir(root: &Path) -> bool {
    has_safetensors_component(root, "text_encoder")
        && has_safetensors_component(root, "text_encoder_2")
        && has_safetensors_component(root, "text_encoder_3")
        && has_safetensors_component(root, "transformer")
        && has_safetensors_component(root, "vae")
}

fn has_safetensors_component(root: &Path, component: &str) -> bool {
    let dir = root.join(component);
    if !dir.is_dir() {
        return false;
    }
    std::fs::read_dir(&dir).ok().is_some_and(|entries| {
        entries.filter_map(Result::ok).any(|entry| {
            entry
                .path()
                .extension()
                .is_some_and(|ext| ext == "safetensors")
        })
    })
}

pub(super) fn resolve_component_safetensors(
    root: &Path,
    component: &str,
) -> Result<Vec<PathBuf>, LoaderError> {
    let component_dir = root.join(component);
    if !component_dir.is_dir() {
        return Err(LoaderError::LoadFailed(anyhow::anyhow!(
            "Missing required SD3.5 component directory: {}",
            component_dir.display()
        )));
    }

    let mut files: Vec<PathBuf> = std::fs::read_dir(&component_dir)
        .map_err(|e| LoaderError::LoadFailed(e.into()))?
        .filter_map(Result::ok)
        .map(|x| x.path())
        .filter(|p| p.is_file() && p.extension().is_some_and(|ext| ext == "safetensors"))
        .collect();

    if files.is_empty() {
        return Err(LoaderError::LoadFailed(anyhow::anyhow!(
            "No .safetensors file found in SD3.5 component directory: {}",
            component_dir.display()
        )));
    }

    files.sort();
    Ok(files)
}

pub(super) fn resolve_component_onnx(root: &Path, component: &str) -> Result<PathBuf, LoaderError> {
    let component_dir = root.join(component);
    if !component_dir.is_dir() {
        return Err(LoaderError::LoadFailed(anyhow::anyhow!(
            "Missing required SD3.5 component directory: {}",
            component_dir.display()
        )));
    }

    let canonical = component_dir.join("model.onnx");
    if canonical.exists() {
        return Ok(canonical);
    }

    let mut onnx_files = std::fs::read_dir(&component_dir)
        .map_err(|e| LoaderError::LoadFailed(e.into()))?
        .filter_map(Result::ok)
        .map(|x| x.path())
        .filter(|p| p.is_file() && p.extension().is_some_and(|ext| ext == "onnx"))
        .collect::<Vec<_>>();

    if onnx_files.is_empty() {
        return Err(LoaderError::LoadFailed(anyhow::anyhow!(
            "No .onnx file found in SD3.5 component directory: {}",
            component_dir.display()
        )));
    }

    onnx_files.sort();
    if let Some(preferred) = onnx_files
        .iter()
        .find(|p| p.file_name().and_then(|x| x.to_str()) == Some("model.int8.onnx"))
    {
        return Ok(preferred.clone());
    }

    Ok(onnx_files[0].clone())
}

pub(super) fn resolve_component_onnx_any(
    root: &Path,
    components: &[&str],
) -> Result<PathBuf, LoaderError> {
    let mut last_error: Option<anyhow::Error> = None;
    for component in components {
        match resolve_component_onnx(root, component) {
            Ok(path) => return Ok(path),
            Err(LoaderError::LoadFailed(err)) => last_error = Some(err),
            Err(other) => return Err(other),
        }
    }
    let tried = components.join(", ");
    Err(LoaderError::LoadFailed(anyhow::anyhow!(
        "Unable to resolve SD3.5 ONNX component from any of: {tried}. Last error: {}",
        last_error
            .map(|e| e.to_string())
            .unwrap_or_else(|| "unknown".to_string())
    )))
}
