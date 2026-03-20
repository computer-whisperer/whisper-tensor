use std::path::PathBuf;
use std::sync::Arc;
use whisper_tensor::interfaces::VideoGenerationInterface;
use whisper_tensor::loader::*;
use whisper_tensor::metadata::TokenizerInfo;
use whisper_tensor::model::Model;

use crate::onnx_graph::weights::SafetensorsWeightManager;

/// Loader for Wan2.1 text-to-video models in HuggingFace Diffusers format.
///
/// Expects a directory containing:
/// - `transformer/` — DiT safetensors + config.json
/// - `vae/` — 3D VAE safetensors
/// - `text_encoder/` — UMT5-XXL safetensors
/// - `scheduler/scheduler_config.json`
pub struct WanLoader;

impl Loader for WanLoader {
    fn name(&self) -> &str {
        "Wan2.1"
    }

    fn description(&self) -> &str {
        "Load Wan2.1 text-to-video model from a HuggingFace Diffusers directory"
    }

    fn config_schema(&self) -> Vec<ConfigField> {
        vec![ConfigField {
            key: "path".to_string(),
            label: "Model Directory".to_string(),
            description: "Path to the Wan2.1 model directory (e.g. Wan-AI/Wan2.1-T2V-1.3B)"
                .to_string(),
            field_type: ConfigFieldType::FilePath,
            required: true,
            default: None,
        }]
    }

    fn load(&self, config: ConfigValues) -> Result<LoaderOutput, LoaderError> {
        let base_path = require_path(&config, "path")?;
        load_wan(base_path)
    }
}

fn load_wan(base_path: PathBuf) -> Result<LoaderOutput, LoaderError> {
    use memmap2::Mmap;

    let storage = super::shared::default_storage();

    let transformer_dir = base_path.join("transformer");
    let vae_dir = base_path.join("vae");
    let text_encoder_dir = base_path.join("text_encoder");

    // Detect variant from transformer config
    let transformer_config_path = transformer_dir.join("config.json");
    let transformer_config_json: serde_json::Value = {
        let data = std::fs::read_to_string(&transformer_config_path)
            .map_err(|e| LoaderError::LoadFailed(e.into()))?;
        serde_json::from_str(&data).map_err(|e| LoaderError::LoadFailed(e.into()))?
    };
    let dim = transformer_config_json["dim"].as_u64().unwrap_or(1536) as usize;

    let (variant, dit_config) = if dim >= 5120 {
        ("14b", crate::models::diffusion::wan::WanTransformerConfig::wan_14b())
    } else {
        ("1.3b", crate::models::diffusion::wan::WanTransformerConfig::wan_1_3b())
    };
    println!("Detected Wan2.1-T2V-{variant}");

    let vae_config = crate::models::diffusion::wan::WanVaeConfig::default_config();

    // Load safetensors from a directory (handles sharded files)
    let load_safetensors_dir =
        |dir: &std::path::Path| -> Result<SafetensorsWeightManager, LoaderError> {
            let mut mmaps = Vec::new();
            let mut paths = Vec::new();
            let mut entries: Vec<_> = std::fs::read_dir(dir)
                .map_err(|e| LoaderError::LoadFailed(e.into()))?
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.path()
                        .extension()
                        .is_some_and(|ext| ext == "safetensors")
                })
                .collect();
            entries.sort_by_key(|e| e.path());
            for entry in entries {
                let path = entry.path();
                let file =
                    std::fs::File::open(&path).map_err(|e| LoaderError::LoadFailed(e.into()))?;
                let mmap =
                    unsafe { Mmap::map(&file) }.map_err(|e| LoaderError::LoadFailed(e.into()))?;
                mmaps.push(Arc::new(mmap));
                paths.push(path);
            }
            if mmaps.is_empty() {
                return Err(LoaderError::LoadFailed(anyhow::anyhow!(
                    "No .safetensors files found in {}",
                    dir.display()
                )));
            }
            SafetensorsWeightManager::new_with_paths(mmaps, paths)
                .map_err(|e| LoaderError::LoadFailed(e.into()))
        };

    // Build T5 encoder (UMT5-XXL has same architecture as T5-XXL)
    println!("Building T5/UMT5-XXL encoder...");
    let t5_wm = load_safetensors_dir(&text_encoder_dir)?;
    let t5_config = crate::models::diffusion::t5::T5Config::t5_xxl(512);
    let t5_onnx = crate::models::diffusion::t5::load_t5_encoder(t5_wm, t5_config, storage.clone())
        .map_err(LoaderError::LoadFailed)?;

    // Build DiT transformer
    println!("Building Wan2.1 DiT transformer...");
    let dit_wm = load_safetensors_dir(&transformer_dir)?;
    let dit_onnx =
        crate::models::diffusion::wan::load_wan_transformer(dit_wm, dit_config, storage.clone())
            .map_err(LoaderError::LoadFailed)?;

    // Build VAE decoder
    println!("Building Wan2.1 VAE decoder...");
    let vae_wm = load_safetensors_dir(&vae_dir)?;
    let vae_onnx =
        crate::models::diffusion::wan::load_wan_vae_decoder(vae_wm, vae_config, storage.clone())
            .map_err(LoaderError::LoadFailed)?;

    // Create Model objects
    let base_dir = Some(base_path.as_path());
    let mut models = Vec::new();
    for (suffix, onnx_data) in [
        ("t5_xxl", t5_onnx),
        ("dit", dit_onnx),
        ("vae_decoder", vae_onnx),
    ] {
        let mut rng = rand::rng();
        let model = Model::new_from_onnx(&onnx_data, &mut rng, base_dir)
            .map_err(|e| LoaderError::LoadFailed(e.into()))?;
        models.push(LoadedModel {
            name: format!("wan2.1-{variant}-{suffix}"),
            model: Arc::new(model),
        });
    }

    // Build VideoGenerationInterface
    // Wan uses flow matching (same as Flux) — RectifiedFlow scheduler with Euler step
    let interface = {
        let mut rng = rand::rng();
        VideoGenerationInterface::new_wan(
            &mut rng,
            TokenizerInfo::HFTokenizer("google/umt5-xxl".to_string()),
            whisper_tensor::dtype::DType::BF16,
        )
    };

    let interfaces = vec![LoadedInterface {
        name: format!("wan2.1-{variant}-VideoGeneration"),
        interface: interface.to_any(),
    }];

    Ok(LoaderOutput { models, interfaces })
}
