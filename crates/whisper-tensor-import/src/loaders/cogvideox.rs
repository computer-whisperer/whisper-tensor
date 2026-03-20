use std::path::PathBuf;
use std::sync::Arc;
use whisper_tensor::interfaces::VideoGenerationInterface;
use whisper_tensor::loader::*;
use whisper_tensor::metadata::TokenizerInfo;
use whisper_tensor::model::Model;

use crate::onnx_graph::weights::SafetensorsWeightManager;

/// Loader for CogVideoX models in HuggingFace Diffusers format.
///
/// Expects a directory containing:
/// - `transformer/` — DiT safetensors + config.json
/// - `vae/` — 3D VAE safetensors + config.json
/// - `text_encoder/` — T5-XXL safetensors
/// - `scheduler/scheduler_config.json`
/// - `tokenizer/` — T5 tokenizer files
pub struct CogVideoXLoader;

impl Loader for CogVideoXLoader {
    fn name(&self) -> &str {
        "CogVideoX"
    }

    fn description(&self) -> &str {
        "Load CogVideoX text-to-video model from a HuggingFace Diffusers directory"
    }

    fn config_schema(&self) -> Vec<ConfigField> {
        vec![ConfigField {
            key: "path".to_string(),
            label: "Model Directory".to_string(),
            description: "Path to the CogVideoX model directory (e.g. THUDM/CogVideoX-2b)"
                .to_string(),
            field_type: ConfigFieldType::FilePath,
            required: true,
            default: None,
        }]
    }

    fn load(&self, config: ConfigValues) -> Result<LoaderOutput, LoaderError> {
        let base_path = require_path(&config, "path")?;
        load_cogvideox(base_path)
    }
}

fn load_cogvideox(base_path: PathBuf) -> Result<LoaderOutput, LoaderError> {
    use memmap2::Mmap;

    let storage = super::shared::default_storage();

    // Detect variant from transformer config
    let transformer_dir = base_path.join("transformer");
    let vae_dir = base_path.join("vae");
    let text_encoder_dir = base_path.join("text_encoder");

    // Read transformer config to detect 2B vs 5B
    let transformer_config_path = transformer_dir.join("config.json");
    let transformer_config_json: serde_json::Value = {
        let data = std::fs::read_to_string(&transformer_config_path)
            .map_err(|e| LoaderError::LoadFailed(e.into()))?;
        serde_json::from_str(&data).map_err(|e| LoaderError::LoadFailed(e.into()))?
    };
    let num_attention_heads = transformer_config_json["num_attention_heads"]
        .as_u64()
        .unwrap_or(30) as usize;
    let use_rope = transformer_config_json["use_rotary_positional_embeddings"]
        .as_bool()
        .unwrap_or(false);

    let (variant, dit_config, vae_config) = if num_attention_heads >= 48 || use_rope {
        (
            "5b",
            crate::models::diffusion::cogvideox::CogVideoXTransformerConfig::cogvideox_5b(),
            crate::models::diffusion::cogvideox::CogVideoXVaeConfig::cogvideox_5b(),
        )
    } else {
        (
            "2b",
            crate::models::diffusion::cogvideox::CogVideoXTransformerConfig::cogvideox_2b(),
            crate::models::diffusion::cogvideox::CogVideoXVaeConfig::cogvideox_2b(),
        )
    };
    println!("Detected CogVideoX-{variant}");

    // Load safetensors files from a directory (may be sharded)
    let load_safetensors_dir =
        |dir: &std::path::Path| -> Result<SafetensorsWeightManager, LoaderError> {
            let mut mmaps = Vec::new();
            let mut paths = Vec::new();
            // Collect all .safetensors files in the directory
            let mut entries: Vec<_> = std::fs::read_dir(dir)
                .map_err(|e| LoaderError::LoadFailed(e.into()))?
                .filter_map(|e| e.ok())
                .filter(|e| e.path().extension().is_some_and(|ext| ext == "safetensors"))
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

    // Build T5-XXL encoder
    println!("Building T5-XXL encoder...");
    let t5_wm = load_safetensors_dir(&text_encoder_dir)?;
    let t5_config = crate::models::diffusion::t5::T5Config::t5_xxl(226);
    let t5_onnx = crate::models::diffusion::t5::load_t5_encoder(t5_wm, t5_config, storage.clone())
        .map_err(LoaderError::LoadFailed)?;

    // Build DiT transformer
    println!("Building CogVideoX DiT transformer...");
    let dit_wm = load_safetensors_dir(&transformer_dir)?;
    let dit_onnx = crate::models::diffusion::cogvideox::load_cogvideox_transformer(
        dit_wm,
        dit_config,
        storage.clone(),
    )
    .map_err(LoaderError::LoadFailed)?;

    // Build VAE decoder
    println!("Building CogVideoX VAE decoder...");
    let vae_wm = load_safetensors_dir(&vae_dir)?;
    let vae_onnx = crate::models::diffusion::cogvideox::load_cogvideox_vae_decoder(
        vae_wm,
        vae_config,
        storage.clone(),
    )
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
            name: format!("cogvideox-{variant}-{suffix}"),
            model: Arc::new(model),
        });
    }

    // Build VideoGenerationInterface
    let interface = {
        let mut rng = rand::rng();
        VideoGenerationInterface::new_cogvideox(
            &mut rng,
            TokenizerInfo::HFTokenizer("google-t5/t5-base".to_string()),
            whisper_tensor::dtype::DType::BF16,
        )
    };

    let interfaces = vec![LoadedInterface {
        name: format!("cogvideox-{variant}-VideoGeneration"),
        interface: interface.to_any(),
    }];

    Ok(LoaderOutput { models, interfaces })
}
