use super::path::resolve_component_safetensors;
use crate::onnx_graph::weights::{SafetensorsWeightManager, WeightManager};
use memmap2::Mmap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use whisper_tensor::dtype::DType;
use whisper_tensor::interfaces::ImageGenerationInterface;
use whisper_tensor::loader::{LoadedInterface, LoadedModel, LoaderError, LoaderOutput};
use whisper_tensor::metadata::TokenizerInfo;
use whisper_tensor::model::Model;

fn with_safetensors_manager<R>(
    files: &[PathBuf],
    f: impl FnOnce(SafetensorsWeightManager) -> Result<R, anyhow::Error>,
) -> Result<R, LoaderError> {
    let canonical_paths: Vec<PathBuf> = files
        .iter()
        .map(|p| std::fs::canonicalize(p).unwrap_or_else(|_| p.clone()))
        .collect();
    let mut mmaps = Vec::with_capacity(files.len());
    for path in files {
        let file = std::fs::File::open(path).map_err(|e| LoaderError::LoadFailed(e.into()))?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| LoaderError::LoadFailed(e.into()))?;
        mmaps.push(Arc::new(mmap));
    }
    let wm = SafetensorsWeightManager::new_with_paths(mmaps, canonical_paths)
        .map_err(|e| LoaderError::LoadFailed(e.into()))?;
    f(wm).map_err(LoaderError::LoadFailed)
}

#[derive(Clone)]
struct ClipProjectionRemapWeightManager {
    root: Arc<SafetensorsWeightManager>,
    prefix_tail: Option<String>,
    prefix: Option<String>,
}

impl ClipProjectionRemapWeightManager {
    fn new(root: SafetensorsWeightManager) -> Self {
        Self {
            root: Arc::new(root),
            prefix_tail: None,
            prefix: None,
        }
    }
}

impl WeightManager for ClipProjectionRemapWeightManager {
    fn prefix(&self, name: &str) -> Self {
        let prefix = Some(if let Some(prefix) = &self.prefix {
            format!("{prefix}.{name}")
        } else {
            name.to_string()
        });
        Self {
            root: self.root.clone(),
            prefix_tail: Some(name.to_string()),
            prefix,
        }
    }

    fn get_tensor(
        &self,
        name: &str,
    ) -> Result<Arc<dyn crate::onnx_graph::tensor::Tensor>, crate::onnx_graph::Error> {
        let full_name = if let Some(prefix) = &self.prefix {
            format!("{prefix}.{name}")
        } else {
            name.to_string()
        };
        let mapped = if let Some(rest) = full_name.strip_prefix("text_model.text_projection.") {
            format!("text_projection.{rest}")
        } else {
            full_name
        };
        self.root.get_tensor(&mapped)
    }

    fn get_prefix_tail(&self) -> Option<&str> {
        self.prefix_tail.as_deref()
    }

    fn get_prefix(&self) -> Option<&str> {
        self.prefix.as_deref()
    }

    fn get_tensor_names(&self) -> Vec<String> {
        self.root.get_tensor_names()
    }
}

fn onnx_dtype_to_runtime(dtype: crate::onnx_graph::tensor::DType) -> Result<DType, LoaderError> {
    match dtype {
        crate::onnx_graph::tensor::DType::F16 => Ok(DType::F16),
        crate::onnx_graph::tensor::DType::BF16 => Ok(DType::BF16),
        crate::onnx_graph::tensor::DType::F32 => Ok(DType::F32),
        other => Err(LoaderError::LoadFailed(anyhow::anyhow!(
            "Unsupported model dtype for SD3.5: {:?}",
            other
        ))),
    }
}

#[derive(serde::Deserialize)]
struct ClipConfigJson {
    hidden_size: usize,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    max_position_embeddings: usize,
    layer_norm_eps: f32,
    hidden_act: String,
}

#[derive(serde::Deserialize)]
struct Sd3TransformerConfigJson {
    attention_head_dim: usize,
    num_attention_heads: usize,
    num_layers: usize,
    dual_attention_layers: Vec<usize>,
    in_channels: usize,
    out_channels: usize,
    patch_size: usize,
    sample_size: usize,
    pos_embed_max_size: usize,
    joint_attention_dim: usize,
    pooled_projection_dim: usize,
}

#[derive(serde::Deserialize)]
struct VaeConfigJson {
    scaling_factor: Option<f32>,
    shift_factor: Option<f32>,
}

fn parse_json_file<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, LoaderError> {
    let file = std::fs::File::open(path).map_err(|e| LoaderError::LoadFailed(e.into()))?;
    serde_json::from_reader(file).map_err(|e| LoaderError::LoadFailed(e.into()))
}

fn clip_hidden_act_to_enum(act: &str) -> crate::models::diffusion::sd_clip::ClipMlpActivation {
    match act {
        "gelu" => crate::models::diffusion::sd_clip::ClipMlpActivation::Gelu,
        _ => crate::models::diffusion::sd_clip::ClipMlpActivation::QuickGelu,
    }
}

pub(super) fn load_diffusers_sd3_pipeline(
    path: &Path,
    clip_tokenizer: String,
    t5_tokenizer: String,
    t5_seq_len_override: Option<usize>,
    vae_scale_factor_override: Option<f32>,
    vae_shift_factor_override: Option<f32>,
) -> Result<LoaderOutput, LoaderError> {
    let storage = crate::onnx_graph::WeightStorageStrategy::OriginReference;

    let clip_l_files = resolve_component_safetensors(path, "text_encoder")?;
    let clip_g_files = resolve_component_safetensors(path, "text_encoder_2")?;
    let t5_files = resolve_component_safetensors(path, "text_encoder_3")?;
    let transformer_files = resolve_component_safetensors(path, "transformer")?;
    let vae_files = resolve_component_safetensors(path, "vae")?;

    let clip_l_cfg: ClipConfigJson =
        parse_json_file(&path.join("text_encoder").join("config.json"))?;
    let clip_g_cfg: ClipConfigJson =
        parse_json_file(&path.join("text_encoder_2").join("config.json"))?;
    let transformer_cfg: Sd3TransformerConfigJson =
        parse_json_file(&path.join("transformer").join("config.json"))?;
    let vae_cfg: VaeConfigJson = parse_json_file(&path.join("vae").join("config.json"))?;

    let model_dtype = with_safetensors_manager(&transformer_files, |wm| {
        Ok(
            crate::models::diffusion::sd_common::detect_model_dtype_with_canary(
                &wm,
                "transformer_blocks.0.attn.to_q.weight",
            ),
        )
    })
    .and_then(onnx_dtype_to_runtime)?;
    let import_dtype = match model_dtype {
        DType::F16 => crate::onnx_graph::tensor::DType::F16,
        DType::BF16 => crate::onnx_graph::tensor::DType::BF16,
        DType::F32 => crate::onnx_graph::tensor::DType::F32,
        _ => crate::onnx_graph::tensor::DType::F16,
    };

    let t5_seq_len = t5_seq_len_override.unwrap_or(256);
    let vae_scale_factor = vae_scale_factor_override
        .or(vae_cfg.scaling_factor)
        .unwrap_or(1.5305);
    let vae_shift_factor = vae_shift_factor_override
        .or(vae_cfg.shift_factor)
        .unwrap_or(0.0609);

    println!("Building SD3.5 CLIP-L text encoder from safetensors...");
    let clip_l_onnx = with_safetensors_manager(&clip_l_files, |wm| {
        let wm = ClipProjectionRemapWeightManager::new(wm);
        let cfg = crate::models::diffusion::sd_clip::ClipTextModelConfig {
            hidden_dim: clip_l_cfg.hidden_size,
            num_heads: clip_l_cfg.num_attention_heads,
            num_layers: clip_l_cfg.num_hidden_layers,
            max_position: clip_l_cfg.max_position_embeddings,
            layer_norm_eps: clip_l_cfg.layer_norm_eps,
            mlp_activation: clip_hidden_act_to_enum(&clip_l_cfg.hidden_act),
            hidden_source: crate::models::diffusion::sd_clip::ClipHiddenStateSource::Penultimate,
            output_pooled_projection: true,
        };
        crate::models::diffusion::sd_clip::build_clip_text_model_with_projection(
            wm,
            import_dtype,
            storage.clone(),
            &clip_l_files[0],
            "text_model",
            cfg,
        )
    })?;

    println!("Building SD3.5 CLIP-G text encoder from safetensors...");
    let clip_g_onnx = with_safetensors_manager(&clip_g_files, |wm| {
        let wm = ClipProjectionRemapWeightManager::new(wm);
        let cfg = crate::models::diffusion::sd_clip::ClipTextModelConfig {
            hidden_dim: clip_g_cfg.hidden_size,
            num_heads: clip_g_cfg.num_attention_heads,
            num_layers: clip_g_cfg.num_hidden_layers,
            max_position: clip_g_cfg.max_position_embeddings,
            layer_norm_eps: clip_g_cfg.layer_norm_eps,
            mlp_activation: clip_hidden_act_to_enum(&clip_g_cfg.hidden_act),
            hidden_source: crate::models::diffusion::sd_clip::ClipHiddenStateSource::Penultimate,
            output_pooled_projection: true,
        };
        crate::models::diffusion::sd_clip::build_clip_text_model_with_projection(
            wm,
            import_dtype,
            storage.clone(),
            &clip_g_files[0],
            "text_model",
            cfg,
        )
    })?;

    println!("Building SD3.5 T5-XXL encoder from safetensors...");
    let t5_onnx = with_safetensors_manager(&t5_files, |wm| {
        let t5_cfg = crate::models::diffusion::t5::T5Config::t5_xxl(t5_seq_len);
        crate::models::diffusion::t5::load_t5_encoder_with_origin(
            wm,
            t5_cfg,
            storage.clone(),
            Some(&t5_files[0]),
        )
    })?;

    println!("Building SD3.5 transformer from safetensors...");
    let sd3_transformer_cfg = crate::models::diffusion::sd3::Sd3TransformerConfig {
        num_layers: transformer_cfg.num_layers,
        num_heads: transformer_cfg.num_attention_heads,
        head_dim: transformer_cfg.attention_head_dim,
        hidden_dim: transformer_cfg.num_attention_heads * transformer_cfg.attention_head_dim,
        latent_channels: transformer_cfg.in_channels,
        out_channels: transformer_cfg.out_channels,
        patch_size: transformer_cfg.patch_size,
        sample_size: transformer_cfg.sample_size,
        pos_embed_max_size: transformer_cfg.pos_embed_max_size,
        joint_attention_dim: transformer_cfg.joint_attention_dim,
        pooled_projection_dim: transformer_cfg.pooled_projection_dim,
        context_seq_len: 77 + t5_seq_len,
        dual_attention_layers: transformer_cfg.dual_attention_layers.into_iter().collect(),
    };
    let transformer_onnx = with_safetensors_manager(&transformer_files, |wm| {
        crate::models::diffusion::sd3::load_sd3_transformer_with_origin(
            wm,
            sd3_transformer_cfg,
            storage.clone(),
            Some(&transformer_files[0]),
        )
    })?;

    println!("Building SD3.5 VAE decoder from safetensors...");
    let vae_onnx = with_safetensors_manager(&vae_files, |wm| {
        crate::models::diffusion::sd3::build_sd3_vae_decoder(
            wm,
            import_dtype,
            storage.clone(),
            &vae_files[0],
        )
    })?;

    let base_name = path
        .file_name()
        .and_then(|x| x.to_str())
        .unwrap_or("sd35")
        .to_string();

    let component_dirs = [
        path.join("text_encoder"),
        path.join("text_encoder_2"),
        path.join("text_encoder_3"),
        path.join("transformer"),
        path.join("vae"),
    ];
    let model_bytes = [
        ("clip_l", clip_l_onnx),
        ("clip_g", clip_g_onnx),
        ("t5", t5_onnx),
        ("transformer", transformer_onnx),
        ("vae_decoder", vae_onnx),
    ];

    let mut models = Vec::new();
    for ((suffix, onnx_data), base_dir) in model_bytes.into_iter().zip(component_dirs.iter()) {
        let mut rng = rand::rng();
        let model = Model::new_from_onnx(&onnx_data, &mut rng, Some(base_dir.as_path()))
            .map_err(|e| LoaderError::LoadFailed(e.into()))?;
        models.push(LoadedModel {
            name: format!("{base_name}-{suffix}"),
            model: Arc::new(model),
        });
    }

    let interface = {
        let mut rng = rand::rng();
        ImageGenerationInterface::new_sd3(
            &mut rng,
            TokenizerInfo::HFTokenizer(clip_tokenizer),
            TokenizerInfo::HFTokenizer(t5_tokenizer),
            model_dtype,
            t5_seq_len,
            "input_ids",
            Some("eos_indices"),
            "last_hidden_state",
            "pooled_output",
            "input_ids",
            Some("eos_indices"),
            "last_hidden_state",
            "pooled_output",
            "input_ids",
            "hidden_states",
            "latent_sample",
            "timestep",
            "encoder_hidden_states",
            "pooled_projections",
            "out_sample",
            "latent_sample",
            "sample",
            vae_scale_factor,
            vae_shift_factor,
            transformer_cfg.in_channels,
        )
    };

    let interfaces = vec![LoadedInterface {
        name: format!("{base_name}-ImageGeneration"),
        interface: interface.to_any(),
    }];

    Ok(LoaderOutput { models, interfaces })
}
