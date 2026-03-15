use crate::onnx_graph::weights::{SafetensorsWeightManager, WeightManager};
use memmap2::Mmap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use whisper_tensor::dtype::DType;
use whisper_tensor::interfaces::ImageGenerationInterface;
use whisper_tensor::loader::*;
use whisper_tensor::metadata::TokenizerInfo;
use whisper_tensor::model::Model;
use whisper_tensor::symbolic_graph::SymbolicGraph;

/// Loader for Stable Diffusion 3.5 ONNX pipelines.
///
/// Expected directory layout:
/// - text_encoder/model.onnx
/// - text_encoder_2/model.onnx
/// - text_encoder_3/model.onnx
/// - transformer/model.onnx
/// - vae_decoder/model.onnx
pub struct SD35Loader;

struct ClipEncoderIo {
    input: String,
    eos_input: Option<String>,
    hidden_output: String,
    pooled_output: String,
}

struct T5EncoderIo {
    input: String,
    hidden_output: String,
}

struct TransformerIo {
    latent_input: String,
    timestep_input: String,
    context_input: String,
    pooled_input: String,
    output: String,
}

struct VaeDecoderIo {
    input: String,
    output: String,
}

impl Loader for SD35Loader {
    fn name(&self) -> &str {
        "Stable Diffusion 3.5"
    }

    fn description(&self) -> &str {
        "Load SD3.5 from official diffusers safetensors or ONNX pipeline directories"
    }

    fn config_schema(&self) -> Vec<ConfigField> {
        vec![
            ConfigField {
                key: "path".to_string(),
                label: "Pipeline Path".to_string(),
                description:
                    "Path to an SD3.5 directory (official diffusers safetensors or ONNX pipeline)"
                        .to_string(),
                field_type: ConfigFieldType::FilePath,
                required: true,
                default: None,
            },
            ConfigField {
                key: "clip_tokenizer".to_string(),
                label: "CLIP Tokenizer".to_string(),
                description:
                    "HF tokenizer for CLIP encoders (default: openai/clip-vit-large-patch14)"
                        .to_string(),
                field_type: ConfigFieldType::String,
                required: false,
                default: Some(ConfigValue::String(
                    "openai/clip-vit-large-patch14".to_string(),
                )),
            },
            ConfigField {
                key: "t5_tokenizer".to_string(),
                label: "T5 Tokenizer".to_string(),
                description: "HF tokenizer for T5 encoder (default: google-t5/t5-base)".to_string(),
                field_type: ConfigFieldType::String,
                required: false,
                default: Some(ConfigValue::String("google-t5/t5-base".to_string())),
            },
            ConfigField {
                key: "t5_seq_len".to_string(),
                label: "T5 Sequence Length".to_string(),
                description: "Prompt token length for T5 encoder (default: inferred or 256)"
                    .to_string(),
                field_type: ConfigFieldType::Integer {
                    min: Some(1),
                    max: Some(1024),
                },
                required: false,
                default: Some(ConfigValue::Integer(256)),
            },
            ConfigField {
                key: "vae_scale_factor".to_string(),
                label: "VAE Scale Factor".to_string(),
                description: "Latent scale factor before VAE decode (default: 1.5305 for SD3.x)"
                    .to_string(),
                field_type: ConfigFieldType::Float {
                    min: Some(0.0001),
                    max: None,
                },
                required: false,
                default: Some(ConfigValue::Float(1.5305)),
            },
            ConfigField {
                key: "vae_shift_factor".to_string(),
                label: "VAE Shift Factor".to_string(),
                description: "Latent shift factor before VAE decode (default: 0.0609 for SD3.x)"
                    .to_string(),
                field_type: ConfigFieldType::Float {
                    min: None,
                    max: None,
                },
                required: false,
                default: Some(ConfigValue::Float(0.0609)),
            },
        ]
    }

    fn load(&self, config: ConfigValues) -> Result<LoaderOutput, LoaderError> {
        let path = require_path(&config, "path")?;
        if !path.is_dir() {
            return Err(LoaderError::LoadFailed(anyhow::anyhow!(
                "SD3.5 loader expects a directory, got: {}",
                path.display()
            )));
        }

        let clip_tokenizer = get_string(&config, "clip_tokenizer")?
            .unwrap_or_else(|| "openai/clip-vit-large-patch14".to_string());
        let t5_tokenizer =
            get_string(&config, "t5_tokenizer")?.unwrap_or_else(|| "google-t5/t5-base".to_string());
        let t5_seq_len_override = match config.get("t5_seq_len") {
            Some(ConfigValue::Integer(v)) => Some(*v as usize),
            Some(_) => {
                return Err(LoaderError::InvalidValue {
                    field: "t5_seq_len".to_string(),
                    reason: "expected integer".to_string(),
                });
            }
            None => None,
        };
        let vae_scale_factor_override = match config.get("vae_scale_factor") {
            Some(ConfigValue::Float(v)) => Some(*v as f32),
            Some(ConfigValue::Integer(v)) => Some(*v as f32),
            Some(_) => {
                return Err(LoaderError::InvalidValue {
                    field: "vae_scale_factor".to_string(),
                    reason: "expected float".to_string(),
                });
            }
            None => None,
        };
        let vae_shift_factor_override = match config.get("vae_shift_factor") {
            Some(ConfigValue::Float(v)) => Some(*v as f32),
            Some(ConfigValue::Integer(v)) => Some(*v as f32),
            Some(_) => {
                return Err(LoaderError::InvalidValue {
                    field: "vae_shift_factor".to_string(),
                    reason: "expected float".to_string(),
                });
            }
            None => None,
        };

        if is_sd3_diffusers_safetensors_dir(&path) {
            return load_diffusers_sd3_pipeline(
                &path,
                clip_tokenizer,
                t5_tokenizer,
                t5_seq_len_override,
                vae_scale_factor_override,
                vae_shift_factor_override,
            );
        }

        let vae_scale_factor = vae_scale_factor_override.unwrap_or(1.5305);
        let vae_shift_factor = vae_shift_factor_override.unwrap_or(0.0609);

        let clip_l_path = resolve_component_onnx(&path, "text_encoder")?;
        let clip_g_path = resolve_component_onnx(&path, "text_encoder_2")?;
        let t5_path = resolve_component_onnx(&path, "text_encoder_3")?;
        let transformer_path = resolve_component_onnx(&path, "transformer")?;
        let vae_path = resolve_component_onnx_any(&path, &["vae_decoder", "vae"])?;

        let clip_l_model = load_onnx_model(&clip_l_path)?;
        let clip_g_model = load_onnx_model(&clip_g_path)?;
        let t5_model = load_onnx_model(&t5_path)?;
        let transformer_model = load_onnx_model(&transformer_path)?;
        let vae_model = load_onnx_model(&vae_path)?;

        let clip_l_io = infer_clip_io(clip_l_model.get_symbolic_graph(), "text_encoder")?;
        let clip_g_io = infer_clip_io(clip_g_model.get_symbolic_graph(), "text_encoder_2")?;
        let t5_io = infer_t5_io(t5_model.get_symbolic_graph(), "text_encoder_3")?;
        let transformer_io =
            infer_transformer_io(transformer_model.get_symbolic_graph(), "transformer")?;
        let vae_io = infer_vae_io(vae_model.get_symbolic_graph(), "vae_decoder")?;

        let model_dtype =
            infer_model_dtype(transformer_model.get_symbolic_graph(), &transformer_io)
                .unwrap_or(DType::F16);
        let latent_channels =
            infer_latent_channels(transformer_model.get_symbolic_graph(), &transformer_io)
                .unwrap_or(16);
        let inferred_t5_seq_len = infer_t5_seq_len(t5_model.get_symbolic_graph(), &t5_io);
        let t5_seq_len = t5_seq_len_override.or(inferred_t5_seq_len).unwrap_or(256);

        let base_name = path
            .file_name()
            .and_then(|x| x.to_str())
            .unwrap_or("sd35")
            .to_string();

        let models = vec![
            LoadedModel {
                name: format!("{base_name}-clip_l"),
                model: clip_l_model.clone(),
            },
            LoadedModel {
                name: format!("{base_name}-clip_g"),
                model: clip_g_model.clone(),
            },
            LoadedModel {
                name: format!("{base_name}-t5"),
                model: t5_model.clone(),
            },
            LoadedModel {
                name: format!("{base_name}-transformer"),
                model: transformer_model.clone(),
            },
            LoadedModel {
                name: format!("{base_name}-vae_decoder"),
                model: vae_model.clone(),
            },
        ];

        let interface = {
            let mut rng = rand::rng();
            ImageGenerationInterface::new_sd3(
                &mut rng,
                TokenizerInfo::HFTokenizer(clip_tokenizer),
                TokenizerInfo::HFTokenizer(t5_tokenizer),
                model_dtype,
                t5_seq_len,
                &clip_l_io.input,
                clip_l_io.eos_input.as_deref(),
                &clip_l_io.hidden_output,
                &clip_l_io.pooled_output,
                &clip_g_io.input,
                clip_g_io.eos_input.as_deref(),
                &clip_g_io.hidden_output,
                &clip_g_io.pooled_output,
                &t5_io.input,
                &t5_io.hidden_output,
                &transformer_io.latent_input,
                &transformer_io.timestep_input,
                &transformer_io.context_input,
                &transformer_io.pooled_input,
                &transformer_io.output,
                &vae_io.input,
                &vae_io.output,
                vae_scale_factor,
                vae_shift_factor,
                latent_channels,
            )
        };

        let interfaces = vec![LoadedInterface {
            name: format!("{base_name}-ImageGeneration"),
            interface: interface.to_any(),
        }];

        Ok(LoaderOutput { models, interfaces })
    }
}

fn load_onnx_model(path: &Path) -> Result<Arc<Model>, LoaderError> {
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

fn is_sd3_diffusers_safetensors_dir(root: &Path) -> bool {
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

fn resolve_component_safetensors(
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

fn load_diffusers_sd3_pipeline(
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

fn resolve_component_onnx(root: &Path, component: &str) -> Result<PathBuf, LoaderError> {
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

fn resolve_component_onnx_any(root: &Path, components: &[&str]) -> Result<PathBuf, LoaderError> {
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

fn infer_clip_io(graph: &SymbolicGraph, model_name: &str) -> Result<ClipEncoderIo, LoaderError> {
    let (inputs, outputs) = list_io_names(graph);
    let input = pick_name(&inputs, &["input_ids"], "input_ids", model_name)?;
    let eos_input = inputs
        .iter()
        .find(|name| name.as_str() == "eos_indices" || name.as_str() == "eos_index")
        .or_else(|| {
            inputs
                .iter()
                .find(|name| name.to_ascii_lowercase().contains("eos"))
        })
        .and_then(|name| {
            let tid = graph.get_tensors_by_name().get(name).copied()?;
            let info = graph.get_tensor_info(tid)?;
            let shape = info.shape.as_ref()?;
            if shape.len() == 1 {
                Some(name.clone())
            } else {
                None
            }
        });
    let hidden_output = pick_name_with_rank(
        graph,
        &outputs,
        &["last_hidden_state", "hidden_states", "prompt_embeds"],
        Some(3),
        "hidden output",
        model_name,
    )?;
    let pooled_output = pick_name_with_rank(
        graph,
        &outputs,
        &[
            "text_embeds",
            "pooled_output",
            "pooler_output",
            "pooled_text_embeds",
        ],
        Some(2),
        "pooled output",
        model_name,
    )?;
    Ok(ClipEncoderIo {
        input,
        eos_input,
        hidden_output,
        pooled_output,
    })
}

fn infer_t5_io(graph: &SymbolicGraph, model_name: &str) -> Result<T5EncoderIo, LoaderError> {
    let (inputs, outputs) = list_io_names(graph);
    let input = pick_name(&inputs, &["input_ids"], "input_ids", model_name)?;
    let hidden_output = pick_name_with_rank(
        graph,
        &outputs,
        &[
            "last_hidden_state",
            "hidden_states",
            "encoder_hidden_states",
        ],
        Some(3),
        "hidden output",
        model_name,
    )?;
    Ok(T5EncoderIo {
        input,
        hidden_output,
    })
}

fn infer_transformer_io(
    graph: &SymbolicGraph,
    model_name: &str,
) -> Result<TransformerIo, LoaderError> {
    let (inputs, outputs) = list_io_names(graph);
    let latent_input = pick_name(
        &inputs,
        &["hidden_states", "latent_sample", "sample"],
        "latent input",
        model_name,
    )?;
    let timestep_input = pick_name(
        &inputs,
        &["timestep", "timesteps", "t"],
        "timestep input",
        model_name,
    )?;
    let context_input = pick_name(
        &inputs,
        &["encoder_hidden_states", "prompt_embeds", "context"],
        "encoder_hidden_states input",
        model_name,
    )?;
    let pooled_input = pick_name(
        &inputs,
        &["pooled_projections", "pooled_prompt_embeds", "text_embeds"],
        "pooled projections input",
        model_name,
    )?;
    let output = pick_name_with_rank(
        graph,
        &outputs,
        &["sample", "out_sample", "model_output"],
        Some(4),
        "model output",
        model_name,
    )?;
    Ok(TransformerIo {
        latent_input,
        timestep_input,
        context_input,
        pooled_input,
        output,
    })
}

fn infer_vae_io(graph: &SymbolicGraph, model_name: &str) -> Result<VaeDecoderIo, LoaderError> {
    let (inputs, outputs) = list_io_names(graph);
    let input = pick_name(
        &inputs,
        &["latent_sample", "latent", "sample"],
        "latent input",
        model_name,
    )?;
    let output = pick_name_with_rank(
        graph,
        &outputs,
        &["sample", "image", "decoded"],
        Some(4),
        "image output",
        model_name,
    )?;
    Ok(VaeDecoderIo { input, output })
}

fn infer_model_dtype(graph: &SymbolicGraph, io: &TransformerIo) -> Option<DType> {
    let tid = graph.get_tensors_by_name().get(&io.latent_input).copied()?;
    graph.get_tensor_info(tid)?.dtype
}

fn infer_t5_seq_len(graph: &SymbolicGraph, io: &T5EncoderIo) -> Option<usize> {
    let tid = graph.get_tensors_by_name().get(&io.input).copied()?;
    let info = graph.get_tensor_info(tid)?;
    let shape = info.shape.as_ref()?;
    shape
        .last()
        .and_then(|dim| dim.as_numeric().copied())
        .map(|v| v as usize)
}

fn infer_latent_channels(graph: &SymbolicGraph, io: &TransformerIo) -> Option<usize> {
    let tid = graph.get_tensors_by_name().get(&io.latent_input).copied()?;
    let info = graph.get_tensor_info(tid)?;
    let shape = info.shape.as_ref()?;
    if shape.len() < 2 {
        return None;
    }
    shape[1].as_numeric().copied().map(|v| v as usize)
}

fn list_io_names(graph: &SymbolicGraph) -> (Vec<String>, Vec<String>) {
    let inputs = graph
        .get_inputs()
        .into_iter()
        .filter_map(|id| graph.get_tensor_name(id).map(str::to_string))
        .collect::<Vec<_>>();
    let outputs = graph
        .get_outputs()
        .into_iter()
        .filter_map(|id| graph.get_tensor_name(id).map(str::to_string))
        .collect::<Vec<_>>();
    (inputs, outputs)
}

fn pick_name(
    available: &[String],
    candidates: &[&str],
    what: &str,
    model_name: &str,
) -> Result<String, LoaderError> {
    for candidate in candidates {
        if let Some(found) = available.iter().find(|x| x.as_str() == *candidate) {
            return Ok(found.clone());
        }
    }
    for candidate in candidates {
        let lower_candidate = candidate.to_ascii_lowercase();
        if let Some(found) = available
            .iter()
            .find(|x| x.to_ascii_lowercase().contains(&lower_candidate))
        {
            return Ok(found.clone());
        }
    }
    Err(LoaderError::LoadFailed(anyhow::anyhow!(
        "Could not infer {what} for {model_name}. Available names: {:?}",
        available
    )))
}

fn pick_name_with_rank(
    graph: &SymbolicGraph,
    available: &[String],
    candidates: &[&str],
    rank: Option<usize>,
    what: &str,
    model_name: &str,
) -> Result<String, LoaderError> {
    for candidate in candidates {
        if let Some(found) = available.iter().find(|x| x.as_str() == *candidate)
            && rank_matches(graph, found, rank)
        {
            return Ok(found.clone());
        }
    }
    for candidate in candidates {
        let lower_candidate = candidate.to_ascii_lowercase();
        if let Some(found) = available.iter().find(|x| {
            x.to_ascii_lowercase().contains(&lower_candidate) && rank_matches(graph, x, rank)
        }) {
            return Ok(found.clone());
        }
    }
    if let Some(expected_rank) = rank
        && let Some(found) = available
            .iter()
            .find(|x| rank_matches(graph, x, Some(expected_rank)))
    {
        return Ok(found.clone());
    }
    Err(LoaderError::LoadFailed(anyhow::anyhow!(
        "Could not infer {what} for {model_name}. Available names: {:?}",
        available
    )))
}

fn rank_matches(graph: &SymbolicGraph, name: &str, rank: Option<usize>) -> bool {
    let Some(expected_rank) = rank else {
        return true;
    };
    let Some(tid) = graph.get_tensors_by_name().get(name).copied() else {
        return false;
    };
    let Some(info) = graph.get_tensor_info(tid) else {
        return false;
    };
    let Some(shape) = &info.shape else {
        return false;
    };
    shape.len() == expected_rank
}
