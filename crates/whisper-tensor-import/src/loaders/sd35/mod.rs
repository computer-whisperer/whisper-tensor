mod diffusers;
mod io_infer;
mod path;

use self::diffusers::load_diffusers_sd3_pipeline;
use self::io_infer::{
    infer_clip_io, infer_latent_channels, infer_model_dtype, infer_t5_io, infer_t5_seq_len,
    infer_transformer_io, infer_vae_io,
};
use self::path::{
    is_sd3_diffusers_safetensors_dir, load_onnx_model, resolve_component_onnx,
    resolve_component_onnx_any,
};
use whisper_tensor::dtype::DType;
use whisper_tensor::interfaces::ImageGenerationInterface;
use whisper_tensor::loader::*;
use whisper_tensor::metadata::TokenizerInfo;

/// Loader for Stable Diffusion 3.5 ONNX pipelines.
///
/// Expected directory layout:
/// - text_encoder/model.onnx
/// - text_encoder_2/model.onnx
/// - text_encoder_3/model.onnx
/// - transformer/model.onnx
/// - vae_decoder/model.onnx
pub struct SD35Loader;

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
