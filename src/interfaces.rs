use crate::metadata::TokenizerInfo;
use crate::model::Model;
use crate::numeric_dtype::NumericDType;
use crate::numeric_scalar::NumericScalar;
use crate::numeric_tensor::NumericTensor;
use crate::pool::Pool;
use crate::super_graph::cache::SuperGraphCache;
use crate::super_graph::data::{SuperGraphData, SuperGraphImage};
use crate::super_graph::links::SuperGraphLink;
use crate::super_graph::{SuperGraph, SuperGraphContext, SuperGraphError};
use crate::tensor_rank::DynRank;
use crate::tokenizer::{AnyTokenizer, Tokenizer};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Find the index of the maximum element in a tensor view (argmax over all elements).
fn argmax_flat(view: &crate::numeric_tensor::NumericTensorView<'_, DynRank>) -> u32 {
    let n = view.numel();
    assert!(n > 0, "argmax on empty tensor");
    let mut best_idx: u32 = 0;
    let mut best_val = f64::NEG_INFINITY;
    for i in 0..n {
        let v = view.read_element(i).to_f64();
        if v > best_val {
            best_val = v;
            best_idx = i as u32;
        }
    }
    best_idx
}

fn alloc_err(e: crate::pool::AllocationError) -> SuperGraphError {
    SuperGraphError::InvalidInputError(format!("allocation: {e}"))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum AnyInterface {
    TextInferenceTokensInLogitOutInterface(TextInferenceTokensInLogitOutInterface),
    MultimodalLanguageInterface(MultimodalLanguageInterface),
    ImageGenerationInterface(ImageGenerationInterface),
    VideoGenerationInterface(VideoGenerationInterface),
    TextToSpeechInterface(TextToSpeechInterface),
    SpeechToTextInterface(SpeechToTextInterface),
}

impl AnyInterface {
    pub fn name(&self) -> String {
        match self {
            AnyInterface::TextInferenceTokensInLogitOutInterface(_) => {
                "TextInferenceTokensInLogitsOut".to_string()
            }
            AnyInterface::MultimodalLanguageInterface(_) => "MultimodalLanguage".to_string(),
            AnyInterface::ImageGenerationInterface(_) => "ImageGeneration".to_string(),
            AnyInterface::VideoGenerationInterface(_) => "VideoGeneration".to_string(),
            AnyInterface::TextToSpeechInterface(_) => "TextToSpeech".to_string(),
            AnyInterface::SpeechToTextInterface(_) => "SpeechToText".to_string(),
        }
    }

    pub fn get_super_graph(&self) -> &SuperGraph {
        match self {
            AnyInterface::TextInferenceTokensInLogitOutInterface(x) => &x.super_graph,
            AnyInterface::MultimodalLanguageInterface(x) => &x.super_graph,
            AnyInterface::ImageGenerationInterface(x) => &x.super_graph,
            AnyInterface::VideoGenerationInterface(x) => &x.super_graph,
            AnyInterface::TextToSpeechInterface(x) => &x.super_graph,
            AnyInterface::SpeechToTextInterface(x) => &x.super_graph,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TextInferenceTokensInLogitOutInterface {
    pub cache_key_input_link: SuperGraphLink,
    pub token_context_input_link: SuperGraphLink,
    pub model_input_link: SuperGraphLink,
    pub logit_output_link: SuperGraphLink,
    pub super_graph: SuperGraph,
    pub tokenizer: TokenizerInfo,
}

impl TextInferenceTokensInLogitOutInterface {
    pub fn run_string_in_string_out<'p, P: Pool + 'p>(
        &self,
        model: &Model,
        text_in: String,
        tokenizer_cache: &mut HashMap<TokenizerInfo, Arc<AnyTokenizer>>,
        super_graph_caches: Option<&mut SuperGraphCache>,
        pool: &'p P,
    ) -> Result<String, SuperGraphError> {
        let tokenizer = {
            if let Some(x) = tokenizer_cache.get(&self.tokenizer) {
                x.clone()
            } else {
                let x = Arc::new(AnyTokenizer::from_tokenizer_info(&self.tokenizer));
                tokenizer_cache.insert(self.tokenizer.clone(), x.clone());
                x
            }
        };
        let tokens = tokenizer.encode(text_in.as_str());
        let tokens_tensor =
            NumericTensor::from_fn(vec![tokens.len() as u64], NumericDType::U32, pool, |i| {
                NumericScalar::from_u32(tokens[i])
            })
            .map_err(alloc_err)?;

        let super_graph_data = {
            let mut super_graph_data = SuperGraphData::new();
            super_graph_data
                .tensor_maps
                .insert(self.model_input_link, model.get_tensor_store());
            super_graph_data
                .tensors
                .insert(self.token_context_input_link, tokens_tensor);
            super_graph_data.hashes.insert(self.cache_key_input_link, 0);
            super_graph_data
        };
        let super_graph_output = {
            let mut observer = ();
            let mut context = SuperGraphContext {
                pool,
                observer: &mut observer,
                caches: super_graph_caches,
                symbolic_graphs: vec![model.get_symbolic_graph()],
            };
            self.super_graph.run(super_graph_data, &mut context)?
        };
        let logits = super_graph_output
            .tensors
            .get(&self.logit_output_link)
            .unwrap();
        let shape = logits.shape();
        // Select last position and argmax
        let last_row = logits
            .slice(&[(shape[0] - 1, shape[0]), (0, shape[1])])
            .map_err(|e| SuperGraphError::InvalidInputError(format!("logits slice: {e}")))?;
        let token_id = argmax_flat(&last_row);

        let token_str = tokenizer.decode(&[token_id])?;
        Ok(token_str)
    }

    pub fn get_tokenizer(&self) -> &TokenizerInfo {
        &self.tokenizer
    }

    pub fn to_any(self) -> AnyInterface {
        AnyInterface::TextInferenceTokensInLogitOutInterface(self)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum MultimodalTensorInputRole {
    Embeddings,
    AttentionMask,
    PositionIds,
    MediaTokenIds,
    MediaGrid,
    AudioFeatures,
    Other(String),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MultimodalTensorInput {
    pub name: String,
    pub role: MultimodalTensorInputRole,
    pub tensor_link: SuperGraphLink,
    pub required: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MultimodalLanguageInterface {
    pub cache_key_input_link: SuperGraphLink,
    pub token_context_input_link: SuperGraphLink,
    pub model_input_link: SuperGraphLink,
    pub modality_inputs: Vec<MultimodalTensorInput>,
    pub logit_output_link: SuperGraphLink,
    pub super_graph: SuperGraph,
    pub tokenizer: TokenizerInfo,
}

impl MultimodalLanguageInterface {
    pub fn run_string_with_modal_inputs_in_string_out<'p, P: Pool + 'p>(
        &self,
        model: &Model,
        text_in: String,
        modal_inputs: HashMap<SuperGraphLink, NumericTensor<'p, DynRank, P>>,
        tokenizer_cache: &mut HashMap<TokenizerInfo, Arc<AnyTokenizer>>,
        super_graph_caches: Option<&mut SuperGraphCache>,
        pool: &'p P,
    ) -> Result<String, SuperGraphError> {
        let tokenizer = {
            if let Some(x) = tokenizer_cache.get(&self.tokenizer) {
                x.clone()
            } else {
                let x = Arc::new(AnyTokenizer::from_tokenizer_info(&self.tokenizer));
                tokenizer_cache.insert(self.tokenizer.clone(), x.clone());
                x
            }
        };
        let tokens = tokenizer.encode(text_in.as_str());
        let tokens_tensor =
            NumericTensor::from_fn(vec![tokens.len() as u64], NumericDType::U32, pool, |i| {
                NumericScalar::from_u32(tokens[i])
            })
            .map_err(alloc_err)?;

        let super_graph_data = {
            let mut super_graph_data = SuperGraphData::new();
            super_graph_data
                .tensor_maps
                .insert(self.model_input_link, model.get_tensor_store());
            super_graph_data.tensors.extend(modal_inputs);
            super_graph_data
                .tensors
                .insert(self.token_context_input_link, tokens_tensor);
            super_graph_data.hashes.insert(self.cache_key_input_link, 0);
            super_graph_data
        };
        let super_graph_output = {
            let mut observer = ();
            let mut context = SuperGraphContext {
                pool,
                observer: &mut observer,
                caches: super_graph_caches,
                symbolic_graphs: vec![model.get_symbolic_graph()],
            };
            self.super_graph.run(super_graph_data, &mut context)?
        };
        let logits = super_graph_output
            .tensors
            .get(&self.logit_output_link)
            .unwrap();
        let shape = logits.shape();
        // Select last position and argmax
        let last_row = logits
            .slice(&[(shape[0] - 1, shape[0]), (0, shape[1])])
            .map_err(|e| SuperGraphError::InvalidInputError(format!("logits slice: {e}")))?;
        let token_id = argmax_flat(&last_row);

        let token_str = tokenizer.decode(&[token_id])?;
        Ok(token_str)
    }

    pub fn run_string_in_string_out<'p, P: Pool + 'p>(
        &self,
        model: &Model,
        text_in: String,
        tokenizer_cache: &mut HashMap<TokenizerInfo, Arc<AnyTokenizer>>,
        super_graph_caches: Option<&mut SuperGraphCache>,
        pool: &'p P,
    ) -> Result<String, SuperGraphError> {
        self.run_string_with_modal_inputs_in_string_out(
            model,
            text_in,
            HashMap::new(),
            tokenizer_cache,
            super_graph_caches,
            pool,
        )
    }

    pub fn get_tokenizer(&self) -> &TokenizerInfo {
        &self.tokenizer
    }

    pub fn to_any(self) -> AnyInterface {
        AnyInterface::MultimodalLanguageInterface(self)
    }
}

// ============================================================================
// Image Generation Interface
// ============================================================================

/// Scheduler type for the denoising loop.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SchedulerType {
    /// Euler discrete scheduler (SD 1.5, SD 2, SDXL).
    /// Initial noise is scaled by init_sigma.
    EulerDiscrete,
    /// Rectified flow scheduler (Flux).
    /// No noise scaling.
    RectifiedFlow,
    /// DDIM with v-prediction (CogVideoX).
    DDIMVPrediction,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImageGenerationInterface {
    pub super_graph: SuperGraph,
    // Prompt inputs
    pub positive_prompt_input: SuperGraphLink,
    pub negative_prompt_input: Option<SuperGraphLink>,
    // Latent / scheduler inputs
    pub initial_latent_input: SuperGraphLink,
    pub timesteps_input: SuperGraphLink,
    pub dt_input: SuperGraphLink,
    pub sigmas_input: SuperGraphLink,
    pub iteration_count_input: SuperGraphLink,
    pub guidance_scale_input: Option<SuperGraphLink>,
    // Model weight maps (in order matching loader's model_ids)
    pub model_weights: Vec<SuperGraphLink>,
    // Output
    pub image_output: SuperGraphLink,
    /// Scheduler type for the denoising loop.
    pub scheduler: SchedulerType,
    /// Number of latent channels (4 for SD/SDXL, 16 for Flux).
    pub latent_channels: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VideoGenerationInterface {
    pub super_graph: SuperGraph,
    // Prompt inputs
    pub positive_prompt_input: SuperGraphLink,
    pub negative_prompt_input: Option<SuperGraphLink>,
    // Latent / scheduler inputs
    pub initial_latent_input: SuperGraphLink,
    /// Per-step timestep values, scanned along axis 0.
    pub timesteps_input: SuperGraphLink,
    /// Per-step scheduler data, scanned along axis 0.
    /// Interpretation depends on scheduler type:
    /// - RectifiedFlow: dt values (scalar per step)
    /// - DDIMVPrediction: `[alpha_{t-1}, sigma_{t-1}]` pairs per step
    pub dt_input: SuperGraphLink,
    /// Per-step scheduler data, scanned along axis 0.
    /// Interpretation depends on scheduler type:
    /// - RectifiedFlow: unused (still required as graph input)
    /// - DDIMVPrediction: `[alpha_t, sigma_t]` pairs per step
    pub sigmas_input: SuperGraphLink,
    pub iteration_count_input: SuperGraphLink,
    pub guidance_scale_input: Option<SuperGraphLink>,
    // Model weight maps (in order matching loader's model_ids)
    pub model_weights: Vec<SuperGraphLink>,
    // Output
    pub video_output: SuperGraphLink,
    /// Scheduler type for the denoising loop.
    pub scheduler: SchedulerType,
    /// Number of latent channels.
    pub latent_channels: usize,
    /// Frames per second for the output video.
    pub fps: f32,
    /// Number of output frames.
    pub num_frames: usize,
}

impl VideoGenerationInterface {
    pub fn to_any(self) -> AnyInterface {
        AnyInterface::VideoGenerationInterface(self)
    }
}

impl ImageGenerationInterface {
    /// Pre-compute the Euler discrete scheduler parameters.
    /// Returns (timestep_values, dt_values, sigma_values, initial_sigma).
    pub fn compute_euler_schedule(
        num_inference_steps: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, f32) {
        let num_train_timesteps = 1000;
        let beta_start: f32 = 0.00085;
        let beta_end: f32 = 0.012;

        // Linear beta schedule (scaled_linear: sqrt of linear interpolation)
        let betas: Vec<f32> = (0..num_train_timesteps)
            .map(|i| {
                let t = i as f32 / (num_train_timesteps - 1) as f32;
                let b = beta_start.sqrt() + t * (beta_end.sqrt() - beta_start.sqrt());
                b * b
            })
            .collect();

        let mut alphas_cumprod = Vec::with_capacity(num_train_timesteps);
        let mut cumprod = 1.0f32;
        for &beta in &betas {
            cumprod *= 1.0 - beta;
            alphas_cumprod.push(cumprod);
        }

        // Evenly spaced timesteps (descending)
        let step_ratio = num_train_timesteps / num_inference_steps;
        let timesteps: Vec<usize> = (0..num_inference_steps)
            .rev()
            .map(|i| i * step_ratio + step_ratio - 1)
            .collect();

        // Compute sigmas from alphas_cumprod
        let sigmas: Vec<f32> = timesteps
            .iter()
            .map(|&t| ((1.0 - alphas_cumprod[t]) / alphas_cumprod[t]).sqrt())
            .collect();

        let init_sigma = sigmas[0];

        // Timestep values as f32
        let timestep_values: Vec<f32> = timesteps.iter().map(|&t| t as f32).collect();

        // dt[i] = sigma[i+1] - sigma[i], with sigma[num_steps] = 0
        let dt_values: Vec<f32> = (0..num_inference_steps)
            .map(|i| {
                let sigma_next = if i + 1 < num_inference_steps {
                    sigmas[i + 1]
                } else {
                    0.0
                };
                sigma_next - sigmas[i]
            })
            .collect();

        (timestep_values, dt_values, sigmas, init_sigma)
    }

    /// Pre-compute the rectified flow scheduler parameters for Flux Schnell.
    ///
    /// Flux uses flow matching: timesteps go from 1.0 (noise) to 0.0 (clean).
    /// Returns (timestep_values, dt_values, sigma_values).
    /// For Flux, sigma values equal timestep values (used for interface compatibility).
    pub fn compute_flux_schedule(num_inference_steps: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        // Linearly spaced timesteps from 1.0 to near 0.0
        // sigmas[i] = 1.0 - i/(N), giving [1.0, 1-1/N, ..., 1/N]
        // Final sigma (after last step) is 0.0
        let sigmas: Vec<f32> = (0..num_inference_steps)
            .map(|i| 1.0 - i as f32 / num_inference_steps as f32)
            .collect();

        let timestep_values = sigmas.clone();

        // dt[i] = sigma[i+1] - sigma[i], with sigma[num_steps] = 0
        let dt_values: Vec<f32> = (0..num_inference_steps)
            .map(|i| {
                let sigma_next = if i + 1 < num_inference_steps {
                    sigmas[i + 1]
                } else {
                    0.0
                };
                sigma_next - sigmas[i]
            })
            .collect();

        (timestep_values, dt_values, sigmas)
    }

    /// Run the full image generation pipeline.
    #[allow(clippy::too_many_arguments)]
    pub fn run<'p, P: Pool + 'p>(
        &self,
        models: &[&Model],
        positive_prompt: String,
        negative_prompt: Option<String>,
        initial_noise: Vec<f32>,
        latent_shape: Vec<usize>,
        num_inference_steps: usize,
        guidance_scale: f32,
        pool: &'p P,
    ) -> Result<SuperGraphImage<'p, P>, SuperGraphError> {
        assert_eq!(
            models.len(),
            self.model_weights.len(),
            "Expected {} models, got {}",
            self.model_weights.len(),
            models.len()
        );

        // Compute schedule and prepare latent based on scheduler type
        let (timestep_values, dt_values, sigma_values, latent_data) = match &self.scheduler {
            SchedulerType::EulerDiscrete => {
                let (ts, dt, sigmas, init_sigma) =
                    Self::compute_euler_schedule(num_inference_steps);
                let scaled: Vec<f32> = initial_noise.iter().map(|&x| x * init_sigma).collect();
                (ts, dt, sigmas, scaled)
            }
            SchedulerType::RectifiedFlow => {
                let (ts, dt, sigmas) = Self::compute_flux_schedule(num_inference_steps);
                (ts, dt, sigmas, initial_noise)
            }
            _ => unreachable!("ImageGenerationInterface only uses EulerDiscrete or RectifiedFlow"),
        };

        let f32_tensor = |data: &[f32],
                          shape: Vec<u64>|
         -> Result<NumericTensor<'p, DynRank, P>, SuperGraphError> {
            NumericTensor::from_fn(shape, NumericDType::F32, pool, |i| {
                NumericScalar::from_f32(data[i])
            })
            .map_err(alloc_err)
        };

        let latent_shape_u64: Vec<u64> = latent_shape.iter().map(|&d| d as u64).collect();
        let n = num_inference_steps as u64;

        // Pack data
        let mut data = SuperGraphData::new();
        data.strings
            .insert(self.positive_prompt_input, positive_prompt);
        if let Some(negative_link) = self.negative_prompt_input {
            data.strings
                .insert(negative_link, negative_prompt.unwrap_or_default());
        }
        data.tensors.insert(
            self.initial_latent_input,
            f32_tensor(&latent_data, latent_shape_u64)?,
        );
        data.tensors
            .insert(self.timesteps_input, f32_tensor(&timestep_values, vec![n])?);
        data.tensors
            .insert(self.dt_input, f32_tensor(&dt_values, vec![n])?);
        data.tensors
            .insert(self.sigmas_input, f32_tensor(&sigma_values, vec![n])?);
        data.tensors.insert(
            self.iteration_count_input,
            NumericTensor::from_fn(vec![1], NumericDType::I64, pool, |_| {
                NumericScalar::from_i64(num_inference_steps as i64)
            })
            .map_err(alloc_err)?,
        );
        if let Some(gs_link) = self.guidance_scale_input {
            data.tensors.insert(
                gs_link,
                NumericTensor::from_fn(vec![], NumericDType::F32, pool, |_| {
                    NumericScalar::from_f32(guidance_scale)
                })
                .map_err(alloc_err)?,
            );
        }
        for (weight_link, model) in self.model_weights.iter().zip(models.iter()) {
            data.tensor_maps
                .insert(*weight_link, model.get_tensor_store());
        }

        // Run
        let mut observer = ();
        let symbolic_graphs: Vec<_> = models.iter().map(|m| m.get_symbolic_graph()).collect();
        let mut context = SuperGraphContext {
            pool,
            observer: &mut observer,
            caches: None,
            symbolic_graphs,
        };

        let mut result = self.super_graph.run(data, &mut context)?;
        let image = result
            .images
            .remove(&self.image_output)
            .expect("image output missing");
        Ok(image)
    }

    pub fn to_any(self) -> AnyInterface {
        AnyInterface::ImageGenerationInterface(self)
    }
}

// ============================================================================
// Text-to-Speech Interface
// ============================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KokoroVoiceEmbedding {
    pub name: String,
    /// Raw little-endian f32 data laid out as [num_rows, 256].
    pub style_table_le_bytes: Vec<u8>,
}

impl KokoroVoiceEmbedding {
    pub const STYLE_DIM: usize = 256;

    pub fn style_for_token_count(&self, num_tokens: usize) -> Result<Vec<f32>, String> {
        let bytes_per_row = Self::STYLE_DIM * 4;
        if self.style_table_le_bytes.len() < bytes_per_row {
            return Err(format!(
                "voice '{}' table is too small: {} bytes",
                self.name,
                self.style_table_le_bytes.len()
            ));
        }
        if !self
            .style_table_le_bytes
            .len()
            .is_multiple_of(bytes_per_row)
        {
            return Err(format!(
                "voice '{}' table size {} is not divisible by row size {}",
                self.name,
                self.style_table_le_bytes.len(),
                bytes_per_row
            ));
        }

        let num_rows = self.style_table_le_bytes.len() / bytes_per_row;
        let row_idx = num_tokens.min(num_rows.saturating_sub(1));
        let start = row_idx * bytes_per_row;
        let end = start + bytes_per_row;
        let row_bytes = &self.style_table_le_bytes[start..end];

        let mut style = Vec::with_capacity(Self::STYLE_DIM);
        for chunk in row_bytes.chunks_exact(4) {
            style.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Ok(style)
    }
}

#[cfg(test)]
mod kokoro_voice_embedding_tests {
    use super::KokoroVoiceEmbedding;

    fn row_bytes(v: f32) -> Vec<u8> {
        let mut out = Vec::new();
        for _ in 0..KokoroVoiceEmbedding::STYLE_DIM {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    #[test]
    fn style_for_token_count_selects_and_clamps_rows() {
        let mut bytes = row_bytes(0.25);
        bytes.extend_from_slice(&row_bytes(0.75));
        let voice = KokoroVoiceEmbedding {
            name: "test".to_string(),
            style_table_le_bytes: bytes,
        };

        let first = voice.style_for_token_count(0).unwrap();
        assert_eq!(first.len(), KokoroVoiceEmbedding::STYLE_DIM);
        assert!((first[0] - 0.25).abs() < 1e-6);

        let clamped = voice.style_for_token_count(1234).unwrap();
        assert_eq!(clamped.len(), KokoroVoiceEmbedding::STYLE_DIM);
        assert!((clamped[0] - 0.75).abs() < 1e-6);
    }

    #[test]
    fn style_for_token_count_rejects_invalid_size() {
        let voice = KokoroVoiceEmbedding {
            name: "bad".to_string(),
            style_table_le_bytes: vec![0, 1, 2],
        };
        assert!(voice.style_for_token_count(0).is_err());
    }
}

/// Model-specific input configuration for TTS.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum TTSInputConfig {
    /// Kokoro-style conditioning inputs.
    Kokoro {
        style_link: SuperGraphLink,
        speed_link: SuperGraphLink,
        voices: Vec<KokoroVoiceEmbedding>,
        default_voice: Option<String>,
    },
    /// Piper VITS conditioning inputs.
    Piper {
        scales_link: SuperGraphLink,
        speaker_id_link: Option<SuperGraphLink>,
        num_speakers: u32,
    },
    /// F5-TTS conditioning inputs.
    /// 3-model pipeline with ODE loop baked into SuperGraph.
    F5 {
        ref_audio_link: SuperGraphLink,
        max_duration_link: SuperGraphLink,
        /// Scan inputs for the ODE denoising loop.
        time_steps_link: SuperGraphLink,
        iteration_count_link: SuperGraphLink,
        /// Number of function evaluations (default 32).
        nfe_steps: u32,
    },
}

/// Unified interface for all text-to-speech models.
///
/// Each model family builds a different SuperGraph that hides its internal
/// complexity (single model, multi-model pipelines, ODE loops, etc.).
/// The caller provides raw text plus model-specific conditioning inputs
/// described by `input_config`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TextToSpeechInterface {
    pub super_graph: SuperGraph,
    /// Raw text input.
    pub text_input_link: SuperGraphLink,
    /// Model weights (one per model in the pipeline).
    pub model_weights: Vec<SuperGraphLink>,
    /// Output audio clip.
    pub audio_output_link: SuperGraphLink,
    /// Sample rate of the output audio in Hz.
    pub sample_rate: u32,
    /// Model-specific non-text conditioning inputs.
    pub input_config: TTSInputConfig,
}

impl TextToSpeechInterface {
    pub fn to_any(self) -> AnyInterface {
        AnyInterface::TextToSpeechInterface(self)
    }
}

/// Interface for speech-to-text models (e.g. Whisper).
///
/// Single-supergraph architecture:
/// 1. Audio features → encoder hidden states
/// 2. Fixed-step autoregressive decoder loop
/// 3. Output token sequence (caller can trim at EOS and decode)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SpeechToTextInterface {
    /// Unified STT supergraph.
    pub super_graph: SuperGraph,

    /// Input audio clip link.
    /// The clip payload carries raw mono waveform samples.
    pub audio_input_link: SuperGraphLink,

    /// Encoder model weights.
    pub encoder_weights_link: SuperGraphLink,
    /// Decoder model weights.
    pub decoder_weights_link: SuperGraphLink,

    /// Output token sequence: [forced_prefix..., generated...].
    pub output_token_link: SuperGraphLink,

    /// Tokenizer for decoding output tokens to text.
    pub tokenizer: TokenizerInfo,
    /// Audio sample rate expected by the model (e.g. 16000).
    pub sample_rate: u32,
    /// Number of mel bins.
    pub num_mel_bins: u32,
    /// Decoder start token ID.
    pub decoder_start_token_id: u32,
    /// End-of-text token ID.
    pub eos_token_id: u32,

    /// Prefix token IDs prepended before generation (start + forced tokens).
    pub decoder_prefix_token_ids: Vec<u32>,

    /// Number of fixed decode iterations executed in the supergraph.
    pub max_decode_steps: u32,
}

impl SpeechToTextInterface {
    pub fn to_any(self) -> AnyInterface {
        AnyInterface::SpeechToTextInterface(self)
    }
}
