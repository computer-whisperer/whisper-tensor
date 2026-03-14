use clap::Parser;
use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use whisper_tensor::backends::ModelLoadedTensorCache;
use whisper_tensor::backends::eval_backend::EvalBackend;
use whisper_tensor::interfaces::AnyInterface;
use whisper_tensor::loader::{ConfigValue, ConfigValues, Loader, LoaderOutput};
use whisper_tensor::numeric_tensor::NumericTensor;
use whisper_tensor::super_graph::cache::SuperGraphCache;
use whisper_tensor::tensor_rank::DynRank;
use whisper_tensor::tokenizer::Tokenizer;

#[derive(Parser)]
#[command(name = "wt", about = "Whisper Tensor CLI")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(clap::Subcommand)]
enum Command {
    /// Generate text from a language model
    Generate {
        /// Path to the model (directory or file)
        model: Option<PathBuf>,

        /// Loader to use
        #[arg(long, value_enum, default_value = "auto")]
        loader: LoaderChoice,

        /// Extra loader config (key=value, repeatable)
        #[arg(long = "config", value_name = "KEY=VALUE")]
        configs: Vec<String>,

        /// Prompt text (reads from stdin if not provided)
        #[arg(long, short)]
        prompt: Option<String>,

        /// Maximum number of tokens to generate
        #[arg(long, short = 'n', default_value = "100")]
        max_tokens: usize,
    },

    /// Generate an image from a diffusion model
    Image {
        /// Path to the model file (shorthand for --config path=<...>)
        model: Option<PathBuf>,

        /// Loader to use
        #[arg(long, value_enum, default_value = "auto")]
        loader: LoaderChoice,

        /// Extra loader config (key=value, repeatable)
        #[arg(long = "config", value_name = "KEY=VALUE")]
        configs: Vec<String>,

        /// Prompt text
        #[arg(long, short)]
        prompt: String,

        /// Negative prompt
        #[arg(long, default_value = "")]
        negative_prompt: String,

        /// Output image path
        #[arg(long, short, default_value = "output.png")]
        output: PathBuf,

        /// Number of inference steps
        #[arg(long, default_value = "20")]
        steps: usize,

        /// Classifier-free guidance scale
        #[arg(long, default_value = "7.5")]
        guidance_scale: f32,

        /// Latent height (image height = latent_h * 8)
        #[arg(long, default_value = "64")]
        latent_h: usize,

        /// Latent width (image width = latent_w * 8)
        #[arg(long, default_value = "64")]
        latent_w: usize,

        /// Random seed
        #[arg(long, default_value = "42")]
        seed: u64,
    },

    /// Synthesize speech from text
    Tts {
        /// Path to the model directory
        model: Option<PathBuf>,

        /// Loader to use
        #[arg(long, value_enum, default_value = "auto")]
        loader: LoaderChoice,

        /// Extra loader config (key=value, repeatable)
        #[arg(long = "config", value_name = "KEY=VALUE")]
        configs: Vec<String>,

        /// Text to speak
        #[arg(long, short)]
        prompt: String,

        /// Voice name (e.g. "af_bella"). Looked up in <model>/voices/<name>.bin
        #[arg(long, default_value = "af")]
        voice: String,

        /// Speech speed multiplier
        #[arg(long, default_value = "1.0")]
        speed: f32,

        /// Output WAV path
        #[arg(long, short, default_value = "output.wav")]
        output: PathBuf,

        /// Reference audio file for F5-TTS (WAV, 24kHz)
        #[arg(long)]
        ref_audio: Option<PathBuf>,

        /// Transcript of the reference audio (for F5-TTS duration estimation)
        #[arg(long)]
        ref_text: Option<String>,
    },

    /// Transcribe speech to text
    Stt {
        /// Path to the model directory
        model: Option<PathBuf>,

        /// Loader to use
        #[arg(long, value_enum, default_value = "whisper")]
        loader: LoaderChoice,

        /// Extra loader config (key=value, repeatable)
        #[arg(long = "config", value_name = "KEY=VALUE")]
        configs: Vec<String>,

        /// Input audio file (WAV)
        #[arg(long, short)]
        audio: PathBuf,
    },
}

#[derive(Clone, clap::ValueEnum)]
enum LoaderChoice {
    Auto,
    Transformers,
    Onnx,
    Rwkv7,
    Gguf,
    Sd15,
    Sd2,
    Sdxl,
    Flux,
    Kokoro,
    Piper,
    F5Tts,
    Whisper,
}

// ============================================================================
// Shared loading
// ============================================================================

/// Build a ConfigValues map from the positional model path and --config args.
fn build_config(model: Option<PathBuf>, configs: &[String]) -> ConfigValues {
    let mut config = ConfigValues::new();
    if let Some(path) = model {
        config.insert("path".to_string(), ConfigValue::FilePath(path));
    }
    for entry in configs {
        if let Some((key, value)) = entry.split_once('=') {
            config.insert(key.to_string(), parse_config_value(value));
        } else {
            eprintln!("Warning: ignoring malformed --config '{entry}' (expected key=value)");
        }
    }
    config
}

/// Infer ConfigValue type from a string.
fn parse_config_value(value: &str) -> ConfigValue {
    let path = PathBuf::from(value);
    if path.exists() || value.contains('/') || value.contains('\\') {
        ConfigValue::FilePath(path)
    } else if value.eq_ignore_ascii_case("true") {
        ConfigValue::Bool(true)
    } else if value.eq_ignore_ascii_case("false") {
        ConfigValue::Bool(false)
    } else if let Ok(i) = value.parse::<i64>() {
        ConfigValue::Integer(i)
    } else if let Ok(f) = value.parse::<f64>() {
        ConfigValue::Float(f)
    } else {
        ConfigValue::String(value.to_string())
    }
}

/// Dispatch to the appropriate loader.
fn load_model(loader: &LoaderChoice, config: ConfigValues) -> LoaderOutput {
    use whisper_tensor_import::loaders::*;

    let result = match loader {
        LoaderChoice::Auto => AutoLoader.load(config),
        LoaderChoice::Transformers => TransformersLoader.load(config),
        LoaderChoice::Onnx => OnnxLoader.load(config),
        LoaderChoice::Rwkv7 => Rwkv7Loader.load(config),
        LoaderChoice::Gguf => GgufLoader.load(config),
        LoaderChoice::Sd15 => SD15Loader.load(config),
        LoaderChoice::Sd2 => SD2Loader.load(config),
        LoaderChoice::Sdxl => SDXLLoader.load(config),
        LoaderChoice::Flux => FluxLoader.load(config),
        LoaderChoice::Kokoro => KokoroLoader.load(config),
        LoaderChoice::Piper => PiperLoader.load(config),
        LoaderChoice::F5Tts => F5TtsLoader.load(config),
        LoaderChoice::Whisper => WhisperLoader.load(config),
    };

    result.unwrap_or_else(|e| {
        eprintln!("Failed to load model: {e}");
        std::process::exit(1);
    })
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    match cli.command {
        Command::Generate {
            model,
            loader,
            configs,
            prompt,
            max_tokens,
        } => {
            let config = build_config(model, &configs);
            eprintln!("Loading model...");
            let output = load_model(&loader, config);
            cmd_generate(output, prompt, max_tokens);
        }
        Command::Image {
            model,
            loader,
            configs,
            prompt,
            negative_prompt,
            output,
            steps,
            guidance_scale,
            latent_h,
            latent_w,
            seed,
        } => {
            let config = build_config(model, &configs);
            eprintln!("Loading model...");
            let loaded = load_model(&loader, config);
            cmd_image(
                loaded,
                prompt,
                negative_prompt,
                output,
                steps,
                guidance_scale,
                latent_h,
                latent_w,
                seed,
            );
        }
        Command::Tts {
            model,
            loader,
            configs,
            prompt,
            voice,
            speed,
            output,
            ref_audio,
            ref_text,
        } => {
            let model_dir = model.clone().unwrap_or_else(|| PathBuf::from("."));
            let config = build_config(model, &configs);
            eprintln!("Loading model...");
            let loaded = load_model(&loader, config);
            cmd_tts(
                loaded, model_dir, prompt, voice, speed, output, ref_audio, ref_text,
            );
        }
        Command::Stt {
            model,
            loader,
            configs,
            audio,
        } => {
            let config = build_config(model.clone(), &configs);
            eprintln!("Loading model...");
            let loaded = load_model(&loader, config);
            cmd_stt(loaded, audio, model);
        }
    }
}

// ============================================================================
// Text generation
// ============================================================================

fn cmd_generate(output: LoaderOutput, prompt: Option<String>, max_tokens: usize) {
    let prompt = match prompt {
        Some(p) => p,
        None => {
            eprint!("Prompt: ");
            std::io::stderr().flush().unwrap();
            let mut line = String::new();
            std::io::stdin().read_line(&mut line).unwrap();
            line.trim_end_matches('\n').to_string()
        }
    };

    let model = &output.models[0].model;

    let interface = output
        .interfaces
        .iter()
        .find_map(|i| match &i.interface {
            AnyInterface::TextInferenceTokensInLogitOutInterface(x) => Some(x),
            _ => None,
        })
        .unwrap_or_else(|| {
            eprintln!("Model has no text inference interface");
            std::process::exit(1);
        });

    eprintln!("Model loaded. Generating...\n");

    let mut backend = EvalBackend::NDArray;
    let mut tokenizer_cache = HashMap::new();
    let mut super_graph_caches = SuperGraphCache::new();
    let mut model_cache = ModelLoadedTensorCache::new();

    print!("{prompt}");
    std::io::stdout().flush().unwrap();

    let mut context = prompt;
    for _ in 0..max_tokens {
        let token = interface
            .run_string_in_string_out(
                model,
                None,
                context.clone(),
                &mut tokenizer_cache,
                Some(&mut model_cache),
                Some(&mut super_graph_caches),
                &mut backend,
            )
            .unwrap_or_else(|e| {
                eprintln!("\nInference error: {e}");
                std::process::exit(1);
            });
        print!("{token}");
        std::io::stdout().flush().unwrap();
        context.push_str(&token);
    }
    println!();
}

// ============================================================================
// Image generation
// ============================================================================

#[allow(clippy::too_many_arguments)]
fn cmd_image(
    output: LoaderOutput,
    prompt: String,
    negative_prompt: String,
    output_path: PathBuf,
    steps: usize,
    guidance_scale: f32,
    latent_h: usize,
    latent_w: usize,
    seed: u64,
) {
    let models: Vec<&_> = output.models.iter().map(|m| m.model.as_ref()).collect();

    let interface = output
        .interfaces
        .iter()
        .find_map(|i| match &i.interface {
            AnyInterface::ImageGenerationInterface(x) => Some(x),
            _ => None,
        })
        .unwrap_or_else(|| {
            eprintln!("No ImageGeneration interface found");
            std::process::exit(1);
        });

    // Generate initial noise
    let channels = interface.latent_channels;
    let latent_n = channels * latent_h * latent_w;
    let initial_noise = generate_gaussian_noise(latent_n, seed);

    eprintln!(
        "Generating {}x{} image ({steps} steps, guidance={guidance_scale}, seed={seed})...",
        latent_w * 8,
        latent_h * 8,
    );

    let mut backend = EvalBackend::NDArray;
    let start = std::time::Instant::now();

    let image_tensor = interface
        .run(
            &models,
            prompt,
            Some(negative_prompt),
            initial_noise,
            vec![1, channels, latent_h, latent_w],
            steps,
            guidance_scale,
            &mut backend,
        )
        .map(|x| x.tensor)
        .unwrap_or_else(|e| {
            eprintln!("Inference error: {e}");
            std::process::exit(1);
        });

    eprintln!("Generated in {:.2?}", start.elapsed());

    save_image_tensor(&image_tensor, &output_path, &mut backend);
    eprintln!("Saved to {}", output_path.display());
}

// ============================================================================
// Text-to-speech
// ============================================================================

fn cmd_tts(
    output: LoaderOutput,
    model_dir: PathBuf,
    text: String,
    voice_name: String,
    speed: f32,
    output_path: PathBuf,
    ref_audio: Option<PathBuf>,
    ref_text: Option<String>,
) {
    use whisper_tensor::interfaces::TTSInputConfig;
    use whisper_tensor::super_graph::SuperGraphContext;
    use whisper_tensor::super_graph::cache::SuperGraphTensorCache;
    use whisper_tensor::super_graph::data::SuperGraphData;

    let interface = output
        .interfaces
        .iter()
        .find_map(|i| match &i.interface {
            AnyInterface::TextToSpeechInterface(x) => Some(x),
            _ => None,
        })
        .unwrap_or_else(|| {
            eprintln!("No TTS interface found in loaded model");
            std::process::exit(1);
        });

    // Build SuperGraph inputs based on model type.
    let mut data = SuperGraphData::new();
    for (i, model_weights_link) in interface.model_weights.iter().enumerate() {
        data.tensor_maps.insert(
            *model_weights_link,
            output.models[i].model.get_tensor_store(),
        );
    }
    data.strings.insert(interface.text_input_link, text.clone());

    match &interface.input_config {
        TTSInputConfig::Kokoro {
            style_link,
            speed_link,
            voices,
            default_voice,
        } => {
            // Kokoro voice style bins are indexed by token count; use a simple
            // text-length approximation here and let the graph handle tokenization.
            let approx_tokens = text.chars().count().saturating_add(2);
            let style_tensor = if !voices.is_empty() {
                let selected_voice = if voices.iter().any(|v| v.name == voice_name) {
                    voice_name.clone()
                } else if let Some(default_voice) = default_voice
                    && voices.iter().any(|v| v.name == *default_voice)
                {
                    eprintln!(
                        "Voice '{}' not found, using default '{}'",
                        voice_name, default_voice
                    );
                    default_voice.clone()
                } else {
                    let fallback = voices.first().unwrap().name.clone();
                    eprintln!("Voice '{}' not found, using '{}'", voice_name, fallback);
                    fallback
                };
                let voice = voices.iter().find(|v| v.name == selected_voice).unwrap();
                let style_values = voice
                    .style_for_token_count(approx_tokens)
                    .unwrap_or_else(|e| {
                        eprintln!("Failed to decode embedded voice '{}': {}", voice.name, e);
                        std::process::exit(1);
                    });
                NumericTensor::<DynRank>::from_vec_shape(style_values, vec![1, 256]).unwrap()
            } else {
                let voice_bin = model_dir.join("voices").join(format!("{voice_name}.bin"));
                if !voice_bin.exists() {
                    eprintln!("No voice file found: {}", voice_bin.display());
                    list_available_voices(&model_dir);
                    std::process::exit(1);
                }
                load_voice_style_bin(&voice_bin, approx_tokens)
            };

            let speed_tensor =
                NumericTensor::<DynRank>::from_vec_shape(vec![speed], vec![1]).unwrap();

            data.tensors.insert(*style_link, style_tensor);
            data.tensors.insert(*speed_link, speed_tensor);
        }
        TTSInputConfig::Piper {
            scales_link,
            speaker_id_link,
            ..
        } => {
            let length_scale = 1.0 / speed;
            let scales_tensor = NumericTensor::<DynRank>::from_vec_shape(
                vec![0.667f32, length_scale, 0.8],
                vec![3],
            )
            .unwrap();

            data.tensors.insert(*scales_link, scales_tensor);
            if let Some(sid_link) = speaker_id_link {
                data.tensors.insert(
                    *sid_link,
                    NumericTensor::<DynRank>::from_vec_shape(vec![0i64], vec![1]).unwrap(),
                );
            }
        }
        TTSInputConfig::F5 {
            ref_audio_link,
            max_duration_link,
            time_steps_link,
            iteration_count_link,
            nfe_steps,
        } => {
            // Load reference audio
            let ref_audio_path = ref_audio.unwrap_or_else(|| {
                eprintln!("F5-TTS requires --ref-audio <path.wav>");
                std::process::exit(1);
            });
            let ref_samples = load_wav_f16(&ref_audio_path, 24000);
            let ref_audio_len = ref_samples.len();
            eprintln!(
                "Reference audio: {} samples ({:.2}s)",
                ref_audio_len,
                ref_audio_len as f64 / 24000.0
            );

            let ref_text_str = ref_text.as_deref().unwrap_or("");
            let gen_text = &text;
            let combined_text = if ref_text_str.is_empty() {
                gen_text.clone()
            } else {
                format!("{ref_text_str} {gen_text}")
            };
            data.strings
                .insert(interface.text_input_link, combined_text.clone());

            // Compute max_duration
            let hop_length: usize = 256;
            let ref_audio_frames = ref_audio_len / hop_length + 1;
            let gen_text_len = gen_text.chars().count().max(1);
            let gen_duration_estimate = if !ref_text_str.is_empty() {
                let ref_text_len = ref_text_str.chars().count().max(1);
                (ref_audio_frames as f64 / ref_text_len as f64 * gen_text_len as f64 / speed as f64)
                    as usize
            } else {
                // No ref text: estimate ~4 chars/sec at 24kHz/256hop = ~94 frames/sec
                (gen_text_len as f64 * 94.0 / 4.0 / speed as f64) as usize
            };
            let max_duration = (ref_audio_frames + gen_duration_estimate).min(2048);
            eprintln!(
                "Max duration: {max_duration} frames ({:.2}s)",
                max_duration as f64 * hop_length as f64 / 24000.0
            );

            // Build input tensors
            let ref_audio_tensor =
                NumericTensor::<DynRank>::from_vec_shape(ref_samples, vec![1, 1, ref_audio_len])
                    .unwrap();

            let max_duration_tensor =
                NumericTensor::<DynRank>::from_vec_shape(vec![max_duration as i64], vec![])
                    .unwrap();

            // Build ODE loop inputs
            let nfe = *nfe_steps;
            let iterations = nfe - 1; // 31 iterations for nfe=32
            let time_steps: Vec<i32> = (0..iterations as i32).collect();
            let time_steps_tensor =
                NumericTensor::<DynRank>::from_vec_shape(time_steps, vec![iterations as usize])
                    .unwrap();
            let iteration_count_tensor =
                NumericTensor::<DynRank>::from_vec_shape(vec![iterations as i64], vec![]).unwrap();

            data.tensors.insert(*ref_audio_link, ref_audio_tensor);
            data.tensors.insert(*max_duration_link, max_duration_tensor);
            data.tensors.insert(*time_steps_link, time_steps_tensor);
            data.tensors
                .insert(*iteration_count_link, iteration_count_tensor);
        }
    }

    // Run inference
    eprintln!("Running inference...");
    let mut backend = EvalBackend::NDArray;
    let start = std::time::Instant::now();

    let symbolic_graphs: Vec<_> = output
        .models
        .iter()
        .map(|m| m.model.get_symbolic_graph())
        .collect();
    let super_graph_output = {
        let mut observer = ();
        let mut super_graph_tensor_cache = SuperGraphTensorCache::new();
        let mut context = SuperGraphContext {
            observer: &mut observer,
            eval_backend: &mut backend,
            super_graph_tensor_cache: &mut super_graph_tensor_cache,
            caches: None,
            symbolic_graphs,
            use_compiled_models: false,
            compiled_models: None,
        };
        interface
            .super_graph
            .run(data, &mut context)
            .unwrap_or_else(|e| {
                eprintln!("Inference error: {e}");
                std::process::exit(1);
            })
    };

    let audio = super_graph_output
        .audio_clips
        .get(&interface.audio_output_link)
        .expect("No audio output clip");

    eprintln!("Generated in {:.2?}", start.elapsed());

    let samples = audio_tensor_to_samples(&audio.samples, &mut backend);
    save_wav(&samples, audio.sample_rate_hz, &output_path);
    eprintln!(
        "Saved {:.1}s of audio to {}",
        samples.len() as f64 / audio.sample_rate_hz as f64,
        output_path.display()
    );
}

/// Extract f32 samples from an audio output tensor.
fn audio_tensor_to_samples(audio: &NumericTensor<DynRank>, backend: &mut EvalBackend) -> Vec<f32> {
    let audio_f32 = audio
        .cast(whisper_tensor::dtype::DType::F32, backend)
        .expect("cast to f32 failed");
    let audio_ndarray = audio_f32.to_ndarray().expect("to_ndarray failed");
    audio_ndarray.flatten().try_into().expect("flatten failed")
}

/// Load a voice style vector from a .bin file (Kokoro format), indexed by token count.
fn load_voice_style_bin(path: &std::path::Path, num_tokens: usize) -> NumericTensor<DynRank> {
    let data = std::fs::read(path).expect("Failed to read voice file");
    let floats: Vec<f32> = data
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();

    let num_vectors = floats.len() / 256;
    let idx = num_tokens.min(num_vectors.saturating_sub(1));
    let start = idx * 256;
    let style: Vec<f32> = floats[start..start + 256].to_vec();

    NumericTensor::<DynRank>::from_vec_shape(style, vec![1, 256]).unwrap()
}

/// List available voices in a model directory.
fn list_available_voices(model_dir: &std::path::Path) {
    let voices_dir = model_dir.join("voices");
    if voices_dir.is_dir() {
        let mut available: Vec<String> = std::fs::read_dir(&voices_dir)
            .unwrap()
            .filter_map(|e| {
                let name = e.ok()?.file_name().to_string_lossy().to_string();
                name.strip_suffix(".bin").map(|s| s.to_string())
            })
            .collect();
        available.sort();
        if !available.is_empty() {
            eprintln!("Available voices: {}", available.join(", "));
        }
    }
}

/// Save f32 samples as a WAV file.
fn save_wav(samples: &[f32], sample_rate: u32, path: &std::path::Path) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec).expect("Failed to create WAV file");
    for &s in samples {
        let sample = (s * 32767.0).clamp(-32768.0, 32767.0) as i16;
        writer.write_sample(sample).expect("Failed to write sample");
    }
    writer.finalize().expect("Failed to finalize WAV");
}

// ============================================================================
// Speech-to-text (Whisper)
// ============================================================================

fn cmd_stt(output: LoaderOutput, audio_path: PathBuf, model_dir: Option<PathBuf>) {
    use whisper_tensor::super_graph::SuperGraphContext;
    use whisper_tensor::super_graph::cache::SuperGraphTensorCache;
    use whisper_tensor::super_graph::data::{SuperGraphAudioClip, SuperGraphData};

    let interface = output
        .interfaces
        .iter()
        .find_map(|i| match &i.interface {
            AnyInterface::SpeechToTextInterface(x) => Some(x),
            _ => None,
        })
        .unwrap_or_else(|| {
            eprintln!("No STT interface found in loaded model");
            std::process::exit(1);
        });

    // Load and preprocess audio → mel spectrogram
    eprintln!("Loading audio: {}", audio_path.display());
    let samples = load_wav_f32(&audio_path, interface.sample_rate);
    let audio_duration = samples.len() as f64 / interface.sample_rate as f64;
    eprintln!(
        "Audio: {:.2}s ({} samples at {}Hz)",
        audio_duration,
        samples.len(),
        interface.sample_rate
    );

    // Load mel filterbank from preprocessor_config.json
    let mel_filters = load_mel_filters(model_dir.as_deref(), interface.num_mel_bins as usize);
    let mel = compute_mel_spectrogram(&samples, interface.num_mel_bins as usize, &mel_filters);
    let mel_len = mel.len() / interface.num_mel_bins as usize;
    eprintln!(
        "Mel spectrogram: {} bins x {} frames",
        interface.num_mel_bins, mel_len
    );

    let mel_tensor = NumericTensor::<DynRank>::from_vec_shape(
        mel,
        vec![1, interface.num_mel_bins as usize, mel_len],
    )
    .unwrap();

    // Run encoder
    eprintln!("Running encoder...");
    let mut backend = EvalBackend::NDArray;
    let start = std::time::Instant::now();

    let encoder_output = {
        let mut data = SuperGraphData::new();
        data.audio_clips.insert(
            interface.audio_input_link,
            SuperGraphAudioClip::new(mel_tensor, interface.sample_rate),
        );
        data.tensor_maps.insert(
            interface.encoder_weights_link,
            output.models[0].model.get_tensor_store(),
        );
        let symbolic_graphs = vec![output.models[0].model.get_symbolic_graph()];
        let mut observer = ();
        let mut tensor_cache = SuperGraphTensorCache::new();
        let mut context = SuperGraphContext {
            observer: &mut observer,
            eval_backend: &mut backend,
            super_graph_tensor_cache: &mut tensor_cache,
            caches: None,
            symbolic_graphs,
            use_compiled_models: false,
            compiled_models: None,
        };
        interface
            .encoder_super_graph
            .run(data, &mut context)
            .unwrap_or_else(|e| {
                eprintln!("Encoder error: {e}");
                std::process::exit(1);
            })
    };

    let encoder_hidden = encoder_output
        .tensors
        .get(&interface.encoder_output_link)
        .expect("No encoder output tensor")
        .clone();
    eprintln!("Encoder done in {:.2?}", start.elapsed());

    // Run decoder autoregressively
    eprintln!("Running decoder...");
    let decode_start = std::time::Instant::now();

    let mut super_graph_caches = SuperGraphCache::new();

    let decoder_sg = &interface.decoder_super_graph;
    let decoder_symbolic_graphs = vec![
        output.models[0].model.get_symbolic_graph(), // dummy for index 0
        output.models[1].model.get_symbolic_graph(), // decoder at index 1
    ];

    // Start with decoder_start_token_id + forced decoder IDs from generation_config
    // For English-only Whisper: [50257(<|startoftranscript|>), 50362(<|notimestamps|>)]
    let forced_ids =
        load_forced_decoder_ids(model_dir.as_deref(), interface.decoder_start_token_id);
    let mut token_ids: Vec<u32> = forced_ids;
    let max_tokens = 448;

    let mut step_start = std::time::Instant::now();
    for _step in 0..max_tokens {
        let mut data = SuperGraphData::new();
        data.tensor_maps.insert(
            interface.decoder_weights_link,
            output.models[1].model.get_tensor_store(),
        );
        data.tensors.insert(
            interface.decoder_encoder_hidden_link,
            encoder_hidden.clone(),
        );

        let token_tensor =
            NumericTensor::<DynRank>::from_vec_shape(token_ids.clone(), vec![token_ids.len()])
                .unwrap();
        data.tensors
            .insert(interface.decoder_token_link, token_tensor);
        data.hashes.insert(interface.decoder_cache_key_link, 0);

        let mut observer = ();
        let mut tensor_cache = SuperGraphTensorCache::new();
        let mut context = SuperGraphContext {
            observer: &mut observer,
            eval_backend: &mut backend,
            super_graph_tensor_cache: &mut tensor_cache,
            caches: Some(&mut super_graph_caches),
            symbolic_graphs: decoder_symbolic_graphs.clone(),
            use_compiled_models: false,
            compiled_models: None,
        };

        let result = decoder_sg.run(data, &mut context).unwrap_or_else(|e| {
            eprintln!("Decoder error at step {_step}: {e}");
            std::process::exit(1);
        });

        // Get logits and pick next token (greedy argmax)
        let logits = result
            .tensors
            .get(&interface.decoder_logit_link)
            .expect("No decoder logit output");

        let logits_f32 = logits
            .cast(whisper_tensor::dtype::DType::F32, &mut backend)
            .expect("cast failed");
        let logits_shape = logits_f32.shape();
        let logits_ndarray = logits_f32.to_ndarray().expect("to_ndarray failed");
        let logits_flat: Vec<f32> = logits_ndarray.flatten().try_into().expect("flatten failed");

        // Take last token's logits — vocab_size is the last dimension
        let vocab_size = *logits_shape.last().unwrap_or(&(logits_flat.len() as u64)) as usize;
        let last_logits = &logits_flat[logits_flat.len() - vocab_size..];

        let next_token = last_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();

        let step_elapsed = step_start.elapsed();
        eprint!("\rDecoder step {_step}: token {next_token} ({step_elapsed:.1?})  ");

        if next_token == interface.eos_token_id {
            eprintln!();
            break;
        }

        token_ids.push(next_token);
        step_start = std::time::Instant::now();
    }

    eprintln!(
        "Decoder done in {:.2?} ({} tokens)",
        decode_start.elapsed(),
        token_ids.len()
    );

    // Decode tokens to text
    let tokenizer =
        whisper_tensor::tokenizer::AnyTokenizer::from_tokenizer_info(&interface.tokenizer);

    let text = tokenizer
        .decode(&token_ids)
        .unwrap_or_else(|e| format!("<decode error: {e}>"));

    eprintln!("Total: {:.2?}", start.elapsed());
    println!("{text}");
}

/// Load a WAV file as f32 samples, resampled to target_sr.
fn load_wav_f32(path: &std::path::Path, target_sr: u32) -> Vec<f32> {
    let mut reader = hound::WavReader::open(path).unwrap_or_else(|e| {
        eprintln!("Failed to open WAV file {}: {e}", path.display());
        std::process::exit(1);
    });
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_val = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.expect("WAV read error") as f32 / max_val)
                .collect()
        }
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .map(|s| s.expect("WAV read error"))
            .collect(),
    };

    let mono: Vec<f32> = if spec.channels > 1 {
        samples
            .chunks(spec.channels as usize)
            .map(|c| c[0])
            .collect()
    } else {
        samples
    };

    if spec.sample_rate != target_sr {
        let ratio = spec.sample_rate as f64 / target_sr as f64;
        let out_len = (mono.len() as f64 / ratio) as usize;
        (0..out_len)
            .map(|i| {
                let pos = i as f64 * ratio;
                let idx = pos as usize;
                let frac = pos - idx as f64;
                let a = mono[idx.min(mono.len() - 1)];
                let b = mono[(idx + 1).min(mono.len() - 1)];
                a + (b - a) * frac as f32
            })
            .collect()
    } else {
        mono
    }
}

/// Load forced decoder IDs from generation_config.json.
/// Returns the initial token sequence: [start_token, forced_id_1, forced_id_2, ...]
fn load_forced_decoder_ids(model_dir: Option<&std::path::Path>, start_token: u32) -> Vec<u32> {
    let mut ids = vec![start_token];
    if let Some(dir) = model_dir {
        let config_path = dir.join("generation_config.json");
        if let Ok(data) = std::fs::read_to_string(&config_path) {
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&data) {
                if let Some(forced) = json["forced_decoder_ids"].as_array() {
                    // forced_decoder_ids: [[position, token_id], ...]
                    let mut pairs: Vec<(u64, u32)> = forced
                        .iter()
                        .filter_map(|pair| {
                            let arr = pair.as_array()?;
                            Some((arr[0].as_u64()?, arr[1].as_u64()? as u32))
                        })
                        .collect();
                    pairs.sort_by_key(|p| p.0);
                    for (_, tok) in pairs {
                        ids.push(tok);
                    }
                    eprintln!("Forced decoder IDs: {:?}", ids);
                }
            }
        }
    }
    ids
}

/// Load mel filterbank from the model's preprocessor_config.json.
/// Falls back to a simple triangular filterbank if the file is not found.
fn load_mel_filters(model_dir: Option<&std::path::Path>, num_mel_bins: usize) -> Vec<f32> {
    let n_freqs = 201; // n_fft/2 + 1 where n_fft=400
    if let Some(dir) = model_dir {
        let config_path = dir.join("preprocessor_config.json");
        if config_path.exists() {
            if let Ok(data) = std::fs::read_to_string(&config_path) {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&data) {
                    if let Some(filters) = json["mel_filters"].as_array() {
                        let mut flat = Vec::with_capacity(num_mel_bins * n_freqs);
                        for row in filters {
                            if let Some(arr) = row.as_array() {
                                for v in arr {
                                    flat.push(v.as_f64().unwrap_or(0.0) as f32);
                                }
                            }
                        }
                        if flat.len() == num_mel_bins * n_freqs {
                            eprintln!("Loaded mel filterbank from {}", config_path.display());
                            return flat;
                        }
                    }
                }
            }
        }
    }
    eprintln!("Using built-in mel filterbank");
    build_mel_filterbank(num_mel_bins, n_freqs, 16000, 400)
}

/// Compute log-mel spectrogram from audio samples.
///
/// Uses n_fft=400, hop_length=160 (Whisper defaults).
/// Returns flattened [num_mel_bins, num_frames] in row-major order.
fn compute_mel_spectrogram(samples: &[f32], num_mel_bins: usize, mel_filters: &[f32]) -> Vec<f32> {
    let n_fft = 400;
    let hop_length = 160;
    let n_freqs = n_fft / 2 + 1; // 201
    assert_eq!(mel_filters.len(), num_mel_bins * n_freqs);

    // Pad to 30 seconds + extra for STFT centering (Whisper expects 3000 frames)
    let target_len = 480000; // 30s * 16000Hz
    let pad = n_fft / 2;
    let total_len = target_len + 2 * pad;
    let mut padded = vec![0.0f32; total_len];
    let copy_len = samples.len().min(target_len);
    padded[pad..pad + copy_len].copy_from_slice(&samples[..copy_len]);

    // Hann window
    let window: Vec<f32> = (0..n_fft)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / n_fft as f32).cos()))
        .collect();

    // STFT — drop last frame to match Whisper's `stft[..., :-1]`
    let num_frames = (padded.len() - n_fft) / hop_length; // stft_frames - 1
    let mut magnitudes = vec![0.0f32; n_freqs * num_frames];

    for frame in 0..num_frames {
        let start = frame * hop_length;
        let mut real = vec![0.0f32; n_fft];

        for i in 0..n_fft {
            real[i] = padded[start + i] * window[i];
        }

        for k in 0..n_freqs {
            let mut re = 0.0f32;
            let mut im = 0.0f32;
            for n in 0..n_fft {
                let angle = -2.0 * std::f32::consts::PI * k as f32 * n as f32 / n_fft as f32;
                re += real[n] * angle.cos();
                im += real[n] * angle.sin();
            }
            magnitudes[k * num_frames + frame] = re * re + im * im;
        }
    }

    // Apply mel filterbank
    let mut mel_spec = vec![0.0f32; num_mel_bins * num_frames];
    for mel in 0..num_mel_bins {
        for frame in 0..num_frames {
            let mut sum = 0.0f32;
            for freq in 0..n_freqs {
                sum += mel_filters[mel * n_freqs + freq] * magnitudes[freq * num_frames + frame];
            }
            mel_spec[mel * num_frames + frame] = sum;
        }
    }

    // Log mel spectrogram
    let log_spec: Vec<f32> = mel_spec.iter().map(|&x| (x.max(1e-10)).log10()).collect();

    // Normalize: clamp to max - 8, then (x + 4) / 4
    let max_val = log_spec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    log_spec
        .iter()
        .map(|&x| (x.max(max_val - 8.0) + 4.0) / 4.0)
        .collect()
}

/// Build a mel filterbank matrix [num_mel_bins, n_freqs].
fn build_mel_filterbank(
    num_mel_bins: usize,
    n_freqs: usize,
    sample_rate: u32,
    n_fft: usize,
) -> Vec<f32> {
    let hz_to_mel = |hz: f32| -> f32 { 2595.0 * (1.0 + hz / 700.0).log10() };
    let mel_to_hz = |mel: f32| -> f32 { 700.0 * (10.0f32.powf(mel / 2595.0) - 1.0) };

    let mel_low = hz_to_mel(0.0);
    let mel_high = hz_to_mel(sample_rate as f32 / 2.0);
    let mel_points: Vec<f32> = (0..num_mel_bins + 2)
        .map(|i| mel_to_hz(mel_low + (mel_high - mel_low) * i as f32 / (num_mel_bins + 1) as f32))
        .collect();

    let freq_bins: Vec<f32> = (0..n_freqs)
        .map(|i| i as f32 * sample_rate as f32 / n_fft as f32)
        .collect();

    let mut filters = vec![0.0f32; num_mel_bins * n_freqs];
    for m in 0..num_mel_bins {
        let (f_low, f_center, f_high) = (mel_points[m], mel_points[m + 1], mel_points[m + 2]);
        for k in 0..n_freqs {
            let freq = freq_bins[k];
            if freq >= f_low && freq <= f_center {
                filters[m * n_freqs + k] = (freq - f_low) / (f_center - f_low);
            } else if freq > f_center && freq <= f_high {
                filters[m * n_freqs + k] = (f_high - freq) / (f_high - f_center);
            }
        }
    }
    filters
}

// ============================================================================
// F5-TTS helpers
// ============================================================================

/// Load a WAV file and return f16 samples (resampled to target_sr if needed).
fn load_wav_f16(path: &std::path::Path, target_sr: u32) -> Vec<half::f16> {
    load_wav_f32(path, target_sr)
        .iter()
        .map(|&s| half::f16::from_f32(s))
        .collect()
}

// ============================================================================
// Utilities
// ============================================================================

/// Generate standard normal noise via Box-Muller transform.
fn generate_gaussian_noise(n: usize, seed: u64) -> Vec<f32> {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut vals = Vec::with_capacity(n);
    while vals.len() + 1 < n {
        let u1: f32 = rand::Rng::random_range(&mut rng, f32::EPSILON..1.0);
        let u2: f32 = rand::Rng::random_range(&mut rng, 0.0f32..std::f32::consts::TAU);
        let r = (-2.0 * u1.ln()).sqrt();
        vals.push(r * u2.cos());
        vals.push(r * u2.sin());
    }
    if vals.len() < n {
        let u1: f32 = rand::Rng::random_range(&mut rng, f32::EPSILON..1.0);
        let u2: f32 = rand::Rng::random_range(&mut rng, 0.0f32..std::f32::consts::TAU);
        vals.push((-2.0 * u1.ln()).sqrt() * u2.cos());
    }
    vals
}

/// Convert an NCHW image tensor (values in [-1, 1]) to an RGB PNG file.
fn save_image_tensor(
    tensor: &NumericTensor<DynRank>,
    path: &std::path::Path,
    backend: &mut EvalBackend,
) {
    let f32_tensor = tensor
        .cast(whisper_tensor::dtype::DType::F32, backend)
        .expect("cast to f32 failed");
    let ndarray = f32_tensor.to_ndarray().expect("to_ndarray failed");
    let image_f32: Vec<f32> = ndarray.flatten().try_into().expect("flatten failed");

    let shape = tensor.shape();
    let ch = shape[1] as usize;
    let img_h = shape[2] as u32;
    let img_w = shape[3] as u32;

    let mut pixels = vec![0u8; (img_h * img_w * 3) as usize];
    for y in 0..img_h as usize {
        for x in 0..img_w as usize {
            for c in 0..ch.min(3) {
                let idx = c * (img_h as usize * img_w as usize) + y * img_w as usize + x;
                let v = (image_f32[idx] + 1.0) * 0.5;
                let byte = (v.clamp(0.0, 1.0) * 255.0) as u8;
                pixels[y * img_w as usize * 3 + x * 3 + c] = byte;
            }
        }
    }

    image::save_buffer(path, &pixels, img_w, img_h, image::ColorType::Rgb8)
        .expect("Failed to save image");
}
