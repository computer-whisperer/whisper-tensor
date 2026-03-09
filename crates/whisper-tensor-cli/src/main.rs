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
use whisper_tensor::tokenizer::AnyTokenizer;

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
    },
}

#[derive(Clone, clap::ValueEnum)]
enum LoaderChoice {
    Auto,
    Transformers,
    Onnx,
    Rwkv7,
    Sd15,
    Sd2,
    Sdxl,
    Flux,
    Kokoro,
    Piper,
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
        LoaderChoice::Sd15 => SD15Loader.load(config),
        LoaderChoice::Sd2 => SD2Loader.load(config),
        LoaderChoice::Sdxl => SDXLLoader.load(config),
        LoaderChoice::Flux => FluxLoader.load(config),
        LoaderChoice::Kokoro => KokoroLoader.load(config),
        LoaderChoice::Piper => PiperLoader.load(config),
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
        } => {
            let model_dir = model.clone().unwrap_or_else(|| PathBuf::from("."));
            let config = build_config(model, &configs);
            eprintln!("Loading model...");
            let loaded = load_model(&loader, config);
            cmd_tts(loaded, model_dir, prompt, voice, speed, output);
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

    // Tokenize positive prompt for all input slots
    let mut prompt_tokens = HashMap::new();
    for pi in &interface.positive_prompts {
        let tokenizer = AnyTokenizer::from_tokenizer_info(&pi.tokenizer);
        let ids = pi.tokenize(&tokenizer, &prompt);
        let tensor = NumericTensor::<DynRank>::from_vec_shape(ids, vec![1, pi.seq_len]).unwrap();
        prompt_tokens.insert(pi.link, tensor);
    }

    // Tokenize negative prompt for all input slots (if CFG)
    if let Some(neg_prompts) = &interface.negative_prompts {
        for pi in neg_prompts {
            let tokenizer = AnyTokenizer::from_tokenizer_info(&pi.tokenizer);
            let ids = pi.tokenize(&tokenizer, &negative_prompt);
            let tensor =
                NumericTensor::<DynRank>::from_vec_shape(ids, vec![1, pi.seq_len]).unwrap();
            prompt_tokens.insert(pi.link, tensor);
        }
    }

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
            prompt_tokens,
            initial_noise,
            vec![1, channels, latent_h, latent_w],
            steps,
            guidance_scale,
            &mut backend,
        )
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

fn cmd_tts(output: LoaderOutput, model_dir: PathBuf, text: String, voice_name: String, speed: f32, output_path: PathBuf) {
    let model = &output.models[0].model;

    // Dispatch based on which interface the loader produced
    if let Some(interface) = output.interfaces.iter().find_map(|i| match &i.interface {
        AnyInterface::TextToSpeechInterface(x) => Some(x),
        _ => None,
    }) {
        cmd_tts_kokoro_style(model, interface, &model_dir, &text, &voice_name, speed, &output_path);
    } else if let Some(interface) = output.interfaces.iter().find_map(|i| match &i.interface {
        AnyInterface::PiperInterface(x) => Some(x),
        _ => None,
    }) {
        cmd_tts_piper(model, interface, &text, speed, &output_path);
    } else {
        eprintln!("No TTS interface found in loaded model");
        std::process::exit(1);
    }
}

/// Run TTS with Kokoro-style interface (input_ids + style + speed).
fn cmd_tts_kokoro_style(
    model: &std::sync::Arc<whisper_tensor::model::Model>,
    interface: &whisper_tensor::interfaces::TextToSpeechInterface,
    model_dir: &std::path::Path,
    text: &str,
    voice_name: &str,
    speed: f32,
    output_path: &std::path::Path,
) {
    use whisper_tensor::super_graph::cache::SuperGraphTensorCache;
    use whisper_tensor::super_graph::data::SuperGraphData;
    use whisper_tensor::super_graph::SuperGraphContext;

    // Phonemize and tokenize: espeak-ng IPA → E2M conversion → char-level vocab lookup
    let phonemes = text_to_kokoro_phonemes(text);
    let vocab = load_tts_vocab(&interface.tokenizer);
    let token_ids: Vec<i64> = {
        let mut ids = vec![0i64]; // BOS ($)
        for ch in phonemes.chars() {
            if let Some(&id) = vocab.get(&ch) {
                ids.push(id as i64);
            }
        }
        ids.push(0i64); // EOS ($)
        ids
    };
    eprintln!("Phonemes: {phonemes}");
    let num_tokens = token_ids.len();
    eprintln!("Tokens: {num_tokens}");

    let input_ids_tensor =
        NumericTensor::<DynRank>::from_vec_shape(token_ids, vec![1, num_tokens]).unwrap();

    // Load voice embedding
    let voice_bin = model_dir.join("voices").join(format!("{voice_name}.bin"));
    let style_tensor = if voice_bin.exists() {
        load_voice_style_bin(&voice_bin, num_tokens)
    } else {
        eprintln!("No voice file found: {}", voice_bin.display());
        list_available_voices(model_dir);
        std::process::exit(1);
    };
    let speed_tensor = NumericTensor::<DynRank>::from_vec_shape(vec![speed], vec![1]).unwrap();

    // Run inference
    eprintln!("Running inference...");
    let mut backend = EvalBackend::NDArray;
    let start = std::time::Instant::now();

    let super_graph_data = {
        let mut data = SuperGraphData::new();
        data.tensor_maps
            .insert(interface.model_weights_link, model.get_tensor_store());
        data.tensors.insert(interface.input_ids_link, input_ids_tensor);
        data.tensors.insert(interface.style_link, style_tensor);
        data.tensors.insert(interface.speed_link, speed_tensor);
        data
    };

    let super_graph_output = {
        let mut observer = ();
        let mut super_graph_tensor_cache = SuperGraphTensorCache::new();
        let mut context = SuperGraphContext {
            observer: &mut observer,
            eval_backend: &mut backend,
            super_graph_tensor_cache: &mut super_graph_tensor_cache,
            caches: None,
            symbolic_graphs: vec![model.get_symbolic_graph()],
            use_compiled_models: false,
            compiled_models: None,
        };
        interface
            .super_graph
            .run(super_graph_data, &mut context)
            .unwrap_or_else(|e| {
                eprintln!("Inference error: {e}");
                std::process::exit(1);
            })
    };

    let audio = super_graph_output
        .tensors
        .get(&interface.audio_output_link)
        .expect("No audio output tensor");

    eprintln!("Generated in {:.2?}", start.elapsed());

    let samples = audio_tensor_to_samples(audio, &mut backend);
    save_wav(&samples, interface.sample_rate, output_path);
    eprintln!(
        "Saved {:.1}s of audio to {}",
        samples.len() as f64 / interface.sample_rate as f64,
        output_path.display()
    );
}

/// Run TTS with Piper VITS interface (input + input_lengths + scales).
fn cmd_tts_piper(
    model: &std::sync::Arc<whisper_tensor::model::Model>,
    interface: &whisper_tensor::interfaces::PiperInterface,
    text: &str,
    speed: f32,
    output_path: &std::path::Path,
) {
    use whisper_tensor::super_graph::cache::SuperGraphTensorCache;
    use whisper_tensor::super_graph::data::SuperGraphData;
    use whisper_tensor::super_graph::SuperGraphContext;

    // Phonemize text via espeak-ng using the voice code from config
    let sentences = espeak_rs::text_to_phonemes(text, &interface.espeak_voice, None, true, false)
        .expect("espeak-ng phonemization failed");
    let ipa = sentences.join(" ");
    eprintln!("Phonemes: {ipa}");

    // Build phoneme ID map from the stored JSON
    let phoneme_id_map = parse_piper_phoneme_id_map(&interface.phoneme_id_map_json);

    // Tokenize: BOS + (phoneme_ids interleaved with PAD) + EOS
    let mut token_ids: Vec<i64> = Vec::new();
    token_ids.push(1); // BOS (^)
    token_ids.push(0); // PAD (_)
    for ch in ipa.chars() {
        if let Some(ids) = phoneme_id_map.get(&ch) {
            for &id in ids {
                token_ids.push(id);
            }
        }
        token_ids.push(0); // PAD interspersed
    }
    token_ids.push(2); // EOS ($)

    let num_tokens = token_ids.len();
    eprintln!("Tokens: {num_tokens}");

    let input_tensor =
        NumericTensor::<DynRank>::from_vec_shape(token_ids, vec![1, num_tokens]).unwrap();
    let input_lengths_tensor =
        NumericTensor::<DynRank>::from_vec_shape(vec![num_tokens as i64], vec![1]).unwrap();

    // Scales: [noise_scale, length_scale, noise_scale_w]
    // length_scale is 1/speed (higher speed = shorter durations)
    let length_scale = 1.0 / speed;
    let scales_tensor =
        NumericTensor::<DynRank>::from_vec_shape(vec![0.667f32, length_scale, 0.8], vec![3]).unwrap();

    eprintln!("Running inference...");
    let mut backend = EvalBackend::NDArray;
    let start = std::time::Instant::now();

    let super_graph_data = {
        let mut data = SuperGraphData::new();
        data.tensor_maps
            .insert(interface.model_weights_link, model.get_tensor_store());
        data.tensors.insert(interface.input_link, input_tensor);
        data.tensors.insert(interface.input_lengths_link, input_lengths_tensor);
        data.tensors.insert(interface.scales_link, scales_tensor);
        if let Some(sid_link) = interface.speaker_id_link {
            let sid_tensor =
                NumericTensor::<DynRank>::from_vec_shape(vec![0i64], vec![1]).unwrap();
            data.tensors.insert(sid_link, sid_tensor);
        }
        data
    };

    let super_graph_output = {
        let mut observer = ();
        let mut super_graph_tensor_cache = SuperGraphTensorCache::new();
        let mut context = SuperGraphContext {
            observer: &mut observer,
            eval_backend: &mut backend,
            super_graph_tensor_cache: &mut super_graph_tensor_cache,
            caches: None,
            symbolic_graphs: vec![model.get_symbolic_graph()],
            use_compiled_models: false,
            compiled_models: None,
        };
        interface
            .super_graph
            .run(super_graph_data, &mut context)
            .unwrap_or_else(|e| {
                eprintln!("Inference error: {e}");
                std::process::exit(1);
            })
    };

    let audio = super_graph_output
        .tensors
        .get(&interface.audio_output_link)
        .expect("No audio output tensor");

    eprintln!("Generated in {:.2?}", start.elapsed());

    let samples = audio_tensor_to_samples(audio, &mut backend);
    save_wav(&samples, interface.sample_rate, output_path);
    eprintln!(
        "Saved {:.1}s of audio to {}",
        samples.len() as f64 / interface.sample_rate as f64,
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

/// Parse Piper's phoneme_id_map JSON into a char → Vec<i64> lookup.
fn parse_piper_phoneme_id_map(json: &str) -> HashMap<char, Vec<i64>> {
    let value: serde_json::Value = serde_json::from_str(json).expect("Invalid phoneme_id_map JSON");
    let obj = value.as_object().expect("phoneme_id_map is not an object");
    let mut map = HashMap::new();
    for (key, val) in obj {
        if let Some(ch) = key.chars().next() {
            if key.chars().count() == 1 {
                let ids: Vec<i64> = val
                    .as_array()
                    .expect("phoneme IDs not an array")
                    .iter()
                    .map(|v| v.as_i64().expect("phoneme ID not i64"))
                    .collect();
                map.insert(ch, ids);
            }
        }
    }
    map
}

/// Load phoneme vocab from tokenizer.json (Kokoro) or config.json (Kitten TTS).
///
/// Load Kokoro tokenizer vocab (character → token ID) from tokenizer.json.
fn load_tts_vocab(info: &whisper_tensor::metadata::TokenizerInfo) -> HashMap<char, u32> {
    let path = match info {
        whisper_tensor::metadata::TokenizerInfo::HFTokenizerLocal(p) => p.clone(),
        _ => panic!("Expected HFTokenizerLocal for TTS tokenizer"),
    };
    let json = std::fs::read_to_string(&path).expect("Failed to read tokenizer file");
    let value: serde_json::Value = serde_json::from_str(&json).expect("Invalid JSON");

    let vocab = value["model"]["vocab"]
        .as_object()
        .expect("Cannot find model.vocab in tokenizer file");
    let mut map = HashMap::new();
    for (key, val) in vocab {
        let id = val.as_u64().expect("vocab id not u64") as u32;
        if key.chars().count() == 1 {
            map.insert(key.chars().next().unwrap(), id);
        }
    }
    map
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
// Phonemization (espeak-ng + E2M conversion for Kokoro)
// ============================================================================

/// Convert English text to Kokoro-compatible phoneme string.
///
/// Pipeline: text -> espeak-ng IPA -> E2M (espeak-to-Misaki) conversion
fn text_to_kokoro_phonemes(text: &str) -> String {
    let sentences = espeak_rs::text_to_phonemes(text, "en-us", None, true, false)
        .expect("espeak-ng phonemization failed");
    let ipa = sentences.join(" ");
    espeak_to_misaki(&ipa)
}

/// Apply espeak-to-Misaki (E2M) phoneme conversion.
///
/// Converts espeak-ng's IPA output into the Misaki phoneme format
/// that Kokoro's tokenizer expects. Replacements are applied in
/// longest-first order to handle multi-character sequences correctly.
fn espeak_to_misaki(ipa: &str) -> String {
    // E2M replacements, applied longest-first
    static E2M: &[(&str, &str)] = &[
        // Multi-char (longest first)
        ("a\u{0361}\u{026a}", "I"),     // a͡ɪ -> I (PRICE)
        ("a\u{0361}\u{028a}", "W"),     // a͡ʊ -> W (MOUTH)
        ("d\u{0361}\u{0292}", "\u{02A4}"), // d͡ʒ -> ʤ
        ("e\u{0361}\u{026a}", "A"),     // e͡ɪ -> A (FACE)
        ("t\u{0361}\u{0283}", "\u{02A7}"), // t͡ʃ -> ʧ
        ("\u{0254}\u{0361}\u{026a}", "Y"), // ɔ͡ɪ -> Y (CHOICE)
        ("o\u{0361}\u{028a}", "O"),     // o͡ʊ -> O (GOAT, US)
        // Without tie bar (fallback)
        ("a\u{026a}", "I"),             // aɪ -> I
        ("a\u{028a}", "W"),             // aʊ -> W
        ("d\u{0292}", "\u{02A4}"),      // dʒ -> ʤ
        ("e\u{026a}", "A"),             // eɪ -> A
        ("t\u{0283}", "\u{02A7}"),      // tʃ -> ʧ
        ("\u{0254}\u{026a}", "Y"),      // ɔɪ -> Y
        ("o\u{028a}", "O"),             // oʊ -> O
        // Syllabic patterns
        ("\u{0294}\u{02cc}n\u{0329}", "t\u{1d4a}n"),  // ʔˌn̩ -> tᵊn
        ("\u{0294}n", "t\u{1d4a}n"),                    // ʔn -> tᵊn
        ("\u{0259}\u{0361}l", "\u{1d4a}l"),            // ə͡l -> ᵊl
        ("\u{0259}l", "\u{1d4a}l"),                     // əl -> ᵊl (no tie)
        // R-colored
        ("\u{025a}", "\u{0259}\u{0279}"),              // ɚ -> əɹ
        // US English specifics
        ("\u{025c}\u{02d0}\u{0279}", "\u{025c}\u{0279}"), // ɜːɹ -> ɜɹ
        ("\u{025c}\u{02d0}", "\u{025c}\u{0279}"),          // ɜː -> ɜɹ
        ("\u{026a}\u{0259}", "i\u{0259}"),                 // ɪə -> iə
        // Single-char
        ("e", "A"),                     // bare e -> A
        ("r", "\u{0279}"),             // r -> ɹ
        ("x", "k"),
        ("\u{00e7}", "k"),             // ç -> k
        ("\u{0250}", "\u{0259}"),      // ɐ -> ə
        ("\u{026c}", "l"),             // ɬ -> l
        ("\u{0294}", "t"),             // ʔ -> t
        ("o", "\u{0254}"),             // o -> ɔ
        ("\u{027e}", "T"),             // ɾ -> T
    ];

    let mut result = ipa.to_string();

    // Remove combining tilde
    result = result.replace('\u{0303}', "");

    // Remove remaining palatalization marker
    result = result.replace('\u{02b2}', "");

    // Apply E2M replacements
    for &(from, to) in E2M {
        result = result.replace(from, to);
    }

    // Remove length marks (US English)
    result = result.replace('\u{02d0}', "");

    // Remove syllabic marker (any remaining)
    result = result.replace('\u{0329}', "");

    result
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
