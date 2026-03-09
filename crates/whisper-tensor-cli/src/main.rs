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
    FluxSchnell,
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
        LoaderChoice::FluxSchnell => FluxSchnellLoader.load(config),
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
