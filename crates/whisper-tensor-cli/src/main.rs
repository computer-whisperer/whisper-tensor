use clap::Parser;
use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use whisper_tensor::backends::ModelLoadedTensorCache;
use whisper_tensor::backends::eval_backend::EvalBackend;
use whisper_tensor::interfaces::AnyInterface;
use whisper_tensor::loader::{ConfigValue, ConfigValues, Loader};
use whisper_tensor::numeric_tensor::NumericTensor;
use whisper_tensor::super_graph::cache::SuperGraphCache;
use whisper_tensor::tensor_rank::DynRank;
use whisper_tensor::tokenizer::{AnyTokenizer, Tokenizer};
use whisper_tensor_import::loaders::{AutoLoader, OnnxLoader, Rwkv7Loader, TransformersLoader};

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
        model: PathBuf,

        /// Loader to use
        #[arg(long, value_enum, default_value = "auto")]
        loader: LoaderChoice,

        /// HuggingFace tokenizer name (for ONNX loader)
        #[arg(long)]
        tokenizer: Option<String>,

        /// Prompt text (reads from stdin if not provided)
        #[arg(long, short)]
        prompt: Option<String>,

        /// Maximum number of tokens to generate
        #[arg(long, short = 'n', default_value = "100")]
        max_tokens: usize,
    },

    /// Generate an image from a Stable Diffusion model
    Image {
        /// Path to the SD checkpoint (.safetensors)
        model: PathBuf,

        /// SD variant to use
        #[arg(long, value_enum, default_value = "sd15")]
        sd_version: SdVersion,

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
}

#[derive(Clone, clap::ValueEnum)]
enum SdVersion {
    Sd15,
    Sd2,
    Sdxl,
}

fn load_model(loader: &LoaderChoice, config: ConfigValues) -> whisper_tensor::loader::LoaderOutput {
    match loader {
        LoaderChoice::Auto => AutoLoader.load(config),
        LoaderChoice::Transformers => TransformersLoader.load(config),
        LoaderChoice::Onnx => OnnxLoader.load(config),
        LoaderChoice::Rwkv7 => Rwkv7Loader.load(config),
    }
    .unwrap_or_else(|e| {
        eprintln!("Failed to load model: {e}");
        std::process::exit(1);
    })
}

fn main() {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    match cli.command {
        Command::Generate {
            model,
            loader,
            tokenizer,
            prompt,
            max_tokens,
        } => cmd_generate(model, loader, tokenizer, prompt, max_tokens),
        Command::Image {
            model,
            sd_version,
            prompt,
            negative_prompt,
            output,
            steps,
            guidance_scale,
            latent_h,
            latent_w,
            seed,
        } => cmd_image(
            model,
            sd_version,
            prompt,
            negative_prompt,
            output,
            steps,
            guidance_scale,
            latent_h,
            latent_w,
            seed,
        ),
    }
}

fn cmd_generate(
    model_path: PathBuf,
    loader: LoaderChoice,
    tokenizer_name: Option<String>,
    prompt: Option<String>,
    max_tokens: usize,
) {
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

    let mut config = ConfigValues::new();
    config.insert(
        "path".to_string(),
        ConfigValue::FilePath(model_path.clone()),
    );
    if let Some(tok) = &tokenizer_name {
        config.insert("tokenizer".to_string(), ConfigValue::String(tok.clone()));
    }

    eprintln!("Loading model from {}...", model_path.display());
    let output = load_model(&loader, config);
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

fn cmd_image(
    model_path: PathBuf,
    sd_version: SdVersion,
    prompt: String,
    negative_prompt: String,
    output_path: PathBuf,
    steps: usize,
    guidance_scale: f32,
    latent_h: usize,
    latent_w: usize,
    seed: u64,
) {
    use whisper_tensor_import::loaders::{SD15Loader, SD2Loader, SDXLLoader};

    let mut config = ConfigValues::new();
    config.insert(
        "path".to_string(),
        ConfigValue::FilePath(model_path.clone()),
    );

    eprintln!("Loading SD model from {}...", model_path.display());
    let output: whisper_tensor::loader::LoaderOutput = match sd_version {
        SdVersion::Sd15 => SD15Loader.load(config),
        SdVersion::Sd2 => SD2Loader.load(config),
        SdVersion::Sdxl => SDXLLoader.load(config),
    }
    .unwrap_or_else(|e| {
        eprintln!("Failed to load model: {e}");
        std::process::exit(1);
    });

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

    // Tokenize prompts
    let tokenizer = Arc::new(AnyTokenizer::from_tokenizer_info(&interface.tokenizer));
    let seq_len = 77;

    let cond_ids = tokenize_clip(&*tokenizer, &prompt, seq_len);
    let uncond_ids = tokenize_clip(&*tokenizer, &negative_prompt, seq_len);

    let cond_input =
        NumericTensor::<DynRank>::from_vec_shape(cond_ids, vec![1, seq_len]).unwrap();
    let uncond_input =
        NumericTensor::<DynRank>::from_vec_shape(uncond_ids, vec![1, seq_len]).unwrap();

    // Generate initial noise (Box-Muller)
    let latent_n = 4 * latent_h * latent_w;
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
            cond_input,
            Some(uncond_input),
            initial_noise,
            vec![1, 4, latent_h, latent_w],
            steps,
            guidance_scale,
            &mut backend,
        )
        .unwrap_or_else(|e| {
            eprintln!("Inference error: {e}");
            std::process::exit(1);
        });

    eprintln!("Generated in {:.2?}", start.elapsed());

    // Convert NCHW float tensor to RGB PNG
    save_image_tensor(&image_tensor, &output_path, &mut backend);
    eprintln!("Saved to {}", output_path.display());
}

/// Tokenize text for CLIP: encode, prepend BOS (49406), append EOS (49407), pad to seq_len with zeros.
fn tokenize_clip(tokenizer: &dyn Tokenizer, text: &str, seq_len: usize) -> Vec<i32> {
    let bos: u32 = 49406;
    let eos: u32 = 49407;

    let encoded = tokenizer.encode(text);

    let mut ids = Vec::with_capacity(seq_len);
    ids.push(bos as i32);
    let max_text_tokens = seq_len - 2; // room for BOS + EOS
    for &id in encoded.iter().take(max_text_tokens) {
        ids.push(id as i32);
    }
    ids.push(eos as i32);
    ids.resize(seq_len, 0);
    ids
}

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
