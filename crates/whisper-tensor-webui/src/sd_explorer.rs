use crate::app::{InterfaceId, LoadedModels, ModelLoadState};
use crate::websockets::ServerRequestManager;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use whisper_tensor::backends::ndarray_backend::NDArrayNumericTensor;
use whisper_tensor::dtype::DType;
use whisper_tensor::interfaces::{AnyInterface, StableDiffusionInterface};
use whisper_tensor::super_graph::links::SuperGraphLinkTensor;
use whisper_tensor_server::{
    SuperGraphRequest, SuperGraphRequestBackendMode, WebsocketClientServerMessage,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct SDExplorerState {
    pub prompt_text: String,
    pub num_steps: usize,
    pub guidance_scale: f32,
    pub latent_h: usize,
    pub latent_w: usize,
    pub seed: u64,
    pub sd_pipeline_path: String,
}

impl Default for SDExplorerState {
    fn default() -> Self {
        Self {
            prompt_text: "a photo of a cat".to_string(),
            num_steps: 20,
            guidance_scale: 7.5,
            latent_h: 8,
            latent_w: 8,
            seed: 42,
            sd_pipeline_path: String::new(),
        }
    }
}

pub(crate) struct SDExplorerApp {
    selected_interface_id: Option<InterfaceId>,
    pending_request: Option<(u64, SuperGraphLinkTensor)>,
    generated_image: Option<egui::TextureHandle>,
    status_message: Option<String>,
}

impl SDExplorerApp {
    pub fn new() -> Self {
        Self {
            selected_interface_id: None,
            pending_request: None,
            generated_image: None,
            status_message: None,
        }
    }

    pub fn update(
        &mut self,
        state: &mut SDExplorerState,
        loaded_models: &mut LoadedModels,
        server_request_manager: &mut ServerRequestManager,
        ui: &mut egui::Ui,
    ) {
        // Handle pending response
        if let Some((request_id, output_link)) = self.pending_request.clone() {
            if let Some(response) = server_request_manager.get_response(request_id) {
                self.pending_request = None;
                match response.result {
                    Ok(mut data) => {
                        if let Some(image_tensor) = data.tensor_outputs.remove(&output_link) {
                            self.status_message =
                                Some(format!("Image generated: {:?}", image_tensor.shape()));
                            self.generated_image =
                                Some(tensor_to_egui_texture(&image_tensor, ui.ctx()));
                        } else {
                            self.status_message =
                                Some("Error: output tensor not found".to_string());
                        }
                    }
                    Err(err) => {
                        self.status_message = Some(format!("Error: {err}"));
                    }
                }
            }
        }

        // Find SD interfaces
        let mut sd_interfaces = HashMap::new();
        for (&interface_id, interface) in &loaded_models.current_interfaces {
            if let AnyInterface::StableDiffusionInterface(_) = &interface.interface {
                sd_interfaces.insert(interface.interface_name.clone(), interface_id);
            }
        }

        // Top bar: interface selection + load button
        ui.horizontal(|ui| {
            if sd_interfaces.is_empty() {
                ui.label("No SD pipelines loaded.");
            }
            let mut keys: Vec<_> = sd_interfaces.keys().cloned().collect();
            keys.sort();
            for name in keys {
                ui.selectable_value(
                    &mut self.selected_interface_id,
                    Some(sd_interfaces[&name]),
                    name,
                );
            }
            if ui.button("Load SD Pipeline").clicked() {
                loaded_models.model_load_state = Some(ModelLoadState::DialogOpen(None));
            }
        });

        ui.separator();

        // Pipeline path + load button
        ui.horizontal(|ui| {
            ui.label("SD Pipeline Path:");
            ui.text_edit_singleline(&mut state.sd_pipeline_path);
            if ui.button("Load").clicked() && !state.sd_pipeline_path.is_empty() {
                server_request_manager
                    .send(WebsocketClientServerMessage::LoadSDPipeline {
                        base_path: state.sd_pipeline_path.clone(),
                    })
                    .unwrap();
                self.status_message = Some("Loading SD pipeline...".to_string());
            }
        });

        ui.separator();

        // Get selected interface
        let interface = self
            .selected_interface_id
            .and_then(|id| loaded_models.current_interfaces.get(&id));

        if let Some(interface) = interface {
            if let AnyInterface::StableDiffusionInterface(sd) = &interface.interface {
                // Parameter controls
                ui.horizontal(|ui| {
                    ui.label("Prompt:");
                    ui.text_edit_singleline(&mut state.prompt_text);
                });

                ui.horizontal(|ui| {
                    ui.label("Steps:");
                    ui.add(egui::DragValue::new(&mut state.num_steps).range(1..=100));
                    ui.label("Guidance:");
                    ui.add(
                        egui::DragValue::new(&mut state.guidance_scale)
                            .speed(0.1)
                            .range(1.0..=30.0),
                    );
                    ui.label("Seed:");
                    ui.add(egui::DragValue::new(&mut state.seed));
                });

                ui.horizontal(|ui| {
                    ui.label("Latent H:");
                    ui.add(egui::DragValue::new(&mut state.latent_h).range(4..=128));
                    ui.label("Latent W:");
                    ui.add(egui::DragValue::new(&mut state.latent_w).range(4..=128));
                    ui.label(format!(
                        "({}x{} pixels)",
                        state.latent_w * 8,
                        state.latent_h * 8
                    ));
                });

                let is_running = self.pending_request.is_some();

                ui.horizontal(|ui| {
                    if is_running {
                        ui.spinner();
                        ui.label("Generating...");
                    } else if ui.button("Generate").clicked() {
                        self.run_generation(
                            state,
                            sd,
                            interface.model_ids.clone(),
                            server_request_manager,
                        );
                    }
                });

                if let Some(msg) = &self.status_message {
                    ui.label(msg);
                }

                // Display image
                if let Some(texture) = &self.generated_image {
                    let size = texture.size_vec2();
                    // Scale up for small images
                    let scale = (400.0 / size.x.max(size.y)).max(1.0);
                    let display_size = egui::vec2(size.x * scale, size.y * scale);
                    ui.image(egui::load::SizedTexture::new(texture.id(), display_size));
                }
            }
        } else if self.selected_interface_id.is_some() {
            ui.label("Selected interface no longer available.");
            self.selected_interface_id = None;
        }
    }

    fn run_generation(
        &mut self,
        state: &SDExplorerState,
        sd: &StableDiffusionInterface,
        model_ids: Vec<whisper_tensor_server::LoadedModelId>,
        server_request_manager: &mut ServerRequestManager,
    ) {
        let seq_len = 77;

        // Simple CLIP tokenization: split on spaces, use hardcoded common tokens
        let cond_ids = simple_clip_tokenize(&state.prompt_text, seq_len);
        let mut uncond_ids = vec![0i32; seq_len];
        uncond_ids[0] = 49406; // BOS
        uncond_ids[1] = 49407; // EOS

        let cond_tensor =
            NDArrayNumericTensor::from_vec_shape(cond_ids, &vec![1, seq_len as u64]).unwrap();
        let uncond_tensor =
            NDArrayNumericTensor::from_vec_shape(uncond_ids, &vec![1, seq_len as u64]).unwrap();

        // Compute scheduler params
        let (timestep_values, dt_values, sigma_values, init_sigma) =
            StableDiffusionInterface::compute_euler_schedule(state.num_steps);

        // Generate random noise
        let latent_n = 1 * 4 * state.latent_h * state.latent_w;
        let initial_noise = generate_normal_noise(latent_n, state.seed);
        let scaled_noise: Vec<f32> = initial_noise.iter().map(|&x| x * init_sigma).collect();

        let latent_tensor = NDArrayNumericTensor::from_vec_shape(
            scaled_noise,
            &vec![1, 4, state.latent_h as u64, state.latent_w as u64],
        )
        .unwrap();

        let timesteps_tensor =
            NDArrayNumericTensor::from_vec_shape(timestep_values, &vec![state.num_steps as u64])
                .unwrap();
        let dt_tensor =
            NDArrayNumericTensor::from_vec_shape(dt_values, &vec![state.num_steps as u64]).unwrap();
        let sigmas_tensor =
            NDArrayNumericTensor::from_vec_shape(sigma_values, &vec![state.num_steps as u64])
                .unwrap();
        let iter_count =
            NDArrayNumericTensor::from_vec_shape(vec![state.num_steps as i64], &vec![1]).unwrap();
        let guidance = NDArrayNumericTensor::from_vec(vec![state.guidance_scale]).to_dyn();

        // model_ids order: [text_encoder, unet, vae_decoder]
        let text_encoder_id = model_ids[0];
        let unet_id = model_ids[1];
        let vae_decoder_id = model_ids[2];

        let token = server_request_manager.submit_supergraph_request(SuperGraphRequest {
            do_node_execution_reports: true,
            abbreviated_tensor_report_settings: None,
            attention_token: None,
            super_graph: sd.super_graph.clone(),
            subscribed_tensors: Vec::new(),
            string_inputs: HashMap::new(),
            use_cache: None,
            backend_mode: SuperGraphRequestBackendMode::NDArray,
            symbolic_graph_ids: vec![text_encoder_id, unet_id, vae_decoder_id],
            tensor_inputs: HashMap::from([
                (sd.cond_ids_input, cond_tensor),
                (sd.uncond_ids_input, uncond_tensor),
                (sd.initial_latent_input, latent_tensor),
                (sd.timesteps_input, timesteps_tensor),
                (sd.dt_input, dt_tensor),
                (sd.sigmas_input, sigmas_tensor),
                (sd.iteration_count_input, iter_count),
                (sd.guidance_scale_input, guidance),
            ]),
            model_inputs: HashMap::from([
                (sd.text_encoder_weights, text_encoder_id),
                (sd.unet_weights, unet_id),
                (sd.vae_decoder_weights, vae_decoder_id),
            ]),
            hash_inputs: HashMap::new(),
        });

        self.pending_request = Some((token, sd.image_output));
        self.status_message = Some("Running SD pipeline...".to_string());
    }
}

/// Convert an output tensor (NCHW, f16) to an egui texture.
pub(crate) fn tensor_to_egui_texture(
    tensor: &NDArrayNumericTensor<whisper_tensor::DynRank>,
    ctx: &egui::Context,
) -> egui::TextureHandle {
    let shape = tensor.shape();
    let ch = shape[1] as usize;
    let img_h = shape[2] as usize;
    let img_w = shape[3] as usize;

    // Cast to f32
    let f32_tensor = tensor.cast(DType::F32).unwrap();
    let flat = f32_tensor.flatten();
    let f32_data: Vec<f32> = flat.try_to_vec().unwrap();

    // NCHW → RGBA pixels, remap [-1, 1] → [0, 255]
    let mut pixels = vec![egui::Color32::BLACK; img_h * img_w];
    for y in 0..img_h {
        for x in 0..img_w {
            let r = if ch > 0 {
                let idx = 0 * img_h * img_w + y * img_w + x;
                ((f32_data[idx] + 1.0) * 0.5).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let g = if ch > 1 {
                let idx = 1 * img_h * img_w + y * img_w + x;
                ((f32_data[idx] + 1.0) * 0.5).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let b = if ch > 2 {
                let idx = 2 * img_h * img_w + y * img_w + x;
                ((f32_data[idx] + 1.0) * 0.5).clamp(0.0, 1.0)
            } else {
                0.0
            };
            pixels[y * img_w + x] =
                egui::Color32::from_rgb((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8);
        }
    }

    let color_image = egui::ColorImage {
        size: [img_w, img_h],
        pixels,
        source_size: egui::Vec2::new(img_w as f32, img_h as f32),
    };

    ctx.load_texture("sd_output", color_image, egui::TextureOptions::NEAREST)
}

/// Simple CLIP tokenizer — just uses hardcoded BOS/EOS and a small lookup table.
/// For real usage, load the proper CLIP tokenizer from HuggingFace.
pub(crate) fn simple_clip_tokenize(text: &str, seq_len: usize) -> Vec<i32> {
    // Common CLIP token IDs for simple words
    let token_map: HashMap<&str, i32> = HashMap::from([
        ("a", 320),
        ("the", 518),
        ("of", 539),
        ("in", 530),
        ("on", 525),
        ("with", 593),
        ("and", 537),
        ("is", 533),
        ("an", 550),
        ("photo", 1125),
        ("photograph", 8853),
        ("picture", 1674),
        ("image", 1491),
        ("painting", 3086),
        ("beautiful", 1546),
        ("landscape", 3067),
        ("cat", 2368),
        ("dog", 1929),
        ("person", 2039),
        ("man", 786),
        ("woman", 2308),
        ("tree", 2995),
        ("house", 1316),
        ("car", 1400),
        ("sky", 2390),
        ("water", 1336),
        ("sun", 3103),
        ("mountain", 3860),
        ("flower", 3656),
        ("bird", 3329),
        ("ocean", 4026),
        ("forest", 3880),
        ("city", 1304),
        ("street", 2956),
        ("night", 1558),
        ("day", 1208),
        ("red", 736),
        ("blue", 1612),
        ("green", 1901),
        ("white", 1579),
        ("black", 1173),
        ("big", 1205),
        ("small", 1236),
        ("old", 793),
        ("new", 665),
    ]);

    let mut ids = vec![0i32; seq_len];
    ids[0] = 49406; // BOS

    let words: Vec<&str> = text.split_whitespace().collect();
    let mut pos = 1;
    for word in &words {
        if pos >= seq_len - 1 {
            break;
        }
        let lower = word.to_lowercase();
        let lower = lower.trim_matches(|c: char| !c.is_alphanumeric());
        if let Some(&token_id) = token_map.get(&*lower) {
            ids[pos] = token_id;
            pos += 1;
        }
        // Unknown words are silently skipped
    }
    ids[pos] = 49407; // EOS

    ids
}

/// Generate normally distributed noise using Box-Muller transform.
pub(crate) fn generate_normal_noise(n: usize, seed: u64) -> Vec<f32> {
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
