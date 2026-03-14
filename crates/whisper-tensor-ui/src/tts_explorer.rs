use crate::app::{InterfaceId, LoadedModels, LoadedTokenizers, ModelLoadState};
use crate::audio_io::{
    download_audio_wav, play_audio_samples, stop_audio_playback, tensor_to_audio_samples,
};
use crate::websockets::ServerRequestManager;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use whisper_tensor::backends::ndarray_backend::NDArrayNumericTensor;
use whisper_tensor::interfaces::{
    AnyInterface, KokoroVoiceEmbedding, TTSInputConfig, TextToSpeechInterface,
};
use whisper_tensor::super_graph::links::SuperGraphLink;
use whisper_tensor_server::{SuperGraphRequest, SuperGraphRequestBackendMode};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct TTSExplorerState {
    pub text: String,
    pub speed: f32,
    pub piper_speaker_id: i64,
    pub kokoro_voice_name: Option<String>,
}

impl Default for TTSExplorerState {
    fn default() -> Self {
        Self {
            text: "Hello from Whisper Tensor".to_string(),
            speed: 1.0,
            piper_speaker_id: 0,
            kokoro_voice_name: None,
        }
    }
}

pub(crate) struct TTSExplorerApp {
    selected_interface_id: Option<InterfaceId>,
    pending_request: Option<(u64, SuperGraphLink, u32)>,
    generated_audio: Option<NDArrayNumericTensor<whisper_tensor::DynRank>>,
    generated_sample_rate_hz: Option<u32>,
    status_message: Option<String>,
}

impl TTSExplorerApp {
    pub fn new() -> Self {
        Self {
            selected_interface_id: None,
            pending_request: None,
            generated_audio: None,
            generated_sample_rate_hz: None,
            status_message: None,
        }
    }

    pub fn update(
        &mut self,
        state: &mut TTSExplorerState,
        loaded_models: &mut LoadedModels,
        _loaded_tokenizers: &mut LoadedTokenizers,
        server_request_manager: &mut ServerRequestManager,
        ui: &mut egui::Ui,
    ) {
        if let Some((request_id, output_link, sample_rate_hz)) = self.pending_request
            && let Some(response) = server_request_manager.get_response(request_id)
        {
            self.pending_request = None;
            match response.result {
                Ok(mut data) => {
                    if let Some(audio_tensor) = data.tensor_outputs.remove(&output_link) {
                        let sample_count = audio_tensor.shape().iter().product::<u64>() as usize;
                        let duration_s = sample_count as f64 / sample_rate_hz as f64;
                        self.status_message = Some(format!(
                            "Generated audio: {sample_count} samples ({duration_s:.2}s @ {sample_rate_hz}Hz)"
                        ));
                        self.generated_audio = Some(audio_tensor);
                        self.generated_sample_rate_hz = Some(sample_rate_hz);
                    } else {
                        self.status_message = Some("Error: output audio not found".to_string());
                        self.generated_sample_rate_hz = None;
                    }
                }
                Err(err) => {
                    self.status_message = Some(format!("Error: {err}"));
                    self.generated_sample_rate_hz = None;
                }
            }
        }

        let mut tts_interfaces = HashMap::new();
        for (&interface_id, interface) in &loaded_models.current_interfaces {
            if let AnyInterface::TextToSpeechInterface(_) = &interface.interface {
                tts_interfaces.insert(interface.interface_name.clone(), interface_id);
            }
        }

        ui.horizontal(|ui| {
            if tts_interfaces.is_empty() {
                ui.label("No TTS interfaces loaded.");
            }
            let mut keys: Vec<_> = tts_interfaces.keys().cloned().collect();
            keys.sort();
            for name in keys {
                ui.selectable_value(
                    &mut self.selected_interface_id,
                    Some(tts_interfaces[&name]),
                    name,
                );
            }
            if ui.button("Load TTS Model").clicked() {
                loaded_models.model_load_state = Some(ModelLoadState::DialogOpen(None));
            }
        });

        ui.separator();

        let interface = self
            .selected_interface_id
            .and_then(|id| loaded_models.current_interfaces.get(&id));

        if let Some(interface) = interface {
            if let AnyInterface::TextToSpeechInterface(tts) = &interface.interface {
                ui.horizontal(|ui| {
                    ui.label("Text:");
                    ui.text_edit_singleline(&mut state.text);
                });
                ui.horizontal(|ui| {
                    ui.label("Speed:");
                    ui.add(
                        egui::DragValue::new(&mut state.speed)
                            .speed(0.05)
                            .range(0.1..=4.0),
                    );
                });
                if let TTSInputConfig::Kokoro {
                    voices,
                    default_voice,
                    ..
                } = &tts.input_config
                {
                    ensure_kokoro_voice_selection(
                        &mut state.kokoro_voice_name,
                        voices,
                        default_voice.as_deref(),
                    );
                    if !voices.is_empty() {
                        let mut selected = state.kokoro_voice_name.clone().unwrap();
                        egui::ComboBox::from_id_salt((
                            "tts_kokoro_voice",
                            interface.interface_name.as_str(),
                        ))
                        .selected_text(selected.clone())
                        .show_ui(ui, |ui| {
                            for voice in voices {
                                ui.selectable_value(
                                    &mut selected,
                                    voice.name.clone(),
                                    voice.name.as_str(),
                                );
                            }
                        });
                        state.kokoro_voice_name = Some(selected);
                    }
                }
                if let TTSInputConfig::Piper {
                    speaker_id_link: Some(_),
                    num_speakers,
                    ..
                } = &tts.input_config
                {
                    ui.horizontal(|ui| {
                        ui.label("Speaker ID:");
                        let max_speaker = (*num_speakers as i64).saturating_sub(1).max(0);
                        ui.add(
                            egui::DragValue::new(&mut state.piper_speaker_id)
                                .range(0..=max_speaker),
                        );
                    });
                }

                let is_running = self.pending_request.is_some();
                ui.horizontal(|ui| {
                    if is_running {
                        ui.spinner();
                        ui.label("Generating...");
                    } else if ui.button("Generate").clicked() {
                        self.run_generation(
                            state,
                            tts,
                            interface.model_ids.clone(),
                            server_request_manager,
                        );
                    }
                });

                if let Some(msg) = &self.status_message {
                    ui.label(msg);
                }

                if let (Some(audio), Some(sample_rate_hz)) =
                    (self.generated_audio.clone(), self.generated_sample_rate_hz)
                {
                    ui.label(format!("Output tensor shape: {:?}", audio.shape()));
                    let mut play_clicked = false;
                    let mut stop_clicked = false;
                    let mut download_clicked = false;
                    ui.horizontal(|ui| {
                        play_clicked = ui.button("Play").clicked();
                        stop_clicked = ui.button("Stop").clicked();
                        download_clicked = ui.button("Download WAV").clicked();
                    });

                    if play_clicked {
                        let result = tensor_to_audio_samples(&audio)
                            .and_then(|samples| play_audio_samples(&samples, sample_rate_hz));
                        match result {
                            Ok(()) => {
                                self.status_message =
                                    Some(format!("Playing audio @ {sample_rate_hz}Hz"));
                            }
                            Err(err) => {
                                self.status_message = Some(format!("Audio playback failed: {err}"));
                            }
                        }
                    }
                    if stop_clicked {
                        stop_audio_playback();
                        self.status_message = Some("Stopped playback".to_string());
                    }
                    if download_clicked {
                        let result = tensor_to_audio_samples(&audio)
                            .and_then(|samples| download_audio_wav(&samples, sample_rate_hz));
                        match result {
                            Ok(()) => {
                                self.status_message = Some("Saved generated_audio.wav".to_string());
                            }
                            Err(err) => {
                                self.status_message = Some(format!("Audio download failed: {err}"));
                            }
                        }
                    }
                } else if let Some(audio) = &self.generated_audio {
                    ui.label(format!("Output tensor shape: {:?}", audio.shape()));
                }
            }
        } else if self.selected_interface_id.is_some() {
            ui.label("Selected interface no longer available.");
            self.selected_interface_id = None;
        }
    }

    fn run_generation(
        &mut self,
        state: &TTSExplorerState,
        tts: &TextToSpeechInterface,
        model_ids: Vec<whisper_tensor_server::LoadedModelId>,
        server_request_manager: &mut ServerRequestManager,
    ) {
        let mut tensor_inputs = HashMap::new();
        let mut string_inputs = HashMap::new();
        string_inputs.insert(tts.text_input_link, state.text.clone());

        match &tts.input_config {
            TTSInputConfig::Kokoro {
                style_link,
                speed_link,
                voices,
                default_voice,
            } => {
                let voice = selected_kokoro_voice(
                    &state.kokoro_voice_name,
                    voices,
                    default_voice.as_deref(),
                )
                .ok_or_else(|| "no Kokoro voice embeddings available".to_string());
                let voice = match voice {
                    Ok(v) => v,
                    Err(err) => {
                        self.status_message = Some(err);
                        return;
                    }
                };
                let approx_tokens = state.text.chars().count().saturating_add(2);
                let style_values = match voice.style_for_token_count(approx_tokens) {
                    Ok(values) => values,
                    Err(err) => {
                        self.status_message = Some(format!(
                            "Failed to decode Kokoro voice '{}': {}",
                            voice.name, err
                        ));
                        return;
                    }
                };
                let style = NDArrayNumericTensor::<whisper_tensor::DynRank>::from_vec_shape(
                    style_values,
                    &vec![1, KokoroVoiceEmbedding::STYLE_DIM as u64],
                )
                .unwrap();
                let speed = NDArrayNumericTensor::<whisper_tensor::DynRank>::from_vec_shape(
                    vec![state.speed],
                    &vec![1],
                )
                .unwrap();
                tensor_inputs.insert(*style_link, style);
                tensor_inputs.insert(*speed_link, speed);
            }
            TTSInputConfig::Piper {
                scales_link,
                speaker_id_link,
                ..
            } => {
                let length_scale = 1.0 / state.speed.max(0.1);
                let scales = NDArrayNumericTensor::<whisper_tensor::DynRank>::from_vec_shape(
                    vec![0.667f32, length_scale, 0.8],
                    &vec![3],
                )
                .unwrap();
                tensor_inputs.insert(*scales_link, scales);
                if let Some(sid_link) = speaker_id_link {
                    let sid = NDArrayNumericTensor::<whisper_tensor::DynRank>::from_vec_shape(
                        vec![state.piper_speaker_id],
                        &vec![1],
                    )
                    .unwrap();
                    tensor_inputs.insert(*sid_link, sid);
                }
            }
            TTSInputConfig::F5 { .. } => {
                self.status_message =
                    Some("F5-TTS UI wiring for reference audio is not implemented yet".to_string());
                return;
            }
        }

        let symbolic_graph_ids: Vec<_> = model_ids.to_vec();
        let model_inputs: HashMap<_, _> = tts
            .model_weights
            .iter()
            .zip(model_ids.iter())
            .map(|(&link, &id)| (link, id))
            .collect();

        let token = server_request_manager.submit_supergraph_request(SuperGraphRequest {
            do_node_execution_reports: true,
            abbreviated_tensor_report_settings: None,
            attention_token: None,
            super_graph: tts.super_graph.clone(),
            subscribed_tensors: Vec::new(),
            string_inputs,
            use_cache: None,
            backend_mode: SuperGraphRequestBackendMode::NDArray,
            symbolic_graph_ids,
            tensor_inputs,
            model_inputs,
            hash_inputs: HashMap::new(),
        });

        self.pending_request = Some((token, tts.audio_output_link, tts.sample_rate));
        self.status_message = Some("Running TTS pipeline...".to_string());
    }
}

fn ensure_kokoro_voice_selection(
    selected: &mut Option<String>,
    voices: &[KokoroVoiceEmbedding],
    default_voice: Option<&str>,
) {
    let selected_valid = selected
        .as_ref()
        .is_some_and(|name| voices.iter().any(|v| v.name == *name));
    if selected_valid {
        return;
    }
    if let Some(default_voice) = default_voice
        && voices.iter().any(|v| v.name == default_voice)
    {
        *selected = Some(default_voice.to_string());
        return;
    }
    *selected = voices.first().map(|v| v.name.clone());
}

fn selected_kokoro_voice<'a>(
    selected: &Option<String>,
    voices: &'a [KokoroVoiceEmbedding],
    default_voice: Option<&str>,
) -> Option<&'a KokoroVoiceEmbedding> {
    if let Some(name) = selected
        && let Some(voice) = voices.iter().find(|v| v.name == *name)
    {
        return Some(voice);
    }
    if let Some(default_voice) = default_voice
        && let Some(voice) = voices.iter().find(|v| v.name == default_voice)
    {
        return Some(voice);
    }
    voices.first()
}
