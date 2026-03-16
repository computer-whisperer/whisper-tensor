use crate::app::{InterfaceId, LoadedModels, LoadedTokenizers, ModelLoadState};
use crate::audio_io::decode_wav_bytes_to_mono_f32;
#[cfg(not(target_arch = "wasm32"))]
use crate::audio_io::pick_audio_file_native;
#[cfg(target_arch = "wasm32")]
use crate::audio_io::{WebAudioFilePickReceiver, start_audio_file_pick_web};
use crate::websockets::ServerRequestManager;
use crate::widgets::progress_report::SuperGraphProgressWidgetState;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use whisper_tensor::DynRank;
use whisper_tensor::backends::ndarray_backend::NDArrayNumericTensor;
use whisper_tensor::dtype::DType;
use whisper_tensor::interfaces::{AnyInterface, SpeechToTextInterface};
use whisper_tensor::metadata::TokenizerInfo;
use whisper_tensor::super_graph::links::SuperGraphLink;
use whisper_tensor::tokenizer::Tokenizer;
use whisper_tensor_server::{
    SuperGraphAudioInput, SuperGraphRequest, SuperGraphRequestBackendMode,
};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub(crate) struct STTExplorerState {}

#[derive(Clone, Debug)]
struct PendingSttRequest {
    request_id: u64,
    output_link: SuperGraphLink,
    eos_token_id: u32,
    tokenizer: TokenizerInfo,
}

pub(crate) struct STTExplorerApp {
    selected_interface_id: Option<InterfaceId>,
    pending_request: Option<PendingSttRequest>,
    selected_audio_name: Option<String>,
    selected_audio_bytes: Option<Vec<u8>>,
    status_message: Option<String>,
    transcription_text: Option<String>,
    transcription_tokens: Option<Vec<u32>>,
    progress_widget_state: SuperGraphProgressWidgetState,
    #[cfg(target_arch = "wasm32")]
    pending_web_audio_pick: Option<WebAudioFilePickReceiver>,
}

impl STTExplorerApp {
    pub fn new() -> Self {
        Self {
            selected_interface_id: None,
            pending_request: None,
            selected_audio_name: None,
            selected_audio_bytes: None,
            status_message: None,
            transcription_text: None,
            transcription_tokens: None,
            progress_widget_state: SuperGraphProgressWidgetState::default(),
            #[cfg(target_arch = "wasm32")]
            pending_web_audio_pick: None,
        }
    }

    pub fn update(
        &mut self,
        _state: &mut STTExplorerState,
        loaded_models: &mut LoadedModels,
        loaded_tokenizers: &mut LoadedTokenizers,
        server_request_manager: &mut ServerRequestManager,
        ui: &mut egui::Ui,
    ) {
        #[cfg(target_arch = "wasm32")]
        self.poll_audio_file_upload_web();

        if let Some(pending) = &self.pending_request
            && let Some(reports) = server_request_manager.get_reports(pending.request_id)
        {
            self.progress_widget_state.ingest_reports(reports);
        }

        self.handle_pending_response(loaded_tokenizers, server_request_manager);

        let mut stt_interfaces = HashMap::new();
        for (&interface_id, interface) in &loaded_models.current_interfaces {
            if let AnyInterface::SpeechToTextInterface(_) = &interface.interface {
                stt_interfaces.insert(interface.interface_name.clone(), interface_id);
            }
        }

        ui.horizontal(|ui| {
            if stt_interfaces.is_empty() {
                ui.label("No STT interfaces loaded.");
            }
            let mut keys: Vec<_> = stt_interfaces.keys().cloned().collect();
            keys.sort();
            for name in keys {
                ui.selectable_value(
                    &mut self.selected_interface_id,
                    Some(stt_interfaces[&name]),
                    name,
                );
            }
            if ui.button("Load STT Model").clicked() {
                loaded_models.model_load_state = Some(ModelLoadState::DialogOpen(None));
            }
        });

        ui.separator();

        let interface = self
            .selected_interface_id
            .and_then(|id| loaded_models.current_interfaces.get(&id));

        if let Some(interface) = interface {
            if let AnyInterface::SpeechToTextInterface(stt) = &interface.interface {
                ui.label(format!(
                    "Input: mono WAV, resampled to {} Hz",
                    stt.sample_rate
                ));
                ui.label(format!(
                    "Decode steps: {} (EOS token {})",
                    stt.max_decode_steps, stt.eos_token_id
                ));
                ui.label(format!("Tokenizer: {}", tokenizer_label(&stt.tokenizer)));

                ui.horizontal(|ui| {
                    ui.label(format!(
                        "Audio: {}",
                        self.selected_audio_name
                            .as_deref()
                            .unwrap_or("<none selected>")
                    ));

                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        if ui.button("Choose WAV...").clicked() {
                            self.load_audio_file_native();
                        }
                    }

                    #[cfg(target_arch = "wasm32")]
                    {
                        if self.pending_web_audio_pick.is_some() {
                            ui.spinner();
                            ui.label("Waiting for file...");
                        } else if ui.button("Choose WAV...").clicked() {
                            self.pending_web_audio_pick = Some(start_audio_file_pick_web());
                        }
                    }

                    if ui.button("Clear").clicked() {
                        self.selected_audio_name = None;
                        self.selected_audio_bytes = None;
                    }
                });

                ui.horizontal(|ui| {
                    if let Some(request_id) = self.pending_request.as_ref().map(|x| x.request_id) {
                        ui.spinner();
                        ui.label("Transcribing...");
                        if ui.button("Cancel").clicked() {
                            server_request_manager.cancel_request(request_id);
                            self.pending_request = None;
                            self.progress_widget_state.clear();
                            self.status_message = Some("Cancelled".to_string());
                        }
                    } else if ui.button("Transcribe").clicked() {
                        self.run_transcription(
                            stt,
                            interface.model_ids.clone(),
                            server_request_manager,
                        );
                    }
                });

                if let Some(msg) = &self.status_message {
                    ui.label(msg);
                }
                if !self.progress_widget_state.is_empty() {
                    self.progress_widget_state.show(ui);
                }

                if let Some(tokens) = &self.transcription_tokens {
                    ui.label(format!("Tokens: {}", tokens.len()));
                }

                if let Some(text) = &self.transcription_text {
                    let mut text = text.clone();
                    ui.label("Transcription:");
                    ui.add(
                        egui::TextEdit::multiline(&mut text)
                            .interactive(false)
                            .desired_rows(8),
                    );
                }
            }
        } else if self.selected_interface_id.is_some() {
            ui.label("Selected interface no longer available.");
            self.selected_interface_id = None;
        }
    }

    fn handle_pending_response(
        &mut self,
        loaded_tokenizers: &mut LoadedTokenizers,
        server_request_manager: &mut ServerRequestManager,
    ) {
        let Some(pending) = self.pending_request.clone() else {
            return;
        };
        let Some(response) = server_request_manager.get_response(pending.request_id) else {
            return;
        };

        self.pending_request = None;
        match response.result {
            Ok(mut data) => {
                let Some(token_tensor) = data.tensor_outputs.remove(&pending.output_link) else {
                    self.status_message = Some("Error: output token tensor not found".to_string());
                    self.transcription_text = None;
                    self.transcription_tokens = None;
                    return;
                };

                let token_tensor = match token_tensor.cast(DType::U32) {
                    Ok(x) => x,
                    Err(err) => {
                        self.status_message = Some(format!("Error: token cast failed: {err}"));
                        self.transcription_text = None;
                        self.transcription_tokens = None;
                        return;
                    }
                };
                let mut token_ids: Vec<u32> = match token_tensor.flatten().try_to_vec() {
                    Ok(x) => x,
                    Err(err) => {
                        self.status_message = Some(format!("Error: token decode failed: {err}"));
                        self.transcription_text = None;
                        self.transcription_tokens = None;
                        return;
                    }
                };
                if let Some(pos) = token_ids
                    .iter()
                    .position(|&token| token == pending.eos_token_id)
                {
                    token_ids.truncate(pos);
                }
                self.transcription_tokens = Some(token_ids.clone());

                let tokenizer = loaded_tokenizers
                    .loaded_tokenizers
                    .get(&pending.tokenizer)
                    .cloned()
                    .flatten();
                match tokenizer {
                    Some(Ok(tokenizer)) => match tokenizer.decode(&token_ids) {
                        Ok(text) => {
                            self.status_message = Some(format!(
                                "Transcription complete ({} tokens)",
                                token_ids.len()
                            ));
                            self.transcription_text = Some(text);
                        }
                        Err(err) => {
                            self.status_message =
                                Some(format!("Token decode failed: {err} (raw tokens are shown)"));
                            self.transcription_text = None;
                        }
                    },
                    Some(Err(err)) => {
                        self.status_message = Some(format!(
                            "Tokenizer load failed: {err} (raw tokens are shown)"
                        ));
                        self.transcription_text = None;
                    }
                    None => {
                        self.status_message =
                            Some("Tokenizer not loaded yet (raw tokens are shown)".to_string());
                        self.transcription_text = None;
                    }
                }
            }
            Err(err) => {
                self.status_message = Some(format!("Error: {err}"));
                self.transcription_text = None;
                self.transcription_tokens = None;
            }
        }
    }

    fn run_transcription(
        &mut self,
        stt: &SpeechToTextInterface,
        model_ids: Vec<whisper_tensor_server::LoadedModelId>,
        server_request_manager: &mut ServerRequestManager,
    ) {
        self.progress_widget_state.clear();
        let Some(audio_bytes) = self.selected_audio_bytes.as_ref() else {
            self.status_message = Some("Select a WAV file first.".to_string());
            return;
        };

        let samples = match decode_wav_bytes_to_mono_f32(audio_bytes, stt.sample_rate) {
            Ok(x) => x,
            Err(err) => {
                self.status_message = Some(format!("Failed to decode WAV: {err}"));
                return;
            }
        };

        if samples.is_empty() {
            self.status_message = Some("Audio file has no samples.".to_string());
            return;
        }

        if model_ids.len() < 2 {
            self.status_message = Some(format!(
                "STT interface expected 2 model IDs (encoder+decoder), found {}",
                model_ids.len()
            ));
            return;
        }

        let audio_tensor = NDArrayNumericTensor::<DynRank>::from_vec_shape(
            samples.clone(),
            &vec![samples.len() as u64],
        )
        .unwrap();

        let request_id = server_request_manager.submit_supergraph_request(SuperGraphRequest {
            do_node_execution_reports: false,
            abbreviated_tensor_report_settings: None,
            attention_token: None,
            super_graph: stt.super_graph.clone(),
            subscribed_tensors: Vec::new(),
            string_inputs: HashMap::new(),
            use_cache: None,
            backend_mode: SuperGraphRequestBackendMode::NDArray,
            symbolic_graph_ids: model_ids.clone(),
            tensor_inputs: HashMap::new(),
            audio_inputs: HashMap::from([(
                stt.audio_input_link,
                SuperGraphAudioInput {
                    samples: audio_tensor,
                    sample_rate_hz: stt.sample_rate,
                },
            )]),
            model_inputs: HashMap::from([
                (stt.encoder_weights_link, model_ids[0]),
                (stt.decoder_weights_link, model_ids[1]),
            ]),
            hash_inputs: HashMap::new(),
        });

        self.pending_request = Some(PendingSttRequest {
            request_id,
            output_link: stt.output_token_link,
            eos_token_id: stt.eos_token_id,
            tokenizer: stt.tokenizer.clone(),
        });
        self.status_message = Some("Running STT pipeline...".to_string());
        self.transcription_text = None;
        self.transcription_tokens = None;
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn load_audio_file_native(&mut self) {
        match pick_audio_file_native() {
            Ok(Some(file)) => {
                self.selected_audio_name = Some(file.name.clone());
                self.selected_audio_bytes = Some(file.bytes);
                self.status_message = Some(format!("Loaded audio file: {}", file.name));
                self.transcription_text = None;
                self.transcription_tokens = None;
            }
            Ok(None) => {}
            Err(err) => {
                self.status_message = Some(format!("Failed to load audio file: {err}"));
            }
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn poll_audio_file_upload_web(&mut self) {
        let Some(receiver) = self.pending_web_audio_pick.as_mut() else {
            return;
        };

        match receiver.try_recv() {
            Ok(Ok(file)) => {
                self.selected_audio_name = Some(file.name.clone());
                self.selected_audio_bytes = Some(file.bytes);
                self.status_message = Some(format!("Loaded audio file: {}", file.name));
                self.transcription_text = None;
                self.transcription_tokens = None;
                self.pending_web_audio_pick = None;
            }
            Ok(Err(err)) => {
                if err != "file selection canceled" {
                    self.status_message = Some(format!("Failed to load audio file: {err}"));
                }
                self.pending_web_audio_pick = None;
            }
            Err(tokio::sync::mpsc::error::TryRecvError::Empty) => {}
            Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => {
                self.pending_web_audio_pick = None;
            }
        }
    }
}

fn tokenizer_label(info: &TokenizerInfo) -> String {
    match info {
        TokenizerInfo::HFTokenizer(name) => format!("Huggingface: {name}"),
        TokenizerInfo::HFTokenizerLocal(path) => format!("Local: {path}"),
        TokenizerInfo::RWKVWorld => "RWKV World".to_string(),
        TokenizerInfo::HFTokenizerJson(_) => "GGUF embedded".to_string(),
    }
}
