use crate::app::{LoadedModels, LoadedTokenizers, ModelLoadState};
use crate::websockets::ServerRequestManager;
use egui::{Color32, CursorIcon, Event, EventFilter, Label, RichText, Sense, Widget};
use log::info;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashMap;
use tokio::sync::mpsc;
use web_sys::js_sys::Atomics::load;
use whisper_tensor::backends::ndarray_backend::NDArrayNumericTensor;
use whisper_tensor::interfaces::AnyInterface;
use whisper_tensor::super_graph::links::SuperGraphLinkTensor;
use whisper_tensor::tokenizer::Tokenizer;
use whisper_tensor_import::onnx_graph::TokenizerInfo;
use whisper_tensor_server::{LoadedModelId, SuperGraphRequest, WebsocketClientServerMessage};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct LLMExplorerState {
    llm_explorer_selected_interface_name: Option<String>,
    current_llm_text: String,
}

impl Default for LLMExplorerState {
    fn default() -> Self {
        Self {
            llm_explorer_selected_interface_name: None,
            current_llm_text: "This is example text".to_string(),
        }
    }
}

fn escape_token_text(input: &str) -> String {
    use std::fmt::Write;

    let mut out = String::with_capacity(input.len());

    for c in input.chars() {
        match c {
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\\' => out.push_str("\\\\"),
            // `is_control` is true for all C0 controls (0x00–0x1F) and DEL (0x7F)
            c if c.is_control() => {
                // \u{XXXX} where XXXX is at least 4 hex digits
                write!(out, "\\u{{{:04X}}}", c as u32).unwrap();
            }
            // Printable, keep as‑is
            c => out.push(c),
        }
    }

    out
}

pub(crate) struct LLMExplorerApp {
    llm_explorer_cached_token_list: Option<Vec<u32>>,
    pending_request: Option<(u64, SuperGraphLinkTensor, Vec<u32>)>,
    pub(crate) latest_logits: Option<(Vec<u32>, Result<Vec<Vec<(u32, f32)>>, String>)>,
}

impl LLMExplorerApp {
    pub fn new() -> Self {
        Self {
            llm_explorer_cached_token_list: None,
            pending_request: None,
            latest_logits: None,
        }
    }

    pub fn update(
        &mut self,
        state: &mut LLMExplorerState,
        loaded_models: &mut LoadedModels,
        loaded_tokenizers: &mut LoadedTokenizers,
        server_request_manager: &mut ServerRequestManager,
        ui: &mut egui::Ui,
    ) {
        // Handle pending requests
        if let Some((request_id, link, tokens)) = self.pending_request.clone() {
            if let Some(mut response) = server_request_manager.get_response(request_id) {
                self.pending_request = None;
                let mut response_tokens = response.tensor_outputs.remove(&link).unwrap();
                let shape = response_tokens.shape();
                let logits_per_token = shape[1];
                let returned_tokens = shape[0];
                let mut outputs = Vec::new();
                for i in 0..returned_tokens as usize {
                    let sliced_output_tensor = response_tokens
                        .slice(&[i..i + 1, 0..logits_per_token as usize])
                        .unwrap();
                    let output = sliced_output_tensor.flatten();
                    let output_vec: Vec<f32> = output.try_into().unwrap();
                    let mut idx_and_val = output_vec
                        .iter()
                        .enumerate()
                        .map(|(a, b)| (a as u32, *b))
                        .collect::<Vec<_>>();
                    idx_and_val.sort_by(|(_, a), (_, b)| {
                        if a < b {
                            Ordering::Greater
                        } else {
                            Ordering::Less
                        }
                    });
                    let clipped_logits = idx_and_val[0..idx_and_val.len().min(100)].to_vec();
                    outputs.push(clipped_logits)
                }
                self.latest_logits = Some((tokens, Ok(outputs)));
            }
        }

        // Find llm interfaces
        let mut llm_interfaces = HashMap::new();
        for (name, interface) in &loaded_models.current_models.interfaces {
            if let AnyInterface::TextInferenceTokensInLogitOutInterface(_x) = &interface.interface {
                llm_interfaces.insert(name, interface);
            }
        }

        ui.horizontal(|ui| {
            if llm_interfaces.is_empty() {
                ui.label("No Models Loaded");
            }
            let mut keys = llm_interfaces.keys().collect::<Vec<_>>();
            keys.sort();
            for &interface_name in keys {
                if ui
                    .selectable_value(
                        &mut state.llm_explorer_selected_interface_name,
                        Some(interface_name.clone()),
                        interface_name,
                    )
                    .clicked()
                {
                    // Good!
                    self.llm_explorer_cached_token_list = None;
                };
            }
            if ui.button("Load New Model").clicked() {
                loaded_models.model_load_state = Some(ModelLoadState::DialogOpen(None));
            };
        });

        let interface = if let Some(selected_model_id) = &state.llm_explorer_selected_interface_name
        {
            loaded_models
                .current_models
                .interfaces
                .get(selected_model_id)
        } else {
            None
        };

        {
            let spacing_mut = ui.spacing_mut();
            spacing_mut.item_spacing.y = 10.0;
        }

        let tokenizer = if let Some(interface) = interface {
            let AnyInterface::TextInferenceTokensInLogitOutInterface(interface) =
                &interface.interface;
            let info = interface.get_tokenizer();
            loaded_tokenizers
                .loaded_tokenizers
                .get(info)
                .cloned()
                .flatten()
        } else {
            None
        };

        match (interface, tokenizer) {
            (Some(interface), Some(Ok(tokenizer))) => {
                let AnyInterface::TextInferenceTokensInLogitOutInterface(llm_interface) =
                    &interface.interface;
                {
                    let v = match &llm_interface.get_tokenizer() {
                        TokenizerInfo::HFTokenizer(x) => {
                            format!("Huggingface: {x}")
                        }
                        TokenizerInfo::RWKVWorld => "RWKV World".to_string(),
                    };
                    ui.label(format!("Using tokenizer: {v}"));
                }
                {
                    let old_text = state.current_llm_text.clone();
                    egui::TextEdit::multiline(&mut state.current_llm_text).ui(ui);
                    if state.current_llm_text != old_text {
                        self.llm_explorer_cached_token_list = None;
                    }
                }
                let frame = egui::Frame::default();
                frame.show(ui, |ui| {
                    let id = egui::Id::new("llm box");

                    // Generate tokens if needed
                    if self.llm_explorer_cached_token_list.is_none() {
                        // Generate it
                        self.llm_explorer_cached_token_list =
                            Some(tokenizer.encode(&state.current_llm_text));
                    }

                    // Display

                    if let Some(tokens) = &self.llm_explorer_cached_token_list {
                        let color_wheel = [
                            Color32::from_rgb(50, 0, 0),
                            Color32::from_rgb(0, 50, 0),
                            Color32::from_rgb(0, 0, 50),
                        ];

                        ui.horizontal_wrapped(|ui| {
                            {
                                let spacing_mut = ui.spacing_mut();
                                spacing_mut.item_spacing.x = 0.0;
                                spacing_mut.item_spacing.y = 2.0;
                            }
                            let mut idx = 0;
                            let mut is_good = true;
                            loop {
                                let color = color_wheel[idx % color_wheel.len()];
                                if idx >= tokens.len() && !is_good {
                                    break;
                                }

                                let chosen_token_id = tokens.get(idx);

                                ui.vertical(|ui| {
                                    let mut next_is_good = is_good;
                                    if let Some(chosen_token) = chosen_token_id {
                                        let label_text = match tokenizer.decode(&[*chosen_token]) {
                                            Ok(token_str) => escape_token_text(&token_str),
                                            Err(_) => "(?)".to_string(),
                                        };

                                        // Token string
                                        let text = RichText::new(label_text)
                                            .background_color(color)
                                            .size(20.0);
                                        Label::new(text).ui(ui);

                                        // Token id
                                        let text = RichText::new(chosen_token.to_string())
                                            .background_color(color)
                                            .size(8.0);
                                        Label::new(text).ui(ui);

                                        // Validate logits
                                        if let Some((b, _c)) = &self.latest_logits {
                                            if idx < b.len() {
                                                next_is_good &= b[idx] == *chosen_token
                                            }
                                        }
                                    } else {
                                        next_is_good = false;
                                    }
                                    {
                                        let spacing_mut = ui.spacing_mut();
                                        spacing_mut.item_spacing.x = 5.0;
                                        spacing_mut.item_spacing.y = 3.0;
                                    }
                                    if is_good {
                                        if let Some((b, c)) = &self.latest_logits {
                                            if idx - 1 < b.len() {
                                                if let Ok(c) = c {
                                                    let logits = &c[idx - 1];
                                                    for j in 0..5 {
                                                        if j >= logits.len() {
                                                            break;
                                                        }
                                                        let (token_id, value) = &logits[j];
                                                        let token_str = tokenizer
                                                            .decode(&[*token_id])
                                                            .unwrap_or("?".to_string());
                                                        let token_str =
                                                            escape_token_text(&token_str);
                                                        //ui.label(format!("{token_id}, {value}, {token_str}"));
                                                        let text = RichText::new(token_str)
                                                            .background_color(Color32::from_rgb(
                                                                20, 20, 20,
                                                            ))
                                                            .size(15.0);
                                                        ui.label(text);
                                                        //let text = RichText::new(token_id.to_string()).size(8.0);
                                                        //ui.label(text);
                                                        let text =
                                                            RichText::new(format!("{value:.02}"))
                                                                .size(8.0);
                                                        ui.label(text);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    is_good = next_is_good;
                                });

                                idx += 1;
                            }
                        });
                    }

                    let mut response = ui.interact(ui.min_rect(), id, Sense::click());

                    if response.hovered() {
                        ui.ctx().set_cursor_icon(CursorIcon::Text);
                    }
                    let event_filter = EventFilter {
                        // moving the cursor is really important
                        horizontal_arrows: true,
                        vertical_arrows: true,
                        tab: false, // tab is used to change focus, not to insert a tab character
                        ..Default::default()
                    };
                    if ui.memory(|mem| mem.has_focus(id)) {
                        ui.memory_mut(|mem| mem.set_focus_lock_filter(id, event_filter));

                        let events = ui.input(|i| i.filtered_events(&event_filter));
                        for event in events {
                            match event {
                                Event::Text(x) => {
                                    state.current_llm_text.push_str(&x);
                                    self.llm_explorer_cached_token_list = None;
                                }
                                Event::Key {
                                    key: egui::Key::Backspace,
                                    pressed: true,
                                    ..
                                } => {
                                    state
                                        .current_llm_text
                                        .remove(state.current_llm_text.len() - 1);
                                    self.llm_explorer_cached_token_list = None;
                                }
                                Event::PointerMoved(_) | Event::PointerGone | Event::Key { .. } => {
                                    // Don't care
                                }
                                _ => {
                                    info!("Unhandled event: {event:?}")
                                }
                            }
                        }
                    }
                    if response.clicked() {
                        ui.memory_mut(|mem| mem.request_focus(response.id));
                    }
                });
                if ui.button("Run").clicked() {
                    // Generate tokens if needed
                    if self.llm_explorer_cached_token_list.is_none() {
                        // Generate it
                        self.llm_explorer_cached_token_list =
                            Some(tokenizer.encode(&state.current_llm_text));
                    }
                    let tokens = self.llm_explorer_cached_token_list.clone().unwrap();
                    let tokens_tensor = NDArrayNumericTensor::from_vec(tokens.clone()).to_dyn();
                    let token =
                        server_request_manager.submit_supergraph_request(SuperGraphRequest {
                            attention_token: None,
                            super_graph: llm_interface.super_graph.clone(),
                            string_inputs: HashMap::new(),
                            tensor_inputs: HashMap::from([(
                                llm_interface.token_context_input_link.clone(),
                                tokens_tensor,
                            )]),
                            model_inputs: HashMap::from([(
                                llm_interface.model_input_link.clone(),
                                interface.model_ids.first().unwrap().clone(),
                            )]),
                            hash_inputs: HashMap::from([(
                                llm_interface.cache_key_input_link.clone(),
                                0u64,
                            )]),
                        });
                    self.pending_request =
                        Some((token, llm_interface.logit_output_link.clone(), tokens));
                    self.latest_logits = None;
                }
            }
            (Some(_), Some(Err(err))) => {
                ui.label(format!("Tokenizer load error: {err}."));
            }
            _ => {
                ui.label("Invalid model selected.");
            }
        }
    }
}
