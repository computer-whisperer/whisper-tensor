use egui::{Color32, CursorIcon, Event, EventFilter, Label, RichText, Sense, Widget};
use log::info;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use whisper_tensor::tokenizer::Tokenizer;
use whisper_tensor_import::onnx_graph::TokenizerInfo;
use whisper_tensor_server::{ForwardLogitRequest, LoadedModelId, ModelTypeMetadata, WebsocketClientServerMessage};
use crate::app::{LoadedModels, LoadedTokenizers, ModelLoadState};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct LLMExplorerState {
    llm_explorer_selected_model_id: Option<LoadedModelId>,
    current_llm_text: String,
}

impl Default for LLMExplorerState {
    fn default() -> Self {
        Self {
            llm_explorer_selected_model_id: None,
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
    pub(crate) latest_logits: Option<(LoadedModelId, Vec<u32>, Result<Vec<Vec<(u32, f32)>>, String>)>
}

impl LLMExplorerApp {
    pub fn new() -> Self {
        Self {
            llm_explorer_cached_token_list: None,
            latest_logits: None
        }
    }

    pub fn update(&mut self,
                  state: &mut LLMExplorerState,
                  loaded_models: &mut LoadedModels,
                  loaded_tokenizers: &mut LoadedTokenizers,
                  websocket_client_server_sender: &mpsc::UnboundedSender<WebsocketClientServerMessage>,
                  ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if loaded_models.current_models.is_empty() {
                ui.label("No Models Loaded");
            }
            for model in &loaded_models.current_models {
                if let ModelTypeMetadata::LLM(_) = model.model_type_metadata {
                    if ui.selectable_value(&mut state.llm_explorer_selected_model_id, Some(model.model_id), format!("({}) {}", model.model_id, model.model_name.clone())).clicked() {
                        // Good!
                        self.llm_explorer_cached_token_list = None;
                    };
                }
            }
            if ui.button("Load New Model").clicked() {
                loaded_models.model_load_state = Some(ModelLoadState::DialogOpen(None));
            };
        });

        let model = if let Some(selected_model_id) =  state.llm_explorer_selected_model_id {
            let mut ret = None;
            for model in &loaded_models.current_models {
                if model.model_id == selected_model_id {
                    ret = Some(model);
                }
            }
            ret
        } else {
            None
        };

        {
            let spacing_mut = ui.spacing_mut();
            spacing_mut.item_spacing.y = 10.0;
        }

        let tokenizer = if let Some(model) = model {
            match loaded_tokenizers.loaded_tokenizers.get(&model.model_id) {
                Some(x) => {
                    x.as_ref()
                },
                _ => { None }
            }
        } else {
            None
        };

        match (model, tokenizer) {
            (Some(model), Some(Ok(tokenizer))) => {
                if let ModelTypeMetadata::LLM(x) = &model.model_type_metadata {
                    let v = match &x.tokenizer_info {
                        TokenizerInfo::HFTokenizer(x) => {
                            format!("Huggingface: {x}")
                        }
                        TokenizerInfo::RWKVWorld => {
                            "RWKV World".to_string()
                        }
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
                        self.llm_explorer_cached_token_list = Some(
                            tokenizer.encode(&state.current_llm_text)
                        );
                    }

                    // Display

                    if let Some(tokens) = &self.llm_explorer_cached_token_list{
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
                                let color = color_wheel[idx%color_wheel.len()];
                                if idx >= tokens.len() && !is_good {
                                    break;
                                }

                                let chosen_token_id = tokens.get(idx);

                                ui.vertical(|ui| {
                                    let mut next_is_good = is_good;
                                    if let Some(chosen_token) = chosen_token_id {
                                        let label_text = match tokenizer.decode(&[*chosen_token]) {
                                            Ok(token_str) => {
                                                escape_token_text(&token_str)
                                            }
                                            Err(_) => {
                                                "(?)".to_string()
                                            }
                                        };

                                        // Token string
                                        let text = RichText::new(label_text).background_color(color).size(20.0);
                                        Label::new(text).ui(ui);

                                        // Token id
                                        let text = RichText::new(chosen_token.to_string()).background_color(color).size(8.0);
                                        Label::new(text).ui(ui);

                                        // Validate logits
                                        if let Some((_a, b, _c)) = &self.latest_logits {
                                            if idx < b.len() {
                                                next_is_good &= b[idx] == *chosen_token
                                            }
                                        }
                                    }
                                    else {
                                        next_is_good = false;
                                    }
                                    {
                                        let spacing_mut = ui.spacing_mut();
                                        spacing_mut.item_spacing.x = 5.0;
                                        spacing_mut.item_spacing.y = 3.0;
                                    }
                                    if is_good {
                                        if let Some((a, b, c)) = &self.latest_logits {
                                            if *a == model.model_id && idx-1 < b.len() {
                                                if let Ok(c) = c {
                                                    let logits = &c[idx-1];
                                                    for j in 0..5 {
                                                        if j >= logits.len() {
                                                            break;
                                                        }
                                                        let (token_id, value) = &logits[j];
                                                        let token_str = tokenizer.decode(&[*token_id]).unwrap_or("?".to_string());
                                                        let token_str = escape_token_text(&token_str);
                                                        //ui.label(format!("{token_id}, {value}, {token_str}"));
                                                        let text = RichText::new(token_str).background_color(Color32::from_rgb(20, 20, 20)).size(15.0);
                                                        ui.label(text);
                                                        //let text = RichText::new(token_id.to_string()).size(8.0);
                                                        //ui.label(text);
                                                        let text = RichText::new(format!("{value:.02}")).size(8.0);
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
                                Event::Key{
                                    key: egui::Key::Backspace,
                                    pressed: true,
                                    ..
                                } => {
                                    state.current_llm_text.remove(state.current_llm_text.len() - 1);
                                    self.llm_explorer_cached_token_list = None;
                                }
                                Event::PointerMoved(_) | Event::PointerGone | Event::Key{..} => {
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
                        self.llm_explorer_cached_token_list = Some(
                            tokenizer.encode(&state.current_llm_text)
                        );
                    }
                    let tokens = self.llm_explorer_cached_token_list.clone().unwrap();
                    let msg = WebsocketClientServerMessage::GetLogits(ForwardLogitRequest{
                        model_id: model.model_id,
                        context_tokens: tokens
                    });
                    websocket_client_server_sender.send(msg).unwrap();
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