use std::cmp::{max, Ordering};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;
use egui::{vec2, Color32, CursorIcon, Event, EventFilter, Label, Layout, Margin, Rect, Response, RichText, Sense, Shape, Stroke, StrokeKind, UiBuilder, Vec2, Widget, WidgetText};
use egui::epaint::{CubicBezierShape, QuadraticBezierShape, RectShape};
use strum::IntoEnumIterator;
use web_sys::WebSocket;
use tokio::sync::mpsc;
use futures::SinkExt;
use log::{debug, info};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::error::TryRecvError;
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::js_sys;
use wasm_bindgen_futures::js_sys::ArrayBuffer;
use wasm_bindgen::prelude::*;
use rand::{random, random_range};
use rwkv_tokenizer::WorldTokenizer;
use whisper_tensor::{DynRank};
use whisper_tensor::backends::ndarray_backend::NDArrayNumericTensor;
use whisper_tensor::scalar_info::ScalarInfoTyped;
use whisper_tensor::symbolic_graph::{OperationId, StoredOrNotTensor, SymbolicGraph, TensorId, TensorType};
use whisper_tensor::symbolic_graph::ops::{AnyOperation, Operation};
use whisper_tensor::symbolic_graph::tensor_store::TensorStoreTensorId;
use whisper_tensor::tokenizer::{AnyTokenizer, Tokenizer, TokenizerError};
use whisper_tensor_import::ModelTypeHint;
use whisper_tensor_import::onnx_graph::TokenizerInfo;
use whisper_tensor_server::{CurrentModelsReportEntry, ForwardLogitRequest, LoadedModelId, ModelLoadType, ModelTypeMetadata, WebsocketClientServerMessage, WebsocketServerClientMessage};
use crate::graph_explorer::{GraphExplorerApp, GraphExplorerState, InspectWindow, OpOrTensorId};
use crate::graph_layout::{GraphLayout, GraphLayoutIOOffsets, GraphLayoutLinkData, GraphLayoutLinkId, GraphLayoutLinkType, GraphLayoutNode, GraphLayoutNodeId, GraphLayoutNodeInitData, GraphLayoutNodeType};
use crate::llm_explorer::{LLMExplorerApp, LLMExplorerState};
use crate::widgets::toggle::toggle_ui;

#[derive(Clone, Debug)]
pub(crate) enum ModelLoadState {
    DialogOpen(Option<String>),
    Loading
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
enum SelectedTab {
    Models,
    GraphExplorer,
    LLMExplorer
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct AppState {
    selected_tab: SelectedTab,
    model_to_load_path_text: String,
    model_type_hint_selected: Option<ModelTypeHint>,
    model_load_type_selected: ModelLoadType,

    graph_explorer_state: GraphExplorerState,
    llm_explorer_state: LLMExplorerState,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            selected_tab: SelectedTab::Models,
            model_to_load_path_text: String::new(),
            model_type_hint_selected: None,
            model_load_type_selected: ModelLoadType::Other,
            graph_explorer_state: GraphExplorerState::default(),
            llm_explorer_state: LLMExplorerState::default(),
        }
    }
}

pub(crate) struct LoadedModels {
    pub(crate) model_load_state: Option<ModelLoadState>,
    pub(crate) current_models: Vec<CurrentModelsReportEntry>,
    pub(crate) currently_requesting_model: Option<LoadedModelId>,
}

pub(crate) struct LoadedTokenizers {
    pub(crate) loaded_tokenizers: HashMap<LoadedModelId, Option<Result<Arc<AnyTokenizer>, String>>>
}

impl LoadedTokenizers {
    pub fn new() -> Self {
        Self {
            loaded_tokenizers: HashMap::new()
        }
    }
}

pub struct WebUIApp {

    websocket_server_client_receiver: mpsc::UnboundedReceiver<WebsocketServerClientMessage>,
    websocket_client_server_sender: mpsc::UnboundedSender<WebsocketClientServerMessage>,

    loaded_models: LoadedModels,
    app_state: AppState,
    graph_explorer_app: GraphExplorerApp,
    llm_explorer_app: LLMExplorerApp,
    loaded_tokenizers: LoadedTokenizers,

}

impl WebUIApp {
    /// Called once before the first frame.
    pub fn new(cc: &eframe::CreationContext<'_>,
               websocket_server_client_receiver: mpsc::UnboundedReceiver<WebsocketServerClientMessage>,
               websocket_client_server_sender: mpsc::UnboundedSender<WebsocketClientServerMessage>) -> Self {
        // This is also where you can customize the look and feel of egui using
        // `cc.egui_ctx.set_visuals` and `cc.egui_ctx.set_fonts`.
        cc.egui_ctx.set_zoom_factor(1.2);

        // Load previous app state (if any).
        // Note that you must enable the `persistence` feature for this to work.
        let app_state = if let Some(storage) = cc.storage {
            eframe::get_value(storage, eframe::APP_KEY).unwrap_or_default()
        } else {
            Default::default()
        };

        Self {
            loaded_models: LoadedModels{
                model_load_state: None,
                current_models: vec![],
                currently_requesting_model: None,
            },
            websocket_server_client_receiver,
            websocket_client_server_sender,
            app_state,
            graph_explorer_app: GraphExplorerApp::new(),
            loaded_tokenizers: LoadedTokenizers::new(),
            llm_explorer_app: LLMExplorerApp::new(),
        }
    }
}


impl eframe::App for WebUIApp {
    /// Called by the frame work to save state before shutdown.
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, &self.app_state);
    }

    /// Called each time the UI needs repainting, which may be many times per second.
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Put your widgets into a `SidePanel`, `TopBottomPanel`, `CentralPanel`, `Window` or `Area`.
        // For inspiration and more examples, go to https://emilk.github.io/egui

        loop {
            match self.websocket_server_client_receiver.try_recv() {
                Ok(msg) => {
                    match msg {
                        WebsocketServerClientMessage::ModelLoadReturn(res) => {
                            match res {
                                Ok(_) => {
                                    self.loaded_models.model_load_state = None;
                                }
                                Err(err) => {
                                    if self.loaded_models.model_load_state.is_some() {
                                        self.loaded_models.model_load_state = Some(ModelLoadState::DialogOpen(Some(err)))
                                    }
                                }
                            }
                        }
                        WebsocketServerClientMessage::CurrentModelsReport(res) => {
                            self.loaded_models.current_models = res;
                            // Prompt tokenizer loading
                            for model in &self.loaded_models.current_models {
                                if let ModelTypeMetadata::LLM(llm_metadata) = &model.model_type_metadata {
                                    match &llm_metadata.tokenizer_info {
                                        TokenizerInfo::HFTokenizer(x) => {
                                            if !self.loaded_tokenizers.loaded_tokenizers.contains_key(&model.model_id) {
                                                self.websocket_client_server_sender.send(WebsocketClientServerMessage::GetHFTokenizer(x.clone())).unwrap();
                                                self.loaded_tokenizers.loaded_tokenizers.insert(model.model_id, None);
                                            }
                                        }
                                        TokenizerInfo::RWKVWorld => {
                                            if !self.loaded_tokenizers.loaded_tokenizers.contains_key(&model.model_id) {
                                                self.loaded_tokenizers.loaded_tokenizers.insert(model.model_id, Some(Ok(Arc::new(AnyTokenizer::Rwkv(WorldTokenizer::new_default())))));
                                            }
                                        }
                                    }

                                }
                            }
                        }
                        WebsocketServerClientMessage::ModelGraphReturn(res) => {
                            let (id, graph_bin) = res.unwrap();
                            if let Some(requesting_id) = self.loaded_models.currently_requesting_model {
                                if requesting_id == id {
                                    self.loaded_models.currently_requesting_model = None;
                                    let graph = ciborium::from_reader::<SymbolicGraph, _>(graph_bin.as_slice()).unwrap();
                                    self.graph_explorer_app.loaded_model_graph = Some((id, graph))
                                }
                            }
                        }
                        WebsocketServerClientMessage::TensorStoreReturn(model_id, stored_tensor_id, res) => {
                            if let Some(selected_model_id) = self.app_state.graph_explorer_state.selected_model_id {
                                if selected_model_id == model_id {
                                    for window in &mut self.graph_explorer_app.inspect_windows {
                                        if let InspectWindow::Tensor(inspect_window_tensor) = window {
                                            if let Some(x) = inspect_window_tensor.stored_value_requested {
                                                if x == stored_tensor_id {
                                                    inspect_window_tensor.stored_value = Some(res.clone());
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        WebsocketServerClientMessage::HFTokenizerReturn(hf_name, bytes_res) => {
                            let tokenizer = match bytes_res {
                                Ok(x) => {
                                    tokenizers::Tokenizer::from_bytes(x).map_err(|x| x.to_string()).map(|x|
                                        Arc::new(AnyTokenizer::Tokenizers(x))
                                    )
                                }
                                Err(err) => {
                                    Err(err)
                                }
                            };

                            for model_entry in &self.loaded_models.current_models {
                                if let ModelTypeMetadata::LLM(x) = &model_entry.model_type_metadata {
                                    if let TokenizerInfo::HFTokenizer(model_hf_name) = &x.tokenizer_info {
                                        if hf_name == *model_hf_name {
                                            if let Some(x) = self.loaded_tokenizers.loaded_tokenizers.get_mut(&model_entry.model_id) {
                                                if x.is_none() {
                                                    *x = Some(tokenizer.clone())
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        WebsocketServerClientMessage::GetLogitsReturn(request, result) => {
                            self.llm_explorer_app.latest_logits = Some((request.model_id, request.context_tokens, result));
                        }
                        _ => {
                            log::debug!("Unhandled message: {:?}", msg);
                        }
                    }
                }
                Err(_) => {
                    break;
                }
            }
        }

        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            // The top panel is often a good place for a menu bar:
            egui::menu::bar(ui, |ui| {
                egui::widgets::global_theme_preference_switch(ui);
                ui.heading("Whisper Tensor");
                ui.selectable_value(&mut self.app_state.selected_tab, SelectedTab::Models, "Manage Models");
                ui.selectable_value(&mut self.app_state.selected_tab, SelectedTab::GraphExplorer, "Graph Explorer");
                ui.selectable_value(&mut self.app_state.selected_tab, SelectedTab::LLMExplorer, "LLM Explorer");
            });
        });

        if let Some(model_load_state) = self.loaded_models.model_load_state.clone() {
            egui::Modal::new(egui::Id::new("Load Model")).show(ctx, |ui| {
                {
                    let spacing_mut = ui.spacing_mut();
                    spacing_mut.item_spacing.x = 10.0;
                    spacing_mut.item_spacing.y = 10.0;
                    spacing_mut.window_margin = Margin::same(50)
                }
                match model_load_state {
                    ModelLoadState::DialogOpen(err) => {
                        ui.label("Load Model");
                        if let Some(err) = err {
                            ui.scope(|ui| {
                                ui.visuals_mut().override_text_color = Some(egui::Color32::RED);
                                ui.style_mut().override_text_style = Some(egui::TextStyle::Monospace);
                                ui.label(err);
                            });
                        }
                        ui.horizontal(|ui| {
                            ui.label("Path: ");
                            ui.text_edit_singleline(&mut self.app_state.model_to_load_path_text);
                            egui::ComboBox::from_id_salt(1245).selected_text(if let Some(which) = &self.app_state.model_type_hint_selected {which.to_string()} else {String::from("No Hint")})
                                .show_ui(ui, |ui|{
                                    ui.selectable_value(&mut self.app_state.model_type_hint_selected, None, "No Hint");
                                    for which in ModelTypeHint::iter() {
                                        ui.selectable_value(&mut self.app_state.model_type_hint_selected, Some(which.clone()), which.to_string());
                                    }
                                });
                            egui::ComboBox::from_id_salt(1246).selected_text(format!("{}", &self.app_state.model_load_type_selected))
                                .show_ui(ui, |ui|{
                                    for which in ModelLoadType::iter() {
                                        ui.selectable_value(&mut self.app_state.model_load_type_selected, which.clone(), which.to_string());
                                    }
                                });
                        });
                        ui.horizontal(|ui| {
                            if ui.button("Load").clicked() {
                                self.websocket_client_server_sender.send(WebsocketClientServerMessage::LoadModel {
                                    model_path: self.app_state.model_to_load_path_text.clone(),
                                    model_type_hint: self.app_state.model_type_hint_selected.clone(),
                                    model_load_type: self.app_state.model_load_type_selected.clone(),
                                }).unwrap();
                                self.loaded_models.model_load_state = Some(ModelLoadState::Loading)
                            }
                            if ui.button("Cancel").clicked() {
                                self.loaded_models.model_load_state = None;
                            }
                        });
                    }
                    ModelLoadState::Loading => {
                        ui.vertical_centered(|ui| {
                            ui.label("Loading Model");
                            ui.spinner();
                        });
                    }
                }
            });
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            // The central panel the region left after adding TopPanel's and SidePanel's
            match &self.app_state.selected_tab {
                SelectedTab::Models => {
                    if ui.button("Load New model").clicked() {
                        self.loaded_models.model_load_state = Some(ModelLoadState::DialogOpen(None));
                    };
                    ui.vertical(|ui| {
                        ui.label("Loaded Models:");
                        egui::Grid::new("loaded_models").striped(true).show(ui, |ui| {
                            for model in &self.loaded_models.current_models {
                                ui.label(model.model_id.to_string());
                                ui.label(model.model_name.clone());
                                ui.label(format!("Operations: {:?}", model.num_ops));
                                match model.model_type_metadata {
                                    ModelTypeMetadata::Other => {
                                        ui.label("Other");
                                    }
                                    ModelTypeMetadata::LLM(_) => {
                                        ui.label("LLM");
                                    }
                                }
                                if let Some(x) = self.loaded_tokenizers.loaded_tokenizers.get(&model.model_id) {
                                    match x {
                                        None => {
                                            ui.label("Loading tokenizer...");
                                        },
                                        Some(Ok(_)) => {
                                            ui.label("Loaded tokenizer");
                                        },
                                        Some(Err(err)) => {
                                            ui.label(format!("Error loading tokenizer: {}", err));
                                        }
                                    }
                                } else {
                                    ui.label("N/A");
                                }
                                if ui.button("Unload").clicked() {
                                    self.websocket_client_server_sender.send(
                                        WebsocketClientServerMessage::UnloadModel(model.model_id)
                                    ).unwrap();
                                }
                                ui.end_row();
                            }
                        });
                    });
                }
                SelectedTab::GraphExplorer => {
                    self.graph_explorer_app.update(
                        &mut self.app_state.graph_explorer_state,
                        &mut self.loaded_models,
                        &self.websocket_client_server_sender,
                        ui
                    )
                }
                SelectedTab::LLMExplorer => {
                    self.llm_explorer_app.update(
                        &mut self.app_state.llm_explorer_state,
                        &mut self.loaded_models,
                        &mut self.loaded_tokenizers,
                        &self.websocket_client_server_sender,
                        ui);
                }
            }


            /*
            if ui.button("Ping").clicked() {
                self.websocket_client_server_message.send(WebsocketClientServerMessage::Ping);

            };*/
        });



        if let SelectedTab::GraphExplorer = self.app_state.selected_tab {
            self.graph_explorer_app.update_inspect_windows(
                &mut self.app_state.graph_explorer_state,
                ctx,
                &self.websocket_client_server_sender
            )
        }

    }
}

