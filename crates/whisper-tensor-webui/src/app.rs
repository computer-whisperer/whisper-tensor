use crate::graph_explorer::inspect_windows::InspectWindow;
use crate::graph_explorer::{GraphExplorerApp, GraphExplorerLayerSelection, GraphExplorerState};
use crate::llm_explorer::{LLMExplorerApp, LLMExplorerState};
use crate::websockets;
use crate::websockets::ServerRequestManager;
use crate::widgets::toggle::toggle_ui;
use egui::epaint::{CubicBezierShape, QuadraticBezierShape, RectShape};
use egui::{
    Color32, CursorIcon, Event, EventFilter, Label, Layout, Margin, Rect, Response, RichText,
    Sense, Shape, Stroke, StrokeKind, UiBuilder, Vec2, Widget, WidgetText, vec2,
};
use futures::SinkExt;
use log::{debug, info};
use rand::{random, random_range};
use rwkv_tokenizer::WorldTokenizer;
use serde::{Deserialize, Serialize};
use std::cmp::{Ordering, max};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;
use strum::IntoEnumIterator;
use tokio::sync::mpsc;
use tokio::sync::mpsc::error::TryRecvError;
use wasm_bindgen::prelude::*;
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::js_sys;
use wasm_bindgen_futures::js_sys::ArrayBuffer;
use web_sys::WebSocket;
use whisper_tensor::DynRank;
use whisper_tensor::backends::ndarray_backend::NDArrayNumericTensor;
use whisper_tensor::interfaces::AnyInterface;
use whisper_tensor::scalar_info::ScalarInfoTyped;
use whisper_tensor::symbolic_graph::ops::{AnyOperation, Operation};
use whisper_tensor::symbolic_graph::tensor_store::TensorStoreTensorId;
use whisper_tensor::symbolic_graph::{
    StoredOrNotTensor, SymbolicGraph, SymbolicGraphOperationId, SymbolicGraphTensorId, TensorType,
};
use whisper_tensor::tokenizer::{AnyTokenizer, Tokenizer, TokenizerError};
use whisper_tensor_import::ModelTypeHint;
use whisper_tensor_import::onnx_graph::TokenizerInfo;
use whisper_tensor_server::{
    CurrentInterfacesReportEntry, CurrentModelsAndInterfacesReport, CurrentModelsReportEntry,
    LoadedModelId, ServerConfigReport, WebsocketClientServerMessage, WebsocketServerClientMessage,
};

#[derive(Clone, Debug)]
pub(crate) enum ModelLoadState {
    DialogOpen(Option<String>),
    Loading,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
enum SelectedTab {
    Models,
    GraphExplorer,
    LLMExplorer,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct AppState {
    selected_tab: SelectedTab,
    model_to_load_path_text: String,
    model_type_hint_selected: Option<ModelTypeHint>,

    graph_explorer_state: GraphExplorerState,
    llm_explorer_state: LLMExplorerState,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            selected_tab: SelectedTab::Models,
            model_to_load_path_text: String::new(),
            model_type_hint_selected: None,
            graph_explorer_state: GraphExplorerState::default(),
            llm_explorer_state: LLMExplorerState::default(),
        }
    }
}

pub(crate) type InterfaceId = u32;

pub(crate) struct LoadedModels {
    pub(crate) model_load_state: Option<ModelLoadState>,
    pub(crate) current_models: Vec<CurrentModelsReportEntry>,
    pub(crate) current_interfaces: HashMap<InterfaceId, CurrentInterfacesReportEntry>,
    pub(crate) currently_requesting_model: Option<LoadedModelId>,
}

pub(crate) struct LoadedTokenizers {
    pub(crate) loaded_tokenizers: HashMap<TokenizerInfo, Option<Result<Arc<AnyTokenizer>, String>>>,
}

impl LoadedTokenizers {
    pub fn new() -> Self {
        Self {
            loaded_tokenizers: HashMap::new(),
        }
    }
}

pub struct WebUIApp {
    websocket_server_client_receiver: mpsc::UnboundedReceiver<WebsocketServerClientMessage>,
    server_request_manager: ServerRequestManager,
    next_interface_id: InterfaceId,
    loaded_models: LoadedModels,
    app_state: AppState,
    graph_explorer_app: GraphExplorerApp,
    llm_explorer_app: LLMExplorerApp,
    loaded_tokenizers: LoadedTokenizers,
    server_config_report: Option<ServerConfigReport>,
}

impl WebUIApp {
    /// Called once before the first frame.
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // This is also where you can customize the look and feel of egui using
        // `cc.egui_ctx.set_visuals` and `cc.egui_ctx.set_fonts`.
        cc.egui_ctx.set_zoom_factor(1.2);

        let (server_client_sender, server_client_receiver) = mpsc::unbounded_channel();
        let (client_server_sender, client_server_receiver) = mpsc::unbounded_channel();

        let ctx = cc.egui_ctx.clone();
        wasm_bindgen_futures::spawn_local(async {
            websockets::websocket_task(server_client_sender, client_server_receiver, ctx).await;
        });
        // Load previous app state (if any).
        // Note that you must enable the `persistence` feature for this to work.
        let app_state = if let Some(storage) = cc.storage {
            eframe::get_value(storage, eframe::APP_KEY).unwrap_or_default()
        } else {
            Default::default()
        };

        Self {
            loaded_models: LoadedModels {
                model_load_state: None,
                current_models: Vec::new(),
                current_interfaces: HashMap::new(),
                currently_requesting_model: None,
            },
            server_request_manager: ServerRequestManager::new(client_server_sender),
            next_interface_id: 0,
            websocket_server_client_receiver: server_client_receiver,
            app_state,
            graph_explorer_app: GraphExplorerApp::new(),
            loaded_tokenizers: LoadedTokenizers::new(),
            llm_explorer_app: LLMExplorerApp::new(),
            server_config_report: None,
        }
    }
}

impl eframe::App for WebUIApp {
    /// Called each time the UI needs repainting, which may be many times per second.
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Put your widgets into a `SidePanel`, `TopBottomPanel`, `CentralPanel`, `Window` or `Area`.
        // For inspiration and more examples, go to https://emilk.github.io/egui

        loop {
            match self.websocket_server_client_receiver.try_recv() {
                Ok(msg) => {
                    match msg {
                        WebsocketServerClientMessage::ModelLoadReturn(res) => match res {
                            Ok(_) => {
                                self.loaded_models.model_load_state = None;
                            }
                            Err(err) => {
                                if self.loaded_models.model_load_state.is_some() {
                                    self.loaded_models.model_load_state =
                                        Some(ModelLoadState::DialogOpen(Some(err)))
                                }
                            }
                        },
                        WebsocketServerClientMessage::CurrentModelsReport(res) => {
                            self.loaded_models.current_models = res.models;
                            // Rebuild interfaces list
                            self.loaded_models.current_interfaces = {
                                let mut new_interfaces = HashMap::new();
                                for interface in res.interfaces {
                                    let mut found = false;
                                    for (&id, existing_interface) in
                                        &self.loaded_models.current_interfaces
                                    {
                                        if interface.interface_name
                                            == existing_interface.interface_name
                                        {
                                            new_interfaces.insert(id, existing_interface.clone());
                                            found = true;
                                            break;
                                        }
                                    }
                                    if !found {
                                        new_interfaces.insert(self.next_interface_id, interface);
                                        self.next_interface_id += 1;
                                    }
                                }
                                new_interfaces
                            };
                            // Prompt tokenizer loading
                            let mut needed_tokenizers = Vec::new();
                            for interface in self.loaded_models.current_interfaces.values() {
                                if let AnyInterface::TextInferenceTokensInLogitOutInterface(
                                    interface,
                                ) = &interface.interface
                                {
                                    needed_tokenizers.push(interface.get_tokenizer().clone());
                                }
                            }
                            for tokenizer_info in needed_tokenizers {
                                if !self
                                    .loaded_tokenizers
                                    .loaded_tokenizers
                                    .contains_key(&tokenizer_info)
                                {
                                    match &tokenizer_info {
                                        TokenizerInfo::HFTokenizer(x) => {
                                            self.server_request_manager
                                                .send(WebsocketClientServerMessage::GetHFTokenizer(
                                                    x.clone(),
                                                ))
                                                .unwrap();
                                            self.loaded_tokenizers
                                                .loaded_tokenizers
                                                .insert(tokenizer_info, None);
                                        }
                                        TokenizerInfo::RWKVWorld => {
                                            self.loaded_tokenizers.loaded_tokenizers.insert(
                                                tokenizer_info,
                                                Some(Ok(Arc::new(AnyTokenizer::Rwkv(
                                                    WorldTokenizer::new_default(),
                                                )))),
                                            );
                                        }
                                    }
                                }
                            }
                        }
                        WebsocketServerClientMessage::ModelGraphReturn(res) => {
                            let (id, graph_bin) = res.unwrap();
                            if let Some(requesting_id) =
                                self.loaded_models.currently_requesting_model
                            {
                                if requesting_id == id {
                                    self.loaded_models.currently_requesting_model = None;
                                    let graph = ciborium::from_reader::<SymbolicGraph, _>(
                                        graph_bin.as_slice(),
                                    )
                                    .unwrap();
                                    self.graph_explorer_app.loaded_models.insert(id, graph);
                                }
                            }
                        }
                        WebsocketServerClientMessage::TensorStoreReturn(
                            model_id,
                            stored_tensor_id,
                            res,
                        ) => {
                            if let Some(selected_model_id) =
                                self.graph_explorer_app.get_model_scope(&self.loaded_models)
                            {
                                if selected_model_id == model_id {
                                    for window in &mut self.graph_explorer_app.inspect_windows {
                                        if let InspectWindow::SymbolicGraphTensor(
                                            inspect_window_tensor,
                                        ) = window
                                        {
                                            if let Some(x) =
                                                inspect_window_tensor.stored_value_requested
                                            {
                                                if x == stored_tensor_id {
                                                    inspect_window_tensor.stored_value =
                                                        Some(res.clone());
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        WebsocketServerClientMessage::HFTokenizerReturn(hf_name, bytes_res) => {
                            let tokenizer = match bytes_res {
                                Ok(x) => tokenizers::Tokenizer::from_bytes(x)
                                    .map_err(|x| x.to_string())
                                    .map(|x| Arc::new(AnyTokenizer::Tokenizers(x))),
                                Err(err) => Err(err),
                            };

                            self.loaded_tokenizers.loaded_tokenizers.insert(
                                TokenizerInfo::HFTokenizer(hf_name.clone()),
                                Some(tokenizer.clone()),
                            );
                        }
                        WebsocketServerClientMessage::SuperGraphResponse(response) => {
                            self.server_request_manager.new_response(response);
                        }
                        WebsocketServerClientMessage::SuperGraphExecutionReport(report) => {
                            self.server_request_manager.new_execution_report(report);
                        }
                        WebsocketServerClientMessage::ServerConfigReport(config) => {
                            self.server_config_report = Some(config);
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
                ui.selectable_value(
                    &mut self.app_state.selected_tab,
                    SelectedTab::Models,
                    "Manage Models",
                );
                ui.selectable_value(
                    &mut self.app_state.selected_tab,
                    SelectedTab::GraphExplorer,
                    "Graph Explorer",
                );
                ui.selectable_value(
                    &mut self.app_state.selected_tab,
                    SelectedTab::LLMExplorer,
                    "LLM Explorer",
                );
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
                                ui.style_mut().override_text_style =
                                    Some(egui::TextStyle::Monospace);
                                ui.label(err);
                            });
                        }
                        ui.horizontal(|ui| {
                            ui.label("Path: ");
                            ui.text_edit_singleline(&mut self.app_state.model_to_load_path_text);
                            egui::ComboBox::from_id_salt(1245)
                                .selected_text(
                                    if let Some(which) = &self.app_state.model_type_hint_selected {
                                        which.to_string()
                                    } else {
                                        String::from("No Hint")
                                    },
                                )
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(
                                        &mut self.app_state.model_type_hint_selected,
                                        None,
                                        "No Hint",
                                    );
                                    for which in ModelTypeHint::iter() {
                                        ui.selectable_value(
                                            &mut self.app_state.model_type_hint_selected,
                                            Some(which.clone()),
                                            which.to_string(),
                                        );
                                    }
                                });
                        });
                        ui.horizontal(|ui| {
                            if ui.button("Load").clicked() {
                                self.server_request_manager
                                    .send(WebsocketClientServerMessage::LoadModel {
                                        model_path: self.app_state.model_to_load_path_text.clone(),
                                        model_type_hint: self
                                            .app_state
                                            .model_type_hint_selected
                                            .clone(),
                                    })
                                    .unwrap();
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
                        self.loaded_models.model_load_state =
                            Some(ModelLoadState::DialogOpen(None));
                    };
                    ui.vertical(|ui| {
                        ui.label("Loaded Models:");
                        egui::Grid::new("loaded_models")
                            .striped(true)
                            .show(ui, |ui| {
                                for model in &self.loaded_models.current_models {
                                    ui.label(model.model_id.to_string());
                                    ui.label(model.model_name.clone());
                                    ui.label(format!("Operations: {:?}", model.num_ops));
                                    ui.label(format!("Compiled: {:}", model.model_compiled));
                                    if ui.button("Unload").clicked() {
                                        self.server_request_manager
                                            .send(WebsocketClientServerMessage::UnloadModel(
                                                model.model_id,
                                            ))
                                            .unwrap();
                                    }
                                    if ui.button("Compile").clicked() {
                                        self.server_request_manager
                                            .send(WebsocketClientServerMessage::CompileModel(
                                                model.model_id,
                                            ))
                                            .unwrap();
                                    }
                                    ui.end_row();
                                }
                            });
                    });
                }
                SelectedTab::GraphExplorer => {
                    if let Some(server_config_report) = &self.server_config_report {
                        self.graph_explorer_app.update(
                            &mut self.app_state.graph_explorer_state,
                            &mut self.loaded_models,
                            &mut self.loaded_tokenizers,
                            &mut self.server_request_manager,
                            server_config_report,
                            ui,
                        )
                    }
                }
                SelectedTab::LLMExplorer => {
                    self.llm_explorer_app.update(
                        &mut self.app_state.llm_explorer_state,
                        &mut self.loaded_models,
                        &mut self.loaded_tokenizers,
                        &mut self.server_request_manager,
                        ui,
                    );
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
                &mut self.loaded_models,
                &mut self.server_request_manager,
            )
        }
    }

    /// Called by the frame work to save state before shutdown.
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, &self.app_state);
    }
}
