use crate::graph_explorer::{GraphExplorerApp, GraphExplorerSettings, GraphRootSubjectSelection};
use crate::llm_explorer::{LLMExplorerApp, LLMExplorerState};
use crate::sd_explorer::{SDExplorerApp, SDExplorerState};
use crate::websockets::ServerRequestManager;
use crate::widgets::toggle::toggle_ui;
use egui::Margin;
use rwkv_tokenizer::WorldTokenizer;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::mpsc;
use whisper_tensor::interfaces::AnyInterface;
use whisper_tensor::loader::{ConfigFieldType, ConfigValue};
use whisper_tensor::metadata::TokenizerInfo;
use whisper_tensor::symbolic_graph::SymbolicGraph;
use whisper_tensor::tokenizer::AnyTokenizer;
use whisper_tensor_server::{
    CurrentInterfacesReportEntry, CurrentModelsReportEntry, LoadedModelId, LoaderRegistryReport,
    ServerConfigReport, WebsocketClientServerMessage, WebsocketServerClientMessage,
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
    SDExplorer,
}

/// Persisted state for the loader dialog's config field values.
/// Keyed by loader_index, then field key → string value.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct LoaderDialogState {
    selected_loader: usize,
    field_values: HashMap<usize, HashMap<String, String>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct AppState {
    selected_tab: SelectedTab,

    loader_dialog: LoaderDialogState,

    graph_explorer_settings: GraphExplorerSettings,
    llm_explorer_state: LLMExplorerState,
    sd_explorer_state: SDExplorerState,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            selected_tab: SelectedTab::Models,
            loader_dialog: LoaderDialogState::default(),
            graph_explorer_settings: GraphExplorerSettings::default(),
            llm_explorer_state: LLMExplorerState::default(),
            sd_explorer_state: SDExplorerState::default(),
        }
    }
}

pub(crate) type InterfaceId = u32;

pub(crate) struct LoadedModels {
    pub(crate) model_load_state: Option<ModelLoadState>,
    pub(crate) current_models: Vec<CurrentModelsReportEntry>,
    pub(crate) current_interfaces: HashMap<InterfaceId, CurrentInterfacesReportEntry>,
    pub(crate) currently_requesting_model: Option<LoadedModelId>,
    pub(crate) loaded_models: HashMap<LoadedModelId, SymbolicGraph>,
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
    selected_graph_explorer_tab: Option<GraphRootSubjectSelection>,
    graph_explorer_app: HashMap<GraphRootSubjectSelection, GraphExplorerApp>,
    llm_explorer_app: LLMExplorerApp,
    sd_explorer_app: SDExplorerApp,
    loaded_tokenizers: LoadedTokenizers,
    server_config_report: Option<ServerConfigReport>,
    loader_registry: Option<LoaderRegistryReport>,
}

impl WebUIApp {
    /// Called once before the first frame.
    pub fn new(
        cc: &eframe::CreationContext<'_>,
        server_client_receiver: mpsc::UnboundedReceiver<WebsocketServerClientMessage>,
        client_server_sender: mpsc::UnboundedSender<WebsocketClientServerMessage>,
    ) -> Self {
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
            loaded_models: LoadedModels {
                model_load_state: None,
                current_models: Vec::new(),
                current_interfaces: HashMap::new(),
                currently_requesting_model: None,
                loaded_models: HashMap::new(),
            },
            server_request_manager: ServerRequestManager::new(client_server_sender),
            next_interface_id: 0,
            websocket_server_client_receiver: server_client_receiver,
            app_state,
            selected_graph_explorer_tab: None,
            graph_explorer_app: HashMap::new(),
            loaded_tokenizers: LoadedTokenizers::new(),
            llm_explorer_app: LLMExplorerApp::new(),
            sd_explorer_app: SDExplorerApp::new(),
            server_config_report: None,
            loader_registry: None,
        }
    }

    fn render_loader_dialog(&mut self, ctx: &egui::Context) {
        let Some(model_load_state) = self.loaded_models.model_load_state.clone() else {
            return;
        };
        let Some(registry) = &self.loader_registry else {
            return;
        };
        let registry = registry.clone();

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

                    // Loader selector
                    if !registry.loaders.is_empty() {
                        let selected = self.app_state.loader_dialog.selected_loader;
                        let selected_name = registry
                            .loaders
                            .get(selected)
                            .map(|l| l.name.as_str())
                            .unwrap_or("Select loader");

                        ui.horizontal(|ui| {
                            ui.label("Loader:");
                            egui::ComboBox::from_id_salt("loader_selector")
                                .selected_text(selected_name)
                                .show_ui(ui, |ui| {
                                    for (i, loader) in registry.loaders.iter().enumerate() {
                                        ui.selectable_value(
                                            &mut self.app_state.loader_dialog.selected_loader,
                                            i,
                                            &loader.name,
                                        );
                                    }
                                });
                        });

                        // Render config fields for selected loader
                        if let Some(loader) = registry.loaders.get(selected) {
                            if !loader.description.is_empty() {
                                ui.label(egui::RichText::new(&loader.description).small().weak());
                            }

                            let field_values = self
                                .app_state
                                .loader_dialog
                                .field_values
                                .entry(selected)
                                .or_default();

                            for field in &loader.config_schema {
                                let value =
                                    field_values.entry(field.key.clone()).or_insert_with(|| {
                                        // Initialize from default
                                        match &field.default {
                                            Some(ConfigValue::String(s)) => s.clone(),
                                            Some(ConfigValue::FilePath(p)) => {
                                                p.to_string_lossy().to_string()
                                            }
                                            Some(ConfigValue::Integer(n)) => n.to_string(),
                                            Some(ConfigValue::Float(f)) => f.to_string(),
                                            Some(ConfigValue::Bool(b)) => b.to_string(),
                                            None => String::new(),
                                        }
                                    });

                                ui.horizontal(|ui| {
                                    let label = if field.required {
                                        format!("{}*:", field.label)
                                    } else {
                                        format!("{}:", field.label)
                                    };
                                    ui.label(label);

                                    match &field.field_type {
                                        ConfigFieldType::FilePath => {
                                            ui.text_edit_singleline(value);
                                            #[cfg(not(target_arch = "wasm32"))]
                                            {
                                                if ui.button("File…").clicked() {
                                                    if let Some(path) =
                                                        rfd::FileDialog::new().pick_file()
                                                    {
                                                        *value = path.to_string_lossy().to_string();
                                                    }
                                                }
                                                if ui.button("Dir…").clicked() {
                                                    if let Some(path) =
                                                        rfd::FileDialog::new().pick_folder()
                                                    {
                                                        *value = path.to_string_lossy().to_string();
                                                    }
                                                }
                                            }
                                        }
                                        ConfigFieldType::String => {
                                            ui.text_edit_singleline(value);
                                        }
                                        ConfigFieldType::Integer { .. } => {
                                            ui.text_edit_singleline(value);
                                        }
                                        ConfigFieldType::Float { .. } => {
                                            ui.text_edit_singleline(value);
                                        }
                                        ConfigFieldType::Bool => {
                                            let mut checked =
                                                value.parse::<bool>().unwrap_or(false);
                                            if ui.checkbox(&mut checked, "").changed() {
                                                *value = checked.to_string();
                                            }
                                        }
                                        ConfigFieldType::Enum { options } => {
                                            egui::ComboBox::from_id_salt(&field.key)
                                                .selected_text(value.as_str())
                                                .show_ui(ui, |ui| {
                                                    for opt in options {
                                                        ui.selectable_value(
                                                            value,
                                                            opt.clone(),
                                                            opt,
                                                        );
                                                    }
                                                });
                                        }
                                    }
                                });
                            }
                        }
                    } else {
                        ui.label("No loaders available (server not connected?)");
                    }

                    ui.horizontal(|ui| {
                        if ui.button("Load").clicked()
                            && let Some(loader) = registry
                                .loaders
                                .get(self.app_state.loader_dialog.selected_loader)
                        {
                            // Build ConfigValues from field strings
                            let field_values = self
                                .app_state
                                .loader_dialog
                                .field_values
                                .get(&self.app_state.loader_dialog.selected_loader)
                                .cloned()
                                .unwrap_or_default();

                            let mut config = HashMap::new();
                            for field in &loader.config_schema {
                                if let Some(raw) = field_values.get(&field.key) {
                                    if raw.is_empty() && !field.required {
                                        continue;
                                    }
                                    let cv = match &field.field_type {
                                        ConfigFieldType::FilePath => {
                                            ConfigValue::FilePath(raw.into())
                                        }
                                        ConfigFieldType::String => ConfigValue::String(raw.clone()),
                                        ConfigFieldType::Integer { .. } => {
                                            match raw.parse::<i64>() {
                                                Ok(n) => ConfigValue::Integer(n),
                                                Err(_) => ConfigValue::String(raw.clone()),
                                            }
                                        }
                                        ConfigFieldType::Float { .. } => match raw.parse::<f64>() {
                                            Ok(f) => ConfigValue::Float(f),
                                            Err(_) => ConfigValue::String(raw.clone()),
                                        },
                                        ConfigFieldType::Bool => {
                                            ConfigValue::Bool(raw.parse::<bool>().unwrap_or(false))
                                        }
                                        ConfigFieldType::Enum { .. } => {
                                            ConfigValue::String(raw.clone())
                                        }
                                    };
                                    config.insert(field.key.clone(), cv);
                                }
                            }

                            self.server_request_manager
                                .send(WebsocketClientServerMessage::RunLoader {
                                    loader_index: self.app_state.loader_dialog.selected_loader,
                                    config,
                                })
                                .unwrap();
                            self.loaded_models.model_load_state = Some(ModelLoadState::Loading);
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
                        WebsocketServerClientMessage::LoaderRegistryReport(report) => {
                            self.loader_registry = Some(report);
                        }
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
                                match &interface.interface {
                                    AnyInterface::TextInferenceTokensInLogitOutInterface(iface) => {
                                        needed_tokenizers.push(iface.get_tokenizer().clone());
                                    }
                                    AnyInterface::ImageGenerationInterface(iface) => {
                                        for pi in &iface.positive_prompts {
                                            needed_tokenizers.push(pi.tokenizer.clone());
                                        }
                                        if let Some(neg) = &iface.negative_prompts {
                                            for pi in neg {
                                                needed_tokenizers.push(pi.tokenizer.clone());
                                            }
                                        }
                                    }
                                    AnyInterface::TextToSpeechInterface(_) => {}
                                    AnyInterface::SpeechToTextInterface(_) => {}
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
                                        TokenizerInfo::HFTokenizerLocal(path) => {
                                            self.server_request_manager
                                                .send(
                                                    WebsocketClientServerMessage::GetTokenizerFile(
                                                        path.clone(),
                                                    ),
                                                )
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
                                        TokenizerInfo::HFTokenizerJson(_) => {
                                            self.loaded_tokenizers.loaded_tokenizers.insert(
                                                tokenizer_info.clone(),
                                                Some(Ok(Arc::new(
                                                    AnyTokenizer::from_tokenizer_info(
                                                        &tokenizer_info,
                                                    ),
                                                ))),
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
                                && requesting_id == id
                            {
                                self.loaded_models.currently_requesting_model = None;
                                let graph =
                                    ciborium::from_reader::<SymbolicGraph, _>(graph_bin.as_slice())
                                        .unwrap();
                                self.loaded_models.loaded_models.insert(id, graph);
                            }
                        }
                        WebsocketServerClientMessage::TensorStoreReturn(
                            _model_id,
                            stored_tensor_id,
                            res,
                        ) => {
                            // Route to all graph explorers that may have requested this tensor
                            for explorer in self.graph_explorer_app.values_mut() {
                                for window in &mut explorer.inspect_windows {
                                    if let crate::graph_explorer::inspect_windows::AnyInspectWindow::GraphLink(link_window) = window
                                        && link_window.stored_value_requested == Some(stored_tensor_id) {
                                            link_window.stored_value = Some(res.clone());
                                            link_window.stored_value_requested = None;
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
                        WebsocketServerClientMessage::TokenizerFileReturn(path, bytes_res) => {
                            let tokenizer = match bytes_res {
                                Ok(x) => tokenizers::Tokenizer::from_bytes(x)
                                    .map_err(|x| x.to_string())
                                    .map(|x| Arc::new(AnyTokenizer::Tokenizers(x))),
                                Err(err) => Err(err),
                            };

                            self.loaded_tokenizers.loaded_tokenizers.insert(
                                TokenizerInfo::HFTokenizerLocal(path.clone()),
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
                Err(mpsc::error::TryRecvError::Empty) => {
                    // No issue
                    break;
                }
                Err(err) => {
                    log::debug!("Websocket error: {err}!");
                    break;
                }
            }
        }

        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            // The top panel is often a good place for a menu bar:
            egui::MenuBar::new().ui(ui, |ui| {
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
                ui.selectable_value(
                    &mut self.app_state.selected_tab,
                    SelectedTab::SDExplorer,
                    "SD Explorer",
                );
            });
        });

        self.render_loader_dialog(ctx);

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
                    ui.horizontal(|ui| {
                        let model_selector_options = {
                            let mut options = vec![];
                            for (interface_id, interface) in &self.loaded_models.current_interfaces
                            {
                                options.push((
                                    GraphRootSubjectSelection::Interface(*interface_id),
                                    format!(
                                        "({}) {}",
                                        interface_id,
                                        interface.interface_name.clone()
                                    ),
                                ));
                            }
                            for model in &self.loaded_models.current_models {
                                options.push((
                                    GraphRootSubjectSelection::Model(model.model_id),
                                    format!("({}) {}", model.model_id, model.model_name.clone()),
                                ));
                            }
                            options
                        };
                        egui::ComboBox::from_id_salt(123661)
                            .selected_text(
                                model_selector_options
                                    .iter()
                                    .find(|(a, _b)| {
                                        self.selected_graph_explorer_tab
                                            .as_ref()
                                            .map(|x| x == a)
                                            .unwrap_or(false)
                                    })
                                    .map(|(_a, b)| b.clone())
                                    .unwrap_or("Select a model or interface".to_string()),
                            )
                            .show_ui(ui, |ui| {
                                for (a, b) in model_selector_options {
                                    ui.selectable_value(
                                        &mut self.selected_graph_explorer_tab,
                                        Some(a),
                                        b.to_string(),
                                    );
                                }
                            });
                        if ui.button("Load New Model").clicked() {
                            self.loaded_models.model_load_state =
                                Some(ModelLoadState::DialogOpen(None));
                        };
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            toggle_ui(
                                ui,
                                &mut self.app_state.graph_explorer_settings.explorer_minimap,
                            );
                            ui.label("Minimap:");
                            if let Some(selected_tab) = self.selected_graph_explorer_tab
                                && let Some(ge) = self.graph_explorer_app.get_mut(&selected_tab)
                            {
                                toggle_ui(ui, &mut ge.show_profiling_window);
                                ui.label("Profiling:");
                            }
                            toggle_ui(
                                ui,
                                &mut self.app_state.graph_explorer_settings.explorer_physics,
                            );
                            ui.label("Physics:");
                            toggle_ui(
                                ui,
                                &mut self.app_state.graph_explorer_settings.explorer_node_wave,
                            );
                            ui.label("Activity:");
                            toggle_ui(
                                ui,
                                &mut self
                                    .app_state
                                    .graph_explorer_settings
                                    .do_all_explorer_swatches,
                            );
                            ui.label("All Swatches:");
                            toggle_ui(
                                ui,
                                &mut self
                                    .app_state
                                    .graph_explorer_settings
                                    .do_explorer_swatches_in_view,
                            );
                            ui.label("Swatches In-frame:");
                        })
                    });
                    if let Some(selected_tab) = self.selected_graph_explorer_tab {
                        let graph_explorer = self
                            .graph_explorer_app
                            .entry(selected_tab)
                            .or_insert_with(|| GraphExplorerApp::new(selected_tab));
                        if let Some(server_config_report) = &self.server_config_report {
                            if self.app_state.graph_explorer_settings.explorer_minimap {
                                ui.horizontal(|ui| graph_explorer.render_minimap(ui));
                            }
                            graph_explorer.update(
                                &mut self.app_state.graph_explorer_settings,
                                &mut self.loaded_models,
                                &mut self.loaded_tokenizers,
                                &mut self.server_request_manager,
                                server_config_report,
                                ui,
                            );
                        }
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
                SelectedTab::SDExplorer => {
                    self.sd_explorer_app.update(
                        &mut self.app_state.sd_explorer_state,
                        &mut self.loaded_models,
                        &mut self.loaded_tokenizers,
                        &mut self.server_request_manager,
                        ui,
                    );
                }
            }
        });

        if let SelectedTab::GraphExplorer = self.app_state.selected_tab
            && let Some(selected_tab) = self.selected_graph_explorer_tab
            && let Some(app) = self.graph_explorer_app.get_mut(&selected_tab)
        {
            app.update_inspect_windows(
                &mut self.app_state.graph_explorer_settings,
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
