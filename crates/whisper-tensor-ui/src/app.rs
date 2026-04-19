use crate::graph_explorer::{GraphExplorerApp, GraphExplorerSettings, GraphRootSubjectSelection};
use crate::llm_explorer::{LLMExplorerApp, LLMExplorerState};
use crate::sd_explorer::{SDExplorerApp, SDExplorerState};
use crate::server_stats::{ServerStatsHistory, render_server_stats};
use crate::stt_explorer::{STTExplorerApp, STTExplorerState};
use crate::tts_explorer::{TTSExplorerApp, TTSExplorerState};
use crate::websockets::ServerRequestManager;
use crate::widgets::toggle::toggle_ui;
use egui::Margin;
use rwkv_tokenizer::WorldTokenizer;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;
use std::sync::Arc;
use tokio::sync::mpsc;
use whisper_tensor::graph::GraphDyn;
use whisper_tensor::interfaces::AnyInterface;
use whisper_tensor::loader::{ConfigFieldType, ConfigValue};
use whisper_tensor::metadata::TokenizerInfo;
use whisper_tensor::milli_graph::MilliOpGraph;
use whisper_tensor::super_graph::SuperGraph;
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
    STTExplorer,
    TTSExplorer,
    BuildInspector,
    Resources,
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
    stt_explorer_state: STTExplorerState,
    tts_explorer_state: TTSExplorerState,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            selected_tab: SelectedTab::Models,
            loader_dialog: LoaderDialogState::default(),
            graph_explorer_settings: GraphExplorerSettings::default(),
            llm_explorer_state: LLMExplorerState::default(),
            sd_explorer_state: SDExplorerState::default(),
            stt_explorer_state: STTExplorerState::default(),
            tts_explorer_state: TTSExplorerState::default(),
        }
    }
}

pub(crate) type InterfaceId = u32;
pub(crate) type ClientGraphId = u32;

#[allow(dead_code)]
pub(crate) enum ClientLoadedGraphData {
    Super(Box<SuperGraph>),
    MilliOp(Box<MilliOpGraph>),
    Symbolic(Box<SymbolicGraph>),
}

impl ClientLoadedGraphData {
    pub(crate) fn as_graph_dyn(&self) -> &dyn GraphDyn {
        match self {
            Self::Super(graph) => graph.as_ref(),
            Self::MilliOp(graph) => graph.as_ref(),
            Self::Symbolic(graph) => graph.as_ref(),
        }
    }

    pub(crate) fn kind_name(&self) -> &'static str {
        match self {
            Self::Super(_) => "SuperGraph",
            Self::MilliOp(_) => "MilliOpGraph",
            Self::Symbolic(_) => "SymbolicGraph",
        }
    }

    pub(crate) fn supports_editing(&self) -> bool {
        matches!(self, Self::Super(_) | Self::MilliOp(_))
    }
}

pub(crate) struct ClientLoadedGraphEntry {
    pub(crate) display_name: String,
    pub(crate) graph: ClientLoadedGraphData,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GraphRootOwnership {
    Server,
    Client,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GraphEditability {
    ServerReadOnly,
    ClientEditable,
    ClientReadOnlyUnsupported,
}

impl GraphEditability {
    pub(crate) fn description(self) -> &'static str {
        match self {
            Self::ServerReadOnly => "server-loaded (read-only)",
            Self::ClientEditable => "client-loaded (editable)",
            Self::ClientReadOnlyUnsupported => "client-loaded (read-only, unsupported type)",
        }
    }

    pub(crate) fn can_edit(self) -> bool {
        matches!(self, Self::ClientEditable)
    }
}

pub(crate) struct ResolvedRootGraph<'a> {
    pub(crate) graph: &'a dyn GraphDyn,
    pub(crate) ownership: GraphRootOwnership,
    pub(crate) editability: GraphEditability,
}

#[allow(dead_code)] // Reserved for upcoming graph-edit command plumbing.
pub(crate) enum EditableClientGraphMut<'a> {
    SuperGraph(&'a mut SuperGraph),
    MilliOpGraph(&'a mut MilliOpGraph),
}

#[allow(dead_code)] // Reserved for upcoming graph-edit command plumbing.
#[derive(Debug)]
pub(crate) enum GraphEditAccessError {
    NotClientGraphRoot,
    ClientGraphNotFound(ClientGraphId),
    UnsupportedClientGraphType(&'static str),
}

impl core::fmt::Display for GraphEditAccessError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotClientGraphRoot => {
                write!(f, "graph root is not a client-loaded graph")
            }
            Self::ClientGraphNotFound(id) => write!(f, "client graph {id} not found"),
            Self::UnsupportedClientGraphType(kind) => {
                write!(f, "client graph type {kind} is not editable")
            }
        }
    }
}

#[derive(Default)]
pub(crate) struct ServerGraphCatalog {
    pub(crate) current_models: Vec<CurrentModelsReportEntry>,
    pub(crate) current_interfaces: HashMap<InterfaceId, CurrentInterfacesReportEntry>,
    pub(crate) currently_requesting_model: Option<LoadedModelId>,
    pub(crate) symbolic_graphs: HashMap<LoadedModelId, SymbolicGraph>,
}

#[derive(Default)]
#[allow(dead_code)]
pub(crate) struct ClientGraphCatalog {
    next_graph_id: ClientGraphId,
    pub(crate) graphs: HashMap<ClientGraphId, ClientLoadedGraphEntry>,
}

#[allow(dead_code)]
impl ClientGraphCatalog {
    pub(crate) fn insert_graph(
        &mut self,
        display_name: String,
        graph: ClientLoadedGraphData,
    ) -> ClientGraphId {
        let id = self.next_graph_id;
        self.next_graph_id = self.next_graph_id.wrapping_add(1);
        self.graphs.insert(
            id,
            ClientLoadedGraphEntry {
                display_name,
                graph,
            },
        );
        id
    }
}

pub(crate) struct LoadedModels {
    pub(crate) model_load_state: Option<ModelLoadState>,
    pub(crate) server_graphs: ServerGraphCatalog,
    pub(crate) client_graphs: ClientGraphCatalog,
}

impl LoadedModels {
    pub(crate) fn root_graph_editability(
        &self,
        selection: GraphRootSubjectSelection,
    ) -> GraphEditability {
        match selection {
            GraphRootSubjectSelection::ServerModel(_)
            | GraphRootSubjectSelection::ServerInterface(_) => GraphEditability::ServerReadOnly,
            GraphRootSubjectSelection::ClientGraph(client_graph_id) => self
                .client_graphs
                .graphs
                .get(&client_graph_id)
                .map_or(GraphEditability::ClientReadOnlyUnsupported, |entry| {
                    if entry.graph.supports_editing() {
                        GraphEditability::ClientEditable
                    } else {
                        GraphEditability::ClientReadOnlyUnsupported
                    }
                }),
        }
    }

    pub(crate) fn resolve_root_graph(
        &self,
        selection: GraphRootSubjectSelection,
    ) -> Option<ResolvedRootGraph<'_>> {
        let editability = self.root_graph_editability(selection);
        match selection {
            GraphRootSubjectSelection::ServerModel(model_id) => self
                .server_graphs
                .symbolic_graphs
                .get(&model_id)
                .map(|graph| ResolvedRootGraph {
                    graph,
                    ownership: GraphRootOwnership::Server,
                    editability,
                }),
            GraphRootSubjectSelection::ServerInterface(interface_id) => self
                .server_graphs
                .current_interfaces
                .get(&interface_id)
                .map(|entry| ResolvedRootGraph {
                    graph: entry.interface.get_super_graph(),
                    ownership: GraphRootOwnership::Server,
                    editability,
                }),
            GraphRootSubjectSelection::ClientGraph(client_graph_id) => self
                .client_graphs
                .graphs
                .get(&client_graph_id)
                .map(|entry| ResolvedRootGraph {
                    graph: entry.graph.as_graph_dyn(),
                    ownership: GraphRootOwnership::Client,
                    editability,
                }),
        }
    }

    #[allow(dead_code)] // Reserved for upcoming graph-edit command plumbing.
    pub(crate) fn with_editable_client_graph_mut<R>(
        &mut self,
        selection: GraphRootSubjectSelection,
        f: impl FnOnce(EditableClientGraphMut<'_>) -> R,
    ) -> Result<R, GraphEditAccessError> {
        let client_graph_id = match selection {
            GraphRootSubjectSelection::ClientGraph(client_graph_id) => client_graph_id,
            _ => return Err(GraphEditAccessError::NotClientGraphRoot),
        };
        let entry = self
            .client_graphs
            .graphs
            .get_mut(&client_graph_id)
            .ok_or(GraphEditAccessError::ClientGraphNotFound(client_graph_id))?;
        match &mut entry.graph {
            ClientLoadedGraphData::Super(graph) => {
                Ok(f(EditableClientGraphMut::SuperGraph(graph.as_mut())))
            }
            ClientLoadedGraphData::MilliOp(graph) => {
                Ok(f(EditableClientGraphMut::MilliOpGraph(graph.as_mut())))
            }
            ClientLoadedGraphData::Symbolic(_) => Err(
                GraphEditAccessError::UnsupportedClientGraphType("SymbolicGraph"),
            ),
        }
    }
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
    stt_explorer_app: STTExplorerApp,
    tts_explorer_app: TTSExplorerApp,
    loaded_tokenizers: LoadedTokenizers,
    server_config_report: Option<ServerConfigReport>,
    loader_registry: Option<LoaderRegistryReport>,
    graph_catalog_status: Option<String>,
    cache_report: Option<whisper_tensor_server::CacheReport>,
    server_stats: ServerStatsHistory,
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
                server_graphs: ServerGraphCatalog::default(),
                client_graphs: ClientGraphCatalog::default(),
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
            stt_explorer_app: STTExplorerApp::new(),
            tts_explorer_app: TTSExplorerApp::new(),
            server_config_report: None,
            loader_registry: None,
            graph_catalog_status: None,
            cache_report: None,
            server_stats: ServerStatsHistory::new(),
        }
    }

    #[allow(dead_code)]
    fn decode_client_graph_bytes(bytes: &[u8]) -> Result<ClientLoadedGraphData, String> {
        let super_res = SuperGraph::from_cbor_bytes(bytes);
        if let Ok(graph) = super_res {
            return Ok(ClientLoadedGraphData::Super(Box::new(graph)));
        }

        let milli_res = MilliOpGraph::from_cbor_bytes(bytes);
        if let Ok(graph) = milli_res {
            return Ok(ClientLoadedGraphData::MilliOp(Box::new(graph)));
        }

        let symbolic_res = ciborium::from_reader::<SymbolicGraph, _>(bytes);
        if let Ok(graph) = symbolic_res {
            return Ok(ClientLoadedGraphData::Symbolic(Box::new(graph)));
        }

        Err(format!(
            "decode failed as SuperGraph ({:?}), MilliOpGraph ({:?}), and SymbolicGraph ({:?})",
            super_res.err(),
            milli_res.err(),
            symbolic_res.err(),
        ))
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn import_client_graph_from_dialog(&mut self) -> Result<Option<String>, String> {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("Whisper Tensor Graphs", &["cbor"])
            .pick_file()
        else {
            return Ok(None);
        };

        let bytes = std::fs::read(&path)
            .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
        let graph = Self::decode_client_graph_bytes(&bytes)?;
        let kind_name = graph.kind_name().to_string();
        let display_name = graph_display_name_from_path(path.as_path());
        let graph_id = self
            .loaded_models
            .client_graphs
            .insert_graph(display_name, graph);
        self.selected_graph_explorer_tab = Some(GraphRootSubjectSelection::ClientGraph(graph_id));

        Ok(Some(format!(
            "Imported {kind_name} from {}",
            path.display()
        )))
    }

    #[cfg(target_arch = "wasm32")]
    fn import_client_graph_from_dialog(&mut self) -> Result<Option<String>, String> {
        Err("Import graph is not available in the web build yet.".to_string())
    }

    fn trigger_graph_import(&mut self) {
        match self.import_client_graph_from_dialog() {
            Ok(Some(status)) => {
                self.graph_catalog_status = Some(status);
            }
            Ok(None) => {}
            Err(err) => {
                self.graph_catalog_status = Some(format!("Import failed: {err}"));
            }
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
                                                if ui.button("File…").clicked()
                                                    && let Some(path) =
                                                        rfd::FileDialog::new().pick_file()
                                                {
                                                    *value = path.to_string_lossy().to_string();
                                                }
                                                if ui.button("Dir…").clicked()
                                                    && let Some(path) =
                                                        rfd::FileDialog::new().pick_folder()
                                                {
                                                    *value = path.to_string_lossy().to_string();
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

    fn render_build_inspector(&mut self, ui: &mut egui::Ui) {
        ui.vertical(|ui| {
            if ui.button("Refresh Cache Report").clicked() {
                self.server_request_manager
                    .send(WebsocketClientServerMessage::GetCacheReport)
                    .ok();
            }

            let Some(report) = &self.cache_report else {
                ui.label("No cache report yet. Click Refresh to query the server.");
                return;
            };

            if report.entries.is_empty() {
                ui.label("Cache is empty. Run a model with caching enabled to populate.");
                return;
            }

            egui::ScrollArea::vertical().show(ui, |ui| {
                for entry in &report.entries {
                    ui.separator();
                    ui.heading(format!("Cache slot {}", entry.cache_key));
                    ui.horizontal(|ui| {
                        ui.label(format!(
                            "RNN: {}  Tensors: {}  Packs: {}",
                            entry.num_rnn_entries,
                            entry.num_tensor_entries,
                            entry.num_tensor_pack_entries,
                        ));
                    });

                    for model in &entry.lowered_models {
                        ui.group(|ui| {
                            ui.strong(format!(
                                "Lowered Model {}  (hash: {:016x})",
                                model.graph_id, model.info_inputs_hash
                            ));

                            // Stats table
                            egui::Grid::new(format!(
                                "lowered_stats_{}_{}",
                                entry.cache_key, model.graph_id
                            ))
                            .striped(true)
                            .show(ui, |ui| {
                                ui.label("Groups");
                                ui.label(model.num_groups.to_string());
                                ui.end_row();
                                ui.label("Total atoms");
                                ui.label(model.total_atoms.to_string());
                                ui.end_row();
                                ui.label("Singleton groups");
                                ui.label(model.singleton_groups.to_string());
                                ui.end_row();
                                ui.label("Symbolic groups");
                                ui.label(model.symbolic_groups.to_string());
                                ui.end_row();
                                ui.label("Tensors");
                                ui.label(model.num_tensors.to_string());
                                ui.end_row();
                                ui.label("Inputs");
                                ui.label(model.num_inputs.to_string());
                                ui.end_row();
                                ui.label("Outputs");
                                ui.label(model.num_outputs.to_string());
                                ui.end_row();
                            });

                            // Nano op breakdown
                            if !model.groups_by_op.is_empty() {
                                egui::CollapsingHeader::new("Nano op breakdown")
                                    .id_salt(format!(
                                        "lowered_nano_{}_{}",
                                        entry.cache_key, model.graph_id
                                    ))
                                    .show(ui, |ui| {
                                        let mut ops: Vec<_> = model.groups_by_op.iter().collect();
                                        ops.sort_by(|a, b| b.1.cmp(a.1));
                                        egui::Grid::new(format!(
                                            "nano_ops_{}_{}",
                                            entry.cache_key, model.graph_id
                                        ))
                                        .striped(true)
                                        .show(ui, |ui| {
                                            ui.strong("Op");
                                            ui.strong("Count");
                                            ui.end_row();
                                            for (op, count) in &ops {
                                                ui.label(*op);
                                                ui.label(count.to_string());
                                                ui.end_row();
                                            }
                                        });
                                    });
                            }

                            // Milli op census
                            if !model.milli_op_census.is_empty() {
                                egui::CollapsingHeader::new("Milli op census")
                                    .id_salt(format!(
                                        "lowered_milli_{}_{}",
                                        entry.cache_key, model.graph_id
                                    ))
                                    .show(ui, |ui| {
                                        let mut ops: Vec<_> =
                                            model.milli_op_census.iter().collect();
                                        ops.sort_by_key(|b| std::cmp::Reverse(b.1.1));
                                        egui::Grid::new(format!(
                                            "milli_ops_{}_{}",
                                            entry.cache_key, model.graph_id
                                        ))
                                        .striped(true)
                                        .show(ui, |ui| {
                                            ui.strong("Milli Op");
                                            ui.strong("Groups");
                                            ui.strong("Atoms");
                                            ui.end_row();
                                            for (op, (groups, atoms)) in &ops {
                                                ui.label(*op);
                                                ui.label(groups.to_string());
                                                ui.label(atoms.to_string());
                                                ui.end_row();
                                            }
                                        });
                                    });
                            }

                            // Unsupported ops
                            if !model.unsupported.is_empty() {
                                egui::CollapsingHeader::new(format!(
                                    "Unsupported ops ({})",
                                    model.unsupported.len()
                                ))
                                .id_salt(format!(
                                    "lowered_unsup_{}_{}",
                                    entry.cache_key, model.graph_id
                                ))
                                .show(ui, |ui| {
                                    for detail in &model.unsupported_details {
                                        ui.label(detail);
                                    }
                                });
                            }
                        });
                    }

                    for plan in &entry.compiled_plans {
                        ui.group(|ui| {
                            ui.strong(format!(
                                "Compiled Plan {}  (hash: {:016x})",
                                plan.graph_id, plan.info_inputs_hash
                            ));
                            let s = &plan.plan_summary;
                            egui::Grid::new(format!(
                                "compiled_stats_{}_{}",
                                entry.cache_key, plan.graph_id
                            ))
                            .striped(true)
                            .show(ui, |ui| {
                                ui.label("Phases");
                                ui.label(s.num_phases.to_string());
                                ui.end_row();
                                ui.label("Lanes");
                                ui.label(s.num_lanes.to_string());
                                ui.end_row();
                                ui.label("Main graph groups");
                                ui.label(s.main_graph_groups.to_string());
                                ui.end_row();
                                ui.label("Main graph atoms");
                                ui.label(s.main_graph_atoms.to_string());
                                ui.end_row();
                                ui.label("Total compute atoms");
                                ui.label(s.total_compute_atoms.to_string());
                                ui.end_row();
                                ui.label("Total input atoms");
                                ui.label(s.total_input_atoms.to_string());
                                ui.end_row();
                                ui.label("Total output atoms");
                                ui.label(s.total_output_atoms.to_string());
                                ui.end_row();
                                ui.label("Model outputs");
                                ui.label(plan.num_outputs.to_string());
                                ui.end_row();
                            });

                            // Milli op census
                            if !s.milli_op_census.is_empty() {
                                egui::CollapsingHeader::new("Milli op census")
                                    .id_salt(format!(
                                        "compiled_milli_{}_{}",
                                        entry.cache_key, plan.graph_id
                                    ))
                                    .show(ui, |ui| {
                                        egui::Grid::new(format!(
                                            "compiled_milli_grid_{}_{}",
                                            entry.cache_key, plan.graph_id
                                        ))
                                        .striped(true)
                                        .show(ui, |ui| {
                                            ui.strong("Milli Op");
                                            ui.strong("Groups");
                                            ui.strong("Atoms");
                                            ui.end_row();
                                            for (op, groups, atoms) in &s.milli_op_census {
                                                ui.label(op);
                                                ui.label(groups.to_string());
                                                ui.label(atoms.to_string());
                                                ui.end_row();
                                            }
                                        });
                                    });
                            }

                            // Per-phase breakdown
                            egui::CollapsingHeader::new(format!("Phases ({})", s.phases.len()))
                                .id_salt(format!(
                                    "compiled_phases_{}_{}",
                                    entry.cache_key, plan.graph_id
                                ))
                                .show(ui, |ui| {
                                    for (pi, phase) in s.phases.iter().enumerate() {
                                        egui::CollapsingHeader::new(format!(
                                            "Phase {}: {} groups, {} atoms, balance {:.1}x",
                                            pi,
                                            phase.num_groups,
                                            phase.compute_atoms,
                                            phase.balance,
                                        ))
                                        .id_salt(format!(
                                            "compiled_phase_{}_{}_{}",
                                            entry.cache_key, plan.graph_id, pi
                                        ))
                                        .show(ui, |ui| {
                                            egui::Grid::new(format!(
                                                "phase_{}_{}_{}",
                                                entry.cache_key, plan.graph_id, pi
                                            ))
                                            .striped(true)
                                            .show(
                                                ui,
                                                |ui| {
                                                    ui.strong("Lane");
                                                    ui.strong("Groups");
                                                    ui.strong("Atoms");
                                                    ui.strong("In");
                                                    ui.strong("Out");
                                                    ui.strong("Nano ops");
                                                    ui.strong("Milli ops");
                                                    ui.end_row();
                                                    for (li, lane) in phase.lanes.iter().enumerate()
                                                    {
                                                        if lane.num_groups == 0 {
                                                            continue;
                                                        }
                                                        ui.label(li.to_string());
                                                        ui.label(lane.num_groups.to_string());
                                                        ui.label(lane.atoms.to_string());
                                                        ui.label(lane.num_inputs.to_string());
                                                        ui.label(lane.num_outputs.to_string());
                                                        let nano: String = lane
                                                            .nano_ops
                                                            .iter()
                                                            .map(|(n, a)| format!("{n}:{a}"))
                                                            .collect::<Vec<_>>()
                                                            .join(" ");
                                                        ui.label(nano);
                                                        let milli: String = lane
                                                            .milli_ops
                                                            .iter()
                                                            .map(|(k, g, a)| {
                                                                format!("{k}({g}g/{a}a)")
                                                            })
                                                            .collect::<Vec<_>>()
                                                            .join(", ");
                                                        ui.label(milli);
                                                        ui.end_row();
                                                    }
                                                },
                                            );
                                        });
                                    }
                                });
                        });
                    }
                }
            });
        });
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn graph_display_name_from_path(path: &Path) -> String {
    if let Some(file_stem) = path.file_stem().and_then(|x| x.to_str())
        && !file_stem.is_empty()
    {
        return file_stem.to_string();
    }
    if let Some(file_name) = path.file_name().and_then(|x| x.to_str())
        && !file_name.is_empty()
    {
        return file_name.to_string();
    }
    "client_graph".to_string()
}

fn interface_type_label(interface: &AnyInterface) -> &'static str {
    match interface {
        AnyInterface::TextInferenceTokensInLogitOutInterface(_) => "Text Inference",
        AnyInterface::MultimodalLanguageInterface(_) => "Multimodal Language",
        AnyInterface::ImageGenerationInterface(_) => "Image Generation",
        AnyInterface::VideoGenerationInterface(_) => "Video Generation",
        AnyInterface::TextToSpeechInterface(_) => "Text to Speech",
        AnyInterface::SpeechToTextInterface(_) => "Speech to Text",
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
                            self.loaded_models.server_graphs.current_models = res.models;
                            // Rebuild interfaces list
                            self.loaded_models.server_graphs.current_interfaces = {
                                let mut new_interfaces = HashMap::new();
                                for interface in res.interfaces {
                                    let mut found = false;
                                    for (&id, existing_interface) in
                                        &self.loaded_models.server_graphs.current_interfaces
                                    {
                                        if interface.interface_name
                                            == existing_interface.interface_name
                                        {
                                            new_interfaces.insert(id, interface.clone());
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
                            for interface in
                                self.loaded_models.server_graphs.current_interfaces.values()
                            {
                                match &interface.interface {
                                    AnyInterface::TextInferenceTokensInLogitOutInterface(iface) => {
                                        needed_tokenizers.push(iface.get_tokenizer().clone());
                                    }
                                    AnyInterface::MultimodalLanguageInterface(iface) => {
                                        needed_tokenizers.push(iface.get_tokenizer().clone());
                                    }
                                    AnyInterface::ImageGenerationInterface(_) => {}
                                    AnyInterface::VideoGenerationInterface(_) => {}
                                    AnyInterface::TextToSpeechInterface(_) => {}
                                    AnyInterface::SpeechToTextInterface(iface) => {
                                        needed_tokenizers.push(iface.tokenizer.clone());
                                    }
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
                                self.loaded_models.server_graphs.currently_requesting_model
                                && requesting_id == id
                            {
                                self.loaded_models.server_graphs.currently_requesting_model = None;
                                let graph =
                                    ciborium::from_reader::<SymbolicGraph, _>(graph_bin.as_slice())
                                        .unwrap();
                                self.loaded_models
                                    .server_graphs
                                    .symbolic_graphs
                                    .insert(id, graph);
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
                        WebsocketServerClientMessage::CacheReportReturn(report) => {
                            self.cache_report = Some(report);
                        }
                        WebsocketServerClientMessage::ServerStatsReport(snapshot) => {
                            self.server_stats.push(snapshot);
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
                ui.selectable_value(
                    &mut self.app_state.selected_tab,
                    SelectedTab::STTExplorer,
                    "STT Explorer",
                );
                ui.selectable_value(
                    &mut self.app_state.selected_tab,
                    SelectedTab::TTSExplorer,
                    "TTS Explorer",
                );
                ui.selectable_value(
                    &mut self.app_state.selected_tab,
                    SelectedTab::BuildInspector,
                    "Build Inspector",
                );
                ui.selectable_value(
                    &mut self.app_state.selected_tab,
                    SelectedTab::Resources,
                    "Resources",
                );
            });
        });

        self.render_loader_dialog(ctx);

        egui::CentralPanel::default().show(ctx, |ui| {
            // The central panel the region left after adding TopPanel's and SidePanel's
            match &self.app_state.selected_tab {
                SelectedTab::Models => {
                    ui.vertical(|ui| {
                        ui.horizontal(|ui| {
                            if ui.button("Load New model").clicked() {
                                self.loaded_models.model_load_state =
                                    Some(ModelLoadState::DialogOpen(None));
                            }
                            if ui.button("Import Graph").clicked() {
                                self.trigger_graph_import();
                            }
                        });
                        if let Some(status) = &self.graph_catalog_status {
                            ui.label(status);
                        }

                        ui.label("Loaded Models:");
                        egui::Grid::new("loaded_models")
                            .striped(true)
                            .show(ui, |ui| {
                                ui.strong("Model ID");
                                ui.strong("Name");
                                ui.strong("Ops");
                                ui.strong("Compiled");
                                ui.strong("");
                                ui.strong("");
                                ui.end_row();
                                for model in &self.loaded_models.server_graphs.current_models {
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

                        ui.separator();
                        ui.label("Server-loaded Interfaces:");
                        if self
                            .loaded_models
                            .server_graphs
                            .current_interfaces
                            .is_empty()
                        {
                            ui.label("None loaded");
                        } else {
                            let mut interface_rows = self
                                .loaded_models
                                .server_graphs
                                .current_interfaces
                                .iter()
                                .map(|(&interface_id, interface)| {
                                    (
                                        interface_id,
                                        interface.interface_name.clone(),
                                        interface_type_label(&interface.interface),
                                        interface
                                            .model_ids
                                            .iter()
                                            .map(|id| id.to_string())
                                            .collect::<Vec<_>>()
                                            .join(", "),
                                    )
                                })
                                .collect::<Vec<_>>();
                            interface_rows.sort_by_key(|row| row.0);

                            egui::Grid::new("server_loaded_interfaces")
                                .striped(true)
                                .show(ui, |ui| {
                                    ui.strong("Interface ID");
                                    ui.strong("Name");
                                    ui.strong("Type");
                                    ui.strong("Referenced Model IDs");
                                    ui.end_row();

                                    for (interface_id, interface_name, interface_type, model_ids) in
                                        interface_rows
                                    {
                                        ui.label(interface_id.to_string());
                                        ui.label(interface_name);
                                        ui.label(interface_type);
                                        if model_ids.is_empty() {
                                            ui.label("-");
                                        } else {
                                            ui.label(model_ids);
                                        }
                                        ui.end_row();
                                    }
                                });
                        }

                        ui.separator();
                        ui.label("Client-loaded Graphs:");
                        if self.loaded_models.client_graphs.graphs.is_empty() {
                            ui.label("None loaded");
                        } else {
                            let mut client_ids = self
                                .loaded_models
                                .client_graphs
                                .graphs
                                .keys()
                                .copied()
                                .collect::<Vec<_>>();
                            client_ids.sort_unstable();

                            let mut to_remove = Vec::new();
                            egui::Grid::new("client_loaded_graphs")
                                .striped(true)
                                .show(ui, |ui| {
                                    ui.strong("Graph ID");
                                    ui.strong("Name");
                                    ui.strong("Kind");
                                    ui.strong("");
                                    ui.strong("");
                                    ui.end_row();

                                    for graph_id in client_ids {
                                        let Some(entry) =
                                            self.loaded_models.client_graphs.graphs.get(&graph_id)
                                        else {
                                            continue;
                                        };
                                        ui.label(format!("client-{graph_id}"));
                                        ui.label(entry.display_name.clone());
                                        ui.label(entry.graph.kind_name());
                                        if ui.button("Open").clicked() {
                                            self.selected_graph_explorer_tab = Some(
                                                GraphRootSubjectSelection::ClientGraph(graph_id),
                                            );
                                            self.app_state.selected_tab =
                                                SelectedTab::GraphExplorer;
                                        }
                                        if ui.button("Remove").clicked() {
                                            to_remove.push(graph_id);
                                        }
                                        ui.end_row();
                                    }
                                });

                            for graph_id in to_remove {
                                self.loaded_models.client_graphs.graphs.remove(&graph_id);
                                let root = GraphRootSubjectSelection::ClientGraph(graph_id);
                                self.graph_explorer_app.remove(&root);
                                if self.selected_graph_explorer_tab == Some(root) {
                                    self.selected_graph_explorer_tab = None;
                                }
                            }
                        }
                    });
                }
                SelectedTab::GraphExplorer => {
                    ui.vertical(|ui| {
                        ui.horizontal(|ui| {
                            let model_selector_options = {
                                let mut options = vec![];
                                for (interface_id, interface) in
                                    &self.loaded_models.server_graphs.current_interfaces
                                {
                                    options.push((
                                        GraphRootSubjectSelection::ServerInterface(*interface_id),
                                        format!(
                                            "({}) {}",
                                            interface_id,
                                            interface.interface_name.clone()
                                        ),
                                    ));
                                }
                                for model in &self.loaded_models.server_graphs.current_models {
                                    options.push((
                                        GraphRootSubjectSelection::ServerModel(model.model_id),
                                        format!(
                                            "({}) {}",
                                            model.model_id,
                                            model.model_name.clone()
                                        ),
                                    ));
                                }
                                let mut client_ids = self
                                    .loaded_models
                                    .client_graphs
                                    .graphs
                                    .keys()
                                    .copied()
                                    .collect::<Vec<_>>();
                                client_ids.sort_unstable();
                                for client_id in client_ids {
                                    if let Some(graph_entry) =
                                        self.loaded_models.client_graphs.graphs.get(&client_id)
                                    {
                                        options.push((
                                            GraphRootSubjectSelection::ClientGraph(client_id),
                                            format!(
                                                "(client-{client_id}) {} [{}]",
                                                graph_entry.display_name,
                                                graph_entry.graph.kind_name(),
                                            ),
                                        ));
                                    }
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
                                        .unwrap_or("Select a graph source".to_string()),
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
                            if ui.button("Import Graph").clicked() {
                                self.trigger_graph_import();
                            }
                            if ui.button("Load New Model").clicked() {
                                self.loaded_models.model_load_state =
                                    Some(ModelLoadState::DialogOpen(None));
                            };
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    toggle_ui(
                                        ui,
                                        &mut self
                                            .app_state
                                            .graph_explorer_settings
                                            .explorer_minimap,
                                    );
                                    ui.label("Minimap:");
                                    if let Some(selected_tab) = self.selected_graph_explorer_tab
                                        && let Some(ge) =
                                            self.graph_explorer_app.get_mut(&selected_tab)
                                    {
                                        toggle_ui(ui, &mut ge.show_profiling_window);
                                        ui.label("Profiling:");
                                    }
                                    toggle_ui(
                                        ui,
                                        &mut self
                                            .app_state
                                            .graph_explorer_settings
                                            .explorer_physics,
                                    );
                                    ui.label("Physics:");
                                    toggle_ui(
                                        ui,
                                        &mut self
                                            .app_state
                                            .graph_explorer_settings
                                            .explorer_node_wave,
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
                                },
                            );
                        });
                        if let Some(status) = &self.graph_catalog_status {
                            ui.label(status);
                        }
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
                SelectedTab::TTSExplorer => {
                    self.tts_explorer_app.update(
                        &mut self.app_state.tts_explorer_state,
                        &mut self.loaded_models,
                        &mut self.loaded_tokenizers,
                        &mut self.server_request_manager,
                        ui,
                    );
                }
                SelectedTab::STTExplorer => {
                    self.stt_explorer_app.update(
                        &mut self.app_state.stt_explorer_state,
                        &mut self.loaded_models,
                        &mut self.loaded_tokenizers,
                        &mut self.server_request_manager,
                        ui,
                    );
                }
                SelectedTab::BuildInspector => {
                    self.render_build_inspector(ui);
                }
                SelectedTab::Resources => {
                    render_server_stats(ui, &self.server_stats);
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
