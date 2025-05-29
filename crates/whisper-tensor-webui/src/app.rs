use std::cmp::{max, Ordering};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;
use egui::{vec2, Color32, Label, Layout, Margin, Rect, Response, RichText, Sense, Shape, Stroke, StrokeKind, UiBuilder, Vec2, Widget};
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
use whisper_tensor::{DynRank, NDArrayNumericTensor};
use whisper_tensor::symbolic_graph::{OperationId, StoredOrNotTensor, SymbolicGraph, TensorId, TensorType};
use whisper_tensor::symbolic_graph::ops::{AnyOperation, Operation};
use whisper_tensor::symbolic_graph::scalar_info::ScalarInfoTyped;
use whisper_tensor::symbolic_graph::tensor_store::TensorStoreTensorId;
use whisper_tensor::tokenizer::AnyTokenizer;
use whisper_tensor_import::ModelTypeHint;
use whisper_tensor_import::onnx_graph::TokenizerInfo;
use whisper_tensor_server::{CurrentModelsReportEntry, LoadedModelId, ModelLoadType, ModelTypeMetadata, WebsocketClientServerMessage, WebsocketServerClientMessage};
use crate::graph_layout::{GraphLayout, GraphLayoutIOOffsets, GraphLayoutLinkData, GraphLayoutLinkId, GraphLayoutLinkType, GraphLayoutNode, GraphLayoutNodeId, GraphLayoutNodeInitData, GraphLayoutNodeType};

pub (crate) async fn websocket_task(server_client_sender: mpsc::UnboundedSender<WebsocketServerClientMessage>,
                                    mut client_server_receiver: mpsc::UnboundedReceiver<WebsocketClientServerMessage>) {
    let ws = WebSocket::new("/ws").unwrap();

    // Set up event handlers
    let onopen_callback = Closure::wrap(Box::new(move || {
        log::debug!("WebSocket connection opened");
    }) as Box<dyn FnMut()>);
    ws.set_onopen(Some(onopen_callback.as_ref().unchecked_ref()));
    onopen_callback.forget();

    // Handle messages coming from the server
    let server_client_sender_clone = server_client_sender.clone();
    let onmessage_callback = Closure::wrap(Box::new(move |e: web_sys::MessageEvent| {
        match e.data().dyn_into::<web_sys::Blob>() {
            Ok(blob) => {
                let fr = web_sys::FileReader::new().unwrap();
                let fr_c = fr.clone();
                // create onLoadEnd callback
                let sender_clone = server_client_sender_clone.clone();
                let onloadend_cb = Closure::<dyn FnMut(_)>::new(move |_e: web_sys::ProgressEvent| {
                    let array = js_sys::Uint8Array::new(&fr_c.result().unwrap());
                    let vec = array.to_vec();
                    log::info!("Blob received {} bytes: {:?}", vec.len(), vec);
                    // here you can for example use the received image/png data

                    match ciborium::from_reader::<WebsocketServerClientMessage, _>(vec.as_slice()) {
                        Ok(msg) => {
                            log::debug!("Decoded message: {:?}", msg);
                            sender_clone.send(msg).unwrap();
                        }
                        Err(err) => {
                            log::warn!("Failed to decode message: {:?}", err);
                        }
                    }
                });
                fr.set_onloadend(Some(onloadend_cb.as_ref().unchecked_ref()));
                fr.read_as_array_buffer(&blob).expect("blob not readable");
                onloadend_cb.forget();
            }
            Err(err) => {
                log::warn!("Failed to decode message: {:?}", err);
            }
        }
    }) as Box<dyn FnMut(web_sys::MessageEvent)>);
    ws.set_onmessage(Some(onmessage_callback.as_ref().unchecked_ref()));
    onmessage_callback.forget();

    // Handle errors
    let onerror_callback = Closure::wrap(Box::new(move |e: web_sys::ErrorEvent| {
        log::error!("WebSocket error: {}", e.message());
    }) as Box<dyn FnMut(web_sys::ErrorEvent)>);
    ws.set_onerror(Some(onerror_callback.as_ref().unchecked_ref()));
    onerror_callback.forget();

    // Handle closing
    let onclose_callback = Closure::wrap(Box::new(move |e: web_sys::CloseEvent| {
        log::debug!("WebSocket closed: {} - {}", e.code(), e.reason());
    }) as Box<dyn FnMut(web_sys::CloseEvent)>);
    ws.set_onclose(Some(onclose_callback.as_ref().unchecked_ref()));
    onclose_callback.forget();

    // Process messages from the client to send to the server
    let ws_clone = ws.clone();
    while let Some(message) = client_server_receiver.recv().await {
        let mut data = Vec::<u8>::new();
        ciborium::into_writer(&message, &mut data).unwrap();
        log::debug!("Sending message to server");
        ws_clone.send_with_u8_array(&data).unwrap();
    }
}

fn toggle_ui(ui: &mut egui::Ui, on: &mut bool) -> egui::Response {
    let desired_size = ui.spacing().interact_size.y * egui::vec2(2.0, 1.0);
    let (rect, mut response) = ui.allocate_exact_size(desired_size, egui::Sense::click());
    if response.clicked() {
        *on = !*on;
        response.mark_changed();
    }
    response.widget_info(|| {
        egui::WidgetInfo::selected(egui::WidgetType::Checkbox, ui.is_enabled(), *on, "")
    });

    if ui.is_rect_visible(rect) {
        let how_on = ui.ctx().animate_bool_responsive(response.id, *on);
        let visuals = ui.style().interact_selectable(&response, *on);
        let rect = rect.expand(visuals.expansion);
        let radius = 0.5 * rect.height();
        ui.painter().rect(
            rect,
            radius,
            visuals.bg_fill,
            visuals.bg_stroke,
            egui::StrokeKind::Inside,
        );
        let circle_x = egui::lerp((rect.left() + radius)..=(rect.right() - radius), how_on);
        let center = egui::pos2(circle_x, rect.center().y);
        ui.painter()
            .circle(center, 0.75 * radius, visuals.bg_fill, visuals.fg_stroke);
    }

    response
}


#[derive(Clone, Debug)]
enum ModelLoadState {
    DialogOpen(Option<String>),
    Loading
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
enum SelectedTab {
    Models,
    GraphExplorer,
    LLMExplorer
}

#[derive(Clone, Debug)]
struct InspectWindowTensor {
    tensor_id: TensorId,
    stored_value_requested: Option<TensorStoreTensorId>,
    stored_value: Option<Result<NDArrayNumericTensor<DynRank>, String>>
}

#[derive(Clone, Debug)]
enum InspectWindow {
    Operation(OperationId),
    Tensor(InspectWindowTensor)
}

impl InspectWindow {
    fn to_op_or_tensor_id(&self) -> OpOrTensorId {
        match self {
            InspectWindow::Operation(op_id) => OpOrTensorId::Op(*op_id),
            InspectWindow::Tensor (x) => OpOrTensorId::Tensor(x.tensor_id),
        }
    }

    fn check_if_already_exists(windows: &[Self], op_or_tensor_id: &OpOrTensorId) -> bool {
        for window in windows {
            if window.to_op_or_tensor_id() == *op_or_tensor_id {
                return true;
            }
        }
        false
    }

    fn new(op_or_tensor_id: OpOrTensorId) -> Self {
        match op_or_tensor_id {
            OpOrTensorId::Op(op_id) => Self::Operation(op_id),
            OpOrTensorId::Tensor(tensor_id) => {
                Self::Tensor(InspectWindowTensor{
                    tensor_id,
                    stored_value_requested: None,
                    stored_value: None
                })
            },
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct AppState {
    selected_tab: SelectedTab,
    model_to_load_path_text: String,
    model_type_hint_selected: Option<ModelTypeHint>,
    model_load_type_selected: ModelLoadType,
    graph_explorer_selected_model_id: Option<LoadedModelId>,
    llm_explorer_selected_model_id: Option<LoadedModelId>,
    current_llm_text: String,
    explorer_physics: bool,
    explorer_minimap: bool,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            selected_tab: SelectedTab::Models,
            model_to_load_path_text: String::new(),
            model_type_hint_selected: None,
            graph_explorer_selected_model_id: None,
            llm_explorer_selected_model_id: None,
            model_load_type_selected: ModelLoadType::Other,
            current_llm_text: String::new(),
            explorer_physics: true,
            explorer_minimap: true,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum OpOrTensorId {
    Op(OperationId),
    Tensor(TensorId)
}

pub struct TemplateApp {
    model_load_state: Option<ModelLoadState>,
    websocket_server_client_receiver: mpsc::UnboundedReceiver<WebsocketServerClientMessage>,
    websocket_client_server_sender: mpsc::UnboundedSender<WebsocketClientServerMessage>,
    current_models: Vec<CurrentModelsReportEntry>,
    graph_layouts: HashMap<LoadedModelId, GraphLayout>,
    loaded_model_graph: Option<(LoadedModelId, SymbolicGraph)>,
    currently_requesting_model: Option<LoadedModelId>,
    model_view_scene_rects: HashMap<LoadedModelId, Rect>,
    app_state: AppState,
    explorer_selection: Option<OpOrTensorId>,
    explorer_hovered: Option<OpOrTensorId>,
    inspect_windows: Vec<InspectWindow>,
    loaded_tokenizers: HashMap<LoadedModelId, Option<Result<Arc<AnyTokenizer>, String>>>,
}

impl TemplateApp {
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
            model_load_state: None,
            websocket_server_client_receiver,
            websocket_client_server_sender,
            current_models: vec![],
            graph_layouts: HashMap::new(),
            loaded_model_graph: None,
            currently_requesting_model: None,
            model_view_scene_rects: HashMap::new(),
            app_state,
            explorer_selection: None,
            explorer_hovered: None,
            inspect_windows: vec![],
            loaded_tokenizers: HashMap::new()
        }
    }
}

fn render_node_contents(ui: &mut egui::Ui, node_type: &GraphLayoutNodeType, num_inputs: usize, num_outputs: usize, symbolic_graph: &SymbolicGraph, is_selected: bool, is_hovered: bool) -> (Response, GraphLayoutIOOffsets) {

    // Decide corner radius
    let corner_radius = match node_type {
        GraphLayoutNodeType::SymbolicGraphOperation(_) => {3.0}
        GraphLayoutNodeType::SymbolicGraphTensor(_) |
        GraphLayoutNodeType::ConnectionByNameSrc(_) |
        GraphLayoutNodeType::ConnectionByNameDest(_) => {10.0}
    };

    /*let stroke_width = if is_selected {
        ui.visuals().widgets.active.fg_stroke.width
    } else {
        ui.visuals().widgets.inactive.fg_stroke.width
    };*/
    let stroke_width = ui.visuals().widgets.active.fg_stroke.width;

    let mut frame = egui::Frame::new().inner_margin(5).stroke(
        Stroke {
            width: stroke_width,
            color: Color32::TRANSPARENT
        }
    ).corner_radius(corner_radius).begin(ui);
    {
        let ui = &mut frame.content_ui;
        match &node_type {
            GraphLayoutNodeType::SymbolicGraphOperation(op_id) => {
                let op = &symbolic_graph.get_operations()[&op_id];
                match &op.op {
                    AnyOperation::Constant(x) => {
                        let text = format!("{} {}", x.value.dtype(), x.value);
                        ui.add(Label::new(text).selectable(false));
                    }
                    _ => {
                        if let Some(name) = &op.name {
                            ui.add(Label::new(name).selectable(false));
                        }
                        ui.add(Label::new(op.op.get_op_type_name()).selectable(false));
                    }
                }
            }
            GraphLayoutNodeType::SymbolicGraphTensor(tensor_id) => {
                let data = symbolic_graph.get_tensor_info(*tensor_id).unwrap();
                let name = if let Some(name) = &data.onnx_name {
                    name.clone()
                } else {
                    format!("Tensor {}", tensor_id)
                };
                match &data.tensor_type {
                    TensorType::Constant(StoredOrNotTensor::NotStored(value)) => {
                        ui.add(Label::new(format!("{} {}", value.dtype(), value)));
                    }
                    _ => {
                        ui.add(Label::new(name).selectable(false));
                    }
                }
            }
            GraphLayoutNodeType::ConnectionByNameSrc(name) => {
                match name.link_type {
                    GraphLayoutLinkType::SymbolicGraphTensor(tensor_id) => {
                        let value = symbolic_graph.get_tensor_info(tensor_id).unwrap();
                        let name = value.onnx_name.clone().unwrap_or(format!("tensor {}", tensor_id));
                        ui.add(Label::new(format!("{} >", name)).selectable(false));
                    }
                }
            }
            GraphLayoutNodeType::ConnectionByNameDest(name) => {
                match name.link_type {
                    GraphLayoutLinkType::SymbolicGraphTensor(tensor_id) => {
                        let value = symbolic_graph.get_tensor_info(tensor_id).unwrap();
                        let name = value.onnx_name.clone().unwrap_or(format!("tensor {}", tensor_id));
                        ui.add(Label::new(format!("> {}", name)).selectable(false));
                    }
                }
            }
            _ => {
                // TODO
            }
        }
    }


    let mut content_rect = frame.content_ui.min_rect();
    content_rect = content_rect
        + frame.frame.inner_margin
        + Margin::from(stroke_width)
        + frame.frame.outer_margin;

    let response = ui.allocate_rect(content_rect, Sense::HOVER | Sense::CLICK | Sense::DRAG);

    let (fill, stroke) = if is_selected {
        (ui.visuals().widgets.active.bg_fill, egui::Stroke{
            width: stroke_width,
            color: Color32::from_rgb(64, 64, 255),
        })
    } else if response.hovered() || is_hovered {
        (ui.visuals().widgets.hovered.bg_fill, ui.visuals().widgets.hovered.fg_stroke)
    } else {
        (ui.visuals().widgets.inactive.bg_fill, ui.visuals().widgets.inactive.fg_stroke)
    };
    frame.frame.fill = fill;
    frame.frame.stroke = stroke;
    frame.paint(ui);

    // Get positions for io ports
    let mut inputs = vec![];
    for i in 0..num_inputs {
        inputs.push(egui::Vec2::new(-ui.min_rect().width()/2.0, (((i as f32 + 1.0) / (num_inputs as f32 + 1.0)) - 0.5) * ui.min_rect().height()));
    }
    let mut outputs = vec![];
    for i in 0..num_outputs {
        outputs.push(egui::Vec2::new(ui.min_rect().width()/2.0, (((i as f32 + 1.0) / (num_outputs as f32 + 1.0)) - 0.5) * ui.min_rect().height()));
    }
    (
        response,
        GraphLayoutIOOffsets {
            inputs,
            outputs
        }
    )
}

impl eframe::App for TemplateApp {
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
                                    self.model_load_state = None;
                                }
                                Err(err) => {
                                    if self.model_load_state.is_some() {
                                        self.model_load_state = Some(ModelLoadState::DialogOpen(Some(err)))
                                    }
                                }
                            }
                        }
                        WebsocketServerClientMessage::CurrentModelsReport(res) => {
                            self.current_models = res;
                            // Prompt tokenizer loading
                            for model in &self.current_models {
                                if let ModelTypeMetadata::LLM(llm_metadata) = &model.model_type_metadata {
                                    match &llm_metadata.tokenizer_info {
                                        TokenizerInfo::HFTokenizer(x) => {
                                            if !self.loaded_tokenizers.contains_key(&model.model_id) {
                                                self.websocket_client_server_sender.send(WebsocketClientServerMessage::GetHFTokenizer(x.clone())).unwrap();
                                                self.loaded_tokenizers.insert(model.model_id, None);
                                            }
                                        }
                                        TokenizerInfo::RWKVWorld => {
                                            if !self.loaded_tokenizers.contains_key(&model.model_id) {
                                                self.loaded_tokenizers.insert(model.model_id, Some(Ok(Arc::new(AnyTokenizer::Rwkv(WorldTokenizer::new_default())))));
                                            }
                                        }
                                    }

                                }
                            }
                        }
                        WebsocketServerClientMessage::ModelGraphReturn(res) => {
                            let (id, graph_bin) = res.unwrap();
                            if let Some(requesting_id) = self.currently_requesting_model {
                                if requesting_id == id {
                                    self.currently_requesting_model = None;
                                    let graph = ciborium::from_reader::<SymbolicGraph, _>(graph_bin.as_slice()).unwrap();
                                    self.loaded_model_graph = Some((id, graph))
                                }
                            }
                        }
                        WebsocketServerClientMessage::TensorStoreReturn(model_id, stored_tensor_id, res) => {
                            if let Some(selected_model_id) = self.app_state.graph_explorer_selected_model_id {
                                if selected_model_id == model_id {
                                    for window in &mut self.inspect_windows {
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

                            for model_entry in &self.current_models {
                                if let ModelTypeMetadata::LLM(x) = &model_entry.model_type_metadata {
                                    if let TokenizerInfo::HFTokenizer(model_hf_name) = &x.tokenizer_info {
                                        if hf_name == *model_hf_name {
                                            if let Some(x) = self.loaded_tokenizers.get_mut(&model_entry.model_id) {
                                                if x.is_none() {
                                                    *x = Some(tokenizer.clone())
                                                }
                                            }
                                        }
                                    }
                                }
                            }
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

        if let Some(model_load_state) = self.model_load_state.clone() {
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
                                self.model_load_state = Some(ModelLoadState::Loading)
                            }
                            if ui.button("Cancel").clicked() {
                                self.model_load_state = None;
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

        let mut new_explorer_hovered = None;

        egui::CentralPanel::default().show(ctx, |ui| {
            // The central panel the region left after adding TopPanel's and SidePanel's
            match &self.app_state.selected_tab {
                SelectedTab::Models => {
                    if ui.button("Load New model").clicked() {
                        self.model_load_state = Some(ModelLoadState::DialogOpen(None));
                    };
                    ui.vertical(|ui| {
                        ui.label("Loaded Models:");
                        egui::Grid::new("loaded_models").striped(true).show(ui, |ui| {
                            for model in &self.current_models {
                                ui.label(model.model_id.to_string());
                                ui.label(model.model_name.clone());
                                ui.label(if let Some(ops) = model.num_ops {format!("Operations: {:?}", ops)} else {"Operations: Unknown".to_string()});
                                match model.model_type_metadata {
                                    ModelTypeMetadata::Other => {
                                        ui.label("Other");
                                    }
                                    ModelTypeMetadata::LLM(_) => {
                                        ui.label("LLM");
                                    }
                                }
                                if let Some(x) = self.loaded_tokenizers.get(&model.model_id) {
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
                    ui.horizontal(|ui| {
                        if self.current_models.is_empty() {
                            ui.label("No Models Loaded");
                        }
                        for model in &self.current_models {
                            if ui.selectable_value(&mut self.app_state.graph_explorer_selected_model_id, Some(model.model_id), format!("({}) {}", model.model_id, model.model_name.clone())).clicked() {
                                self.explorer_selection = None;
                                self.explorer_hovered = None;
                                self.inspect_windows.clear();
                            };
                        }
                        if ui.button("Load New Model").clicked() {
                            self.model_load_state = Some(ModelLoadState::DialogOpen(None));
                        };
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            toggle_ui(ui, &mut self.app_state.explorer_minimap);
                            ui.label("Minimap:");
                            toggle_ui(ui, &mut self.app_state.explorer_physics);
                            ui.label("Physics:");
                        })
                    });

                    let model_id = if let Some(model_id) = self.app_state.graph_explorer_selected_model_id {
                        let mut ret = None;
                        for model in &self.current_models {
                            if model.model_id == model_id {
                                ret = Some(model_id)
                            }
                        }
                        ret
                    } else {
                        None
                    };

                    if let Some(model_id) = model_id {
                        let symbolic_graph = if let Some((loaded_id, loaded_graph)) = &mut self.loaded_model_graph {
                            if *loaded_id == model_id {
                                Some(loaded_graph)
                            }
                            else {
                                None
                            }
                        } else {
                            None
                        };

                        if let Some(graph) = symbolic_graph {
                            if let Some(graph_layout) = self.graph_layouts.get(&model_id) {
                                if self.app_state.explorer_minimap {
                                    let node_bounding_rect = graph_layout.get_bounding_rect();

                                    // Get frame min and max
                                    let minimap_frame = egui::Frame::default().stroke(ui.visuals().window_stroke);
                                    minimap_frame.show(ui, |ui| {
                                        let shape_request = vec2(ui.available_size_before_wrap().x, 100.0);
                                        let (outer_rect, outer_response) = ui.allocate_exact_size(shape_request, Sense::drag());
                                        let transform = outer_rect.size() /  node_bounding_rect.size();
                                        if outer_response.dragged() {
                                            let outer_pos = outer_response.interact_pointer_pos().unwrap();
                                            let inner_pos = node_bounding_rect.min + ((outer_pos - outer_rect.min) / transform);
                                            if let Some(selected_area) = self.model_view_scene_rects.get_mut(&model_id) {
                                                *selected_area = Rect::from_center_size(inner_pos, selected_area.size());
                                            }
                                        }
                                        for (_node_id, node) in graph_layout.get_nodes() {
                                            let pos = (node.position - node_bounding_rect.min) * transform;
                                            let (is_selected, is_hovered) = match &node.node_type {
                                                GraphLayoutNodeType::SymbolicGraphOperation(op_id) => {
                                                    (
                                                        if let Some(OpOrTensorId::Op(selected_op_id)) = &self.explorer_selection {
                                                            *selected_op_id == *op_id
                                                        } else {
                                                            false
                                                        },
                                                        if let Some(OpOrTensorId::Op(hovered_op_id)) = &self.explorer_hovered {
                                                            *hovered_op_id == *op_id
                                                        } else {
                                                            false
                                                        }
                                                    )
                                                }
                                                GraphLayoutNodeType::SymbolicGraphTensor(tensor_id) => {
                                                    (
                                                        if let Some(OpOrTensorId::Tensor(selected_tensor_id)) = &self.explorer_selection {
                                                            *selected_tensor_id == *tensor_id
                                                        } else {
                                                            false
                                                        },
                                                        if let Some(OpOrTensorId::Tensor(hovered_tensor_id)) = &self.explorer_hovered {
                                                            *hovered_tensor_id == *tensor_id
                                                        } else {
                                                            false
                                                        }
                                                    )
                                                }
                                                GraphLayoutNodeType::ConnectionByNameSrc(link_data) |
                                                GraphLayoutNodeType::ConnectionByNameDest(link_data) => {
                                                    match link_data.link_type {
                                                        GraphLayoutLinkType::SymbolicGraphTensor(tensor_id) => {
                                                            (
                                                                if let Some(OpOrTensorId::Tensor(selected_tensor_id)) = &self.explorer_selection {
                                                                    *selected_tensor_id == tensor_id
                                                                } else {
                                                                    false
                                                                },
                                                                if let Some(OpOrTensorId::Tensor(hovered_tensor_id)) = &self.explorer_hovered {
                                                                    *hovered_tensor_id == tensor_id
                                                                } else {
                                                                    false
                                                                }
                                                            )
                                                        }
                                                    }
                                                }
                                            };

                                            let color = if is_selected {
                                                egui::Color32::from_rgba_unmultiplied(64, 64, 255, 128)
                                            } else if is_hovered {
                                                egui::Color32::from_rgba_unmultiplied(80, 80, 80, 128)
                                            }
                                            else {
                                                egui::Color32::from_rgba_unmultiplied(32, 32, 32, 128)
                                            };
                                            let radius = if is_selected { 3.0 } else { 1.0 };

                                            ui.painter().add(egui::Shape::circle_filled(outer_rect.min + pos, radius, color));
                                        }
                                        if let Some(selected_area) = self.model_view_scene_rects.get(&model_id) {
                                            let transformed_area = Rect::from_min_max(
                                                (outer_rect.min + ((selected_area.min - node_bounding_rect.min) * transform)).max(outer_rect.min),
                                                (outer_rect.min + ((selected_area.max - node_bounding_rect.min) * transform)).min(outer_rect.max)
                                            );
                                            ui.painter().add(egui::Shape::rect_stroke(
                                                transformed_area,
                                                2.0,
                                                (1.0, egui::Color32::from_rgba_unmultiplied(64, 64, 255, 128)),
                                                StrokeKind::Inside
                                            ));
                                        }
                                    });
                                }
                            }

                            if !self.graph_layouts.contains_key(&model_id) {
                                let mut next_link_id = 0;

                                // Map tensors to link IDs
                                let mut tensor_link_ids = HashMap::new();
                                let mut link_data = HashMap::new();
                                for (tensor_id, _) in graph.get_tensors() {
                                    let new_link_id = GraphLayoutLinkId(next_link_id);
                                    next_link_id += 1;
                                    tensor_link_ids.insert(*tensor_id, new_link_id);
                                    link_data.insert(new_link_id, GraphLayoutLinkData{
                                        link_type: GraphLayoutLinkType::SymbolicGraphTensor(*tensor_id)
                                    });
                                }

                                // Build node init data for ops and I/O tensors
                                let mut next_node_id = 0;
                                let mut node_init_data = HashMap::new();
                                for (op_id, op) in graph.get_operations() {
                                    let new_node_id = GraphLayoutNodeId(next_node_id);
                                    next_node_id += 1;

                                    let mut inputs = vec![];
                                    for tensor_id in op.op.get_inputs() {
                                        inputs.push(tensor_link_ids[&tensor_id]);
                                    }
                                    let mut outputs = vec![];
                                    for tensor_id in op.op.get_outputs() {
                                        outputs.push(tensor_link_ids[&tensor_id]);
                                    }
                                    node_init_data.insert(new_node_id, GraphLayoutNodeInitData {
                                        node_type: GraphLayoutNodeType::SymbolicGraphOperation(*op_id),
                                        inputs,
                                        outputs
                                    });
                                }

                                let mut io_tensor_node_ids = HashMap::new();
                                for (tensor_id, tensor_data) in graph.get_tensors() {
                                    match tensor_data.tensor_type {
                                        TensorType::Input(_) => {
                                            let node_id = GraphLayoutNodeId(next_node_id);
                                            io_tensor_node_ids.insert(*tensor_id, node_id);
                                            next_node_id += 1;

                                            node_init_data.insert(node_id, GraphLayoutNodeInitData {
                                                node_type: GraphLayoutNodeType::SymbolicGraphTensor(*tensor_id),
                                                inputs: vec![],
                                                outputs: vec![tensor_link_ids[tensor_id]]
                                            });
                                        }
                                        TensorType::Output => {
                                            let node_id = GraphLayoutNodeId(next_node_id);
                                            io_tensor_node_ids.insert(*tensor_id, node_id);
                                            next_node_id += 1;

                                            node_init_data.insert(node_id, GraphLayoutNodeInitData {
                                                node_type: GraphLayoutNodeType::SymbolicGraphTensor(*tensor_id),
                                                inputs: vec![tensor_link_ids[tensor_id]],
                                                outputs: vec![]
                                            });
                                        }
                                        TensorType::Intermediate => {
                                            continue;
                                        }
                                        TensorType::Constant(_) => {
                                            let node_id = GraphLayoutNodeId(next_node_id);
                                            io_tensor_node_ids.insert(*tensor_id, node_id);
                                            next_node_id += 1;

                                            node_init_data.insert(node_id, GraphLayoutNodeInitData {
                                                node_type: GraphLayoutNodeType::SymbolicGraphTensor(*tensor_id),
                                                inputs: vec![],
                                                outputs: vec![tensor_link_ids[tensor_id]]
                                            });
                                        }
                                    }
                                }
                                let initial_layout = GraphLayout::new(node_init_data, link_data, ui, |ui, node_init_data| {
                                    render_node_contents(ui,
                                                         &node_init_data.node_type,
                                                         node_init_data.inputs.len(),
                                                         node_init_data.outputs.len(),
                                                         graph,
                                                         false, false).1
                                });
                                self.graph_layouts.insert(model_id, initial_layout);
                            }
                            let graph_layout = self.graph_layouts.get_mut(&model_id).unwrap();

                            // Update positions
                            if self.app_state.explorer_physics {
                                if graph_layout.update_layout(5000) {
                                    ui.ctx().request_repaint_after(Duration::from_millis(20));
                                }
                            }

                            let frame = egui::Frame::default().stroke(ui.visuals().window_stroke);
                            frame.show(ui, |ui| {
                                let frame_shape = ui.max_rect().size();
                                let mut scene_rect = self.model_view_scene_rects.get(&model_id).map(|x| x.clone()).unwrap_or(Rect::from_center_size(
                                    egui::pos2(-300.0 + frame_shape.x, 0.0),
                                    frame_shape * 2.0
                                ));
                                let cull_rect = scene_rect.expand(300.0);
                                let scene = egui::Scene::new();
                                scene.show(ui, &mut scene_rect, |ui| {

                                    // Find all ops actually in scene
                                    let mut nodes_to_render_vec = graph_layout.find_nodes_within(&cull_rect.center(), cull_rect.size().length()/2.0).clone();
                                    let mut nodes_to_render = HashSet::<GraphLayoutNodeId>::from_iter(nodes_to_render_vec.iter().map(|x| *x));

                                    let mut edges_to_render = vec![];

                                    // Render both sides of all visible edges
                                    for ((src_id, src_id_i), (dst_id, dst_id_i), link_id) in graph_layout.get_edges() {
                                        if nodes_to_render.contains(src_id) || nodes_to_render.contains(dst_id) {
                                            if !nodes_to_render.contains(dst_id) {
                                                nodes_to_render.insert(*dst_id);
                                                nodes_to_render_vec.push(*dst_id);
                                            }
                                            if !nodes_to_render.contains(src_id) {
                                                nodes_to_render.insert(*src_id);
                                                nodes_to_render_vec.push(*src_id);
                                            }
                                            // Pre-allocate the shape in the paint queue
                                            let shape_idx = ui.painter().add(Shape::Noop);
                                            edges_to_render.push(((*src_id, *src_id_i), (*dst_id, *dst_id_i), *link_id, shape_idx));
                                        }
                                    }

                                    let mut node_io_connections = HashMap::new();
                                    let mut node_bounding_boxes = HashMap::new();

                                    for node_id in nodes_to_render_vec {
                                        let current_node_data = graph_layout.get_nodes_mut();
                                        let pos = current_node_data[&node_id].position;
                                        let cell_shape = current_node_data[&node_id].shape;
                                        let op_rect = Rect::from_min_max(pos.clone() - cell_shape/2.0, pos.clone() + cell_shape);
                                        let ui_builder = UiBuilder::new().max_rect(op_rect);
                                        let mut ui_child = ui.new_child(ui_builder);

                                        let (is_selected, is_hovered) = match &current_node_data[&node_id].node_type {
                                            GraphLayoutNodeType::SymbolicGraphOperation(op_id) => {
                                                (
                                                    if let Some(OpOrTensorId::Op(selected_op_id)) = &self.explorer_selection {
                                                        *selected_op_id == *op_id
                                                    } else {
                                                        false
                                                    },
                                                    if let Some(OpOrTensorId::Op(hovered_op_id)) = &self.explorer_hovered {
                                                        *hovered_op_id == *op_id
                                                    } else {
                                                        false
                                                    }
                                                )
                                            }
                                            GraphLayoutNodeType::SymbolicGraphTensor(tensor_id) => {
                                                (
                                                    if let Some(OpOrTensorId::Tensor(selected_tensor_id)) = &self.explorer_selection {
                                                        *selected_tensor_id == *tensor_id
                                                    } else {
                                                        false
                                                    },
                                                    if let Some(OpOrTensorId::Tensor(hovered_tensor_id)) = &self.explorer_hovered {
                                                        *hovered_tensor_id == *tensor_id
                                                    } else {
                                                        false
                                                    }
                                                )
                                            }
                                            GraphLayoutNodeType::ConnectionByNameSrc(link_data) |
                                            GraphLayoutNodeType::ConnectionByNameDest(link_data) => {
                                                match link_data.link_type {
                                                    GraphLayoutLinkType::SymbolicGraphTensor(tensor_id) => {
                                                        (
                                                            if let Some(OpOrTensorId::Tensor(selected_tensor_id)) = &self.explorer_selection {
                                                                *selected_tensor_id == tensor_id
                                                            } else {
                                                                false
                                                            },
                                                            if let Some(OpOrTensorId::Tensor(hovered_tensor_id)) = &self.explorer_hovered {
                                                                *hovered_tensor_id == tensor_id
                                                            } else {
                                                                false
                                                            }
                                                        )
                                                    }
                                                }
                                            }
                                        };

                                        let (resp, io_connections) =
                                            render_node_contents(
                                                &mut ui_child,
                                                &current_node_data[&node_id].node_type,
                                                current_node_data[&node_id].inputs.len(),
                                                current_node_data[&node_id].outputs.len(),
                                                &graph, is_selected, is_hovered);
                                        node_io_connections.insert(node_id, io_connections);
                                        node_bounding_boxes.insert(node_id, ui_child.min_rect());

                                        let this_op_or_tensor_id = match &current_node_data[&node_id].node_type  {
                                            GraphLayoutNodeType::SymbolicGraphOperation(op_id) => OpOrTensorId::Op(*op_id),
                                            GraphLayoutNodeType::SymbolicGraphTensor(tensor_id) => OpOrTensorId::Tensor(*tensor_id),
                                            GraphLayoutNodeType::ConnectionByNameSrc(tensor_data) | GraphLayoutNodeType::ConnectionByNameDest(tensor_data) => {
                                                match tensor_data.link_type {
                                                    GraphLayoutLinkType::SymbolicGraphTensor(tensor_id) => {
                                                        OpOrTensorId::Tensor(tensor_id)
                                                    }
                                                }
                                            }
                                        };

                                        if resp.hovered() {
                                            new_explorer_hovered = Some(this_op_or_tensor_id.clone());
                                        }
                                        if resp.clicked() {
                                            self.explorer_selection = Some(this_op_or_tensor_id);
                                        }
                                        if resp.dragged() {
                                            current_node_data.get_mut(&node_id).unwrap().position += resp.drag_delta();
                                            current_node_data.get_mut(&node_id).unwrap().velocity = Vec2::ZERO;
                                        }
                                        if resp.double_clicked() {
                                            match &current_node_data[&node_id].node_type {
                                                GraphLayoutNodeType::SymbolicGraphOperation(op_id) => {
                                                    let new_inspect = OpOrTensorId::Op(*op_id);
                                                    if !InspectWindow::check_if_already_exists(&self.inspect_windows, &new_inspect) {
                                                        self.inspect_windows.push(InspectWindow::new(new_inspect));
                                                    }
                                                }
                                                GraphLayoutNodeType::SymbolicGraphTensor(tensor_id) => {
                                                    let new_inspect = OpOrTensorId::Tensor(*tensor_id);
                                                    if !InspectWindow::check_if_already_exists(&self.inspect_windows, &new_inspect) {
                                                        self.inspect_windows.push(InspectWindow::new(new_inspect));
                                                    }
                                                }
                                                GraphLayoutNodeType::ConnectionByNameSrc(link_data) | GraphLayoutNodeType::ConnectionByNameDest(link_data) => {
                                                    match &link_data.link_type {
                                                        GraphLayoutLinkType::SymbolicGraphTensor(tensor_id) => {
                                                            let new_inspect = OpOrTensorId::Tensor(*tensor_id);
                                                            if !InspectWindow::check_if_already_exists(&self.inspect_windows, &new_inspect) {
                                                                self.inspect_windows.push(InspectWindow::new(new_inspect));
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }

                                    // Draw lines
                                    let link_data = graph_layout.get_link_data();
                                    for ((src_id, src_id_i), (dst_id, dst_id_i), link_id, paint_idx) in edges_to_render {
                                        let source_connection = node_bounding_boxes[&src_id].center() + node_io_connections[&src_id].outputs[src_id_i];
                                        let dest_connection = node_bounding_boxes[&dst_id].center() + node_io_connections[&dst_id].inputs[dst_id_i];
                                        let points = [
                                            source_connection,
                                            egui::pos2(source_connection.x + 40.0, source_connection.y),
                                            egui::pos2(dest_connection.x - 40.0, dest_connection.y),
                                            dest_connection
                                        ];
                                        let (is_selected, is_hovered) = if let Some(link_data) = link_data.get(&link_id) {
                                            match link_data.link_type {
                                                GraphLayoutLinkType::SymbolicGraphTensor(tensor_id) => {
                                                    (
                                                        if let Some(OpOrTensorId::Tensor(selected_tensor_id)) = self.explorer_selection {
                                                            tensor_id == selected_tensor_id
                                                        }
                                                        else {
                                                            false
                                                        },
                                                        if let Some(OpOrTensorId::Tensor(selected_tensor_id)) = self.explorer_hovered {
                                                            tensor_id == selected_tensor_id
                                                        }
                                                        else {
                                                            false
                                                        }
                                                    )
                                                }
                                            }
                                        } else {
                                            (false, false)
                                        };

                                        let stroke = if is_selected {
                                            Stroke {
                                                width: ui.visuals().widgets.active.fg_stroke.width,
                                                color: egui::Color32::from_rgb(64, 64, 255)
                                            }
                                        } else if is_hovered {
                                            ui.visuals().widgets.hovered.fg_stroke
                                        } else {
                                            ui.visuals().widgets.noninteractive.fg_stroke
                                        };

                                        let shape = CubicBezierShape::from_points_stroke(points, false, Color32::TRANSPARENT, stroke);
                                        ui.painter().set(paint_idx, shape);
                                    }
                                });
                                self.model_view_scene_rects.insert(model_id, scene_rect);
                            });
                        }
                        else {
                            ui.label("Loading Model Graph...");
                            ui.spinner();
                            if self.currently_requesting_model.map(|id| id != model_id).unwrap_or(true) {
                                self.websocket_client_server_sender.send(WebsocketClientServerMessage::GetModelGraph(model_id)).unwrap();
                                self.currently_requesting_model = Some(model_id);
                            }
                        }
                    } else {
                        ui.label("No Model Selected");
                    }
                }
                SelectedTab::LLMExplorer => {
                    ui.horizontal(|ui| {
                        if self.current_models.is_empty() {
                            ui.label("No Models Loaded");
                        }
                        for model in &self.current_models {
                            if let ModelTypeMetadata::LLM(_) = model.model_type_metadata {
                                if ui.selectable_value(&mut self.app_state.graph_explorer_selected_model_id, Some(model.model_id), format!("({}) {}", model.model_id, model.model_name.clone())).clicked() {
                                    self.explorer_selection = None;
                                    self.explorer_hovered = None;
                                    self.inspect_windows.clear();
                                };
                            }
                        }
                        if ui.button("Load New Model").clicked() {
                            self.model_load_state = Some(ModelLoadState::DialogOpen(None));
                        };
                    });
                    egui::TextEdit::singleline(&mut self.app_state.current_llm_text).ui(ui);
                    let frame = egui::Frame::default().stroke(ui.visuals().window_stroke);
                    frame.show(ui, |ui| {
                        ui.label(&self.app_state.current_llm_text);
                    });
                }
            }


            /*
            if ui.button("Ping").clicked() {
                self.websocket_client_server_message.send(WebsocketClientServerMessage::Ping);

            };*/
        });

        fn format_shape(val: &Vec<ScalarInfoTyped<u64>>) -> String {
            let joined = val.iter().map(|x| match x {
                ScalarInfoTyped::Numeric(x) => {x.to_string()}
                ScalarInfoTyped::Symbolic(x) => {"?".to_string()}
            }).collect::<Vec::<_>>().join(", ");
            format!("({joined:})")
        }

        if let SelectedTab::GraphExplorer = self.app_state.selected_tab {
            let mut new_inspect_windows = vec![];
            self.inspect_windows.retain_mut(|inspect_window| {
                let mut local_open = true;
                let default_pos = ctx.input(|x| {x.pointer.latest_pos()});
                if let Some((model_id, model_graph)) = &self.loaded_model_graph {
                    if let Some(selected_model_id) = self.app_state.graph_explorer_selected_model_id {
                        if *model_id == selected_model_id {
                            match inspect_window {
                                InspectWindow::Operation(op_id) => {
                                    let mut is_hovering_tensor = false;
                                    let op_info =  &model_graph.get_operations()[op_id];
                                    let name = op_info.name.clone().map(|x| format!("Op: {x}")).unwrap_or(format!("Op Id: {op_id:?}"));
                                    let mut window = egui::Window::new(RichText::from(name.clone()).size(14.)).open(&mut local_open).resizable(false);
                                    if let Some(default_pos) = default_pos {
                                        window = window.default_pos(default_pos);
                                    }
                                    let resp = window.show(ctx, |ui| {
                                        let mut resp = ui.label(format!("ONNX Name: {:}", op_info.name.clone().unwrap_or("N/A".to_string())));
                                        resp = resp.union(ui.label(format!("Op Type: {:}", op_info.op.get_op_type_name())));
                                        resp = resp.union(ui.label("Inputs:"));
                                        fn format_tensor_row(ui: &mut egui::Ui, i: usize, tensor_id: TensorId, model_graph: &SymbolicGraph, new_inspect_windows: &mut Vec<OpOrTensorId>) -> Response {
                                            let tensor_info = model_graph.get_tensor_info(tensor_id).unwrap();
                                            let mut response = ui.label(format!("{i}"));
                                            response = response.union(ui.label(format!("{tensor_id:?}")));
                                            response = response.union(ui.label(tensor_info.onnx_name.clone().unwrap_or("N/A".to_string())));
                                            response = response.union(ui.label(tensor_info.dtype.map(|x| x.to_string()).clone().unwrap_or("N/A".to_string())));
                                            response = response.union(ui.label(tensor_info.shape().map(|x| format_shape(&x)).unwrap_or("N/A".to_string())));
                                            if ui.button("Inspect").clicked() || response.double_clicked() {
                                                new_inspect_windows.push(OpOrTensorId::Tensor(tensor_id));
                                            }
                                            ui.end_row();
                                            response
                                        }
                                        egui::Grid::new(egui::Id::new(("grid_inputs", name.clone()))).striped(true).show(ui, |ui| {
                                            for (i, tensor_id) in op_info.op.get_inputs().iter().enumerate() {
                                                let resp = format_tensor_row(ui, i, *tensor_id, &model_graph, &mut new_inspect_windows);
                                                if resp.hovered() {
                                                    new_explorer_hovered = Some(OpOrTensorId::Tensor(*tensor_id));
                                                    is_hovering_tensor = true;
                                                }
                                                if resp.clicked() {
                                                    self.explorer_selection = Some(OpOrTensorId::Tensor(*tensor_id));
                                                }
                                            }
                                        });
                                        resp = resp.union(ui.label("Outputs:"));
                                        egui::Grid::new(egui::Id::new(("grid_outputs", name))).striped(true).show(ui, |ui| {
                                            for (i, tensor_id) in op_info.op.get_outputs().iter().enumerate()  {
                                                let resp = format_tensor_row(ui, i, *tensor_id, &model_graph, &mut new_inspect_windows);
                                                if resp.hovered() {
                                                    new_explorer_hovered = Some(OpOrTensorId::Tensor(*tensor_id));
                                                    is_hovering_tensor = true;
                                                }
                                                if resp.clicked() {
                                                    self.explorer_selection = Some(OpOrTensorId::Tensor(*tensor_id));
                                                }
                                            }
                                        });
                                        match &op_info.op {
                                            AnyOperation::Constant(x) => {
                                                resp = resp.union(ui.label(format!("Value: {}", x.value.to_string())));
                                            }
                                            _ => {}
                                        }

                                        if resp.clicked() {
                                            self.explorer_selection = Some(OpOrTensorId::Op(*op_id));
                                        }
                                    });
                                    if let Some(resp) = resp {
                                        if resp.response.contains_pointer() && !is_hovering_tensor {
                                            new_explorer_hovered = Some(OpOrTensorId::Op(*op_id))
                                        }
                                    }
                                }
                                InspectWindow::Tensor(inspect_window_tensor) => {
                                    let tensor_id = inspect_window_tensor.tensor_id;
                                    let tensor_info = model_graph.get_tensor_info(tensor_id).unwrap();
                                    let name = tensor_info.onnx_name.clone().map(|x| format!("Tensor: {x}")).unwrap_or(format!("Tensor Id: {tensor_id}"));
                                    let mut window = egui::Window::new(RichText::from(name.clone()).size(14.)).open(&mut local_open).resizable(false);
                                    if let Some(default_pos) = default_pos {
                                        window = window.default_pos(default_pos);
                                    }

                                    let resp = window.show(ctx, |ui| {
                                        let mut resp = ui.label(format!("ONNX Name: {:}", tensor_info.onnx_name.clone().unwrap_or("N/A".to_string())));
                                        resp = resp.union(ui.label(format!("DType: {:}", tensor_info.dtype.map(|x| x.to_string()).clone().unwrap_or("N/A".to_string()))));
                                        resp = resp.union(ui.label(format!("Shape: {:}", tensor_info.shape().map(|x| format_shape(&x)).unwrap_or("N/A".to_string()))));
                                        let mut stored_tensor = None;
                                        match &tensor_info.tensor_type {
                                            TensorType::Input(x) => {
                                                resp = resp.union(ui.label("Tensor Type: Model Input"));
                                                if let Some(x) = x {
                                                    match x {
                                                        StoredOrNotTensor::Stored(stored_tensor_id) => {
                                                            stored_tensor = Some(*stored_tensor_id);
                                                        }
                                                        StoredOrNotTensor::NotStored(x) => {
                                                            ui.label(format!("Initial Value: {x}"));
                                                        }
                                                    };
                                                }
                                            }
                                            TensorType::Output => {
                                                resp = resp.union(ui.label("Tensor Type: Model Output"));
                                            }
                                            TensorType::Intermediate => {
                                                resp = resp.union(ui.label("Tensor Type: Intermediate"));
                                            }
                                            TensorType::Constant(x) => {
                                                resp = resp.union(ui.label("Tensor Type: Constant"));
                                                match x {
                                                    StoredOrNotTensor::Stored(stored_tensor_id) => {
                                                        stored_tensor = Some(*stored_tensor_id);
                                                    }
                                                    StoredOrNotTensor::NotStored(x) => {
                                                        resp = resp.union(ui.label(format!("Value: {x}")));
                                                    }
                                                };

                                            }
                                        }
                                        if let Some(stored_tensor_id) = stored_tensor {
                                            if inspect_window_tensor.stored_value_requested.is_none() {
                                                inspect_window_tensor.stored_value_requested = Some(stored_tensor_id);
                                                let msg = WebsocketClientServerMessage::GetStoredTensor(*model_id, stored_tensor_id);
                                                self.websocket_client_server_sender.send(msg).unwrap();
                                            }
                                            if let Some(x) = &inspect_window_tensor.stored_value {
                                                match x {
                                                    Ok(x) => {
                                                        resp = resp.union(ui.label(format!("Value: {x}")));
                                                    }
                                                    Err(err) => {
                                                        ui.scope(|ui| {
                                                            ui.visuals_mut().override_text_color = Some(egui::Color32::RED);
                                                            ui.style_mut().override_text_style = Some(egui::TextStyle::Monospace);
                                                            ui.label(err);
                                                        });
                                                    }
                                                }
                                            }
                                            else {
                                                resp = resp.union(ui.label("Loading Tensor..."));
                                                resp = resp.union(ui.spinner());
                                            }
                                        }
                                        if resp.clicked() {
                                            self.explorer_selection = Some(OpOrTensorId::Tensor(tensor_id));
                                        }
                                    });
                                    if let Some(resp) = resp {
                                        if resp.response.contains_pointer() {
                                            new_explorer_hovered = Some(OpOrTensorId::Tensor(tensor_id))
                                        }
                                    }


                                }
                            }
                        }
                    }
                }
                local_open
            });
            for new_inspect_window in new_inspect_windows {
                if !InspectWindow::check_if_already_exists(&self.inspect_windows, &new_inspect_window) {
                    self.inspect_windows.push(InspectWindow::new(new_inspect_window));
                }
            }
        }
        self.explorer_hovered = new_explorer_hovered;
    }
}

