use std::cmp::{max, Ordering};
use std::collections::{HashMap, HashSet};
use std::time::Duration;
use egui::{vec2, Color32, Label, Layout, Margin, Rect, Response, Sense, Shape, StrokeKind, UiBuilder, Vec2};
use egui::epaint::{CubicBezierShape, QuadraticBezierShape, RectShape};
use strum::IntoEnumIterator;
use web_sys::WebSocket;
use tokio::sync::mpsc;
use futures::SinkExt;
use log::{debug, info};
use rmp_serde::decode::Error;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::error::TryRecvError;
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::js_sys;
use wasm_bindgen_futures::js_sys::ArrayBuffer;
use crate::{CurrentModelsReportEntry, WebsocketClientServerMessage, WebsocketServerClientMessage};
use wasm_bindgen::prelude::*;
use rand::{random, random_range};
use whisper_tensor::symbolic_graph::{OperationId, SymbolicGraph, TensorId, TensorType};
use whisper_tensor::symbolic_graph::ops::{AnyOperation, Operation};
use crate::graph_layout::{GraphLayout, GraphLayoutNodeId, GraphLayoutNodeInitData, GraphLayoutNodeType};

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

                    match rmp_serde::from_slice::<WebsocketServerClientMessage>(&vec) {
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
        let data = rmp_serde::to_vec(&message).unwrap();
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
    Explorer,
    OtherStuff
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct AppState {
    selected_tab: SelectedTab,
    model_to_load_path_text: String,
    model_type_hint_selected: Option<onnx_import::ModelTypeHint>,
    explorer_selected_model_id: Option<u32>,
    explorer_physics: bool
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            selected_tab: SelectedTab::Models,
            model_to_load_path_text: String::new(),
            model_type_hint_selected: None,
            explorer_selected_model_id: None,
            explorer_physics: true
        }
    }
}

pub struct TemplateApp {
    model_load_state: Option<ModelLoadState>,
    websocket_server_client_receiver: mpsc::UnboundedReceiver<WebsocketServerClientMessage>,
    websocket_client_server_message: mpsc::UnboundedSender<WebsocketClientServerMessage>,
    current_models: Vec<CurrentModelsReportEntry>,
    graph_layouts: HashMap<u32, GraphLayout>,
    loaded_model_graph: Option<(u32, SymbolicGraph)>,
    currently_requesting_model: Option<u32>,
    model_view_scene_rects: HashMap<u32, Rect>,
    app_state: AppState,
    explorer_selection: Option<GraphLayoutNodeId>,
    explorer_connection_by_name_selection: Option<String>
}

impl TemplateApp {
    /// Called once before the first frame.
    pub fn new(cc: &eframe::CreationContext<'_>,
               websocket_server_client_receiver: mpsc::UnboundedReceiver<WebsocketServerClientMessage>,
               websocket_client_server_message: mpsc::UnboundedSender<WebsocketClientServerMessage>) -> Self {
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
            websocket_client_server_message,
            current_models: vec![],
            graph_layouts: HashMap::new(),
            loaded_model_graph: None,
            currently_requesting_model: None,
            model_view_scene_rects: HashMap::new(),
            app_state,
            explorer_selection: None,
            explorer_connection_by_name_selection: None
        }
    }
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
                        }
                        WebsocketServerClientMessage::ModelGraphReturn(res) => {
                            let (id, graph_bin) = res.unwrap();
                            if let Some(requesting_id) = self.currently_requesting_model {
                                if requesting_id == id {
                                    self.currently_requesting_model = None;
                                    let graph = rmp_serde::from_slice::<SymbolicGraph>(&graph_bin).unwrap();
                                    if !self.graph_layouts.contains_key(&id) {
                                        // Come up with IDs
                                        let mut op_ids = HashMap::new();
                                        let mut next_graph_id = 0;
                                        for (id, _) in graph.get_operations() {
                                            op_ids.insert(id, GraphLayoutNodeId(next_graph_id));
                                            next_graph_id += 1;
                                        }
                                        let mut io_tensor_node_ids = HashMap::new();
                                        for ((tensor_id, tensor_data)) in graph.get_tensors() {
                                            match tensor_data.tensor_type {
                                                TensorType::Input(_) => {
                                                    io_tensor_node_ids.insert(*tensor_id, GraphLayoutNodeId(next_graph_id));
                                                    next_graph_id += 1;
                                                }
                                                TensorType::Output => {
                                                    io_tensor_node_ids.insert(*tensor_id, GraphLayoutNodeId(next_graph_id));
                                                    next_graph_id += 1;
                                                }
                                                TensorType::Intermediate => {

                                                }
                                                TensorType::Constant(_) => {
                                                    io_tensor_node_ids.insert(*tensor_id, GraphLayoutNodeId(next_graph_id));
                                                    next_graph_id += 1;
                                                }
                                            }
                                        }

                                        // Map tensors to node IDs
                                        let mut all_tensor_node_ids = HashMap::new();
                                        for (id, op) in graph.get_operations() {
                                            for output in op.op.get_outputs() {
                                                all_tensor_node_ids.insert(output, op_ids[&id]);
                                            }
                                        }

                                        // Build node init data
                                        let mut node_init_data = HashMap::new();
                                        for (op_id, op) in graph.get_operations() {

                                            let op_shape = match op.op {
                                                AnyOperation::Shape(_) |
                                                AnyOperation::Cast(_) => {
                                                    egui::vec2(60.0, 40.0)
                                                }
                                                AnyOperation::Squeeze(_) |
                                                AnyOperation::Unsqueeze(_) => {
                                                    egui::vec2(100.0, 40.0)
                                                }
                                                AnyOperation::ConstantOfShape(_) => {
                                                    egui::vec2(140.0, 40.0)
                                                }
                                                AnyOperation::Reshape(_) => {
                                                    egui::vec2(100.0, 40.0)
                                                }
                                                AnyOperation::Constant(_) => {
                                                    egui::vec2(100.0, 40.0)
                                                }
                                                _ => {
                                                    // Default shape
                                                    egui::vec2(150.0, 200.0)
                                                }
                                            };
                                            let mut inputs = vec![];
                                            for tensor_id in op.op.get_inputs() {
                                                if let Some(node_id) = io_tensor_node_ids.get(&tensor_id) {
                                                    inputs.push(*node_id);
                                                }
                                                else {
                                                    if let Some(node_id) = all_tensor_node_ids.get(&tensor_id) {
                                                        inputs.push(*node_id);
                                                    }
                                                }
                                            }
                                            node_init_data.insert(op_ids[&op_id], GraphLayoutNodeInitData {
                                                node_type: GraphLayoutNodeType::SymbolicGraphOperation(*op_id),
                                                shape: op_shape,
                                                inputs,
                                            });
                                        }
                                        for (tensor_id, node_id) in io_tensor_node_ids {
                                            node_init_data.insert(node_id, GraphLayoutNodeInitData {
                                                node_type: GraphLayoutNodeType::SymbolicGraphTensor(tensor_id),
                                                shape: egui::vec2(150.0, 40.0),
                                                inputs: all_tensor_node_ids.get(&tensor_id).map(|x| vec![*x]).unwrap_or_default(),
                                            });
                                        }
                                        let initial_layout = GraphLayout::new(node_init_data);
                                        self.graph_layouts.insert(id, initial_layout);
                                    }
                                    self.loaded_model_graph = Some((id, graph))
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
                ui.selectable_value(&mut self.app_state.selected_tab, SelectedTab::Explorer, "Model Explorer");
                ui.selectable_value(&mut self.app_state.selected_tab, SelectedTab::OtherStuff, "Other Stuff");
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
                                    for which in onnx_import::ModelTypeHint::iter() {
                                        ui.selectable_value(&mut self.app_state.model_type_hint_selected, Some(which.clone()), which.to_string());
                                    }
                                });
                        });
                        ui.horizontal(|ui| {
                            if ui.button("Load").clicked() {
                                self.websocket_client_server_message.send(WebsocketClientServerMessage::LoadModel {
                                    model_path: self.app_state.model_to_load_path_text.clone(),
                                    model_type_hint: self.app_state.model_type_hint_selected.clone()
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
                                if ui.button("Unload").clicked() {
                                    self.websocket_client_server_message.send(
                                        WebsocketClientServerMessage::UnloadModel(model.model_id)
                                    ).unwrap();
                                }
                                ui.end_row();
                            }
                        });
                    });
                }
                SelectedTab::Explorer => {
                    ui.horizontal(|ui| {
                        if self.current_models.is_empty() {
                            ui.label("No Models Loaded");
                        }
                        for model in &self.current_models {
                            if ui.selectable_value(&mut self.app_state.explorer_selected_model_id, Some(model.model_id), format!("({}) {}", model.model_id, model.model_name.clone())).clicked() {
                                self.explorer_connection_by_name_selection = None;
                                self.explorer_selection = None;
                            };
                        }
                        if ui.button("Load New Model").clicked() {
                            self.model_load_state = Some(ModelLoadState::DialogOpen(None));
                        };
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.label("Physics");
                            toggle_ui(ui, &mut self.app_state.explorer_physics);
                        })
                    });

                    let model_id = if let Some(model_id) = self.app_state.explorer_selected_model_id {
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
                        let graph = if let Some((loaded_id, loaded_graph)) = &mut self.loaded_model_graph {
                            let graph_layout = self.graph_layouts.get_mut(loaded_id).unwrap();
                            if *loaded_id == model_id {
                                Some((loaded_graph, graph_layout))
                            }
                            else {
                                None
                            }
                        } else {
                            None
                        };

                        if let Some((graph, graph_layout)) = graph {
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
                                let operations = graph.get_operations();
                                let scene = egui::Scene::new();
                                scene.show(ui, &mut scene_rect, |ui| {


                                    // Find all ops actually in scene
                                    let visible_nodes_vec = graph_layout.find_nodes_within(&cull_rect.center(), cull_rect.size().length()/2.0).clone();
                                    let visible_nodes = HashSet::<GraphLayoutNodeId>::from_iter(visible_nodes_vec.iter().map(|x| *x));

                                    // Draw lines
                                    for (src_id, dst_id) in graph_layout.get_edges() {
                                        if visible_nodes.contains(src_id) || visible_nodes.contains(dst_id) {
                                            let current_node_data = graph_layout.get_nodes();
                                            let src_data = &current_node_data[src_id];
                                            let dst_data = &current_node_data[dst_id];
                                            let source_pos = src_data.position;
                                            let dest_pos = dst_data.position;
                                            let source_shape = src_data.shape;
                                            let dest_shape = dst_data.shape;
                                            let source_pos = egui::pos2(source_pos.x + source_shape.x/2.0, source_pos.y);
                                            let dest_pos = egui::pos2(dest_pos.x - dest_shape.x/2.0, dest_pos.y);
                                            let points = [
                                                source_pos,
                                                egui::pos2(source_pos.x + 40.0, source_pos.y),
                                                egui::pos2(dest_pos.x - 40.0, dest_pos.y),
                                                dest_pos
                                            ];
                                            let shape = CubicBezierShape::from_points_stroke(points, false, Color32::TRANSPARENT, ui.visuals().widgets.noninteractive.fg_stroke);
                                            ui.painter().add(shape);
                                        }
                                    }

                                    for op_id in visible_nodes_vec {
                                        let current_node_data = graph_layout.get_nodes_mut();
                                        let pos = current_node_data[&op_id].position;
                                        if cull_rect.contains(pos.clone()) {
                                            let cell_shape = current_node_data[&op_id].shape;
                                            let op_rect = Rect::from_center_size(pos.clone(), cell_shape);
                                            let mut ui_builder = UiBuilder::new().max_rect(op_rect);
                                            let mut ui_child = ui.new_child(ui_builder);

                                            let resp = ui_child.interact(op_rect, egui::Id::new(op_id), Sense::CLICK | Sense::DRAG );
                                            if resp.clicked() {
                                                self.explorer_selection = Some(op_id);
                                                match &current_node_data[&op_id].node_type {
                                                    GraphLayoutNodeType::ConnectionByNameSrc(name) => {
                                                        self.explorer_connection_by_name_selection = Some(name.clone());
                                                    }
                                                    GraphLayoutNodeType::ConnectionByNameDest(name) => {
                                                        self.explorer_connection_by_name_selection = Some(name.clone());
                                                    }
                                                    _ => {
                                                        self.explorer_connection_by_name_selection = None;
                                                    }
                                                }
                                            }
                                            if resp.dragged() {
                                                current_node_data.get_mut(&op_id).unwrap().position += resp.drag_delta();
                                                current_node_data.get_mut(&op_id).unwrap().velocity = Vec2::ZERO;
                                            }

                                            let is_selected = if let Some(selected_node_id) = self.explorer_selection {
                                                selected_node_id == op_id || match &current_node_data[&op_id].node_type {
                                                    GraphLayoutNodeType::ConnectionByNameSrc(name) => {
                                                        self.explorer_connection_by_name_selection.as_ref().map(|n| n == name).unwrap_or(false)
                                                    }
                                                    GraphLayoutNodeType::ConnectionByNameDest(name) => {
                                                        self.explorer_connection_by_name_selection.as_ref().map(|n| n == name).unwrap_or(false)
                                                    }
                                                    _ => {false}
                                                }
                                            } else {
                                                false
                                            };

                                            let rect_shape = if is_selected {
                                                RectShape::new(
                                                    op_rect,
                                                    5.0,
                                                    ui.visuals().widgets.active.bg_fill,
                                                    ui.visuals().widgets.active.fg_stroke,
                                                    StrokeKind::Inside,
                                                )
                                            } else {
                                                RectShape::new(
                                                    op_rect,
                                                    5.0,
                                                    ui.visuals().widgets.inactive.bg_fill,
                                                    ui.visuals().widgets.inactive.fg_stroke,
                                                    StrokeKind::Inside,
                                                )
                                            };
                                            let frame_shape = Shape::Rect(rect_shape);
                                            ui_child.painter().add(frame_shape);
                                            ui_child.vertical_centered(|ui| {
                                                match &current_node_data[&op_id].node_type {
                                                    GraphLayoutNodeType::SymbolicGraphOperation(op_id) => {
                                                        let op = &operations[&op_id];
                                                        if let Some(name) = &op.name {
                                                            ui.add(Label::new(name).selectable(false));
                                                        }
                                                        ui.add(Label::new(op.op.get_op_type_name()).selectable(false));
                                                    }
                                                    GraphLayoutNodeType::SymbolicGraphTensor(tensor_id) => {
                                                        let data = graph.get_tensor_info(*tensor_id).unwrap();
                                                        if let Some(name) = &data.onnx_name {
                                                            ui.put(op_rect, Label::new(name).selectable(false)) ;
                                                        } else {
                                                            ui.put(op_rect, Label::new(format!("tensor {}", tensor_id)).selectable(false));
                                                        }
                                                    }
                                                    GraphLayoutNodeType::ConnectionByNameSrc(name) => {
                                                        ui.put(op_rect, Label::new(format!("{} >", name)).selectable(false)) ;
                                                    }
                                                    GraphLayoutNodeType::ConnectionByNameDest(name) => {
                                                        ui.put(op_rect, Label::new(format!("> {}", name)).selectable(false)) ;
                                                    }
                                                    _ => {
                                                        // TODO
                                                    }
                                                }
                                            });


                                        }
                                    }
                                });
                                self.model_view_scene_rects.insert(model_id, scene_rect);
                            });
                        }
                        else {
                            ui.label("Loading Model Graph...");
                            ui.spinner();
                            if self.currently_requesting_model.map(|id| id != model_id).unwrap_or(true) {
                                self.websocket_client_server_message.send(WebsocketClientServerMessage::GetModelGraph(model_id)).unwrap();
                                self.currently_requesting_model = Some(model_id);
                            }
                        }
                    } else {
                        ui.label("No Model Selected");
                    }
                }
                SelectedTab::OtherStuff => {
                    ui.label("Other Stuff");
                }
            }


            /*
            if ui.button("Ping").clicked() {
                self.websocket_client_server_message.send(WebsocketClientServerMessage::Ping);

            };*/
        });
    }
}

