use std::collections::HashMap;
use std::time::Duration;
use egui::{Color32, Margin, Rect, Shape, StrokeKind, UiBuilder};
use egui::epaint::RectShape;
use strum::IntoEnumIterator;
use web_sys::WebSocket;
use tokio::sync::mpsc;
use futures::SinkExt;
use rmp_serde::decode::Error;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::error::TryRecvError;
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::js_sys;
use wasm_bindgen_futures::js_sys::ArrayBuffer;
use crate::{CurrentModelsReportEntry, WebsocketClientServerMessage, WebsocketServerClientMessage};
use wasm_bindgen::prelude::*;
use rand::{random, random_range};
use whisper_tensor::symbolic_graph::{OperationId, SymbolicGraph, TensorId};
use whisper_tensor::symbolic_graph::ops::Operation;

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
                    log::info!("Blob received {}bytes: {:?}", vec.len(), vec);
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

pub struct GraphLayout {
    nodes: HashMap<OperationId, egui::Pos2>,
    node_map: HashMap<(i32, i32), Vec<OperationId>>,
    node_heights: HashMap<OperationId, usize>,
    node_velocities: HashMap<OperationId, egui::Vec2>,
    edges: Vec<(OperationId, OperationId)>,
    op_inputs: HashMap<OperationId, Vec<OperationId>>,
    op_outputs: HashMap<OperationId, Vec<OperationId>>,
    layout_clock: f32
}

fn calculate_height(op_id: OperationId, graph: &SymbolicGraph, tensor_sources: &HashMap<TensorId, OperationId>, node_heights: &mut HashMap<OperationId, usize>) -> usize {
    if let Some(x) = node_heights.get(&op_id) {
        *x
    } else {
        let mut max_height = 0;
        for input in &graph.get_operations()[&op_id].op.get_inputs() {
            if let Some(input_op) = tensor_sources.get(input) {
                max_height = max_height.max(1 + calculate_height(*input_op, graph, tensor_sources, node_heights));
            }
        }
        node_heights.insert(op_id, max_height);
        max_height
    }
}

impl GraphLayout {
    
    fn get_index_for(pos: &egui::Pos2) -> (i32, i32) {
        ((pos.x/400.0) as i32, (pos.y/400.0) as i32)
    }
    
    fn find_nodes_within(&self, pos: &egui::Pos2, distance: f32) -> Vec<OperationId> {
        let top_left = egui::pos2(pos.x - distance, pos.y - distance);
        let bot_right = egui::pos2(pos.x + distance, pos.y + distance);
        let top_left_index = Self::get_index_for(&top_left);
        let bot_right_index = Self::get_index_for(&bot_right);
        let mut ret = vec![];
        for x in top_left_index.0 .. bot_right_index.0+1 {
            for y in top_left_index.1 .. bot_right_index.1+1 {
                if let Some(stuff) = self.node_map.get(&(x, y)) {
                    for op_id in stuff {
                        if let Some(op_pos) = self.nodes.get(op_id) {
                            if op_pos.distance(*pos) < distance {
                                ret.push(*op_id)
                            }
                        }
                    }
                }
            }
        }
        ret
    }
    
    fn new(graph: &SymbolicGraph) -> Self {
        // Get sources of all tensors
        let mut tensor_sources = HashMap::new();
        for (op_id, b) in graph.get_operations() {
            for output in b.op.get_outputs() {
                tensor_sources.insert(output, *op_id);
            }
        }
        // Get heights
        let mut max_height = 0;
        let mut node_heights = HashMap::new();
        for (op_id, _op) in graph.get_operations() {
            let height = calculate_height(*op_id, graph, &tensor_sources, &mut node_heights);
            max_height = max_height.max(height);
        }
        let mut nodes = HashMap::new();
        let mut node_map = HashMap::new();
        let mut node_velocities = HashMap::new();
        for (op_id, _op) in graph.get_operations() {
            let height = node_heights[op_id];
            let pos = egui::pos2(random_range(-300.0 .. 300.0), height as f32 * 100.0);
            nodes.insert(*op_id, pos);
            node_map.entry(Self::get_index_for(&pos)).or_insert(Vec::new()).push(*op_id);
            node_velocities.insert(*op_id, egui::vec2(random::<f32>(), random::<f32>()));
        }

        // Draw lines
        let mut edges = Vec::new();
        let mut op_inputs = HashMap::new();
        let mut op_outputs = HashMap::new();
        for (op_id, b) in graph.get_operations() {
            for input in b.op.get_inputs() {
                if let Some(source) = tensor_sources.get(&input) {
                    edges.push((*source, *op_id));
                    op_inputs.entry(*op_id).or_insert(Vec::new()).push(*source);
                    op_outputs.entry(*source).or_insert(Vec::new()).push(*op_id);
                }
            }
        }

        Self {
            nodes,
            node_map,
            node_velocities,
            edges,
            op_inputs,
            op_outputs,
            node_heights,
            layout_clock: 0.0
        }
    }

    fn get_current_layout(&self) -> (&HashMap<OperationId, egui::Pos2>, &Vec<(OperationId, OperationId)>) {
        (&self.nodes, &self.edges)
    }

    fn update_layout(&mut self, max_nodes_to_update: u32) -> bool {
        let mut did_update = false;
        for _ in 0..max_nodes_to_update {
            let i = random_range(0..self.nodes.len());
            let op_id = *self.nodes.keys().nth(i).unwrap();
            let node_pos = &self.nodes[&op_id];
            let mut applied_force = (0.0, 0.0);
            for other_node in self.find_nodes_within(node_pos, 400.0) {
                if other_node == op_id {
                    continue;
                }
                let other_node_pos = self.nodes[&other_node];
                let delta = (node_pos.x - other_node_pos.x, node_pos.y - other_node_pos.y);
                let distance = (delta.0 * delta.0 + delta.1 * delta.1).sqrt();
                let normalized_delta = (delta.0 / distance, delta.1 / distance);

                if distance > 0.01 {
                    let force = if delta.0.abs() < 300.0 && delta.1.abs() < 120.0 {
                        (100.0 / distance)
                    }
                    else {
                        0.0
                    }.min(self.layout_clock / 2000.0);
                    applied_force.0 += force * normalized_delta.0;
                    applied_force.1 += force * normalized_delta.1;
                }
            }
            let mut links = vec![];
            if let Some(inputs) = self.op_inputs.get(&op_id) {
                for input in inputs {
                    links.push((*input, op_id));
                }
            }
            if let Some(outputs) = self.op_outputs.get(&op_id) {
                for output in outputs {
                    links.push((op_id, *output));
                }
            }
            
            for (src, dst) in links {
                // Applied to dst, inverse applied to src
                let mut link_force = (0.0, 0.0);
                
                let delta = (self.nodes[&dst].x - self.nodes[&src].x, self.nodes[&dst].y - self.nodes[&src].y);
                let distance = (delta.0 * delta.0 + delta.1 * delta.1).sqrt();
                let normalized_delta = (delta.0 / distance, delta.1 / distance);
                let force = if distance > 100.0 {
                    (-distance / 300.0).min(0.5)
                } else {
                    0.0
                };
                link_force.0 = force * normalized_delta.0;
                link_force.1 = force * normalized_delta.1;
                if self.nodes[&src].y + 100.0 > self.nodes[&dst].y {
                    // Inputs must be pushed above the node
                    link_force.1 += 3.0;
                }
                
                if dst == op_id {
                    applied_force.0 += link_force.0;
                    applied_force.1 += link_force.1;
                }
                if src == op_id {
                    applied_force.0 -= link_force.0;
                    applied_force.1 -= link_force.1;
                }
            }
            
            // Weak draw towards 0,0
            applied_force.0 -= node_pos.x * 0.0002;
            applied_force.1 -= node_pos.y * 0.0002;

            // Constrain force
            let temperature = (4000.0 / self.layout_clock).min(1.0);
            let applied_force = (applied_force.0 * temperature, applied_force.1 * temperature);
            let mut velocity = self.node_velocities[&op_id];
            velocity.x += applied_force.0;
            velocity.y += applied_force.1;
            // Dampen
            velocity.x -= velocity.x * 0.3;
            velocity.y -= velocity.y * 0.3;
            // Clip velocity magnitude
            let velocity_magnitude  = velocity.length();
            let clipped_velocity = velocity_magnitude.min(1.0);
            let velocity = egui::vec2(clipped_velocity*velocity.x/velocity_magnitude, clipped_velocity*velocity.y/velocity_magnitude);
            
            let velocity_magnitude  = velocity.length();
            let min_movement =  self.layout_clock / 400000.0;
            if velocity_magnitude.is_finite() && velocity_magnitude > min_movement {
                did_update = true;
                let old_index = Self::get_index_for(&node_pos);
                let new_position = egui::pos2(node_pos.x + velocity.x, node_pos.y + velocity.y);
                let new_index = Self::get_index_for(&new_position);
                if old_index != new_index {
                    // Update position on map
                    if let Some(x) = self.node_map.get_mut(&old_index) {
                        // Remove
                        x.retain_mut(|x| {*x != op_id});
                    } else {  
                        // Should not be possible
                        panic!();
                    }
                    // Add to map
                    self.node_map.entry(new_index).or_insert(Vec::new()).push(op_id);
                }
                
                self.nodes.insert(op_id, new_position);
                self.node_velocities.insert(op_id, velocity);
            } else {
                self.node_velocities.insert(op_id, egui::vec2(0.0, 0.0));
            }
        }
        self.layout_clock += max_nodes_to_update as f32 / self.nodes.len() as f32;
        did_update
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct AppState {
    selected_tab: SelectedTab,
    model_to_load_path_text: String,
    model_type_hint_selected: Option<onnx_import::ModelTypeHint>,
    explorer_selected_model_id: Option<u32>,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            selected_tab: SelectedTab::Models,
            model_to_load_path_text: String::new(),
            model_type_hint_selected: None,
            explorer_selected_model_id: None,
        }
    }
}

pub struct TemplateApp {
    model_load_state: Option<ModelLoadState>,
    websocket_server_client_receiver: mpsc::UnboundedReceiver<WebsocketServerClientMessage>,
    websocket_client_server_message: mpsc::UnboundedSender<WebsocketClientServerMessage>,
    current_models: Vec<CurrentModelsReportEntry>,
    loaded_model_graph: Option<(u32, SymbolicGraph, GraphLayout)>,
    currently_requesting_model: Option<u32>,
    model_view_scene_rect: Option<Rect>,
    app_state: AppState
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
            loaded_model_graph: None,
            currently_requesting_model: None,
            model_view_scene_rect: None,
            app_state
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
                                    let mut initial_layout = GraphLayout::new(&graph);
                                    self.model_view_scene_rect = None;
                                    self.loaded_model_graph = Some((id, graph, initial_layout))
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
                ui.selectable_value(&mut self.app_state.selected_tab, SelectedTab::Explorer, "ONNX Explorer");
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
                        for model in &self.current_models {
                            ui.horizontal(|ui| {
                                ui.label(model.model_id.to_string());
                                ui.label(model.model_name.clone());
                                if ui.button("Unload").clicked() {
                                    self.websocket_client_server_message.send(
                                        WebsocketClientServerMessage::UnloadModel(model.model_id)
                                    ).unwrap();
                                }
                            });
                        }
                    });
                }
                SelectedTab::Explorer => {
                    ui.horizontal(|ui| {
                        if self.current_models.is_empty() {
                            ui.label("No Models Loaded");
                        }
                        for model in &self.current_models {
                            ui.selectable_value(&mut self.app_state.explorer_selected_model_id, Some(model.model_id), format!("({}) {}", model.model_id, model.model_name.clone()));
                        }
                        if ui.button("Load New Model").clicked() {
                            self.model_load_state = Some(ModelLoadState::DialogOpen(None));
                        };
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
                        let graph = if let Some((loaded_id, loaded_graph, graph_layout)) = &mut self.loaded_model_graph {
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
                            if graph_layout.update_layout(5000) {
                                ui.ctx().request_repaint_after(Duration::from_millis(20));
                            }
                            let (current_graph_layout, current_edge_layout) = graph_layout.get_current_layout();
                            let frame = egui::Frame::default().stroke(ui.visuals().window_stroke);
                            frame.show(ui, |ui| {
                                let frame_shape = ui.max_rect().size();
                                let mut scene_rect = self.model_view_scene_rect.unwrap_or(Rect::from_center_size(
                                    egui::pos2(0.0, -200.0 + frame_shape.y/2.0),
                                    frame_shape
                                ));
                                let cull_rect = scene_rect.expand(300.0);
                                let operations = graph.get_operations();
                                let scene = egui::Scene::new();
                                scene.show(ui, &mut scene_rect, |ui| {
                                    // Draw lines

                                    // Draw lines
                                    for (a, b) in current_edge_layout {
                                        let source_pos = current_graph_layout[a];
                                        let dest_pos = current_graph_layout[b];
                                        ui.painter().line_segment([source_pos, dest_pos], (1.0, ui.visuals().widgets.inactive.fg_stroke.color));
                                    }

                                    for op_id in graph_layout.find_nodes_within(&cull_rect.center(), cull_rect.size().length()/2.0) {
                                        let pos = current_graph_layout[&op_id];
                                        if cull_rect.contains(pos.clone()) {
                                            let op_rect = Rect::from_center_size(pos.clone(), egui::vec2(120.0, 50.0));
                                            let mut ui_child = ui.new_child(
                                                UiBuilder::new().max_rect(op_rect)
                                            );
                                            let frame_shape = Shape::Rect(RectShape::new(
                                                op_rect,
                                                5.0,
                                                ui.visuals().window_fill,
                                                ui.visuals().window_stroke,
                                                StrokeKind::Inside,
                                            ));
                                            ui_child.painter().add(frame_shape);
                                            ui_child.vertical_centered(|ui| {
                                                let op = &operations[&op_id];
                                                if let Some(name) = &op.name {
                                                    ui.label(name);
                                                }
                                                ui.label(op.op.get_op_type_name());
                                            });
                                        }
                                    }
                                });
                                self.model_view_scene_rect = Some(scene_rect);
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

