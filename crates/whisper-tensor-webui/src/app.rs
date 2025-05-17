use std::time::Duration;
use egui::{Margin, Rect, UiBuilder};
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
use whisper_tensor::symbolic_graph::{OperationId, SymbolicGraph};
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

#[derive(Clone, Debug, PartialEq)]
enum SelectedTab {
    Models,
    Explorer,
    OtherStuff
}

pub struct GraphLayout {
    nodes: Vec<(OperationId, egui::Pos2)>
}

impl GraphLayout {
    fn new(graph: &SymbolicGraph) -> Self {
        let mut nodes = vec![];
        for op in graph.get_operations() {
            nodes.push((*op.0, egui::pos2(random::<f32>(), random::<f32>())));
        }
        Self {
            nodes
        }
    }

    fn get_current_layout(&self) -> &[(OperationId, egui::Pos2)] {
        &self.nodes
    }
    
    fn update_layout(&mut self, max_nodes_to_update: u32) {
        // Every node gets a cycle
        for _ in 0..max_nodes_to_update {
            let i = random_range(0..self.nodes.len());
            let (node_id, node_pos) = &self.nodes[i];
            let mut applied_force = (0.0, 0.0);
            for other_node in &self.nodes {
                if other_node.0 == *node_id {
                    continue;
                }
                let delta = (node_pos.x - other_node.1.x, node_pos.y - other_node.1.y);
                let distance = (delta.0 * delta.0 + delta.1 * delta.1).sqrt();
                if distance < 20.0 {
                    let force = 10.0 / distance;
                    applied_force.0 += force * delta.0;
                    applied_force.1 += force * delta.1;
                }
                
            }
            // Constrain force
            let force_magnitude = (applied_force.0 * applied_force.0 + applied_force.1 * applied_force.1).sqrt();
            let normalized_force = (applied_force.0 / force_magnitude, applied_force.1 / force_magnitude);
            let force_magnitude = force_magnitude.min(1.0);
            let applied_force = (normalized_force.0 * force_magnitude, normalized_force.1 * force_magnitude);
            self.nodes[i].1 = egui::pos2(node_pos.x + applied_force.0, node_pos.y + applied_force.1);
        }
    }
}

/// We derive Deserialize/Serialize so we can persist app state on shutdown.
// if we add new fields, give them default values when deserializing old state
pub struct TemplateApp {
    model_load_state: Option<ModelLoadState>,
    model_to_load_path_text: String,
    model_type_hint_selected: Option<onnx_import::ModelTypeHint>,
    websocket_server_client_receiver: mpsc::UnboundedReceiver<WebsocketServerClientMessage>,
    websocket_client_server_message: mpsc::UnboundedSender<WebsocketClientServerMessage>,
    current_models: Vec<CurrentModelsReportEntry>,
    explorer_selected_model_id: Option<u32>,
    selected_tab: SelectedTab,
    loaded_model_graph: Option<(u32, SymbolicGraph, GraphLayout)>,
    currently_requesting_model: Option<u32>,
    model_view_scene_rect: Option<Rect>,
}



impl TemplateApp {
    /// Called once before the first frame.
    pub fn new(_cc: &eframe::CreationContext<'_>,
               websocket_server_client_receiver: mpsc::UnboundedReceiver<WebsocketServerClientMessage>,
               websocket_client_server_message: mpsc::UnboundedSender<WebsocketClientServerMessage>) -> Self {
        // This is also where you can customize the look and feel of egui using
        // `cc.egui_ctx.set_visuals` and `cc.egui_ctx.set_fonts`.

        // Load previous app state (if any).
        // Note that you must enable the `persistence` feature for this to work.
        /*
        if let Some(storage) = cc.storage {
            return eframe::get_value(storage, eframe::APP_KEY).unwrap_or_default();
        }*/

        Self {
            model_load_state: None,
            model_to_load_path_text: String::new(),
            model_type_hint_selected: None,
            websocket_server_client_receiver,
            websocket_client_server_message,
            current_models: vec![],
            selected_tab: SelectedTab::Models,
            explorer_selected_model_id: None,
            loaded_model_graph: None,
            currently_requesting_model: None,
            model_view_scene_rect: None,
        }
    }
}

impl eframe::App for TemplateApp {
    /// Called by the frame work to save state before shutdown.
    fn save(&mut self, _storage: &mut dyn eframe::Storage) {
        //eframe::set_value(storage, eframe::APP_KEY, self);
    }

    /// Called each time the UI needs repainting, which may be many times per second.
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Put your widgets into a `SidePanel`, `TopBottomPanel`, `CentralPanel`, `Window` or `Area`.
        // For inspiration and more examples, go to https://emilk.github.io/egui

        ctx.set_zoom_factor(1.2);

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
                                    let initial_layout = GraphLayout::new(&graph);
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
                ui.selectable_value(&mut self.selected_tab, SelectedTab::Models, "Manage Models");
                ui.selectable_value(&mut self.selected_tab, SelectedTab::Explorer, "ONNX Explorer");
                ui.selectable_value(&mut self.selected_tab, SelectedTab::OtherStuff, "Other Stuff");
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
                            ui.text_edit_singleline(&mut self.model_to_load_path_text);
                            egui::ComboBox::from_id_salt(1245).selected_text(if let Some(which) = &self.model_type_hint_selected {which.to_string()} else {String::from("No Hint")})
                                .show_ui(ui, |ui|{
                                    ui.selectable_value(&mut self.model_type_hint_selected, None, "No Hint");
                                    for which in onnx_import::ModelTypeHint::iter() {
                                        ui.selectable_value(&mut self.model_type_hint_selected, Some(which.clone()), which.to_string());
                                    }
                                });
                        });
                        ui.horizontal(|ui| {
                            if ui.button("Load").clicked() {
                                self.websocket_client_server_message.send(WebsocketClientServerMessage::LoadModel {
                                    model_path: self.model_to_load_path_text.clone(),
                                    model_type_hint: self.model_type_hint_selected.clone()
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
            match &self.selected_tab {
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
                            ui.selectable_value(&mut self.explorer_selected_model_id, Some(model.model_id), format!("({}) {}", model.model_id, model.model_name.clone()));
                        }
                        if ui.button("Load New Model").clicked() {
                            self.model_load_state = Some(ModelLoadState::DialogOpen(None));
                        };
                    });

                    let model_id = if let Some(model_id) = self.explorer_selected_model_id {
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
                            graph_layout.update_layout(100);
                            ui.ctx().request_repaint_after(Duration::from_millis(20));
                            let frame = egui::Frame::default().stroke(ui.visuals().window_stroke);
                            frame.show(ui, |ui| {
                                let mut scene_rect = self.model_view_scene_rect.unwrap_or(egui::Rect::from_center_size(egui::pos2(0.0, 0.0), egui::vec2(1000.0, 1000.0)));
                                let cull_rect = scene_rect.expand(0.0);
                                let operations = graph.get_operations();
                                let scene = egui::Scene::new();
                                scene.show(ui, &mut scene_rect, |ui| {
                                    for (id, pos) in graph_layout.get_current_layout() {
                                        if cull_rect.contains(pos.clone()) {
                                            let mut ui_child = ui.new_child(
                                                UiBuilder::new().max_rect(Rect::from_center_size(pos.clone(), egui::vec2(100.0, 50.0)))
                                            );
                                            let mut frame = egui::Frame::default().stroke(ui.visuals().window_stroke);
                                            let mut prepared = frame.begin(&mut ui_child);

                                            //area = area.fixed_pos(pos.clone());
                                            let op = &operations[id];
                                            if let Some(name) = &op.name {
                                                prepared.content_ui.label(name);
                                            }
                                            prepared.content_ui.label(op.op.get_op_type_name());

                                            prepared.end(&mut ui_child);
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

