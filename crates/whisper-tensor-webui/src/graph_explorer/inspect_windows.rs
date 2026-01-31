use crate::app::LoadedModels;
use crate::graph_explorer::{
    GraphExplorerApp, GraphExplorerSettings, GraphRootSubjectSelection, LoadableGraphState,
    format_shape, get_inner_graph,
};
use crate::websockets::ServerRequestManager;
use crate::widgets::tensor_view::{TensorViewState, tensor_view};
use whisper_tensor::DynRank;
use whisper_tensor::backends::ndarray_backend::NDArrayNumericTensor;
use whisper_tensor::graph::{GlobalId, Graph, GraphDyn};
use whisper_tensor::super_graph::SuperGraph;
use whisper_tensor::super_graph::links::SuperGraphAnyLink;
use whisper_tensor::symbolic_graph::SymbolicGraph;
use whisper_tensor::symbolic_graph::tensor_store::TensorStoreTensorId;
use whisper_tensor::symbolic_graph::{StoredOrNotTensor, TensorType};
use whisper_tensor_server::{LoadedModelId, WebsocketClientServerMessage};

#[derive(Clone, Debug)]
pub(crate) struct InspectWindowGraphLink {
    pub(crate) path: Vec<GlobalId>,
    pub(crate) stored_value_requested: Option<TensorStoreTensorId>,
    pub(crate) stored_value: Option<Result<NDArrayNumericTensor<DynRank>, String>>,
    pub(crate) value_view_state: TensorViewState,
    pub(crate) subscribed_view_state: TensorViewState,
}

impl InspectWindowGraphLink {
    pub(crate) fn new(path: Vec<GlobalId>) -> Self {
        Self {
            path,
            stored_value_requested: None,
            stored_value: None,
            value_view_state: TensorViewState::default(),
            subscribed_view_state: TensorViewState::default(),
        }
    }

    pub(crate) fn to_any(self) -> AnyInspectWindow {
        AnyInspectWindow::GraphLink(self)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct InspectWindowGraphNode {
    pub(crate) path: Vec<GlobalId>,
    #[allow(dead_code)]
    pub(crate) value_view_state: TensorViewState,
}

impl InspectWindowGraphNode {
    pub(crate) fn new(path: Vec<GlobalId>) -> Self {
        Self {
            path,
            value_view_state: TensorViewState::default(),
        }
    }

    pub(crate) fn to_any(self) -> AnyInspectWindow {
        AnyInspectWindow::GraphNode(self)
    }
}

#[derive(Clone, Debug)]
pub(crate) enum AnyInspectWindow {
    GraphLink(InspectWindowGraphLink),
    GraphNode(InspectWindowGraphNode),
}

impl AnyInspectWindow {
    pub(crate) fn check_if_already_exists(windows: &[Self], subject: &[GlobalId]) -> bool {
        for window in windows {
            match window {
                AnyInspectWindow::GraphNode(node) => {
                    if node.path == *subject {
                        return true;
                    }
                }
                AnyInspectWindow::GraphLink(link) => {
                    if link.path == *subject {
                        return true;
                    }
                }
            }
        }
        false
    }
}

/// Format a GlobalId path for display (shortened version)
fn format_path_short(path: &[GlobalId]) -> String {
    if path.is_empty() {
        return "root".to_string();
    }
    if let Some(last) = path.last() {
        // Use Display impl which shows GlobalId(...)
        format!("{}", last)
    } else {
        "?".to_string()
    }
}

impl GraphExplorerApp {
    /// Resolves a path to the containing graph and the target element ID.
    /// Returns (graph containing the element, element's GlobalId)
    fn resolve_path_to_graph<'a>(
        &self,
        path: &[GlobalId],
        root_graph: &'a dyn GraphDyn,
        loaded_models: &'a LoadedModels,
    ) -> Option<(&'a dyn GraphDyn, GlobalId)> {
        if path.is_empty() {
            return None;
        }

        let mut current_graph: &dyn GraphDyn = root_graph;

        // Navigate through path, stopping before the last element
        for &node_id in &path[..path.len() - 1] {
            match get_inner_graph(current_graph, node_id, self.root_selection, loaded_models) {
                LoadableGraphState::Loaded(inner) => {
                    current_graph = inner;
                }
                _ => return None,
            }
        }

        Some((current_graph, *path.last().unwrap()))
    }

    /// Check if a node has a navigable subgraph
    fn node_has_subgraph(
        &self,
        graph: &dyn GraphDyn,
        node_id: GlobalId,
        loaded_models: &LoadedModels,
    ) -> bool {
        matches!(
            get_inner_graph(graph, node_id, self.root_selection, loaded_models),
            LoadableGraphState::Loaded(_) | LoadableGraphState::Unloaded(_)
        )
    }

    /// Render type-specific link information
    fn render_link_type_info(&self, ui: &mut egui::Ui, graph: &dyn GraphDyn, link_id: GlobalId) {
        // For SuperGraph, show link type
        if let Some(super_graph) = graph.as_any().downcast_ref::<SuperGraph>() {
            if let Some(link) = super_graph.links_by_global_id.get(&link_id) {
                let type_name = match link {
                    SuperGraphAnyLink::Tensor(_) => "Tensor",
                    SuperGraphAnyLink::String(_) => "String",
                    SuperGraphAnyLink::TensorMap(_) => "TensorMap (Model Weights)",
                    SuperGraphAnyLink::Tokenizer(_) => "Tokenizer",
                    SuperGraphAnyLink::Hash(_) => "Hash",
                };
                ui.horizontal(|ui| {
                    ui.strong("Type:");
                    ui.label(type_name);
                });
            }
        }

        // For SymbolicGraph, show tensor info (shape, dtype)
        if let Some(symbolic_graph) = graph.as_any().downcast_ref::<SymbolicGraph>() {
            if let Some(tensor_info) = symbolic_graph.get_tensors().get(&link_id) {
                if let Some(dtype) = tensor_info.dtype() {
                    ui.horizontal(|ui| {
                        ui.strong("DType:");
                        ui.label(format!("{:?}", dtype));
                    });
                }
                if let Some(shape) = tensor_info.shape() {
                    ui.horizontal(|ui| {
                        ui.strong("Shape:");
                        ui.label(format_shape(&shape));
                    });
                }

                let type_str = match &tensor_info.tensor_type {
                    TensorType::Input(_) => "Input",
                    TensorType::Output => "Output",
                    TensorType::Intermediate => "Intermediate",
                    TensorType::Constant(_) => "Constant",
                };
                ui.horizontal(|ui| {
                    ui.strong("Category:");
                    ui.label(type_str);
                });
            }
        }
    }

    /// Check if we can fetch a stored tensor for this link
    fn can_fetch_stored_tensor(&self, graph: &dyn GraphDyn, link_id: GlobalId) -> bool {
        if let Some(symbolic_graph) = graph.as_any().downcast_ref::<SymbolicGraph>() {
            if let Some(tensor_info) = symbolic_graph.get_tensors().get(&link_id) {
                if let TensorType::Constant(StoredOrNotTensor::Stored(_)) = &tensor_info.tensor_type
                {
                    return true;
                }
                if let TensorType::Input(Some(StoredOrNotTensor::Stored(_))) =
                    &tensor_info.tensor_type
                {
                    return true;
                }
            }
        }
        false
    }

    /// Find the model ID that owns a SymbolicGraph
    fn find_model_id_for_graph(
        &self,
        graph: &dyn GraphDyn,
        loaded_models: &LoadedModels,
    ) -> Option<LoadedModelId> {
        if let Some(symbolic_graph) = graph.as_any().downcast_ref::<SymbolicGraph>() {
            let target_id = <SymbolicGraph as Graph>::global_id(symbolic_graph);
            for (&model_id, model_graph) in &loaded_models.loaded_models {
                if <SymbolicGraph as Graph>::global_id(model_graph) == target_id {
                    return Some(model_id);
                }
            }
        }
        None
    }

    /// Request a stored tensor from the server
    fn request_stored_tensor(
        &self,
        window: &mut InspectWindowGraphLink,
        graph: &dyn GraphDyn,
        link_id: GlobalId,
        loaded_models: &LoadedModels,
        server_request_manager: &mut ServerRequestManager,
    ) {
        if let Some(symbolic_graph) = graph.as_any().downcast_ref::<SymbolicGraph>() {
            if let Some(tensor_info) = symbolic_graph.get_tensors().get(&link_id) {
                let stored_id = match &tensor_info.tensor_type {
                    TensorType::Constant(StoredOrNotTensor::Stored(id)) => Some(*id),
                    TensorType::Input(Some(StoredOrNotTensor::Stored(id))) => Some(*id),
                    _ => None,
                };

                if let Some(tensor_store_id) = stored_id {
                    if let Some(model_id) = self.find_model_id_for_graph(graph, loaded_models) {
                        window.stored_value_requested = Some(tensor_store_id);
                        let _ = server_request_manager.send(
                            WebsocketClientServerMessage::GetStoredTensor(
                                model_id,
                                tensor_store_id,
                            ),
                        );
                    }
                }
            }
        }
    }

    /// Render a node inspect window. Returns true if the window should stay open.
    fn render_node_inspect_window(
        &mut self,
        ctx: &egui::Context,
        window: &mut InspectWindowGraphNode,
        root_graph: &dyn GraphDyn,
        loaded_models: &LoadedModels,
    ) -> bool {
        let mut is_open = true;

        let window_id = egui::Id::new(("node_inspect", window.path.clone()));
        let title = format!("Node: {}", format_path_short(&window.path));

        egui::Window::new(title)
            .id(window_id)
            .open(&mut is_open)
            .resizable(true)
            .default_size([400.0, 300.0])
            .show(ctx, |ui| {
                if let Some((graph, node_id)) =
                    self.resolve_path_to_graph(&window.path, root_graph, loaded_models)
                {
                    if let Some(node) = graph.get_node_by_id(&node_id) {
                        // Operation type
                        ui.horizontal(|ui| {
                            ui.strong("Operation:");
                            ui.label(node.op_kind());
                        });

                        // Label (if any)
                        if let Some(label) = node.label() {
                            ui.horizontal(|ui| {
                                ui.strong("Label:");
                                ui.label(label);
                            });
                        }

                        // GlobalId
                        ui.horizontal(|ui| {
                            ui.strong("ID:");
                            ui.monospace(format!("{}", node_id));
                        });

                        // Execution timing (if available)
                        if let Some(duration) = self.node_execution_durations.get(&window.path) {
                            ui.horizontal(|ui| {
                                ui.strong("Last Execution:");
                                ui.label(format!("{:.3}ms", duration.as_secs_f64() * 1000.0));
                            });
                        }

                        ui.separator();

                        // Inputs section
                        let inputs: Vec<_> = node.inputs().collect();
                        ui.collapsing(format!("Inputs ({})", inputs.len()), |ui| {
                            for (i, input_id) in inputs.iter().enumerate() {
                                ui.horizontal(|ui| {
                                    ui.label(format!("[{}]", i));
                                    if let Some(link) = graph.get_link_by_id(input_id) {
                                        if let Some(label) = link.label() {
                                            ui.label(&label);
                                        }
                                    }
                                    ui.monospace(format!("{}", input_id));
                                });
                            }
                        });

                        // Outputs section
                        let outputs: Vec<_> = node.outputs().collect();
                        ui.collapsing(format!("Outputs ({})", outputs.len()), |ui| {
                            for (i, output_id) in outputs.iter().enumerate() {
                                ui.horizontal(|ui| {
                                    ui.label(format!("[{}]", i));
                                    if let Some(link) = graph.get_link_by_id(output_id) {
                                        if let Some(label) = link.label() {
                                            ui.label(&label);
                                        }
                                    }
                                    ui.monospace(format!("{}", output_id));
                                });
                            }
                        });

                        // Check for subgraph navigation
                        if self.node_has_subgraph(graph, node_id, loaded_models) {
                            ui.separator();
                            if ui.button("Navigate to Subgraph").clicked() {
                                self.next_graph_subject_path = Some(window.path.clone());
                            }
                        }
                    } else {
                        ui.label("Node not found in graph");
                    }
                } else {
                    ui.label("Could not resolve path");
                }
            });

        is_open
    }

    /// Render a link inspect window. Returns true if the window should stay open.
    fn render_link_inspect_window(
        &mut self,
        ctx: &egui::Context,
        window: &mut InspectWindowGraphLink,
        root_graph: &dyn GraphDyn,
        loaded_models: &LoadedModels,
        server_request_manager: &mut ServerRequestManager,
    ) -> bool {
        let mut is_open = true;
        let mut should_subscribe = false;
        let mut should_unsubscribe = false;
        let mut should_fetch_stored = false;

        let window_id = egui::Id::new(("link_inspect", window.path.clone()));
        let title = format!("Link: {}", format_path_short(&window.path));

        egui::Window::new(title)
            .id(window_id)
            .open(&mut is_open)
            .resizable(true)
            .default_size([500.0, 400.0])
            .show(ctx, |ui| {
                if let Some((graph, link_id)) = self.resolve_path_to_graph(&window.path, root_graph, loaded_models) {
                    // Link label and ID
                    if let Some(link) = graph.get_link_by_id(&link_id) {
                        if let Some(label) = link.label() {
                            ui.horizontal(|ui| {
                                ui.strong("Label:");
                                ui.label(label);
                            });
                        }
                    }

                    ui.horizontal(|ui| {
                        ui.strong("ID:");
                        ui.monospace(format!("{}", link_id));
                    });

                    // Type-specific info
                    self.render_link_type_info(ui, graph, link_id);

                    ui.separator();

                    // Subscription controls
                    let is_subscribed = self.inspect_window_tensor_subscriptions.contains(&window.path);

                    ui.horizontal(|ui| {
                        if is_subscribed {
                            if ui.button("Unsubscribe").clicked() {
                                should_unsubscribe = true;
                            }
                            ui.label("(Live updates enabled)");
                        } else {
                            if ui.button("Subscribe").clicked() {
                                should_subscribe = true;
                            }
                        }
                    });

                    ui.separator();

                    // Display tensor value
                    // Priority: subscribed value > stored value
                    if let Some(tensor) = self.inspect_window_tensor_subscription_returns.get(&window.path) {
                        ui.label("Subscribed Value:");
                        tensor_view(ui, tensor, &mut window.subscribed_view_state);
                    } else if let Some(result) = &window.stored_value {
                        match result {
                            Ok(tensor) => {
                                ui.label("Stored Value:");
                                tensor_view(ui, tensor, &mut window.value_view_state);
                            }
                            Err(err) => {
                                ui.colored_label(egui::Color32::RED, format!("Error: {}", err));
                            }
                        }
                    } else if window.stored_value_requested.is_some() {
                        ui.spinner();
                        ui.label("Loading stored value...");
                    } else {
                        // Check if this is a constant/stored tensor and offer to fetch
                        if self.can_fetch_stored_tensor(graph, link_id) {
                            if ui.button("Fetch Stored Value").clicked() {
                                should_fetch_stored = true;
                            }
                        } else {
                            ui.label("No value available. Subscribe to see live values during execution.");
                        }
                    }
                } else {
                    ui.label("Could not resolve path");
                }
            });

        // Handle deferred actions (to avoid borrow conflicts)
        if should_subscribe {
            self.inspect_window_tensor_subscriptions
                .insert(window.path.clone());
        }
        if should_unsubscribe {
            self.inspect_window_tensor_subscriptions
                .remove(&window.path);
        }
        if should_fetch_stored {
            if let Some((graph, link_id)) =
                self.resolve_path_to_graph(&window.path, root_graph, loaded_models)
            {
                self.request_stored_tensor(
                    window,
                    graph,
                    link_id,
                    loaded_models,
                    server_request_manager,
                );
            }
        }

        is_open
    }

    pub(crate) fn update_inspect_windows(
        &mut self,
        _state: &mut GraphExplorerSettings,
        ctx: &egui::Context,
        loaded_models: &mut LoadedModels,
        server_request_manager: &mut ServerRequestManager,
    ) {
        // Get root graph based on selection
        let root_graph: Option<&dyn GraphDyn> = match self.root_selection {
            GraphRootSubjectSelection::Model(model_id) => loaded_models
                .loaded_models
                .get(&model_id)
                .map(|g| g as &dyn GraphDyn),
            GraphRootSubjectSelection::Interface(interface_id) => loaded_models
                .current_interfaces
                .get(&interface_id)
                .map(|iface| iface.interface.get_super_graph() as &dyn GraphDyn),
        };

        let Some(root_graph) = root_graph else {
            return;
        };

        // Track which windows to keep and paths to unsubscribe
        let mut windows_to_keep = Vec::new();
        let mut paths_to_unsubscribe = Vec::new();

        // Take ownership temporarily to allow mutable access
        let windows = std::mem::take(&mut self.inspect_windows);

        for mut window in windows {
            let keep = match &mut window {
                AnyInspectWindow::GraphNode(node_window) => {
                    self.render_node_inspect_window(ctx, node_window, root_graph, loaded_models)
                }
                AnyInspectWindow::GraphLink(link_window) => self.render_link_inspect_window(
                    ctx,
                    link_window,
                    root_graph,
                    loaded_models,
                    server_request_manager,
                ),
            };

            if keep {
                windows_to_keep.push(window);
            } else {
                // Track subscription cleanup for closed link windows
                if let AnyInspectWindow::GraphLink(link_window) = &window {
                    paths_to_unsubscribe.push(link_window.path.clone());
                }
            }
        }

        self.inspect_windows = windows_to_keep;

        // Clean up subscriptions for closed windows
        for path in paths_to_unsubscribe {
            self.inspect_window_tensor_subscriptions.remove(&path);
        }
    }
}
