use crate::app::LoadedModels;
use crate::graph_explorer::graph_layout::GraphLayout;
use crate::graph_explorer::{
    GraphExplorerApp, GraphExplorerLayerSelection, GraphExplorerSelectable, GraphExplorerState,
    GraphSubject, format_shape,
};
use crate::websockets::ServerRequestManager;
use crate::widgets::tensor_view::{TensorViewState, tensor_view};
use egui::{Response, RichText};
use rand::random_range;
use std::collections::HashSet;
use whisper_tensor::DynRank;
use whisper_tensor::backends::ndarray_backend::NDArrayNumericTensor;
use whisper_tensor::graph::Node;
use whisper_tensor::super_graph::nodes::SuperGraphAnyNode;
use whisper_tensor::symbolic_graph::ops::{AnyOperation, Operation};
use whisper_tensor::symbolic_graph::tensor_store::TensorStoreTensorId;
use whisper_tensor::symbolic_graph::{
    StoredOrNotTensor, SymbolicGraphInner, SymbolicGraphOperationId, SymbolicGraphTensorId,
    TensorType,
};
use whisper_tensor_server::WebsocketClientServerMessage;

#[derive(Clone, Debug)]
pub(crate) struct InspectWindowSymbolicGraphTensor {
    pub(crate) tensor_id: SymbolicGraphTensorId,
    pub(crate) stored_value_requested: Option<TensorStoreTensorId>,
    pub(crate) stored_value: Option<Result<NDArrayNumericTensor<DynRank>, String>>,
    pub(crate) value_view_state: TensorViewState,
    pub(crate) subscribed_view_state: TensorViewState,
}

#[derive(Clone, Debug)]
pub(crate) struct InspectWindowSymbolicGraphOperation {
    pub(crate) op_id: SymbolicGraphOperationId,
    pub(crate) value_view_state: TensorViewState,
}

#[derive(Clone, Debug)]
pub(crate) enum InspectWindow {
    SymbolicGraphOperation(InspectWindowSymbolicGraphOperation),
    SymbolicGraphTensor(InspectWindowSymbolicGraphTensor),
}

impl InspectWindow {
    fn to_graph_selectable(&self) -> GraphExplorerSelectable {
        match self {
            InspectWindow::SymbolicGraphOperation(x) => {
                GraphExplorerSelectable::SymbolicGraphOperationId(x.op_id)
            }
            InspectWindow::SymbolicGraphTensor(x) => {
                GraphExplorerSelectable::SymbolicGraphTensorId(x.tensor_id)
            }
        }
    }

    pub(crate) fn check_if_already_exists(
        windows: &[Self],
        subject: &GraphExplorerSelectable,
    ) -> bool {
        for window in windows {
            if window.to_graph_selectable() == *subject {
                return true;
            }
        }
        false
    }

    pub(crate) fn new(subject: GraphExplorerSelectable) -> Option<Self> {
        match subject {
            GraphExplorerSelectable::SymbolicGraphOperationId(op_id) => Some(
                Self::SymbolicGraphOperation(InspectWindowSymbolicGraphOperation {
                    op_id,
                    value_view_state: TensorViewState::default(),
                }),
            ),
            GraphExplorerSelectable::SymbolicGraphTensorId(tensor_id) => Some(
                Self::SymbolicGraphTensor(InspectWindowSymbolicGraphTensor {
                    tensor_id,
                    stored_value_requested: None,
                    stored_value: None,
                    value_view_state: TensorViewState::default(),
                    subscribed_view_state: TensorViewState::default(),
                }),
            ),
            _ => None,
        }
    }
}

impl GraphExplorerApp {
    pub(crate) fn update_inspect_windows(
        &mut self,
        _state: &mut GraphExplorerState,
        ctx: &egui::Context,
        loaded_models: &mut LoadedModels,
        server_request_manager: &mut ServerRequestManager,
    ) {
        // Resolve context
        let (graph_subject, model_scope) = {
            let mut graph_subject = None;
            let mut implied_model = None;
            let mut model_scope = None;
            for a in &self.graph_subject_path {
                match a {
                    GraphExplorerLayerSelection::Model(model_id) => {
                        if let Some(x) = self.loaded_models.get(model_id) {
                            graph_subject = Some(GraphSubject::SymbolicGraphInner(&x.inner_graph));
                            model_scope = Some(model_id);
                        } else {
                            graph_subject = None;
                        }
                    }
                    GraphExplorerLayerSelection::Interface(interface_id) => {
                        if let Some(x) = loaded_models.current_interfaces.get(interface_id) {
                            implied_model = x.model_ids.first();
                            graph_subject = Some(GraphSubject::SuperGraphInner(
                                &x.interface.get_super_graph().inner,
                            ));
                        } else {
                            graph_subject = None;
                        }
                    }
                    GraphExplorerLayerSelection::SymbolicGraphOperationId((
                        op_id,
                        sub_graph_idx,
                    )) => {
                        if let Some(GraphSubject::SymbolicGraphInner(symbolic_graph)) =
                            graph_subject
                            && let Some(x) = symbolic_graph.get_operations().get(op_id)
                            && let Some(inner_graph) = x.op.get_sub_graphs().get(*sub_graph_idx)
                        {
                            graph_subject = Some(GraphSubject::SymbolicGraphInner(inner_graph));
                        } else {
                            graph_subject = None;
                        }
                    }
                    GraphExplorerLayerSelection::SuperGraphNodeId(node_id) => {
                        if let Some(GraphSubject::SuperGraphInner(graph)) = graph_subject
                            && let Some(node) = graph.nodes.get(node_id)
                        {
                            match node {
                                SuperGraphAnyNode::ModelExecution(_) => {
                                    if let Some(model_id) = implied_model
                                        && let Some(model_graph) = self.loaded_models.get(model_id)
                                    {
                                        model_scope = Some(model_id);
                                        graph_subject = Some(GraphSubject::SymbolicGraphInner(
                                            &model_graph.inner_graph,
                                        ));
                                    } else {
                                        graph_subject = None;
                                    }
                                }
                                SuperGraphAnyNode::MilliOpGraph(node) => {
                                    graph_subject = Some(GraphSubject::MilliOpGraphB(&node.graph));
                                }
                                _ => {
                                    if let Some(x) = node.get_sub_graph() {
                                        graph_subject = Some(GraphSubject::SuperGraphInner(x));
                                    } else {
                                        graph_subject = None;
                                    }
                                }
                            }
                        } else {
                            graph_subject = None;
                        }
                    }
                }
            }
            (graph_subject, model_scope)
        };
        let mut new_subscribed_tensors = HashSet::new();
        let super_graph_path = self.get_super_graph_graph_path(loaded_models);
        if let Some(graph_subject) = graph_subject {
            let mut new_inspect_windows = vec![];
            self.inspect_windows.retain_mut(|inspect_window| {
                let mut local_open = true;
                let default_pos = ctx.input(|x| x.pointer.latest_pos());
                match (inspect_window, &graph_subject) {
                    (
                        InspectWindow::SymbolicGraphOperation(inspect_window),
                        GraphSubject::SymbolicGraphInner(model_graph),
                    ) => {
                        let op_id = inspect_window.op_id;
                        let mut is_hovering_tensor = false;
                        let op_info = &model_graph.get_operations()[&op_id];
                        let name = op_info
                            .name
                            .clone()
                            .map(|x| format!("Op: {x}"))
                            .unwrap_or(format!("Op Id: {op_id:?}"));
                        let mut window = egui::Window::new(RichText::from(name.clone()).size(14.))
                            .open(&mut local_open)
                            .resizable(false);
                        if let Some(default_pos) = default_pos {
                            window = window.default_pos(default_pos);
                        }
                        let resp = window.show(ctx, |ui| {
                            let mut resp = ui.label(format!(
                                "ONNX Name: {:}",
                                op_info.name.clone().unwrap_or("N/A".to_string())
                            ));
                            resp =
                                resp.union(ui.label(format!("Op Type: {:}", op_info.op.op_kind())));
                            if let Some(super_graph_path) = &super_graph_path {
                                let node_path = super_graph_path.push_symbolic_node(op_id);
                                if let Some(x) = self.node_execution_durations.get(&node_path) {
                                    resp = resp.union(
                                        ui.label(format!("Last execution duration: {:?}", x)),
                                    );
                                }
                            }
                            resp = resp.union(ui.label("Inputs:"));
                            fn format_tensor_row(
                                ui: &mut egui::Ui,
                                i: usize,
                                tensor_id: SymbolicGraphTensorId,
                                model_graph: &SymbolicGraphInner,
                                new_inspect_windows: &mut Vec<GraphExplorerSelectable>,
                            ) -> Response {
                                let tensor_info = model_graph.get_tensor_info(tensor_id).unwrap();
                                let mut response = ui.label(format!("{i}"));
                                response = response.union(ui.label(format!("{tensor_id:?}")));
                                response = response.union(ui.label(
                                    tensor_info.onnx_name.clone().unwrap_or("N/A".to_string()),
                                ));
                                response = response.union(
                                    ui.label(
                                        tensor_info
                                            .dtype
                                            .map(|x| x.to_string())
                                            .clone()
                                            .unwrap_or("N/A".to_string()),
                                    ),
                                );
                                response = response.union(
                                    ui.label(
                                        tensor_info
                                            .shape()
                                            .map(|x| format_shape(&x))
                                            .unwrap_or("N/A".to_string()),
                                    ),
                                );
                                if ui.button("Inspect").clicked() || response.double_clicked() {
                                    new_inspect_windows.push(
                                        GraphExplorerSelectable::SymbolicGraphTensorId(tensor_id),
                                    );
                                }
                                ui.end_row();
                                response
                            }
                            egui::Grid::new(egui::Id::new(("grid_inputs", name.clone())))
                                .striped(true)
                                .show(ui, |ui| {
                                    for (i, tensor_id) in op_info.op.inputs().enumerate() {
                                        let resp = format_tensor_row(
                                            ui,
                                            i,
                                            tensor_id,
                                            &model_graph,
                                            &mut new_inspect_windows,
                                        );
                                        if resp.hovered() {
                                            self.next_explorer_hovered = Some(
                                                GraphExplorerSelectable::SymbolicGraphTensorId(
                                                    tensor_id,
                                                ),
                                            );
                                            is_hovering_tensor = true;
                                        }
                                        if resp.clicked() {
                                            self.explorer_selection = Some(
                                                GraphExplorerSelectable::SymbolicGraphTensorId(
                                                    tensor_id,
                                                ),
                                            );
                                        }
                                    }
                                });
                            resp = resp.union(ui.label("Outputs:"));
                            egui::Grid::new(egui::Id::new(("grid_outputs", name)))
                                .striped(true)
                                .show(ui, |ui| {
                                    for (i, tensor_id) in op_info.op.outputs().enumerate() {
                                        let resp = format_tensor_row(
                                            ui,
                                            i,
                                            tensor_id,
                                            &model_graph,
                                            &mut new_inspect_windows,
                                        );
                                        if resp.hovered() {
                                            self.next_explorer_hovered = Some(
                                                GraphExplorerSelectable::SymbolicGraphTensorId(
                                                    tensor_id,
                                                ),
                                            );
                                            is_hovering_tensor = true;
                                        }
                                        if resp.clicked() {
                                            self.explorer_selection = Some(
                                                GraphExplorerSelectable::SymbolicGraphTensorId(
                                                    tensor_id,
                                                ),
                                            );
                                        }
                                    }
                                });
                            match &op_info.op {
                                AnyOperation::Constant(x) => {
                                    resp = resp.union(tensor_view(
                                        ui,
                                        &x.value,
                                        &mut inspect_window.value_view_state,
                                    ));
                                }
                                _ => {}
                            }

                            if resp.clicked() {
                                self.explorer_selection =
                                    Some(GraphExplorerSelectable::SymbolicGraphOperationId(op_id));
                            }
                        });
                        if let Some(resp) = resp {
                            if resp.response.contains_pointer() && !is_hovering_tensor {
                                self.next_explorer_hovered =
                                    Some(GraphExplorerSelectable::SymbolicGraphOperationId(op_id))
                            }
                        }
                    }
                    (
                        InspectWindow::SymbolicGraphTensor(inspect_window_tensor),
                        GraphSubject::SymbolicGraphInner(model_graph),
                    ) => {
                        let tensor_id = inspect_window_tensor.tensor_id;
                        let tensor_info = model_graph.get_tensor_info(tensor_id).unwrap();
                        let name = tensor_info
                            .onnx_name
                            .clone()
                            .map(|x| format!("Tensor: {x}"))
                            .unwrap_or(format!("Tensor Id: {tensor_id}"));
                        let mut window = egui::Window::new(RichText::from(name.clone()).size(14.))
                            .open(&mut local_open)
                            .resizable(false);
                        if let Some(default_pos) = default_pos {
                            window = window.default_pos(default_pos);
                        }

                        let resp = window.show(ctx, |ui| {
                            let mut resp = ui.label(format!(
                                "ONNX Name: {:}",
                                tensor_info.onnx_name.clone().unwrap_or("N/A".to_string())
                            ));
                            resp = resp.union(ui.label(format!(
                                "DType: {:}",
                                tensor_info
                                    .dtype
                                    .map(|x| x.to_string())
                                    .clone()
                                    .unwrap_or("N/A".to_string())
                            )));
                            resp = resp.union(ui.label(format!(
                                "Shape: {:}",
                                tensor_info
                                    .shape()
                                    .map(|x| format_shape(&x))
                                    .unwrap_or("N/A".to_string())
                            )));
                            let mut stored_tensor = None;
                            let mut subscribed_tensor = None;
                            match &tensor_info.tensor_type {
                                TensorType::Input(x) => {
                                    resp = resp.union(ui.label("Tensor Type: Model Input"));
                                    if let Some(x) = x {
                                        match x {
                                            StoredOrNotTensor::Stored(stored_tensor_id) => {
                                                stored_tensor = Some(*stored_tensor_id);
                                            }
                                            StoredOrNotTensor::NotStored(x) => {
                                                resp = resp.union(ui.label("Initial Value:"));
                                                resp = resp.union(tensor_view(
                                                    ui,
                                                    &x,
                                                    &mut inspect_window_tensor.value_view_state,
                                                ));
                                            }
                                        };
                                    } else {
                                        if let Some(x) = &super_graph_path {
                                            subscribed_tensor =
                                                Some(x.push_symbolic_tensor(tensor_id));
                                        }
                                    }
                                }
                                TensorType::Output => {
                                    resp = resp.union(ui.label("Tensor Type: Model Output"));
                                    if let Some(x) = &super_graph_path {
                                        subscribed_tensor = Some(x.push_symbolic_tensor(tensor_id));
                                    }
                                }
                                TensorType::Intermediate => {
                                    resp = resp.union(ui.label("Tensor Type: Intermediate"));
                                    if let Some(x) = &super_graph_path {
                                        subscribed_tensor = Some(x.push_symbolic_tensor(tensor_id));
                                    }
                                }
                                TensorType::Constant(x) => {
                                    resp = resp.union(ui.label("Tensor Type: Constant"));
                                    match x {
                                        StoredOrNotTensor::Stored(stored_tensor_id) => {
                                            stored_tensor = Some(*stored_tensor_id);
                                        }
                                        StoredOrNotTensor::NotStored(x) => {
                                            resp = resp.union(ui.label("Value:"));
                                            resp = resp.union(tensor_view(
                                                ui,
                                                &x,
                                                &mut inspect_window_tensor.value_view_state,
                                            ));
                                        }
                                    };
                                }
                            }
                            if let Some(stored_tensor_id) = stored_tensor {
                                if inspect_window_tensor.stored_value_requested.is_none() {
                                    inspect_window_tensor.stored_value_requested =
                                        Some(stored_tensor_id);
                                    let msg = WebsocketClientServerMessage::GetStoredTensor(
                                        *model_scope.unwrap(),
                                        stored_tensor_id,
                                    );
                                    server_request_manager.send(msg).unwrap();
                                }
                                if let Some(x) = &inspect_window_tensor.stored_value {
                                    match x {
                                        Ok(x) => {
                                            resp = resp.union(ui.label("Value:"));
                                            resp = resp.union(tensor_view(
                                                ui,
                                                &x,
                                                &mut inspect_window_tensor.value_view_state,
                                            ));
                                        }
                                        Err(err) => {
                                            ui.scope(|ui| {
                                                ui.visuals_mut().override_text_color =
                                                    Some(egui::Color32::RED);
                                                ui.style_mut().override_text_style =
                                                    Some(egui::TextStyle::Monospace);
                                                ui.label(err);
                                            });
                                        }
                                    }
                                } else {
                                    resp = resp.union(ui.label("Loading Tensor..."));
                                    resp = resp.union(ui.spinner());
                                }
                            }
                            if let Some(x) = subscribed_tensor {
                                if let Some(x) =
                                    self.inspect_window_tensor_subscription_returns.get(&x)
                                {
                                    resp = resp.union(ui.label("Last value:"));
                                    resp = resp.union(tensor_view(
                                        ui,
                                        &x,
                                        &mut inspect_window_tensor.subscribed_view_state,
                                    ));
                                } else {
                                    resp = resp.union(ui.label("No Value Received Yet"));
                                }
                                new_subscribed_tensors.insert(x);
                            }
                            if resp.clicked() {
                                self.explorer_selection =
                                    Some(GraphExplorerSelectable::SymbolicGraphTensorId(tensor_id));
                            }
                        });
                        if let Some(resp) = resp {
                            if resp.response.contains_pointer() {
                                self.next_explorer_hovered =
                                    Some(GraphExplorerSelectable::SymbolicGraphTensorId(tensor_id))
                            }
                        }
                    }
                    _ => {}
                }
                local_open
            });
            for new_inspect_window in new_inspect_windows {
                if !InspectWindow::check_if_already_exists(
                    &self.inspect_windows,
                    &new_inspect_window,
                ) && let Some(x) = InspectWindow::new(new_inspect_window)
                {
                    self.inspect_windows.push(x);
                }
            }
        }
        self.inspect_window_tensor_subscriptions = new_subscribed_tensors;
    }
}
