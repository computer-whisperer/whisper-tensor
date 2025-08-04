use crate::app::{InterfaceId, LoadedModels, ModelLoadState};
use crate::graph_layout::{
    GraphLayout, GraphLayoutIOOffsets, GraphLayoutLinkData, GraphLayoutLinkId, GraphLayoutLinkType,
    GraphLayoutNodeId, GraphLayoutNodeInitData, GraphLayoutNodeType,
};
use crate::websockets::ServerRequestManager;
use crate::widgets::toggle::toggle_ui;
use egui::epaint::CubicBezierShape;
use egui::{
    Color32, Label, Margin, Rect, Response, RichText, Sense, Shape, Stroke, StrokeKind, Ui,
    UiBuilder, Vec2, vec2,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::Duration;
use whisper_tensor::DynRank;
use whisper_tensor::backends::ndarray_backend::NDArrayNumericTensor;
use whisper_tensor::interfaces::AnyInterface;
use whisper_tensor::scalar_info::ScalarInfoTyped;
use whisper_tensor::symbolic_graph::ops::{AnyOperation, Operation};
use whisper_tensor::symbolic_graph::tensor_store::TensorStoreTensorId;
use whisper_tensor::symbolic_graph::{
    OperationId, StoredOrNotTensor, SymbolicGraph, TensorId, TensorType,
};
use whisper_tensor_server::{LoadedModelId, WebsocketClientServerMessage};

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub(crate) enum RootGraphSubjectIdentifier {
    Model(LoadedModelId),
    Interface(InterfaceId),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct GraphExplorerState {
    explorer_physics: bool,
    explorer_minimap: bool,
}

impl Default for GraphExplorerState {
    fn default() -> Self {
        Self {
            explorer_physics: false,
            explorer_minimap: false,
        }
    }
}

pub(crate) enum GraphSubject<'a> {
    Model(&'a SymbolicGraph),
    Interface(&'a AnyInterface),
}

pub(crate) struct GraphExplorerApp {
    pub(crate) selected_root_subject: Option<RootGraphSubjectIdentifier>,
    pub(crate) explorer_selection: Option<OpOrTensorId>,
    pub(crate) explorer_hovered: Option<OpOrTensorId>,
    pub(crate) next_explorer_hovered: Option<OpOrTensorId>,
    pub inspect_windows: Vec<InspectWindow>,
    graph_layouts: HashMap<RootGraphSubjectIdentifier, GraphLayout>,
    pub loaded_models: HashMap<LoadedModelId, SymbolicGraph>,
    model_view_scene_rects: HashMap<RootGraphSubjectIdentifier, Rect>,
}

fn render_node_contents(
    ui: &mut egui::Ui,
    node_type: &GraphLayoutNodeType,
    num_inputs: usize,
    num_outputs: usize,
    symbolic_graphs: &HashMap<LoadedModelId, SymbolicGraph>,
    is_selected: bool,
    is_hovered: bool,
) -> (Response, GraphLayoutIOOffsets) {
    // Decide corner radius
    let corner_radius = match node_type {
        GraphLayoutNodeType::SymbolicGraphOperation(_) | GraphLayoutNodeType::SuperGraphNode(_) => {
            3.0
        }
        GraphLayoutNodeType::SymbolicGraphTensor(_)
        | GraphLayoutNodeType::SuperGraphLink(_)
        | GraphLayoutNodeType::ConnectionByNameSrc(_)
        | GraphLayoutNodeType::ConnectionByNameDest(_) => 10.0,
    };

    /*let stroke_width = if is_selected {
        ui.visuals().widgets.active.fg_stroke.width
    } else {
        ui.visuals().widgets.inactive.fg_stroke.width
    };*/
    let stroke_width = ui.visuals().widgets.active.fg_stroke.width;

    let mut frame = egui::Frame::new()
        .inner_margin(5)
        .stroke(Stroke {
            width: stroke_width,
            color: Color32::TRANSPARENT,
        })
        .corner_radius(corner_radius)
        .begin(ui);
    {
        let ui = &mut frame.content_ui;
        match &node_type {
            GraphLayoutNodeType::SuperGraphNode((_interface_id, node_id)) => {
                ui.add(Label::new(format!("Node {node_id:?}")).selectable(false));
            }
            GraphLayoutNodeType::SuperGraphLink((_interface_id, link)) => {
                ui.add(Label::new(format!("Link {link:?}")).selectable(false));
            }
            GraphLayoutNodeType::SymbolicGraphOperation((model_id, op_id)) => {
                if let Some(symbolic_graph) = symbolic_graphs.get(model_id) {
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
            }
            GraphLayoutNodeType::SymbolicGraphTensor((model_id, tensor_id)) => {
                if let Some(symbolic_graph) = symbolic_graphs.get(model_id) {
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
            }
            GraphLayoutNodeType::ConnectionByNameSrc(name) => match &name.link_type {
                GraphLayoutLinkType::SymbolicGraphTensor((model_id, tensor_id)) => {
                    if let Some(symbolic_graph) = symbolic_graphs.get(model_id) {
                        let value = symbolic_graph.get_tensor_info(tensor_id.clone()).unwrap();
                        let name = value
                            .onnx_name
                            .clone()
                            .unwrap_or(format!("tensor {}", tensor_id));
                        ui.add(Label::new(format!("{} >", name)).selectable(false));
                    }
                }
                GraphLayoutLinkType::SuperGraphLink((_interface_id, link)) => {
                    ui.add(Label::new(format!("{:?} >", link)).selectable(false));
                }
            },
            GraphLayoutNodeType::ConnectionByNameDest(name) => match &name.link_type {
                GraphLayoutLinkType::SymbolicGraphTensor((model_id, tensor_id)) => {
                    if let Some(symbolic_graph) = symbolic_graphs.get(model_id) {
                        let value = symbolic_graph.get_tensor_info(tensor_id.clone()).unwrap();
                        let name = value
                            .onnx_name
                            .clone()
                            .unwrap_or(format!("tensor {}", tensor_id));
                        ui.add(Label::new(format!("> {}", name)).selectable(false));
                    }
                }
                GraphLayoutLinkType::SuperGraphLink((_interface_id, link)) => {
                    ui.add(Label::new(format!("> {:?}", link)).selectable(false));
                }
            },
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
        (
            ui.visuals().widgets.active.bg_fill,
            egui::Stroke {
                width: stroke_width,
                color: Color32::from_rgb(64, 64, 255),
            },
        )
    } else if response.hovered() || is_hovered {
        (
            ui.visuals().widgets.hovered.bg_fill,
            ui.visuals().widgets.hovered.fg_stroke,
        )
    } else {
        (
            ui.visuals().widgets.inactive.bg_fill,
            ui.visuals().widgets.inactive.fg_stroke,
        )
    };
    frame.frame.fill = fill;
    frame.frame.stroke = stroke;
    frame.paint(ui);

    // Get positions for io ports
    let mut inputs = vec![];
    for i in 0..num_inputs {
        inputs.push(egui::Vec2::new(
            -ui.min_rect().width() / 2.0,
            (((i as f32 + 1.0) / (num_inputs as f32 + 1.0)) - 0.5) * ui.min_rect().height(),
        ));
    }
    let mut outputs = vec![];
    for i in 0..num_outputs {
        outputs.push(egui::Vec2::new(
            ui.min_rect().width() / 2.0,
            (((i as f32 + 1.0) / (num_outputs as f32 + 1.0)) - 0.5) * ui.min_rect().height(),
        ));
    }
    (response, GraphLayoutIOOffsets { inputs, outputs })
}

fn format_shape(val: &Vec<ScalarInfoTyped<u64>>) -> String {
    let joined = val
        .iter()
        .map(|x| match x {
            ScalarInfoTyped::Numeric(x) => x.to_string(),
            ScalarInfoTyped::Symbolic(_x) => "?".to_string(),
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("({joined:})")
}

impl GraphExplorerApp {
    pub(crate) fn new() -> Self {
        Self {
            selected_root_subject: None,
            explorer_selection: None,
            explorer_hovered: None,
            next_explorer_hovered: None,
            inspect_windows: Vec::new(),
            graph_layouts: HashMap::new(),
            loaded_models: HashMap::new(),
            model_view_scene_rects: HashMap::new(),
        }
    }

    pub(crate) fn update(
        &mut self,
        state: &mut GraphExplorerState,
        loaded_models: &mut LoadedModels,
        server_request_manager: &mut ServerRequestManager,
        ui: &mut Ui,
    ) {
        self.explorer_hovered = self.next_explorer_hovered.clone();
        self.next_explorer_hovered = None;

        ui.horizontal(|ui| {
            if loaded_models.current_models.is_empty() {
                ui.label("No Models Loaded");
            }
            for model in &loaded_models.current_models {
                if ui
                    .selectable_value(
                        &mut self.selected_root_subject,
                        Some(RootGraphSubjectIdentifier::Model(model.model_id)),
                        format!("({}) {}", model.model_id, model.model_name.clone()),
                    )
                    .clicked()
                {
                    self.explorer_selection = None;
                    self.explorer_hovered = None;
                    self.inspect_windows.clear();
                };
            }
            for (interface_id, interface) in &loaded_models.current_interfaces {
                if ui
                    .selectable_value(
                        &mut self.selected_root_subject,
                        Some(RootGraphSubjectIdentifier::Interface(interface_id.clone())),
                        interface.interface_name.clone(),
                    )
                    .clicked()
                {
                    self.explorer_selection = None;
                    self.explorer_hovered = None;
                    self.inspect_windows.clear();
                };
            }
            if ui.button("Load New Model").clicked() {
                loaded_models.model_load_state = Some(ModelLoadState::DialogOpen(None));
            };
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                toggle_ui(ui, &mut state.explorer_minimap);
                ui.label("Minimap:");
                toggle_ui(ui, &mut state.explorer_physics);
                ui.label("Physics:");
            })
        });

        let graph_subject = if let Some(graph_subject_id) = &self.selected_root_subject {
            match graph_subject_id {
                RootGraphSubjectIdentifier::Model(model_id) => {
                    self.loaded_models.get(model_id).map(GraphSubject::Model)
                }
                RootGraphSubjectIdentifier::Interface(interface_id) => loaded_models
                    .current_interfaces
                    .get(interface_id)
                    .map(|x| GraphSubject::Interface(&x.interface)),
            }
        } else {
            None
        };

        if let Some(graph_subject) = graph_subject {
            let graph_subject_id = self.selected_root_subject.as_ref().unwrap();
            if let Some(graph_layout) = self.graph_layouts.get(graph_subject_id) {
                if state.explorer_minimap {
                    let node_bounding_rect = graph_layout.get_bounding_rect();

                    // Get frame min and max
                    let minimap_frame = egui::Frame::default().stroke(ui.visuals().window_stroke);
                    minimap_frame.show(ui, |ui| {
                        let shape_request = vec2(ui.available_size_before_wrap().x, 100.0);
                        let (outer_rect, outer_response) =
                            ui.allocate_exact_size(shape_request, Sense::drag());
                        let transform = outer_rect.size() / node_bounding_rect.size();
                        if outer_response.dragged() {
                            let outer_pos = outer_response.interact_pointer_pos().unwrap();
                            let inner_pos =
                                node_bounding_rect.min + ((outer_pos - outer_rect.min) / transform);
                            if let Some(selected_area) =
                                self.model_view_scene_rects.get_mut(&graph_subject_id)
                            {
                                *selected_area =
                                    Rect::from_center_size(inner_pos, selected_area.size());
                            }
                        }
                        for (_node_id, node) in graph_layout.get_nodes() {
                            let pos = (node.position - node_bounding_rect.min) * transform;
                            let (is_selected, is_hovered) = match &node.node_type {
                                GraphLayoutNodeType::SymbolicGraphOperation((_model_id, op_id)) => {
                                    (
                                        if let Some(OpOrTensorId::Op(selected_op_id)) =
                                            &self.explorer_selection
                                        {
                                            *selected_op_id == *op_id
                                        } else {
                                            false
                                        },
                                        if let Some(OpOrTensorId::Op(hovered_op_id)) =
                                            &self.explorer_hovered
                                        {
                                            *hovered_op_id == *op_id
                                        } else {
                                            false
                                        },
                                    )
                                }
                                GraphLayoutNodeType::SymbolicGraphTensor((
                                    _model_id,
                                    tensor_id,
                                )) => (
                                    if let Some(OpOrTensorId::Tensor(selected_tensor_id)) =
                                        &self.explorer_selection
                                    {
                                        *selected_tensor_id == *tensor_id
                                    } else {
                                        false
                                    },
                                    if let Some(OpOrTensorId::Tensor(hovered_tensor_id)) =
                                        &self.explorer_hovered
                                    {
                                        *hovered_tensor_id == *tensor_id
                                    } else {
                                        false
                                    },
                                ),
                                GraphLayoutNodeType::SuperGraphNode(_) => (false, false),
                                GraphLayoutNodeType::SuperGraphLink(_) => (false, false),
                                GraphLayoutNodeType::ConnectionByNameSrc(link_data)
                                | GraphLayoutNodeType::ConnectionByNameDest(link_data) => {
                                    match &link_data.link_type {
                                        GraphLayoutLinkType::SymbolicGraphTensor((
                                            _model_id,
                                            tensor_id,
                                        )) => (
                                            if let Some(OpOrTensorId::Tensor(selected_tensor_id)) =
                                                &self.explorer_selection
                                            {
                                                *selected_tensor_id == *tensor_id
                                            } else {
                                                false
                                            },
                                            if let Some(OpOrTensorId::Tensor(hovered_tensor_id)) =
                                                &self.explorer_hovered
                                            {
                                                *hovered_tensor_id == *tensor_id
                                            } else {
                                                false
                                            },
                                        ),
                                        GraphLayoutLinkType::SuperGraphLink((_a, _b)) => {
                                            (false, false)
                                        }
                                    }
                                }
                            };

                            let color = if is_selected {
                                egui::Color32::from_rgba_unmultiplied(64, 64, 255, 128)
                            } else if is_hovered {
                                egui::Color32::from_rgba_unmultiplied(80, 80, 80, 128)
                            } else {
                                egui::Color32::from_rgba_unmultiplied(32, 32, 32, 128)
                            };
                            let radius = if is_selected { 3.0 } else { 1.0 };

                            ui.painter().add(egui::Shape::circle_filled(
                                outer_rect.min + pos,
                                radius,
                                color,
                            ));
                        }
                        if let Some(selected_area) =
                            self.model_view_scene_rects.get(&graph_subject_id)
                        {
                            let transformed_area = Rect::from_min_max(
                                (outer_rect.min
                                    + ((selected_area.min - node_bounding_rect.min) * transform))
                                    .max(outer_rect.min),
                                (outer_rect.min
                                    + ((selected_area.max - node_bounding_rect.min) * transform))
                                    .min(outer_rect.max),
                            );
                            ui.painter().add(egui::Shape::rect_stroke(
                                transformed_area,
                                2.0,
                                (1.0, egui::Color32::from_rgba_unmultiplied(64, 64, 255, 128)),
                                StrokeKind::Inside,
                            ));
                        }
                    });
                }
            }

            if !self.graph_layouts.contains_key(&graph_subject_id) {
                match graph_subject {
                    GraphSubject::Model(graph) => {
                        let model_id =
                            if let RootGraphSubjectIdentifier::Model(graph) = graph_subject_id {
                                *graph
                            } else {
                                panic!();
                            };

                        let mut next_link_id = 0;

                        // Map tensors to link IDs
                        let mut tensor_link_ids = HashMap::new();
                        let mut link_data = HashMap::new();
                        for (tensor_id, _) in graph.get_tensors() {
                            let new_link_id = GraphLayoutLinkId(next_link_id);
                            next_link_id += 1;
                            tensor_link_ids.insert(*tensor_id, new_link_id);
                            link_data.insert(
                                new_link_id,
                                GraphLayoutLinkData {
                                    link_type: GraphLayoutLinkType::SymbolicGraphTensor((
                                        model_id, *tensor_id,
                                    )),
                                },
                            );
                        }

                        // Build node init data for ops and I/O tensors
                        let mut sourced_links = HashSet::new();
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
                                sourced_links.insert(tensor_link_ids[&tensor_id]);
                                outputs.push(tensor_link_ids[&tensor_id]);
                            }
                            node_init_data.insert(
                                new_node_id,
                                GraphLayoutNodeInitData {
                                    node_type: GraphLayoutNodeType::SymbolicGraphOperation((
                                        model_id, *op_id,
                                    )),
                                    inputs,
                                    outputs,
                                },
                            );
                        }

                        let mut io_tensor_node_ids = HashMap::new();
                        for (tensor_id, tensor_data) in graph.get_tensors() {
                            match tensor_data.tensor_type {
                                TensorType::Input(_) => {
                                    let node_id = GraphLayoutNodeId(next_node_id);
                                    io_tensor_node_ids.insert(*tensor_id, node_id);
                                    next_node_id += 1;
                                    sourced_links.insert(tensor_link_ids[&tensor_id]);
                                    node_init_data.insert(
                                        node_id,
                                        GraphLayoutNodeInitData {
                                            node_type: GraphLayoutNodeType::SymbolicGraphTensor((
                                                model_id, *tensor_id,
                                            )),
                                            inputs: vec![],
                                            outputs: vec![tensor_link_ids[tensor_id]],
                                        },
                                    );
                                }
                                TensorType::Output => {
                                    let node_id = GraphLayoutNodeId(next_node_id);
                                    io_tensor_node_ids.insert(*tensor_id, node_id);
                                    next_node_id += 1;

                                    node_init_data.insert(
                                        node_id,
                                        GraphLayoutNodeInitData {
                                            node_type: GraphLayoutNodeType::SymbolicGraphTensor((
                                                model_id, *tensor_id,
                                            )),
                                            inputs: vec![tensor_link_ids[tensor_id]],
                                            outputs: vec![],
                                        },
                                    );
                                }
                                TensorType::Intermediate => {
                                    if !sourced_links.contains(&tensor_link_ids[tensor_id]) {
                                        let node_id = GraphLayoutNodeId(next_node_id);
                                        io_tensor_node_ids.insert(*tensor_id, node_id);
                                        next_node_id += 1;
                                        sourced_links.insert(tensor_link_ids[&tensor_id]);
                                        node_init_data.insert(
                                            node_id,
                                            GraphLayoutNodeInitData {
                                                node_type: GraphLayoutNodeType::SymbolicGraphTensor(
                                                    (model_id, *tensor_id),
                                                ),
                                                inputs: vec![],
                                                outputs: vec![tensor_link_ids[tensor_id]],
                                            },
                                        );
                                    }
                                    continue;
                                }
                                TensorType::Constant(_) => {
                                    let node_id = GraphLayoutNodeId(next_node_id);
                                    io_tensor_node_ids.insert(*tensor_id, node_id);
                                    next_node_id += 1;
                                    sourced_links.insert(tensor_link_ids[&tensor_id]);
                                    node_init_data.insert(
                                        node_id,
                                        GraphLayoutNodeInitData {
                                            node_type: GraphLayoutNodeType::SymbolicGraphTensor((
                                                model_id, *tensor_id,
                                            )),
                                            inputs: vec![],
                                            outputs: vec![tensor_link_ids[tensor_id]],
                                        },
                                    );
                                }
                            }
                        }
                        let initial_layout = GraphLayout::new(
                            node_init_data,
                            link_data,
                            ui,
                            |ui, node_init_data| {
                                render_node_contents(
                                    ui,
                                    &node_init_data.node_type,
                                    node_init_data.inputs.len(),
                                    node_init_data.outputs.len(),
                                    &self.loaded_models,
                                    false,
                                    false,
                                )
                                .1
                            },
                        );
                        self.graph_layouts
                            .insert(graph_subject_id.clone(), initial_layout);
                    }
                    GraphSubject::Interface(interface) => {
                        let interface_id =
                            if let RootGraphSubjectIdentifier::Interface(name) = graph_subject_id {
                                name.clone()
                            } else {
                                panic!();
                            };

                        let super_graph = interface.get_super_graph();

                        // Map links
                        let mut link_ids = HashMap::new();
                        let mut next_link_id = 0;
                        let mut link_data = HashMap::new();
                        for super_graph_link in super_graph.get_all_links() {
                            let new_link_id = GraphLayoutLinkId(next_link_id);
                            next_link_id += 1;
                            link_ids.insert(super_graph_link.clone(), new_link_id);
                            link_data.insert(
                                new_link_id,
                                GraphLayoutLinkData {
                                    link_type: GraphLayoutLinkType::SuperGraphLink((
                                        interface_id,
                                        super_graph_link,
                                    )),
                                },
                            );
                        }

                        // Map nodes
                        let mut sourced_links = HashSet::new();
                        let mut next_node_id = 0;
                        let mut node_init_data = HashMap::new();
                        for (node_id, node) in super_graph.inner.nodes.iter() {
                            let new_node_id = GraphLayoutNodeId(next_node_id);
                            next_node_id += 1;

                            let mut inputs = vec![];
                            for tensor_id in node.get_inputs() {
                                inputs.push(link_ids[&tensor_id]);
                            }
                            let mut outputs = vec![];
                            for tensor_id in node.get_outputs() {
                                sourced_links.insert(link_ids[&tensor_id]);
                                outputs.push(link_ids[&tensor_id]);
                            }
                            node_init_data.insert(
                                new_node_id,
                                GraphLayoutNodeInitData {
                                    node_type: GraphLayoutNodeType::SuperGraphNode((
                                        interface_id,
                                        node_id.clone(),
                                    )),
                                    inputs,
                                    outputs,
                                },
                            );
                        }

                        // I/O links
                        for input_link in &super_graph.inner.input_links {
                            let new_node_id = GraphLayoutNodeId(next_node_id);
                            next_node_id += 1;

                            let mut inputs = vec![];
                            let mut outputs = vec![link_ids[input_link]];
                            sourced_links.insert(link_ids[input_link]);

                            node_init_data.insert(
                                new_node_id,
                                GraphLayoutNodeInitData {
                                    node_type: GraphLayoutNodeType::SuperGraphLink((
                                        interface_id,
                                        input_link.clone(),
                                    )),
                                    inputs,
                                    outputs,
                                },
                            );
                        }
                        for output_link in &super_graph.inner.output_links {
                            let new_node_id = GraphLayoutNodeId(next_node_id);
                            next_node_id += 1;

                            let mut inputs = vec![link_ids[output_link]];
                            let mut outputs = vec![];
                            node_init_data.insert(
                                new_node_id,
                                GraphLayoutNodeInitData {
                                    node_type: GraphLayoutNodeType::SuperGraphLink((
                                        interface_id,
                                        output_link.clone(),
                                    )),
                                    inputs,
                                    outputs,
                                },
                            );
                        }

                        let initial_layout = GraphLayout::new(
                            node_init_data,
                            link_data,
                            ui,
                            |ui, node_init_data| {
                                render_node_contents(
                                    ui,
                                    &node_init_data.node_type,
                                    node_init_data.inputs.len(),
                                    node_init_data.outputs.len(),
                                    &self.loaded_models,
                                    false,
                                    false,
                                )
                                .1
                            },
                        );
                        self.graph_layouts
                            .insert(graph_subject_id.clone(), initial_layout);
                    }
                }
            }

            let graph_layout = self.graph_layouts.get_mut(&graph_subject_id).unwrap();

            // Update positions
            if state.explorer_physics {
                if graph_layout.update_layout(5000) {
                    ui.ctx().request_repaint_after(Duration::from_millis(20));
                }
            }

            let frame = egui::Frame::default().stroke(ui.visuals().window_stroke);
            frame.show(ui, |ui| {
                let frame_shape = ui.max_rect().size();
                let mut scene_rect = self
                    .model_view_scene_rects
                    .get(&graph_subject_id)
                    .map(|x| x.clone())
                    .unwrap_or(Rect::from_center_size(
                        egui::pos2(-300.0 + frame_shape.x, 0.0),
                        frame_shape * 2.0,
                    ));
                let cull_rect = scene_rect.expand(300.0);
                let scene = egui::Scene::new();
                scene.show(ui, &mut scene_rect, |ui| {
                    // Find all ops actually in scene
                    let mut nodes_to_render_vec = graph_layout
                        .find_nodes_within(&cull_rect.center(), cull_rect.size().length() / 2.0)
                        .clone();
                    let mut nodes_to_render = HashSet::<GraphLayoutNodeId>::from_iter(
                        nodes_to_render_vec.iter().map(|x| *x),
                    );

                    let mut edges_to_render = vec![];

                    // Render both sides of all visible edges
                    for ((src_id, src_id_i), (dst_id, dst_id_i), link_id) in
                        graph_layout.get_edges()
                    {
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
                            edges_to_render.push((
                                (*src_id, *src_id_i),
                                (*dst_id, *dst_id_i),
                                *link_id,
                                shape_idx,
                            ));
                        }
                    }

                    let mut node_io_connections = HashMap::new();
                    let mut node_bounding_boxes = HashMap::new();

                    for node_id in nodes_to_render_vec {
                        let current_node_data = graph_layout.get_nodes_mut();
                        let pos = current_node_data[&node_id].position;
                        let cell_shape = current_node_data[&node_id].shape;
                        let op_rect = Rect::from_min_max(
                            pos.clone() - cell_shape / 2.0,
                            pos.clone() + cell_shape,
                        );
                        let ui_builder = UiBuilder::new().max_rect(op_rect);
                        let mut ui_child = ui.new_child(ui_builder);

                        let (is_selected, is_hovered) = match &current_node_data[&node_id].node_type
                        {
                            GraphLayoutNodeType::SymbolicGraphOperation((_model_id, op_id)) => (
                                if let Some(OpOrTensorId::Op(selected_op_id)) =
                                    &self.explorer_selection
                                {
                                    *selected_op_id == *op_id
                                } else {
                                    false
                                },
                                if let Some(OpOrTensorId::Op(hovered_op_id)) =
                                    &self.explorer_hovered
                                {
                                    *hovered_op_id == *op_id
                                } else {
                                    false
                                },
                            ),
                            GraphLayoutNodeType::SymbolicGraphTensor((_model_id, tensor_id)) => (
                                if let Some(OpOrTensorId::Tensor(selected_tensor_id)) =
                                    &self.explorer_selection
                                {
                                    *selected_tensor_id == *tensor_id
                                } else {
                                    false
                                },
                                if let Some(OpOrTensorId::Tensor(hovered_tensor_id)) =
                                    &self.explorer_hovered
                                {
                                    *hovered_tensor_id == *tensor_id
                                } else {
                                    false
                                },
                            ),
                            GraphLayoutNodeType::SuperGraphNode(_) => (false, false),
                            GraphLayoutNodeType::SuperGraphLink(_) => (false, false),
                            GraphLayoutNodeType::ConnectionByNameSrc(link_data)
                            | GraphLayoutNodeType::ConnectionByNameDest(link_data) => {
                                match &link_data.link_type {
                                    GraphLayoutLinkType::SymbolicGraphTensor((
                                        _model_id,
                                        tensor_id,
                                    )) => (
                                        if let Some(OpOrTensorId::Tensor(selected_tensor_id)) =
                                            &self.explorer_selection
                                        {
                                            *selected_tensor_id == *tensor_id
                                        } else {
                                            false
                                        },
                                        if let Some(OpOrTensorId::Tensor(hovered_tensor_id)) =
                                            &self.explorer_hovered
                                        {
                                            *hovered_tensor_id == *tensor_id
                                        } else {
                                            false
                                        },
                                    ),
                                    GraphLayoutLinkType::SuperGraphLink((_a, _b)) => (false, false),
                                }
                            }
                        };

                        let (resp, io_connections) = render_node_contents(
                            &mut ui_child,
                            &current_node_data[&node_id].node_type,
                            current_node_data[&node_id].inputs.len(),
                            current_node_data[&node_id].outputs.len(),
                            &self.loaded_models,
                            is_selected,
                            is_hovered,
                        );
                        node_io_connections.insert(node_id, io_connections);
                        node_bounding_boxes.insert(node_id, ui_child.min_rect());

                        let this_op_or_tensor_id = match &current_node_data[&node_id].node_type {
                            GraphLayoutNodeType::SymbolicGraphOperation((_model_id, op_id)) => {
                                Some(OpOrTensorId::Op(*op_id))
                            }
                            GraphLayoutNodeType::SymbolicGraphTensor((_model_id, tensor_id)) => {
                                Some(OpOrTensorId::Tensor(*tensor_id))
                            }
                            GraphLayoutNodeType::SuperGraphNode(_) => None,
                            GraphLayoutNodeType::SuperGraphLink(_) => None,
                            GraphLayoutNodeType::ConnectionByNameSrc(tensor_data)
                            | GraphLayoutNodeType::ConnectionByNameDest(tensor_data) => {
                                match &tensor_data.link_type {
                                    GraphLayoutLinkType::SymbolicGraphTensor((
                                        _model_id,
                                        tensor_id,
                                    )) => Some(OpOrTensorId::Tensor(*tensor_id)),
                                    GraphLayoutLinkType::SuperGraphLink((_a, _b)) => None,
                                }
                            }
                        };

                        if resp.hovered() {
                            self.next_explorer_hovered = this_op_or_tensor_id.clone();
                        }
                        if resp.clicked() {
                            self.explorer_selection = this_op_or_tensor_id;
                        }
                        if resp.dragged() {
                            current_node_data.get_mut(&node_id).unwrap().position +=
                                resp.drag_delta();
                            current_node_data.get_mut(&node_id).unwrap().velocity = Vec2::ZERO;
                        }
                        if resp.double_clicked() {
                            match &current_node_data[&node_id].node_type {
                                GraphLayoutNodeType::SymbolicGraphOperation((_model_id, op_id)) => {
                                    let new_inspect = OpOrTensorId::Op(*op_id);
                                    if !InspectWindow::check_if_already_exists(
                                        &self.inspect_windows,
                                        &new_inspect,
                                    ) {
                                        self.inspect_windows.push(InspectWindow::new(new_inspect));
                                    }
                                }
                                GraphLayoutNodeType::SymbolicGraphTensor((
                                    _model_id,
                                    tensor_id,
                                )) => {
                                    let new_inspect = OpOrTensorId::Tensor(*tensor_id);
                                    if !InspectWindow::check_if_already_exists(
                                        &self.inspect_windows,
                                        &new_inspect,
                                    ) {
                                        self.inspect_windows.push(InspectWindow::new(new_inspect));
                                    }
                                }
                                GraphLayoutNodeType::SuperGraphNode(_) => {
                                    // TODO: Inspect this
                                }
                                GraphLayoutNodeType::SuperGraphLink(_) => {
                                    // TODO: Inspect this
                                }
                                GraphLayoutNodeType::ConnectionByNameSrc(link_data)
                                | GraphLayoutNodeType::ConnectionByNameDest(link_data) => {
                                    match &link_data.link_type {
                                        GraphLayoutLinkType::SymbolicGraphTensor((
                                            _model_id,
                                            tensor_id,
                                        )) => {
                                            let new_inspect = OpOrTensorId::Tensor(*tensor_id);
                                            if !InspectWindow::check_if_already_exists(
                                                &self.inspect_windows,
                                                &new_inspect,
                                            ) {
                                                self.inspect_windows
                                                    .push(InspectWindow::new(new_inspect));
                                            }
                                        }
                                        GraphLayoutLinkType::SuperGraphLink((_a, _b)) => {
                                            // TODO: inspect super graph link
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Draw lines
                    let link_data = graph_layout.get_link_data();
                    for ((src_id, src_id_i), (dst_id, dst_id_i), link_id, paint_idx) in
                        edges_to_render
                    {
                        let source_connection = node_bounding_boxes[&src_id].center()
                            + node_io_connections[&src_id].outputs[src_id_i];
                        let dest_connection = node_bounding_boxes[&dst_id].center()
                            + node_io_connections[&dst_id].inputs[dst_id_i];
                        let points = [
                            source_connection,
                            egui::pos2(source_connection.x + 40.0, source_connection.y),
                            egui::pos2(dest_connection.x - 40.0, dest_connection.y),
                            dest_connection,
                        ];
                        let (is_selected, is_hovered) =
                            if let Some(link_data) = link_data.get(&link_id) {
                                match link_data.link_type {
                                    GraphLayoutLinkType::SymbolicGraphTensor((
                                        _model_id,
                                        tensor_id,
                                    )) => (
                                        if let Some(OpOrTensorId::Tensor(selected_tensor_id)) =
                                            self.explorer_selection
                                        {
                                            tensor_id == selected_tensor_id
                                        } else {
                                            false
                                        },
                                        if let Some(OpOrTensorId::Tensor(selected_tensor_id)) =
                                            self.explorer_hovered
                                        {
                                            tensor_id == selected_tensor_id
                                        } else {
                                            false
                                        },
                                    ),
                                    GraphLayoutLinkType::SuperGraphLink(_) => (false, false),
                                }
                            } else {
                                (false, false)
                            };

                        let stroke = if is_selected {
                            Stroke {
                                width: ui.visuals().widgets.active.fg_stroke.width,
                                color: egui::Color32::from_rgb(64, 64, 255),
                            }
                        } else if is_hovered {
                            ui.visuals().widgets.hovered.fg_stroke
                        } else {
                            ui.visuals().widgets.noninteractive.fg_stroke
                        };

                        let shape = CubicBezierShape::from_points_stroke(
                            points,
                            false,
                            Color32::TRANSPARENT,
                            stroke,
                        );
                        ui.painter().set(paint_idx, shape);
                    }
                });
                self.model_view_scene_rects
                    .insert(graph_subject_id.clone(), scene_rect);
            });
        } else if let Some(graph_subject_id) = &self.selected_root_subject {
            if let RootGraphSubjectIdentifier::Model(model_id) = graph_subject_id {
                ui.label("Loading Model Graph...");
                ui.spinner();
                if loaded_models
                    .currently_requesting_model
                    .map(|id| id != *model_id)
                    .unwrap_or(true)
                {
                    server_request_manager
                        .send(WebsocketClientServerMessage::GetModelGraph(*model_id))
                        .unwrap();
                    loaded_models.currently_requesting_model = Some(*model_id);
                }
            }
        } else {
            ui.label("No graph selected");
        }
    }

    pub(crate) fn update_inspect_windows(
        &mut self,
        _state: &mut GraphExplorerState,
        ctx: &egui::Context,
        server_request_manager: &mut ServerRequestManager,
    ) {
        let mut new_inspect_windows = vec![];
        self.inspect_windows.retain_mut(|inspect_window| {
            let mut local_open = true;
            let default_pos = ctx.input(|x| x.pointer.latest_pos());
            if let Some(RootGraphSubjectIdentifier::Model(model_id)) = &self.selected_root_subject {
                if let Some(model_graph) = self.loaded_models.get(model_id) {
                    match inspect_window {
                        InspectWindow::Operation(op_id) => {
                            let mut is_hovering_tensor = false;
                            let op_info = &model_graph.get_operations()[op_id];
                            let name = op_info
                                .name
                                .clone()
                                .map(|x| format!("Op: {x}"))
                                .unwrap_or(format!("Op Id: {op_id:?}"));
                            let mut window =
                                egui::Window::new(RichText::from(name.clone()).size(14.))
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
                                    resp.union(ui.label(format!(
                                        "Op Type: {:}",
                                        op_info.op.get_op_type_name()
                                    )));
                                resp = resp.union(ui.label("Inputs:"));
                                fn format_tensor_row(
                                    ui: &mut egui::Ui,
                                    i: usize,
                                    tensor_id: TensorId,
                                    model_graph: &SymbolicGraph,
                                    new_inspect_windows: &mut Vec<OpOrTensorId>,
                                ) -> Response {
                                    let tensor_info =
                                        model_graph.get_tensor_info(tensor_id).unwrap();
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
                                        new_inspect_windows.push(OpOrTensorId::Tensor(tensor_id));
                                    }
                                    ui.end_row();
                                    response
                                }
                                egui::Grid::new(egui::Id::new(("grid_inputs", name.clone())))
                                    .striped(true)
                                    .show(ui, |ui| {
                                        for (i, tensor_id) in
                                            op_info.op.get_inputs().iter().enumerate()
                                        {
                                            let resp = format_tensor_row(
                                                ui,
                                                i,
                                                *tensor_id,
                                                &model_graph,
                                                &mut new_inspect_windows,
                                            );
                                            if resp.hovered() {
                                                self.next_explorer_hovered =
                                                    Some(OpOrTensorId::Tensor(*tensor_id));
                                                is_hovering_tensor = true;
                                            }
                                            if resp.clicked() {
                                                self.explorer_selection =
                                                    Some(OpOrTensorId::Tensor(*tensor_id));
                                            }
                                        }
                                    });
                                resp = resp.union(ui.label("Outputs:"));
                                egui::Grid::new(egui::Id::new(("grid_outputs", name)))
                                    .striped(true)
                                    .show(ui, |ui| {
                                        for (i, tensor_id) in
                                            op_info.op.get_outputs().iter().enumerate()
                                        {
                                            let resp = format_tensor_row(
                                                ui,
                                                i,
                                                *tensor_id,
                                                &model_graph,
                                                &mut new_inspect_windows,
                                            );
                                            if resp.hovered() {
                                                self.next_explorer_hovered =
                                                    Some(OpOrTensorId::Tensor(*tensor_id));
                                                is_hovering_tensor = true;
                                            }
                                            if resp.clicked() {
                                                self.explorer_selection =
                                                    Some(OpOrTensorId::Tensor(*tensor_id));
                                            }
                                        }
                                    });
                                match &op_info.op {
                                    AnyOperation::Constant(x) => {
                                        resp = resp.union(
                                            ui.label(format!("Value: {}", x.value.to_string())),
                                        );
                                    }
                                    _ => {}
                                }

                                if resp.clicked() {
                                    self.explorer_selection = Some(OpOrTensorId::Op(*op_id));
                                }
                            });
                            if let Some(resp) = resp {
                                if resp.response.contains_pointer() && !is_hovering_tensor {
                                    self.next_explorer_hovered = Some(OpOrTensorId::Op(*op_id))
                                }
                            }
                        }
                        InspectWindow::Tensor(inspect_window_tensor) => {
                            let tensor_id = inspect_window_tensor.tensor_id;
                            let tensor_info = model_graph.get_tensor_info(tensor_id).unwrap();
                            let name = tensor_info
                                .onnx_name
                                .clone()
                                .map(|x| format!("Tensor: {x}"))
                                .unwrap_or(format!("Tensor Id: {tensor_id}"));
                            let mut window =
                                egui::Window::new(RichText::from(name.clone()).size(14.))
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
                                        inspect_window_tensor.stored_value_requested =
                                            Some(stored_tensor_id);
                                        let msg = WebsocketClientServerMessage::GetStoredTensor(
                                            *model_id,
                                            stored_tensor_id,
                                        );
                                        server_request_manager.send(msg).unwrap();
                                    }
                                    if let Some(x) = &inspect_window_tensor.stored_value {
                                        match x {
                                            Ok(x) => {
                                                resp = resp.union(ui.label(format!("Value: {x}")));
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
                                if resp.clicked() {
                                    self.explorer_selection = Some(OpOrTensorId::Tensor(tensor_id));
                                }
                            });
                            if let Some(resp) = resp {
                                if resp.response.contains_pointer() {
                                    self.next_explorer_hovered =
                                        Some(OpOrTensorId::Tensor(tensor_id))
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
                self.inspect_windows
                    .push(InspectWindow::new(new_inspect_window));
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum OpOrTensorId {
    Op(OperationId),
    Tensor(TensorId),
}

#[derive(Clone, Debug)]
pub(crate) struct InspectWindowTensor {
    pub(crate) tensor_id: TensorId,
    pub(crate) stored_value_requested: Option<TensorStoreTensorId>,
    pub(crate) stored_value: Option<Result<NDArrayNumericTensor<DynRank>, String>>,
}

#[derive(Clone, Debug)]
pub(crate) enum InspectWindow {
    Operation(OperationId),
    Tensor(InspectWindowTensor),
}

impl InspectWindow {
    fn to_op_or_tensor_id(&self) -> OpOrTensorId {
        match self {
            InspectWindow::Operation(op_id) => OpOrTensorId::Op(*op_id),
            InspectWindow::Tensor(x) => OpOrTensorId::Tensor(x.tensor_id),
        }
    }

    pub(crate) fn check_if_already_exists(
        windows: &[Self],
        op_or_tensor_id: &OpOrTensorId,
    ) -> bool {
        for window in windows {
            if window.to_op_or_tensor_id() == *op_or_tensor_id {
                return true;
            }
        }
        false
    }

    pub(crate) fn new(op_or_tensor_id: OpOrTensorId) -> Self {
        match op_or_tensor_id {
            OpOrTensorId::Op(op_id) => Self::Operation(op_id),
            OpOrTensorId::Tensor(tensor_id) => Self::Tensor(InspectWindowTensor {
                tensor_id,
                stored_value_requested: None,
                stored_value: None,
            }),
        }
    }
}
