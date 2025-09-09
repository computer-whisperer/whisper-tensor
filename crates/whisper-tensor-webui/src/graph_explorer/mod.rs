mod graph_layout;
pub mod inspect_windows;
mod tensor_swatch;

use crate::app::{InterfaceId, LoadedModels, LoadedTokenizers, ModelLoadState};
use crate::graph_explorer::inspect_windows::InspectWindow;
use crate::websockets::ServerRequestManager;
use crate::widgets::toggle::toggle_ui;
use crate::widgets::tokenized_rich_text::TokenizedRichText;
use egui::epaint::CubicBezierShape;
use egui::{
    Color32, ColorImage, Context, Label, Margin, Mesh, Pos2, Rect, Response, Sense, Shape, Stroke,
    StrokeKind, TextureHandle, Ui, UiBuilder, Vec2, Widget, vec2,
};
use graph_layout::{
    GraphLayout, GraphLayoutError, GraphLayoutIOOffsets, GraphLayoutLinkData, GraphLayoutLinkId,
    GraphLayoutLinkType, GraphLayoutNodeId, GraphLayoutNodeInitData, GraphLayoutNodeType,
};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tensor_swatch::build_tensor_swatch;
use web_time::{Duration, Instant};
use whisper_tensor::DynRank;
use whisper_tensor::backends::ndarray_backend::NDArrayNumericTensor;
use whisper_tensor::graph::{InnerGraph, Node};
use whisper_tensor::interfaces::AnyInterface;
use whisper_tensor::milli_graph::ops::MilliOp;
use whisper_tensor::milli_graph::{MilliOpGraph, MilliOpGraphNodeId, MilliOpGraphTensorId};
use whisper_tensor::scalar_info::ScalarInfoTyped;
use whisper_tensor::super_graph::nodes::SuperGraphAnyNode;
use whisper_tensor::super_graph::{
    SuperGraphAnyLink, SuperGraphGraphPath, SuperGraphInner, SuperGraphLinkTensor,
    SuperGraphNodeId, SuperGraphNodePath, SuperGraphTensorPath,
};
use whisper_tensor::symbolic_graph::ops::{AnyOperation, Operation};
use whisper_tensor::symbolic_graph::{
    StoredOrNotTensor, SymbolicGraph, SymbolicGraphInner, SymbolicGraphOperationId,
    SymbolicGraphTensorId, TensorType,
};
use whisper_tensor::tokenizer::Tokenizer;
use whisper_tensor_import::onnx_graph::TokenizerInfo;
use whisper_tensor_import::onnx_graph::tensor::Tensor;
use whisper_tensor_server::{
    AbbreviatedTensorReportSettings, AbbreviatedTensorValue, LoadedModelId, ServerConfigReport,
    SuperGraphRequest, SuperGraphRequestBackendMode, WebsocketClientServerMessage,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct GraphExplorerState {
    explorer_physics: bool,
    explorer_minimap: bool,
    do_all_explorer_swatches: bool,
    do_explorer_swatches_in_view: bool,
    explorer_node_wave: bool,
    swatch_dimension: usize,
}

impl Default for GraphExplorerState {
    fn default() -> Self {
        Self {
            explorer_physics: false,
            explorer_minimap: false,
            do_all_explorer_swatches: false,
            do_explorer_swatches_in_view: false,
            explorer_node_wave: false,
            swatch_dimension: 32,
        }
    }
}

pub(crate) enum GraphSubject<'a> {
    SymbolicGraphInner(&'a SymbolicGraphInner),
    SuperGraphInner(&'a SuperGraphInner),
    MilliOpGraphB(&'a MilliOpGraph<SuperGraphLinkTensor>),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum GraphExplorerLayerSelection {
    Model(LoadedModelId),
    Interface(InterfaceId),
    SymbolicGraphOperationId((SymbolicGraphOperationId, usize)),
    SuperGraphNodeId(SuperGraphNodeId),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum GraphExplorerSelectable {
    SymbolicGraphOperationId(SymbolicGraphOperationId),
    SymbolicGraphTensorId(SymbolicGraphTensorId),
    SuperGraphNodeId(SuperGraphNodeId),
    SuperGraphLink(SuperGraphAnyLink),
    MilliOpGraphTensor(MilliOpGraphTensorId),
    MilliOpGraphNode(MilliOpGraphNodeId),
}

#[derive(Clone, Debug, Default)]
struct TextInferenceData {
    tokens: Vec<u32>,
    logits: HashMap<Vec<u32>, Vec<(u32, f32)>>,
    pending_request: Option<(u64, SuperGraphLinkTensor, Vec<u32>)>,
    use_cache: bool,
    selected_mode: SuperGraphRequestBackendMode,
}

pub(crate) struct GraphExplorerApp {
    pub(crate) explorer_selection: Option<GraphExplorerSelectable>,
    pub(crate) explorer_hovered: Option<GraphExplorerSelectable>,
    pub(crate) next_explorer_hovered: Option<GraphExplorerSelectable>,
    inspect_window_tensor_subscriptions: HashSet<SuperGraphTensorPath>,
    inspect_window_tensor_subscription_returns:
        HashMap<SuperGraphTensorPath, NDArrayNumericTensor<DynRank>>,
    pub inspect_windows: Vec<InspectWindow>,
    graph_layouts: HashMap<Vec<GraphExplorerLayerSelection>, Result<GraphLayout, GraphLayoutError>>,
    pub loaded_models: HashMap<LoadedModelId, SymbolicGraph>,
    model_view_scene_rects: HashMap<Vec<GraphExplorerLayerSelection>, Rect>,
    pub(crate) graph_subject_path: Vec<GraphExplorerLayerSelection>,
    pub(crate) next_graph_subject_path: Option<Vec<GraphExplorerLayerSelection>>,
    pub(crate) text_inference_data: HashMap<InterfaceId, TextInferenceData>,
    node_execution_timestamps: HashMap<SuperGraphNodePath, Instant>,
    node_execution_durations: HashMap<SuperGraphNodePath, Duration>,
    abbreviated_tensor_reports: HashMap<SuperGraphTensorPath, AbbreviatedTensorValue>,
    rendered_tensor_swatches: HashMap<SuperGraphTensorPath, TextureHandle>,
    tensors_in_view: HashSet<SuperGraphTensorPath>,
    nodes_in_view: HashSet<SuperGraphNodePath>,
    error_popup: Option<String>,
}

fn render_node_contents<'a>(
    ui: &mut egui::Ui,
    node_type: &GraphLayoutNodeType,
    num_inputs: usize,
    num_outputs: usize,
    graph_subject: &GraphSubject<'a>,
    is_selected: bool,
    is_hovered: bool,
    time_since_exec: Option<Duration>,
) -> (Response, GraphLayoutIOOffsets) {
    // Decide corner radius
    let corner_radius = match node_type {
        GraphLayoutNodeType::SymbolicGraphOperation(_)
        | GraphLayoutNodeType::SuperGraphNode(_)
        | GraphLayoutNodeType::MilliOpGraphNode(_) => 3.0,
        _ => 10.0,
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
            GraphLayoutNodeType::SuperGraphNode(node_id) => {
                if let GraphSubject::SuperGraphInner(super_graph) = graph_subject {
                    let node = &super_graph.nodes[&node_id];
                    ui.add(Label::new(node.get_name()).selectable(false));
                }
            }
            GraphLayoutNodeType::SuperGraphLink(link) => {
                let text = match link {
                    SuperGraphAnyLink::Tensor(_) => "Tensor",
                    SuperGraphAnyLink::String(_) => "String",
                    SuperGraphAnyLink::Model(_) => "Model",
                    SuperGraphAnyLink::Tokenizer(_) => "Tokenizer",
                    SuperGraphAnyLink::Hash(_) => "Hash",
                };
                ui.add(Label::new(text.to_string()).selectable(false));
            }
            GraphLayoutNodeType::SymbolicGraphOperation(op_id) => {
                if let GraphSubject::SymbolicGraphInner(symbolic_graph) = graph_subject {
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
            GraphLayoutNodeType::SymbolicGraphTensor(tensor_id) => {
                if let GraphSubject::SymbolicGraphInner(symbolic_graph) = graph_subject {
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
            GraphLayoutNodeType::MilliOpGraphNode(node_id) => {
                if let GraphSubject::MilliOpGraphB(milli_graph) = graph_subject {
                    if let Some(op) = milli_graph.get_node(node_id) {
                        ui.add(Label::new(op.op_kind()).selectable(false));
                    }
                }
            }
            GraphLayoutNodeType::MilliOpGraphInput(_) => {
                ui.add(Label::new("Input").selectable(false));
            }
            GraphLayoutNodeType::MilliOpGraphOutput(_) => {
                ui.add(Label::new("Output").selectable(false));
            }
            GraphLayoutNodeType::ConnectionByNameSrc(name) => match &name.link_type {
                GraphLayoutLinkType::SymbolicGraphTensor(tensor_id) => {
                    if let GraphSubject::SymbolicGraphInner(symbolic_graph) = graph_subject {
                        let value = symbolic_graph.get_tensor_info(tensor_id.clone()).unwrap();
                        let name = value
                            .onnx_name
                            .clone()
                            .unwrap_or(format!("tensor {}", tensor_id));
                        ui.add(Label::new(format!("{} >", name)).selectable(false));
                    }
                }
                GraphLayoutLinkType::SuperGraphLink(link) => {
                    ui.add(Label::new(format!("{:?} >", link)).selectable(false));
                }
                GraphLayoutLinkType::MilliOpGraphTensor(link) => {
                    ui.add(Label::new(format!("{:?} >", link)).selectable(false));
                }
            },
            GraphLayoutNodeType::ConnectionByNameDest(name) => match &name.link_type {
                GraphLayoutLinkType::SymbolicGraphTensor(tensor_id) => {
                    if let GraphSubject::SymbolicGraphInner(symbolic_graph) = graph_subject {
                        let value = symbolic_graph.get_tensor_info(tensor_id.clone()).unwrap();
                        let name = value
                            .onnx_name
                            .clone()
                            .unwrap_or(format!("tensor {}", tensor_id));
                        ui.add(Label::new(format!("> {}", name)).selectable(false));
                    }
                }
                GraphLayoutLinkType::SuperGraphLink(link) => {
                    ui.add(Label::new(format!("> {:?}", link)).selectable(false));
                }
                GraphLayoutLinkType::MilliOpGraphTensor(link) => {
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

    let active_pulse = if let Some(x) = time_since_exec {
        (1.0f32 - x.as_secs_f32() / 0.5f32).max(0.0f32)
    } else {
        0.0
    };

    let active_color_border =
        egui::Color32::from_rgba_unmultiplied(255, 222, 33, (active_pulse * 128.0) as u8);
    let active_color_fill =
        egui::Color32::from_rgba_unmultiplied(255, 222, 33, (active_pulse * 64.0) as u8);

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
    frame.frame.fill = fill + active_color_fill;
    frame.frame.stroke = Stroke {
        width: stroke.width,
        color: stroke.color + active_color_border,
    };
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
            explorer_selection: None,
            explorer_hovered: None,
            next_explorer_hovered: None,
            inspect_window_tensor_subscription_returns: HashMap::new(),
            inspect_window_tensor_subscriptions: HashSet::new(),
            inspect_windows: Vec::new(),
            graph_layouts: HashMap::new(),
            loaded_models: HashMap::new(),
            model_view_scene_rects: HashMap::new(),
            graph_subject_path: Vec::new(),
            next_graph_subject_path: None,
            text_inference_data: HashMap::new(),
            node_execution_timestamps: HashMap::new(),
            node_execution_durations: HashMap::new(),
            abbreviated_tensor_reports: HashMap::new(),
            rendered_tensor_swatches: HashMap::new(),
            nodes_in_view: HashSet::new(),
            tensors_in_view: HashSet::new(),
            error_popup: None,
        }
    }

    pub(crate) fn get_tensor_swatch(
        rendered_tensor_swatches: &mut HashMap<SuperGraphTensorPath, TextureHandle>,
        abbreviated_tensor_reports: &HashMap<SuperGraphTensorPath, AbbreviatedTensorValue>,
        state: &mut GraphExplorerState,
        ctx: &Context,
        path: &SuperGraphTensorPath,
    ) -> Option<TextureHandle> {
        if let Some(swatch) = rendered_tensor_swatches.get(&path) {
            Some(swatch.clone())
        } else if let Some(x) = abbreviated_tensor_reports.get(path) {
            if let Some(raw_texture) =
                build_tensor_swatch(x, state.swatch_dimension, state.swatch_dimension)
            {
                let image_data = Arc::new(ColorImage::new(
                    [state.swatch_dimension, state.swatch_dimension],
                    raw_texture,
                ));
                let texture = ctx.load_texture("swatch", image_data, egui::TextureOptions::NEAREST);
                rendered_tensor_swatches.insert(path.clone(), texture.clone());
                Some(texture)
            } else {
                None
            }
        } else {
            None
        }
    }

    pub(crate) fn update(
        &mut self,
        state: &mut GraphExplorerState,
        loaded_models: &mut LoadedModels,
        loaded_tokenizers: &mut LoadedTokenizers,
        server_request_manager: &mut ServerRequestManager,
        server_config_report: &ServerConfigReport,
        ui: &mut Ui,
    ) {
        if let Some(err) = self.error_popup.clone() {
            egui::Modal::new(egui::Id::new("Eval Error")).show(ui.ctx(), |ui| {
                ui.scope(|ui| {
                    ui.visuals_mut().override_text_color = Some(egui::Color32::RED);
                    ui.style_mut().override_text_style = Some(egui::TextStyle::Monospace);
                    ui.label(err);
                });
                if ui.button("Dismiss").clicked() {
                    self.error_popup = None;
                }
            });
        }

        self.explorer_hovered = self.next_explorer_hovered.clone();
        self.next_explorer_hovered = None;

        ui.horizontal(|ui| {
            let model_selector_options = {
                let mut options = vec![];
                for (interface_id, interface) in &loaded_models.current_interfaces {
                    options.push((
                        GraphExplorerLayerSelection::Interface(*interface_id),
                        format!("({}) {}", interface_id, interface.interface_name.clone()),
                    ));
                }
                for model in &loaded_models.current_models {
                    options.push((
                        GraphExplorerLayerSelection::Model(model.model_id),
                        format!("({}) {}", model.model_id, model.model_name.clone()),
                    ));
                }
                options
            };
            let mut value = self.graph_subject_path.first().cloned();
            let initial_value = value.clone();
            egui::ComboBox::from_id_salt(123662)
                .selected_text(
                    model_selector_options
                        .iter()
                        .find(|(a, b)| value.as_ref().map(|x| x == a).unwrap_or(false))
                        .map(|(a, b)| b.clone())
                        .unwrap_or("Select a model or interface".to_string()),
                )
                .show_ui(ui, |ui| {
                    for (a, b) in model_selector_options {
                        ui.selectable_value(&mut value, Some(a), b.to_string());
                    }
                });
            if value != initial_value {
                self.next_graph_subject_path = value.map(|x| vec![x]);
            }
            if ui.button("Load New Model").clicked() {
                loaded_models.model_load_state = Some(ModelLoadState::DialogOpen(None));
            };
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                toggle_ui(ui, &mut state.explorer_minimap);
                ui.label("Minimap:");
                toggle_ui(ui, &mut state.explorer_physics);
                ui.label("Physics:");
                toggle_ui(ui, &mut state.explorer_node_wave);
                ui.label("Activity:");
                toggle_ui(ui, &mut state.do_all_explorer_swatches);
                ui.label("All Swatches:");
                toggle_ui(ui, &mut state.do_explorer_swatches_in_view);
                ui.label("Swatches In-frame:");
            })
        });

        if let Some(next_path) = self.next_graph_subject_path.take() {
            if next_path.first() != self.graph_subject_path.first() {
                self.node_execution_timestamps.clear();
                self.abbreviated_tensor_reports.clear();
                self.rendered_tensor_swatches.clear();
            }
            self.graph_subject_path = next_path;
            self.explorer_selection = None;
            self.explorer_hovered = None;
            self.inspect_window_tensor_subscription_returns.clear();
            self.inspect_window_tensor_subscriptions.clear();
            self.inspect_windows.clear();
        }

        let (graph_subjects, selected_interface, selected_interface_id, is_loading) = {
            let mut graph_subjects: Vec<(
                GraphSubject,
                Vec<GraphExplorerLayerSelection>,
                Option<SuperGraphGraphPath>,
                Option<GraphExplorerLayerSelection>,
            )> = vec![];
            let mut compiled_path = vec![];
            let mut current_graph_path = None;
            let mut selected_interface = None;
            let mut selected_interface_id = None;
            let mut is_loading = false;
            for layer_selection in &self.graph_subject_path {
                let layer_subject = match layer_selection {
                    GraphExplorerLayerSelection::Model(model_id) => {
                        current_graph_path = None;
                        let res = self
                            .loaded_models
                            .get(model_id)
                            .map(|x| GraphSubject::SymbolicGraphInner(&x.inner_graph));
                        if res.is_none() {
                            is_loading = true;
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
                        res
                    }
                    GraphExplorerLayerSelection::Interface(interface_id) => {
                        if let Some(interface) = loaded_models.current_interfaces.get(interface_id)
                        {
                            selected_interface_id = Some(*interface_id);
                            selected_interface = Some(interface);
                            current_graph_path = Some(SuperGraphGraphPath::SuperGraph(vec![]));
                            Some(GraphSubject::SuperGraphInner(
                                &interface.interface.get_super_graph().inner,
                            ))
                        } else {
                            None
                        }
                    }
                    GraphExplorerLayerSelection::SymbolicGraphOperationId((operation_id, i)) => {
                        if let Some(_x) = &current_graph_path {
                            // todo: when we have nested symbolic graph support
                            current_graph_path = None;
                        }
                        if let Some((GraphSubject::SymbolicGraphInner(graph_subject), _, _, _)) =
                            graph_subjects.last()
                        {
                            graph_subject
                                .get_operations()
                                .get(operation_id)
                                .map(|x| {
                                    x.op.get_sub_graphs()
                                        .get(*i)
                                        .map(|x| GraphSubject::SymbolicGraphInner(*x))
                                })
                                .flatten()
                        } else {
                            None
                        }
                    }
                    GraphExplorerLayerSelection::SuperGraphNodeId(node_id) => {
                        if let Some((GraphSubject::SuperGraphInner(graph_subject), _, _, _)) =
                            graph_subjects.last()
                        {
                            let node = graph_subject.nodes.get(node_id);
                            if let Some(node) = node {
                                match node {
                                    SuperGraphAnyNode::ModelExecution(_) => {
                                        if let Some(interface) = selected_interface
                                            && let Some(model_id) = interface.model_ids.first()
                                        {
                                            if let Some(x) = &current_graph_path {
                                                current_graph_path = Some(
                                                    x.push_model_execution_node_graph(*node_id),
                                                );
                                            }
                                            let res = self.loaded_models.get(model_id).map(|x| {
                                                GraphSubject::SymbolicGraphInner(&x.inner_graph)
                                            });
                                            if res.is_none() {
                                                is_loading = true;
                                                if loaded_models
                                                    .currently_requesting_model
                                                    .map(|id| id != *model_id)
                                                    .unwrap_or(true)
                                                {
                                                    server_request_manager
                                                        .send(WebsocketClientServerMessage::GetModelGraph(*model_id))
                                                        .unwrap();
                                                    loaded_models.currently_requesting_model =
                                                        Some(*model_id);
                                                }
                                            }
                                            res
                                        } else {
                                            None
                                        }
                                    }
                                    SuperGraphAnyNode::MilliOpGraph(x) => {
                                        if let Some(x) = &current_graph_path {
                                            current_graph_path =
                                                Some(x.push_super_milli_op_node_graph(*node_id));
                                        }
                                        Some(GraphSubject::MilliOpGraphB(&x.graph))
                                    }
                                    _ => {
                                        if let Some(inner_graph) = node.get_sub_graph() {
                                            if let Some(x) = &current_graph_path {
                                                current_graph_path =
                                                    Some(x.push_super_node_graph(*node_id));
                                            }
                                            Some(GraphSubject::SuperGraphInner(inner_graph))
                                        } else {
                                            None
                                        }
                                    }
                                }
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    }
                };
                compiled_path.push(layer_selection.clone());
                if let Some(x) = graph_subjects.last_mut() {
                    x.3 = Some(layer_selection.clone());
                }
                if let Some(layer_subject) = layer_subject {
                    graph_subjects.push((
                        layer_subject,
                        compiled_path.clone(),
                        current_graph_path.clone(),
                        None,
                    ));
                }
            }
            (
                graph_subjects,
                selected_interface,
                selected_interface_id,
                is_loading,
            )
        };

        let current_time = Instant::now();
        ui.horizontal(|ui| {
            if state.explorer_minimap {
                let mut do_request_redraw = false;
                for (i, (_graph_subject, graph_subject_path, super_graph_path, selected_entry)) in
                    graph_subjects.iter().enumerate()
                {
                    let maps_left = graph_subjects.len() - i;
                    let subject_width = if maps_left <= 1 {
                        ui.available_size_before_wrap().x
                    } else {
                        (ui.available_size_before_wrap().x / (maps_left as f32)) - 5.0
                    };
                    if let Some(Ok(graph_layout)) = self.graph_layouts.get(graph_subject_path) {
                        let node_bounding_rect = graph_layout.get_bounding_rect().expand(15.0);

                        let height = 80.0;
                        let width = (node_bounding_rect.width()
                            * (height / node_bounding_rect.height()))
                        .min(subject_width)
                        .max(30.0);

                        // Get frame min and max
                        let minimap_frame =
                            egui::Frame::default().stroke(ui.visuals().window_stroke);
                        minimap_frame.show(ui, |ui| {
                            let shape_request = vec2(width, 80.0);
                            let (outer_rect, outer_response) =
                                ui.allocate_exact_size(shape_request, Sense::drag());
                            let transform = outer_rect.size() / node_bounding_rect.size();
                            if outer_response.dragged() {
                                if let Some(outer_pos) = outer_response.interact_pointer_pos() {
                                    let inner_pos = node_bounding_rect.min
                                        + ((outer_pos - outer_rect.min) / transform);
                                    if let Some(selected_area) =
                                        self.model_view_scene_rects.get_mut(graph_subject_path)
                                    {
                                        *selected_area =
                                            Rect::from_center_size(inner_pos, selected_area.size());
                                    }
                                }
                                if maps_left > 1 {
                                    self.next_graph_subject_path = Some(graph_subject_path.clone());
                                }
                            }
                            for (_node_id, node) in graph_layout.get_nodes() {
                                let pos = (node.position - node_bounding_rect.min) * transform;

                                let this_selectable = node.node_type.get_graph_selectable();

                                let is_selected = self.explorer_selection
                                    == Some(this_selectable.clone())
                                    && selected_entry.is_none();
                                let is_hovered = self.explorer_hovered
                                    == Some(this_selectable.clone())
                                    && selected_entry.is_none();

                                let node_path = if let Some(graph_path) = super_graph_path {
                                    match &node.node_type {
                                        GraphLayoutNodeType::SuperGraphNode(super_graph_node) => {
                                            Some(graph_path.push_super_node(*super_graph_node))
                                        }
                                        GraphLayoutNodeType::SymbolicGraphOperation(x) => {
                                            Some(graph_path.push_symbolic_node(*x))
                                        }
                                        GraphLayoutNodeType::SymbolicGraphTensor(_x) => None,
                                        GraphLayoutNodeType::SuperGraphLink(_x) => None,
                                        GraphLayoutNodeType::MilliOpGraphNode(x) => {
                                            Some(graph_path.push_milli_op_node(*x))
                                        }
                                        GraphLayoutNodeType::MilliOpGraphInput(_x) => None,
                                        GraphLayoutNodeType::MilliOpGraphOutput(_x) => None,
                                        GraphLayoutNodeType::ConnectionByNameSrc(_x) => None,
                                        GraphLayoutNodeType::ConnectionByNameDest(_x) => None,
                                    }
                                } else {
                                    None
                                };

                                let time_since_last_eval: Option<Duration> = if let Some(x) =
                                    node_path
                                    && let Some(eval_time) = self.node_execution_timestamps.get(&x)
                                {
                                    Some(current_time - *eval_time)
                                } else {
                                    None
                                };

                                let active_pulse = if let Some(x) = time_since_last_eval {
                                    (1.0f32 - x.as_secs_f32() / 0.5f32).max(0.0f32)
                                } else {
                                    0.0
                                };

                                if active_pulse > 0.0 {
                                    do_request_redraw = true;
                                }

                                let is_selected_subgraph = match &node.node_type {
                                    GraphLayoutNodeType::SymbolicGraphOperation(op_id) => {
                                        if let Some(
                                            GraphExplorerLayerSelection::SymbolicGraphOperationId(
                                                (selected_op_id, _),
                                            ),
                                        ) = selected_entry
                                        {
                                            *selected_op_id == *op_id
                                        } else {
                                            false
                                        }
                                    }
                                    GraphLayoutNodeType::SuperGraphNode(super_graph_node) => {
                                        if let Some(
                                            GraphExplorerLayerSelection::SuperGraphNodeId(
                                                selected_node_id,
                                            ),
                                        ) = selected_entry
                                        {
                                            *selected_node_id == *super_graph_node
                                        } else {
                                            false
                                        }
                                    }
                                    _ => false,
                                };

                                let color = if is_selected || is_selected_subgraph {
                                    egui::Color32::from_rgba_unmultiplied(64, 64, 255, 128)
                                } else if is_hovered {
                                    egui::Color32::from_rgba_unmultiplied(80, 80, 80, 128)
                                } else {
                                    egui::Color32::from_rgba_unmultiplied(64, 64, 64, 128)
                                };

                                let active_color = egui::Color32::from_rgba_unmultiplied(
                                    255,
                                    222,
                                    33,
                                    (active_pulse * 128.0) as u8,
                                );

                                let color = color + active_color;

                                let mut size = node.shape * transform;
                                size.x = size.x.max(2.0);
                                size.y = size.y.max(2.0);

                                let radius = 10.0 * (transform.x + transform.y) / 2.0;

                                ui.painter().add(egui::Shape::rect_filled(
                                    Rect::from_center_size(outer_rect.min + pos, size),
                                    radius,
                                    color,
                                ));
                                /*
                                ui.painter().add(egui::Shape::circle_filled(
                                    outer_rect.min + pos,
                                    radius,
                                    color,
                                ));*/
                            }
                            if let Some(selected_area) =
                                self.model_view_scene_rects.get(graph_subject_path)
                            {
                                let transformed_area = Rect::from_min_max(
                                    (outer_rect.min
                                        + ((selected_area.min - node_bounding_rect.min)
                                            * transform))
                                        .max(outer_rect.min),
                                    (outer_rect.min
                                        + ((selected_area.max - node_bounding_rect.min)
                                            * transform))
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
                if do_request_redraw {
                    ui.ctx().request_repaint_after(Duration::from_millis(20));
                }
            }
        });

        let do_interface_panel = selected_interface.is_some();
        let interface_panel_height = 150.0;

        if is_loading {
            ui.label("Loading Model Graph...");
            ui.spinner();
        } else if let Some((graph_subject, _, graph_path, _)) = graph_subjects.last() {
            if !self.graph_layouts.contains_key(&self.graph_subject_path) {
                match graph_subject {
                    GraphSubject::SymbolicGraphInner(graph) => {
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
                                    link_type: GraphLayoutLinkType::SymbolicGraphTensor(*tensor_id),
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
                                    node_type: GraphLayoutNodeType::SymbolicGraphOperation(*op_id),
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
                                            node_type: GraphLayoutNodeType::SymbolicGraphTensor(
                                                *tensor_id,
                                            ),
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
                                            node_type: GraphLayoutNodeType::SymbolicGraphTensor(
                                                *tensor_id,
                                            ),
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
                                                    *tensor_id,
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
                                            node_type: GraphLayoutNodeType::SymbolicGraphTensor(
                                                *tensor_id,
                                            ),
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
                                    &graph_subject,
                                    false,
                                    false,
                                    None,
                                )
                                .1
                            },
                        );
                        self.graph_layouts
                            .insert(self.graph_subject_path.clone(), initial_layout);
                    }
                    GraphSubject::SuperGraphInner(super_graph_inner) => {
                        // Map links
                        let mut link_ids = HashMap::new();
                        let mut next_link_id = 0;
                        let mut link_data = HashMap::new();
                        for super_graph_link in super_graph_inner.get_all_links() {
                            let new_link_id = GraphLayoutLinkId(next_link_id);
                            next_link_id += 1;
                            link_ids.insert(super_graph_link.clone(), new_link_id);
                            link_data.insert(
                                new_link_id,
                                GraphLayoutLinkData {
                                    link_type: GraphLayoutLinkType::SuperGraphLink(
                                        super_graph_link,
                                    ),
                                },
                            );
                        }

                        // Map nodes
                        let mut sourced_links = HashSet::new();
                        let mut next_node_id = 0;
                        let mut node_init_data = HashMap::new();
                        for (node_id, node) in super_graph_inner.nodes.iter() {
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
                                    node_type: GraphLayoutNodeType::SuperGraphNode(node_id.clone()),
                                    inputs,
                                    outputs,
                                },
                            );
                        }

                        // I/O links
                        for input_link in &super_graph_inner.input_links {
                            let new_node_id = GraphLayoutNodeId(next_node_id);
                            next_node_id += 1;

                            let mut inputs = vec![];
                            let mut outputs = vec![link_ids[input_link]];
                            sourced_links.insert(link_ids[input_link]);

                            node_init_data.insert(
                                new_node_id,
                                GraphLayoutNodeInitData {
                                    node_type: GraphLayoutNodeType::SuperGraphLink(
                                        input_link.clone(),
                                    ),
                                    inputs,
                                    outputs,
                                },
                            );
                        }
                        for output_link in &super_graph_inner.output_links {
                            let new_node_id = GraphLayoutNodeId(next_node_id);
                            next_node_id += 1;

                            let mut inputs = vec![link_ids[output_link]];
                            let mut outputs = vec![];
                            node_init_data.insert(
                                new_node_id,
                                GraphLayoutNodeInitData {
                                    node_type: GraphLayoutNodeType::SuperGraphLink(
                                        output_link.clone(),
                                    ),
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
                                    &graph_subject,
                                    false,
                                    false,
                                    None,
                                )
                                .1
                            },
                        );
                        self.graph_layouts
                            .insert(self.graph_subject_path.clone(), initial_layout);
                    }
                    GraphSubject::MilliOpGraphB(milli_op_graph) => {
                        // Map links
                        let mut link_ids = HashMap::new();
                        let mut next_link_id = 0;
                        let mut link_data = HashMap::new();
                        for tensor_id in milli_op_graph.get_all_tensors() {
                            let new_link_id = GraphLayoutLinkId(next_link_id);
                            next_link_id += 1;
                            link_ids.insert(tensor_id.clone(), new_link_id);
                            link_data.insert(
                                new_link_id,
                                GraphLayoutLinkData {
                                    link_type: GraphLayoutLinkType::MilliOpGraphTensor(tensor_id),
                                },
                            );
                        }

                        // Map nodes
                        let mut sourced_links = HashSet::new();
                        let mut next_node_id = 0;
                        let mut node_init_data = HashMap::new();
                        for node_id in milli_op_graph.nodes() {
                            let new_node_id = GraphLayoutNodeId(next_node_id);
                            next_node_id += 1;

                            let node = milli_op_graph.get_node(&node_id).expect("node exists");

                            let mut inputs = vec![];
                            for tensor_id in node.inputs() {
                                inputs.push(link_ids[&tensor_id]);
                            }
                            let mut outputs = vec![];
                            for tensor_id in node.outputs() {
                                outputs.push(link_ids[&tensor_id]);
                            }
                            node_init_data.insert(
                                new_node_id,
                                GraphLayoutNodeInitData {
                                    node_type: GraphLayoutNodeType::MilliOpGraphNode(node_id),
                                    inputs,
                                    outputs,
                                },
                            );
                        }

                        // I/O links
                        for input_link in milli_op_graph.input_map.values() {
                            let new_node_id = GraphLayoutNodeId(next_node_id);
                            next_node_id += 1;

                            let mut inputs = vec![];
                            let mut outputs = vec![link_ids[input_link]];
                            sourced_links.insert(link_ids[input_link]);

                            node_init_data.insert(
                                new_node_id,
                                GraphLayoutNodeInitData {
                                    node_type: GraphLayoutNodeType::MilliOpGraphInput(
                                        input_link.clone(),
                                    ),
                                    inputs,
                                    outputs,
                                },
                            );
                        }
                        for output_link in milli_op_graph.output_map.as_ref().unwrap().keys() {
                            let new_node_id = GraphLayoutNodeId(next_node_id);
                            next_node_id += 1;

                            let mut inputs = vec![link_ids[output_link]];
                            let mut outputs = vec![];
                            node_init_data.insert(
                                new_node_id,
                                GraphLayoutNodeInitData {
                                    node_type: GraphLayoutNodeType::MilliOpGraphOutput(
                                        output_link.clone(),
                                    ),
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
                                    &graph_subject,
                                    false,
                                    false,
                                    None,
                                )
                                .1
                            },
                        );
                        self.graph_layouts
                            .insert(self.graph_subject_path.clone(), initial_layout);
                    }
                }
            }

            match self.graph_layouts.get_mut(&self.graph_subject_path) {
                Some(Ok(graph_layout)) => {
                    // Update positions
                    if state.explorer_physics {
                        if graph_layout.update_layout(5000) {
                            ui.ctx().request_repaint_after(Duration::from_millis(20));
                        }
                    }
                    let mut frame_shape = ui.available_size_before_wrap();
                    if do_interface_panel {
                        frame_shape.y -= interface_panel_height;
                    }
                    let frame_base = ui.available_rect_before_wrap().min;
                    let ui_builder = UiBuilder::new()
                        .max_rect(Rect::from_min_max(frame_base, frame_base + frame_shape));
                    ui.scope_builder(ui_builder, |ui| {
                        let frame = egui::Frame::default().stroke(ui.visuals().window_stroke);
                        frame.show(ui, |ui| {

                            let mut scene_rect =  if let Some(x) =  self.model_view_scene_rects.get(&self.graph_subject_path) {
                                x.clone()
                            } else {
                                // No bound of graph should exceed this (unless necessary for aspect ratio)
                                let max_frame = graph_layout.get_bounding_rect().expand(30.0);

                                let (min_x, max_x) = if max_frame.width() > frame_shape.x{
                                    (max_frame.min.x, max_frame.min.x + frame_shape.x)
                                } else {
                                    (max_frame.min.x, max_frame.max.x)
                                };

                                let center_y = max_frame.center().y;
                                let center_pos = egui::pos2(min_x + (max_x - min_x) / 2.0, center_y);
                                let scale = (frame_shape.x / (max_x - min_x)).max(0.0).min(0.8);

                                Rect::from_center_size(
                                    center_pos,
                                    frame_shape * scale,
                                )
                            };
                            let cull_rect = scene_rect.expand(300.0);
                            let scene = egui::Scene::new().max_inner_size(frame_shape);
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
                                        let shape_idx_a = ui.painter().add(Shape::Noop);
                                        edges_to_render.push((
                                            (*src_id, *src_id_i),
                                            (*dst_id, *dst_id_i),
                                            *link_id,
                                            shape_idx_a,
                                        ));
                                    }
                                }
                                // Also allocate shapes for swatches, but at a higher level
                                let edges_to_render = edges_to_render.into_iter().map(|x| (x.0, x.1, x.2, (x.3, ui.painter().add(Shape::Noop), ui.painter().add(Shape::Noop)))).collect::<Vec<_>>();

                                let mut node_io_connections = HashMap::new();
                                let mut node_bounding_boxes = HashMap::new();

                                self.nodes_in_view.clear();
                                let current_time = Instant::now();
                                let current_node_data = graph_layout.get_nodes();
                                let mut node_position_updates = HashMap::new();
                                for node_id in nodes_to_render_vec {
                                    let node_path = if let Some(graph_path) = graph_path {
                                        match &current_node_data[&node_id].node_type {
                                            GraphLayoutNodeType::SuperGraphNode(super_graph_node) => {
                                                Some(graph_path.push_super_node(*super_graph_node))
                                            }
                                            GraphLayoutNodeType::SymbolicGraphOperation(x) => {
                                                Some(graph_path.push_symbolic_node(*x))
                                            }
                                            GraphLayoutNodeType::SymbolicGraphTensor(_x) => {
                                                None
                                            }
                                            GraphLayoutNodeType::SuperGraphLink(_x) => {
                                                None
                                            }
                                            GraphLayoutNodeType::MilliOpGraphNode(x) => {
                                                Some(graph_path.push_milli_op_node(*x))
                                            }
                                            GraphLayoutNodeType::MilliOpGraphInput(_x) => {
                                                None
                                            }
                                            GraphLayoutNodeType::MilliOpGraphOutput(_x) => {None}
                                            GraphLayoutNodeType::ConnectionByNameSrc(_x) => {None}
                                            GraphLayoutNodeType::ConnectionByNameDest(_x) => {None}
                                        }
                                    } else {
                                        None
                                    };
                                    if let Some(node_path) = &node_path {
                                        self.nodes_in_view.insert(node_path.clone());
                                    }
                                    let duration_since_eval = if let Some(node_path) = &node_path {
                                        if let Some(x) = self.node_execution_timestamps.get(node_path) {
                                            Some(current_time - *x)
                                        } else {
                                            None
                                        }
                                    } else {
                                        None
                                    };
                                    let pos = current_node_data[&node_id].position;
                                    let cell_shape = current_node_data[&node_id].shape;
                                    let op_rect = Rect::from_min_max(
                                        pos.clone() - cell_shape / 2.0,
                                        pos.clone() + cell_shape,
                                    );
                                    let ui_builder = UiBuilder::new().max_rect(op_rect);
                                    let mut ui_child = ui.new_child(ui_builder);

                                    let this_selectable = current_node_data[&node_id].node_type.get_graph_selectable();

                                    let is_selected = if let Some(selected) = &self.explorer_selection {
                                        this_selectable == *selected
                                    } else {
                                        false
                                    };
                                    let is_hovered = if let Some(hovered) = &self.explorer_hovered {
                                        this_selectable == *hovered
                                    } else {
                                        false
                                    };

                                    let (resp, io_connections) = render_node_contents(
                                        &mut ui_child,
                                        &current_node_data[&node_id].node_type,
                                        current_node_data[&node_id].inputs.len(),
                                        current_node_data[&node_id].outputs.len(),
                                        &graph_subject,
                                        is_selected,
                                        is_hovered,
                                        duration_since_eval,
                                    );
                                    node_io_connections.insert(node_id, io_connections);
                                    node_bounding_boxes.insert(node_id, ui_child.min_rect());

                                    if resp.hovered() {
                                        self.next_explorer_hovered = Some(this_selectable.clone());
                                    }
                                    if resp.clicked() {
                                        self.explorer_selection = Some(this_selectable.clone());
                                    }
                                    if resp.dragged() {
                                        node_position_updates.insert(node_id, current_node_data.get(&node_id).unwrap().position + resp.drag_delta());
                                    }
                                    if resp.double_clicked() {
                                        match &current_node_data[&node_id].node_type {
                                            GraphLayoutNodeType::SymbolicGraphOperation(_) => {
                                                if !InspectWindow::check_if_already_exists(
                                                    &self.inspect_windows,
                                                    &this_selectable.clone(),
                                                ) && let Some(x) = InspectWindow::new(this_selectable){
                                                    self.inspect_windows.push(x);
                                                }
                                            }
                                            GraphLayoutNodeType::SymbolicGraphTensor(_tensor_id) => {
                                                if !InspectWindow::check_if_already_exists(
                                                    &self.inspect_windows,
                                                    &this_selectable.clone(),
                                                ) && let Some(x) = InspectWindow::new(this_selectable){
                                                    self.inspect_windows.push(x);
                                                }
                                            }
                                            GraphLayoutNodeType::SuperGraphNode(node_id) => {
                                                if let GraphSubject::SuperGraphInner(x) = &graph_subject {
                                                    if let Some(node) = x.nodes.get(node_id) {
                                                        match node {
                                                            SuperGraphAnyNode::MilliOpGraph(_) => {
                                                                let mut path = self.graph_subject_path.clone();
                                                                path.push(GraphExplorerLayerSelection::SuperGraphNodeId(node_id.clone()));
                                                                self.next_graph_subject_path = Some(path);
                                                            }
                                                            SuperGraphAnyNode::ModelExecution(_) => {
                                                                let mut path = self.graph_subject_path.clone();
                                                                path.push(GraphExplorerLayerSelection::SuperGraphNodeId(node_id.clone()));
                                                                self.next_graph_subject_path = Some(path);
                                                            }
                                                            _ => {
                                                                if let Some(_) = node.get_sub_graph() {
                                                                    let mut path = self.graph_subject_path.clone();
                                                                    path.push(GraphExplorerLayerSelection::SuperGraphNodeId(node_id.clone()));
                                                                    self.next_graph_subject_path = Some(path);
                                                                }
                                                            }
                                                        }

                                                    }
                                                }
                                                // TODO: Inspect this
                                            }
                                            GraphLayoutNodeType::ConnectionByNameSrc(link_data)
                                            | GraphLayoutNodeType::ConnectionByNameDest(link_data) => {
                                                match &link_data.link_type {
                                                    GraphLayoutLinkType::SymbolicGraphTensor(
                                                        tensor_id,
                                                    ) => {
                                                        let new_inspect = GraphExplorerSelectable::SymbolicGraphTensorId(*tensor_id);
                                                        if !InspectWindow::check_if_already_exists(
                                                            &self.inspect_windows,
                                                            &new_inspect,
                                                        ) && let Some(x) = InspectWindow::new(new_inspect) {
                                                            self.inspect_windows.push(x);
                                                        }
                                                    }
                                                    _ => {
                                                        // TODO: inspect other types
                                                    }
                                                }
                                            }
                                            _ => {
                                                // TODO: Inspect other types
                                            }
                                        }
                                    }
                                }
                                for (node_id, position) in node_position_updates {
                                    graph_layout.move_node(node_id, position, Vec2::ZERO)
                                }

                                // Draw lines
                                self.tensors_in_view.clear();
                                let link_data = graph_layout.get_link_data();
                                for ((src_id, src_id_i), (dst_id, dst_id_i), link_id, (paint_idx_a, paint_idx_b, paint_idx_c)) in
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
                                            let this_selectable = link_data.link_type.get_graph_selectable();
                                            let is_selected = self.explorer_selection == Some(this_selectable.clone());
                                            let is_hovered = self.explorer_hovered == Some(this_selectable);
                                            (is_selected, is_hovered)
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
                                    ui.painter().set(paint_idx_a, shape);

                                    // Render swatch
                                    if let Some(graph_path) = graph_path &&
                                        let Some(link_data) = link_data.get(&link_id) {

                                        let path = match link_data.link_type {
                                            GraphLayoutLinkType::SymbolicGraphTensor(x) => {
                                                Some(graph_path.push_symbolic_tensor(x))
                                            }
                                            GraphLayoutLinkType::SuperGraphLink(x) => {
                                                match x {
                                                    SuperGraphAnyLink::Tensor(x) => {
                                                        Some(graph_path.push_super_tensor(x))
                                                    }
                                                    _ => None
                                                }
                                            }
                                            GraphLayoutLinkType::MilliOpGraphTensor(x) => {
                                                Some(graph_path.push_milli_op_tensor(x))
                                            }
                                        };
                                        if let Some(path) = &path {
                                            self.tensors_in_view.insert(path.clone());
                                        }
                                        if let Some(path) = path && let Some(texture) = Self::get_tensor_swatch(
                                            &mut self.rendered_tensor_swatches,
                                            &self.abbreviated_tensor_reports,
                                            state, ui.ctx(), &path) {
                                            let midpoint = points[1].lerp(points[2], 0.5);
                                            let rect = Rect::from_center_size(midpoint, Vec2::splat(32.0));
                                            let mut shape = Mesh::with_texture(texture.id());
                                            shape.add_rect_with_uv(rect, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);
                                            ui.painter().set(paint_idx_c, shape);
                                            let color = ui.visuals().widgets.inactive.bg_fill;
                                            let shape = egui::Shape::rect_filled(rect.expand(4.0), 4.0, color);
                                            ui.painter().set(paint_idx_b, shape);
                                        }
                                    }


                                }
                            });
                            self.model_view_scene_rects
                                .insert(self.graph_subject_path.clone(), scene_rect);
                        });
                    });
                }
                Some(Err(error)) => {
                    ui.label(format!("Error generating graph: {error:?}"));
                }
                None => {
                    ui.label("No graph generated");
                }
            }
        } else {
            ui.label("No graph selected");
        }

        if let Some(interface) = selected_interface {
            let interface_id = selected_interface_id.unwrap();
            let frame = egui::Frame::default().stroke(ui.visuals().window_stroke);
            frame.show(ui, |ui| {
                match &interface.interface {
                    AnyInterface::TextInferenceTokensInLogitOutInterface(llm_interface) => {
                        ui.horizontal_top(|ui| {
                            let tokenizer_info = llm_interface.get_tokenizer();
                            if let Some(tokenizer) = loaded_tokenizers
                                .loaded_tokenizers
                                .get(tokenizer_info)
                                .cloned()
                                .flatten()
                            {
                                match tokenizer {
                                    Ok(tokenizer) => {
                                        let mut text_inference_data = {
                                            if let Some(text_inference_data) =
                                                self.text_inference_data.get_mut(&interface_id)
                                            {
                                                text_inference_data
                                            } else {
                                                let mut inference_data =
                                                    TextInferenceData::default();
                                                inference_data.tokens =
                                                    tokenizer.encode("Hello World!");
                                                self.text_inference_data
                                                    .insert(interface_id, inference_data);
                                                self.text_inference_data
                                                    .get_mut(&interface_id)
                                                    .unwrap()
                                            }
                                        };
                                        if let Some((request_id, _, _)) =
                                            &text_inference_data.pending_request
                                        {
                                            if let Some(reports) = server_request_manager.get_reports(*request_id) {
                                                let time_now = Instant::now();
                                                for report in reports {
                                                    for (path, value) in report.tensor_assignments {
                                                        self.inspect_window_tensor_subscription_returns.insert(path, value);
                                                    }
                                                    for (node_path, age, execution_duration) in report.node_executions {
                                                        let time = time_now - age;
                                                        self.node_execution_timestamps.insert(node_path.clone(), time);
                                                        self.node_execution_durations.insert(node_path, execution_duration);
                                                    }
                                                    for (tensor_path, value) in report.abbreviated_tensor_assignments {
                                                        self.rendered_tensor_swatches.remove(&tensor_path);
                                                        self.abbreviated_tensor_reports.insert(tensor_path, value);
                                                    }
                                                }
                                            }
                                            if let Some(mut response) =
                                                server_request_manager.get_response(*request_id)
                                            {
                                                let (_, link, tokens) = text_inference_data
                                                    .pending_request
                                                    .take()
                                                    .unwrap();
                                                text_inference_data.pending_request = None;
                                                match response.result {
                                                    Ok(mut data) => {
                                                        let mut response_tokens =
                                                            data.tensor_outputs.remove(&link).unwrap();
                                                        let shape = response_tokens.shape();
                                                        let logits_per_token = shape[1];
                                                        let returned_tokens = shape[0];
                                                        for i in 0..returned_tokens as usize {
                                                            let sliced_output_tensor = response_tokens
                                                                .slice(&[
                                                                    i..i + 1,
                                                                    0..logits_per_token as usize,
                                                                ])
                                                                .unwrap();
                                                            let output = sliced_output_tensor.flatten();
                                                            let output_vec: Vec<f32> =
                                                                output.try_into().unwrap();
                                                            let mut idx_and_val = output_vec
                                                                .iter()
                                                                .enumerate()
                                                                .map(|(a, b)| (a as u32, *b))
                                                                .collect::<Vec<_>>();
                                                            idx_and_val.sort_by(|(_, a), (_, b)| {
                                                                if a < b {
                                                                    Ordering::Greater
                                                                } else {
                                                                    Ordering::Less
                                                                }
                                                            });
                                                            let clipped_logits = idx_and_val
                                                                [0..idx_and_val.len().min(100)]
                                                                .to_vec();
                                                            let context_end = tokens.len() - returned_tokens as usize + i + 1;
                                                            let context = tokens[0..context_end].to_vec();
                                                            text_inference_data
                                                                .logits
                                                                .insert(context, clipped_logits);
                                                        }
                                                    }
                                                    Err(err) => {
                                                        self.error_popup = Some(err);
                                                    }
                                                }
                                            }
                                        }
                                        let frame = egui::Frame::default()
                                            .stroke(ui.visuals().window_stroke)
                                            .inner_margin(5.0);
                                        frame.show(ui, |ui| {
                                            ui.vertical(|ui| {
                                                ui.heading("Text Inference");
                                                {
                                                    let v = match &tokenizer_info {
                                                        TokenizerInfo::HFTokenizer(x) => {
                                                            format!("Huggingface: {x}")
                                                        }
                                                        TokenizerInfo::RWKVWorld => {
                                                            "RWKV World".to_string()
                                                        }
                                                    };
                                                    ui.label(format!("Tokenizer: {v}"));
                                                }
                                                if text_inference_data.pending_request.is_some() {
                                                    ui.spinner();
                                                } else {
                                                    ui.horizontal(|ui| {
                                                        toggle_ui(ui, &mut text_inference_data.use_cache);
                                                        ui.label("Cache");
                                                    });
                                                    egui::ComboBox::from_id_salt(121151)
                                                        .selected_text(text_inference_data.selected_mode.to_string())
                                                        .show_ui(ui, |ui| {
                                                            ui.selectable_value(
                                                                &mut text_inference_data.selected_mode,
                                                                SuperGraphRequestBackendMode::NDArray,
                                                                SuperGraphRequestBackendMode::NDArray.to_string());
                                                            if server_config_report.vulkan_available {
                                                                ui.selectable_value(
                                                                    &mut text_inference_data.selected_mode,
                                                                    SuperGraphRequestBackendMode::Vulkan,
                                                                    SuperGraphRequestBackendMode::Vulkan.to_string());
                                                            }
                                                    });
                                                    if ui.button("Run").clicked() {
                                                        let tokens =
                                                            text_inference_data.tokens.clone();
                                                        let tokens_tensor =
                                                            NDArrayNumericTensor::from_vec(
                                                                tokens.clone(),
                                                            )
                                                                .to_dyn();
                                                        let mut swatch_settings = if state.do_explorer_swatches_in_view || state.do_all_explorer_swatches {
                                                            Some(AbbreviatedTensorReportSettings{
                                                                downsampled_size: (state.swatch_dimension*state.swatch_dimension) as u64,
                                                                subscribed_tensors: self.tensors_in_view.iter().cloned().collect(),
                                                                do_all: state.do_all_explorer_swatches,
                                                            })
                                                        } else {
                                                            None
                                                        };
                                                        let token = server_request_manager
                                                            .submit_supergraph_request(
                                                                SuperGraphRequest {
                                                                    abbreviated_tensor_report_settings: swatch_settings,
                                                                    do_node_execution_reports: state.explorer_node_wave,
                                                                    attention_token: None,
                                                                    super_graph: llm_interface
                                                                        .super_graph
                                                                        .clone(),
                                                                    string_inputs: HashMap::new(),
                                                                    subscribed_tensors: self.inspect_window_tensor_subscriptions.iter().cloned().collect(),
                                                                    tensor_inputs: HashMap::from([(
                                                                        llm_interface
                                                                            .token_context_input_link
                                                                            .clone(),
                                                                        tokens_tensor,
                                                                    )]),
                                                                    model_inputs: HashMap::from([(
                                                                        llm_interface
                                                                            .model_input_link
                                                                            .clone(),
                                                                        interface
                                                                            .model_ids
                                                                            .first()
                                                                            .unwrap()
                                                                            .clone(),
                                                                    )]),
                                                                    hash_inputs: HashMap::from([(
                                                                        llm_interface
                                                                            .cache_key_input_link
                                                                            .clone(),
                                                                        12u64,
                                                                    )]),
                                                                    use_cache: if text_inference_data.use_cache {Some(100 + interface_id as u64)} else {None},
                                                                    backend_mode: text_inference_data.selected_mode
                                                                },
                                                            );
                                                        text_inference_data.pending_request =
                                                            Some((
                                                                token,
                                                                llm_interface
                                                                    .logit_output_link
                                                                    .clone(),
                                                                tokens,
                                                            ));
                                                    }
                                                }
                                            });
                                        });
                                        let logits = {
                                            let mut logits = vec![];
                                            let mut context = vec![];
                                            for token in &text_inference_data.tokens {
                                                context.push(token.clone());
                                                if let Some(x) =
                                                    text_inference_data.logits.get(&context)
                                                {
                                                    logits.push(x.as_slice());
                                                } else {
                                                    break;
                                                }
                                            }
                                            logits
                                        };
                                        TokenizedRichText::new().ui(
                                            ui,
                                            tokenizer.as_ref(),
                                            &mut text_inference_data.tokens,
                                            Some(&logits),
                                        );
                                    }
                                    Err(e) => {
                                        ui.label(format!("Error loading tokenizer: {}", e));
                                    }
                                }
                            } else {
                                ui.label("Tokenizer not loaded");
                            }
                        });
                    }
                    _ => {
                        ui.label("Unsupported Interface");
                    }
                }
                ui.allocate_exact_size(ui.available_size_before_wrap(), Sense::click());
            });
        }
    }

    pub(crate) fn get_model_scope(
        &mut self,
        loaded_models: &LoadedModels,
    ) -> Option<LoadedModelId> {
        let (_graph_subject, model_scope) = {
            let mut graph_subject = None;
            let mut implied_model = None;
            let mut model_scope = None;
            for a in &self.graph_subject_path {
                match a {
                    GraphExplorerLayerSelection::Model(model_id) => {
                        if let Some(x) = self.loaded_models.get(model_id) {
                            graph_subject = Some(GraphSubject::SymbolicGraphInner(&x.inner_graph));
                            model_scope = Some(*model_id);
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
                                        model_scope = Some(*model_id);
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
        model_scope
    }

    pub(crate) fn get_super_graph_graph_path(
        &self,
        loaded_models: &LoadedModels,
    ) -> Option<SuperGraphGraphPath> {
        let mut path: Option<SuperGraphGraphPath> = None;
        let mut subject = None;
        let mut implied_model = None;
        for a in &self.graph_subject_path {
            (path, subject) = match (a, &path, &subject) {
                (GraphExplorerLayerSelection::Model(_model_id), _, _) => {
                    // No super graph if the root selection is a model
                    (None, None)
                }
                (GraphExplorerLayerSelection::Interface(interface_id), _, _) => {
                    if let Some(x) = loaded_models.current_interfaces.get(interface_id) {
                        implied_model = x.model_ids.first();
                        (
                            Some(SuperGraphGraphPath::SuperGraph(vec![])),
                            Some(GraphSubject::SuperGraphInner(
                                &x.interface.get_super_graph().inner,
                            )),
                        )
                    } else {
                        (None, None)
                    }
                }
                (
                    GraphExplorerLayerSelection::SymbolicGraphOperationId((_a, _b)),
                    Some(_path),
                    _,
                ) => {
                    todo!();
                    // Don't support symbolic graph nesting yet
                }
                (
                    GraphExplorerLayerSelection::SuperGraphNodeId(node_id),
                    Some(path),
                    Some(GraphSubject::SuperGraphInner(subject)),
                ) => {
                    if let Some(x) = subject.nodes.get(node_id) {
                        match x {
                            SuperGraphAnyNode::MilliOpGraph(x) => (
                                Some(path.push_super_milli_op_node_graph(*node_id)),
                                Some(GraphSubject::MilliOpGraphB(&x.graph)),
                            ),
                            SuperGraphAnyNode::ModelExecution(_x) => {
                                if let Some(model_id) = implied_model
                                    && let Some(symbolic_graph) = self.loaded_models.get(model_id)
                                {
                                    (
                                        Some(path.push_model_execution_node_graph(*node_id)),
                                        Some(GraphSubject::SymbolicGraphInner(
                                            &symbolic_graph.inner_graph,
                                        )),
                                    )
                                } else {
                                    (None, None)
                                }
                            }
                            _ => {
                                if let Some(x) = x.get_sub_graph() {
                                    (
                                        Some(path.push_super_node_graph(*node_id)),
                                        Some(GraphSubject::SuperGraphInner(x)),
                                    )
                                } else {
                                    (None, None)
                                }
                            }
                        }
                    } else {
                        (None, None)
                    }
                }
                _ => (None, None),
            };
        }
        path
    }
}
