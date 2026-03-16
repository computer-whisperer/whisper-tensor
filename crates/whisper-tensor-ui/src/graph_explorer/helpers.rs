use super::graph_layout::{
    GraphLayoutIOOffsets, GraphLayoutLinkData, GraphLayoutLinkId, GraphLayoutNodeType,
};
use super::{GraphExplorerApp, GraphRootSubjectSelection};
use crate::app::{InterfaceId, LoadedModels};
use crate::widgets::progress_report::SuperGraphProgressWidgetState;
use egui::{Color32, Label, Margin, Response, RichText, Sense, Stroke, Ui};
use std::any::Any;
use std::collections::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};
#[cfg(target_arch = "wasm32")]
use web_time::{Duration, Instant};
use whisper_tensor::graph::{GlobalId, Graph, GraphDyn};
use whisper_tensor::scalar_info::ScalarInfoTyped;
use whisper_tensor::super_graph::SuperGraph;
use whisper_tensor::super_graph::nodes::SuperGraphAnyNode;
use whisper_tensor_server::LoadedModelId;

#[allow(clippy::too_many_arguments)]
pub(super) fn render_node_contents(
    ui: &mut egui::Ui,
    node_type: &GraphLayoutNodeType,
    num_inputs: usize,
    num_outputs: usize,
    graph_subject: &dyn GraphDyn,
    is_selected: bool,
    is_hovered: bool,
    time_since_exec: Option<Duration>,
) -> (Response, GraphLayoutIOOffsets) {
    // Decide corner radius
    let corner_radius = match node_type {
        GraphLayoutNodeType::GraphNode(_) => 3.0,
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
            GraphLayoutNodeType::GraphNode(node_id) => {
                if let Some(node) = &graph_subject.get_node_by_id(node_id) {
                    let op_kind = node.op_kind();
                    if let Some(label) = node.label()
                        && label != op_kind
                    {
                        ui.add(Label::new(label).selectable(false));
                        ui.add(Label::new(RichText::new(op_kind).size(9.0)).selectable(false));
                    } else {
                        ui.add(Label::new(op_kind).selectable(false));
                    }
                }
            }
            GraphLayoutNodeType::InputLinkNode(link_id) => {
                let text = if let Some(link) = graph_subject.get_link_by_id(link_id)
                    && let Some(label) = link.label()
                {
                    label.to_string()
                } else {
                    "Input".to_string()
                };
                ui.add(Label::new(text).selectable(false));
            }
            GraphLayoutNodeType::OutputLinkNode(link_id) => {
                let text = if let Some(link) = graph_subject.get_link_by_id(link_id)
                    && let Some(label) = link.label()
                {
                    label.to_string()
                } else {
                    "Output".to_string()
                };
                ui.add(Label::new(text).selectable(false));
            }
            GraphLayoutNodeType::ConstantLinkNode(link_id) => {
                let text = if let Some(link) = graph_subject.get_link_by_id(link_id)
                    && let Some(label) = link.label()
                {
                    label.to_string()
                } else {
                    "Constant".to_string()
                };
                ui.add(Label::new(text).selectable(false));
            }
            GraphLayoutNodeType::ConnectionByNameSrc(link_id) => {
                let text = if let Some(link) = graph_subject.get_link_by_id(link_id)
                    && let Some(label) = link.label()
                {
                    format!("{} >", label)
                } else {
                    format!("{} >", link_id)
                };
                ui.add(Label::new(text).selectable(false));
            }
            GraphLayoutNodeType::ConnectionByNameDest(link_id) => {
                let text = if let Some(link) = graph_subject.get_link_by_id(link_id)
                    && let Some(label) = link.label()
                {
                    format!("> {}", label)
                } else {
                    format!("> {}", link_id)
                };
                ui.add(Label::new(text).selectable(false));
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

pub(crate) fn format_shape(val: &[ScalarInfoTyped<u64>]) -> String {
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

pub(super) fn ensure_layout_link_id(
    tensor_id: GlobalId,
    tensor_link_ids: &mut HashMap<GlobalId, GraphLayoutLinkId>,
    link_data: &mut HashMap<GraphLayoutLinkId, GraphLayoutLinkData>,
    next_link_id: &mut usize,
) -> GraphLayoutLinkId {
    if let Some(link_id) = tensor_link_ids.get(&tensor_id).copied() {
        return link_id;
    }
    let link_id = GraphLayoutLinkId(*next_link_id);
    *next_link_id += 1;
    tensor_link_ids.insert(tensor_id, link_id);
    link_data.insert(
        link_id,
        GraphLayoutLinkData {
            global_id: tensor_id,
        },
    );
    link_id
}

pub(super) struct NodeExecutionActivityMapsMut<'a> {
    pub(super) node_execution_timestamps: &'a mut HashMap<Vec<GlobalId>, Instant>,
    pub(super) node_last_child_active_timestamps: &'a mut HashMap<Vec<GlobalId>, Instant>,
    pub(super) node_execution_durations: &'a mut HashMap<Vec<GlobalId>, Duration>,
    pub(super) node_execution_op_kinds: &'a mut HashMap<Vec<GlobalId>, String>,
}

pub(super) fn record_node_execution_activity(
    maps: &mut NodeExecutionActivityMapsMut<'_>,
    node_path: Vec<GlobalId>,
    op_kind: String,
    execution_time: Instant,
    execution_duration: Duration,
) {
    maps.node_execution_timestamps
        .insert(node_path.clone(), execution_time);
    maps.node_execution_durations
        .insert(node_path.clone(), execution_duration);
    if !op_kind.is_empty() {
        maps.node_execution_op_kinds
            .insert(node_path.clone(), op_kind);
    }

    if node_path.len() > 1 {
        let mut ancestor_path = Vec::with_capacity(node_path.len() - 1);
        for ancestor_id in node_path.iter().take(node_path.len() - 1) {
            ancestor_path.push(*ancestor_id);
            maps.node_last_child_active_timestamps
                .insert(ancestor_path.clone(), execution_time);
        }
    }
}

pub(super) fn duration_since_node_activity(
    node_execution_timestamps: &HashMap<Vec<GlobalId>, Instant>,
    node_last_child_active_timestamps: &HashMap<Vec<GlobalId>, Instant>,
    node_path: &[GlobalId],
    current_time: Instant,
) -> Option<Duration> {
    let own = node_execution_timestamps
        .get(node_path)
        .map(|x| current_time.saturating_duration_since(*x));
    let child = node_last_child_active_timestamps
        .get(node_path)
        .map(|x| current_time.saturating_duration_since(*x));
    match (own, child) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

pub(crate) enum LoadableGraphState<'a> {
    None,
    Unloaded(LoadedModelId),
    Loaded(&'a dyn GraphDyn),
}

pub(crate) fn get_inner_graph<'a>(
    graph: &'a dyn GraphDyn,
    node_id: GlobalId,
    root_selection: GraphRootSubjectSelection,
    loaded_models: &'a LoadedModels,
) -> LoadableGraphState<'a> {
    if let Some(super_graph) = <dyn Any>::downcast_ref::<SuperGraph>(graph.as_any())
        && let Some(node) = Graph::get_node_by_id(super_graph, &node_id)
    {
        match node {
            SuperGraphAnyNode::ModelExecution(model_execution) => {
                let local_model_id = model_execution.symbolic_graph_id;
                if let GraphRootSubjectSelection::ServerInterface(interface_id) = root_selection
                    && let Some(interface) = loaded_models
                        .server_graphs
                        .current_interfaces
                        .get(&interface_id)
                    && let Some(model_id) = interface.model_ids.get(local_model_id)
                {
                    if let Some(model) = loaded_models.server_graphs.symbolic_graphs.get(model_id) {
                        return LoadableGraphState::Loaded(model);
                    } else {
                        return LoadableGraphState::Unloaded(*model_id);
                    }
                }
            }
            SuperGraphAnyNode::MilliOpGraph(milli_op_graph) => {
                return LoadableGraphState::Loaded(&milli_op_graph.graph);
            }
            _ => {
                if let Some(x) = node.get_sub_graph() {
                    return LoadableGraphState::Loaded(x);
                }
            }
        }
    }
    LoadableGraphState::None
}

impl GraphExplorerApp {
    pub(super) fn resolve_op_kind(
        &self,
        path: &[GlobalId],
        root_graph: &dyn GraphDyn,
        loaded_models: &LoadedModels,
    ) -> String {
        // First check cached op_kind from observer
        if let Some(op_kind) = self.node_execution_op_kinds.get(path)
            && !op_kind.is_empty()
        {
            return op_kind.clone();
        }
        // Resolve through graph hierarchy like inspect windows do
        if let Some((graph, node_id)) = self.resolve_path_to_graph(path, root_graph, loaded_models)
        {
            let graph: &dyn GraphDyn = graph;
            if let Some(node) = graph.get_node_by_id(&node_id) {
                return node.op_kind();
            }
        }
        "?".to_string()
    }

    pub(super) fn breadcrumb_node_text(graph: &dyn GraphDyn, node_id: GlobalId) -> String {
        if let Some(node) = graph.get_node_by_id(&node_id) {
            if let Some(label) = node.label()
                && !label.is_empty()
            {
                return label;
            }
            let op_kind = node.op_kind();
            if !op_kind.is_empty() {
                return op_kind;
            }
        }
        node_id.to_string()
    }

    pub(super) fn build_graph_breadcrumb_items(
        &self,
        root_graph: &dyn GraphDyn,
        working_path: &[GlobalId],
        loaded_models: &LoadedModels,
    ) -> Vec<(String, Vec<GlobalId>)> {
        let mut items = vec![("Root".to_string(), Vec::<GlobalId>::new())];
        let mut current_graph = root_graph;
        let mut current_path = Vec::<GlobalId>::new();
        for node_id in working_path {
            current_path.push(*node_id);
            items.push((
                Self::breadcrumb_node_text(current_graph, *node_id),
                current_path.clone(),
            ));
            match get_inner_graph(current_graph, *node_id, self.root_selection, loaded_models) {
                LoadableGraphState::Loaded(next_graph) => {
                    current_graph = next_graph;
                }
                LoadableGraphState::Unloaded(_) | LoadableGraphState::None => {
                    break;
                }
            }
        }
        items
    }

    pub(super) fn interface_progress_widget_state(
        &self,
        interface_id: InterfaceId,
    ) -> Option<&SuperGraphProgressWidgetState> {
        self.text_inference_data
            .get(&interface_id)
            .map(|x| &x.progress_widget_state)
            .or_else(|| {
                self.sd_inference_data
                    .get(&interface_id)
                    .map(|x| &x.progress_widget_state)
            })
            .or_else(|| {
                self.tts_inference_data
                    .get(&interface_id)
                    .map(|x| &x.progress_widget_state)
            })
            .or_else(|| {
                self.stt_inference_data
                    .get(&interface_id)
                    .map(|x| &x.progress_widget_state)
            })
    }

    pub(super) fn render_profiling_window(
        &self,
        ui: &mut Ui,
        root_graph: &dyn GraphDyn,
        loaded_models: &LoadedModels,
    ) {
        if self.node_execution_durations.is_empty() {
            ui.label("No profiling data yet. Run a graph to collect timing.");
            return;
        }

        let prefix = &self.graph_subject_path;

        // Show graph-level total if available
        if let Some(graph_dur) = self.node_execution_durations.get(prefix) {
            ui.strong(format!("Total: {:.1}ms", graph_dur.as_secs_f64() * 1000.0));
        }

        // Filter to descendants of the current graph, excluding the graph node itself
        let mut nodes_by_duration: Vec<_> = self
            .node_execution_durations
            .iter()
            .filter(|(path, _)| path.starts_with(prefix) && path.len() > prefix.len())
            .map(|(path, dur)| {
                let op_kind = self.resolve_op_kind(path, root_graph, loaded_models);
                (path, op_kind, *dur)
            })
            .collect();
        nodes_by_duration.sort_by(|a, b| b.2.cmp(&a.2));

        // Accumulated by op type
        let mut by_op_type: HashMap<String, (Duration, usize)> = HashMap::new();
        for (_, op_kind, dur) in &nodes_by_duration {
            let entry = by_op_type.entry(op_kind.clone()).or_default();
            entry.0 += *dur;
            entry.1 += 1;
        }
        let mut op_type_sorted: Vec<_> = by_op_type.into_iter().collect();
        op_type_sorted.sort_by(|a, b| b.1.0.cmp(&a.1.0));

        ui.label(format!("{} nodes", nodes_by_duration.len()));
        ui.separator();

        egui::CollapsingHeader::new("By Op Type")
            .default_open(true)
            .show(ui, |ui| {
                egui::Grid::new("profiling_op_type_grid")
                    .num_columns(4)
                    .striped(true)
                    .show(ui, |ui| {
                        ui.strong("Op");
                        ui.strong("Total");
                        ui.strong("Count");
                        ui.strong("Avg");
                        ui.end_row();
                        for (op, (total, count)) in op_type_sorted.iter().take(15) {
                            ui.label(op);
                            ui.label(format!("{:.2}ms", total.as_secs_f64() * 1000.0));
                            ui.label(format!("{}", count));
                            ui.label(format!(
                                "{:.2}ms",
                                total.as_secs_f64() * 1000.0 / *count as f64
                            ));
                            ui.end_row();
                        }
                    });
            });

        ui.separator();

        egui::CollapsingHeader::new("Top Individual Nodes")
            .default_open(true)
            .show(ui, |ui| {
                egui::Grid::new("profiling_individual_grid")
                    .num_columns(3)
                    .striped(true)
                    .show(ui, |ui| {
                        ui.strong("Op");
                        ui.strong("Duration");
                        ui.strong("Path");
                        ui.end_row();
                        for (path, op_kind, dur) in nodes_by_duration.iter().take(15) {
                            ui.label(op_kind.as_str());
                            ui.label(format!("{:.2}ms", dur.as_secs_f64() * 1000.0));
                            let path_str = path
                                .iter()
                                .map(|id| id.to_string())
                                .collect::<Vec<_>>()
                                .join("/");
                            ui.label(path_str);
                            ui.end_row();
                        }
                    });
            });
    }
}
