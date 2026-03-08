mod graph_layout;
pub mod inspect_windows;
mod tensor_swatch;

use crate::app::{InterfaceId, LoadedModels, LoadedTokenizers};
use crate::graph_explorer::inspect_windows::{
    AnyInspectWindow, InspectWindowGraphLink, InspectWindowGraphNode,
};
use crate::sd_explorer::{clip_tokenize, generate_normal_noise, tensor_to_egui_texture};
use crate::websockets::ServerRequestManager;
use crate::widgets::toggle::toggle_ui;
use crate::widgets::tokenized_rich_text::TokenizedRichText;
use egui::epaint::CubicBezierShape;
use egui::{
    Color32, ColorImage, Context, Label, Margin, Mesh, Pos2, Rect, Response, Sense, Shape, Stroke,
    StrokeKind, TextureHandle, Ui, UiBuilder, Vec2, vec2,
};
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::wasm_bindgen;

#[wasm_bindgen(inline_js = "
export function js_copy_image_to_clipboard(rgba_data, width, height) {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    const img_data = new ImageData(new Uint8ClampedArray(rgba_data), width, height);
    ctx.putImageData(img_data, 0, 0);
    canvas.toBlob(function(blob) {
        navigator.clipboard.write([new ClipboardItem({'image/png': blob})]);
    }, 'image/png');
}
")]
extern "C" {
    fn js_copy_image_to_clipboard(rgba_data: &[u8], width: u32, height: u32);
}
use graph_layout::{
    GraphLayout, GraphLayoutError, GraphLayoutIOOffsets, GraphLayoutLinkData, GraphLayoutLinkId,
    GraphLayoutNodeId, GraphLayoutNodeInitData, GraphLayoutNodeType,
};
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tensor_swatch::build_tensor_swatch;
use web_time::{Duration, Instant};
use whisper_tensor::DynRank;
use whisper_tensor::backends::ndarray_backend::NDArrayNumericTensor;
use whisper_tensor::graph::{GlobalId, Graph, GraphDyn};
use whisper_tensor::interfaces::{AnyInterface, ImageGenerationInterface};
use whisper_tensor::metadata::TokenizerInfo;
use whisper_tensor::scalar_info::ScalarInfoTyped;
use whisper_tensor::super_graph::nodes::SuperGraphAnyNode;
use whisper_tensor::super_graph::{SuperGraph, SuperGraphLinkTensor};
use whisper_tensor::tokenizer::Tokenizer;
use whisper_tensor_server::{
    AbbreviatedTensorReportSettings, AbbreviatedTensorValue, LoadedModelId, ServerConfigReport,
    SuperGraphRequest, SuperGraphRequestBackendMode, WebsocketClientServerMessage,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct GraphExplorerSettings {
    pub(crate) explorer_physics: bool,
    pub(crate) explorer_minimap: bool,
    pub(crate) do_all_explorer_swatches: bool,
    pub(crate) do_explorer_swatches_in_view: bool,
    pub(crate) explorer_node_wave: bool,
    swatch_dimension: usize,
}

impl Default for GraphExplorerSettings {
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum GraphRootSubjectSelection {
    Model(LoadedModelId),
    Interface(InterfaceId),
}

#[derive(Clone, Debug, Default)]
pub(crate) struct TextInferenceData {
    tokens: Vec<u32>,
    logits: HashMap<Vec<u32>, Vec<(u32, f32)>>,
    pending_request: Option<(u64, SuperGraphLinkTensor, Vec<u32>)>,
    use_cache: bool,
    selected_mode: SuperGraphRequestBackendMode,
}

#[derive(Clone)]
pub(crate) struct SDInferenceData {
    prompt: String,
    num_steps: usize,
    guidance_scale: f32,
    latent_h: usize,
    latent_w: usize,
    seed: u64,
    use_cache: bool,
    selected_mode: SuperGraphRequestBackendMode,
    pending_request: Option<(u64, SuperGraphLinkTensor)>,
    generated_image: Option<(TextureHandle, ColorImage)>,
    status_message: Option<String>,
    show_image_window: bool,
}

impl Default for SDInferenceData {
    fn default() -> Self {
        Self {
            prompt: "a photo of a cat".to_string(),
            num_steps: 20,
            guidance_scale: 7.5,
            latent_h: 8,
            latent_w: 8,
            seed: 42,
            use_cache: false,
            selected_mode: SuperGraphRequestBackendMode::NDArray,
            pending_request: None,
            generated_image: None,
            status_message: None,
            show_image_window: false,
        }
    }
}

pub(crate) struct GraphExplorerApp {
    pub(crate) root_selection: GraphRootSubjectSelection,
    pub(crate) explorer_selection: Option<GlobalId>,
    pub(crate) explorer_hovered: Option<GlobalId>,
    pub(crate) next_explorer_hovered: Option<GlobalId>,
    inspect_window_tensor_subscriptions: HashSet<Vec<GlobalId>>,
    inspect_window_tensor_subscription_returns:
        HashMap<Vec<GlobalId>, NDArrayNumericTensor<DynRank>>,
    pub inspect_windows: Vec<AnyInspectWindow>,
    graph_layouts: HashMap<Vec<GlobalId>, Result<GraphLayout, GraphLayoutError>>,
    model_view_scene_rects: HashMap<Vec<GlobalId>, Rect>,
    pub(crate) graph_subject_path: Vec<GlobalId>,
    pub(crate) next_graph_subject_path: Option<Vec<GlobalId>>,
    pub(crate) text_inference_data: HashMap<InterfaceId, TextInferenceData>,
    pub(crate) sd_inference_data: HashMap<InterfaceId, SDInferenceData>,
    node_execution_timestamps: HashMap<Vec<GlobalId>, Instant>,
    node_execution_durations: HashMap<Vec<GlobalId>, Duration>,
    node_execution_op_kinds: HashMap<Vec<GlobalId>, String>,
    abbreviated_tensor_reports: HashMap<Vec<GlobalId>, AbbreviatedTensorValue>,
    rendered_tensor_swatches: HashMap<Vec<GlobalId>, TextureHandle>,
    tensors_in_view: HashSet<Vec<GlobalId>>,
    nodes_in_view: HashSet<Vec<GlobalId>>,
    error_popup: Option<String>,
    pub(crate) show_profiling_window: bool,
}

#[allow(clippy::too_many_arguments)]
fn render_node_contents(
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
                    ui.add(Label::new(node.op_kind()).selectable(false));
                    if let Some(label) = node.label() {
                        ui.add(Label::new(label).selectable(false));
                    }
                }
            }
            GraphLayoutNodeType::InputLinkNode(link_id) => {
                let text = if let Some(link) = graph_subject.get_link_by_id(link_id)
                    && let Some(label) = link.label()
                {
                    format!("Input: {}", label)
                } else {
                    "Input".to_string()
                };
                ui.add(Label::new(text).selectable(false));
            }
            GraphLayoutNodeType::OutputLinkNode(link_id) => {
                let text = if let Some(link) = graph_subject.get_link_by_id(link_id)
                    && let Some(label) = link.label()
                {
                    format!("Output: {}", label)
                } else {
                    "Output".to_string()
                };
                ui.add(Label::new(text).selectable(false));
            }
            GraphLayoutNodeType::ConstantLinkNode(link_id) => {
                let text = if let Some(link) = graph_subject.get_link_by_id(link_id)
                    && let Some(label) = link.label()
                {
                    format!("Constant: {}", label)
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

fn format_shape(val: &[ScalarInfoTyped<u64>]) -> String {
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
                if let GraphRootSubjectSelection::Interface(interface_id) = root_selection
                    && let Some(interface) = loaded_models.current_interfaces.get(&interface_id)
                    && let Some(model_id) = interface.model_ids.get(local_model_id)
                {
                    if let Some(model) = loaded_models.loaded_models.get(model_id) {
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
    pub(crate) fn new(subject: GraphRootSubjectSelection) -> Self {
        Self {
            root_selection: subject,
            explorer_selection: None,
            explorer_hovered: None,
            next_explorer_hovered: None,
            inspect_window_tensor_subscription_returns: HashMap::new(),
            inspect_window_tensor_subscriptions: HashSet::new(),
            inspect_windows: Vec::new(),
            graph_layouts: HashMap::new(),
            model_view_scene_rects: HashMap::new(),
            graph_subject_path: Vec::new(),
            next_graph_subject_path: None,
            text_inference_data: HashMap::new(),
            sd_inference_data: HashMap::new(),
            node_execution_timestamps: HashMap::new(),
            node_execution_durations: HashMap::new(),
            node_execution_op_kinds: HashMap::new(),
            abbreviated_tensor_reports: HashMap::new(),
            rendered_tensor_swatches: HashMap::new(),
            nodes_in_view: HashSet::new(),
            tensors_in_view: HashSet::new(),
            error_popup: None,
            show_profiling_window: false,
        }
    }

    fn resolve_op_kind(
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

    fn render_profiling_window(
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

    pub(crate) fn get_tensor_swatch(
        rendered_tensor_swatches: &mut HashMap<Vec<GlobalId>, TextureHandle>,
        abbreviated_tensor_reports: &HashMap<Vec<GlobalId>, AbbreviatedTensorValue>,
        state: &mut GraphExplorerSettings,
        ctx: &Context,
        path: &[GlobalId],
    ) -> Option<TextureHandle> {
        if let Some(swatch) = rendered_tensor_swatches.get(path) {
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
                rendered_tensor_swatches.insert(path.to_vec(), texture.clone());
                Some(texture)
            } else {
                None
            }
        } else {
            None
        }
    }

    pub(crate) fn render_minimap(&mut self, ui: &mut egui::Ui) {
        let current_time = Instant::now();

        let mut do_request_redraw = false;

        let max_layer_i = self.graph_subject_path.len() + 1;
        for layer_i in 0..max_layer_i {
            let selected_global_id = self.graph_subject_path.get(layer_i).cloned();
            let current_path = self.graph_subject_path[0..layer_i].to_vec();
            let maps_left = max_layer_i - layer_i;
            let subject_width = if maps_left <= 1 {
                ui.available_size_before_wrap().x
            } else {
                (ui.available_size_before_wrap().x / (maps_left as f32)) - 5.0
            };
            if let Some(Ok(graph_layout)) = self.graph_layouts.get(&current_path) {
                let node_bounding_rect = graph_layout.get_bounding_rect().expand(15.0);

                let height = 80.0;
                let width = (node_bounding_rect.width() * (height / node_bounding_rect.height()))
                    .min(subject_width)
                    .max(30.0);

                // Get frame min and max
                let minimap_frame = egui::Frame::default().stroke(ui.visuals().window_stroke);
                minimap_frame.show(ui, |ui| {
                    let shape_request = vec2(width, 80.0);
                    let (outer_rect, outer_response) =
                        ui.allocate_exact_size(shape_request, Sense::drag());
                    let transform = outer_rect.size() / node_bounding_rect.size();
                    if outer_response.dragged() {
                        if let Some(outer_pos) = outer_response.interact_pointer_pos() {
                            let inner_pos =
                                node_bounding_rect.min + ((outer_pos - outer_rect.min) / transform);
                            if let Some(selected_area) =
                                self.model_view_scene_rects.get_mut(&current_path)
                            {
                                *selected_area =
                                    Rect::from_center_size(inner_pos, selected_area.size());
                            }
                        }
                        if maps_left > 1 {
                            self.next_graph_subject_path = Some(current_path.clone());
                        }
                    }
                    for node in graph_layout.get_nodes().values() {
                        let pos = (node.position - node_bounding_rect.min) * transform;

                        let node_id = node.node_type.global_id();
                        let node_path = {
                            let mut path = current_path.clone();
                            path.push(node_id);
                            path
                        };

                        let is_selected = self.explorer_selection == Some(node_id);
                        let is_hovered = self.explorer_hovered == Some(node_id);

                        let time_since_last_eval: Option<Duration> = self
                            .node_execution_timestamps
                            .get(&node_path)
                            .map(|eval_time| current_time - *eval_time);

                        let active_pulse = if let Some(x) = time_since_last_eval {
                            (1.0f32 - x.as_secs_f32() / 0.5f32).max(0.0f32)
                        } else {
                            0.0
                        };

                        if active_pulse > 0.0 {
                            do_request_redraw = true;
                        }

                        let is_selected_subgraph = selected_global_id == Some(node_id);

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
                    if let Some(selected_area) = self.model_view_scene_rects.get(&current_path) {
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

        if do_request_redraw {
            ui.ctx().request_repaint_after(Duration::from_millis(20));
        }
    }

    pub(crate) fn update(
        &mut self,
        state: &mut GraphExplorerSettings,
        loaded_models: &mut LoadedModels,
        loaded_tokenizers: &mut LoadedTokenizers,
        server_request_manager: &mut ServerRequestManager,
        server_config_report: &ServerConfigReport,
        ui: &mut Ui,
    ) {
        let mut models_to_load = HashSet::<LoadedModelId>::new();
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

        self.explorer_hovered = self.next_explorer_hovered;
        self.next_explorer_hovered = None;

        if let Some(next_path) = self.next_graph_subject_path.take() {
            self.graph_subject_path = next_path;
            self.explorer_selection = None;
            self.explorer_hovered = None;
            self.inspect_window_tensor_subscription_returns.clear();
            self.inspect_window_tensor_subscriptions.clear();
            self.inspect_windows.clear();
        }

        let do_interface_panel =
            matches!(self.root_selection, GraphRootSubjectSelection::Interface(_));
        let interface_panel_height = 150.0;

        // Find the graph we are working with
        let root_graph: Option<&dyn GraphDyn> = match self.root_selection {
            GraphRootSubjectSelection::Model(model_id) => {
                let res = loaded_models
                    .loaded_models
                    .get(&model_id)
                    .map(|x| x as &dyn GraphDyn);
                if res.is_none() {
                    models_to_load.insert(model_id);
                }
                res
            }
            GraphRootSubjectSelection::Interface(interface_id) => loaded_models
                .current_interfaces
                .get(&interface_id)
                .map(|interface| interface.interface.get_super_graph() as &dyn GraphDyn),
        };

        // Profiling window
        if self.show_profiling_window {
            let mut open = true;
            egui::Window::new("Profiling")
                .open(&mut open)
                .resizable(true)
                .default_size([500.0, 400.0])
                .show(ui.ctx(), |ui| {
                    if let Some(root_graph) = root_graph {
                        self.render_profiling_window(ui, root_graph, loaded_models);
                    } else {
                        ui.label("No graph loaded.");
                    }
                });
            if !open {
                self.show_profiling_window = false;
            }
        }

        let graph_and_path = {
            if let Some(graph) = root_graph {
                let mut current_graph = graph;
                let mut current_path = vec![];
                for node_id in &self.graph_subject_path {
                    match get_inner_graph(
                        current_graph,
                        *node_id,
                        self.root_selection,
                        loaded_models,
                    ) {
                        LoadableGraphState::Loaded(next_graph) => {
                            current_graph = next_graph;
                            current_path.push(*node_id);
                        }
                        LoadableGraphState::Unloaded(model_id) => {
                            models_to_load.insert(model_id);
                        }
                        LoadableGraphState::None => {
                            // Invalid selection
                        }
                    }
                }
                Some((current_graph, current_path))
            } else {
                None
            }
        };
        if let Some((working_graph, working_path)) = graph_and_path {
            if !self.graph_layouts.contains_key(&self.graph_subject_path) {
                // Map tensors to link IDs
                let mut tensor_link_ids = HashMap::new();
                let mut link_data = HashMap::new();
                for (i, link_global_id) in working_graph.inner_link_ids().enumerate() {
                    let new_link_id = GraphLayoutLinkId(i);
                    tensor_link_ids.insert(link_global_id, new_link_id);
                    link_data.insert(
                        new_link_id,
                        GraphLayoutLinkData {
                            global_id: link_global_id,
                        },
                    );
                }

                // Build node init data for ops and I/O tensors
                let mut sourced_links = HashSet::new();
                let mut next_node_id = 0;
                let mut node_init_data = HashMap::new();
                for node_global_id in working_graph.node_ids() {
                    if let Some(node) = working_graph.get_node_by_id(&node_global_id) {
                        let new_node_id = GraphLayoutNodeId(next_node_id);
                        next_node_id += 1;

                        let mut inputs = vec![];
                        for tensor_id in node.inputs() {
                            inputs.push(tensor_link_ids[&tensor_id]);
                        }
                        let mut outputs = vec![];
                        for tensor_id in node.outputs() {
                            sourced_links.insert(tensor_link_ids[&tensor_id]);
                            outputs.push(tensor_link_ids[&tensor_id]);
                        }
                        node_init_data.insert(
                            new_node_id,
                            GraphLayoutNodeInitData {
                                node_type: GraphLayoutNodeType::GraphNode(node_global_id),
                                inputs,
                                outputs,
                            },
                        );
                    }
                }

                let mut io_tensor_node_ids = HashMap::new();
                for (_outer_id, inner_id) in working_graph.input_link_ids() {
                    if let Some(link_id) = tensor_link_ids.get(&inner_id) {
                        let node_id = GraphLayoutNodeId(next_node_id);
                        io_tensor_node_ids.insert(inner_id, node_id);
                        next_node_id += 1;
                        sourced_links.insert(*link_id);
                        node_init_data.insert(
                            node_id,
                            GraphLayoutNodeInitData {
                                node_type: GraphLayoutNodeType::InputLinkNode(inner_id),
                                inputs: vec![],
                                outputs: vec![tensor_link_ids[&inner_id]],
                            },
                        );
                    }
                }
                for (_outer_id, inner_id) in working_graph.output_link_ids() {
                    if let Some(link_id) = tensor_link_ids.get(&inner_id) {
                        let node_id = GraphLayoutNodeId(next_node_id);
                        io_tensor_node_ids.insert(inner_id, node_id);
                        next_node_id += 1;

                        node_init_data.insert(
                            node_id,
                            GraphLayoutNodeInitData {
                                node_type: GraphLayoutNodeType::OutputLinkNode(inner_id),
                                inputs: vec![*link_id],
                                outputs: vec![],
                            },
                        );
                    }
                }
                for inner_id in working_graph.constant_link_ids() {
                    if let Some(link_id) = tensor_link_ids.get(&inner_id) {
                        let node_id = GraphLayoutNodeId(next_node_id);
                        io_tensor_node_ids.insert(inner_id, node_id);
                        next_node_id += 1;
                        sourced_links.insert(*link_id);
                        node_init_data.insert(
                            node_id,
                            GraphLayoutNodeInitData {
                                node_type: GraphLayoutNodeType::ConstantLinkNode(inner_id),
                                inputs: vec![],
                                outputs: vec![tensor_link_ids[&inner_id]],
                            },
                        );
                    }
                }
                for global_id in tensor_link_ids.iter().filter_map(|(global_id, link_id)| {
                    if !sourced_links.contains(link_id) {
                        Some(global_id)
                    } else {
                        None
                    }
                }) {
                    let node_id = GraphLayoutNodeId(next_node_id);
                    io_tensor_node_ids.insert(*global_id, node_id);
                    next_node_id += 1;
                    node_init_data.insert(
                        node_id,
                        GraphLayoutNodeInitData {
                            node_type: GraphLayoutNodeType::InputLinkNode(*global_id),
                            inputs: vec![],
                            outputs: vec![tensor_link_ids[global_id]],
                        },
                    );
                }

                let initial_layout =
                    GraphLayout::new(node_init_data, link_data, ui, |ui, node_init_data| {
                        render_node_contents(
                            ui,
                            &node_init_data.node_type,
                            node_init_data.inputs.len(),
                            node_init_data.outputs.len(),
                            working_graph,
                            false,
                            false,
                            None,
                        )
                        .1
                    });
                self.graph_layouts
                    .insert(working_path.clone(), initial_layout);
            }

            match self.graph_layouts.get_mut(&self.graph_subject_path) {
                Some(Ok(graph_layout)) => {
                    // Update positions
                    if state.explorer_physics && graph_layout.update_layout(5000) {
                        ui.ctx().request_repaint_after(Duration::from_millis(20));
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
                            let mut scene_rect = if let Some(x) =
                                self.model_view_scene_rects.get(&self.graph_subject_path)
                            {
                                *x
                            } else {
                                // No bound of graph should exceed this (unless necessary for aspect ratio)
                                let max_frame = graph_layout.get_bounding_rect().expand(30.0);

                                let (min_x, max_x) = if max_frame.width() > frame_shape.x {
                                    (max_frame.min.x, max_frame.min.x + frame_shape.x)
                                } else {
                                    (max_frame.min.x, max_frame.max.x)
                                };

                                let center_y = max_frame.center().y;
                                let center_pos =
                                    egui::pos2(min_x + (max_x - min_x) / 2.0, center_y);
                                let scale = (frame_shape.x / (max_x - min_x)).clamp(0.0, 0.8);

                                Rect::from_center_size(center_pos, frame_shape * scale)
                            };
                            let cull_rect = scene_rect.expand(300.0);
                            let scene = egui::Scene::new().max_inner_size(frame_shape);
                            scene.show(ui, &mut scene_rect, |ui| {
                                // Find all ops actually in scene
                                let mut nodes_to_render_vec = graph_layout
                                    .find_nodes_within(
                                        &cull_rect.center(),
                                        cull_rect.size().length() / 2.0,
                                    )
                                    .clone();
                                let mut nodes_to_render = HashSet::<GraphLayoutNodeId>::from_iter(
                                    nodes_to_render_vec.iter().copied(),
                                );

                                let mut edges_to_render = vec![];

                                // Render both sides of all visible edges
                                for ((src_id, src_id_i), (dst_id, dst_id_i), link_id) in
                                    graph_layout.get_edges()
                                {
                                    if nodes_to_render.contains(src_id)
                                        || nodes_to_render.contains(dst_id)
                                    {
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
                                let edges_to_render = edges_to_render
                                    .into_iter()
                                    .map(|x| {
                                        (
                                            x.0,
                                            x.1,
                                            x.2,
                                            (
                                                x.3,
                                                ui.painter().add(Shape::Noop),
                                                ui.painter().add(Shape::Noop),
                                            ),
                                        )
                                    })
                                    .collect::<Vec<_>>();

                                let mut node_io_connections = HashMap::new();
                                let mut node_bounding_boxes = HashMap::new();

                                self.nodes_in_view.clear();
                                let current_time = Instant::now();
                                let current_node_data = graph_layout.get_nodes();
                                let mut node_position_updates = HashMap::new();
                                for node_id in nodes_to_render_vec {
                                    let node_path = match &current_node_data[&node_id].node_type {
                                        GraphLayoutNodeType::GraphNode(global_id) => Some(
                                            working_path
                                                .iter()
                                                .cloned()
                                                .chain(core::iter::once(*global_id))
                                                .collect::<Vec<GlobalId>>(),
                                        ),
                                        _ => None,
                                    };
                                    let global_id =
                                        current_node_data[&node_id].node_type.global_id();
                                    if let Some(node_path) = &node_path {
                                        self.nodes_in_view.insert(node_path.clone());
                                    }
                                    let duration_since_eval = if let Some(node_path) = &node_path {
                                        self.node_execution_timestamps
                                            .get(node_path)
                                            .map(|x| current_time - *x)
                                    } else {
                                        None
                                    };
                                    let pos = current_node_data[&node_id].position;
                                    let cell_shape = current_node_data[&node_id].shape;
                                    let op_rect = Rect::from_min_max(
                                        pos - cell_shape / 2.0,
                                        pos + cell_shape,
                                    );
                                    let ui_builder = UiBuilder::new().max_rect(op_rect);
                                    let mut ui_child = ui.new_child(ui_builder);

                                    let is_selected =
                                        if let Some(selected) = &self.explorer_selection {
                                            global_id == *selected
                                        } else {
                                            false
                                        };
                                    let is_hovered = if let Some(hovered) = &self.explorer_hovered {
                                        global_id == *hovered
                                    } else {
                                        false
                                    };

                                    let (resp, io_connections) = render_node_contents(
                                        &mut ui_child,
                                        &current_node_data[&node_id].node_type,
                                        current_node_data[&node_id].inputs.len(),
                                        current_node_data[&node_id].outputs.len(),
                                        working_graph,
                                        is_selected,
                                        is_hovered,
                                        duration_since_eval,
                                    );
                                    node_io_connections.insert(node_id, io_connections);
                                    node_bounding_boxes.insert(node_id, ui_child.min_rect());

                                    if resp.hovered() {
                                        self.next_explorer_hovered = Some(global_id);
                                    }
                                    if resp.clicked() {
                                        self.explorer_selection = Some(global_id);
                                    }
                                    if resp.dragged() {
                                        node_position_updates.insert(
                                            node_id,
                                            current_node_data.get(&node_id).unwrap().position
                                                + resp.drag_delta(),
                                        );
                                    }
                                    if resp.double_clicked() {
                                        match &current_node_data[&node_id].node_type {
                                            GraphLayoutNodeType::GraphNode(global_id) => {
                                                let node_path = working_path
                                                                .iter()
                                                                .cloned()
                                                                .chain(core::iter::once(*global_id))
                                                                .collect::<Vec<_>>();
                                                match get_inner_graph(
                                                    working_graph,
                                                    *global_id,
                                                    self.root_selection,
                                                    loaded_models,
                                                ) {
                                                    LoadableGraphState::Unloaded(_) |
                                                    LoadableGraphState::Loaded(_) => {
                                                        self.next_graph_subject_path = Some(
                                                            node_path,
                                                        );
                                                    }
                                                    LoadableGraphState::None => {
                                                        // No sub graph
                                                        if !AnyInspectWindow::check_if_already_exists(
                                                            &self.inspect_windows, &node_path) {
                                                            self.inspect_windows.push(
                                                                InspectWindowGraphNode::new(node_path.clone()).into_any()
                                                            )
                                                        }
                                                    }
                                                }
                                            }
                                            GraphLayoutNodeType::InputLinkNode(link_id) |
                                            GraphLayoutNodeType::OutputLinkNode(link_id) |
                                            GraphLayoutNodeType::ConstantLinkNode(link_id) |
                                            GraphLayoutNodeType::ConnectionByNameSrc(link_id) |
                                            GraphLayoutNodeType::ConnectionByNameDest(link_id) => {
                                                let link_path = working_path
                                                    .iter()
                                                    .cloned()
                                                    .chain(core::iter::once(*link_id))
                                                    .collect::<Vec<_>>();
                                                if !AnyInspectWindow::check_if_already_exists(
                                                    &self.inspect_windows, &link_path) {
                                                    self.inspect_windows.push(
                                                        InspectWindowGraphLink::new(link_path.clone()).into_any()
                                                    )
                                                }
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
                                for (
                                    (src_id, src_id_i),
                                    (dst_id, dst_id_i),
                                    link_id,
                                    (paint_idx_a, paint_idx_b, paint_idx_c),
                                ) in edges_to_render
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
                                            let this_selectable = link_data.global_id;
                                            let is_selected =
                                                self.explorer_selection == Some(this_selectable);
                                            let is_hovered =
                                                self.explorer_hovered == Some(this_selectable);
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
                                    if let Some(link_data) = link_data.get(&link_id) {
                                        let link_path = working_path
                                            .iter()
                                            .cloned()
                                            .chain(core::iter::once(link_data.global_id))
                                            .collect::<Vec<GlobalId>>();
                                        self.tensors_in_view.insert(link_path.clone());
                                        if let Some(texture) = Self::get_tensor_swatch(
                                            &mut self.rendered_tensor_swatches,
                                            &self.abbreviated_tensor_reports,
                                            state,
                                            ui.ctx(),
                                            &link_path,
                                        ) {
                                            let midpoint = points[1].lerp(points[2], 0.5);
                                            let rect =
                                                Rect::from_center_size(midpoint, Vec2::splat(32.0));
                                            let mut shape = Mesh::with_texture(texture.id());
                                            shape.add_rect_with_uv(
                                                rect,
                                                Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                                                Color32::WHITE,
                                            );
                                            ui.painter().set(paint_idx_c, shape);
                                            let color = ui.visuals().widgets.inactive.bg_fill;
                                            let shape = egui::Shape::rect_filled(
                                                rect.expand(4.0),
                                                4.0,
                                                color,
                                            );
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
        if let GraphRootSubjectSelection::Interface(interface_id) = self.root_selection
            && let Some(interface) = loaded_models.current_interfaces.get(&interface_id)
        {
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
                                        let text_inference_data = {
                                            if let Some(text_inference_data) =
                                                self.text_inference_data.get_mut(&interface_id)
                                            {
                                                text_inference_data
                                            } else {
                                                let inference_data = TextInferenceData {
                                                    tokens: tokenizer.encode("Hello World!"),
                                                    .. Default::default()
                                                };
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
                                                    for (node_path, op_kind, age, execution_duration) in report.node_executions {
                                                        let time = time_now - age;
                                                        self.node_execution_timestamps.insert(node_path.clone(), time);
                                                        self.node_execution_durations.insert(node_path.clone(), execution_duration);
                                                        if !op_kind.is_empty() {
                                                            self.node_execution_op_kinds.insert(node_path, op_kind);
                                                        }
                                                    }
                                                    for (tensor_path, value) in report.abbreviated_tensor_assignments {
                                                        self.rendered_tensor_swatches.remove(&tensor_path);
                                                        self.abbreviated_tensor_reports.insert(tensor_path, value);
                                                    }
                                                }
                                            }
                                            if let Some(response) =
                                                server_request_manager.get_response(*request_id)
                                            {
                                                let (_, link, tokens) = text_inference_data
                                                    .pending_request
                                                    .take()
                                                    .unwrap();
                                                text_inference_data.pending_request = None;
                                                match response.result {
                                                    Ok(mut data) => {
                                                        let response_tokens =
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
                                                        TokenizerInfo::HFTokenizerLocal(path) => {
                                                            format!("Local: {path}")
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
                                                            ui.selectable_value(
                                                                &mut text_inference_data.selected_mode,
                                                                SuperGraphRequestBackendMode::Compiler,
                                                                SuperGraphRequestBackendMode::Compiler.to_string());
                                                    });
                                                    if ui.button("Run").clicked() {
                                                        let tokens =
                                                            text_inference_data.tokens.clone();
                                                        let tokens_tensor =
                                                            NDArrayNumericTensor::from_vec(
                                                                tokens.clone(),
                                                            )
                                                                .to_dyn();
                                                        let swatch_settings = if state.do_explorer_swatches_in_view || state.do_all_explorer_swatches {
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
                                                                            .token_context_input_link,
                                                                        tokens_tensor,
                                                                    )]),
                                                                    symbolic_graph_ids: interface.model_ids.clone(),
                                                                    model_inputs: HashMap::from([(
                                                                        llm_interface
                                                                            .model_input_link,
                                                                        *interface
                                                                            .model_ids
                                                                            .first()
                                                                            .unwrap(),
                                                                    )]),
                                                                    hash_inputs: HashMap::from([(
                                                                        llm_interface
                                                                            .cache_key_input_link,
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
                                                                    .logit_output_link,
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
                                                context.push(*token);
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
                    AnyInterface::ImageGenerationInterface(sd_interface) => {
                        let sd_data = self
                            .sd_inference_data
                            .entry(interface_id)
                            .or_default();

                        // Handle pending reports (node wave, tensor swatches)
                        if let Some((request_id, _)) = &sd_data.pending_request {
                            if let Some(reports) = server_request_manager.get_reports(*request_id) {
                                let time_now = Instant::now();
                                for report in reports {
                                    for (path, value) in report.tensor_assignments {
                                        self.inspect_window_tensor_subscription_returns.insert(path, value);
                                    }
                                    for (node_path, op_kind, age, execution_duration) in report.node_executions {
                                        let time = time_now - age;
                                        self.node_execution_timestamps.insert(node_path.clone(), time);
                                        self.node_execution_durations.insert(node_path.clone(), execution_duration);
                                        if !op_kind.is_empty() {
                                            self.node_execution_op_kinds.insert(node_path, op_kind);
                                        }
                                    }
                                    for (tensor_path, value) in report.abbreviated_tensor_assignments {
                                        self.rendered_tensor_swatches.remove(&tensor_path);
                                        self.abbreviated_tensor_reports.insert(tensor_path, value);
                                    }
                                }
                            }
                            if let Some(response) = server_request_manager.get_response(*request_id) {
                                let (_, output_link) = sd_data.pending_request.take().unwrap();
                                match response.result {
                                    Ok(mut data) => {
                                        if let Some(image_tensor) = data.tensor_outputs.remove(&output_link) {
                                            sd_data.status_message = Some(format!("Image generated: {:?}", image_tensor.shape()));
                                            sd_data.generated_image = Some(tensor_to_egui_texture(&image_tensor, ui.ctx()));
                                        } else {
                                            sd_data.status_message = Some("Error: output tensor not found".to_string());
                                        }
                                    }
                                    Err(err) => {
                                        self.error_popup = Some(err);
                                        sd_data.status_message = Some("Error (see popup)".to_string());
                                    }
                                }
                            }
                        }

                        // UI panel
                        ui.horizontal_top(|ui| {
                        let frame = egui::Frame::default()
                            .stroke(ui.visuals().window_stroke)
                            .inner_margin(5.0);
                        frame.show(ui, |ui| {
                            ui.vertical(|ui| {
                                ui.heading("Stable Diffusion");

                                ui.horizontal(|ui| {
                                    ui.label("Prompt:");
                                    ui.text_edit_singleline(&mut sd_data.prompt);
                                });

                                ui.horizontal(|ui| {
                                    ui.label("Steps:");
                                    ui.add(egui::DragValue::new(&mut sd_data.num_steps).range(1..=100));
                                    ui.label("Guidance:");
                                    ui.add(egui::DragValue::new(&mut sd_data.guidance_scale).speed(0.1).range(1.0..=30.0));
                                    ui.label("Seed:");
                                    ui.add(egui::DragValue::new(&mut sd_data.seed));
                                });

                                ui.horizontal(|ui| {
                                    ui.label("Latent H:");
                                    ui.add(egui::DragValue::new(&mut sd_data.latent_h).range(4..=128));
                                    ui.label("Latent W:");
                                    ui.add(egui::DragValue::new(&mut sd_data.latent_w).range(4..=128));
                                    ui.label(format!("({}x{} px)", sd_data.latent_w * 8, sd_data.latent_h * 8));
                                });

                                if sd_data.pending_request.is_some() {
                                    ui.horizontal(|ui| {
                                        ui.spinner();
                                        ui.label("Generating...");
                                    });
                                } else {
                                    ui.horizontal(|ui| {
                                        egui::ComboBox::from_id_salt("sd_backend_mode")
                                            .selected_text(sd_data.selected_mode.to_string())
                                            .show_ui(ui, |ui| {
                                                ui.selectable_value(
                                                    &mut sd_data.selected_mode,
                                                    SuperGraphRequestBackendMode::NDArray,
                                                    SuperGraphRequestBackendMode::NDArray.to_string(),
                                                );
                                                if server_config_report.vulkan_available {
                                                    ui.selectable_value(
                                                        &mut sd_data.selected_mode,
                                                        SuperGraphRequestBackendMode::Vulkan,
                                                        SuperGraphRequestBackendMode::Vulkan.to_string(),
                                                    );
                                                }
                                                ui.selectable_value(
                                                    &mut sd_data.selected_mode,
                                                    SuperGraphRequestBackendMode::Compiler,
                                                    SuperGraphRequestBackendMode::Compiler.to_string(),
                                                );
                                            });
                                        toggle_ui(ui, &mut sd_data.use_cache);
                                        ui.label("Cache");
                                        let tokenizer = loaded_tokenizers
                                            .loaded_tokenizers
                                            .get(&sd_interface.tokenizer)
                                            .cloned()
                                            .flatten();
                                        let can_generate = matches!(&tokenizer, Some(Ok(_)));
                                        if !can_generate {
                                            if let Some(Err(err)) = &tokenizer {
                                                ui.label(format!("Tokenizer error: {err}"));
                                            } else {
                                                ui.spinner();
                                            }
                                        }
                                        if can_generate && ui.button("Generate").clicked() {
                                            let tokenizer = tokenizer.unwrap().unwrap();
                                            let seq_len = 77;
                                            let cond_ids = clip_tokenize(tokenizer.as_ref(), &sd_data.prompt, seq_len);
                                            let uncond_ids = clip_tokenize(tokenizer.as_ref(), "", seq_len);

                                            let cond_tensor = NDArrayNumericTensor::from_vec_shape(cond_ids, &vec![1, seq_len as u64]).unwrap();
                                            let uncond_tensor = NDArrayNumericTensor::from_vec_shape(uncond_ids, &vec![1, seq_len as u64]).unwrap();

                                            let (timestep_values, dt_values, sigma_values, init_sigma) =
                                                ImageGenerationInterface::compute_euler_schedule(sd_data.num_steps);

                                            let latent_n = 4 * sd_data.latent_h * sd_data.latent_w;
                                            let initial_noise = generate_normal_noise(latent_n, sd_data.seed);
                                            let scaled_noise: Vec<f32> = initial_noise.iter().map(|&x| x * init_sigma).collect();

                                            let latent_tensor = NDArrayNumericTensor::from_vec_shape(
                                                scaled_noise,
                                                &vec![1, 4, sd_data.latent_h as u64, sd_data.latent_w as u64],
                                            ).unwrap();

                                            let timesteps_tensor = NDArrayNumericTensor::from_vec_shape(
                                                timestep_values,
                                                &vec![sd_data.num_steps as u64],
                                            ).unwrap();
                                            let dt_tensor = NDArrayNumericTensor::from_vec_shape(
                                                dt_values,
                                                &vec![sd_data.num_steps as u64],
                                            ).unwrap();
                                            let sigmas_tensor = NDArrayNumericTensor::from_vec_shape(
                                                sigma_values,
                                                &vec![sd_data.num_steps as u64],
                                            ).unwrap();
                                            let iter_count = NDArrayNumericTensor::from_vec_shape(
                                                vec![sd_data.num_steps as i64],
                                                &vec![1],
                                            ).unwrap();
                                            let guidance = NDArrayNumericTensor::from_vec(vec![sd_data.guidance_scale]).to_dyn();

                                            let symbolic_graph_ids: Vec<_> = interface.model_ids.to_vec();
                                            let model_inputs: HashMap<_, _> = sd_interface
                                                .model_weights
                                                .iter()
                                                .zip(interface.model_ids.iter())
                                                .map(|(&link, &id)| (link, id))
                                                .collect();

                                            let mut tensor_inputs = HashMap::from([
                                                (sd_interface.cond_ids_input, cond_tensor),
                                                (sd_interface.initial_latent_input, latent_tensor),
                                                (sd_interface.timesteps_input, timesteps_tensor),
                                                (sd_interface.dt_input, dt_tensor),
                                                (sd_interface.sigmas_input, sigmas_tensor),
                                                (sd_interface.iteration_count_input, iter_count),
                                                (sd_interface.guidance_scale_input, guidance),
                                            ]);
                                            if let Some(neg_link) = sd_interface.negative_cond_ids_input {
                                                tensor_inputs.insert(neg_link, uncond_tensor);
                                            }

                                            let swatch_settings = if state.do_explorer_swatches_in_view || state.do_all_explorer_swatches {
                                                Some(AbbreviatedTensorReportSettings {
                                                    downsampled_size: (state.swatch_dimension * state.swatch_dimension) as u64,
                                                    subscribed_tensors: self.tensors_in_view.iter().cloned().collect(),
                                                    do_all: state.do_all_explorer_swatches,
                                                })
                                            } else {
                                                None
                                            };

                                            let token = server_request_manager.submit_supergraph_request(SuperGraphRequest {
                                                do_node_execution_reports: state.explorer_node_wave,
                                                abbreviated_tensor_report_settings: swatch_settings,
                                                attention_token: None,
                                                super_graph: sd_interface.super_graph.clone(),
                                                subscribed_tensors: self.inspect_window_tensor_subscriptions.iter().cloned().collect(),
                                                string_inputs: HashMap::new(),
                                                use_cache: if sd_data.use_cache { Some(200 + interface_id as u64) } else { None },
                                                backend_mode: sd_data.selected_mode,
                                                symbolic_graph_ids,
                                                tensor_inputs,
                                                model_inputs,
                                                hash_inputs: HashMap::new(),
                                            });

                                            sd_data.pending_request = Some((token, sd_interface.image_output));
                                            sd_data.status_message = Some("Running SD pipeline...".to_string());
                                        }
                                    });
                                }

                                if let Some(msg) = &sd_data.status_message {
                                    ui.label(msg);
                                }
                            });
                        });

                        // Display generated image thumbnail to the right of controls
                        if let Some((texture, _color_image)) = &sd_data.generated_image {
                            ui.vertical(|ui| {
                                let size = texture.size_vec2();
                                let thumb_max = 200.0;
                                let scale = (thumb_max / size.x.max(size.y)).min(1.0);
                                let display_size = egui::vec2(size.x * scale, size.y * scale);
                                let response = ui.add(
                                    egui::Image::new(egui::load::SizedTexture::new(texture.id(), display_size))
                                        .sense(Sense::click()),
                                );
                                if response.clicked() {
                                    sd_data.show_image_window = !sd_data.show_image_window;
                                }
                                response.on_hover_text("Click to inspect");
                            });
                        }
                        }); // end horizontal_top

                        // Floating inspect window for full-size image
                        if sd_data.show_image_window
                            && let Some((texture, color_image)) = &sd_data.generated_image {
                                let size = texture.size_vec2();
                                let color_image = color_image.clone();
                                let mut open = sd_data.show_image_window;
                                egui::Window::new("Generated Image")
                                    .open(&mut open)
                                    .resizable(true)
                                    .default_size(size)
                                    .show(ui.ctx(), |ui| {
                                        ui.image(egui::load::SizedTexture::new(texture.id(), size));
                                        ui.horizontal(|ui| {
                                            if ui.button("Save image").clicked() {
                                                save_image_to_download(&color_image);
                                            }
                                            if ui.button("Copy to clipboard").clicked() {
                                                copy_image_to_clipboard(&color_image);
                                            }
                                        });
                                    });
                                sd_data.show_image_window = open;
                        }
                    }
                }
                ui.allocate_exact_size(ui.available_size_before_wrap(), Sense::click());
            });
        }
        // Prompt model loading

        for model_id in models_to_load {
            if !loaded_models.loaded_models.contains_key(&model_id)
                && loaded_models.currently_requesting_model.is_none()
            {
                log::info!("Loading model: {}", model_id);
                server_request_manager
                    .send(WebsocketClientServerMessage::GetModelGraph(model_id))
                    .unwrap();
                loaded_models.currently_requesting_model = Some(model_id);
            }
        }
    }
}

fn save_image_to_download(color_image: &ColorImage) {
    let [w, h] = color_image.size;
    let bmp_data = encode_bmp(w, h, &color_image.pixels);
    let _ = trigger_browser_download("generated_image.bmp", &bmp_data, "image/bmp");
}

fn copy_image_to_clipboard(color_image: &ColorImage) {
    let [w, h] = color_image.size;
    let mut rgba = Vec::with_capacity(w * h * 4);
    for pixel in &color_image.pixels {
        rgba.push(pixel.r());
        rgba.push(pixel.g());
        rgba.push(pixel.b());
        rgba.push(pixel.a());
    }
    js_copy_image_to_clipboard(&rgba, w as u32, h as u32);
}

fn encode_bmp(w: usize, h: usize, pixels: &[Color32]) -> Vec<u8> {
    let row_size = w * 3;
    let row_padding = (4 - (row_size % 4)) % 4;
    let padded_row = row_size + row_padding;
    let pixel_data_size = padded_row * h;
    let file_size = 54 + pixel_data_size;

    let mut data = Vec::with_capacity(file_size);

    // BMP file header (14 bytes)
    data.extend_from_slice(b"BM");
    data.extend_from_slice(&(file_size as u32).to_le_bytes());
    data.extend_from_slice(&0u16.to_le_bytes()); // reserved
    data.extend_from_slice(&0u16.to_le_bytes()); // reserved
    data.extend_from_slice(&54u32.to_le_bytes()); // pixel data offset

    // DIB header (40 bytes)
    data.extend_from_slice(&40u32.to_le_bytes()); // header size
    data.extend_from_slice(&(w as i32).to_le_bytes());
    data.extend_from_slice(&(h as i32).to_le_bytes());
    data.extend_from_slice(&1u16.to_le_bytes()); // planes
    data.extend_from_slice(&24u16.to_le_bytes()); // bits per pixel
    data.extend_from_slice(&0u32.to_le_bytes()); // no compression
    data.extend_from_slice(&(pixel_data_size as u32).to_le_bytes());
    data.extend_from_slice(&2835u32.to_le_bytes()); // h resolution (72 dpi)
    data.extend_from_slice(&2835u32.to_le_bytes()); // v resolution
    data.extend_from_slice(&0u32.to_le_bytes()); // colors in palette
    data.extend_from_slice(&0u32.to_le_bytes()); // important colors

    // Pixel data (bottom-up, BGR)
    for y in (0..h).rev() {
        for x in 0..w {
            let c = pixels[y * w + x];
            data.push(c.b());
            data.push(c.g());
            data.push(c.r());
        }
        data.resize(data.len() + row_padding, 0);
    }

    data
}

fn trigger_browser_download(
    filename: &str,
    data: &[u8],
    mime_type: &str,
) -> Result<(), wasm_bindgen::JsValue> {
    let uint8_array = js_sys::Uint8Array::from(data);
    let array = js_sys::Array::new();
    array.push(&uint8_array.buffer());

    let options = web_sys::BlobPropertyBag::new();
    options.set_type(mime_type);
    let blob = web_sys::Blob::new_with_u8_array_sequence_and_options(&array, &options)?;

    let url = web_sys::Url::create_object_url_with_blob(&blob)?;

    let window = web_sys::window().ok_or("no window")?;
    let document = window.document().ok_or("no document")?;
    let a = document
        .create_element("a")?
        .dyn_into::<web_sys::HtmlAnchorElement>()?;
    a.set_href(&url);
    a.set_download(filename);
    a.style().set_property("display", "none")?;
    document.body().ok_or("no body")?.append_child(&a)?;
    a.click();
    a.remove();
    web_sys::Url::revoke_object_url(&url)?;
    Ok(())
}
