mod graph_layout;
pub mod inspect_windows;
mod tensor_swatch;

use crate::app::{
    ClientGraphId, EditableClientGraphMut, GraphEditability, GraphRootOwnership, InterfaceId,
    LoadedModels, LoadedTokenizers,
};
#[cfg(not(target_arch = "wasm32"))]
use crate::audio_io::pick_audio_file_native;
#[cfg(target_arch = "wasm32")]
use crate::audio_io::{WebAudioFilePickReceiver, start_audio_file_pick_web};
use crate::audio_io::{
    decode_wav_bytes_to_mono_f32, download_audio_wav, play_audio_samples, stop_audio_playback,
    tensor_to_audio_samples,
};
use crate::graph_explorer::inspect_windows::{
    AnyInspectWindow, InspectWindowGraphLink, InspectWindowGraphNode,
};
use crate::sd_explorer::{generate_normal_noise, tensor_to_egui_texture};
use crate::websockets::ServerRequestManager;
use crate::widgets::progress_report::SuperGraphProgressWidgetState;
use crate::widgets::toggle::toggle_ui;
use crate::widgets::tokenized_rich_text::TokenizedRichText;
use egui::epaint::CubicBezierShape;
use egui::{
    Color32, ColorImage, Context, Label, Margin, Mesh, Pos2, Rect, Response, RichText, Sense,
    Shape, Stroke, StrokeKind, TextureHandle, Ui, UiBuilder, Vec2, vec2,
};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::wasm_bindgen;

#[cfg(target_arch = "wasm32")]
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
#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};
use tensor_swatch::build_tensor_swatch;
#[cfg(target_arch = "wasm32")]
use web_time::{Duration, Instant};
use whisper_tensor::DynRank;
use whisper_tensor::backends::ndarray_backend::NDArrayNumericTensor;
use whisper_tensor::dtype::DType;
use whisper_tensor::graph::{GlobalId, Graph, GraphDyn, Node, SlotDirection};
use whisper_tensor::graph_format::{MILLI_OP_GRAPH_FILE_EXTENSION, SUPER_GRAPH_FILE_EXTENSION};
use whisper_tensor::interfaces::{
    AnyInterface, ImageGenerationInterface, KokoroVoiceEmbedding, TTSInputConfig,
};
use whisper_tensor::metadata::TokenizerInfo;
use whisper_tensor::milli_graph::MilliOpGraph;
use whisper_tensor::scalar_info::ScalarInfoTyped;
use whisper_tensor::super_graph::nodes::{SuperGraphAnyNode, SuperGraphNode};
use whisper_tensor::super_graph::{SuperGraph, SuperGraphLink, SuperGraphLinkInfo};
use whisper_tensor::tokenizer::Tokenizer;
use whisper_tensor_server::{
    AbbreviatedTensorReportSettings, AbbreviatedTensorValue, LoadedModelId, ServerConfigReport,
    SuperGraphAudioInput, SuperGraphRequest, SuperGraphRequestBackendMode,
    WebsocketClientServerMessage,
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
    ServerModel(LoadedModelId),
    ServerInterface(InterfaceId),
    ClientGraph(ClientGraphId),
}

#[derive(Clone, Debug, Default)]
pub(crate) struct TextInferenceData {
    tokens: Vec<u32>,
    logits: HashMap<Vec<u32>, Vec<(u32, f32)>>,
    pending_request: Option<(u64, SuperGraphLink, Vec<u32>)>,
    use_cache: bool,
    selected_mode: SuperGraphRequestBackendMode,
    progress_widget_state: SuperGraphProgressWidgetState,
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
    pending_request: Option<(u64, SuperGraphLink)>,
    progress_widget_state: SuperGraphProgressWidgetState,
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
            progress_widget_state: SuperGraphProgressWidgetState::default(),
            generated_image: None,
            status_message: None,
            show_image_window: false,
        }
    }
}

#[derive(Clone)]
pub(crate) struct TTSInferenceData {
    text: String,
    speed: f32,
    piper_speaker_id: i64,
    kokoro_voice_name: Option<String>,
    use_cache: bool,
    selected_mode: SuperGraphRequestBackendMode,
    pending_request: Option<(u64, SuperGraphLink, u32)>,
    progress_widget_state: SuperGraphProgressWidgetState,
    generated_audio: Option<NDArrayNumericTensor<DynRank>>,
    generated_sample_rate_hz: Option<u32>,
    status_message: Option<String>,
}

impl Default for TTSInferenceData {
    fn default() -> Self {
        Self {
            text: "Hello from Whisper Tensor".to_string(),
            speed: 1.0,
            piper_speaker_id: 0,
            kokoro_voice_name: None,
            use_cache: false,
            selected_mode: SuperGraphRequestBackendMode::NDArray,
            pending_request: None,
            progress_widget_state: SuperGraphProgressWidgetState::default(),
            generated_audio: None,
            generated_sample_rate_hz: None,
            status_message: None,
        }
    }
}

pub(crate) struct STTInferenceData {
    use_cache: bool,
    selected_mode: SuperGraphRequestBackendMode,
    pending_request: Option<(u64, SuperGraphLink, u32, TokenizerInfo)>,
    progress_widget_state: SuperGraphProgressWidgetState,
    selected_audio_name: Option<String>,
    selected_audio_bytes: Option<Vec<u8>>,
    transcription_text: Option<String>,
    transcription_tokens: Option<Vec<u32>>,
    status_message: Option<String>,
    #[cfg(target_arch = "wasm32")]
    pending_web_audio_pick: Option<WebAudioFilePickReceiver>,
}

impl Default for STTInferenceData {
    fn default() -> Self {
        Self {
            use_cache: false,
            selected_mode: SuperGraphRequestBackendMode::NDArray,
            pending_request: None,
            progress_widget_state: SuperGraphProgressWidgetState::default(),
            selected_audio_name: None,
            selected_audio_bytes: None,
            transcription_text: None,
            transcription_tokens: None,
            status_message: None,
            #[cfg(target_arch = "wasm32")]
            pending_web_audio_pick: None,
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
    pub(crate) tts_inference_data: HashMap<InterfaceId, TTSInferenceData>,
    pub(crate) stt_inference_data: HashMap<InterfaceId, STTInferenceData>,
    node_execution_timestamps: HashMap<Vec<GlobalId>, Instant>,
    node_last_child_active_timestamps: HashMap<Vec<GlobalId>, Instant>,
    node_execution_durations: HashMap<Vec<GlobalId>, Duration>,
    node_execution_op_kinds: HashMap<Vec<GlobalId>, String>,
    abbreviated_tensor_reports: HashMap<Vec<GlobalId>, AbbreviatedTensorValue>,
    rendered_tensor_swatches: HashMap<Vec<GlobalId>, TextureHandle>,
    tensors_in_view: HashSet<Vec<GlobalId>>,
    nodes_in_view: HashSet<Vec<GlobalId>>,
    actions_status: Option<String>,
    error_popup: Option<String>,
    pub(crate) show_profiling_window: bool,
    undo_history: Vec<GraphEditHistoryEntry>,
    redo_history: Vec<GraphEditHistoryEntry>,
    pending_link_drag: Option<PendingLinkDrag>,
    pending_link_edit_request: Option<(Vec<GlobalId>, SlotPipEndpoint, SlotPipEndpoint)>,
    pending_history_action: Option<PendingHistoryAction>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SlotPipOwner {
    Node(GlobalId),
    InputLink(GlobalId),
    OutputLink(GlobalId),
    ConstantLink(GlobalId),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SlotPipEndpoint {
    owner: SlotPipOwner,
    direction: SlotDirection,
    slot_index: usize,
    link_id: Option<GlobalId>,
    layout_link_id: Option<GraphLayoutLinkId>,
    screen_pos: Pos2,
}

#[derive(Clone, Debug)]
struct PendingLinkDrag {
    graph_path: Vec<GlobalId>,
    source: SlotPipEndpoint,
}

#[derive(Clone, Debug)]
struct LinkEditApplyResult {
    status: String,
    link_global_id: GlobalId,
    requires_layout_refresh: bool,
    history_entry: Option<GraphEditHistoryEntry>,
}

#[derive(Clone, Debug)]
enum EditableGraphSnapshot {
    SuperGraph(Box<SuperGraph>),
    MilliOpGraph(Box<MilliOpGraph>),
}

#[derive(Clone, Debug)]
struct GraphEditHistoryEntry {
    label: String,
    before: EditableGraphSnapshot,
    after: EditableGraphSnapshot,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PendingHistoryAction {
    Undo,
    Redo,
}

const MAX_GRAPH_UNDO_HISTORY: usize = 64;

fn resolve_supergraph_mut_at_path<'a>(
    root_graph: &'a mut SuperGraph,
    path: &[GlobalId],
) -> Option<&'a mut SuperGraph> {
    let mut graph = root_graph;
    for node_id in path {
        let node = graph.nodes.get_mut(node_id)?;
        graph = node.get_sub_graph_mut()?;
    }
    Some(graph)
}

fn graph_layout_node_type_for_slot_owner(owner: SlotPipOwner) -> GraphLayoutNodeType {
    match owner {
        SlotPipOwner::Node(node_id) => GraphLayoutNodeType::GraphNode(node_id),
        SlotPipOwner::InputLink(link_id) => GraphLayoutNodeType::InputLinkNode(link_id),
        SlotPipOwner::OutputLink(link_id) => GraphLayoutNodeType::OutputLinkNode(link_id),
        SlotPipOwner::ConstantLink(link_id) => GraphLayoutNodeType::ConstantLinkNode(link_id),
    }
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

fn ensure_layout_link_id(
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

struct NodeExecutionActivityMapsMut<'a> {
    node_execution_timestamps: &'a mut HashMap<Vec<GlobalId>, Instant>,
    node_last_child_active_timestamps: &'a mut HashMap<Vec<GlobalId>, Instant>,
    node_execution_durations: &'a mut HashMap<Vec<GlobalId>, Duration>,
    node_execution_op_kinds: &'a mut HashMap<Vec<GlobalId>, String>,
}

fn record_node_execution_activity(
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

fn duration_since_node_activity(
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
            tts_inference_data: HashMap::new(),
            stt_inference_data: HashMap::new(),
            node_execution_timestamps: HashMap::new(),
            node_last_child_active_timestamps: HashMap::new(),
            node_execution_durations: HashMap::new(),
            node_execution_op_kinds: HashMap::new(),
            abbreviated_tensor_reports: HashMap::new(),
            rendered_tensor_swatches: HashMap::new(),
            nodes_in_view: HashSet::new(),
            tensors_in_view: HashSet::new(),
            actions_status: None,
            error_popup: None,
            show_profiling_window: false,
            undo_history: Vec::new(),
            redo_history: Vec::new(),
            pending_link_drag: None,
            pending_link_edit_request: None,
            pending_history_action: None,
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

    fn breadcrumb_node_text(graph: &dyn GraphDyn, node_id: GlobalId) -> String {
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

    fn build_graph_breadcrumb_items(
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

    fn interface_progress_widget_state(
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

                        let time_since_last_activity = duration_since_node_activity(
                            &self.node_execution_timestamps,
                            &self.node_last_child_active_timestamps,
                            &node_path,
                            current_time,
                        );

                        let active_pulse = if let Some(x) = time_since_last_activity {
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

    fn push_history_entry(&mut self, entry: GraphEditHistoryEntry) {
        self.undo_history.push(entry);
        if self.undo_history.len() > MAX_GRAPH_UNDO_HISTORY {
            self.undo_history.drain(..1);
        }
        self.redo_history.clear();
    }

    fn restore_root_graph_snapshot(
        &mut self,
        loaded_models: &mut LoadedModels,
        snapshot: &EditableGraphSnapshot,
    ) -> Result<(), String> {
        let restore_result =
            loaded_models.with_editable_client_graph_mut(self.root_selection, |graph| {
                match (graph, snapshot) {
                    (
                        EditableClientGraphMut::SuperGraph(target_graph),
                        EditableGraphSnapshot::SuperGraph(snapshot_graph),
                    ) => {
                        *target_graph = snapshot_graph.as_ref().clone();
                        Ok(())
                    }
                    (
                        EditableClientGraphMut::MilliOpGraph(target_graph),
                        EditableGraphSnapshot::MilliOpGraph(snapshot_graph),
                    ) => {
                        *target_graph = snapshot_graph.as_ref().clone();
                        Ok(())
                    }
                    (
                        EditableClientGraphMut::MilliOpGraph(_),
                        EditableGraphSnapshot::SuperGraph(_),
                    ) => Err("snapshot type mismatch: expected MilliOpGraph snapshot".to_string()),
                    (
                        EditableClientGraphMut::SuperGraph(_),
                        EditableGraphSnapshot::MilliOpGraph(_),
                    ) => Err("snapshot type mismatch: expected SuperGraph snapshot".to_string()),
                }
            });
        match restore_result {
            Ok(inner) => inner,
            Err(err) => Err(err.to_string()),
        }
    }

    fn invalidate_graph_view_cache(&mut self) {
        self.graph_layouts.clear();
        self.model_view_scene_rects.clear();
        self.pending_link_drag = None;
        self.pending_link_edit_request = None;
    }

    fn render_graph_actions_panel(
        &mut self,
        ui: &mut Ui,
        working_graph: &dyn GraphDyn,
        working_path: &[GlobalId],
        ownership: GraphRootOwnership,
        editability: GraphEditability,
    ) {
        ui.vertical(|ui| {
            ui.label(egui::RichText::new("Actions").size(11.0).strong());
            let ownership_text = match ownership {
                GraphRootOwnership::Server => "server",
                GraphRootOwnership::Client => "client",
            };
            ui.label(egui::RichText::new(format!("ownership: {ownership_text}")).size(11.0));
            ui.label(egui::RichText::new(editability.description()).size(11.0));
            if editability.can_edit() {
                ui.label(egui::RichText::new("edit actions: enabled").size(11.0));
                ui.horizontal(|ui| {
                    let undo_count = self.undo_history.len();
                    let redo_count = self.redo_history.len();
                    let undo_clicked = ui
                        .add_enabled(
                            undo_count > 0,
                            egui::Button::new(format!("Undo ({undo_count})")),
                        )
                        .clicked();
                    let redo_clicked = ui
                        .add_enabled(
                            redo_count > 0,
                            egui::Button::new(format!("Redo ({redo_count})")),
                        )
                        .clicked();
                    if undo_clicked {
                        self.pending_history_action = Some(PendingHistoryAction::Undo);
                    } else if redo_clicked {
                        self.pending_history_action = Some(PendingHistoryAction::Redo);
                    }
                });
            }

            let export_result = if let Some(super_graph) =
                <dyn Any>::downcast_ref::<SuperGraph>(working_graph.as_any())
            {
                let default_filename = graph_export_default_filename(
                    "super_graph",
                    SUPER_GRAPH_FILE_EXTENSION,
                    working_path,
                );
                let clicked = ui
                    .button("Export SuperGraph")
                    .on_hover_text(format!("Write {default_filename}"))
                    .clicked();
                if clicked {
                    match super_graph.to_cbor_bytes() {
                        Ok(bytes) => Some(export_graph_bytes(&default_filename, &bytes)),
                        Err(err) => Some(Err(format!("failed to encode SuperGraph: {err}"))),
                    }
                } else {
                    None
                }
            } else if let Some(milli_op_graph) =
                <dyn Any>::downcast_ref::<MilliOpGraph>(working_graph.as_any())
            {
                let default_filename = graph_export_default_filename(
                    "milli_op_graph",
                    MILLI_OP_GRAPH_FILE_EXTENSION,
                    working_path,
                );
                let clicked = ui
                    .button("Export MilliOpGraph")
                    .on_hover_text(format!("Write {default_filename}"))
                    .clicked();
                if clicked {
                    match milli_op_graph.to_cbor_bytes() {
                        Ok(bytes) => Some(export_graph_bytes(&default_filename, &bytes)),
                        Err(err) => Some(Err(format!("failed to encode MilliOpGraph: {err}"))),
                    }
                } else {
                    None
                }
            } else {
                ui.label(
                    egui::RichText::new("No export available for this graph type.").size(11.0),
                );
                None
            };

            if let Some(result) = export_result {
                match result {
                    Ok(status) => {
                        self.actions_status = Some(status);
                    }
                    Err(err) => {
                        let full_error = format!("Export failed: {err}");
                        self.actions_status = Some(full_error.clone());
                        self.error_popup = Some(full_error);
                    }
                }
            }

            if let Some(status) = &self.actions_status {
                ui.label(egui::RichText::new(status).size(11.0));
            }
        });
    }

    fn apply_link_drag_edit(
        &mut self,
        loaded_models: &mut LoadedModels,
        working_path: &[GlobalId],
        source: SlotPipEndpoint,
        target: SlotPipEndpoint,
        editability: GraphEditability,
    ) -> Result<LinkEditApplyResult, String> {
        if !editability.can_edit() {
            return Err(
                "Link editing is only enabled for client-loaded editable graphs.".to_string(),
            );
        }

        let (output_endpoint, input_endpoint) = match (source.direction, target.direction) {
            (SlotDirection::Output, SlotDirection::Input) => (source, target),
            (SlotDirection::Input, SlotDirection::Output) => (target, source),
            _ => {
                return Err(
                    "Drag must connect an output slot to an input slot (or vice versa)."
                        .to_string(),
                );
            }
        };

        let edit_result = loaded_models.with_editable_client_graph_mut(
            self.root_selection,
            |graph| match graph {
                EditableClientGraphMut::SuperGraph(root_graph) => {
                    let before_snapshot =
                        EditableGraphSnapshot::SuperGraph(Box::new(root_graph.clone()));

                    let mut requires_layout_refresh = false;

                    let (status, output_link_global_id) = {
                        let graph = resolve_supergraph_mut_at_path(root_graph, working_path)
                            .ok_or_else(|| {
                                "Failed to resolve mutable SuperGraph at current path.".to_string()
                            })?;
                        let link_global_id = output_endpoint
                            .link_id
                            .or(input_endpoint.link_id)
                            .unwrap_or_else(|| {
                                let mut rng = rand::rng();
                                GlobalId::new(&mut rng)
                            });

                        match output_endpoint.owner {
                            SlotPipOwner::Node(node_id) => {
                                let output_node = graph
                                    .nodes
                                    .get_mut(&node_id)
                                    .ok_or_else(|| format!("Missing output node {node_id}"))?;
                                Node::set_output_slot(
                                    output_node,
                                    output_endpoint.slot_index,
                                    Some(link_global_id),
                                )
                                .map_err(|err| {
                                    format!(
                                        "Failed to set output slot {} on {}: {err:?}",
                                        output_endpoint.slot_index, node_id
                                    )
                                })?;
                            }
                            SlotPipOwner::InputLink(link_id)
                            | SlotPipOwner::ConstantLink(link_id) => {
                                if link_id != link_global_id {
                                    return Err(format!(
                                        "Cannot retarget source link {} to {}",
                                        link_id, link_global_id
                                    ));
                                }
                            }
                            SlotPipOwner::OutputLink(link_id) => {
                                return Err(format!(
                                    "Graph output link {} cannot be used as a source endpoint",
                                    link_id
                                ));
                            }
                        }

                        match input_endpoint.owner {
                            SlotPipOwner::Node(node_id) => {
                                let input_node = graph
                                    .nodes
                                    .get_mut(&node_id)
                                    .ok_or_else(|| format!("Missing input node {node_id}"))?;
                                Node::set_input_slot(
                                    input_node,
                                    input_endpoint.slot_index,
                                    Some(link_global_id),
                                )
                                .map_err(|err| {
                                    format!(
                                        "Failed to set input slot {} on {}: {err:?}",
                                        input_endpoint.slot_index, node_id
                                    )
                                })?;
                            }
                            SlotPipOwner::OutputLink(link_id) => {
                                if !graph.output_links.iter().any(|x| x.global_id() == link_id) {
                                    return Err(format!("Missing graph output link {}", link_id));
                                }
                            }
                            SlotPipOwner::InputLink(link_id) => {
                                return Err(format!(
                                    "Graph input link {} cannot be used as an input endpoint",
                                    link_id
                                ));
                            }
                            SlotPipOwner::ConstantLink(link_id) => {
                                return Err(format!(
                                    "Constant link {} cannot be used as an input endpoint",
                                    link_id
                                ));
                            }
                        }

                        let output_link = match output_endpoint.owner {
                            SlotPipOwner::Node(node_id) => {
                                let output_node = graph
                                    .nodes
                                    .get(&node_id)
                                    .ok_or_else(|| format!("Missing output node {node_id}"))?;
                                SuperGraphNode::output_slots(output_node)
                                    .nth(output_endpoint.slot_index)
                                    .flatten()
                                    .ok_or_else(|| {
                                        format!(
                                            "Output slot {} on {} has no link after update",
                                            output_endpoint.slot_index, node_id
                                        )
                                    })?
                            }
                            SlotPipOwner::InputLink(link_id)
                            | SlotPipOwner::OutputLink(link_id)
                            | SlotPipOwner::ConstantLink(link_id) => graph
                                .links_by_global_id
                                .get(&link_id)
                                .map(|x| x.link())
                                .ok_or_else(|| {
                                    format!("Missing link metadata for endpoint link {}", link_id)
                                })?,
                        };

                        graph
                            .links_by_global_id
                            .entry(output_link.global_id())
                            .or_insert_with(|| SuperGraphLinkInfo::new(output_link, None));

                        if let SlotPipOwner::OutputLink(existing_output_global_id) =
                            input_endpoint.owner
                            && existing_output_global_id != output_link.global_id()
                        {
                            let existing_output_link = graph
                                .output_links
                                .iter()
                                .copied()
                                .find(|x| x.global_id() == existing_output_global_id)
                                .ok_or_else(|| {
                                    format!(
                                        "Missing graph output link {}",
                                        existing_output_global_id
                                    )
                                })?;
                            if existing_output_link.kind() != output_link.kind() {
                                return Err(format!(
                                    "Cannot retarget output link {} ({}) to {} ({})",
                                    existing_output_global_id,
                                    existing_output_link.kind().as_str(),
                                    output_link.global_id(),
                                    output_link.kind().as_str()
                                ));
                            }
                            graph.output_links.remove(&existing_output_link);
                            graph.output_links.insert(output_link);
                            requires_layout_refresh = true;
                        }

                        let describe_endpoint = |endpoint: &SlotPipEndpoint| -> String {
                            match endpoint.owner {
                                SlotPipOwner::Node(node_id) => {
                                    format!("node {} slot {}", node_id, endpoint.slot_index)
                                }
                                SlotPipOwner::InputLink(link_id) => {
                                    format!(
                                        "graph input link {} slot {}",
                                        link_id, endpoint.slot_index
                                    )
                                }
                                SlotPipOwner::OutputLink(link_id) => {
                                    format!(
                                        "graph output link {} slot {}",
                                        link_id, endpoint.slot_index
                                    )
                                }
                                SlotPipOwner::ConstantLink(link_id) => {
                                    format!(
                                        "constant link {} slot {}",
                                        link_id, endpoint.slot_index
                                    )
                                }
                            }
                        };

                        (
                            format!(
                                "Linked {} -> {} ({})",
                                describe_endpoint(&output_endpoint),
                                describe_endpoint(&input_endpoint),
                                output_link.global_id()
                            ),
                            output_link.global_id(),
                        )
                    };

                    let after_snapshot =
                        EditableGraphSnapshot::SuperGraph(Box::new(root_graph.clone()));
                    let history_entry = GraphEditHistoryEntry {
                        label: status.clone(),
                        before: before_snapshot,
                        after: after_snapshot,
                    };

                    Ok(LinkEditApplyResult {
                        status,
                        link_global_id: output_link_global_id,
                        requires_layout_refresh,
                        history_entry: Some(history_entry),
                    })
                }
                EditableClientGraphMut::MilliOpGraph(graph) => {
                    if !working_path.is_empty() {
                        return Err(
                            "MilliOpGraph link editing does not support nested graph paths."
                                .to_string(),
                        );
                    }

                    let before_snapshot =
                        EditableGraphSnapshot::MilliOpGraph(Box::new(graph.clone()));
                    let mut requires_layout_refresh = false;

                    let (status, link_global_id) = {
                        let link_global_id = output_endpoint
                            .link_id
                            .or(input_endpoint.link_id)
                            .unwrap_or_else(|| {
                                let mut rng = rand::rng();
                                GlobalId::new(&mut rng)
                            });
                        graph.ensure_tensor_with_id(link_global_id);

                        match output_endpoint.owner {
                            SlotPipOwner::Node(node_id) => {
                                let output_node = graph
                                    .get_op_mut(&node_id)
                                    .ok_or_else(|| format!("Missing output node {node_id}"))?;
                                Node::set_output_slot(
                                    output_node,
                                    output_endpoint.slot_index,
                                    Some(link_global_id),
                                )
                                .map_err(|err| {
                                    format!(
                                        "Failed to set output slot {} on {}: {err:?}",
                                        output_endpoint.slot_index, node_id
                                    )
                                })?;
                            }
                            SlotPipOwner::InputLink(link_id) => {
                                if !graph.has_internal_input_link(link_id) {
                                    return Err(format!("Missing graph input link {}", link_id));
                                }
                                if link_id != link_global_id {
                                    return Err(format!(
                                        "Cannot retarget source link {} to {}",
                                        link_id, link_global_id
                                    ));
                                }
                            }
                            SlotPipOwner::ConstantLink(link_id) => {
                                return Err(format!(
                                    "Constant link {} cannot be used as a source endpoint",
                                    link_id
                                ));
                            }
                            SlotPipOwner::OutputLink(link_id) => {
                                return Err(format!(
                                    "Graph output link {} cannot be used as a source endpoint",
                                    link_id
                                ));
                            }
                        }

                        match input_endpoint.owner {
                            SlotPipOwner::Node(node_id) => {
                                let input_node = graph
                                    .get_op_mut(&node_id)
                                    .ok_or_else(|| format!("Missing input node {node_id}"))?;
                                Node::set_input_slot(
                                    input_node,
                                    input_endpoint.slot_index,
                                    Some(link_global_id),
                                )
                                .map_err(|err| {
                                    format!(
                                        "Failed to set input slot {} on {}: {err:?}",
                                        input_endpoint.slot_index, node_id
                                    )
                                })?;
                            }
                            SlotPipOwner::OutputLink(link_id) => {
                                let changed = graph
                                    .retarget_output_internal_link(link_id, link_global_id)
                                    .map_err(|err| {
                                        format!(
                                            "Cannot retarget output sink link {} to {}: {err}",
                                            link_id, link_global_id
                                        )
                                    })?;
                                if changed {
                                    requires_layout_refresh = true;
                                }
                            }
                            SlotPipOwner::InputLink(link_id) => {
                                return Err(format!(
                                    "Graph input link {} cannot be used as an input endpoint",
                                    link_id
                                ));
                            }
                            SlotPipOwner::ConstantLink(link_id) => {
                                return Err(format!(
                                    "Constant link {} cannot be used as an input endpoint",
                                    link_id
                                ));
                            }
                        }

                        let describe_endpoint = |endpoint: &SlotPipEndpoint| -> String {
                            match endpoint.owner {
                                SlotPipOwner::Node(node_id) => {
                                    format!("node {} slot {}", node_id, endpoint.slot_index)
                                }
                                SlotPipOwner::InputLink(link_id) => {
                                    format!(
                                        "graph input link {} slot {}",
                                        link_id, endpoint.slot_index
                                    )
                                }
                                SlotPipOwner::OutputLink(link_id) => {
                                    format!(
                                        "graph output link {} slot {}",
                                        link_id, endpoint.slot_index
                                    )
                                }
                                SlotPipOwner::ConstantLink(link_id) => {
                                    format!(
                                        "constant link {} slot {}",
                                        link_id, endpoint.slot_index
                                    )
                                }
                            }
                        };

                        (
                            format!(
                                "Linked {} -> {} ({})",
                                describe_endpoint(&output_endpoint),
                                describe_endpoint(&input_endpoint),
                                link_global_id
                            ),
                            link_global_id,
                        )
                    };

                    let after_snapshot =
                        EditableGraphSnapshot::MilliOpGraph(Box::new(graph.clone()));
                    let history_entry = GraphEditHistoryEntry {
                        label: status.clone(),
                        before: before_snapshot,
                        after: after_snapshot,
                    };

                    Ok(LinkEditApplyResult {
                        status,
                        link_global_id,
                        requires_layout_refresh,
                        history_entry: Some(history_entry),
                    })
                }
            },
        );

        match edit_result {
            Ok(result) => result,
            Err(err) => Err(err.to_string()),
        }
    }

    fn apply_link_drag_edit_to_layout(
        graph_layout: &mut GraphLayout,
        source: SlotPipEndpoint,
        target: SlotPipEndpoint,
        link_global_id: GlobalId,
    ) -> Result<(), String> {
        let (output_endpoint, input_endpoint) = match (source.direction, target.direction) {
            (SlotDirection::Output, SlotDirection::Input) => (source, target),
            (SlotDirection::Input, SlotDirection::Output) => (target, source),
            _ => {
                return Err(
                    "Drag must connect an output slot to an input slot (or vice versa)."
                        .to_string(),
                );
            }
        };

        let layout_link_id = output_endpoint
            .layout_link_id
            .filter(|_| output_endpoint.link_id == Some(link_global_id))
            .or_else(|| {
                input_endpoint
                    .layout_link_id
                    .filter(|_| input_endpoint.link_id == Some(link_global_id))
            })
            .unwrap_or_else(|| graph_layout.ensure_link_id_for_global(link_global_id));

        match output_endpoint.owner {
            SlotPipOwner::Node(_) | SlotPipOwner::InputLink(_) | SlotPipOwner::ConstantLink(_) => {}
            SlotPipOwner::OutputLink(link_id) => {
                return Err(format!(
                    "Graph output link {} cannot be used as a source endpoint",
                    link_id
                ));
            }
        }

        match input_endpoint.owner {
            SlotPipOwner::Node(_) | SlotPipOwner::OutputLink(_) => {}
            SlotPipOwner::InputLink(link_id) => {
                return Err(format!(
                    "Graph input link {} cannot be used as an input endpoint",
                    link_id
                ));
            }
            SlotPipOwner::ConstantLink(link_id) => {
                return Err(format!(
                    "Constant link {} cannot be used as an input endpoint",
                    link_id
                ));
            }
        }

        let output_node_type = graph_layout_node_type_for_slot_owner(output_endpoint.owner);
        graph_layout
            .set_slot_link(
                &output_node_type,
                SlotDirection::Output,
                output_endpoint.slot_index,
                Some(layout_link_id),
            )
            .map_err(|err| {
                format!(
                    "Failed to set layout output slot {} for {:?}: {err}",
                    output_endpoint.slot_index, output_node_type
                )
            })?;

        let input_node_type = graph_layout_node_type_for_slot_owner(input_endpoint.owner);
        graph_layout
            .set_slot_link(
                &input_node_type,
                SlotDirection::Input,
                input_endpoint.slot_index,
                Some(layout_link_id),
            )
            .map_err(|err| {
                format!(
                    "Failed to set layout input slot {} for {:?}: {err}",
                    input_endpoint.slot_index, input_node_type
                )
            })?;

        graph_layout.rebuild_connectivity();
        Ok(())
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
            self.actions_status = None;
            self.pending_link_drag = None;
            self.pending_link_edit_request = None;
            self.pending_history_action = None;
        }

        if let Some((working_path, source_endpoint, target_endpoint)) =
            self.pending_link_edit_request.take()
        {
            let editability = loaded_models.root_graph_editability(self.root_selection);
            match self.apply_link_drag_edit(
                loaded_models,
                working_path.as_slice(),
                source_endpoint,
                target_endpoint,
                editability,
            ) {
                Ok(result) => {
                    let LinkEditApplyResult {
                        status,
                        link_global_id,
                        requires_layout_refresh,
                        history_entry,
                    } = result;
                    self.actions_status = Some(status.clone());
                    if let Some(history_entry) = history_entry {
                        self.push_history_entry(history_entry);
                    }

                    let mut should_relayout = requires_layout_refresh;
                    let mut relayout_reason = if requires_layout_refresh {
                        Some("graph I/O endpoint identity changed".to_string())
                    } else {
                        None
                    };
                    if !should_relayout {
                        match self.graph_layouts.get_mut(&working_path) {
                            Some(Ok(graph_layout)) => {
                                if let Err(err) = Self::apply_link_drag_edit_to_layout(
                                    graph_layout,
                                    source_endpoint,
                                    target_endpoint,
                                    link_global_id,
                                ) {
                                    should_relayout = true;
                                    relayout_reason = Some(err);
                                }
                            }
                            Some(Err(err)) => {
                                should_relayout = true;
                                relayout_reason = Some(format!(
                                    "Cached layout was already invalid before edit: {err}"
                                ));
                            }
                            None => {
                                should_relayout = true;
                            }
                        }
                    }

                    if should_relayout {
                        self.graph_layouts.remove(&working_path);
                        self.model_view_scene_rects.remove(&working_path);
                        if let Some(reason) = relayout_reason {
                            self.actions_status =
                                Some(format!("{} (layout refresh fallback: {reason})", status));
                        }
                    }
                }
                Err(err) => {
                    let full_error = format!("Link edit failed: {err}");
                    self.actions_status = Some(full_error.clone());
                    self.error_popup = Some(full_error);
                }
            }
        }

        if let Some(history_action) = self.pending_history_action.take() {
            match history_action {
                PendingHistoryAction::Undo => {
                    if let Some(entry) = self.undo_history.pop() {
                        match self.restore_root_graph_snapshot(loaded_models, &entry.before) {
                            Ok(()) => {
                                self.invalidate_graph_view_cache();
                                self.actions_status = Some(format!("Undo: {}", entry.label));
                                self.redo_history.push(entry);
                                if self.redo_history.len() > MAX_GRAPH_UNDO_HISTORY {
                                    self.redo_history.drain(..1);
                                }
                            }
                            Err(err) => {
                                self.undo_history.push(entry);
                                let full_error = format!("Undo failed: {err}");
                                self.actions_status = Some(full_error.clone());
                                self.error_popup = Some(full_error);
                            }
                        }
                    } else {
                        self.actions_status = Some("Nothing to undo.".to_string());
                    }
                }
                PendingHistoryAction::Redo => {
                    if let Some(entry) = self.redo_history.pop() {
                        match self.restore_root_graph_snapshot(loaded_models, &entry.after) {
                            Ok(()) => {
                                self.invalidate_graph_view_cache();
                                self.actions_status = Some(format!("Redo: {}", entry.label));
                                self.undo_history.push(entry);
                                if self.undo_history.len() > MAX_GRAPH_UNDO_HISTORY {
                                    self.undo_history.drain(..1);
                                }
                            }
                            Err(err) => {
                                self.redo_history.push(entry);
                                let full_error = format!("Redo failed: {err}");
                                self.actions_status = Some(full_error.clone());
                                self.error_popup = Some(full_error);
                            }
                        }
                    } else {
                        self.actions_status = Some("Nothing to redo.".to_string());
                    }
                }
            }
        }

        let do_interface_panel = matches!(
            self.root_selection,
            GraphRootSubjectSelection::ServerInterface(_)
        );
        let available_height = ui.available_size_before_wrap().y;
        let interface_panel_height = if do_interface_panel {
            // Keep graph view dominant while still reserving enough room for controls.
            150.0f32.min((available_height - 120.0).max(80.0))
        } else {
            0.0
        };
        let mut graph_pane_rect: Option<Rect> = None;
        let mut graph_breadcrumb_items: Option<Vec<(String, Vec<GlobalId>)>> = None;

        // Find the graph we are working with
        let root_graph_resolution = loaded_models.resolve_root_graph(self.root_selection);
        let root_graph = root_graph_resolution.as_ref().map(|x| x.graph);
        let root_ownership = root_graph_resolution
            .as_ref()
            .map(|x| x.ownership)
            .unwrap_or(match self.root_selection {
                GraphRootSubjectSelection::ServerModel(_)
                | GraphRootSubjectSelection::ServerInterface(_) => GraphRootOwnership::Server,
                GraphRootSubjectSelection::ClientGraph(_) => GraphRootOwnership::Client,
            });
        let root_editability = root_graph_resolution
            .as_ref()
            .map(|x| x.editability)
            .unwrap_or_else(|| loaded_models.root_graph_editability(self.root_selection));
        if let GraphRootSubjectSelection::ServerModel(model_id) = self.root_selection
            && root_graph.is_none()
        {
            models_to_load.insert(model_id);
        }

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
        let export_actions_graph_and_path = graph_and_path
            .as_ref()
            .map(|(graph, path)| (*graph, path.clone()));
        if let Some((working_graph, working_path)) = graph_and_path {
            if let Some(root_graph) = root_graph {
                graph_breadcrumb_items = Some(self.build_graph_breadcrumb_items(
                    root_graph,
                    working_path.as_slice(),
                    loaded_models,
                ));
            }
            if !self.graph_layouts.contains_key(&self.graph_subject_path) {
                // Map tensors to link IDs
                let mut tensor_link_ids = HashMap::new();
                let mut link_data = HashMap::new();
                let mut next_link_id = 0usize;
                for (i, link_global_id) in working_graph.inner_link_ids().enumerate() {
                    let new_link_id = GraphLayoutLinkId(i);
                    tensor_link_ids.insert(link_global_id, new_link_id);
                    link_data.insert(
                        new_link_id,
                        GraphLayoutLinkData {
                            global_id: link_global_id,
                        },
                    );
                    next_link_id = i + 1;
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
                        for maybe_tensor_id in node.input_slots() {
                            inputs.push(maybe_tensor_id.map(|tensor_id| {
                                ensure_layout_link_id(
                                    tensor_id,
                                    &mut tensor_link_ids,
                                    &mut link_data,
                                    &mut next_link_id,
                                )
                            }));
                        }
                        let mut outputs = vec![];
                        for maybe_tensor_id in node.output_slots() {
                            outputs.push(maybe_tensor_id.map(|tensor_id| {
                                let link_id = ensure_layout_link_id(
                                    tensor_id,
                                    &mut tensor_link_ids,
                                    &mut link_data,
                                    &mut next_link_id,
                                );
                                sourced_links.insert(link_id);
                                link_id
                            }));
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
                    let link_id = ensure_layout_link_id(
                        inner_id,
                        &mut tensor_link_ids,
                        &mut link_data,
                        &mut next_link_id,
                    );
                    let node_id = GraphLayoutNodeId(next_node_id);
                    io_tensor_node_ids.insert(inner_id, node_id);
                    next_node_id += 1;
                    sourced_links.insert(link_id);
                    node_init_data.insert(
                        node_id,
                        GraphLayoutNodeInitData {
                            node_type: GraphLayoutNodeType::InputLinkNode(inner_id),
                            inputs: vec![],
                            outputs: vec![Some(link_id)],
                        },
                    );
                }
                for (_outer_id, inner_id) in working_graph.output_link_ids() {
                    let link_id = ensure_layout_link_id(
                        inner_id,
                        &mut tensor_link_ids,
                        &mut link_data,
                        &mut next_link_id,
                    );
                    let node_id = GraphLayoutNodeId(next_node_id);
                    io_tensor_node_ids.insert(inner_id, node_id);
                    next_node_id += 1;

                    node_init_data.insert(
                        node_id,
                        GraphLayoutNodeInitData {
                            node_type: GraphLayoutNodeType::OutputLinkNode(inner_id),
                            inputs: vec![Some(link_id)],
                            outputs: vec![],
                        },
                    );
                }
                for inner_id in working_graph.constant_link_ids() {
                    let link_id = ensure_layout_link_id(
                        inner_id,
                        &mut tensor_link_ids,
                        &mut link_data,
                        &mut next_link_id,
                    );
                    let node_id = GraphLayoutNodeId(next_node_id);
                    io_tensor_node_ids.insert(inner_id, node_id);
                    next_node_id += 1;
                    sourced_links.insert(link_id);
                    node_init_data.insert(
                        node_id,
                        GraphLayoutNodeInitData {
                            node_type: GraphLayoutNodeType::ConstantLinkNode(inner_id),
                            inputs: vec![],
                            outputs: vec![Some(link_id)],
                        },
                    );
                }
                for (global_id, link_id) in
                    tensor_link_ids.iter().filter_map(|(global_id, link_id)| {
                        if !sourced_links.contains(link_id) {
                            Some((global_id, link_id))
                        } else {
                            None
                        }
                    })
                {
                    let node_id = GraphLayoutNodeId(next_node_id);
                    io_tensor_node_ids.insert(*global_id, node_id);
                    next_node_id += 1;
                    node_init_data.insert(
                        node_id,
                        GraphLayoutNodeInitData {
                            node_type: GraphLayoutNodeType::InputLinkNode(*global_id),
                            inputs: vec![],
                            outputs: vec![Some(*link_id)],
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

            let mut pending_link_edit_request: Option<(SlotPipEndpoint, SlotPipEndpoint)> = None;
            match self.graph_layouts.get_mut(&self.graph_subject_path) {
                Some(Ok(graph_layout)) => {
                    let working_is_super =
                        <dyn Any>::downcast_ref::<SuperGraph>(working_graph.as_any()).is_some();
                    let working_is_milli =
                        <dyn Any>::downcast_ref::<MilliOpGraph>(working_graph.as_any()).is_some();
                    let root_is_super = root_graph.is_some_and(|graph| {
                        <dyn Any>::downcast_ref::<SuperGraph>(graph.as_any()).is_some()
                    });
                    let root_is_milli = root_graph.is_some_and(|graph| {
                        <dyn Any>::downcast_ref::<MilliOpGraph>(graph.as_any()).is_some()
                    });
                    let can_edit_links = root_editability.can_edit()
                        && ((root_is_super && working_is_super)
                            || (root_is_milli && working_is_milli));
                    if !can_edit_links {
                        self.pending_link_drag = None;
                    }
                    if let Some(link_drag) = &self.pending_link_drag
                        && link_drag.graph_path != working_path
                    {
                        self.pending_link_drag = None;
                    }

                    // Update positions
                    if state.explorer_physics && graph_layout.update_layout(5000) {
                        ui.ctx().request_repaint_after(Duration::from_millis(20));
                    }
                    let mut frame_shape = ui.available_size_before_wrap();
                    if do_interface_panel {
                        frame_shape.y -= interface_panel_height;
                    }
                    let frame_base = ui.available_rect_before_wrap().min;
                    graph_pane_rect =
                        Some(Rect::from_min_max(frame_base, frame_base + frame_shape));
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
                                let mut hovered_slot_endpoint: Option<SlotPipEndpoint> = None;
                                let mut nodes_with_active_slot_drag = HashSet::new();
                                let link_data = graph_layout.get_link_data().clone();

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
                                        duration_since_node_activity(
                                            &self.node_execution_timestamps,
                                            &self.node_last_child_active_timestamps,
                                            node_path,
                                            current_time,
                                        )
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

                                    let slot_owner = match current_node_data[&node_id].node_type {
                                        GraphLayoutNodeType::GraphNode(graph_node_id) => {
                                            Some(SlotPipOwner::Node(graph_node_id))
                                        }
                                        GraphLayoutNodeType::InputLinkNode(link_id) => {
                                            Some(SlotPipOwner::InputLink(link_id))
                                        }
                                        GraphLayoutNodeType::OutputLinkNode(link_id) => {
                                            Some(SlotPipOwner::OutputLink(link_id))
                                        }
                                        GraphLayoutNodeType::ConstantLinkNode(link_id) => {
                                            Some(SlotPipOwner::ConstantLink(link_id))
                                        }
                                        GraphLayoutNodeType::ConnectionByNameSrc(_)
                                        | GraphLayoutNodeType::ConnectionByNameDest(_) => None,
                                    };
                                    if can_edit_links
                                        && let Some(slot_owner) = slot_owner
                                    {
                                        let node_center = node_bounding_boxes[&node_id].center();
                                        let slot_pip_radius = 4.0f32;
                                        let slot_pip_rect_radius = 8.0f32;
                                        let interaction_sense = Sense::click_and_drag();

                                        for (slot_index, offset) in
                                            node_io_connections[&node_id].inputs.iter().enumerate()
                                        {
                                            let center = node_center + *offset;
                                            let maybe_layout_link = current_node_data[&node_id]
                                                .inputs
                                                .get(slot_index)
                                                .copied()
                                                .flatten();
                                            let maybe_link = maybe_layout_link.and_then(|layout_link| {
                                                link_data.get(&layout_link).map(|x| x.global_id)
                                            });
                                            let endpoint = SlotPipEndpoint {
                                                owner: slot_owner,
                                                direction: SlotDirection::Input,
                                                slot_index,
                                                link_id: maybe_link,
                                                layout_link_id: maybe_layout_link,
                                                screen_pos: center,
                                            };

                                            let pip_rect = Rect::from_center_size(
                                                center,
                                                Vec2::splat(slot_pip_rect_radius * 2.0),
                                            );
                                            let pip_resp = ui.interact(
                                                pip_rect,
                                                ui.id().with((
                                                    "slot_pip",
                                                    global_id,
                                                    0u8,
                                                    slot_index,
                                                )),
                                                interaction_sense,
                                            );

                                            if pip_resp.drag_started() {
                                                nodes_with_active_slot_drag.insert(node_id);
                                                self.pending_link_drag = Some(PendingLinkDrag {
                                                    graph_path: working_path.clone(),
                                                    source: endpoint,
                                                });
                                            }
                                            if pip_resp.dragged() {
                                                nodes_with_active_slot_drag.insert(node_id);
                                            }
                                            if let Some(drag) = &self.pending_link_drag
                                                && drag.graph_path == working_path
                                                && drag.source.direction != endpoint.direction
                                                && pip_resp.hovered()
                                            {
                                                hovered_slot_endpoint = Some(endpoint);
                                            }

                                            let is_drag_source = self
                                                .pending_link_drag
                                                .as_ref()
                                                .is_some_and(|drag| {
                                                    drag.graph_path == working_path
                                                        && drag.source.owner == endpoint.owner
                                                        && drag.source.direction
                                                            == endpoint.direction
                                                        && drag.source.slot_index
                                                            == endpoint.slot_index
                                                });
                                            let fill_color = if is_drag_source {
                                                Color32::from_rgb(255, 197, 48)
                                            } else if maybe_link.is_some() {
                                                Color32::from_rgb(98, 190, 250)
                                            } else {
                                                Color32::from_rgb(76, 84, 97)
                                            };
                                            let stroke_color = if pip_resp.hovered() {
                                                Color32::from_rgb(236, 244, 255)
                                            } else {
                                                Color32::from_gray(12)
                                            };
                                            ui.painter().circle_filled(
                                                center,
                                                slot_pip_radius,
                                                fill_color,
                                            );
                                            ui.painter().circle_stroke(
                                                center,
                                                slot_pip_radius,
                                                Stroke::new(1.0, stroke_color),
                                            );
                                        }

                                        for (slot_index, offset) in
                                            node_io_connections[&node_id].outputs.iter().enumerate()
                                        {
                                            let center = node_center + *offset;
                                            let maybe_layout_link = current_node_data[&node_id]
                                                .outputs
                                                .get(slot_index)
                                                .copied()
                                                .flatten();
                                            let maybe_link = maybe_layout_link.and_then(|layout_link| {
                                                link_data.get(&layout_link).map(|x| x.global_id)
                                            });
                                            let endpoint = SlotPipEndpoint {
                                                owner: slot_owner,
                                                direction: SlotDirection::Output,
                                                slot_index,
                                                link_id: maybe_link,
                                                layout_link_id: maybe_layout_link,
                                                screen_pos: center,
                                            };

                                            let pip_rect = Rect::from_center_size(
                                                center,
                                                Vec2::splat(slot_pip_rect_radius * 2.0),
                                            );
                                            let pip_resp = ui.interact(
                                                pip_rect,
                                                ui.id().with((
                                                    "slot_pip",
                                                    global_id,
                                                    1u8,
                                                    slot_index,
                                                )),
                                                interaction_sense,
                                            );

                                            if pip_resp.drag_started() {
                                                nodes_with_active_slot_drag.insert(node_id);
                                                self.pending_link_drag = Some(PendingLinkDrag {
                                                    graph_path: working_path.clone(),
                                                    source: endpoint,
                                                });
                                            }
                                            if pip_resp.dragged() {
                                                nodes_with_active_slot_drag.insert(node_id);
                                            }
                                            if let Some(drag) = &self.pending_link_drag
                                                && drag.graph_path == working_path
                                                && drag.source.direction != endpoint.direction
                                                && pip_resp.hovered()
                                            {
                                                hovered_slot_endpoint = Some(endpoint);
                                            }

                                            let is_drag_source = self
                                                .pending_link_drag
                                                .as_ref()
                                                .is_some_and(|drag| {
                                                    drag.graph_path == working_path
                                                        && drag.source.owner == endpoint.owner
                                                        && drag.source.direction
                                                            == endpoint.direction
                                                        && drag.source.slot_index
                                                            == endpoint.slot_index
                                                });
                                            let fill_color = if is_drag_source {
                                                Color32::from_rgb(255, 197, 48)
                                            } else if maybe_link.is_some() {
                                                Color32::from_rgb(95, 219, 140)
                                            } else {
                                                Color32::from_rgb(76, 84, 97)
                                            };
                                            let stroke_color = if pip_resp.hovered() {
                                                Color32::from_rgb(236, 244, 255)
                                            } else {
                                                Color32::from_gray(12)
                                            };
                                            ui.painter().circle_filled(
                                                center,
                                                slot_pip_radius,
                                                fill_color,
                                            );
                                            ui.painter().circle_stroke(
                                                center,
                                                slot_pip_radius,
                                                Stroke::new(1.0, stroke_color),
                                            );
                                        }
                                    }

                                    if resp.dragged()
                                        && !nodes_with_active_slot_drag.contains(&node_id)
                                    {
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

                                if let Some(link_drag) = &self.pending_link_drag
                                    && link_drag.graph_path == working_path
                                {
                                    if let Some(pointer_pos_global) =
                                        ui.input(|x| x.pointer.interact_pos())
                                    {
                                        let pointer_pos = ui
                                            .ctx()
                                            .layer_transform_from_global(ui.layer_id())
                                            .map_or(pointer_pos_global, |from_global| {
                                                from_global * pointer_pos_global
                                            });
                                        let points = [
                                            link_drag.source.screen_pos,
                                            egui::pos2(
                                                link_drag.source.screen_pos.x + 40.0,
                                                link_drag.source.screen_pos.y,
                                            ),
                                            egui::pos2(pointer_pos.x - 40.0, pointer_pos.y),
                                            pointer_pos,
                                        ];
                                        let preview_stroke = Stroke {
                                            width: 2.0,
                                            color: Color32::from_rgb(255, 197, 48),
                                        };
                                        ui.painter().add(CubicBezierShape::from_points_stroke(
                                            points,
                                            false,
                                            Color32::TRANSPARENT,
                                            preview_stroke,
                                        ));
                                    }

                                    if ui.input(|x| x.pointer.primary_released()) {
                                        if let Some(target_endpoint) = hovered_slot_endpoint {
                                            pending_link_edit_request =
                                                Some((link_drag.source, target_endpoint));
                                        }
                                        self.pending_link_drag = None;
                                    } else {
                                        ui.ctx().request_repaint_after(Duration::from_millis(20));
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

            if let Some((source_endpoint, target_endpoint)) = pending_link_edit_request.take() {
                self.pending_link_edit_request =
                    Some((working_path.clone(), source_endpoint, target_endpoint));
            }
        } else {
            ui.label("No graph selected");
        }
        if let GraphRootSubjectSelection::ServerInterface(interface_id) = self.root_selection
            && let Some(interface) = loaded_models
                .server_graphs
                .current_interfaces
                .get(&interface_id)
        {
            egui::ScrollArea::vertical()
                .id_salt(("graph_interface_panel", interface_id))
                .max_height(interface_panel_height)
                .auto_shrink([false, false])
                .show(ui, |ui| {
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
                                            let swatch_settings = if state
                                                .do_explorer_swatches_in_view
                                                || state.do_all_explorer_swatches
                                            {
                                                Some(AbbreviatedTensorReportSettings {
                                                    downsampled_size: (state.swatch_dimension
                                                        * state.swatch_dimension)
                                                        as u64,
                                                    subscribed_tensors: self
                                                        .tensors_in_view
                                                        .iter()
                                                        .cloned()
                                                        .collect(),
                                                    do_all: state.do_all_explorer_swatches,
                                                })
                                            } else {
                                                None
                                            };
                                            server_request_manager.update_observer_settings(
                                                *request_id,
                                                self.inspect_window_tensor_subscriptions
                                                    .iter()
                                                    .cloned()
                                                    .collect(),
                                                state.explorer_node_wave,
                                                swatch_settings,
                                            );
                                            if let Some(reports) = server_request_manager.get_reports(*request_id) {
                                                let time_now = Instant::now();
                                                let mut node_activity_maps = NodeExecutionActivityMapsMut {
                                                    node_execution_timestamps: &mut self.node_execution_timestamps,
                                                    node_last_child_active_timestamps: &mut self.node_last_child_active_timestamps,
                                                    node_execution_durations: &mut self.node_execution_durations,
                                                    node_execution_op_kinds: &mut self.node_execution_op_kinds,
                                                };
                                                for report in reports {
                                                    text_inference_data
                                                        .progress_widget_state
                                                        .ingest_report(report.clone());
                                                    for (path, value) in report.tensor_assignments {
                                                        self.inspect_window_tensor_subscription_returns.insert(path, value);
                                                    }
                                                    for (node_path, op_kind, age, execution_duration) in report.node_executions {
                                                        let time = time_now - age;
                                                        record_node_execution_activity(
                                                            &mut node_activity_maps,
                                                            node_path,
                                                            op_kind,
                                                            time,
                                                            execution_duration,
                                                        );
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
                                                        TokenizerInfo::HFTokenizerJson(_) => {
                                                            "GGUF embedded".to_string()
                                                        }
                                                    };
                                                    ui.label(format!("Tokenizer: {v}"));
                                                }
                                                if let Some(request_id) = text_inference_data
                                                    .pending_request
                                                    .as_ref()
                                                    .map(|x| x.0)
                                                {
                                                    ui.horizontal(|ui| {
                                                        ui.spinner();
                                                        ui.label("Running...");
                                                        if ui.button("Cancel").clicked() {
                                                            server_request_manager
                                                                .cancel_request(request_id);
                                                            text_inference_data.pending_request =
                                                                None;
                                                            text_inference_data
                                                                .progress_widget_state
                                                                .clear();
                                                        }
                                                    });
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
                                                        text_inference_data.progress_widget_state.clear();
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
                                                                    audio_inputs: HashMap::new(),
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
                    AnyInterface::MultimodalLanguageInterface(mm_interface) => {
                        ui.label("Multimodal language interface");
                        ui.label(format!(
                            "Modality input slots: {}",
                            mm_interface.modality_inputs.len()
                        ));
                        let required = mm_interface
                            .modality_inputs
                            .iter()
                            .filter(|x| x.required)
                            .map(|x| x.name.clone())
                            .collect::<Vec<_>>();
                        if !required.is_empty() {
                            ui.label(format!(
                                "Required modalities are not yet editable in this panel: {}",
                                required.join(", ")
                            ));
                        }
                    }
                    AnyInterface::ImageGenerationInterface(sd_interface) => {
                        let sd_data = self
                            .sd_inference_data
                            .entry(interface_id)
                            .or_default();

                        // Handle pending reports (node wave, tensor swatches)
                        if let Some((request_id, _)) = &sd_data.pending_request {
                            let swatch_settings =
                                if state.do_explorer_swatches_in_view || state.do_all_explorer_swatches {
                                    Some(AbbreviatedTensorReportSettings {
                                        downsampled_size: (state.swatch_dimension
                                            * state.swatch_dimension)
                                            as u64,
                                        subscribed_tensors: self
                                            .tensors_in_view
                                            .iter()
                                            .cloned()
                                            .collect(),
                                        do_all: state.do_all_explorer_swatches,
                                    })
                                } else {
                                    None
                                };
                            server_request_manager.update_observer_settings(
                                *request_id,
                                self.inspect_window_tensor_subscriptions
                                    .iter()
                                    .cloned()
                                    .collect(),
                                state.explorer_node_wave,
                                swatch_settings,
                            );
                            if let Some(reports) = server_request_manager.get_reports(*request_id) {
                                let time_now = Instant::now();
                                let mut node_activity_maps = NodeExecutionActivityMapsMut {
                                    node_execution_timestamps: &mut self.node_execution_timestamps,
                                    node_last_child_active_timestamps: &mut self.node_last_child_active_timestamps,
                                    node_execution_durations: &mut self.node_execution_durations,
                                    node_execution_op_kinds: &mut self.node_execution_op_kinds,
                                };
                                for report in reports {
                                    sd_data.progress_widget_state.ingest_report(report.clone());
                                    for (path, value) in report.tensor_assignments {
                                        self.inspect_window_tensor_subscription_returns.insert(path, value);
                                    }
                                    for (node_path, op_kind, age, execution_duration) in report.node_executions {
                                        let time = time_now - age;
                                        record_node_execution_activity(
                                            &mut node_activity_maps,
                                            node_path,
                                            op_kind,
                                            time,
                                            execution_duration,
                                        );
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
                        ui.columns(2, |columns| {
                            let (left, right) = columns.split_at_mut(1);
                            let controls_ui = &mut left[0];
                            let results_ui = &mut right[0];

                            let frame = egui::Frame::default()
                                .stroke(controls_ui.visuals().window_stroke)
                                .inner_margin(5.0);
                            frame.show(controls_ui, |ui| {
                                ui.vertical(|ui| {
                                    ui.heading("Stable Diffusion");

                                    ui.horizontal(|ui| {
                                        ui.label("Prompt:");
                                        ui.text_edit_singleline(&mut sd_data.prompt);
                                    });

                                    ui.horizontal(|ui| {
                                        ui.label("Steps:");
                                        ui.add(
                                            egui::DragValue::new(&mut sd_data.num_steps)
                                                .range(1..=100),
                                        );
                                        ui.label("Guidance:");
                                        ui.add(
                                            egui::DragValue::new(&mut sd_data.guidance_scale)
                                                .speed(0.1)
                                                .range(1.0..=30.0),
                                        );
                                        ui.label("Seed:");
                                        ui.add(egui::DragValue::new(&mut sd_data.seed));
                                    });

                                    ui.horizontal(|ui| {
                                        ui.label("Latent H:");
                                        ui.add(
                                            egui::DragValue::new(&mut sd_data.latent_h)
                                                .range(4..=128),
                                        );
                                        ui.label("Latent W:");
                                        ui.add(
                                            egui::DragValue::new(&mut sd_data.latent_w)
                                                .range(4..=128),
                                        );
                                        ui.label(format!(
                                            "({}x{} px)",
                                            sd_data.latent_w * 8,
                                            sd_data.latent_h * 8
                                        ));
                                    });

                                    if let Some(request_id) =
                                        sd_data.pending_request.as_ref().map(|x| x.0)
                                    {
                                        ui.horizontal(|ui| {
                                            ui.spinner();
                                            ui.label("Generating...");
                                            if ui.button("Cancel").clicked() {
                                                server_request_manager.cancel_request(request_id);
                                                sd_data.pending_request = None;
                                                sd_data.progress_widget_state.clear();
                                                sd_data.status_message =
                                                    Some("Cancelled".to_string());
                                            }
                                        });
                                    } else {
                                        ui.horizontal(|ui| {
                                            egui::ComboBox::from_id_salt("sd_backend_mode")
                                                .selected_text(sd_data.selected_mode.to_string())
                                                .show_ui(ui, |ui| {
                                                    ui.selectable_value(
                                                        &mut sd_data.selected_mode,
                                                        SuperGraphRequestBackendMode::NDArray,
                                                        SuperGraphRequestBackendMode::NDArray
                                                            .to_string(),
                                                    );
                                                    if server_config_report.vulkan_available {
                                                        ui.selectable_value(
                                                            &mut sd_data.selected_mode,
                                                            SuperGraphRequestBackendMode::Vulkan,
                                                            SuperGraphRequestBackendMode::Vulkan
                                                                .to_string(),
                                                        );
                                                    }
                                                    ui.selectable_value(
                                                        &mut sd_data.selected_mode,
                                                        SuperGraphRequestBackendMode::Compiler,
                                                        SuperGraphRequestBackendMode::Compiler
                                                            .to_string(),
                                                    );
                                                });
                                            toggle_ui(ui, &mut sd_data.use_cache);
                                            ui.label("Cache");
                                            if ui.button("Generate").clicked() {
                                                sd_data.progress_widget_state.clear();
                                                let mut tensor_inputs = HashMap::new();
                                                let mut string_inputs = HashMap::new();
                                                string_inputs.insert(
                                                    sd_interface.positive_prompt_input,
                                                    sd_data.prompt.clone(),
                                                );
                                                if let Some(negative_link) =
                                                    sd_interface.negative_prompt_input
                                                {
                                                    string_inputs
                                                        .insert(negative_link, String::new());
                                                }

                                                let channels = sd_interface.latent_channels;
                                                let (
                                                    timestep_values,
                                                    dt_values,
                                                    sigma_values,
                                                    initial_noise,
                                                ) = match &sd_interface.scheduler {
                                                    whisper_tensor::interfaces::SchedulerType::EulerDiscrete => {
                                                        let (ts, dt, sigmas, init_sigma) =
                                                            ImageGenerationInterface::compute_euler_schedule(sd_data.num_steps);
                                                        let latent_n = channels
                                                            * sd_data.latent_h
                                                            * sd_data.latent_w;
                                                        let noise =
                                                            generate_normal_noise(latent_n, sd_data.seed);
                                                        let scaled: Vec<f32> = noise
                                                            .iter()
                                                            .map(|&x| x * init_sigma)
                                                            .collect();
                                                        (ts, dt, sigmas, scaled)
                                                    }
                                                    whisper_tensor::interfaces::SchedulerType::RectifiedFlow => {
                                                        let (ts, dt, sigmas) =
                                                            ImageGenerationInterface::compute_flux_schedule(sd_data.num_steps);
                                                        let latent_n = channels
                                                            * sd_data.latent_h
                                                            * sd_data.latent_w;
                                                        let noise =
                                                            generate_normal_noise(latent_n, sd_data.seed);
                                                        (ts, dt, sigmas, noise)
                                                    }
                                                };

                                                let latent_tensor =
                                                    NDArrayNumericTensor::from_vec_shape(
                                                        initial_noise,
                                                        &vec![
                                                            1,
                                                            channels as u64,
                                                            sd_data.latent_h as u64,
                                                            sd_data.latent_w as u64,
                                                        ],
                                                    )
                                                    .unwrap();

                                                let timesteps_tensor =
                                                    NDArrayNumericTensor::from_vec_shape(
                                                        timestep_values,
                                                        &vec![sd_data.num_steps as u64],
                                                    )
                                                    .unwrap();
                                                let dt_tensor =
                                                    NDArrayNumericTensor::from_vec_shape(
                                                        dt_values,
                                                        &vec![sd_data.num_steps as u64],
                                                    )
                                                    .unwrap();
                                                let sigmas_tensor =
                                                    NDArrayNumericTensor::from_vec_shape(
                                                        sigma_values,
                                                        &vec![sd_data.num_steps as u64],
                                                    )
                                                    .unwrap();
                                                let iter_count =
                                                    NDArrayNumericTensor::from_vec_shape(
                                                        vec![sd_data.num_steps as i64],
                                                        &vec![1],
                                                    )
                                                    .unwrap();

                                                tensor_inputs.insert(
                                                    sd_interface.initial_latent_input,
                                                    latent_tensor,
                                                );
                                                tensor_inputs.insert(
                                                    sd_interface.timesteps_input,
                                                    timesteps_tensor,
                                                );
                                                tensor_inputs
                                                    .insert(sd_interface.dt_input, dt_tensor);
                                                tensor_inputs.insert(
                                                    sd_interface.sigmas_input,
                                                    sigmas_tensor,
                                                );
                                                tensor_inputs.insert(
                                                    sd_interface.iteration_count_input,
                                                    iter_count,
                                                );
                                                if let Some(gs_link) =
                                                    sd_interface.guidance_scale_input
                                                {
                                                    let guidance =
                                                        NDArrayNumericTensor::from_vec(vec![
                                                            sd_data.guidance_scale,
                                                        ])
                                                        .to_dyn();
                                                    tensor_inputs.insert(gs_link, guidance);
                                                }

                                                let symbolic_graph_ids: Vec<_> =
                                                    interface.model_ids.to_vec();
                                                let model_inputs: HashMap<_, _> = sd_interface
                                                    .model_weights
                                                    .iter()
                                                    .zip(interface.model_ids.iter())
                                                    .map(|(&link, &id)| (link, id))
                                                    .collect();

                                                let swatch_settings = if state
                                                    .do_explorer_swatches_in_view
                                                    || state.do_all_explorer_swatches
                                                {
                                                    Some(AbbreviatedTensorReportSettings {
                                                        downsampled_size: (state.swatch_dimension
                                                            * state.swatch_dimension)
                                                            as u64,
                                                        subscribed_tensors: self
                                                            .tensors_in_view
                                                            .iter()
                                                            .cloned()
                                                            .collect(),
                                                        do_all: state.do_all_explorer_swatches,
                                                    })
                                                } else {
                                                    None
                                                };

                                                let token = server_request_manager
                                                    .submit_supergraph_request(
                                                        SuperGraphRequest {
                                                            do_node_execution_reports: state
                                                                .explorer_node_wave,
                                                            abbreviated_tensor_report_settings:
                                                                swatch_settings,
                                                            attention_token: None,
                                                            super_graph: sd_interface
                                                                .super_graph
                                                                .clone(),
                                                            subscribed_tensors: self
                                                                .inspect_window_tensor_subscriptions
                                                                .iter()
                                                                .cloned()
                                                                .collect(),
                                                            string_inputs,
                                                            use_cache: if sd_data.use_cache {
                                                                Some(200 + interface_id as u64)
                                                            } else {
                                                                None
                                                            },
                                                            backend_mode: sd_data.selected_mode,
                                                            symbolic_graph_ids,
                                                            tensor_inputs,
                                                            audio_inputs: HashMap::new(),
                                                            model_inputs,
                                                            hash_inputs: HashMap::new(),
                                                        },
                                                    );

                                                sd_data.pending_request =
                                                    Some((token, sd_interface.image_output));
                                                sd_data.status_message =
                                                    Some("Running SD pipeline...".to_string());
                                            }
                                        });
                                    }
                                });
                            });

                            let frame = egui::Frame::default()
                                .stroke(results_ui.visuals().window_stroke)
                                .inner_margin(5.0);
                            frame.show(results_ui, |ui| {
                                ui.vertical(|ui| {
                                    ui.heading("Result");
                                    if let Some(msg) = &sd_data.status_message {
                                        ui.label(msg);
                                    } else {
                                        ui.label("No image generated yet.");
                                    }
                                    if let Some((texture, _color_image)) = &sd_data.generated_image
                                    {
                                        let size = texture.size_vec2();
                                        let thumb_max = 200.0;
                                        let scale = (thumb_max / size.x.max(size.y)).min(1.0);
                                        let display_size =
                                            egui::vec2(size.x * scale, size.y * scale);
                                        let response = ui.add(
                                            egui::Image::new(egui::load::SizedTexture::new(
                                                texture.id(),
                                                display_size,
                                            ))
                                            .sense(Sense::click()),
                                        );
                                        if response.clicked() {
                                            sd_data.show_image_window = !sd_data.show_image_window;
                                        }
                                        response.on_hover_text("Click to inspect");
                                    }
                                });
                            });
                        });

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
                    AnyInterface::TextToSpeechInterface(tts_interface) => {
                        let tts_data = self
                            .tts_inference_data
                            .entry(interface_id)
                            .or_default();

                        if let Some((request_id, _, _)) = &tts_data.pending_request {
                            let swatch_settings =
                                if state.do_explorer_swatches_in_view || state.do_all_explorer_swatches {
                                    Some(AbbreviatedTensorReportSettings {
                                        downsampled_size: (state.swatch_dimension
                                            * state.swatch_dimension)
                                            as u64,
                                        subscribed_tensors: self
                                            .tensors_in_view
                                            .iter()
                                            .cloned()
                                            .collect(),
                                        do_all: state.do_all_explorer_swatches,
                                    })
                                } else {
                                    None
                                };
                            server_request_manager.update_observer_settings(
                                *request_id,
                                self.inspect_window_tensor_subscriptions
                                    .iter()
                                    .cloned()
                                    .collect(),
                                state.explorer_node_wave,
                                swatch_settings,
                            );
                            if let Some(reports) = server_request_manager.get_reports(*request_id) {
                                let time_now = Instant::now();
                                let mut node_activity_maps = NodeExecutionActivityMapsMut {
                                    node_execution_timestamps: &mut self.node_execution_timestamps,
                                    node_last_child_active_timestamps: &mut self.node_last_child_active_timestamps,
                                    node_execution_durations: &mut self.node_execution_durations,
                                    node_execution_op_kinds: &mut self.node_execution_op_kinds,
                                };
                                for report in reports {
                                    tts_data.progress_widget_state.ingest_report(report.clone());
                                    for (path, value) in report.tensor_assignments {
                                        self.inspect_window_tensor_subscription_returns.insert(path, value);
                                    }
                                    for (node_path, op_kind, age, execution_duration) in report.node_executions {
                                        let time = time_now - age;
                                        record_node_execution_activity(
                                            &mut node_activity_maps,
                                            node_path,
                                            op_kind,
                                            time,
                                            execution_duration,
                                        );
                                    }
                                    for (tensor_path, value) in report.abbreviated_tensor_assignments {
                                        self.rendered_tensor_swatches.remove(&tensor_path);
                                        self.abbreviated_tensor_reports.insert(tensor_path, value);
                                    }
                                }
                            }
                            if let Some(response) = server_request_manager.get_response(*request_id) {
                                let (_, output_link, sample_rate_hz) = tts_data.pending_request.take().unwrap();
                                match response.result {
                                    Ok(mut data) => {
                                        if let Some(audio_tensor) = data.tensor_outputs.remove(&output_link) {
                                            let sample_count =
                                                audio_tensor.shape().iter().copied().product::<u64>() as usize;
                                            let duration_s = sample_count as f64 / sample_rate_hz as f64;
                                            tts_data.status_message = Some(format!(
                                                "Generated audio: {sample_count} samples ({duration_s:.2}s @ {sample_rate_hz}Hz)"
                                            ));
                                            tts_data.generated_audio = Some(audio_tensor);
                                            tts_data.generated_sample_rate_hz =
                                                Some(sample_rate_hz);
                                        } else {
                                            tts_data.status_message =
                                                Some("Error: output audio not found".to_string());
                                            tts_data.generated_sample_rate_hz = None;
                                        }
                                    }
                                    Err(err) => {
                                        self.error_popup = Some(err);
                                        tts_data.status_message = Some("Error (see popup)".to_string());
                                        tts_data.generated_sample_rate_hz = None;
                                    }
                                }
                            }
                        }

                        ui.columns(2, |columns| {
                            let (left, right) = columns.split_at_mut(1);
                            let controls_ui = &mut left[0];
                            let results_ui = &mut right[0];

                            let frame = egui::Frame::default()
                                .stroke(controls_ui.visuals().window_stroke)
                                .inner_margin(5.0);
                            frame.show(controls_ui, |ui| {
                                ui.vertical(|ui| {
                                    ui.heading("Text to Speech");

                                    ui.horizontal(|ui| {
                                        ui.label("Text:");
                                        ui.text_edit_singleline(&mut tts_data.text);
                                    });
                                    ui.horizontal(|ui| {
                                        ui.label("Speed:");
                                        ui.add(
                                            egui::DragValue::new(&mut tts_data.speed)
                                                .speed(0.05)
                                                .range(0.1..=4.0),
                                        );
                                    });
                                    if let TTSInputConfig::Kokoro {
                                        voices,
                                        default_voice,
                                        ..
                                    } = &tts_interface.input_config
                                    {
                                        ensure_kokoro_voice_selection(
                                            &mut tts_data.kokoro_voice_name,
                                            voices,
                                            default_voice.as_deref(),
                                        );
                                        if !voices.is_empty() {
                                            let mut selected =
                                                tts_data.kokoro_voice_name.clone().unwrap();
                                            egui::ComboBox::from_id_salt((
                                                "graph_tts_kokoro_voice",
                                                interface_id,
                                            ))
                                            .selected_text(selected.clone())
                                            .show_ui(ui, |ui| {
                                                for voice in voices {
                                                    ui.selectable_value(
                                                        &mut selected,
                                                        voice.name.clone(),
                                                        voice.name.as_str(),
                                                    );
                                                }
                                            });
                                            tts_data.kokoro_voice_name = Some(selected);
                                        }
                                    }
                                    if let TTSInputConfig::Piper {
                                        speaker_id_link: Some(_),
                                        num_speakers,
                                        ..
                                    } = &tts_interface.input_config
                                    {
                                        ui.horizontal(|ui| {
                                            ui.label("Speaker ID:");
                                            let max_speaker =
                                                (*num_speakers as i64).saturating_sub(1).max(0);
                                            ui.add(
                                                egui::DragValue::new(&mut tts_data.piper_speaker_id)
                                                    .range(0..=max_speaker),
                                            );
                                        });
                                    }

                                    if let Some(request_id) =
                                        tts_data.pending_request.as_ref().map(|x| x.0)
                                    {
                                        ui.horizontal(|ui| {
                                            ui.spinner();
                                            ui.label("Generating...");
                                            if ui.button("Cancel").clicked() {
                                                server_request_manager.cancel_request(request_id);
                                                tts_data.pending_request = None;
                                                tts_data.progress_widget_state.clear();
                                                tts_data.status_message =
                                                    Some("Cancelled".to_string());
                                            }
                                        });
                                    } else {
                                        ui.horizontal(|ui| {
                                            egui::ComboBox::from_id_salt((
                                                "tts_backend_mode",
                                                interface_id,
                                            ))
                                            .selected_text(tts_data.selected_mode.to_string())
                                            .show_ui(ui, |ui| {
                                                ui.selectable_value(
                                                    &mut tts_data.selected_mode,
                                                    SuperGraphRequestBackendMode::NDArray,
                                                    SuperGraphRequestBackendMode::NDArray.to_string(),
                                                );
                                                if server_config_report.vulkan_available {
                                                    ui.selectable_value(
                                                        &mut tts_data.selected_mode,
                                                        SuperGraphRequestBackendMode::Vulkan,
                                                        SuperGraphRequestBackendMode::Vulkan
                                                            .to_string(),
                                                    );
                                                }
                                                ui.selectable_value(
                                                    &mut tts_data.selected_mode,
                                                    SuperGraphRequestBackendMode::Compiler,
                                                    SuperGraphRequestBackendMode::Compiler
                                                        .to_string(),
                                                );
                                            });
                                            toggle_ui(ui, &mut tts_data.use_cache);
                                            ui.label("Cache");
                                            if ui.button("Generate").clicked() {
                                                tts_data.progress_widget_state.clear();
                                                let mut tensor_inputs = HashMap::new();
                                                let mut string_inputs = HashMap::new();
                                                string_inputs.insert(
                                                    tts_interface.text_input_link,
                                                    tts_data.text.clone(),
                                                );

                                                let mut should_submit = true;
                                                match &tts_interface.input_config {
                                                    TTSInputConfig::Kokoro {
                                                        style_link,
                                                        speed_link,
                                                        voices,
                                                        default_voice,
                                                    } => {
                                                        if let Some(voice) = selected_kokoro_voice(
                                                            &tts_data.kokoro_voice_name,
                                                            voices,
                                                            default_voice.as_deref(),
                                                        ) {
                                                            let approx_tokens = tts_data
                                                                .text
                                                                .chars()
                                                                .count()
                                                                .saturating_add(2);
                                                            match voice
                                                                .style_for_token_count(approx_tokens)
                                                            {
                                                                Ok(style_values) => {
                                                                    let style = NDArrayNumericTensor::<DynRank>::from_vec_shape(
                                                                        style_values,
                                                                        &vec![1, KokoroVoiceEmbedding::STYLE_DIM as u64],
                                                                    ).unwrap();
                                                                    let speed = NDArrayNumericTensor::<DynRank>::from_vec_shape(
                                                                        vec![tts_data.speed],
                                                                        &vec![1],
                                                                    ).unwrap();
                                                                    tensor_inputs
                                                                        .insert(*style_link, style);
                                                                    tensor_inputs
                                                                        .insert(*speed_link, speed);
                                                                }
                                                                Err(err) => {
                                                                    tts_data.status_message = Some(
                                                                        format!(
                                                                            "Failed to decode Kokoro voice '{}': {}",
                                                                            voice.name, err
                                                                        ),
                                                                    );
                                                                    should_submit = false;
                                                                }
                                                            }
                                                        } else {
                                                            tts_data.status_message = Some(
                                                                "no Kokoro voice embeddings available"
                                                                    .to_string(),
                                                            );
                                                            should_submit = false;
                                                        }
                                                    }
                                                    TTSInputConfig::Piper {
                                                        scales_link,
                                                        speaker_id_link,
                                                        ..
                                                    } => {
                                                        let length_scale =
                                                            1.0 / tts_data.speed.max(0.1);
                                                        let scales = NDArrayNumericTensor::<DynRank>::from_vec_shape(
                                                            vec![0.667f32, length_scale, 0.8],
                                                            &vec![3],
                                                        ).unwrap();
                                                        tensor_inputs.insert(*scales_link, scales);
                                                        if let Some(sid_link) = speaker_id_link {
                                                            let speaker_id = NDArrayNumericTensor::<DynRank>::from_vec_shape(
                                                                vec![tts_data.piper_speaker_id],
                                                                &vec![1],
                                                            ).unwrap();
                                                            tensor_inputs.insert(
                                                                *sid_link,
                                                                speaker_id,
                                                            );
                                                        }
                                                    }
                                                    TTSInputConfig::F5 { .. } => {
                                                        tts_data.status_message = Some(
                                                            "F5-TTS UI wiring for reference audio is not implemented yet"
                                                                .to_string(),
                                                        );
                                                        should_submit = false;
                                                    }
                                                }

                                                if should_submit {
                                                    let symbolic_graph_ids: Vec<_> =
                                                        interface.model_ids.to_vec();
                                                    let model_inputs: HashMap<_, _> = tts_interface
                                                        .model_weights
                                                        .iter()
                                                        .zip(interface.model_ids.iter())
                                                        .map(|(&link, &id)| (link, id))
                                                        .collect();

                                                    let swatch_settings = if state
                                                        .do_explorer_swatches_in_view
                                                        || state.do_all_explorer_swatches
                                                    {
                                                        Some(AbbreviatedTensorReportSettings {
                                                            downsampled_size: (state.swatch_dimension
                                                                * state.swatch_dimension)
                                                                as u64,
                                                            subscribed_tensors: self
                                                                .tensors_in_view
                                                                .iter()
                                                                .cloned()
                                                                .collect(),
                                                            do_all: state.do_all_explorer_swatches,
                                                        })
                                                    } else {
                                                        None
                                                    };

                                                    let token = server_request_manager
                                                        .submit_supergraph_request(
                                                            SuperGraphRequest {
                                                                do_node_execution_reports: state
                                                                    .explorer_node_wave,
                                                                abbreviated_tensor_report_settings:
                                                                    swatch_settings,
                                                                attention_token: None,
                                                                super_graph: tts_interface
                                                                    .super_graph
                                                                    .clone(),
                                                                subscribed_tensors: self
                                                                    .inspect_window_tensor_subscriptions
                                                                    .iter()
                                                                    .cloned()
                                                                    .collect(),
                                                                string_inputs,
                                                                use_cache: if tts_data.use_cache {
                                                                    Some(300 + interface_id as u64)
                                                                } else {
                                                                    None
                                                                },
                                                                backend_mode: tts_data.selected_mode,
                                                                symbolic_graph_ids,
                                                                tensor_inputs,
                                                                audio_inputs: HashMap::new(),
                                                                model_inputs,
                                                                hash_inputs: HashMap::new(),
                                                            },
                                                        );

                                                    tts_data.pending_request = Some((
                                                        token,
                                                        tts_interface.audio_output_link,
                                                        tts_interface.sample_rate,
                                                    ));
                                                    tts_data.status_message =
                                                        Some("Running TTS pipeline...".to_string());
                                                }
                                            }
                                        });
                                    }
                                });
                            });

                            let frame = egui::Frame::default()
                                .stroke(results_ui.visuals().window_stroke)
                                .inner_margin(5.0);
                            frame.show(results_ui, |ui| {
                                ui.vertical(|ui| {
                                    ui.heading("Result");
                                    if let Some(msg) = &tts_data.status_message {
                                        ui.label(msg);
                                    } else {
                                        ui.label("No audio generated yet.");
                                    }
                                    if let (Some(audio), Some(sample_rate_hz)) = (
                                        tts_data.generated_audio.clone(),
                                        tts_data.generated_sample_rate_hz,
                                    ) {
                                        ui.label(format!("Output tensor shape: {:?}", audio.shape()));
                                        let mut play_clicked = false;
                                        let mut stop_clicked = false;
                                        let mut download_clicked = false;
                                        ui.horizontal(|ui| {
                                            play_clicked = ui.button("Play").clicked();
                                            stop_clicked = ui.button("Stop").clicked();
                                            download_clicked = ui.button("Download WAV").clicked();
                                        });

                                        if play_clicked {
                                            let result = tensor_to_audio_samples(&audio).and_then(
                                                |samples| {
                                                    play_audio_samples(&samples, sample_rate_hz)
                                                },
                                            );
                                            match result {
                                                Ok(()) => {
                                                    tts_data.status_message = Some(format!(
                                                        "Playing audio @ {sample_rate_hz}Hz"
                                                    ));
                                                }
                                                Err(err) => {
                                                    tts_data.status_message = Some(format!(
                                                        "Audio playback failed: {err}"
                                                    ));
                                                }
                                            }
                                        }
                                        if stop_clicked {
                                            stop_audio_playback();
                                            tts_data.status_message =
                                                Some("Stopped playback".to_string());
                                        }
                                        if download_clicked {
                                            let result =
                                                tensor_to_audio_samples(&audio).and_then(|samples| {
                                                    download_audio_wav(&samples, sample_rate_hz)
                                                });
                                            match result {
                                                Ok(()) => {
                                                    tts_data.status_message =
                                                        Some("Saved generated_audio.wav".to_string());
                                                }
                                                Err(err) => {
                                                    tts_data.status_message = Some(format!(
                                                        "Audio download failed: {err}"
                                                    ));
                                                }
                                            }
                                        }
                                    } else if let Some(audio) = &tts_data.generated_audio {
                                        ui.label(format!("Output tensor shape: {:?}", audio.shape()));
                                    }
                                });
                            });
                        });
                    }
                        AnyInterface::SpeechToTextInterface(stt_interface) => {
                            let stt_data = self
                                .stt_inference_data
                                .entry(interface_id)
                                .or_default();

                            #[cfg(target_arch = "wasm32")]
                            if let Some(receiver) = stt_data.pending_web_audio_pick.as_mut() {
                                match receiver.try_recv() {
                                    Ok(Ok(file)) => {
                                        stt_data.selected_audio_name = Some(file.name.clone());
                                        stt_data.selected_audio_bytes = Some(file.bytes);
                                        stt_data.status_message =
                                            Some(format!("Loaded audio file: {}", file.name));
                                        stt_data.transcription_text = None;
                                        stt_data.transcription_tokens = None;
                                        stt_data.pending_web_audio_pick = None;
                                    }
                                    Ok(Err(err)) => {
                                        if err != "file selection canceled" {
                                            stt_data.status_message =
                                                Some(format!("Failed to load audio file: {err}"));
                                        }
                                        stt_data.pending_web_audio_pick = None;
                                    }
                                    Err(tokio::sync::mpsc::error::TryRecvError::Empty) => {}
                                    Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => {
                                        stt_data.pending_web_audio_pick = None;
                                    }
                                }
                            }

                            if let Some((request_id, _, _, _)) = &stt_data.pending_request {
                                let swatch_settings = if state.do_explorer_swatches_in_view
                                    || state.do_all_explorer_swatches
                                {
                                    Some(AbbreviatedTensorReportSettings {
                                        downsampled_size: (state.swatch_dimension
                                            * state.swatch_dimension)
                                            as u64,
                                        subscribed_tensors: self
                                            .tensors_in_view
                                            .iter()
                                            .cloned()
                                            .collect(),
                                        do_all: state.do_all_explorer_swatches,
                                    })
                                } else {
                                    None
                                };
                                server_request_manager.update_observer_settings(
                                    *request_id,
                                    self.inspect_window_tensor_subscriptions
                                        .iter()
                                        .cloned()
                                        .collect(),
                                    state.explorer_node_wave,
                                    swatch_settings,
                                );
                                if let Some(reports) = server_request_manager.get_reports(*request_id)
                                {
                                    let time_now = Instant::now();
                                    let mut node_activity_maps = NodeExecutionActivityMapsMut {
                                        node_execution_timestamps: &mut self.node_execution_timestamps,
                                        node_last_child_active_timestamps: &mut self.node_last_child_active_timestamps,
                                        node_execution_durations: &mut self.node_execution_durations,
                                        node_execution_op_kinds: &mut self.node_execution_op_kinds,
                                    };
                                    for report in reports {
                                        stt_data.progress_widget_state.ingest_report(report.clone());
                                        for (path, value) in report.tensor_assignments {
                                            self.inspect_window_tensor_subscription_returns
                                                .insert(path, value);
                                        }
                                        for (node_path, op_kind, age, execution_duration) in
                                            report.node_executions
                                        {
                                            let time = time_now - age;
                                            record_node_execution_activity(
                                                &mut node_activity_maps,
                                                node_path,
                                                op_kind,
                                                time,
                                                execution_duration,
                                            );
                                        }
                                        for (tensor_path, value) in
                                            report.abbreviated_tensor_assignments
                                        {
                                            self.rendered_tensor_swatches.remove(&tensor_path);
                                            self.abbreviated_tensor_reports
                                                .insert(tensor_path, value);
                                        }
                                    }
                                }
                                if let Some(response) = server_request_manager.get_response(*request_id)
                                {
                                    let (_, output_link, eos_token_id, tokenizer_info) =
                                        stt_data.pending_request.take().unwrap();
                                    match response.result {
                                        Ok(mut data) => {
                                            if let Some(token_tensor) =
                                                data.tensor_outputs.remove(&output_link)
                                            {
                                                match token_tensor.cast(DType::U32) {
                                                    Ok(token_tensor) => {
                                                        match token_tensor.flatten().try_to_vec() {
                                                            Ok(mut token_ids) => {
                                                                if let Some(pos) = token_ids
                                                                    .iter()
                                                                    .position(|&token| token == eos_token_id)
                                                                {
                                                                    token_ids.truncate(pos);
                                                                }
                                                                stt_data.transcription_tokens =
                                                                    Some(token_ids.clone());
                                                                stt_data.transcription_text = None;
                                                                match loaded_tokenizers
                                                                    .loaded_tokenizers
                                                                    .get(&tokenizer_info)
                                                                    .cloned()
                                                                    .flatten()
                                                                {
                                                                    Some(Ok(tokenizer)) => {
                                                                        match tokenizer.decode(&token_ids) {
                                                                            Ok(text) => {
                                                                                stt_data.status_message = Some(
                                                                                    format!(
                                                                                        "Transcription complete ({} tokens)",
                                                                                        token_ids.len()
                                                                                    ),
                                                                                );
                                                                                stt_data.transcription_text =
                                                                                    Some(text);
                                                                            }
                                                                            Err(err) => {
                                                                                stt_data.status_message = Some(
                                                                                    format!(
                                                                                        "Token decode failed: {err} (raw tokens shown)"
                                                                                    ),
                                                                                );
                                                                            }
                                                                        }
                                                                    }
                                                                    Some(Err(err)) => {
                                                                        stt_data.status_message = Some(format!(
                                                                            "Tokenizer load failed: {err} (raw tokens shown)"
                                                                        ));
                                                                    }
                                                                    None => {
                                                                        stt_data.status_message = Some(
                                                                            "Tokenizer not loaded yet (raw tokens shown)"
                                                                                .to_string(),
                                                                        );
                                                                    }
                                                                }
                                                            }
                                                            Err(err) => {
                                                                stt_data.status_message = Some(format!(
                                                                    "Error: token decode failed: {err}"
                                                                ));
                                                                stt_data.transcription_text = None;
                                                                stt_data.transcription_tokens = None;
                                                            }
                                                        }
                                                    }
                                                    Err(err) => {
                                                        stt_data.status_message = Some(format!(
                                                            "Error: token cast failed: {err}"
                                                        ));
                                                        stt_data.transcription_text = None;
                                                        stt_data.transcription_tokens = None;
                                                    }
                                                }
                                            } else {
                                                stt_data.status_message = Some(
                                                    "Error: output token tensor not found"
                                                        .to_string(),
                                                );
                                                stt_data.transcription_text = None;
                                                stt_data.transcription_tokens = None;
                                            }
                                        }
                                        Err(err) => {
                                            self.error_popup = Some(err);
                                            stt_data.status_message =
                                                Some("Error (see popup)".to_string());
                                            stt_data.transcription_text = None;
                                            stt_data.transcription_tokens = None;
                                        }
                                    }
                                }
                            }

                            ui.columns(2, |columns| {
                                let (left, right) = columns.split_at_mut(1);
                                let controls_ui = &mut left[0];
                                let results_ui = &mut right[0];

                                let frame = egui::Frame::default()
                                    .stroke(controls_ui.visuals().window_stroke)
                                    .inner_margin(5.0);
                                frame.show(controls_ui, |ui| {
                                    ui.vertical(|ui| {
                                        ui.heading("Speech to Text");
                                        ui.label(format!(
                                            "Input: mono WAV, resampled to {} Hz",
                                            stt_interface.sample_rate
                                        ));
                                        ui.label(format!(
                                            "Decode steps: {} (EOS token {})",
                                            stt_interface.max_decode_steps, stt_interface.eos_token_id
                                        ));
                                        ui.label(format!(
                                            "Tokenizer: {}",
                                            match &stt_interface.tokenizer {
                                                TokenizerInfo::HFTokenizer(name) => {
                                                    format!("Huggingface: {name}")
                                                }
                                                TokenizerInfo::HFTokenizerLocal(path) => {
                                                    format!("Local: {path}")
                                                }
                                                TokenizerInfo::RWKVWorld => "RWKV World".to_string(),
                                                TokenizerInfo::HFTokenizerJson(_) => {
                                                    "GGUF embedded".to_string()
                                                }
                                            }
                                        ));

                                        ui.horizontal(|ui| {
                                            ui.label(format!(
                                                "Audio: {}",
                                                stt_data
                                                    .selected_audio_name
                                                    .as_deref()
                                                    .unwrap_or("<none selected>")
                                            ));

                                            #[cfg(not(target_arch = "wasm32"))]
                                            if ui.button("Choose WAV...").clicked() {
                                                match pick_audio_file_native() {
                                                    Ok(Some(file)) => {
                                                        stt_data.selected_audio_name =
                                                            Some(file.name.clone());
                                                        stt_data.selected_audio_bytes =
                                                            Some(file.bytes);
                                                        stt_data.status_message = Some(format!(
                                                            "Loaded audio file: {}",
                                                            file.name
                                                        ));
                                                        stt_data.transcription_text = None;
                                                        stt_data.transcription_tokens = None;
                                                    }
                                                    Ok(None) => {}
                                                    Err(err) => {
                                                        stt_data.status_message = Some(format!(
                                                            "Failed to load audio file: {err}"
                                                        ));
                                                    }
                                                }
                                            }

                                            #[cfg(target_arch = "wasm32")]
                                            {
                                                if stt_data.pending_web_audio_pick.is_some() {
                                                    ui.spinner();
                                                    ui.label("Waiting for file...");
                                                } else if ui.button("Choose WAV...").clicked() {
                                                    stt_data.pending_web_audio_pick =
                                                        Some(start_audio_file_pick_web());
                                                }
                                            }

                                            if ui.button("Clear").clicked() {
                                                stt_data.selected_audio_name = None;
                                                stt_data.selected_audio_bytes = None;
                                            }
                                        });

                                        if let Some(request_id) =
                                            stt_data.pending_request.as_ref().map(|x| x.0)
                                        {
                                            ui.horizontal(|ui| {
                                                ui.spinner();
                                                ui.label("Transcribing...");
                                                if ui.button("Cancel").clicked() {
                                                    server_request_manager
                                                        .cancel_request(request_id);
                                                    stt_data.pending_request = None;
                                                    stt_data.progress_widget_state.clear();
                                                    stt_data.status_message =
                                                        Some("Cancelled".to_string());
                                                }
                                            });
                                        } else {
                                            ui.horizontal(|ui| {
                                                egui::ComboBox::from_id_salt((
                                                    "stt_backend_mode",
                                                    interface_id,
                                                ))
                                                .selected_text(stt_data.selected_mode.to_string())
                                                .show_ui(ui, |ui| {
                                                    ui.selectable_value(
                                                        &mut stt_data.selected_mode,
                                                        SuperGraphRequestBackendMode::NDArray,
                                                        SuperGraphRequestBackendMode::NDArray
                                                            .to_string(),
                                                    );
                                                    if server_config_report.vulkan_available {
                                                        ui.selectable_value(
                                                            &mut stt_data.selected_mode,
                                                            SuperGraphRequestBackendMode::Vulkan,
                                                            SuperGraphRequestBackendMode::Vulkan
                                                                .to_string(),
                                                        );
                                                    }
                                                    ui.selectable_value(
                                                        &mut stt_data.selected_mode,
                                                        SuperGraphRequestBackendMode::Compiler,
                                                        SuperGraphRequestBackendMode::Compiler
                                                            .to_string(),
                                                    );
                                                });
                                                toggle_ui(ui, &mut stt_data.use_cache);
                                                ui.label("Cache");
                                                if ui.button("Transcribe").clicked() {
                                                    if let Some(audio_bytes) =
                                                        stt_data.selected_audio_bytes.as_ref()
                                                    {
                                                        match decode_wav_bytes_to_mono_f32(
                                                            audio_bytes,
                                                            stt_interface.sample_rate,
                                                        ) {
                                                            Ok(samples) => {
                                                                if samples.is_empty() {
                                                                    stt_data.status_message = Some(
                                                                        "Audio file has no samples."
                                                                            .to_string(),
                                                                    );
                                                                } else if interface.model_ids.len() < 2 {
                                                                    stt_data.status_message = Some(format!(
                                                                        "STT interface expected 2 model IDs (encoder+decoder), found {}",
                                                                        interface.model_ids.len()
                                                                    ));
                                                                } else {
                                                                    let audio_tensor = NDArrayNumericTensor::<DynRank>::from_vec_shape(
                                                                        samples.clone(),
                                                                        &vec![samples.len() as u64],
                                                                    )
                                                                    .unwrap();

                                                                    let symbolic_graph_ids: Vec<_> =
                                                                        interface.model_ids.to_vec();
                                                                    let model_inputs = HashMap::from([
                                                                        (
                                                                            stt_interface.encoder_weights_link,
                                                                            interface.model_ids[0],
                                                                        ),
                                                                        (
                                                                            stt_interface.decoder_weights_link,
                                                                            interface.model_ids[1],
                                                                        ),
                                                                    ]);

                                                                    let swatch_settings = if state
                                                                        .do_explorer_swatches_in_view
                                                                        || state.do_all_explorer_swatches
                                                                    {
                                                                        Some(AbbreviatedTensorReportSettings {
                                                                            downsampled_size: (state.swatch_dimension
                                                                                * state.swatch_dimension)
                                                                                as u64,
                                                                            subscribed_tensors: self
                                                                                .tensors_in_view
                                                                                .iter()
                                                                                .cloned()
                                                                                .collect(),
                                                                            do_all: state.do_all_explorer_swatches,
                                                                        })
                                                                    } else {
                                                                        None
                                                                    };

                                                                    let token = server_request_manager
                                                                        .submit_supergraph_request(
                                                                            SuperGraphRequest {
                                                                                do_node_execution_reports: state
                                                                                    .explorer_node_wave,
                                                                                abbreviated_tensor_report_settings:
                                                                                    swatch_settings,
                                                                                attention_token: None,
                                                                                super_graph: stt_interface
                                                                                    .super_graph
                                                                                    .clone(),
                                                                                subscribed_tensors: self
                                                                                    .inspect_window_tensor_subscriptions
                                                                                    .iter()
                                                                                    .cloned()
                                                                                    .collect(),
                                                                                string_inputs: HashMap::new(),
                                                                                use_cache: if stt_data.use_cache {
                                                                                    Some(400 + interface_id as u64)
                                                                                } else {
                                                                                    None
                                                                                },
                                                                                backend_mode: stt_data.selected_mode,
                                                                                symbolic_graph_ids,
                                                                                tensor_inputs: HashMap::new(),
                                                                                audio_inputs: HashMap::from([(
                                                                                    stt_interface.audio_input_link,
                                                                                    SuperGraphAudioInput {
                                                                                        samples: audio_tensor,
                                                                                        sample_rate_hz: stt_interface.sample_rate,
                                                                                    },
                                                                                )]),
                                                                                model_inputs,
                                                                                hash_inputs: HashMap::new(),
                                                                            },
                                                                        );
                                                                    stt_data.progress_widget_state.clear();
                                                                    stt_data.pending_request = Some((
                                                                        token,
                                                                        stt_interface.output_token_link,
                                                                        stt_interface.eos_token_id,
                                                                        stt_interface.tokenizer.clone(),
                                                                    ));
                                                                    stt_data.status_message =
                                                                        Some("Running STT pipeline...".to_string());
                                                                    stt_data.transcription_text = None;
                                                                    stt_data.transcription_tokens = None;
                                                                }
                                                            }
                                                            Err(err) => {
                                                                stt_data.status_message = Some(format!(
                                                                    "Failed to decode WAV: {err}"
                                                                ));
                                                            }
                                                        }
                                                    } else {
                                                        stt_data.status_message = Some(
                                                            "Select a WAV file first.".to_string(),
                                                        );
                                                    }
                                                }
                                            });
                                        }
                                    });
                                });

                                let frame = egui::Frame::default()
                                    .stroke(results_ui.visuals().window_stroke)
                                    .inner_margin(5.0);
                                frame.show(results_ui, |ui| {
                                    ui.vertical(|ui| {
                                        ui.heading("Result");
                                        if let Some(msg) = &stt_data.status_message {
                                            ui.label(msg);
                                        } else {
                                            ui.label("No transcription yet.");
                                        }
                                        if let Some(tokens) = &stt_data.transcription_tokens {
                                            ui.label(format!("Tokens: {}", tokens.len()));
                                        }
                                        if let Some(text) = &stt_data.transcription_text {
                                            let mut text = text.clone();
                                            ui.add(
                                                egui::TextEdit::multiline(&mut text)
                                                    .interactive(false)
                                                    .desired_rows(8),
                                            );
                                        } else if let Some(tokens) = &stt_data.transcription_tokens {
                                            let preview = tokens
                                                .iter()
                                                .take(32)
                                                .map(|x| x.to_string())
                                                .collect::<Vec<_>>()
                                                .join(", ");
                                            ui.label(format!(
                                                "Raw token preview: [{}{}]",
                                                preview,
                                                if tokens.len() > 32 { ", ..." } else { "" }
                                            ));
                                        }
                                    });
                                });
                            });
                        }
                    }
                });
            });
        }

        if let GraphRootSubjectSelection::ServerInterface(interface_id) = self.root_selection
            && let Some(graph_rect) = graph_pane_rect
            && let Some(progress_state) = self.interface_progress_widget_state(interface_id)
            && !progress_state.is_empty()
        {
            egui::Area::new(egui::Id::new(("graph_progress_overlay", interface_id)))
                .order(egui::Order::Foreground)
                .pivot(egui::Align2::LEFT_BOTTOM)
                .fixed_pos(graph_rect.left_bottom() + vec2(10.0, -10.0))
                .show(ui.ctx(), |ui| {
                    progress_state.show(ui);
                });
        }

        if let Some(graph_rect) = graph_pane_rect
            && let Some(breadcrumb_items) = graph_breadcrumb_items
        {
            let last_idx = breadcrumb_items.len().saturating_sub(1);
            egui::Area::new(egui::Id::new((
                "graph_breadcrumb_overlay",
                self.root_selection,
            )))
            .order(egui::Order::Foreground)
            .pivot(egui::Align2::LEFT_TOP)
            .fixed_pos(graph_rect.left_top() + vec2(10.0, 10.0))
            .show(ui.ctx(), |ui| {
                egui::Frame::default()
                    .stroke(ui.visuals().window_stroke)
                    .inner_margin(egui::Margin::same(6))
                    .show(ui, |ui| {
                        ui.set_max_width(460.0);
                        ui.horizontal_wrapped(|ui| {
                            for (idx, (text, path)) in breadcrumb_items.into_iter().enumerate() {
                                if idx > 0 {
                                    ui.label(egui::RichText::new("/").size(11.0));
                                }
                                if idx < last_idx {
                                    let response =
                                        ui.link(egui::RichText::new(text.as_str()).size(11.0));
                                    if response.clicked() {
                                        self.next_graph_subject_path = Some(path);
                                    }
                                } else {
                                    ui.label(egui::RichText::new(text).size(11.0));
                                }
                            }
                        });
                    });
            });
        }

        if let Some((working_graph, working_path)) = &export_actions_graph_and_path
            && let Some(graph_rect) = graph_pane_rect
        {
            egui::Area::new(egui::Id::new((
                "graph_actions_overlay",
                self.root_selection,
            )))
            .order(egui::Order::Foreground)
            .pivot(egui::Align2::RIGHT_TOP)
            .fixed_pos(graph_rect.right_top() + vec2(-10.0, 10.0))
            .show(ui.ctx(), |ui| {
                egui::Frame::default()
                    .stroke(ui.visuals().window_stroke)
                    .inner_margin(egui::Margin::same(6))
                    .show(ui, |ui| {
                        ui.set_max_width(360.0);
                        self.render_graph_actions_panel(
                            ui,
                            *working_graph,
                            working_path,
                            root_ownership,
                            root_editability,
                        );
                    });
            });
        }

        // Prompt model loading

        for model_id in models_to_load {
            if !loaded_models
                .server_graphs
                .symbolic_graphs
                .contains_key(&model_id)
                && loaded_models
                    .server_graphs
                    .currently_requesting_model
                    .is_none()
            {
                log::info!("Loading model: {}", model_id);
                server_request_manager
                    .send(WebsocketClientServerMessage::GetModelGraph(model_id))
                    .unwrap();
                loaded_models.server_graphs.currently_requesting_model = Some(model_id);
            }
        }
    }
}

fn ensure_kokoro_voice_selection(
    selected: &mut Option<String>,
    voices: &[KokoroVoiceEmbedding],
    default_voice: Option<&str>,
) {
    let selected_valid = selected
        .as_ref()
        .is_some_and(|name| voices.iter().any(|v| v.name == *name));
    if selected_valid {
        return;
    }
    if let Some(default_voice) = default_voice
        && voices.iter().any(|v| v.name == default_voice)
    {
        *selected = Some(default_voice.to_string());
        return;
    }
    *selected = voices.first().map(|v| v.name.clone());
}

fn selected_kokoro_voice<'a>(
    selected: &Option<String>,
    voices: &'a [KokoroVoiceEmbedding],
    default_voice: Option<&str>,
) -> Option<&'a KokoroVoiceEmbedding> {
    if let Some(name) = selected
        && let Some(voice) = voices.iter().find(|v| v.name == *name)
    {
        return Some(voice);
    }
    if let Some(default_voice) = default_voice
        && let Some(voice) = voices.iter().find(|v| v.name == default_voice)
    {
        return Some(voice);
    }
    voices.first()
}

fn graph_export_default_filename(base: &str, extension: &str, working_path: &[GlobalId]) -> String {
    if working_path.is_empty() {
        format!("{base}.{extension}")
    } else {
        format!("{base}_depth_{}.{}", working_path.len(), extension)
    }
}

fn export_graph_bytes(default_filename: &str, bytes: &[u8]) -> Result<String, String> {
    #[cfg(target_arch = "wasm32")]
    {
        trigger_browser_download(default_filename, bytes, "application/cbor")
            .map_err(|err| format!("browser download failed: {err:?}"))?;
        Ok(format!("Downloaded {default_filename}"))
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let Some(path) = rfd::FileDialog::new()
            .set_file_name(default_filename)
            .save_file()
        else {
            return Ok("Export canceled".to_string());
        };
        std::fs::write(&path, bytes)
            .map_err(|err| format!("failed to write {}: {err}", path.display()))?;
        Ok(format!("Saved {}", path.display()))
    }
}

fn save_image_to_download(color_image: &ColorImage) {
    let [w, h] = color_image.size;
    let bmp_data = encode_bmp(w, h, &color_image.pixels);
    #[cfg(target_arch = "wasm32")]
    {
        let _ = trigger_browser_download("generated_image.bmp", &bmp_data, "image/bmp");
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = std::fs::write("generated_image.bmp", &bmp_data);
    }
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
    #[cfg(target_arch = "wasm32")]
    js_copy_image_to_clipboard(&rgba, w as u32, h as u32);
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = (&rgba, w, h); // TODO: native clipboard support
    }
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

#[cfg(target_arch = "wasm32")]
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
