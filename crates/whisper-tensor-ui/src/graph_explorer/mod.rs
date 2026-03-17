mod actions_panel;
mod edit_interaction_ui;
mod editing;
mod graph_layout;
mod helpers;
pub mod inspect_windows;
mod media_helpers;
mod tensor_swatch;
mod update;
mod update_flow;
mod view_helpers;
mod voice_helpers;

use crate::app::{ClientGraphId, GraphRootOwnership, InterfaceId, LoadedModels, LoadedTokenizers};
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
    Color32, ColorImage, Context, Mesh, Pos2, Rect, Sense, Shape, Stroke, StrokeKind,
    TextureHandle, Ui, UiBuilder, Vec2, vec2,
};
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
use editing::{
    GraphEditHistoryEntry, LinkDragController, LinkEditApplyResult, MAX_GRAPH_UNDO_HISTORY,
    PendingHistoryAction, SlotPipEndpoint, SlotPipOwner,
};
use graph_layout::{
    GraphLayout, GraphLayoutError, GraphLayoutLinkData, GraphLayoutLinkId, GraphLayoutNodeId,
    GraphLayoutNodeInitData, GraphLayoutNodeType,
};
pub(crate) use helpers::{LoadableGraphState, format_shape, get_inner_graph};
use helpers::{
    NodeExecutionActivityMapsMut, duration_since_node_activity, ensure_layout_link_id,
    record_node_execution_activity, render_node_contents,
};
use media_helpers::{copy_image_to_clipboard, save_image_to_download};
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};
use tensor_swatch::build_tensor_swatch;
use voice_helpers::{ensure_kokoro_voice_selection, selected_kokoro_voice};
#[cfg(target_arch = "wasm32")]
use web_time::{Duration, Instant};
use whisper_tensor::DynRank;
use whisper_tensor::backends::ndarray_backend::NDArrayNumericTensor;
use whisper_tensor::dtype::DType;
use whisper_tensor::graph::{GlobalId, SlotDirection};
use whisper_tensor::interfaces::{
    AnyInterface, ImageGenerationInterface, KokoroVoiceEmbedding, TTSInputConfig,
};
use whisper_tensor::metadata::TokenizerInfo;
use whisper_tensor::milli_graph::MilliOpGraph;
use whisper_tensor::super_graph::{SuperGraph, SuperGraphLink};
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
    link_drag_controller: LinkDragController,
    pending_link_edit_request: Option<(Vec<GlobalId>, SlotPipEndpoint, SlotPipEndpoint)>,
    pending_history_action: Option<PendingHistoryAction>,
}
