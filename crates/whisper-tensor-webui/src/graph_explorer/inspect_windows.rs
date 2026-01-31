use crate::app::LoadedModels;
use crate::graph_explorer::{
    GraphExplorerApp, GraphExplorerSettings, GraphRootSubjectSelection, LoadableGraphState,
    format_shape, get_inner_graph,
};
use crate::websockets::ServerRequestManager;
use crate::widgets::tensor_view::{TensorViewState, tensor_view};
use egui::{Color32, CornerRadius, RichText, Stroke, Vec2};
use whisper_tensor::DynRank;
use whisper_tensor::backends::ndarray_backend::NDArrayNumericTensor;
use whisper_tensor::graph::{GlobalId, Graph, GraphDyn, Property};
use whisper_tensor::scalar_info::ScalarInfoTyped;
use whisper_tensor::symbolic_graph::SymbolicGraph;
use whisper_tensor::symbolic_graph::tensor_store::TensorStoreTensorId;
use whisper_tensor::symbolic_graph::{StoredOrNotTensor, TensorType};
use whisper_tensor_server::{LoadedModelId, WebsocketClientServerMessage};

// ============================================================================
// Helper Types
// ============================================================================

/// Statistics computed from a tensor value
#[derive(Clone, Debug, Default)]
pub(crate) struct TensorStats {
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub std: f64,
    pub nan_count: usize,
    pub inf_count: usize,
    pub zero_count: usize,
    pub total_elements: usize,
}

/// Which tensor value source to prefer displaying
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum ValueSourcePreference {
    #[default]
    Live,   // Prefer subscribed/live value
    Stored, // Prefer stored value
}

/// Metadata about a link extracted from the graph for UI display.
#[derive(Clone, Debug, Default)]
pub(crate) struct LinkMetadataUI {
    pub label: Option<String>,
    pub dtype: Option<String>,
    pub shape: Option<Vec<ScalarInfoTyped<u64>>>,
    pub category: Option<String>,
    pub properties: Vec<Property>,
}

/// Response from rendering an interactive link row
#[derive(Default)]
pub(crate) struct LinkRowInteraction {
    pub hovered: bool,
    pub clicked: bool,
    pub open_requested: bool,
}

/// Response from rendering an interactive node row
#[derive(Default)]
pub(crate) struct NodeRowInteraction {
    pub hovered: bool,
    pub clicked: bool,
    pub open_requested: bool,
}

// ============================================================================
// Inspect Window Structs
// ============================================================================

#[derive(Clone, Debug)]
pub(crate) struct InspectWindowGraphLink {
    pub(crate) path: Vec<GlobalId>,

    // Stored tensor fetching
    pub(crate) stored_value_requested: Option<TensorStoreTensorId>,
    pub(crate) stored_value: Option<Result<NDArrayNumericTensor<DynRank>, String>>,

    // Tensor view states
    pub(crate) value_view_state: TensorViewState,
    pub(crate) subscribed_view_state: TensorViewState,

    // Cached statistics (computed when tensor loads)
    pub(crate) cached_stats: Option<TensorStats>,

    // UI state
    pub(crate) show_connectivity: bool,
    pub(crate) show_debug_info: bool,
    pub(crate) value_source_preference: ValueSourcePreference,
}

impl InspectWindowGraphLink {
    pub(crate) fn new(path: Vec<GlobalId>) -> Self {
        Self {
            path,
            stored_value_requested: None,
            stored_value: None,
            value_view_state: TensorViewState::default(),
            subscribed_view_state: TensorViewState::default(),
            cached_stats: None,
            show_connectivity: true,
            show_debug_info: false,
            value_source_preference: ValueSourcePreference::default(),
        }
    }

    pub(crate) fn to_any(self) -> AnyInspectWindow {
        AnyInspectWindow::GraphLink(self)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct InspectWindowGraphNode {
    pub(crate) path: Vec<GlobalId>,

    // For constant nodes - display inline value (TODO: implement)
    #[allow(dead_code)]
    pub(crate) value_view_state: TensorViewState,

    // UI state
    #[allow(dead_code)] // TODO: implement op params section
    pub(crate) show_op_params: bool,
    pub(crate) inputs_expanded: bool,
    pub(crate) outputs_expanded: bool,
    pub(crate) show_debug_info: bool,
}

impl InspectWindowGraphNode {
    pub(crate) fn new(path: Vec<GlobalId>) -> Self {
        Self {
            path,
            value_view_state: TensorViewState::default(),
            show_op_params: true,
            inputs_expanded: true,
            outputs_expanded: true,
            show_debug_info: false,
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


// ============================================================================
// Metadata Extraction Helpers
// ============================================================================

/// Get link metadata from any graph type using the new introspection traits.
fn get_link_metadata(graph: &dyn GraphDyn, link_id: GlobalId) -> LinkMetadataUI {
    let mut meta = LinkMetadataUI::default();

    // Use the new metadata trait for generic access
    if let Some(link_meta) = graph.get_link_metadata_by_id(&link_id) {
        meta.label = graph.get_link_by_id(&link_id).and_then(|l| l.label());
        meta.dtype = link_meta.dtype().map(|d| format!("{:?}", d));
        meta.shape = link_meta.shape();
        meta.category = link_meta.category().map(|c| c.to_string());

        // Include additional properties from the link
        meta.properties = link_meta.properties();
    }

    meta
}

/// Format a path as a breadcrumb string using labels where available
fn format_path_breadcrumb(
    path: &[GlobalId],
    root_graph: &dyn GraphDyn,
    loaded_models: &LoadedModels,
    root_selection: GraphRootSubjectSelection,
) -> String {
    if path.is_empty() {
        return "root".to_string();
    }

    let mut parts = Vec::new();
    let mut current_graph: &dyn GraphDyn = root_graph;

    for (i, &id) in path.iter().enumerate() {
        // Try to get a label for this element
        let label = if let Some(node) = current_graph.get_node_by_id(&id) {
            node.label().unwrap_or_else(|| node.op_kind().to_string())
        } else if let Some(link) = current_graph.get_link_by_id(&id) {
            link.label().unwrap_or_else(|| format!("link_{}", i))
        } else {
            format!("#{}", i)
        };

        parts.push(label);

        // Navigate to inner graph for next iteration (if not the last element)
        if i < path.len() - 1 {
            match get_inner_graph(current_graph, id, root_selection, loaded_models) {
                LoadableGraphState::Loaded(inner) => {
                    current_graph = inner;
                }
                _ => break,
            }
        }
    }

    parts.join(" > ")
}

/// Compute statistics from a tensor
pub(crate) fn compute_tensor_stats(tensor: &NDArrayNumericTensor<DynRank>) -> TensorStats {
    let mut stats = TensorStats::default();

    // Get total elements
    let shape = tensor.shape();
    stats.total_elements = shape.iter().map(|&x| x as usize).product();

    if stats.total_elements == 0 {
        return stats;
    }

    // Flatten and convert to Vec<f32> for stats computation
    let flattened = tensor.flatten();
    if let Ok(values) = TryInto::<Vec<f32>>::try_into(flattened) {
        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;
        stats.min = f64::INFINITY;
        stats.max = f64::NEG_INFINITY;

        for &v in &values {
            let v64 = v as f64;
            if v.is_nan() {
                stats.nan_count += 1;
            } else if v.is_infinite() {
                stats.inf_count += 1;
            } else {
                if v == 0.0 {
                    stats.zero_count += 1;
                }
                stats.min = stats.min.min(v64);
                stats.max = stats.max.max(v64);
                sum += v64;
                sum_sq += v64 * v64;
            }
        }

        let valid_count = stats.total_elements - stats.nan_count - stats.inf_count;
        if valid_count > 0 {
            stats.mean = sum / valid_count as f64;
            let variance = (sum_sq / valid_count as f64) - (stats.mean * stats.mean);
            stats.std = variance.max(0.0).sqrt();
        }
    }

    stats
}

/// Calculate memory size in bytes for a tensor
fn calculate_memory_size(shape: &[ScalarInfoTyped<u64>], dtype: Option<&str>) -> Option<usize> {
    // Only calculate if all dimensions are concrete
    let total_elements: u64 = shape
        .iter()
        .map(|s| match s {
            ScalarInfoTyped::Numeric(n) => Some(*n),
            ScalarInfoTyped::Symbolic(_) => None,
        })
        .collect::<Option<Vec<_>>>()?
        .iter()
        .product();

    let bytes_per_element = match dtype {
        Some("Float32") | Some("Int32") | Some("UInt32") => 4,
        Some("Float64") | Some("Int64") | Some("UInt64") => 8,
        Some("Float16") | Some("BFloat16") | Some("Int16") | Some("UInt16") => 2,
        Some("Int8") | Some("UInt8") | Some("Bool") => 1,
        _ => return None,
    };

    Some((total_elements as usize) * bytes_per_element)
}

/// Format bytes as human-readable size
fn format_bytes(bytes: usize) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if bytes >= 1024 * 1024 {
        format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.2} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}

// ============================================================================
// UI Styling Helpers
// ============================================================================

/// Colors for UI elements
mod colors {
    use egui::Color32;

    pub const DTYPE_BG: Color32 = Color32::from_rgb(60, 80, 100);
    pub const CATEGORY_INPUT: Color32 = Color32::from_rgb(70, 100, 70);
    pub const CATEGORY_OUTPUT: Color32 = Color32::from_rgb(100, 70, 70);
    pub const CATEGORY_CONST: Color32 = Color32::from_rgb(80, 70, 100);
    pub const CATEGORY_INTERMEDIATE: Color32 = Color32::from_rgb(70, 70, 80);
    pub const STAT_LABEL: Color32 = Color32::from_rgb(140, 140, 140);
    pub const STAT_VALUE: Color32 = Color32::from_rgb(220, 220, 220);
    pub const WARNING: Color32 = Color32::from_rgb(220, 180, 80);
    pub const SECTION_BG: Color32 = Color32::from_rgb(35, 38, 42);
}

/// Format a float with appropriate precision based on magnitude
fn format_stat_value(v: f64) -> String {
    if v == 0.0 {
        "0".to_string()
    } else if v.abs() >= 1000.0 || v.abs() < 0.001 {
        format!("{:.3e}", v)
    } else if v.abs() >= 1.0 {
        format!("{:.4}", v)
    } else {
        format!("{:.6}", v)
    }
}

/// Render a styled badge/tag
fn render_badge(ui: &mut egui::Ui, text: &str, bg_color: Color32) {
    let padding = Vec2::new(6.0, 2.0);
    egui::Frame::new()
        .fill(bg_color)
        .corner_radius(CornerRadius::same(3))
        .inner_margin(padding)
        .show(ui, |ui| {
            ui.label(RichText::new(text).small().strong().color(Color32::WHITE));
        });
}

/// Render a key-value pair with styling
fn render_kv_row(ui: &mut egui::Ui, label: &str, value: &str, monospace: bool) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(label).color(colors::STAT_LABEL));
        if monospace {
            ui.label(RichText::new(value).monospace().color(colors::STAT_VALUE));
        } else {
            ui.label(RichText::new(value).strong().color(colors::STAT_VALUE));
        }
    });
}

/// Render a stat card (label on top, large value below)
fn render_stat_card(ui: &mut egui::Ui, label: &str, value: &str) {
    ui.vertical(|ui| {
        ui.label(RichText::new(label).small().color(colors::STAT_LABEL));
        ui.label(RichText::new(value).monospace().size(14.0).color(colors::STAT_VALUE));
    });
}

/// Get background color for a category
fn category_color(category: &str) -> Color32 {
    match category {
        "Input" => colors::CATEGORY_INPUT,
        "Output" => colors::CATEGORY_OUTPUT,
        "Constant" => colors::CATEGORY_CONST,
        _ => colors::CATEGORY_INTERMEDIATE,
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

    /// Render an interactive link row in a table. Returns interaction state.
    fn render_link_row(
        &self,
        ui: &mut egui::Ui,
        index: usize,
        link_id: GlobalId,
        graph: &dyn GraphDyn,
    ) -> LinkRowInteraction {
        let mut interaction = LinkRowInteraction::default();
        let meta = get_link_metadata(graph, link_id);

        // Index column
        ui.label(format!("{}", index));

        // Name column
        let name = meta.label.as_deref().unwrap_or("-");
        let name_resp = ui.label(name);
        if name_resp.hovered() {
            interaction.hovered = true;
        }
        if name_resp.clicked() {
            interaction.clicked = true;
        }
        if name_resp.double_clicked() {
            interaction.open_requested = true;
        }

        // DType column
        ui.label(meta.dtype.as_deref().unwrap_or("-"));

        // Shape column
        let shape_str = meta
            .shape
            .as_ref()
            .map(|s| format_shape(s))
            .unwrap_or_else(|| "-".to_string());
        ui.label(shape_str);

        // Open button
        if ui.small_button("Open").clicked() {
            interaction.open_requested = true;
        }

        ui.end_row();

        interaction
    }

    /// Render a node inspect window. Returns true if the window should stay open.
    fn render_node_inspect_window(
        &mut self,
        ctx: &egui::Context,
        window: &mut InspectWindowGraphNode,
        root_graph: &dyn GraphDyn,
        loaded_models: &LoadedModels,
    ) -> (bool, Vec<Vec<GlobalId>>) {
        let mut is_open = true;
        let mut new_windows: Vec<Vec<GlobalId>> = Vec::new();
        let mut hovered_link: Option<GlobalId> = None;
        let mut clicked_link: Option<GlobalId> = None;

        let window_id = egui::Id::new(("node_inspect", window.path.clone()));

        // Build title from op type and label
        let title = if let Some((graph, node_id)) =
            self.resolve_path_to_graph(&window.path, root_graph, loaded_models)
        {
            if let Some(node) = graph.get_node_by_id(&node_id) {
                let op = node.op_kind();
                if let Some(label) = node.label() {
                    format!("{}: {}", op, label)
                } else {
                    op.to_string()
                }
            } else {
                "Node".to_string()
            }
        } else {
            "Node".to_string()
        };

        egui::Window::new(title)
            .id(window_id)
            .open(&mut is_open)
            .resizable(true)
            .default_size([450.0, 350.0])
            .show(ctx, |ui| {
                if let Some((graph, node_id)) =
                    self.resolve_path_to_graph(&window.path, root_graph, loaded_models)
                {
                    if let Some(node) = graph.get_node_by_id(&node_id) {
                        // Breadcrumb path
                        let breadcrumb = format_path_breadcrumb(
                            &window.path,
                            root_graph,
                            loaded_models,
                            self.root_selection,
                        );
                        ui.label(egui::RichText::new(breadcrumb).weak().italics());

                        // Execution timing (if available)
                        if let Some(duration) = self.node_execution_durations.get(&window.path) {
                            ui.horizontal(|ui| {
                                ui.label("Last execution:");
                                ui.strong(format!("{:.3}ms", duration.as_secs_f64() * 1000.0));
                            });
                        }

                        ui.add_space(4.0);

                        // Inputs section with interactive table
                        let inputs: Vec<_> = node.inputs().collect();
                        let parent_path: Vec<GlobalId> =
                            window.path[..window.path.len().saturating_sub(1)].to_vec();

                        egui::CollapsingHeader::new(format!("Inputs ({})", inputs.len()))
                            .default_open(window.inputs_expanded)
                            .show(ui, |ui| {
                                if inputs.is_empty() {
                                    ui.label(RichText::new("None").italics().color(colors::STAT_LABEL));
                                } else {
                                    egui::Frame::new()
                                        .fill(colors::SECTION_BG)
                                        .corner_radius(CornerRadius::same(4))
                                        .inner_margin(6.0)
                                        .show(ui, |ui| {
                                            egui::Grid::new(ui.id().with("inputs_grid"))
                                                .num_columns(5)
                                                .striped(true)
                                                .spacing([8.0, 4.0])
                                                .min_col_width(40.0)
                                                .show(ui, |ui| {
                                                    // Header row
                                                    ui.label(RichText::new("#").small().color(colors::STAT_LABEL));
                                                    ui.label(RichText::new("Name").small().color(colors::STAT_LABEL));
                                                    ui.label(RichText::new("DType").small().color(colors::STAT_LABEL));
                                                    ui.label(RichText::new("Shape").small().color(colors::STAT_LABEL));
                                                    ui.label("");
                                                    ui.end_row();

                                                    for (i, &input_id) in inputs.iter().enumerate() {
                                                        let interaction =
                                                            self.render_link_row(ui, i, input_id, graph);

                                                        if interaction.hovered {
                                                            hovered_link = Some(input_id);
                                                        }
                                                        if interaction.clicked {
                                                            clicked_link = Some(input_id);
                                                        }
                                                        if interaction.open_requested {
                                                            let mut link_path = parent_path.clone();
                                                            link_path.push(input_id);
                                                            new_windows.push(link_path);
                                                        }
                                                    }
                                                });
                                        });
                                }
                            });

                        // Outputs section with interactive table
                        let outputs: Vec<_> = node.outputs().collect();

                        egui::CollapsingHeader::new(format!("Outputs ({})", outputs.len()))
                            .default_open(window.outputs_expanded)
                            .show(ui, |ui| {
                                if outputs.is_empty() {
                                    ui.label(RichText::new("None").italics().color(colors::STAT_LABEL));
                                } else {
                                    egui::Frame::new()
                                        .fill(colors::SECTION_BG)
                                        .corner_radius(CornerRadius::same(4))
                                        .inner_margin(6.0)
                                        .show(ui, |ui| {
                                            egui::Grid::new(ui.id().with("outputs_grid"))
                                                .num_columns(5)
                                                .striped(true)
                                                .spacing([8.0, 4.0])
                                                .min_col_width(40.0)
                                                .show(ui, |ui| {
                                                    // Header row
                                                    ui.label(RichText::new("#").small().color(colors::STAT_LABEL));
                                                    ui.label(RichText::new("Name").small().color(colors::STAT_LABEL));
                                                    ui.label(RichText::new("DType").small().color(colors::STAT_LABEL));
                                                    ui.label(RichText::new("Shape").small().color(colors::STAT_LABEL));
                                                    ui.label("");
                                                    ui.end_row();

                                                    for (i, &output_id) in outputs.iter().enumerate() {
                                                        let interaction =
                                                            self.render_link_row(ui, i, output_id, graph);

                                                        if interaction.hovered {
                                                            hovered_link = Some(output_id);
                                                        }
                                                        if interaction.clicked {
                                                            clicked_link = Some(output_id);
                                                        }
                                                        if interaction.open_requested {
                                                            let mut link_path = parent_path.clone();
                                                            link_path.push(output_id);
                                                            new_windows.push(link_path);
                                                        }
                                                    }
                                                });
                                        });
                                }
                            });

                        // Parameters section (using new metadata trait)
                        if let Some(node_meta) = graph.get_node_metadata_by_id(&node_id) {
                            let params = node_meta.parameters();
                            if !params.is_empty() {
                                egui::CollapsingHeader::new(format!("Parameters ({})", params.len()))
                                    .default_open(window.show_op_params)
                                    .show(ui, |ui| {
                                        egui::Frame::new()
                                            .fill(colors::SECTION_BG)
                                            .corner_radius(CornerRadius::same(4))
                                            .inner_margin(8.0)
                                            .show(ui, |ui| {
                                                egui::Grid::new(ui.id().with("params_grid"))
                                                    .num_columns(2)
                                                    .spacing([12.0, 4.0])
                                                    .show(ui, |ui| {
                                                        for param in params {
                                                            ui.label(RichText::new(&*param.name).color(colors::STAT_LABEL));
                                                            ui.label(RichText::new(format!("{}", param.value)).monospace().color(colors::STAT_VALUE));
                                                            ui.end_row();
                                                        }
                                                    });
                                            });
                                    });
                            }
                        }

                        // Subgraph navigation
                        if self.node_has_subgraph(graph, node_id, loaded_models) {
                            ui.add_space(4.0);
                            ui.horizontal(|ui| {
                                ui.label("Contains subgraph");
                                if ui.button("Open Subgraph").clicked() {
                                    self.next_graph_subject_path = Some(window.path.clone());
                                }
                            });
                        }

                        // Debug info (collapsed by default)
                        ui.add_space(8.0);
                        egui::CollapsingHeader::new(RichText::new("Debug Info").small().color(colors::STAT_LABEL))
                            .default_open(window.show_debug_info)
                            .show(ui, |ui| {
                                egui::Frame::new()
                                    .fill(colors::SECTION_BG)
                                    .corner_radius(CornerRadius::same(4))
                                    .inner_margin(6.0)
                                    .show(ui, |ui| {
                                        render_kv_row(ui, "GlobalId:", &format!("{}", node_id), true);
                                        render_kv_row(ui, "Path:", &format!("{:?}", window.path), true);
                                    });
                            });
                    } else {
                        ui.colored_label(Color32::RED, "Node not found in graph");
                    }
                } else {
                    ui.colored_label(Color32::RED, "Could not resolve path");
                }
            });

        // Apply hover/selection to graph
        if let Some(link_id) = hovered_link {
            self.next_explorer_hovered = Some(link_id);
        }
        if let Some(link_id) = clicked_link {
            self.explorer_selection = Some(link_id);
        }

        (is_open, new_windows)
    }

    /// Render a styled node row for connectivity display
    fn render_node_row_styled(
        &self,
        ui: &mut egui::Ui,
        node_id: GlobalId,
        input_idx: Option<usize>,
        graph: &dyn GraphDyn,
    ) -> NodeRowInteraction {
        let mut interaction = NodeRowInteraction::default();

        if let Some(node) = graph.get_node_by_id(&node_id) {
            let op = node.op_kind();
            let label = node.label();

            let resp = egui::Frame::new()
                .fill(Color32::from_rgb(45, 48, 55))
                .stroke(Stroke::new(1.0, Color32::from_rgb(60, 65, 75)))
                .corner_radius(CornerRadius::same(3))
                .inner_margin(Vec2::new(8.0, 4.0))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        // Op type badge
                        ui.label(RichText::new(&op).strong().color(Color32::from_rgb(130, 180, 220)));

                        // Label if present
                        if let Some(label) = label {
                            ui.label(RichText::new(label).color(colors::STAT_VALUE));
                        }

                        // Input index if present
                        if let Some(idx) = input_idx {
                            ui.label(RichText::new(format!("[{}]", idx)).small().color(colors::STAT_LABEL));
                        }

                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.small_button("→").on_hover_text("Open").clicked() {
                                interaction.open_requested = true;
                            }
                        });
                    });
                })
                .response;

            if resp.hovered() {
                interaction.hovered = true;
            }
            if resp.clicked() {
                interaction.clicked = true;
            }
        } else {
            ui.label(RichText::new("Unknown node").italics().color(colors::STAT_LABEL));
        }

        interaction
    }

    /// Render a link inspect window. Returns (is_open, new_windows_to_open).
    fn render_link_inspect_window(
        &mut self,
        ctx: &egui::Context,
        window: &mut InspectWindowGraphLink,
        root_graph: &dyn GraphDyn,
        loaded_models: &LoadedModels,
        server_request_manager: &mut ServerRequestManager,
    ) -> (bool, Vec<Vec<GlobalId>>) {
        let mut is_open = true;
        let mut new_windows: Vec<Vec<GlobalId>> = Vec::new();
        let mut should_subscribe = false;
        let mut should_unsubscribe = false;
        let mut should_fetch_stored = false;
        let mut hovered_node: Option<GlobalId> = None;
        let mut clicked_node: Option<GlobalId> = None;

        let window_id = egui::Id::new(("link_inspect", window.path.clone()));

        // Build title from label
        let title = if let Some((graph, link_id)) =
            self.resolve_path_to_graph(&window.path, root_graph, loaded_models)
        {
            let meta = get_link_metadata(graph, link_id);
            if let Some(label) = meta.label {
                format!("Tensor: {}", label)
            } else {
                "Tensor".to_string()
            }
        } else {
            "Tensor".to_string()
        };

        egui::Window::new(title)
            .id(window_id)
            .open(&mut is_open)
            .resizable(true)
            .default_size([500.0, 450.0])
            .show(ctx, |ui| {
                if let Some((graph, link_id)) =
                    self.resolve_path_to_graph(&window.path, root_graph, loaded_models)
                {
                    let meta = get_link_metadata(graph, link_id);
                    let parent_path: Vec<GlobalId> =
                        window.path[..window.path.len().saturating_sub(1)].to_vec();

                    // Breadcrumb path
                    let breadcrumb = format_path_breadcrumb(
                        &window.path,
                        root_graph,
                        loaded_models,
                        self.root_selection,
                    );
                    ui.label(egui::RichText::new(breadcrumb).weak().italics());

                    // Header row with badges
                    ui.horizontal(|ui| {
                        if let Some(dtype) = &meta.dtype {
                            render_badge(ui, dtype, colors::DTYPE_BG);
                        }
                        if let Some(category) = &meta.category {
                            render_badge(ui, category, category_color(category));
                        }
                        if let Some(shape) = &meta.shape {
                            if let Some(bytes) = calculate_memory_size(shape, meta.dtype.as_deref()) {
                                render_badge(ui, &format_bytes(bytes), Color32::from_rgb(60, 60, 70));
                            }
                        }
                    });

                    ui.add_space(6.0);

                    // Shape display - more prominent
                    if let Some(shape) = &meta.shape {
                        ui.horizontal(|ui| {
                            ui.label(RichText::new("Shape:").color(colors::STAT_LABEL));
                            ui.label(RichText::new(format_shape(shape)).monospace().strong().size(15.0));
                        });

                        // Calculate total elements
                        let total: Option<u64> = shape
                            .iter()
                            .map(|s| match s {
                                ScalarInfoTyped::Numeric(n) => Some(*n),
                                ScalarInfoTyped::Symbolic(_) => None,
                            })
                            .collect::<Option<Vec<_>>>()
                            .map(|v| v.iter().product());

                        if let Some(total) = total {
                            ui.horizontal(|ui| {
                                ui.label(RichText::new("Elements:").color(colors::STAT_LABEL));
                                ui.label(RichText::new(format!("{}", total)).monospace());
                            });
                        }
                    }

                    // Additional properties (if any)
                    if !meta.properties.is_empty() {
                        ui.add_space(4.0);
                        for prop in &meta.properties {
                            render_kv_row(ui, &format!("{}:", prop.name), &format!("{}", prop.value), true);
                        }
                    }

                    // Connectivity section
                    egui::CollapsingHeader::new("Connectivity")
                        .default_open(window.show_connectivity)
                        .show(ui, |ui| {
                            // Find source node (which node outputs this link)
                            let mut source_node: Option<GlobalId> = None;
                            let mut dest_nodes: Vec<(GlobalId, usize)> = Vec::new();

                            for node_id in graph.node_ids() {
                                if let Some(node) = graph.get_node_by_id(&node_id) {
                                    // Check if this node outputs our link
                                    for output_id in node.outputs() {
                                        if output_id == link_id {
                                            source_node = Some(node_id);
                                            break;
                                        }
                                    }
                                    // Check if this node takes our link as input
                                    for (i, input_id) in node.inputs().enumerate() {
                                        if input_id == link_id {
                                            dest_nodes.push((node_id, i));
                                        }
                                    }
                                }
                            }

                            egui::Frame::new()
                                .fill(colors::SECTION_BG)
                                .corner_radius(CornerRadius::same(4))
                                .inner_margin(8.0)
                                .show(ui, |ui| {
                                    // Source
                                    ui.horizontal(|ui| {
                                        ui.label(RichText::new("From").color(colors::STAT_LABEL).small());
                                        ui.label(RichText::new("→").color(Color32::from_rgb(100, 180, 100)));
                                    });
                                    if let Some(src_id) = source_node {
                                        let interaction =
                                            self.render_node_row_styled(ui, src_id, None, graph);
                                        if interaction.hovered {
                                            hovered_node = Some(src_id);
                                        }
                                        if interaction.clicked {
                                            clicked_node = Some(src_id);
                                        }
                                        if interaction.open_requested {
                                            let mut node_path = parent_path.clone();
                                            node_path.push(src_id);
                                            new_windows.push(node_path);
                                        }
                                    } else {
                                        ui.label(RichText::new("Graph Input").italics().color(colors::STAT_LABEL));
                                    }

                                    ui.add_space(8.0);

                                    // Destinations
                                    ui.horizontal(|ui| {
                                        ui.label(RichText::new("To").color(colors::STAT_LABEL).small());
                                        ui.label(RichText::new("→").color(Color32::from_rgb(180, 100, 100)));
                                        if dest_nodes.len() > 1 {
                                            ui.label(RichText::new(format!("({})", dest_nodes.len())).small().color(colors::STAT_LABEL));
                                        }
                                    });
                                    if dest_nodes.is_empty() {
                                        ui.label(RichText::new("Graph Output").italics().color(colors::STAT_LABEL));
                                    } else {
                                        for (dest_id, input_idx) in &dest_nodes {
                                            let interaction =
                                                self.render_node_row_styled(ui, *dest_id, Some(*input_idx), graph);
                                            if interaction.hovered {
                                                hovered_node = Some(*dest_id);
                                            }
                                            if interaction.clicked {
                                                clicked_node = Some(*dest_id);
                                            }
                                            if interaction.open_requested {
                                                let mut node_path = parent_path.clone();
                                                node_path.push(*dest_id);
                                                new_windows.push(node_path);
                                            }
                                        }
                                    }
                                });
                        });

                    ui.add_space(4.0);

                    // Subscription controls
                    let is_subscribed =
                        self.inspect_window_tensor_subscriptions.contains(&window.path);

                    ui.horizontal(|ui| {
                        if is_subscribed {
                            ui.colored_label(egui::Color32::GREEN, "Live");
                            if ui.small_button("Unsubscribe").clicked() {
                                should_unsubscribe = true;
                            }
                        } else {
                            ui.label("Not subscribed");
                            if ui.small_button("Subscribe").clicked() {
                                should_subscribe = true;
                            }
                        }
                    });

                    ui.add_space(4.0);

                    // Tensor value section
                    let has_subscribed = self
                        .inspect_window_tensor_subscription_returns
                        .get(&window.path);
                    let has_stored = window.stored_value.as_ref();

                    // Show value source toggle if we have multiple sources
                    if has_subscribed.is_some() && has_stored.is_some() {
                        ui.horizontal(|ui| {
                            ui.label("Show:");
                            ui.selectable_value(
                                &mut window.value_source_preference,
                                ValueSourcePreference::Live,
                                "Live",
                            );
                            ui.selectable_value(
                                &mut window.value_source_preference,
                                ValueSourcePreference::Stored,
                                "Stored",
                            );
                        });
                    }

                    // Determine which tensor to show
                    let tensor_to_show = match window.value_source_preference {
                        ValueSourcePreference::Live => has_subscribed.or(has_stored.and_then(|r| r.as_ref().ok())),
                        ValueSourcePreference::Stored => has_stored.and_then(|r| r.as_ref().ok()).or(has_subscribed),
                    };

                    if let Some(tensor) = tensor_to_show {
                        // Compute and cache stats if needed
                        if window.cached_stats.is_none() {
                            window.cached_stats = Some(compute_tensor_stats(tensor));
                        }

                        // Show statistics
                        if let Some(stats) = &window.cached_stats {
                            egui::CollapsingHeader::new("Statistics")
                                .default_open(true)
                                .show(ui, |ui| {
                                    // Stats in a framed box
                                    egui::Frame::new()
                                        .fill(colors::SECTION_BG)
                                        .corner_radius(CornerRadius::same(4))
                                        .inner_margin(8.0)
                                        .show(ui, |ui| {
                                            // First row: Min/Max with visual range indicator
                                            ui.horizontal(|ui| {
                                                ui.spacing_mut().item_spacing.x = 16.0;
                                                render_stat_card(ui, "Min", &format_stat_value(stats.min));
                                                render_stat_card(ui, "Max", &format_stat_value(stats.max));
                                                render_stat_card(ui, "Mean", &format_stat_value(stats.mean));
                                                render_stat_card(ui, "Std", &format_stat_value(stats.std));
                                            });

                                            // Warning indicators for NaN/Inf
                                            if stats.nan_count > 0 || stats.inf_count > 0 || stats.zero_count > 0 {
                                                ui.add_space(6.0);
                                                ui.horizontal(|ui| {
                                                    if stats.nan_count > 0 {
                                                        ui.label(RichText::new(format!("{} NaN", stats.nan_count))
                                                            .color(colors::WARNING)
                                                            .small());
                                                    }
                                                    if stats.inf_count > 0 {
                                                        ui.label(RichText::new(format!("{} Inf", stats.inf_count))
                                                            .color(colors::WARNING)
                                                            .small());
                                                    }
                                                    if stats.zero_count > 0 {
                                                        ui.label(RichText::new(format!("{} zeros", stats.zero_count))
                                                            .color(colors::STAT_LABEL)
                                                            .small());
                                                    }
                                                });
                                            }
                                        });
                                });
                        }

                        // Show tensor view
                        let view_state = if has_subscribed.is_some()
                            && window.value_source_preference == ValueSourcePreference::Live
                        {
                            &mut window.subscribed_view_state
                        } else {
                            &mut window.value_view_state
                        };
                        tensor_view(ui, tensor, view_state);
                    } else if window.stored_value_requested.is_some() {
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.label("Loading stored value...");
                        });
                    } else if let Some(Err(err)) = &window.stored_value {
                        ui.colored_label(Color32::RED, format!("Error: {}", err));
                    } else {
                        // No value yet
                        if self.can_fetch_stored_tensor(graph, link_id) {
                            if ui.button("Fetch Stored Value").clicked() {
                                should_fetch_stored = true;
                            }
                        } else {
                            ui.label("No value available.");
                            ui.label("Subscribe to see live values during execution.");
                        }
                    }

                    // Debug info (collapsed by default)
                    ui.add_space(8.0);
                    egui::CollapsingHeader::new(RichText::new("Debug Info").small().color(colors::STAT_LABEL))
                        .default_open(window.show_debug_info)
                        .show(ui, |ui| {
                            egui::Frame::new()
                                .fill(colors::SECTION_BG)
                                .corner_radius(CornerRadius::same(4))
                                .inner_margin(6.0)
                                .show(ui, |ui| {
                                    render_kv_row(ui, "GlobalId:", &format!("{}", link_id), true);
                                    render_kv_row(ui, "Path:", &format!("{:?}", window.path), true);
                                });
                        });
                } else {
                    ui.colored_label(Color32::RED, "Could not resolve path");
                }
            });

        // Handle deferred actions
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

        // Apply hover/selection to graph
        if let Some(node_id) = hovered_node {
            self.next_explorer_hovered = Some(node_id);
        }
        if let Some(node_id) = clicked_node {
            self.explorer_selection = Some(node_id);
        }

        (is_open, new_windows)
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

        // Track which windows to keep, paths to unsubscribe, and new windows to open
        let mut windows_to_keep = Vec::new();
        let mut paths_to_unsubscribe = Vec::new();
        let mut new_window_paths: Vec<Vec<GlobalId>> = Vec::new();

        // Take ownership temporarily to allow mutable access
        let windows = std::mem::take(&mut self.inspect_windows);

        for mut window in windows {
            let (keep, new_windows) = match &mut window {
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

            // Collect new windows to open
            new_window_paths.extend(new_windows);

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

        // Open new windows requested from within existing windows
        for path in new_window_paths {
            if !AnyInspectWindow::check_if_already_exists(&self.inspect_windows, &path) {
                // Determine if this is a node or link by checking the graph
                let is_node = if let Some((graph, element_id)) =
                    self.resolve_path_to_graph(&path, root_graph, loaded_models)
                {
                    graph.get_node_by_id(&element_id).is_some()
                } else {
                    false
                };

                if is_node {
                    self.inspect_windows
                        .push(InspectWindowGraphNode::new(path).to_any());
                } else {
                    self.inspect_windows
                        .push(InspectWindowGraphLink::new(path).to_any());
                }
            }
        }
    }
}
