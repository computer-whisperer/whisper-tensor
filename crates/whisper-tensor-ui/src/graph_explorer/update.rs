use super::*;

impl GraphExplorerApp {
    pub(crate) fn update(
        &mut self,
        state: &mut GraphExplorerSettings,
        loaded_models: &mut LoadedModels,
        loaded_tokenizers: &mut LoadedTokenizers,
        server_request_manager: &mut ServerRequestManager,
        _server_config_report: &ServerConfigReport,
        ui: &mut Ui,
    ) {
        let mut models_to_load = HashSet::<LoadedModelId>::new();
        self.render_error_popup_if_any(ui);
        self.apply_pending_path_change();
        self.apply_pending_link_edit(loaded_models);
        self.apply_pending_history_action(loaded_models);

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
                    let can_edit_links =
                        root_editability.can_edit() && (working_is_super || working_is_milli);
                    if !can_edit_links {
                        self.link_drag_controller.clear();
                    }
                    self.link_drag_controller
                        .clear_if_graph_mismatch(working_path.as_slice());

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
                                let mut nodes_with_active_slot_drag = HashSet::new();
                                let link_data = graph_layout.get_link_data().clone();

                                self.nodes_in_view.clear();
                                self.link_drag_controller
                                    .clear_hover_target_for_graph(working_path.as_slice());
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

                                    if can_edit_links {
                                        let node_has_active_slot_drag =
                                            edit_interaction_ui::render_node_slot_pips(
                                            &mut self.link_drag_controller,
                                            ui,
                                            working_path.as_slice(),
                                            global_id,
                                            &current_node_data[&node_id].node_type,
                                            node_bounding_boxes[&node_id].center(),
                                            &node_io_connections[&node_id].inputs,
                                            &node_io_connections[&node_id].outputs,
                                            &current_node_data[&node_id].inputs,
                                            &current_node_data[&node_id].outputs,
                                            &link_data,
                                        );
                                        if node_has_active_slot_drag {
                                            nodes_with_active_slot_drag.insert(node_id);
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

                                if let Some((source_endpoint, target_endpoint)) =
                                    edit_interaction_ui::render_pending_link_drag_preview(
                                        &mut self.link_drag_controller,
                                        ui,
                                        working_path.as_slice(),
                                    )
                                {
                                    pending_link_edit_request =
                                        Some((source_endpoint, target_endpoint));
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
                                                        // Interface returns [batch, seq, vocab];
                                                        // UI runs at batch=1, so we read row 0.
                                                        let returned_tokens = shape[1];
                                                        let logits_per_token = shape[2];
                                                        for i in 0..returned_tokens as usize {
                                                            let row_start = i * logits_per_token as usize;
                                                            let output_vec: Vec<f32> = (0..logits_per_token as usize)
                                                                .map(|j| response_tokens.read_element(row_start + j).to_f64() as f32)
                                                                .collect();
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
                                                    eval_options_ui(ui, &mut text_inference_data.eval_options);
                                                    if ui.button("Run").clicked() {
                                                        let tokens =
                                                            text_inference_data.tokens.clone();
                                                        // Interface contract is [batch, seq]; UI runs at batch=1.
                                                        let tokens_tensor = NumericTensor::<DynRank, SystemPool>::from_fn(
                                                            vec![1, tokens.len() as u64],
                                                            NumericDType::U32,
                                                            &SystemPool,
                                                            |i| NumericScalar::from_u32(tokens[i]),
                                                        ).unwrap();
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
                                                                    eval_options: text_inference_data.eval_options.clone(),
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
                                            toggle_ui(ui, &mut sd_data.use_cache);
                                            ui.label("Cache");
                                        });
                                        eval_options_ui(ui, &mut sd_data.eval_options);
                                        {
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
                                                    whisper_tensor::interfaces::SchedulerType::DDIMVPrediction => {
                                                        // Video-only scheduler; not supported in image UI
                                                        return;
                                                    }
                                                };

                                                let latent_shape = vec![
                                                    1,
                                                    channels as u64,
                                                    sd_data.latent_h as u64,
                                                    sd_data.latent_w as u64,
                                                ];
                                                let latent_tensor =
                                                    NumericTensor::<DynRank, SystemPool>::from_fn(
                                                        latent_shape,
                                                        NumericDType::F32,
                                                        &SystemPool,
                                                        |i| NumericScalar::from_f32(initial_noise[i]),
                                                    )
                                                    .unwrap();

                                                let timesteps_tensor =
                                                    NumericTensor::<DynRank, SystemPool>::from_fn(
                                                        vec![sd_data.num_steps as u64],
                                                        NumericDType::F32,
                                                        &SystemPool,
                                                        |i| NumericScalar::from_f32(timestep_values[i]),
                                                    )
                                                    .unwrap();
                                                let dt_tensor =
                                                    NumericTensor::<DynRank, SystemPool>::from_fn(
                                                        vec![sd_data.num_steps as u64],
                                                        NumericDType::F32,
                                                        &SystemPool,
                                                        |i| NumericScalar::from_f32(dt_values[i]),
                                                    )
                                                    .unwrap();
                                                let sigmas_tensor =
                                                    NumericTensor::<DynRank, SystemPool>::from_fn(
                                                        vec![sd_data.num_steps as u64],
                                                        NumericDType::F32,
                                                        &SystemPool,
                                                        |i| NumericScalar::from_f32(sigma_values[i]),
                                                    )
                                                    .unwrap();
                                                let iter_count =
                                                    NumericTensor::<DynRank, SystemPool>::from_fn(
                                                        vec![1],
                                                        NumericDType::I64,
                                                        &SystemPool,
                                                        |_| NumericScalar::from_i64(sd_data.num_steps as i64),
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
                                                        NumericTensor::<DynRank, SystemPool>::from_fn(
                                                            vec![1],
                                                            NumericDType::F32,
                                                            &SystemPool,
                                                            |_| NumericScalar::from_f32(sd_data.guidance_scale),
                                                        )
                                                        .unwrap();
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
                                                            eval_options: sd_data.eval_options.clone(),
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
                                        }
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
                    AnyInterface::VideoGenerationInterface(_video_interface) => {
                        ui.label("Video generation interface (not yet runnable from UI)");
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
                                            toggle_ui(ui, &mut tts_data.use_cache);
                                            ui.label("Cache");
                                        });
                                        eval_options_ui(ui, &mut tts_data.eval_options);
                                        {
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
                                                                    let style = NumericTensor::<DynRank, SystemPool>::from_fn(
                                                                        vec![1, KokoroVoiceEmbedding::STYLE_DIM as u64],
                                                                        NumericDType::F32,
                                                                        &SystemPool,
                                                                        |i| NumericScalar::from_f32(style_values[i]),
                                                                    ).unwrap();
                                                                    let speed = NumericTensor::<DynRank, SystemPool>::from_fn(
                                                                        vec![1],
                                                                        NumericDType::F32,
                                                                        &SystemPool,
                                                                        |_| NumericScalar::from_f32(tts_data.speed),
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
                                                        let scale_values = [0.667f32, length_scale, 0.8];
                                                        let scales = NumericTensor::<DynRank, SystemPool>::from_fn(
                                                            vec![3],
                                                            NumericDType::F32,
                                                            &SystemPool,
                                                            |i| NumericScalar::from_f32(scale_values[i]),
                                                        ).unwrap();
                                                        tensor_inputs.insert(*scales_link, scales);
                                                        if let Some(sid_link) = speaker_id_link {
                                                            let speaker_id = NumericTensor::<DynRank, SystemPool>::from_fn(
                                                                vec![1],
                                                                NumericDType::I64,
                                                                &SystemPool,
                                                                |_| NumericScalar::from_i64(tts_data.piper_speaker_id),
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
                                                                eval_options: tts_data.eval_options.clone(),
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
                                        }
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
                                                let mut token_ids: Vec<u32> = (0..token_tensor.numel())
                                                    .map(|i| token_tensor.read_element(i).to_i64() as u32)
                                                    .collect();
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
                                                toggle_ui(ui, &mut stt_data.use_cache);
                                                ui.label("Cache");
                                            });
                                            eval_options_ui(ui, &mut stt_data.eval_options);
                                            {
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
                                                                    let audio_tensor = NumericTensor::<DynRank, SystemPool>::from_fn(
                                                                        vec![samples.len() as u64],
                                                                        NumericDType::F32,
                                                                        &SystemPool,
                                                                        |i| NumericScalar::from_f32(samples[i]),
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
                                                                                eval_options: stt_data.eval_options.clone(),
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
                                            }
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

        self.request_missing_models(models_to_load, loaded_models, server_request_manager);
    }
}
