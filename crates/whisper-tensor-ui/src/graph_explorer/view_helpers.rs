use super::*;

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
}
