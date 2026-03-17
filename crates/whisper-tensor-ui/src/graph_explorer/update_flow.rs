use super::*;

impl GraphExplorerApp {
    pub(super) fn render_error_popup_if_any(&mut self, ui: &mut Ui) {
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
    }

    pub(super) fn apply_pending_path_change(&mut self) {
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
            self.link_drag_controller.clear();
            self.pending_link_edit_request = None;
            self.pending_history_action = None;
        }
    }

    pub(super) fn apply_pending_link_edit(&mut self, loaded_models: &mut LoadedModels) {
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
                        layout_patch,
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
                    if !should_relayout
                        && let Some(layout_patch) = layout_patch.as_ref()
                        && !layout_patch.is_empty()
                    {
                        match self.graph_layouts.get_mut(&working_path) {
                            Some(Ok(graph_layout)) => {
                                if let Err(err) =
                                    Self::apply_layout_patch_to_layout(graph_layout, layout_patch)
                                {
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
                            None => {}
                        }
                    }

                    if should_relayout {
                        self.invalidate_graph_view_cache_for_path(&working_path);
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
    }

    pub(super) fn apply_pending_history_action(&mut self, loaded_models: &mut LoadedModels) {
        if let Some(history_action) = self.pending_history_action.take() {
            match history_action {
                PendingHistoryAction::Undo => {
                    if let Some(entry) = self.undo_history.pop() {
                        let graph_path = entry.graph_path.clone();
                        let mut should_relayout = entry.requires_layout_refresh;
                        let mut relayout_reason = if should_relayout {
                            Some("graph I/O endpoint identity changed".to_string())
                        } else {
                            None
                        };

                        match self.apply_edit_command(loaded_models, &entry.undo_command) {
                            Ok(()) => {
                                if !should_relayout
                                    && let Some(layout_patch) = entry.undo_layout_patch.as_ref()
                                    && !layout_patch.is_empty()
                                {
                                    match self.graph_layouts.get_mut(&graph_path) {
                                        Some(Ok(graph_layout)) => {
                                            if let Err(err) = Self::apply_layout_patch_to_layout(
                                                graph_layout,
                                                layout_patch,
                                            ) {
                                                should_relayout = true;
                                                relayout_reason = Some(err);
                                            }
                                        }
                                        Some(Err(err)) => {
                                            should_relayout = true;
                                            relayout_reason = Some(format!(
                                                "Cached layout was already invalid before undo: {err}"
                                            ));
                                        }
                                        None => {}
                                    }
                                }
                                if should_relayout {
                                    self.invalidate_graph_view_cache_for_path(&graph_path);
                                } else {
                                    self.link_drag_controller.clear();
                                    self.pending_link_edit_request = None;
                                }

                                if let Some(reason) = relayout_reason {
                                    self.actions_status = Some(format!(
                                        "Undo: {} (layout refresh fallback: {reason})",
                                        entry.label
                                    ));
                                } else {
                                    self.actions_status = Some(format!("Undo: {}", entry.label));
                                }
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
                        let graph_path = entry.graph_path.clone();
                        let mut should_relayout = entry.requires_layout_refresh;
                        let mut relayout_reason = if should_relayout {
                            Some("graph I/O endpoint identity changed".to_string())
                        } else {
                            None
                        };

                        match self.apply_edit_command(loaded_models, &entry.redo_command) {
                            Ok(()) => {
                                if !should_relayout
                                    && let Some(layout_patch) = entry.redo_layout_patch.as_ref()
                                    && !layout_patch.is_empty()
                                {
                                    match self.graph_layouts.get_mut(&graph_path) {
                                        Some(Ok(graph_layout)) => {
                                            if let Err(err) = Self::apply_layout_patch_to_layout(
                                                graph_layout,
                                                layout_patch,
                                            ) {
                                                should_relayout = true;
                                                relayout_reason = Some(err);
                                            }
                                        }
                                        Some(Err(err)) => {
                                            should_relayout = true;
                                            relayout_reason = Some(format!(
                                                "Cached layout was already invalid before redo: {err}"
                                            ));
                                        }
                                        None => {}
                                    }
                                }

                                if should_relayout {
                                    self.invalidate_graph_view_cache_for_path(&graph_path);
                                } else {
                                    self.link_drag_controller.clear();
                                    self.pending_link_edit_request = None;
                                }

                                if let Some(reason) = relayout_reason {
                                    self.actions_status = Some(format!(
                                        "Redo: {} (layout refresh fallback: {reason})",
                                        entry.label
                                    ));
                                } else {
                                    self.actions_status = Some(format!("Redo: {}", entry.label));
                                }
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
    }

    pub(super) fn request_missing_models(
        &mut self,
        models_to_load: HashSet<LoadedModelId>,
        loaded_models: &mut LoadedModels,
        server_request_manager: &mut ServerRequestManager,
    ) {
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
