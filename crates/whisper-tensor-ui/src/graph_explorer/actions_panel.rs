use super::GraphExplorerApp;
use crate::app::{GraphEditability, GraphRootOwnership};
use egui::Ui;
use std::any::Any;
use whisper_tensor::graph::{GlobalId, GraphDyn};
use whisper_tensor::graph_format::{MILLI_OP_GRAPH_FILE_EXTENSION, SUPER_GRAPH_FILE_EXTENSION};
use whisper_tensor::milli_graph::MilliOpGraph;
use whisper_tensor::super_graph::SuperGraph;

impl GraphExplorerApp {
    pub(super) fn render_graph_actions_panel(
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
                        self.pending_history_action = Some(super::PendingHistoryAction::Undo);
                    } else if redo_clicked {
                        self.pending_history_action = Some(super::PendingHistoryAction::Redo);
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
        super::media_helpers::trigger_browser_download(default_filename, bytes, "application/cbor")
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
