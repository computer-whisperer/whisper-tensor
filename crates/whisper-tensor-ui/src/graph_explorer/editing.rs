use super::{GraphExplorerApp, GraphLayout, GraphLayoutNodeType};
use crate::app::{GraphEditability, LoadedModels};
use egui::Pos2;
use whisper_tensor::graph::{GlobalId, SlotDirection};

mod adapters;
mod drag_controller;

pub(super) use adapters::GraphEditCommand;
pub(super) use drag_controller::LinkDragController;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SlotPipOwner {
    Node(GlobalId),
    InputLink(GlobalId),
    OutputLink(GlobalId),
    ConstantLink(GlobalId),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct SlotPipEndpoint {
    pub(super) owner: SlotPipOwner,
    pub(super) direction: SlotDirection,
    pub(super) slot_index: usize,
    pub(super) link_id: Option<GlobalId>,
    pub(super) layout_link_id: Option<super::GraphLayoutLinkId>,
    pub(super) screen_pos: Pos2,
}

#[derive(Clone, Debug)]
pub(super) struct LinkEditApplyResult {
    pub(super) status: String,
    pub(super) link_global_id: GlobalId,
    pub(super) requires_layout_refresh: bool,
    pub(super) history_entry: Option<GraphEditHistoryEntry>,
}

#[derive(Clone, Debug)]
pub(super) struct GraphEditHistoryEntry {
    pub(super) label: String,
    pub(super) undo_command: GraphEditCommand,
    pub(super) redo_command: GraphEditCommand,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PendingHistoryAction {
    Undo,
    Redo,
}

pub(super) const MAX_GRAPH_UNDO_HISTORY: usize = 64;

fn graph_layout_node_type_for_slot_owner(owner: SlotPipOwner) -> GraphLayoutNodeType {
    match owner {
        SlotPipOwner::Node(node_id) => GraphLayoutNodeType::GraphNode(node_id),
        SlotPipOwner::InputLink(link_id) => GraphLayoutNodeType::InputLinkNode(link_id),
        SlotPipOwner::OutputLink(link_id) => GraphLayoutNodeType::OutputLinkNode(link_id),
        SlotPipOwner::ConstantLink(link_id) => GraphLayoutNodeType::ConstantLinkNode(link_id),
    }
}

impl GraphExplorerApp {
    pub(super) fn push_history_entry(&mut self, entry: GraphEditHistoryEntry) {
        self.undo_history.push(entry);
        if self.undo_history.len() > MAX_GRAPH_UNDO_HISTORY {
            self.undo_history.drain(..1);
        }
        self.redo_history.clear();
    }

    pub(super) fn apply_edit_command(
        &mut self,
        loaded_models: &mut LoadedModels,
        command: &GraphEditCommand,
    ) -> Result<(), String> {
        let apply_result = loaded_models
            .with_editable_client_graph_mut(self.root_selection, |graph| {
                adapters::apply_graph_edit_command(graph, command)
            });
        match apply_result {
            Ok(inner) => inner,
            Err(err) => Err(err.to_string()),
        }
    }

    pub(super) fn invalidate_graph_view_cache(&mut self) {
        self.graph_layouts.clear();
        self.model_view_scene_rects.clear();
        self.link_drag_controller.clear();
        self.pending_link_edit_request = None;
    }

    pub(super) fn apply_link_drag_edit(
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

        let edit_result =
            loaded_models.with_editable_client_graph_mut(self.root_selection, |graph| {
                adapters::plan_and_apply_link_drag_edit(
                    graph,
                    working_path,
                    output_endpoint,
                    input_endpoint,
                )
            });

        let plan = match edit_result {
            Ok(inner) => inner,
            Err(err) => Err(err.to_string()),
        }?;

        let history_entry = GraphEditHistoryEntry {
            label: plan.status.clone(),
            undo_command: plan.undo_command,
            redo_command: plan.forward_command,
        };

        Ok(LinkEditApplyResult {
            status: plan.status,
            link_global_id: plan.link_global_id,
            requires_layout_refresh: plan.requires_layout_refresh,
            history_entry: Some(history_entry),
        })
    }

    pub(super) fn apply_link_drag_edit_to_layout(
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
}
