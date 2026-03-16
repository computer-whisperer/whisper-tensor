use super::{GraphExplorerApp, GraphLayout, GraphLayoutNodeType};
use crate::app::{EditableClientGraphMut, GraphEditability, LoadedModels};
use egui::Pos2;
use whisper_tensor::graph::{GlobalId, Node, SlotDirection};
use whisper_tensor::milli_graph::MilliOpGraph;
use whisper_tensor::super_graph::nodes::SuperGraphNode;
use whisper_tensor::super_graph::{SuperGraph, SuperGraphLinkInfo};

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
pub(super) struct PendingLinkDrag {
    pub(super) graph_path: Vec<GlobalId>,
    pub(super) source: SlotPipEndpoint,
}

#[derive(Clone, Debug)]
pub(super) struct LinkEditApplyResult {
    pub(super) status: String,
    pub(super) link_global_id: GlobalId,
    pub(super) requires_layout_refresh: bool,
    pub(super) history_entry: Option<GraphEditHistoryEntry>,
}

#[derive(Clone, Debug)]
pub(super) enum EditableGraphSnapshot {
    SuperGraph(Box<SuperGraph>),
    MilliOpGraph(Box<MilliOpGraph>),
}

#[derive(Clone, Debug)]
pub(super) struct GraphEditHistoryEntry {
    pub(super) label: String,
    pub(super) before: EditableGraphSnapshot,
    pub(super) after: EditableGraphSnapshot,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PendingHistoryAction {
    Undo,
    Redo,
}

pub(super) const MAX_GRAPH_UNDO_HISTORY: usize = 64;

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

impl GraphExplorerApp {
    pub(super) fn push_history_entry(&mut self, entry: GraphEditHistoryEntry) {
        self.undo_history.push(entry);
        if self.undo_history.len() > MAX_GRAPH_UNDO_HISTORY {
            self.undo_history.drain(..1);
        }
        self.redo_history.clear();
    }

    pub(super) fn restore_root_graph_snapshot(
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

    pub(super) fn invalidate_graph_view_cache(&mut self) {
        self.graph_layouts.clear();
        self.model_view_scene_rects.clear();
        self.pending_link_drag = None;
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
