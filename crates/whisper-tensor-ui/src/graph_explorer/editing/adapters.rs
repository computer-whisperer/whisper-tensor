use super::{GraphLayoutPatch, GraphLayoutSlotPatch, SlotPipEndpoint, SlotPipOwner};
use crate::app::EditableClientGraphMut;
use whisper_tensor::graph::{GlobalId, Graph, Node, SlotDirection};
use whisper_tensor::milli_graph::MilliOpGraph;
use whisper_tensor::super_graph::nodes::{SuperGraphAnyNode, SuperGraphNode};
use whisper_tensor::super_graph::{SuperGraph, SuperGraphLink, SuperGraphLinkInfo};

#[derive(Clone, Debug)]
pub(super) struct GraphEditPlan {
    pub(super) status: String,
    pub(super) requires_layout_refresh: bool,
    pub(super) forward_layout_patch: Option<GraphLayoutPatch>,
    pub(super) undo_layout_patch: Option<GraphLayoutPatch>,
    pub(super) forward_command: GraphEditCommand,
    pub(super) undo_command: GraphEditCommand,
}

#[derive(Clone, Debug)]
pub(crate) enum GraphEditCommand {
    SuperGraph(SuperGraphEditCommandBatch),
    MilliOpGraph(MilliOpGraphEditCommandBatch),
}

#[derive(Clone, Debug)]
pub(crate) struct SuperGraphEditCommandBatch {
    graph_path: Vec<GlobalId>,
    ops: Vec<SuperGraphEditOp>,
}

#[derive(Clone, Debug)]
enum SuperGraphEditOp {
    SetNodeInputSlot {
        node_id: GlobalId,
        slot_index: usize,
        link: Option<GlobalId>,
    },
    SetNodeOutputSlot {
        node_id: GlobalId,
        slot_index: usize,
        link: Option<GlobalId>,
    },
    RetargetGraphOutputLink {
        existing_output_link_id: GlobalId,
        replacement_link_id: GlobalId,
    },
    EnsureLinkMetadata {
        link: SuperGraphLink,
    },
}

#[derive(Clone, Debug)]
pub(crate) struct MilliOpGraphEditCommandBatch {
    graph_path: Vec<GlobalId>,
    ops: Vec<MilliOpGraphEditOp>,
}

#[derive(Clone, Debug)]
enum MilliOpGraphEditOp {
    SetNodeInputSlot {
        node_id: GlobalId,
        slot_index: usize,
        link: Option<GlobalId>,
    },
    SetNodeOutputSlot {
        node_id: GlobalId,
        slot_index: usize,
        link: Option<GlobalId>,
    },
    RetargetGraphOutputLink {
        old_internal_id: GlobalId,
        new_internal_id: GlobalId,
    },
    EnsureTensor {
        tensor_id: GlobalId,
    },
}

enum ResolvedEditableGraphRef<'a> {
    SuperGraph,
    Milli(&'a MilliOpGraph),
}

pub(super) fn plan_and_apply_link_drag_edit(
    graph: EditableClientGraphMut<'_>,
    working_path: &[GlobalId],
    output_endpoint: SlotPipEndpoint,
    input_endpoint: SlotPipEndpoint,
) -> Result<GraphEditPlan, String> {
    match graph {
        EditableClientGraphMut::SuperGraph(root_graph) => {
            let resolved_graph =
                resolve_editable_graph_in_supergraph_at_path(root_graph, working_path).ok_or_else(
                    || {
                        "Failed to resolve editable graph at current path inside SuperGraph root."
                            .to_string()
                    },
                )?;
            match resolved_graph {
                ResolvedEditableGraphRef::SuperGraph => {
                    let plan = plan_supergraph_link_drag(
                        root_graph,
                        working_path,
                        output_endpoint,
                        input_endpoint,
                    )?;
                    let GraphEditCommand::SuperGraph(forward_batch) = &plan.forward_command else {
                        unreachable!();
                    };
                    apply_supergraph_edit_command(root_graph, forward_batch)?;
                    Ok(plan)
                }
                ResolvedEditableGraphRef::Milli(graph) => {
                    let plan = plan_milli_graph_link_drag(
                        graph,
                        working_path,
                        output_endpoint,
                        input_endpoint,
                    )?;
                    let GraphEditCommand::MilliOpGraph(forward_batch) = &plan.forward_command
                    else {
                        unreachable!();
                    };
                    apply_milli_graph_edit_command_in_supergraph(root_graph, forward_batch)?;
                    Ok(plan)
                }
            }
        }
        EditableClientGraphMut::MilliOpGraph(graph) => {
            let plan =
                plan_milli_graph_link_drag(graph, working_path, output_endpoint, input_endpoint)?;
            let GraphEditCommand::MilliOpGraph(forward_batch) = &plan.forward_command else {
                unreachable!();
            };
            apply_milli_graph_edit_command(graph, forward_batch)?;
            Ok(plan)
        }
    }
}

pub(super) fn apply_graph_edit_command(
    graph: EditableClientGraphMut<'_>,
    command: &GraphEditCommand,
) -> Result<(), String> {
    match (graph, command) {
        (EditableClientGraphMut::SuperGraph(root_graph), GraphEditCommand::SuperGraph(batch)) => {
            apply_supergraph_edit_command(root_graph, batch)
        }
        (EditableClientGraphMut::SuperGraph(root_graph), GraphEditCommand::MilliOpGraph(batch)) => {
            apply_milli_graph_edit_command_in_supergraph(root_graph, batch)
        }
        (EditableClientGraphMut::MilliOpGraph(graph), GraphEditCommand::MilliOpGraph(batch)) => {
            apply_milli_graph_edit_command(graph, batch)
        }
        (EditableClientGraphMut::MilliOpGraph(_), GraphEditCommand::SuperGraph(_)) => {
            Err("graph edit command mismatch: expected MilliOpGraph command".to_string())
        }
    }
}

fn plan_supergraph_link_drag(
    root_graph: &SuperGraph,
    working_path: &[GlobalId],
    output_endpoint: SlotPipEndpoint,
    input_endpoint: SlotPipEndpoint,
) -> Result<GraphEditPlan, String> {
    let graph = resolve_supergraph_at_path(root_graph, working_path)
        .ok_or_else(|| "Failed to resolve SuperGraph at current path.".to_string())?;

    let link_global_id = output_endpoint
        .link_id
        .or(input_endpoint.link_id)
        .unwrap_or_else(|| {
            let mut rng = rand::rng();
            GlobalId::new(&mut rng)
        });

    let mut requires_layout_refresh = false;

    let source_template_link = resolve_supergraph_source_template_link(graph, output_endpoint)?;
    let mut target_link = graph
        .links_by_global_id
        .get(&link_global_id)
        .map(|x| x.link())
        .unwrap_or_else(|| {
            SuperGraphLink::with_global_id(link_global_id, source_template_link.kind())
        });

    let mut forward_ops = Vec::new();
    let mut undo_ops = Vec::new();

    match output_endpoint.owner {
        SlotPipOwner::Node(node_id) => {
            let old_link = supergraph_node_output_slot(graph, node_id, output_endpoint.slot_index)?;
            forward_ops.push(SuperGraphEditOp::SetNodeOutputSlot {
                node_id,
                slot_index: output_endpoint.slot_index,
                link: Some(link_global_id),
            });
            undo_ops.push(SuperGraphEditOp::SetNodeOutputSlot {
                node_id,
                slot_index: output_endpoint.slot_index,
                link: old_link,
            });

            if old_link == Some(link_global_id)
                && let Some(existing_link) = old_link.and_then(|link_id| {
                    graph
                        .links_by_global_id
                        .get(&link_id)
                        .map(|link_info| link_info.link())
                })
            {
                target_link = existing_link;
            }
        }
        SlotPipOwner::InputLink(link_id) | SlotPipOwner::ConstantLink(link_id) => {
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
            let old_link = supergraph_node_input_slot(graph, node_id, input_endpoint.slot_index)?;
            forward_ops.push(SuperGraphEditOp::SetNodeInputSlot {
                node_id,
                slot_index: input_endpoint.slot_index,
                link: Some(link_global_id),
            });
            undo_ops.push(SuperGraphEditOp::SetNodeInputSlot {
                node_id,
                slot_index: input_endpoint.slot_index,
                link: old_link,
            });
        }
        SlotPipOwner::OutputLink(existing_output_link_id) => {
            let existing_output_link = graph
                .output_links
                .iter()
                .copied()
                .find(|x| x.global_id() == existing_output_link_id)
                .ok_or_else(|| format!("Missing graph output link {}", existing_output_link_id))?;

            if existing_output_link_id != link_global_id {
                if existing_output_link.kind() != target_link.kind() {
                    return Err(format!(
                        "Cannot retarget output link {} ({}) to {} ({})",
                        existing_output_link_id,
                        existing_output_link.kind().as_str(),
                        target_link.global_id(),
                        target_link.kind().as_str()
                    ));
                }

                forward_ops.push(SuperGraphEditOp::RetargetGraphOutputLink {
                    existing_output_link_id,
                    replacement_link_id: link_global_id,
                });
                undo_ops.push(SuperGraphEditOp::RetargetGraphOutputLink {
                    existing_output_link_id: link_global_id,
                    replacement_link_id: existing_output_link_id,
                });
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

    forward_ops.insert(
        0,
        SuperGraphEditOp::EnsureLinkMetadata { link: target_link },
    );
    undo_ops.insert(
        0,
        SuperGraphEditOp::EnsureLinkMetadata { link: target_link },
    );

    let status = format!(
        "Linked {} -> {} ({})",
        describe_endpoint(&output_endpoint),
        describe_endpoint(&input_endpoint),
        target_link.global_id()
    );

    let forward_layout_patch = if requires_layout_refresh {
        None
    } else {
        supergraph_layout_patch_from_ops(&forward_ops)
    };
    let undo_layout_patch = if requires_layout_refresh {
        None
    } else {
        supergraph_layout_patch_from_ops(&undo_ops)
    };

    let forward_command = GraphEditCommand::SuperGraph(SuperGraphEditCommandBatch {
        graph_path: working_path.to_vec(),
        ops: forward_ops,
    });
    let undo_command = GraphEditCommand::SuperGraph(SuperGraphEditCommandBatch {
        graph_path: working_path.to_vec(),
        ops: undo_ops,
    });

    Ok(GraphEditPlan {
        status,
        requires_layout_refresh,
        forward_layout_patch,
        undo_layout_patch,
        forward_command,
        undo_command,
    })
}

fn apply_supergraph_edit_command(
    root_graph: &mut SuperGraph,
    command: &SuperGraphEditCommandBatch,
) -> Result<(), String> {
    let graph = resolve_supergraph_mut_at_path(root_graph, &command.graph_path)
        .ok_or_else(|| "Failed to resolve mutable SuperGraph at command path.".to_string())?;

    for op in &command.ops {
        match op {
            SuperGraphEditOp::SetNodeInputSlot {
                node_id,
                slot_index,
                link,
            } => {
                let input_node = graph
                    .nodes
                    .get_mut(node_id)
                    .ok_or_else(|| format!("Missing input node {node_id}"))?;
                Node::set_input_slot(input_node, *slot_index, *link).map_err(|err| {
                    format!(
                        "Failed to set input slot {} on {}: {err:?}",
                        slot_index, node_id
                    )
                })?;
            }
            SuperGraphEditOp::SetNodeOutputSlot {
                node_id,
                slot_index,
                link,
            } => {
                let output_node = graph
                    .nodes
                    .get_mut(node_id)
                    .ok_or_else(|| format!("Missing output node {node_id}"))?;
                Node::set_output_slot(output_node, *slot_index, *link).map_err(|err| {
                    format!(
                        "Failed to set output slot {} on {}: {err:?}",
                        slot_index, node_id
                    )
                })?;
            }
            SuperGraphEditOp::RetargetGraphOutputLink {
                existing_output_link_id,
                replacement_link_id,
            } => {
                if existing_output_link_id == replacement_link_id {
                    continue;
                }
                let existing_output_link = graph
                    .output_links
                    .iter()
                    .copied()
                    .find(|x| x.global_id() == *existing_output_link_id)
                    .ok_or_else(|| {
                        format!("Missing graph output link {}", existing_output_link_id)
                    })?;
                let replacement_link = graph
                    .links_by_global_id
                    .get(replacement_link_id)
                    .map(|x| x.link())
                    .ok_or_else(|| {
                        format!(
                            "Missing link metadata for output link {}",
                            replacement_link_id
                        )
                    })?;
                if existing_output_link.kind() != replacement_link.kind() {
                    return Err(format!(
                        "Cannot retarget output link {} ({}) to {} ({})",
                        existing_output_link_id,
                        existing_output_link.kind().as_str(),
                        replacement_link.global_id(),
                        replacement_link.kind().as_str()
                    ));
                }
                graph.output_links.remove(&existing_output_link);
                graph.output_links.insert(replacement_link);
            }
            SuperGraphEditOp::EnsureLinkMetadata { link } => {
                graph
                    .links_by_global_id
                    .entry(link.global_id())
                    .or_insert_with(|| SuperGraphLinkInfo::new(*link, None));
            }
        }
    }

    Ok(())
}

fn plan_milli_graph_link_drag(
    graph: &MilliOpGraph,
    working_path: &[GlobalId],
    output_endpoint: SlotPipEndpoint,
    input_endpoint: SlotPipEndpoint,
) -> Result<GraphEditPlan, String> {
    let link_global_id = output_endpoint
        .link_id
        .or(input_endpoint.link_id)
        .unwrap_or_else(|| {
            let mut rng = rand::rng();
            GlobalId::new(&mut rng)
        });

    let mut requires_layout_refresh = false;
    let mut forward_ops = vec![MilliOpGraphEditOp::EnsureTensor {
        tensor_id: link_global_id,
    }];
    let mut undo_ops = vec![MilliOpGraphEditOp::EnsureTensor {
        tensor_id: link_global_id,
    }];

    match output_endpoint.owner {
        SlotPipOwner::Node(node_id) => {
            let old_link =
                milli_graph_node_output_slot(graph, node_id, output_endpoint.slot_index)?;
            forward_ops.push(MilliOpGraphEditOp::SetNodeOutputSlot {
                node_id,
                slot_index: output_endpoint.slot_index,
                link: Some(link_global_id),
            });
            undo_ops.push(MilliOpGraphEditOp::SetNodeOutputSlot {
                node_id,
                slot_index: output_endpoint.slot_index,
                link: old_link,
            });
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
            let old_link = milli_graph_node_input_slot(graph, node_id, input_endpoint.slot_index)?;
            forward_ops.push(MilliOpGraphEditOp::SetNodeInputSlot {
                node_id,
                slot_index: input_endpoint.slot_index,
                link: Some(link_global_id),
            });
            undo_ops.push(MilliOpGraphEditOp::SetNodeInputSlot {
                node_id,
                slot_index: input_endpoint.slot_index,
                link: old_link,
            });
        }
        SlotPipOwner::OutputLink(old_internal_id) => {
            if old_internal_id != link_global_id {
                forward_ops.push(MilliOpGraphEditOp::RetargetGraphOutputLink {
                    old_internal_id,
                    new_internal_id: link_global_id,
                });
                undo_ops.push(MilliOpGraphEditOp::RetargetGraphOutputLink {
                    old_internal_id: link_global_id,
                    new_internal_id: old_internal_id,
                });
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

    let status = format!(
        "Linked {} -> {} ({})",
        describe_endpoint(&output_endpoint),
        describe_endpoint(&input_endpoint),
        link_global_id
    );

    let forward_layout_patch = if requires_layout_refresh {
        None
    } else {
        milli_layout_patch_from_ops(&forward_ops)
    };
    let undo_layout_patch = if requires_layout_refresh {
        None
    } else {
        milli_layout_patch_from_ops(&undo_ops)
    };

    let forward_command = GraphEditCommand::MilliOpGraph(MilliOpGraphEditCommandBatch {
        graph_path: working_path.to_vec(),
        ops: forward_ops,
    });
    let undo_command = GraphEditCommand::MilliOpGraph(MilliOpGraphEditCommandBatch {
        graph_path: working_path.to_vec(),
        ops: undo_ops,
    });

    Ok(GraphEditPlan {
        status,
        requires_layout_refresh,
        forward_layout_patch,
        undo_layout_patch,
        forward_command,
        undo_command,
    })
}

fn supergraph_layout_patch_from_ops(ops: &[SuperGraphEditOp]) -> Option<GraphLayoutPatch> {
    let mut slot_patches = Vec::new();
    for op in ops {
        match op {
            SuperGraphEditOp::SetNodeInputSlot {
                node_id,
                slot_index,
                link,
            } => slot_patches.push(GraphLayoutSlotPatch {
                owner: SlotPipOwner::Node(*node_id),
                direction: SlotDirection::Input,
                slot_index: *slot_index,
                link_global_id: *link,
            }),
            SuperGraphEditOp::SetNodeOutputSlot {
                node_id,
                slot_index,
                link,
            } => slot_patches.push(GraphLayoutSlotPatch {
                owner: SlotPipOwner::Node(*node_id),
                direction: SlotDirection::Output,
                slot_index: *slot_index,
                link_global_id: *link,
            }),
            SuperGraphEditOp::RetargetGraphOutputLink { .. }
            | SuperGraphEditOp::EnsureLinkMetadata { .. } => {}
        }
    }

    if slot_patches.is_empty() {
        None
    } else {
        Some(GraphLayoutPatch { slot_patches })
    }
}

fn milli_layout_patch_from_ops(ops: &[MilliOpGraphEditOp]) -> Option<GraphLayoutPatch> {
    let mut slot_patches = Vec::new();
    for op in ops {
        match op {
            MilliOpGraphEditOp::SetNodeInputSlot {
                node_id,
                slot_index,
                link,
            } => slot_patches.push(GraphLayoutSlotPatch {
                owner: SlotPipOwner::Node(*node_id),
                direction: SlotDirection::Input,
                slot_index: *slot_index,
                link_global_id: *link,
            }),
            MilliOpGraphEditOp::SetNodeOutputSlot {
                node_id,
                slot_index,
                link,
            } => slot_patches.push(GraphLayoutSlotPatch {
                owner: SlotPipOwner::Node(*node_id),
                direction: SlotDirection::Output,
                slot_index: *slot_index,
                link_global_id: *link,
            }),
            MilliOpGraphEditOp::RetargetGraphOutputLink { .. }
            | MilliOpGraphEditOp::EnsureTensor { .. } => {}
        }
    }

    if slot_patches.is_empty() {
        None
    } else {
        Some(GraphLayoutPatch { slot_patches })
    }
}

fn apply_milli_graph_edit_command(
    graph: &mut MilliOpGraph,
    command: &MilliOpGraphEditCommandBatch,
) -> Result<(), String> {
    if !command.graph_path.is_empty() {
        return Err("MilliOpGraph edit command path must be empty.".to_string());
    }

    apply_milli_graph_edit_ops(graph, &command.ops)
}

fn apply_milli_graph_edit_command_in_supergraph(
    root_graph: &mut SuperGraph,
    command: &MilliOpGraphEditCommandBatch,
) -> Result<(), String> {
    let graph = resolve_milli_graph_mut_in_supergraph_at_path(root_graph, &command.graph_path)
        .ok_or_else(|| {
            "Failed to resolve mutable MilliOpGraph at command path inside SuperGraph root."
                .to_string()
        })?;

    apply_milli_graph_edit_ops(graph, &command.ops)
}

fn apply_milli_graph_edit_ops(
    graph: &mut MilliOpGraph,
    ops: &[MilliOpGraphEditOp],
) -> Result<(), String> {
    for op in ops {
        match op {
            MilliOpGraphEditOp::SetNodeInputSlot {
                node_id,
                slot_index,
                link,
            } => {
                let input_node = graph
                    .get_op_mut(node_id)
                    .ok_or_else(|| format!("Missing input node {node_id}"))?;
                Node::set_input_slot(input_node, *slot_index, *link).map_err(|err| {
                    format!(
                        "Failed to set input slot {} on {}: {err:?}",
                        slot_index, node_id
                    )
                })?;
            }
            MilliOpGraphEditOp::SetNodeOutputSlot {
                node_id,
                slot_index,
                link,
            } => {
                let output_node = graph
                    .get_op_mut(node_id)
                    .ok_or_else(|| format!("Missing output node {node_id}"))?;
                Node::set_output_slot(output_node, *slot_index, *link).map_err(|err| {
                    format!(
                        "Failed to set output slot {} on {}: {err:?}",
                        slot_index, node_id
                    )
                })?;
            }
            MilliOpGraphEditOp::RetargetGraphOutputLink {
                old_internal_id,
                new_internal_id,
            } => {
                graph
                    .retarget_output_internal_link(*old_internal_id, *new_internal_id)
                    .map_err(|err| {
                        format!(
                            "Cannot retarget output sink link {} to {}: {err}",
                            old_internal_id, new_internal_id
                        )
                    })?;
            }
            MilliOpGraphEditOp::EnsureTensor { tensor_id } => {
                graph.ensure_tensor_with_id(*tensor_id);
            }
        }
    }

    Ok(())
}

fn resolve_editable_graph_in_supergraph_at_path<'a>(
    root_graph: &'a SuperGraph,
    path: &[GlobalId],
) -> Option<ResolvedEditableGraphRef<'a>> {
    let Some((head, tail)) = path.split_first() else {
        return Some(ResolvedEditableGraphRef::SuperGraph);
    };

    let node = root_graph.nodes.get(head)?;
    if let SuperGraphAnyNode::MilliOpGraph(milli_node) = node {
        if tail.is_empty() {
            return Some(ResolvedEditableGraphRef::Milli(&milli_node.graph));
        }
        return None;
    }
    node.get_sub_graph()
        .and_then(|next| resolve_editable_graph_in_supergraph_at_path(next, tail))
}

fn resolve_milli_graph_mut_in_supergraph_at_path<'a>(
    root_graph: &'a mut SuperGraph,
    path: &[GlobalId],
) -> Option<&'a mut MilliOpGraph> {
    let (head, tail) = path.split_first()?;

    let node = root_graph.nodes.get_mut(head)?;
    if let SuperGraphAnyNode::MilliOpGraph(milli_node) = node {
        if tail.is_empty() {
            return Some(&mut milli_node.graph);
        }
        return None;
    }
    node.get_sub_graph_mut()
        .and_then(|next| resolve_milli_graph_mut_in_supergraph_at_path(next, tail))
}

fn resolve_supergraph_at_path<'a>(
    root_graph: &'a SuperGraph,
    path: &[GlobalId],
) -> Option<&'a SuperGraph> {
    let mut graph = root_graph;
    for node_id in path {
        let node = graph.nodes.get(node_id)?;
        graph = node.get_sub_graph()?;
    }
    Some(graph)
}

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

fn supergraph_node_input_slot(
    graph: &SuperGraph,
    node_id: GlobalId,
    slot_index: usize,
) -> Result<Option<GlobalId>, String> {
    let node = graph
        .nodes
        .get(&node_id)
        .ok_or_else(|| format!("Missing input node {node_id}"))?;
    let Some(slot_link) = SuperGraphNode::input_slots(node).nth(slot_index) else {
        return Err(format!(
            "Input slot {} is out of range for node {}",
            slot_index, node_id
        ));
    };
    Ok(slot_link.map(|link| link.global_id()))
}

fn supergraph_node_output_slot(
    graph: &SuperGraph,
    node_id: GlobalId,
    slot_index: usize,
) -> Result<Option<GlobalId>, String> {
    let node = graph
        .nodes
        .get(&node_id)
        .ok_or_else(|| format!("Missing output node {node_id}"))?;
    let Some(slot_link) = SuperGraphNode::output_slots(node).nth(slot_index) else {
        return Err(format!(
            "Output slot {} is out of range for node {}",
            slot_index, node_id
        ));
    };
    Ok(slot_link.map(|link| link.global_id()))
}

fn resolve_supergraph_source_template_link(
    graph: &SuperGraph,
    output_endpoint: SlotPipEndpoint,
) -> Result<SuperGraphLink, String> {
    match output_endpoint.owner {
        SlotPipOwner::Node(node_id) => {
            let node = graph
                .nodes
                .get(&node_id)
                .ok_or_else(|| format!("Missing output node {node_id}"))?;
            SuperGraphNode::output_slots(node)
                .nth(output_endpoint.slot_index)
                .flatten()
                .ok_or_else(|| {
                    format!(
                        "Output slot {} on {} has no link kind to reuse",
                        output_endpoint.slot_index, node_id
                    )
                })
        }
        SlotPipOwner::InputLink(link_id) | SlotPipOwner::ConstantLink(link_id) => graph
            .links_by_global_id
            .get(&link_id)
            .map(|x| x.link())
            .ok_or_else(|| format!("Missing link metadata for endpoint link {}", link_id)),
        SlotPipOwner::OutputLink(link_id) => Err(format!(
            "Graph output link {} cannot be used as a source endpoint",
            link_id
        )),
    }
}

fn milli_graph_node_input_slot(
    graph: &MilliOpGraph,
    node_id: GlobalId,
    slot_index: usize,
) -> Result<Option<GlobalId>, String> {
    let node = Graph::get_node_by_id(graph, &node_id)
        .ok_or_else(|| format!("Missing input node {node_id}"))?;
    let Some(link) = Node::input_slots(node).nth(slot_index) else {
        return Err(format!(
            "Input slot {} is out of range for node {}",
            slot_index, node_id
        ));
    };
    Ok(link)
}

fn milli_graph_node_output_slot(
    graph: &MilliOpGraph,
    node_id: GlobalId,
    slot_index: usize,
) -> Result<Option<GlobalId>, String> {
    let node = Graph::get_node_by_id(graph, &node_id)
        .ok_or_else(|| format!("Missing output node {node_id}"))?;
    let Some(link) = Node::output_slots(node).nth(slot_index) else {
        return Err(format!(
            "Output slot {} is out of range for node {}",
            slot_index, node_id
        ));
    };
    Ok(link)
}

fn describe_endpoint(endpoint: &SlotPipEndpoint) -> String {
    match endpoint.owner {
        SlotPipOwner::Node(node_id) => {
            format!("node {} slot {}", node_id, endpoint.slot_index)
        }
        SlotPipOwner::InputLink(link_id) => {
            format!("graph input link {} slot {}", link_id, endpoint.slot_index)
        }
        SlotPipOwner::OutputLink(link_id) => {
            format!("graph output link {} slot {}", link_id, endpoint.slot_index)
        }
        SlotPipOwner::ConstantLink(link_id) => {
            format!("constant link {} slot {}", link_id, endpoint.slot_index)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::EditableClientGraphMut;
    use egui::Pos2;
    use whisper_tensor::graph::{Graph, SlotDirection};
    use whisper_tensor::milli_graph::ops::SimpleUnaryOp;
    use whisper_tensor::super_graph::nodes::{
        SuperGraphAnyNode, SuperGraphNode, SuperGraphNodeMilliOpGraph, SuperGraphNodeReportProgress,
    };
    use whisper_tensor::super_graph::{SuperGraphBuilder, SuperGraphLink};

    fn endpoint(
        owner: SlotPipOwner,
        direction: SlotDirection,
        slot_index: usize,
        link_id: Option<GlobalId>,
    ) -> SlotPipEndpoint {
        SlotPipEndpoint {
            owner,
            direction,
            slot_index,
            link_id,
            layout_link_id: None,
            screen_pos: Pos2::ZERO,
        }
    }

    #[test]
    fn supergraph_node_slot_edit_supports_undo_and_redo() {
        let mut rng = rand::rng();
        let mut builder = SuperGraphBuilder::new();

        let source_old = builder.new_tensor_link(&mut rng);
        let source_new = builder.new_tensor_link(&mut rng);
        let numerator = builder.new_tensor_link(&mut rng);
        let denominator = builder.new_tensor_link(&mut rng);
        let node = SuperGraphNodeReportProgress::new(source_old, numerator, denominator, &mut rng);
        let node_id = SuperGraphNode::global_id(&node);
        builder.add_node(node.to_any());

        let mut graph = builder.build(
            &mut rng,
            &[
                source_old.to_any(),
                source_new.to_any(),
                numerator.to_any(),
                denominator.to_any(),
            ],
            &[],
        );

        let before_slot = supergraph_node_input_slot(&graph, node_id, 0).unwrap();
        assert_eq!(before_slot, Some(source_old.global_id()));

        let output_endpoint = endpoint(
            SlotPipOwner::InputLink(source_new.global_id()),
            SlotDirection::Output,
            0,
            Some(source_new.global_id()),
        );
        let input_endpoint = endpoint(
            SlotPipOwner::Node(node_id),
            SlotDirection::Input,
            0,
            Some(source_old.global_id()),
        );

        let plan = plan_and_apply_link_drag_edit(
            EditableClientGraphMut::SuperGraph(&mut graph),
            &[],
            output_endpoint,
            input_endpoint,
        )
        .unwrap();

        assert!(!plan.requires_layout_refresh);
        assert_eq!(
            supergraph_node_input_slot(&graph, node_id, 0).unwrap(),
            Some(source_new.global_id())
        );

        apply_graph_edit_command(
            EditableClientGraphMut::SuperGraph(&mut graph),
            &plan.undo_command,
        )
        .unwrap();
        assert_eq!(
            supergraph_node_input_slot(&graph, node_id, 0).unwrap(),
            Some(source_old.global_id())
        );

        apply_graph_edit_command(
            EditableClientGraphMut::SuperGraph(&mut graph),
            &plan.forward_command,
        )
        .unwrap();
        assert_eq!(
            supergraph_node_input_slot(&graph, node_id, 0).unwrap(),
            Some(source_new.global_id())
        );
    }

    #[test]
    fn supergraph_output_retarget_supports_undo_and_redo() {
        let mut rng = rand::rng();
        let mut builder = SuperGraphBuilder::new();

        let old_source = builder.new_tensor_link(&mut rng);
        let new_source = builder.new_tensor_link(&mut rng);
        let mut graph = builder.build(
            &mut rng,
            &[old_source.to_any(), new_source.to_any()],
            &[old_source.to_any()],
        );

        let output_endpoint = endpoint(
            SlotPipOwner::InputLink(new_source.global_id()),
            SlotDirection::Output,
            0,
            Some(new_source.global_id()),
        );
        let input_endpoint = endpoint(
            SlotPipOwner::OutputLink(old_source.global_id()),
            SlotDirection::Input,
            0,
            Some(old_source.global_id()),
        );

        let plan = plan_and_apply_link_drag_edit(
            EditableClientGraphMut::SuperGraph(&mut graph),
            &[],
            output_endpoint,
            input_endpoint,
        )
        .unwrap();

        assert!(plan.requires_layout_refresh);
        assert!(
            graph
                .output_links
                .iter()
                .any(|x| x.global_id() == new_source.global_id()),
            "expected output link to retarget to new source"
        );
        assert!(
            !graph
                .output_links
                .iter()
                .any(|x| x.global_id() == old_source.global_id()),
            "expected old output link to be removed"
        );

        apply_graph_edit_command(
            EditableClientGraphMut::SuperGraph(&mut graph),
            &plan.undo_command,
        )
        .unwrap();
        assert!(
            graph
                .output_links
                .iter()
                .any(|x| x.global_id() == old_source.global_id()),
            "expected undo to restore old output link"
        );
        assert!(
            !graph
                .output_links
                .iter()
                .any(|x| x.global_id() == new_source.global_id()),
            "expected undo to remove new output link"
        );

        apply_graph_edit_command(
            EditableClientGraphMut::SuperGraph(&mut graph),
            &plan.forward_command,
        )
        .unwrap();
        assert!(
            graph
                .output_links
                .iter()
                .any(|x| x.global_id() == new_source.global_id()),
            "expected redo to restore new output link"
        );
    }

    #[test]
    fn milli_node_slot_edit_supports_undo_and_redo() {
        let mut rng = rand::rng();
        let external_a = GlobalId::new(&mut rng);
        let external_b = GlobalId::new(&mut rng);
        let (mut graph, input_map) = MilliOpGraph::new([external_a, external_b], &mut rng);
        let internal_a = input_map[&external_a];
        let internal_b = input_map[&external_b];

        let _out = SimpleUnaryOp::neg(&mut graph, internal_a, &mut rng);
        let node_id = Graph::node_ids(&graph).next().unwrap();

        let output_endpoint = endpoint(
            SlotPipOwner::InputLink(internal_b),
            SlotDirection::Output,
            0,
            Some(internal_b),
        );
        let input_endpoint = endpoint(
            SlotPipOwner::Node(node_id),
            SlotDirection::Input,
            0,
            Some(internal_a),
        );

        let plan = plan_and_apply_link_drag_edit(
            EditableClientGraphMut::MilliOpGraph(&mut graph),
            &[],
            output_endpoint,
            input_endpoint,
        )
        .unwrap();

        assert!(!plan.requires_layout_refresh);
        assert_eq!(
            milli_graph_node_input_slot(&graph, node_id, 0).unwrap(),
            Some(internal_b)
        );

        apply_graph_edit_command(
            EditableClientGraphMut::MilliOpGraph(&mut graph),
            &plan.undo_command,
        )
        .unwrap();
        assert_eq!(
            milli_graph_node_input_slot(&graph, node_id, 0).unwrap(),
            Some(internal_a)
        );

        apply_graph_edit_command(
            EditableClientGraphMut::MilliOpGraph(&mut graph),
            &plan.forward_command,
        )
        .unwrap();
        assert_eq!(
            milli_graph_node_input_slot(&graph, node_id, 0).unwrap(),
            Some(internal_b)
        );
    }

    #[test]
    fn nested_milli_inside_supergraph_supports_edit_undo_redo() {
        let mut rng = rand::rng();
        let external_input_a = GlobalId::new(&mut rng);
        let external_input_b = GlobalId::new(&mut rng);
        let external_output = GlobalId::new(&mut rng);

        let (mut inner_milli, input_map) =
            MilliOpGraph::new([external_input_a, external_input_b], &mut rng);
        let internal_a = input_map[&external_input_a];
        let internal_b = input_map[&external_input_b];
        let _inner_output = SimpleUnaryOp::neg(&mut inner_milli, internal_a, &mut rng);
        let inner_node_id = Graph::node_ids(&inner_milli).next().unwrap();
        inner_milli.set_output_map([(internal_a, external_output)]);

        let mut builder = SuperGraphBuilder::new();
        let milli_node =
            SuperGraphAnyNode::MilliOpGraph(SuperGraphNodeMilliOpGraph::new(inner_milli, &mut rng));
        let milli_node_id = builder.add_node(milli_node);
        let mut graph = builder.build(
            &mut rng,
            &[
                SuperGraphLink::tensor(external_input_a).to_any(),
                SuperGraphLink::tensor(external_input_b).to_any(),
            ],
            &[SuperGraphLink::tensor(external_output).to_any()],
        );

        let output_endpoint = endpoint(
            SlotPipOwner::InputLink(internal_b),
            SlotDirection::Output,
            0,
            Some(internal_b),
        );
        let input_endpoint = endpoint(
            SlotPipOwner::Node(inner_node_id),
            SlotDirection::Input,
            0,
            Some(internal_a),
        );
        let working_path = vec![milli_node_id];

        let plan = plan_and_apply_link_drag_edit(
            EditableClientGraphMut::SuperGraph(&mut graph),
            &working_path,
            output_endpoint,
            input_endpoint,
        )
        .unwrap();

        assert!(!plan.requires_layout_refresh);
        let GraphEditCommand::MilliOpGraph(forward_batch) = &plan.forward_command else {
            panic!("expected milli command");
        };
        assert_eq!(forward_batch.graph_path, working_path);

        let nested_graph = match resolve_editable_graph_in_supergraph_at_path(&graph, &working_path)
        {
            Some(ResolvedEditableGraphRef::Milli(nested)) => nested,
            Some(ResolvedEditableGraphRef::SuperGraph) => panic!("expected nested milli graph"),
            None => panic!("expected working path to resolve"),
        };
        assert_eq!(
            milli_graph_node_input_slot(nested_graph, inner_node_id, 0).unwrap(),
            Some(internal_b)
        );

        apply_graph_edit_command(
            EditableClientGraphMut::SuperGraph(&mut graph),
            &plan.undo_command,
        )
        .unwrap();
        let nested_graph = match resolve_editable_graph_in_supergraph_at_path(&graph, &working_path)
        {
            Some(ResolvedEditableGraphRef::Milli(nested)) => nested,
            Some(ResolvedEditableGraphRef::SuperGraph) => panic!("expected nested milli graph"),
            None => panic!("expected working path to resolve"),
        };
        assert_eq!(
            milli_graph_node_input_slot(nested_graph, inner_node_id, 0).unwrap(),
            Some(internal_a)
        );

        apply_graph_edit_command(
            EditableClientGraphMut::SuperGraph(&mut graph),
            &plan.forward_command,
        )
        .unwrap();
        let nested_graph = match resolve_editable_graph_in_supergraph_at_path(&graph, &working_path)
        {
            Some(ResolvedEditableGraphRef::Milli(nested)) => nested,
            Some(ResolvedEditableGraphRef::SuperGraph) => panic!("expected nested milli graph"),
            None => panic!("expected working path to resolve"),
        };
        assert_eq!(
            milli_graph_node_input_slot(nested_graph, inner_node_id, 0).unwrap(),
            Some(internal_b)
        );
    }

    #[test]
    fn milli_output_retarget_supports_undo_and_redo() {
        let mut rng = rand::rng();
        let external_old = GlobalId::new(&mut rng);
        let external_new = GlobalId::new(&mut rng);
        let (mut graph, input_map) = MilliOpGraph::new([external_old, external_new], &mut rng);
        let old_internal = input_map[&external_old];
        let new_internal = input_map[&external_new];
        graph.set_outputs(vec![old_internal]);

        let output_endpoint = endpoint(
            SlotPipOwner::InputLink(new_internal),
            SlotDirection::Output,
            0,
            Some(new_internal),
        );
        let input_endpoint = endpoint(
            SlotPipOwner::OutputLink(old_internal),
            SlotDirection::Input,
            0,
            Some(old_internal),
        );

        let plan = plan_and_apply_link_drag_edit(
            EditableClientGraphMut::MilliOpGraph(&mut graph),
            &[],
            output_endpoint,
            input_endpoint,
        )
        .unwrap();

        assert!(plan.requires_layout_refresh);
        assert!(
            Graph::output_link_ids(&graph).any(|(_, internal_id)| internal_id == new_internal),
            "expected output retarget to new internal link"
        );

        apply_graph_edit_command(
            EditableClientGraphMut::MilliOpGraph(&mut graph),
            &plan.undo_command,
        )
        .unwrap();
        assert!(
            Graph::output_link_ids(&graph).any(|(_, internal_id)| internal_id == old_internal),
            "expected undo to restore old internal output link"
        );

        apply_graph_edit_command(
            EditableClientGraphMut::MilliOpGraph(&mut graph),
            &plan.forward_command,
        )
        .unwrap();
        assert!(
            Graph::output_link_ids(&graph).any(|(_, internal_id)| internal_id == new_internal),
            "expected redo to restore new internal output link"
        );
    }
}
