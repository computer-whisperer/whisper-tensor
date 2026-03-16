use super::{SlotPipEndpoint, SlotPipOwner};
use crate::app::EditableClientGraphMut;
use whisper_tensor::graph::{GlobalId, Graph, Node};
use whisper_tensor::milli_graph::MilliOpGraph;
use whisper_tensor::super_graph::nodes::SuperGraphNode;
use whisper_tensor::super_graph::{SuperGraph, SuperGraphLink, SuperGraphLinkInfo};

#[derive(Clone, Debug)]
pub(super) struct GraphEditPlan {
    pub(super) status: String,
    pub(super) link_global_id: GlobalId,
    pub(super) requires_layout_refresh: bool,
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

pub(super) fn plan_and_apply_link_drag_edit(
    graph: EditableClientGraphMut<'_>,
    working_path: &[GlobalId],
    output_endpoint: SlotPipEndpoint,
    input_endpoint: SlotPipEndpoint,
) -> Result<GraphEditPlan, String> {
    match graph {
        EditableClientGraphMut::SuperGraph(root_graph) => {
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
        (EditableClientGraphMut::MilliOpGraph(graph), GraphEditCommand::MilliOpGraph(batch)) => {
            apply_milli_graph_edit_command(graph, batch)
        }
        (EditableClientGraphMut::SuperGraph(_), GraphEditCommand::MilliOpGraph(_)) => {
            Err("graph edit command mismatch: expected SuperGraph command".to_string())
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
        link_global_id: target_link.global_id(),
        requires_layout_refresh,
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
    if !working_path.is_empty() {
        return Err("MilliOpGraph link editing does not support nested graph paths.".to_string());
    }

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
        link_global_id,
        requires_layout_refresh,
        forward_command,
        undo_command,
    })
}

fn apply_milli_graph_edit_command(
    graph: &mut MilliOpGraph,
    command: &MilliOpGraphEditCommandBatch,
) -> Result<(), String> {
    if !command.graph_path.is_empty() {
        return Err("MilliOpGraph edit command path must be empty.".to_string());
    }

    for op in &command.ops {
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
