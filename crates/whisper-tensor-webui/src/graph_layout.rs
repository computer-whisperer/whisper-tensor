use crate::app::InterfaceId;
use egui::{Rect, UiBuilder, vec2};
use rand::{random, random_range};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use whisper_tensor::super_graph::links::SuperGraphAnyLink;
use whisper_tensor_server::LoadedModelId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct GraphLayoutNodeId(pub(crate) usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct GraphLayoutLinkId(pub(crate) usize);

#[derive(Debug, Clone)]
pub(crate) struct GraphLayoutIOOffsets {
    pub(crate) inputs: Vec<egui::Vec2>,
    pub(crate) outputs: Vec<egui::Vec2>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum GraphLayoutNodeType {
    SymbolicGraphOperation((LoadedModelId, whisper_tensor::symbolic_graph::OperationId)),
    SymbolicGraphTensor((LoadedModelId, whisper_tensor::symbolic_graph::TensorId)),
    SuperGraphLink((InterfaceId, whisper_tensor::super_graph::SuperGraphAnyLink)),
    SuperGraphNode((InterfaceId, whisper_tensor::super_graph::SuperGraphNodeId)),
    ConnectionByNameSrc(GraphLayoutLinkData),
    ConnectionByNameDest(GraphLayoutLinkData),
}

#[derive(Debug, Clone)]
pub(crate) struct GraphLayoutNode {
    pub(crate) node_type: GraphLayoutNodeType,
    pub(crate) position: egui::Pos2,
    pub(crate) velocity: egui::Vec2,
    pub(crate) shape: egui::Vec2,
    pub(crate) inputs: Vec<GraphLayoutLinkId>,
    pub(crate) outputs: Vec<GraphLayoutLinkId>,
    pub(crate) io_offsets: GraphLayoutIOOffsets,
}

#[derive(Debug, Clone)]
pub(crate) struct GraphLayoutNodeInitData {
    pub node_type: GraphLayoutNodeType,
    pub inputs: Vec<GraphLayoutLinkId>,
    pub outputs: Vec<GraphLayoutLinkId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum GraphLayoutLinkType {
    SymbolicGraphTensor((LoadedModelId, whisper_tensor::symbolic_graph::TensorId)),
    SuperGraphLink((InterfaceId, SuperGraphAnyLink)),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct GraphLayoutLinkData {
    pub(crate) link_type: GraphLayoutLinkType,
}

pub(crate) struct GraphLayout {
    nodes: HashMap<GraphLayoutNodeId, GraphLayoutNode>,
    node_map: HashMap<(i32, i32), Vec<GraphLayoutNodeId>>,
    edges: Vec<(
        (GraphLayoutNodeId, usize),
        (GraphLayoutNodeId, usize),
        GraphLayoutLinkId,
    )>,
    layout_clock: f32,
    max_cell_shape: egui::Vec2,
    bounding_rect: Rect,
    upstream_node_for_link: HashMap<GraphLayoutLinkId, (GraphLayoutNodeId, usize)>,
    downstream_nodes_for_link: HashMap<GraphLayoutLinkId, Vec<(GraphLayoutNodeId, usize)>>,
    link_data: HashMap<GraphLayoutLinkId, GraphLayoutLinkData>,
}

fn calculate_height(
    node_id: GraphLayoutNodeId,
    nodes: &HashMap<GraphLayoutNodeId, GraphLayoutNodeInitData>,
    upstream_node_for_link: &HashMap<GraphLayoutLinkId, (GraphLayoutNodeId, usize)>,
    node_heights: &mut HashMap<GraphLayoutNodeId, usize>,
) -> usize {
    if let Some(x) = node_heights.get(&node_id) {
        *x
    } else {
        let mut min_height = 0;
        for link_id in &nodes[&node_id].inputs {
            let node_id = upstream_node_for_link[link_id].0;
            min_height = min_height
                .max(1 + calculate_height(node_id, nodes, upstream_node_for_link, node_heights));
        }
        node_heights.insert(node_id, min_height);
        min_height
    }
}

impl GraphLayout {
    fn get_index_for(pos: &egui::Pos2) -> (i32, i32) {
        ((pos.x / 1000.0) as i32, (pos.y / 1000.0) as i32)
    }

    pub(crate) fn find_nodes_within(
        &self,
        pos: &egui::Pos2,
        distance: f32,
    ) -> Vec<GraphLayoutNodeId> {
        let top_left = egui::pos2(pos.x - distance, pos.y - distance);
        let bot_right = egui::pos2(pos.x + distance, pos.y + distance);
        let top_left_index = Self::get_index_for(&top_left);
        let bot_right_index = Self::get_index_for(&bot_right);
        let mut ret = vec![];
        for x in top_left_index.0..bot_right_index.0 + 1 {
            for y in top_left_index.1..bot_right_index.1 + 1 {
                if let Some(stuff) = self.node_map.get(&(x, y)) {
                    for op_id in stuff {
                        if let Some(op_pos) = self.nodes.get(op_id) {
                            if op_pos.position.distance(*pos) < distance {
                                ret.push(*op_id)
                            }
                        }
                    }
                }
            }
        }
        ret
    }

    pub(crate) fn new(
        input_node_init_data: HashMap<GraphLayoutNodeId, GraphLayoutNodeInitData>,
        link_data: HashMap<GraphLayoutLinkId, GraphLayoutLinkData>,
        ui: &mut egui::Ui,
        node_render_tester: impl Fn(&mut egui::Ui, &GraphLayoutNodeInitData) -> GraphLayoutIOOffsets,
    ) -> Self {
        let mut max_existing_node_id = 0;
        let mut max_existing_link_id = 0;
        for (id, data) in &input_node_init_data {
            max_existing_node_id = max_existing_node_id.max(id.0);
            for link_id in &data.inputs {
                max_existing_link_id = max_existing_link_id.max(link_id.0);
            }
            for link_id in &data.outputs {
                max_existing_link_id = max_existing_link_id.max(link_id.0);
            }
        }
        for (id, _) in &link_data {
            max_existing_link_id = max_existing_link_id.max(id.0);
        }
        let mut next_node_id = max_existing_node_id + 1;
        let mut next_link_id = max_existing_link_id + 1;

        // Resolve upstream/downstream nodes for each link
        let mut upstream_node_for_link = HashMap::new();
        let mut downstream_nodes_for_link = HashMap::new();
        for (node_id, node_data) in &input_node_init_data {
            for (i, link_id) in node_data.outputs.iter().enumerate() {
                upstream_node_for_link.insert(*link_id, (*node_id, i));
            }
            for (i, link_id) in node_data.inputs.iter().enumerate() {
                downstream_nodes_for_link
                    .entry(*link_id)
                    .or_insert_with(Vec::new)
                    .push((*node_id, i));
            }
        }

        // Get heights
        let mut max_height = 0;
        let mut node_heights = HashMap::new();
        for (op_id, _op) in &input_node_init_data {
            let height = calculate_height(
                *op_id,
                &input_node_init_data,
                &upstream_node_for_link,
                &mut node_heights,
            );
            max_height = max_height.max(height);
        }

        // Sort by height
        let mut nodes_and_heights = node_heights.clone().into_iter().collect::<Vec<_>>();
        nodes_and_heights.sort_by(|(_, a), (_, b)| b.cmp(a));

        // Iterate over backwards and push nodes up to right under their children

        for (node_id, _) in nodes_and_heights.iter() {
            let mut upper_bound = None;
            for link_id in &input_node_init_data[node_id].outputs {
                for (node_id, _) in &downstream_nodes_for_link[link_id] {
                    let height = &node_heights[&node_id];
                    if let Some(x) = upper_bound.clone() {
                        if *height < x {
                            upper_bound = Some(*height);
                        }
                    } else {
                        upper_bound = Some(*height);
                    }
                }
            }

            if let Some(upper_bound) = upper_bound {
                node_heights.insert(*node_id, upper_bound - 1);
            }
        }

        // Break long edges with connection by name (involves rewriting the input list)
        let mut link_data = link_data;
        let mut node_init_data = HashMap::new();
        let mut links_with_source_node = HashSet::new();
        for (node_id, _op) in &input_node_init_data {
            let node = &input_node_init_data[node_id];
            let mut new_node_inputs = vec![];
            for link_id in &node.inputs {
                let (upstream_node_id, _) = upstream_node_for_link[link_id];
                let height_delta = node_heights[node_id] - node_heights[&upstream_node_id];
                if height_delta > 10 {
                    // Break into connection by name

                    if !links_with_source_node.contains(link_id) {
                        links_with_source_node.insert(*link_id);
                        let new_node = GraphLayoutNodeInitData {
                            node_type: GraphLayoutNodeType::ConnectionByNameSrc(
                                link_data[link_id].clone(),
                            ),
                            inputs: vec![*link_id],
                            outputs: vec![],
                        };
                        let new_node_id = GraphLayoutNodeId(next_node_id);
                        next_node_id += 1;

                        node_init_data.insert(new_node_id, new_node);
                        node_heights.insert(new_node_id, node_heights[&upstream_node_id] + 1);
                    }

                    // Add new dest node
                    let new_link_id = GraphLayoutLinkId(next_link_id);
                    link_data.insert(new_link_id, link_data[link_id].clone());
                    next_link_id += 1;
                    let new_node = GraphLayoutNodeInitData {
                        node_type: GraphLayoutNodeType::ConnectionByNameDest(
                            link_data[link_id].clone(),
                        ),
                        inputs: vec![],
                        outputs: vec![new_link_id],
                    };
                    let new_node_id = GraphLayoutNodeId(next_node_id);
                    next_node_id += 1;
                    node_init_data.insert(new_node_id, new_node);
                    node_heights.insert(new_node_id, node_heights[node_id] - 1);

                    new_node_inputs.push(new_link_id);
                } else {
                    new_node_inputs.push(*link_id)
                }
            }
            node_init_data.insert(
                *node_id,
                GraphLayoutNodeInitData {
                    node_type: node.node_type.clone(),
                    inputs: new_node_inputs,
                    outputs: node.outputs.clone(),
                },
            );
        }
        drop(input_node_init_data);

        // Re-resolve upstream/downstream nodes for each link
        let mut upstream_node_for_link = HashMap::new();
        let mut downstream_nodes_for_link = HashMap::new();
        for (node_id, node_data) in &node_init_data {
            for (i, link_id) in node_data.outputs.iter().enumerate() {
                upstream_node_for_link.insert(*link_id, (*node_id, i));
            }
            for (i, link_id) in node_data.inputs.iter().enumerate() {
                downstream_nodes_for_link
                    .entry(*link_id)
                    .or_insert_with(Vec::new)
                    .push((*node_id, i));
            }
        }

        let mut node_shapes = HashMap::new();
        let mut node_io_offsets = HashMap::new();
        let mut max_node_shape = egui::Vec2::ZERO;
        for (node_id, node_data) in node_init_data.iter_mut() {
            let ui_builder = UiBuilder::new();
            let mut ui_child = ui.new_child(ui_builder);
            ui_child.set_invisible();
            let io_offsets = node_render_tester(&mut ui_child, &node_data);
            node_io_offsets.insert(*node_id, io_offsets);
            let new_shape = ui_child.min_size();
            max_node_shape = max_node_shape.max(new_shape);
            node_shapes.insert(*node_id, new_shape);
        }

        // Write edges list
        let mut edges = vec![];
        for (node_id, node_data) in &node_init_data {
            for (i, link_id) in node_data.inputs.iter().enumerate() {
                let (upstream_node_id, j) = &upstream_node_for_link[link_id];
                edges.push(((*upstream_node_id, *j), (*node_id, i), *link_id));
            }
        }

        // Sort by height again
        let mut nodes_and_heights = node_heights.clone().into_iter().collect::<Vec<_>>();
        nodes_and_heights.sort_by(|(_, a), (_, b)| b.cmp(a));

        // Resolve y positions
        let mut node_y_positions = HashMap::new();
        let mut height_val = nodes_and_heights.first().unwrap().1;
        let mut height_siblings = vec![];
        let mut i = 0;
        loop {
            // Collect all values with the same height
            loop {
                if i >= nodes_and_heights.len() {
                    break;
                }
                let (op_id, height) = nodes_and_heights[i];
                if height < height_val {
                    break;
                } else {
                    height_siblings.push(op_id);
                }
                i += 1;
            }
            if height_siblings.len() > 0 {
                // Get avg lateral position of downstream nodes
                let mut lat_means = vec![];
                let mut max_shape = egui::Vec2::ZERO;
                for node_id in &height_siblings {
                    let node = &node_init_data[node_id];
                    max_shape = max_shape.max(node_shapes[node_id]);
                    let mut num = 0;
                    let mut total = 0.0;
                    for link_id in &node.outputs {
                        for (downstream_node_id, input_idx) in &downstream_nodes_for_link[link_id] {
                            let link_offset = (*input_idx as f32
                                / node_init_data[downstream_node_id].inputs.len() as f32)
                                * 0.2;
                            num += 1;
                            total += node_y_positions[downstream_node_id] + link_offset;
                        }
                    }
                    if num > 0 {
                        lat_means.push((node_id, total / num as f32))
                    } else {
                        lat_means.push((node_id, 0.0))
                    }
                }
                // Sort and place
                lat_means.sort_by(|(_, a), (_, b)| {
                    if a > b {
                        Ordering::Greater
                    } else {
                        Ordering::Less
                    }
                });
                let num_vals = lat_means.len();
                // Get total len
                for (i, (a, _b)) in lat_means.into_iter().enumerate() {
                    let y = ((i as f32) - (num_vals as f32 / 2.0)) * (max_shape.y + 30.0);
                    node_y_positions.insert(*a, y);
                }
                height_siblings.clear();
            }
            if height_val == 0 {
                break;
            }
            height_val -= 1;
            if i >= nodes_and_heights.len() {
                break;
            }
        }

        // Calculate node x positions
        // Sort by height again
        let mut nodes_and_heights = node_heights.clone().into_iter().collect::<Vec<_>>();
        nodes_and_heights.sort_by(|(_, a), (_, b)| a.cmp(b));

        let mut node_x_positions = HashMap::new();
        let mut last_height_max_x = 0.0f32;
        let mut this_height_max_x = 0.0f32;
        let mut last_height = 0;
        for (node_id, height) in nodes_and_heights.iter() {
            if *height != last_height {
                last_height = *height;
                last_height_max_x = this_height_max_x;
            }
            let margin = 50.0;
            let x_pos = last_height_max_x + margin;
            node_x_positions.insert(*node_id, x_pos + node_shapes[node_id].x / 2.0);
            this_height_max_x = this_height_max_x.max(x_pos + node_shapes[node_id].x);
        }

        let mut nodes = HashMap::new();
        let mut node_map = HashMap::new();
        for (node_id, data) in node_init_data {
            let pos = egui::pos2(node_x_positions[&node_id], node_y_positions[&node_id]);
            let vel = egui::vec2(random::<f32>(), random::<f32>());

            node_map
                .entry(Self::get_index_for(&pos))
                .or_insert(Vec::new())
                .push(node_id);

            let new_node = GraphLayoutNode {
                node_type: data.node_type,
                position: pos,
                velocity: vel,
                shape: node_shapes[&node_id],
                inputs: data.inputs,
                outputs: data.outputs,
                io_offsets: node_io_offsets[&node_id].clone(),
            };
            nodes.insert(node_id, new_node);
        }

        let bounding_rect = Self::get_bounding_rect_from_nodes(&nodes);

        Self {
            nodes,
            node_map,
            edges,
            layout_clock: 0.0,
            max_cell_shape: max_node_shape,
            bounding_rect,
            upstream_node_for_link,
            downstream_nodes_for_link,
            link_data,
        }
    }

    pub(crate) fn get_nodes(&self) -> &HashMap<GraphLayoutNodeId, GraphLayoutNode> {
        &self.nodes
    }

    pub(crate) fn get_edges(
        &self,
    ) -> &Vec<(
        (GraphLayoutNodeId, usize),
        (GraphLayoutNodeId, usize),
        GraphLayoutLinkId,
    )> {
        &self.edges
    }

    pub(crate) fn get_link_data(&self) -> &HashMap<GraphLayoutLinkId, GraphLayoutLinkData> {
        &self.link_data
    }

    pub(crate) fn get_nodes_mut(&mut self) -> &mut HashMap<GraphLayoutNodeId, GraphLayoutNode> {
        &mut self.nodes
    }

    pub(crate) fn get_bounding_rect(&self) -> Rect {
        self.bounding_rect
    }

    fn get_bounding_rect_from_nodes(nodes: &HashMap<GraphLayoutNodeId, GraphLayoutNode>) -> Rect {
        let mut min_vec = egui::pos2(0.0, 0.0);
        let mut max_vec = egui::pos2(0.0, 0.0);
        for (_id, node) in nodes {
            min_vec = min_vec.min(node.position - node.shape / 2.0);
            max_vec = max_vec.max(node.position + node.shape / 2.0);
        }
        Rect::from_min_max(min_vec, max_vec)
    }

    pub(crate) fn update_layout(&mut self, max_nodes_to_update: u32) -> bool {
        let mut did_update = false;
        for _ in 0..max_nodes_to_update {
            let i = random_range(0..self.nodes.len());
            let op_id = *self.nodes.keys().nth(i).unwrap();
            let node_data = &self.nodes[&op_id];
            let mut applied_force = (0.0, 0.0);
            for other_node in
                self.find_nodes_within(&node_data.position, self.max_cell_shape.length() * 1.5)
            {
                if other_node == op_id {
                    continue;
                }
                let other_node_data = &self.nodes[&other_node];
                let delta = node_data.position - other_node_data.position;
                let distance = delta.length();
                let normalized_delta = (delta.x / distance, delta.y / distance);

                if distance > 0.01 {
                    let min_x_dist = (node_data.shape.x + other_node_data.shape.x) * 0.8;
                    let min_y_dist = (node_data.shape.y + other_node_data.shape.y) * 0.8;
                    if delta.x.abs() < min_x_dist && delta.y.abs() < min_y_dist {
                        let force = (node_data.shape.length() + other_node_data.shape.length())
                            * 1.4
                            / distance;
                        applied_force.0 += force * normalized_delta.0;
                        applied_force.1 += force * normalized_delta.1;
                    }
                }
            }
            let mut links = vec![];
            for (i, link_id) in node_data.inputs.iter().enumerate() {
                let (upstream_node_id, j) = self.upstream_node_for_link[link_id];
                links.push(((upstream_node_id, j), (op_id, i)));
            }
            for (i, link_id) in node_data.outputs.iter().enumerate() {
                for (downstream_node_id, j) in &self.downstream_nodes_for_link[link_id] {
                    links.push(((op_id, i), (*downstream_node_id, *j)));
                }
            }

            for ((src_node_id, src_i), (dst_node_id, dst_i)) in links {
                // Applied to dst, inverse applied to src
                let mut link_force = (0.0, 0.0);
                let src_data = &self.nodes[&src_node_id];
                let dst_data = &self.nodes[&dst_node_id];

                let source_io_position = src_data.position + src_data.io_offsets.outputs[src_i];
                let dest_io_position = dst_data.position + dst_data.io_offsets.inputs[dst_i];

                let delta = dest_io_position - source_io_position;
                let distance = delta.length();
                let normalized_delta = (delta.x / distance, delta.y / distance);
                let force = if distance > 40.0 {
                    (-distance / 300.0).min(0.5)
                } else {
                    0.0
                };

                link_force.0 = force * normalized_delta.0;
                link_force.1 = force * normalized_delta.1;
                let hierarchy_error = src_data.position.x
                    + (src_data.shape.x + dst_data.shape.x) * 0.8
                    - dst_data.position.x;
                if hierarchy_error > 0.0 {
                    // Inputs must be pushed above the node
                    link_force.0 += 5.0 + 0.01 * hierarchy_error;
                }

                if dst_node_id == op_id {
                    applied_force.0 += link_force.0;
                    applied_force.1 += link_force.1;
                }
                if src_node_id == op_id {
                    applied_force.0 -= link_force.0;
                    applied_force.1 -= link_force.1;
                }
            }

            // Weak draw towards 0,0
            //applied_force.0 -= node_pos.x * 0.0001;
            applied_force.1 -= node_data.position.y * 0.0001;

            // Constrain force
            let temperature = (4000.0 / self.layout_clock).min(1.0);
            let applied_force = (applied_force.0 * temperature, applied_force.1 * temperature);
            let mut velocity = node_data.velocity;
            velocity.x += applied_force.0;
            velocity.y += applied_force.1;
            // Dampen
            velocity.x -= velocity.x * 0.3;
            velocity.y -= velocity.y * 0.3;
            // Clip velocity magnitude
            let velocity_magnitude = velocity.length();
            let clipped_velocity = velocity_magnitude.min(15.0);
            let velocity = egui::vec2(
                clipped_velocity * velocity.x / velocity_magnitude,
                clipped_velocity * velocity.y / velocity_magnitude,
            );

            let velocity_magnitude = velocity.length();
            let min_movement = self.layout_clock / 6000.0;
            if velocity_magnitude.is_finite() && velocity_magnitude > min_movement {
                did_update = true;
                let old_index = Self::get_index_for(&node_data.position);
                let new_position = node_data.position + velocity;
                let new_index = Self::get_index_for(&new_position);
                if old_index != new_index {
                    // Update position on map
                    if let Some(x) = self.node_map.get_mut(&old_index) {
                        // Remove
                        x.retain_mut(|x| *x != op_id);
                    } else {
                        // Should not be possible
                        panic!();
                    }
                    // Add to map
                    self.node_map
                        .entry(new_index)
                        .or_insert(Vec::new())
                        .push(op_id);
                }

                self.nodes.get_mut(&op_id).unwrap().position = new_position;
                self.nodes.get_mut(&op_id).unwrap().velocity = velocity;
            } else {
                self.nodes.get_mut(&op_id).unwrap().velocity = vec2(0.0, 0.0);
            }
        }
        self.layout_clock += max_nodes_to_update as f32 / self.nodes.len() as f32;
        if did_update {
            self.bounding_rect = Self::get_bounding_rect_from_nodes(&self.nodes)
        }
        did_update
    }
}
