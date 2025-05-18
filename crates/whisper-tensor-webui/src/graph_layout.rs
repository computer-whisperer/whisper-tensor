use std::cmp::Ordering;
use std::collections::HashMap;
use rand::{random, random_range};
use egui::vec2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct GraphLayoutNodeId(pub(crate) u32);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum GraphLayoutNodeType {
    SymbolicGraphOperation(whisper_tensor::symbolic_graph::OperationId),
    SymbolicGraphTensor(whisper_tensor::symbolic_graph::TensorId),
    ConnectionByNameSrc(String),
    ConnectionByNameDest(String)
}

#[derive(Debug, Clone)]
pub(crate) struct GraphLayoutNode {
    pub(crate) node_type: GraphLayoutNodeType,
    pub(crate) position: egui::Pos2,
    pub(crate) velocity: egui::Vec2,
    pub(crate) shape: egui::Vec2,
    inputs: Vec<GraphLayoutNodeId>,
}

#[derive(Debug, Clone)]
pub(crate) struct GraphLayoutNodeInitData {
    pub node_type: GraphLayoutNodeType,
    pub shape: egui::Vec2,
    pub inputs: Vec<GraphLayoutNodeId>
}

pub(crate) struct GraphLayout {
    nodes: HashMap<GraphLayoutNodeId, GraphLayoutNode>,
    node_map: HashMap<(i32, i32), Vec<GraphLayoutNodeId>>,
    edges: Vec<(GraphLayoutNodeId, GraphLayoutNodeId)>,
    op_inputs: HashMap<GraphLayoutNodeId, Vec<GraphLayoutNodeId>>,
    op_outputs: HashMap<GraphLayoutNodeId, Vec<GraphLayoutNodeId>>,
    layout_clock: f32,
    max_cell_shape: egui::Vec2
}

fn calculate_height(node_id: GraphLayoutNodeId, nodes: &HashMap<GraphLayoutNodeId, GraphLayoutNodeInitData>, node_heights: &mut HashMap<GraphLayoutNodeId, usize>) -> usize {
    if let Some(x) = node_heights.get(&node_id) {
        *x
    } else {
        let mut min_height = 0;
        for input in &nodes[&node_id].inputs {
            min_height = min_height.max(1 + calculate_height(*input, nodes, node_heights));
        }
        node_heights.insert(node_id, min_height);
        min_height
    }
}

impl GraphLayout {

    fn get_index_for(pos: &egui::Pos2) -> (i32, i32) {
        ((pos.x/1000.0) as i32, (pos.y/1000.0) as i32)
    }

    pub(crate) fn find_nodes_within(&self, pos: &egui::Pos2, distance: f32) -> Vec<GraphLayoutNodeId> {
        let top_left = egui::pos2(pos.x - distance, pos.y - distance);
        let bot_right = egui::pos2(pos.x + distance, pos.y + distance);
        let top_left_index = Self::get_index_for(&top_left);
        let bot_right_index = Self::get_index_for(&bot_right);
        let mut ret = vec![];
        for x in top_left_index.0 .. bot_right_index.0+1 {
            for y in top_left_index.1 .. bot_right_index.1+1 {
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

    pub(crate) fn new(input_init_data: HashMap<GraphLayoutNodeId, GraphLayoutNodeInitData>) -> Self {
        // Get max op shape
        let mut max_existing_id = 0;
        let mut max_node_shape = vec2(1.0, 1.0);
        for (id, data) in &input_init_data {
            max_node_shape.x = max_node_shape.x.max(data.shape.x);
            max_node_shape.y = max_node_shape.y.max(data.shape.y);
            max_existing_id = max_existing_id.max(id.0);
        }
        let mut next_node_id = max_existing_id + 1;

        // get edges
        let mut op_inputs = HashMap::new();
        let mut op_outputs = HashMap::new();
        for (op_id, data) in &input_init_data {
            for input in &data.inputs {
                op_inputs.entry(*op_id).or_insert(Vec::new()).push(*input);
                op_outputs.entry(*input).or_insert(Vec::new()).push(*op_id);
            }
        }

        // Get heights
        let mut max_height = 0;
        let mut node_heights = HashMap::new();
        for (op_id, _op) in &input_init_data {
            let height = calculate_height(*op_id, &input_init_data, &mut node_heights);
            max_height = max_height.max(height);
        }

        // Sort by height
        let mut nodes_and_heights = node_heights.clone().into_iter().collect::<Vec<_>>();
        nodes_and_heights.sort_by(|(_, a), (_, b)| {b.cmp(a)});

        // Iterate over backwards and push nodes up to right under their children
        for (op_id, _) in nodes_and_heights.iter() {
            let mut upper_bound = None;
            if let Some(outputs) = op_outputs.get(op_id) {
                for output in outputs {
                    let height = &node_heights[output];
                    if let Some(x) = upper_bound.clone() {
                        if *height < x {
                            upper_bound = Some(*height);
                        }
                    }
                    else {
                        upper_bound = Some(*height);
                    }
                }
            }

            if let Some(upper_bound) = upper_bound {
                node_heights.insert(*op_id, upper_bound-1);
            }
        }

        // Break long edges with connection by name
        let mut connection_by_name_ids: HashMap<GraphLayoutNodeId, String> = HashMap::new();
        let mut init_data = input_init_data.clone();
        let mut connection_by_name_next_id = 0;
        for (op_id, _op) in &input_init_data {
            let mut new_inputs = vec![];
            if let Some(inputs) = op_inputs.get(op_id) {
                for input_id in inputs.clone() {
                    let height_delta = node_heights[&op_id] - node_heights[&input_id];
                    if height_delta > 10 {  
                        // Break into connection by name

                        // Add source node if necessary
                        let source_node_id: String = if let Some(node) = connection_by_name_ids.get(&input_id) {
                            node.to_string()
                        } else {
                            let connection_by_name_id = connection_by_name_next_id.to_string();
                            let new_node = GraphLayoutNodeInitData{
                                node_type: GraphLayoutNodeType::ConnectionByNameSrc(connection_by_name_id.clone()),
                                inputs: vec![input_id],
                                shape: vec2(80.0, 30.0)
                            };
                            let new_node_id = GraphLayoutNodeId(next_node_id);
                            next_node_id += 1;
                            connection_by_name_next_id += 1;
                            connection_by_name_ids.insert(input_id, connection_by_name_id.clone());

                            init_data.insert(new_node_id, new_node);
                            node_heights.insert(new_node_id, node_heights[&input_id]+1);

                            op_inputs.insert(new_node_id, vec![input_id]);
                            op_outputs.entry(input_id).and_modify(|v| {v.push(new_node_id)});

                            connection_by_name_id
                        };
                        // Add new dest node
                        let new_node = GraphLayoutNodeInitData{
                            node_type: GraphLayoutNodeType::ConnectionByNameDest(source_node_id.clone()),
                            inputs: vec![],
                            shape: vec2(80.0, 30.0)
                        };
                        let new_node_id = GraphLayoutNodeId(next_node_id);
                        next_node_id += 1;
                        init_data.insert(new_node_id, new_node);
                        node_heights.insert(new_node_id, node_heights[op_id]-1);

                        op_outputs.insert(new_node_id, vec![*op_id]);
                        new_inputs.push(new_node_id);
                        // Delete us from the output list of the source node
                        op_outputs.entry(input_id).and_modify(|x| {
                            let idx = x.iter().position(|&r| r == *op_id).unwrap();
                            x.remove(idx);
                        });
                    } else {
                        new_inputs.push(input_id)
                    }
                }
            }
            op_inputs.insert(*op_id, new_inputs);
        }
        
        // List all edges
        let mut edges = vec![];
        for (op_id, inputs) in op_inputs.iter() {
            for input in inputs {
                edges.push((*input, *op_id));
            }
        }

        // Sort by height again
        let mut nodes_and_heights = node_heights.clone().into_iter().collect::<Vec<_>>();
        nodes_and_heights.sort_by(|(_, a), (_, b)| {b.cmp(a)});

        // Assign lateral positions to minimize crossovers
        let mut node_lat_vals = HashMap::new();
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
                // Get avg lateral position of parents
                let mut lat_means = vec![];
                for op_id in &height_siblings {
                    if let Some(outputs) = op_outputs.get(op_id) {
                        let outputs_len = outputs.len();
                        let mut total = 0.0;
                        for output in outputs {
                            total += node_lat_vals[output]
                        }
                        lat_means.push((op_id, total / outputs_len as f32))
                    }
                    else {
                        // No outputs
                        lat_means.push((op_id, 0.0))
                    }
                }
                // Sort and place
                lat_means.sort_by(|(_, a), (_, b)| {if a > b {Ordering::Greater} else {Ordering::Less}});
                let num_vals = lat_means.len();
                for (i, (a, b)) in lat_means.into_iter().enumerate() {
                    let x = (i as f32) - (num_vals as f32 / 2.0);
                    node_lat_vals.insert(*a, (x + b)/2.0);
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

        let mut nodes = HashMap::new();
        let mut node_map = HashMap::new();
        for (op_id, data) in init_data {
            let pos = egui::pos2(node_heights[&op_id] as f32 * max_node_shape.x*1.5, node_lat_vals[&op_id] * max_node_shape.y*1.5);
            let vel = egui::vec2(random::<f32>(), random::<f32>());

            node_map.entry(Self::get_index_for(&pos)).or_insert(Vec::new()).push(op_id);

            let new_node = GraphLayoutNode {
                node_type: data.node_type,
                position: pos,
                velocity: vel,
                shape: data.shape,
                inputs: data.inputs,
            };
            nodes.insert(op_id, new_node);
        }

        Self {
            nodes,
            node_map,
            edges,
            op_inputs,
            op_outputs,
            layout_clock: 0.0,
            max_cell_shape: max_node_shape
        }
    }

    pub(crate) fn get_nodes(&self) -> &HashMap<GraphLayoutNodeId, GraphLayoutNode> {
        &self.nodes
    }

    
    pub(crate) fn get_edges(&self) -> &Vec<(GraphLayoutNodeId, GraphLayoutNodeId)> {
        &self.edges
    }
    
    pub(crate) fn get_nodes_mut(&mut self) -> &mut HashMap<GraphLayoutNodeId, GraphLayoutNode> {
        &mut self.nodes
    }

    pub(crate) fn update_layout(&mut self, max_nodes_to_update: u32) -> bool {
        let mut did_update = false;
        for _ in 0..max_nodes_to_update {
            let i = random_range(0..self.nodes.len());
            let op_id = *self.nodes.keys().nth(i).unwrap();
            let node_data = &self.nodes[&op_id];
            let mut applied_force = (0.0, 0.0);
            for other_node in self.find_nodes_within(&node_data.position, self.max_cell_shape.length()*1.5) {
                if other_node == op_id {
                    continue;
                }
                let other_node_data = &self.nodes[&other_node];
                let delta = node_data.position - other_node_data.position;
                let distance = delta.length();
                let normalized_delta = (delta.x / distance, delta.y / distance);

                if distance > 0.01 {
                    let min_x_dist = (node_data.shape.x + other_node_data.shape.x)*0.8;
                    let min_y_dist = (node_data.shape.y + other_node_data.shape.y)*0.8;
                    if delta.x.abs() < min_x_dist && delta.y.abs() < min_y_dist {
                        let force = (node_data.shape.length() + other_node_data.shape.length())*1.4 / distance;
                        applied_force.0 += force * normalized_delta.0;
                        applied_force.1 += force * normalized_delta.1;
                    }
                }
            }
            let mut links = vec![];
            if let Some(inputs) = self.op_inputs.get(&op_id) {
                for input in inputs {
                    links.push((*input, op_id));
                }
            }
            if let Some(outputs) = self.op_outputs.get(&op_id) {
                for output in outputs {
                    links.push((op_id, *output));
                }
            }

            for (src, dst) in links {
                // Applied to dst, inverse applied to src
                let mut link_force = (0.0, 0.0);
                let src_data = &self.nodes[&src];
                let dst_data = &self.nodes[&dst];

                let diag_dist = (src_data.shape.length() + dst_data.shape.length())/2.0;
                let horiz_dist = (src_data.shape.x + dst_data.shape.x)/2.0;

                let delta = dst_data.position - src_data.position;
                let distance = delta.length();
                let normalized_delta = (delta.x / distance, delta.y / distance);
                let force = if distance > horiz_dist * 1.2 {
                    (-distance / (100.0 + diag_dist)).min(0.5)
                } else {
                    0.0
                };

                link_force.0 = force * normalized_delta.0;
                link_force.1 = force * normalized_delta.1;
                let hierarchy_error = src_data.position.x + (src_data.shape.x + dst_data.shape.x)*0.8 - dst_data.position.x;
                if hierarchy_error > 0.0 {
                    // Inputs must be pushed above the node
                    link_force.0 += 5.0 + 0.01* hierarchy_error;
                }

                if dst == op_id {
                    applied_force.0 += link_force.0;
                    applied_force.1 += link_force.1;
                }
                if src == op_id {
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
            let velocity_magnitude  = velocity.length();
            let clipped_velocity = velocity_magnitude.min(15.0);
            let velocity = egui::vec2(clipped_velocity*velocity.x/velocity_magnitude, clipped_velocity*velocity.y/velocity_magnitude);

            let velocity_magnitude  = velocity.length();
            let min_movement =  self.layout_clock / 6000.0;
            if velocity_magnitude.is_finite() && velocity_magnitude > min_movement {
                did_update = true;
                let old_index = Self::get_index_for(&node_data.position);
                let new_position = node_data.position + velocity;
                let new_index = Self::get_index_for(&new_position);
                if old_index != new_index {
                    // Update position on map
                    if let Some(x) = self.node_map.get_mut(&old_index) {
                        // Remove
                        x.retain_mut(|x| {*x != op_id});
                    } else {
                        // Should not be possible
                        panic!();
                    }
                    // Add to map
                    self.node_map.entry(new_index).or_insert(Vec::new()).push(op_id);
                }

                self.nodes.get_mut(&op_id).unwrap().position = new_position;
                self.nodes.get_mut(&op_id).unwrap().velocity = velocity;
            } else {
                self.nodes.get_mut(&op_id).unwrap().velocity = vec2(0.0, 0.0);
            }
        }
        self.layout_clock += max_nodes_to_update as f32 / self.nodes.len() as f32;
        did_update
    }
}