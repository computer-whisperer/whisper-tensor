use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn render_node_slot_pips(
    link_drag_controller: &mut LinkDragController,
    ui: &mut Ui,
    working_path: &[GlobalId],
    global_id: GlobalId,
    node_type: &GraphLayoutNodeType,
    node_center: Pos2,
    input_offsets: &[Vec2],
    output_offsets: &[Vec2],
    input_slots: &[Option<GraphLayoutLinkId>],
    output_slots: &[Option<GraphLayoutLinkId>],
    link_data: &HashMap<GraphLayoutLinkId, GraphLayoutLinkData>,
) -> bool {
    let slot_owner = match node_type {
        GraphLayoutNodeType::GraphNode(graph_node_id) => SlotPipOwner::Node(*graph_node_id),
        GraphLayoutNodeType::InputLinkNode(link_id) => SlotPipOwner::InputLink(*link_id),
        GraphLayoutNodeType::OutputLinkNode(link_id) => SlotPipOwner::OutputLink(*link_id),
        GraphLayoutNodeType::ConstantLinkNode(link_id) => SlotPipOwner::ConstantLink(*link_id),
        GraphLayoutNodeType::ConnectionByNameSrc(_)
        | GraphLayoutNodeType::ConnectionByNameDest(_) => {
            return false;
        }
    };

    let mut active_slot_drag = false;
    let slot_pip_radius = 4.0f32;
    let slot_pip_rect_radius = 8.0f32;
    let interaction_sense = Sense::click_and_drag();

    active_slot_drag |= render_slot_pips_for_direction(
        link_drag_controller,
        ui,
        working_path,
        global_id,
        slot_owner,
        SlotDirection::Input,
        node_center,
        input_offsets,
        input_slots,
        link_data,
        slot_pip_radius,
        slot_pip_rect_radius,
        interaction_sense,
        Color32::from_rgb(98, 190, 250),
    );

    active_slot_drag |= render_slot_pips_for_direction(
        link_drag_controller,
        ui,
        working_path,
        global_id,
        slot_owner,
        SlotDirection::Output,
        node_center,
        output_offsets,
        output_slots,
        link_data,
        slot_pip_radius,
        slot_pip_rect_radius,
        interaction_sense,
        Color32::from_rgb(95, 219, 140),
    );

    active_slot_drag
}

#[allow(clippy::too_many_arguments)]
fn render_slot_pips_for_direction(
    link_drag_controller: &mut LinkDragController,
    ui: &mut Ui,
    working_path: &[GlobalId],
    global_id: GlobalId,
    slot_owner: SlotPipOwner,
    direction: SlotDirection,
    node_center: Pos2,
    slot_offsets: &[Vec2],
    slot_links: &[Option<GraphLayoutLinkId>],
    link_data: &HashMap<GraphLayoutLinkId, GraphLayoutLinkData>,
    slot_pip_radius: f32,
    slot_pip_rect_radius: f32,
    interaction_sense: Sense,
    connected_fill_color: Color32,
) -> bool {
    let direction_salt = match direction {
        SlotDirection::Input => 0u8,
        SlotDirection::Output => 1u8,
    };

    let mut active_slot_drag = false;
    for (slot_index, offset) in slot_offsets.iter().enumerate() {
        let center = node_center + *offset;
        let maybe_layout_link = slot_links.get(slot_index).copied().flatten();
        let maybe_link = maybe_layout_link.and_then(|layout_link| {
            link_data
                .get(&layout_link)
                .map(|layout_link_data| layout_link_data.global_id)
        });
        let endpoint = SlotPipEndpoint {
            owner: slot_owner,
            direction,
            slot_index,
            link_id: maybe_link,
            layout_link_id: maybe_layout_link,
            screen_pos: center,
        };

        let pip_rect = Rect::from_center_size(center, Vec2::splat(slot_pip_rect_radius * 2.0));
        let pip_resp = ui.interact(
            pip_rect,
            ui.id()
                .with(("slot_pip", global_id, direction_salt, slot_index)),
            interaction_sense,
        );

        if pip_resp.drag_started() {
            active_slot_drag = true;
            link_drag_controller.begin_drag(working_path.to_vec(), endpoint);
        }
        if pip_resp.dragged() {
            active_slot_drag = true;
        }
        if pip_resp.hovered() {
            link_drag_controller.set_hover_target(working_path, endpoint);
        }

        let is_drag_source = link_drag_controller.is_drag_source(working_path, endpoint);
        let fill_color = if is_drag_source {
            Color32::from_rgb(255, 197, 48)
        } else if maybe_link.is_some() {
            connected_fill_color
        } else {
            Color32::from_rgb(76, 84, 97)
        };
        let stroke_color = if pip_resp.hovered() {
            Color32::from_rgb(236, 244, 255)
        } else {
            Color32::from_gray(12)
        };

        ui.painter()
            .circle_filled(center, slot_pip_radius, fill_color);
        ui.painter()
            .circle_stroke(center, slot_pip_radius, Stroke::new(1.0, stroke_color));
    }

    active_slot_drag
}

pub(super) fn render_pending_link_drag_preview(
    link_drag_controller: &mut LinkDragController,
    ui: &mut Ui,
    working_path: &[GlobalId],
) -> Option<(SlotPipEndpoint, SlotPipEndpoint)> {
    let source_endpoint = link_drag_controller.source_for_graph(working_path)?;

    if let Some(pointer_pos_global) = ui.input(|x| x.pointer.interact_pos()) {
        let pointer_pos = ui
            .ctx()
            .layer_transform_from_global(ui.layer_id())
            .map_or(pointer_pos_global, |from_global| {
                from_global * pointer_pos_global
            });
        let points = [
            source_endpoint.screen_pos,
            egui::pos2(
                source_endpoint.screen_pos.x + 40.0,
                source_endpoint.screen_pos.y,
            ),
            egui::pos2(pointer_pos.x - 40.0, pointer_pos.y),
            pointer_pos,
        ];
        let preview_stroke = Stroke {
            width: 2.0,
            color: Color32::from_rgb(255, 197, 48),
        };
        ui.painter().add(CubicBezierShape::from_points_stroke(
            points,
            false,
            Color32::TRANSPARENT,
            preview_stroke,
        ));
    }

    let pointer_released = ui.input(|x| x.pointer.primary_released());
    let maybe_result = link_drag_controller.finish_if_released(working_path, pointer_released);
    if maybe_result.is_none() && !pointer_released {
        ui.ctx().request_repaint_after(Duration::from_millis(20));
    }

    maybe_result
}
