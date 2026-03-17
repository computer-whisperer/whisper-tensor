use super::SlotPipEndpoint;
use whisper_tensor::graph::GlobalId;

#[derive(Clone, Debug, Default)]
pub(crate) struct LinkDragController {
    state: LinkDragState,
}

#[derive(Clone, Debug, Default)]
enum LinkDragState {
    #[default]
    Idle,
    Dragging {
        graph_path: Vec<GlobalId>,
        source: SlotPipEndpoint,
        hovered_target: Option<SlotPipEndpoint>,
    },
}

impl LinkDragController {
    pub(crate) fn clear(&mut self) {
        self.state = LinkDragState::Idle;
    }

    pub(crate) fn clear_if_graph_mismatch(&mut self, graph_path: &[GlobalId]) {
        if let LinkDragState::Dragging {
            graph_path: drag_path,
            ..
        } = &self.state
            && drag_path != graph_path
        {
            self.clear();
        }
    }

    pub(crate) fn clear_hover_target_for_graph(&mut self, graph_path: &[GlobalId]) {
        if let LinkDragState::Dragging {
            graph_path: drag_path,
            hovered_target,
            ..
        } = &mut self.state
            && drag_path == graph_path
        {
            *hovered_target = None;
        }
    }

    pub(crate) fn begin_drag(&mut self, graph_path: Vec<GlobalId>, source: SlotPipEndpoint) {
        self.state = LinkDragState::Dragging {
            graph_path,
            source,
            hovered_target: None,
        };
    }

    pub(crate) fn set_hover_target(&mut self, graph_path: &[GlobalId], target: SlotPipEndpoint) {
        if let LinkDragState::Dragging {
            graph_path: drag_path,
            source,
            hovered_target,
        } = &mut self.state
            && drag_path == graph_path
            && source.direction != target.direction
        {
            *hovered_target = Some(target);
        }
    }

    pub(crate) fn is_drag_source(
        &self,
        graph_path: &[GlobalId],
        endpoint: SlotPipEndpoint,
    ) -> bool {
        matches!(
            &self.state,
            LinkDragState::Dragging {
                graph_path: drag_path,
                source,
                ..
            } if drag_path == graph_path
                && source.owner == endpoint.owner
                && source.direction == endpoint.direction
                && source.slot_index == endpoint.slot_index
        )
    }

    pub(crate) fn source_for_graph(&self, graph_path: &[GlobalId]) -> Option<SlotPipEndpoint> {
        match &self.state {
            LinkDragState::Dragging {
                graph_path: drag_path,
                source,
                ..
            } if drag_path == graph_path => Some(*source),
            _ => None,
        }
    }

    pub(crate) fn finish_if_released(
        &mut self,
        graph_path: &[GlobalId],
        primary_released: bool,
    ) -> Option<(SlotPipEndpoint, SlotPipEndpoint)> {
        if !primary_released {
            return None;
        }

        let state = core::mem::take(&mut self.state);
        match state {
            LinkDragState::Dragging {
                graph_path: drag_path,
                source,
                hovered_target,
            } if drag_path == graph_path => hovered_target.map(|target| (source, target)),
            other_state => {
                self.state = other_state;
                None
            }
        }
    }
}
