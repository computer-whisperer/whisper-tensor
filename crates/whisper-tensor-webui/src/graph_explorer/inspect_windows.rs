use crate::app::LoadedModels;
use crate::graph_explorer::{
    GraphExplorerApp, GraphExplorerSettings,
    format_shape,
};
use crate::websockets::ServerRequestManager;
use crate::widgets::tensor_view::{TensorViewState, tensor_view};
use egui::{Response, RichText};
use std::collections::HashSet;
use whisper_tensor::DynRank;
use whisper_tensor::backends::ndarray_backend::NDArrayNumericTensor;
use whisper_tensor::graph::{GlobalId, Node};
use whisper_tensor::super_graph::nodes::SuperGraphAnyNode;
use whisper_tensor::symbolic_graph::ops::{AnyOperation, Operation};
use whisper_tensor::symbolic_graph::tensor_store::TensorStoreTensorId;
use whisper_tensor::symbolic_graph::{
    StoredOrNotTensor,
    TensorType,
};
use whisper_tensor_server::WebsocketClientServerMessage;

#[derive(Clone, Debug)]
pub(crate) struct InspectWindowGraphLink {
    pub(crate) path: Vec<GlobalId>,
    pub(crate) stored_value_requested: Option<TensorStoreTensorId>,
    pub(crate) stored_value: Option<Result<NDArrayNumericTensor<DynRank>, String>>,
    pub(crate) value_view_state: TensorViewState,
    pub(crate) subscribed_view_state: TensorViewState,
}

impl InspectWindowGraphLink {
    pub(crate) fn new(path: Vec<GlobalId>) -> Self {
        Self {
            path,
            stored_value_requested: None,
            stored_value: None,
            value_view_state: TensorViewState::default(),
            subscribed_view_state: TensorViewState::default(),
        }
    }

    pub(crate) fn to_any(self) -> AnyInspectWindow {
        AnyInspectWindow::GraphLink(self)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct InspectWindowGraphNode {
    pub(crate) path: Vec<GlobalId>,
    pub(crate) value_view_state: TensorViewState,
}

impl InspectWindowGraphNode {
    pub(crate) fn new(path: Vec<GlobalId>) -> Self {
        Self {
            path,
            value_view_state: TensorViewState::default(),
        }
    }

    pub(crate) fn to_any(self) -> AnyInspectWindow {
        AnyInspectWindow::GraphNode(self)
    }
}

#[derive(Clone, Debug)]
pub(crate) enum AnyInspectWindow {
    GraphLink(InspectWindowGraphLink),
    GraphNode(InspectWindowGraphNode),
}

impl AnyInspectWindow {

    pub(crate) fn check_if_already_exists(
        windows: &[Self],
        subject: &[GlobalId]
    ) -> bool {
        for window in windows {
            match window {
                AnyInspectWindow::GraphNode(node) => {
                    if node.path == *subject {
                        return true;
                    }
                }
                AnyInspectWindow::GraphLink(node) => {
                    if node.path == *subject {
                        return true;
                    }
                }

                _ => {}
            }
        }
        false
    }
}

impl GraphExplorerApp {
    pub(crate) fn update_inspect_windows(
        &mut self,
        _state: &mut GraphExplorerSettings,
        ctx: &egui::Context,
        loaded_models: &mut LoadedModels,
        server_request_manager: &mut ServerRequestManager,
    ) {
        // TODO: Implement
    }
}
