//! Backbone traits to enforce a common graph paradigm across implementations.
//!
//! Goals:
//! - Provide a prescriptive shape (graph, inner graph, node, link, path, observer).
//! - Future graph layers (import, symbolic, milli-op, super) implement these to align.
//! - Enable shared tooling and passes across all graphs.
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::hash::Hash;

#[derive(Clone, Debug, Serialize, Deserialize, Hash, PartialEq, Eq, Copy, Ord, PartialOrd)]
pub struct GlobalId(pub(crate) u64);

impl GlobalId {
    pub fn new(rng: &mut impl Rng) -> Self {
        GlobalId(rng.next_u64())
    }
}

/// A directed connection between a producer node output and a consumer node input.
pub trait Link {
    /// Unique identifier.
    fn global_id(&self) -> GlobalId;
}

/// Node within a graph. Carries op kind and its interface to links.
pub trait Node {
    type OpKind: AsRef<str> + Clone + Debug;
    type AnyInnerGraph: InnerGraph;

    /// Unique identifier.
    fn global_id(&self) -> GlobalId;
    /// Op name or other identifier.
    fn op_kind(&self) -> Self::OpKind;
    /// Incoming link handles in input index order.
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId> + '_>;
    /// Outgoing link handles grouped by output index order.
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId> + '_>;
    fn inner_graph_ids(&self) -> Box<dyn Iterator<Item = GlobalId> + '_>;
    fn get_inner_graph_by_id(&self, id: &GlobalId) -> Option<&dyn InnerGraphDyn>;
}

pub trait NodeDyn {
    fn global_id(&self) -> GlobalId;
    fn op_kind(&self) -> String;
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId> + '_>;
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId> + '_>;
}

impl<N: Node> NodeDyn for N
{

    fn global_id(&self) -> GlobalId {
        self.global_id()
    }

    fn op_kind(&self) -> String {
        self.op_kind().as_ref().to_string()
    }

    fn inputs(&self) -> Box<dyn Iterator<Item=GlobalId> + '_> {
        self.inputs()
    }

    fn outputs(&self)  -> Box<dyn Iterator<Item=GlobalId> + '_> {
        self.outputs()
    }
}


/// The inner structure of a graph: nodes and links, plus IO interface.
pub trait InnerGraph {
    type Error: Debug;
    type AnyNode: Node;
    type AnyLink: Link;

    /// Unique identifier for the graph layer.
    fn global_id(&self) -> GlobalId;

    /// Deterministic iteration over nodes and links.
    fn node_ids(&self) -> impl Iterator<Item = GlobalId>;
    fn inner_link_ids(&self) -> impl Iterator<Item = GlobalId>;

    /// Resolve handles.
    fn get_node_by_id(&self, id: &GlobalId) -> Option<&Self::AnyNode>;
    fn get_link_by_id(&self, id: &GlobalId) -> Option<&Self::AnyLink>;

    /// External interface tensors.
    fn input_link_ids(&self) -> impl Iterator<Item = (GlobalId, GlobalId)>;
    fn output_link_ids(&self) -> impl Iterator<Item = (GlobalId, GlobalId)>;

    /// Optional topological order of node handles.
    fn topological_order(&self) -> Option<Box<dyn Iterator<Item = GlobalId>>> {
        None
    }
}


pub trait InnerGraphDyn {
    fn global_id(&self) -> GlobalId;

    fn input_link_ids(&self) -> Box<dyn Iterator<Item = (GlobalId, GlobalId)> + '_>;
    fn output_link_ids(&self) -> Box<dyn Iterator<Item = (GlobalId, GlobalId)> + '_>;

    fn get_node_by_id(&self, id: &GlobalId) -> Option<&dyn InnerGraphDyn>;
    fn get_link_by_id(&self, id: &GlobalId) -> Option<&dyn InnerGraphDyn>;
}

impl<G: InnerGraph> InnerGraphDyn for G {
    fn global_id(&self) -> GlobalId {
        self.global_id()
    }
    fn input_link_ids(&self) -> Box<dyn Iterator<Item = (GlobalId, GlobalId)> + '_> {
        Box::new(<G as InnerGraph>::input_link_ids(self))
    }
    fn output_link_ids(&self) -> Box<dyn Iterator<Item = (GlobalId, GlobalId)> + '_> {
        Box::new(<G as InnerGraph>::output_link_ids(self))
    }
    fn get_node_by_id(&self, id: &GlobalId) -> Option<&dyn InnerGraphDyn> {
        self.get_node_by_id(id).map(|x| x as &dyn InnerGraphDyn)
    }
    fn get_link_by_id(&self, id: &GlobalId) -> Option<&dyn InnerGraphDyn> {
        self.get_link_by_id(id).map(|x| x as &dyn InnerGraphDyn)
    }
}

/// Root graph abstraction that can host an InnerGraph and provide naming/paths.
pub trait Graph {
    type Inner: InnerGraph;

    /// Access inner graph by path (root or nested).
    fn inner(&self) -> &Self::Inner;
}

pub trait GraphDyn {
    fn inner(&self) -> &dyn InnerGraphDyn;
}

impl<G: Graph> GraphDyn for G where G::Inner: InnerGraphDyn{
    fn inner(&self) -> &dyn InnerGraphDyn {
        self.inner()
    }
}

/// Observer API for instrumentation across graph execution and transformations.
pub trait Observer<G: Graph> {
    fn on_node_scheduled(&mut self, _path: &[GlobalId], _node: &<G::Inner as InnerGraph>::AnyNode) {
    }
    fn on_node_executed(&mut self, _path: &[GlobalId], _node: &<G::Inner as InnerGraph>::AnyNode) {}
    fn on_tensor_assigned(
        &mut self,
        _path: &[GlobalId],
        _tensor: &<G::Inner as InnerGraph>::AnyLink,
    ) {
    }
}
