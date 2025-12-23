//! Backbone traits to enforce a common graph paradigm across implementations.
//!
//! Goals:
//! - Provide a prescriptive shape (graph, inner graph, node, link, path, observer).
//! - Future graph layers (import, symbolic, milli-op, super) implement these to align.
//! - Enable shared tooling and passes across all graphs.

use std::any::Any;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::fmt::{Debug, Display};
use std::hash::Hash;

#[derive(Clone, Debug, Serialize, Deserialize, Hash, PartialEq, Eq, Copy, Ord, PartialOrd)]
pub struct GlobalId(pub(crate) u64);

impl GlobalId {
    pub fn new(rng: &mut impl Rng) -> Self {
        GlobalId(rng.next_u64())
    }
}

impl Display for GlobalId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GlobalId({})", self.0)
    }
}

/// A directed connection between a producer node output and a consumer node input.
pub trait Link {
    /// Unique identifier.
    fn global_id(&self) -> GlobalId;
    fn label(&self) -> Option<String> {
        None
    }
}

/// Node within a graph. Carries op kind and its interface to links.
pub trait Node {
    type OpKind: AsRef<str> + Clone + Debug;

    /// Unique identifier.
    fn global_id(&self) -> GlobalId;
    /// Op name or other identifier.
    fn op_kind(&self) -> Self::OpKind;
    /// Incoming link handles in input index order.
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId> + '_>;
    /// Outgoing link handles grouped by output index order.
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId> + '_>;
    /// Optional label for debugging.
    fn label(&self) -> Option<String> {
        None
    }
}

pub trait NodeDyn {
    /// Unique identifier.
    fn global_id(&self) -> GlobalId;
    /// Op name or other identifier.
    fn op_kind(&self) -> String;
    /// Incoming link handles in input index order.
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId> + '_>;
    /// Outgoing link handles grouped by output index order.
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId> + '_>;
    /// Optional label for debugging.
    fn label(&self) -> Option<String>;
}

impl<N: Node> NodeDyn for N
{
    fn global_id(&self) -> GlobalId {
        self.global_id()
    }

    fn op_kind(&self) -> String {
        self.op_kind().as_ref().to_string()
    }

    fn label(&self) -> Option<String> {
        self.label()
    }

    fn inputs(&self) -> Box<dyn Iterator<Item=GlobalId> + '_> {
        self.inputs()
    }

    fn outputs(&self)  -> Box<dyn Iterator<Item=GlobalId> + '_> {
        self.outputs()
    }
}

/// Root graph abstraction that can host an InnerGraph and provide naming/paths.
pub trait Graph {
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

    /// Input links to the graph, (outer scope, inner scope).
    fn input_link_ids(&self) -> impl Iterator<Item = (GlobalId, GlobalId)>;
    /// Output links to the graph, (outer scope, inner scope).
    fn output_link_ids(&self) -> impl Iterator<Item = (GlobalId, GlobalId)>;
    /// Constant values in the graph.
    fn constant_link_ids(&self) -> impl Iterator<Item = GlobalId>;

    /// Optional topological order of node handles.
    fn topological_order(&self) -> Option<Box<dyn Iterator<Item = GlobalId>>> {
        None
    }
}

pub trait GraphDyn {
    fn global_id(&self) -> GlobalId;

    fn node_ids(&self) -> Box<dyn Iterator<Item = GlobalId> + '_>;
    fn inner_link_ids(&self) -> Box<dyn Iterator<Item = GlobalId> + '_>;
    /// Input links to the graph, (outer scope, inner scope).
    fn input_link_ids(&self) -> Box<dyn Iterator<Item = (GlobalId, GlobalId)> + '_>;
    /// Output links to the graph, (outer scope, inner scope).
    fn output_link_ids(&self) -> Box<dyn Iterator<Item = (GlobalId, GlobalId)> + '_>;
    /// Constant values in the graph.
    fn constant_link_ids(&self) -> Box<dyn Iterator<Item = GlobalId> + '_>;
    fn get_node_by_id(&self, id: &GlobalId) -> Option<&dyn NodeDyn>;
    fn get_link_by_id(&self, id: &GlobalId) -> Option<&dyn Link>;
    fn as_any(&self) -> &dyn Any;
}

impl<G: Graph + 'static> GraphDyn for G {
    fn global_id(&self) -> GlobalId {
        self.global_id()
    }
    fn node_ids(&self) -> Box<dyn Iterator<Item = GlobalId> + '_> {
        Box::new(self.node_ids())
    }
    fn inner_link_ids(&self) -> Box<dyn Iterator<Item = GlobalId> + '_> {
        Box::new(self.inner_link_ids())
    }
    fn input_link_ids(&self) -> Box<dyn Iterator<Item = (GlobalId, GlobalId)> + '_> {
        Box::new(<G as Graph>::input_link_ids(self))
    }
    fn output_link_ids(&self) -> Box<dyn Iterator<Item = (GlobalId, GlobalId)> + '_> {
        Box::new(<G as Graph>::output_link_ids(self))
    }
    fn constant_link_ids(&self) -> Box<dyn Iterator<Item = GlobalId> + '_> {
        Box::new(<G as Graph>::constant_link_ids(self))
    }
    fn get_node_by_id(&self, id: &GlobalId) -> Option<&dyn NodeDyn> {
        self.get_node_by_id(id).map(|x| x as &dyn NodeDyn)
    }
    fn get_link_by_id(&self, id: &GlobalId) -> Option<&dyn Link> {
        self.get_link_by_id(id).map(|x| x as &dyn Link)
    }
    fn as_any<'a>(&'a self) -> &'a dyn Any {
        self as &'a dyn Any
    }
}

/// Observer API for instrumentation across graph execution and transformations.
pub trait Observer<G: Graph> {
    fn on_node_scheduled(&mut self, _path: &[GlobalId], _node: &G::AnyNode) {
    }
    fn on_node_executed(&mut self, _path: &[GlobalId], _node: &G::AnyNode) {}
    fn on_tensor_assigned(
        &mut self,
        _path: &[GlobalId],
        _tensor: &G::AnyLink,
    ) {
    }
}
