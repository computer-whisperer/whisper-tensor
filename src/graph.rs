
//! Backbone traits to enforce a common graph paradigm across implementations.
//!
//! Goals:
//! - Provide a prescriptive shape (graph, inner graph, node, link, path, observer).
//! - Future graph layers (import, symbolic, milli-op, super) implement these to align.
//! - Enable shared tooling and passes across all graphs.
use std::fmt::Debug;
use std::hash::Hash;

/// Path addressing the graph itself (root and nested subgraphs).
pub trait GraphPath: Clone + Debug + Eq + Hash {}

/// Path addressing a node instance within a graph (may be nested via GraphPath semantics).
pub trait NodePath: Clone + Debug + Eq + Hash {}

/// Path addressing a link (edge) within a graph (may be nested via GraphPath semantics).
pub trait LinkPath: Clone + Debug + Eq + Hash {}

/// A directed connection between a producer node output and a consumer node input.
pub trait Link<LinkIdT: Clone + Eq + Hash + Debug> {
    /// The tensor that flows along this link (stable id at graph scope).
    fn link_id(&self) -> LinkIdT;
}

/// Node within a graph. Carries op kind and its interface to links.
pub trait Node<LinkIdT: Clone + Eq + Hash + Debug> {

    /// Incoming link handles in input index order.
    fn inputs(&self) -> impl Iterator<Item = LinkIdT>;
    /// Outgoing link handles grouped by output index order.
    fn outputs(&self) -> impl Iterator<Item = LinkIdT>;
}

/// The inner structure of a graph: nodes and links, plus IO interface.
pub trait InnerGraph {
    type NodeId: Clone + Eq + Hash + Debug;
    type LinkId: Clone + Eq + Hash + Debug;
    type Error: Debug;
    type AnyNode: Node<Self::LinkId>;
    type AnyLink: Link<Self::LinkId>;
    type InputLinkId: Clone + Eq + Hash + Debug;
    type OutputLinkId: Clone + Eq + Hash + Debug;

    /// Deterministic iteration over nodes and links.
    fn nodes(&self) -> impl Iterator<Item = Self::NodeId>;
    fn links(&self) -> impl Iterator<Item = Self::LinkId>;

    /// Resolve handles.
    fn get_node(&self, id: &Self::NodeId) -> Option<&Self::AnyNode>;
    fn get_link(&self, id: &Self::LinkId) -> Option<&Self::AnyLink>;

    /// External interface tensors.
    fn input_links(&self) -> impl Iterator<Item = (Self::InputLinkId, Self::LinkId)>;
    fn output_links(&self) -> impl Iterator<Item = (Self::OutputLinkId, Self::LinkId)>;

    /// Optional topological order of node handles.
    fn topological_order(&self) -> Option<Box<dyn Iterator<Item = Self::NodeId>>> {
        None
    }
}

/// Root graph abstraction that can host an InnerGraph and provide naming/paths.
pub trait Graph {
    type GraphPath: GraphPath;
    type NodePath: NodePath;
    type LinkPath: LinkPath;
    type Inner: InnerGraph;
    type AnySubGraph;

    /// Access inner graph by path (root or nested).
    fn inner(&self, path: &Self::GraphPath) -> &Self::AnySubGraph;
}

/// Observer API for instrumentation across graph execution and transformations.
pub trait Observer<G: Graph> {
    fn on_node_scheduled(
        &mut self,
        _path: &G::GraphPath,
        _node: &<G::Inner as InnerGraph>::AnyNode,
    ) {
    }
    fn on_node_executed(
        &mut self,
        _path: &G::GraphPath,
        _node: &<G::Inner as InnerGraph>::AnyNode,
    ) {
    }
    fn on_tensor_assigned(
        &mut self,
        _path: &G::GraphPath,
        _tensor: &<G::Inner as InnerGraph>::AnyLink,
    ) {
    }
}