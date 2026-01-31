//! Backbone traits to enforce a common graph paradigm across implementations.
//!
//! Goals:
//! - Provide a prescriptive shape (graph, inner graph, node, link, path, observer).
//! - Future graph layers (import, symbolic, milli-op, super) implement these to align.
//! - Enable shared tooling and passes across all graphs.

use std::any::Any;
use std::borrow::Cow;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::fmt::{Debug, Display};
use std::hash::Hash;
use crate::dtype::DType;
use crate::scalar_info::ScalarInfoTyped;

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

// ============================================================================
// Metadata Types
// ============================================================================

/// Dynamic property value for introspection of node parameters.
#[derive(Clone, Debug, PartialEq)]
pub enum PropertyValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
    IntList(Vec<i64>),
    FloatList(Vec<f64>),
    DType(DType),
    GlobalId(GlobalId),
    GlobalIdList(Vec<GlobalId>),
    None,
}

impl Display for PropertyValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PropertyValue::Int(v) => write!(f, "{}", v),
            PropertyValue::Float(v) => write!(f, "{:.6}", v),
            PropertyValue::Bool(v) => write!(f, "{}", v),
            PropertyValue::String(v) => write!(f, "{}", v),
            PropertyValue::IntList(v) => write!(f, "{:?}", v),
            PropertyValue::FloatList(v) => write!(f, "{:?}", v),
            PropertyValue::DType(v) => write!(f, "{:?}", v),
            PropertyValue::GlobalId(v) => write!(f, "{}", v),
            PropertyValue::GlobalIdList(v) => write!(f, "{:?}", v),
            PropertyValue::None => write!(f, "None"),
        }
    }
}

/// A named property with its value.
#[derive(Clone, Debug)]
pub struct Property {
    pub name: Cow<'static, str>,
    pub value: PropertyValue,
}

impl Property {
    pub fn new(name: impl Into<Cow<'static, str>>, value: PropertyValue) -> Self {
        Self {
            name: name.into(),
            value,
        }
    }
}

/// Category of a link within a graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinkCategory {
    Input,
    Output,
    Intermediate,
    Constant,
}

impl Display for LinkCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LinkCategory::Input => write!(f, "Input"),
            LinkCategory::Output => write!(f, "Output"),
            LinkCategory::Intermediate => write!(f, "Intermediate"),
            LinkCategory::Constant => write!(f, "Constant"),
        }
    }
}

/// Metadata about a link (tensor/value) in a graph.
pub trait LinkMetadata: Link {
    /// The data type of this link, if known.
    fn dtype(&self) -> Option<DType> {
        None
    }

    /// The shape of this link, if known. Each dimension may be concrete or symbolic.
    fn shape(&self) -> Option<Vec<ScalarInfoTyped<u64>>> {
        None
    }

    /// The category of this link (input, output, intermediate, constant).
    fn category(&self) -> Option<LinkCategory> {
        None
    }

    /// Additional properties specific to this link type.
    fn properties(&self) -> Vec<Property> {
        Vec::new()
    }
}

/// Metadata about a node (operation) in a graph.
pub trait NodeMetadata: Node {
    /// Operation parameters as key-value pairs for introspection.
    fn parameters(&self) -> Vec<Property> {
        Vec::new()
    }

    /// Whether this node contains a subgraph (e.g., If, Scan operations).
    fn has_subgraph(&self) -> bool {
        false
    }
}

/// Object-safe version of LinkMetadata for dynamic dispatch.
pub trait LinkMetadataDyn: Link {
    fn dtype(&self) -> Option<DType>;
    fn shape(&self) -> Option<Vec<ScalarInfoTyped<u64>>>;
    fn category(&self) -> Option<LinkCategory>;
    fn properties(&self) -> Vec<Property>;
}

impl<L: LinkMetadata> LinkMetadataDyn for L {
    fn dtype(&self) -> Option<DType> {
        LinkMetadata::dtype(self)
    }

    fn shape(&self) -> Option<Vec<ScalarInfoTyped<u64>>> {
        LinkMetadata::shape(self)
    }

    fn category(&self) -> Option<LinkCategory> {
        LinkMetadata::category(self)
    }

    fn properties(&self) -> Vec<Property> {
        LinkMetadata::properties(self)
    }
}

/// Object-safe version of NodeMetadata for dynamic dispatch.
pub trait NodeMetadataDyn: NodeDyn {
    fn parameters(&self) -> Vec<Property>;
    fn has_subgraph(&self) -> bool;
}

impl<N: NodeMetadata> NodeMetadataDyn for N {
    fn parameters(&self) -> Vec<Property> {
        NodeMetadata::parameters(self)
    }

    fn has_subgraph(&self) -> bool {
        NodeMetadata::has_subgraph(self)
    }
}

// ============================================================================
// Core Graph Traits
// ============================================================================

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
    type AnyNode: Node + NodeMetadata;
    type AnyLink: Link + LinkMetadata;

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

    // Metadata accessors
    fn get_node_metadata_by_id(&self, id: &GlobalId) -> Option<&dyn NodeMetadataDyn>;
    fn get_link_metadata_by_id(&self, id: &GlobalId) -> Option<&dyn LinkMetadataDyn>;
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
    fn get_node_metadata_by_id(&self, id: &GlobalId) -> Option<&dyn NodeMetadataDyn> {
        self.get_node_by_id(id).map(|x| x as &dyn NodeMetadataDyn)
    }
    fn get_link_metadata_by_id(&self, id: &GlobalId) -> Option<&dyn LinkMetadataDyn> {
        self.get_link_by_id(id).map(|x| x as &dyn LinkMetadataDyn)
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
