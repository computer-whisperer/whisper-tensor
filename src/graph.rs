//! Backbone traits to enforce a common graph paradigm across implementations.
//!
//! Goals:
//! - Provide a prescriptive shape (graph, inner graph, node, link, path, observer).
//! - Future graph layers (import, symbolic, milli-op, super) implement these to align.
//! - Enable shared tooling and passes across all graphs.

use crate::dtype::DType;
use crate::scalar_info::ScalarInfoTyped;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::borrow::Cow;
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
    /// Incoming link slots in index order. `None` indicates a disconnected slot.
    fn input_slots(&self) -> Box<dyn Iterator<Item = Option<GlobalId>> + '_> {
        Box::new(self.inputs().map(Some))
    }
    /// Outgoing link slots in index order. `None` indicates an unbound output slot.
    fn output_slots(&self) -> Box<dyn Iterator<Item = Option<GlobalId>> + '_> {
        Box::new(self.outputs().map(Some))
    }
    /// Optional display labels aligned with input slot indices.
    fn input_slot_labels(&self) -> Box<dyn Iterator<Item = Option<String>> + '_> {
        Box::new(self.input_slots().map(|_| None))
    }
    /// Optional display labels aligned with output slot indices.
    fn output_slot_labels(&self) -> Box<dyn Iterator<Item = Option<String>> + '_> {
        Box::new(self.output_slots().map(|_| None))
    }
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
    /// Incoming link slots in index order. `None` indicates a disconnected slot.
    fn input_slots(&self) -> Box<dyn Iterator<Item = Option<GlobalId>> + '_>;
    /// Outgoing link slots in index order. `None` indicates an unbound output slot.
    fn output_slots(&self) -> Box<dyn Iterator<Item = Option<GlobalId>> + '_>;
    /// Optional display labels aligned with input slot indices.
    fn input_slot_labels(&self) -> Box<dyn Iterator<Item = Option<String>> + '_>;
    /// Optional display labels aligned with output slot indices.
    fn output_slot_labels(&self) -> Box<dyn Iterator<Item = Option<String>> + '_>;
    /// Optional label for debugging.
    fn label(&self) -> Option<String>;
}

impl<N: Node> NodeDyn for N {
    fn global_id(&self) -> GlobalId {
        self.global_id()
    }

    fn op_kind(&self) -> String {
        self.op_kind().as_ref().to_string()
    }

    fn label(&self) -> Option<String> {
        self.label()
    }

    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId> + '_> {
        self.inputs()
    }

    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId> + '_> {
        self.outputs()
    }

    fn input_slots(&self) -> Box<dyn Iterator<Item = Option<GlobalId>> + '_> {
        self.input_slots()
    }

    fn output_slots(&self) -> Box<dyn Iterator<Item = Option<GlobalId>> + '_> {
        self.output_slots()
    }

    fn input_slot_labels(&self) -> Box<dyn Iterator<Item = Option<String>> + '_> {
        self.input_slot_labels()
    }

    fn output_slot_labels(&self) -> Box<dyn Iterator<Item = Option<String>> + '_> {
        self.output_slot_labels()
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlotDirection {
    Input,
    Output,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DisconnectedNodeSlot {
    pub node_id: GlobalId,
    pub op_kind: String,
    pub direction: SlotDirection,
    pub slot_index: usize,
    pub slot_label: Option<String>,
}

impl DisconnectedNodeSlot {
    pub fn describe(&self) -> String {
        let direction = match self.direction {
            SlotDirection::Input => "input",
            SlotDirection::Output => "output",
        };
        if let Some(label) = &self.slot_label {
            format!(
                "node {} ({}) has disconnected {} slot {} ('{}')",
                self.node_id, self.op_kind, direction, self.slot_index, label
            )
        } else {
            format!(
                "node {} ({}) has disconnected {} slot {}",
                self.node_id, self.op_kind, direction, self.slot_index
            )
        }
    }
}

pub fn collect_disconnected_node_slots(graph: &dyn GraphDyn) -> Vec<DisconnectedNodeSlot> {
    let mut issues = Vec::new();

    for node_id in graph.node_ids() {
        let Some(node) = graph.get_node_by_id(&node_id) else {
            continue;
        };
        let op_kind = node.op_kind();

        let input_slots = node.input_slots().collect::<Vec<_>>();
        let input_labels = node.input_slot_labels().collect::<Vec<_>>();
        let input_len = input_slots.len().max(input_labels.len());
        for slot_index in 0..input_len {
            let slot = input_slots.get(slot_index).copied().unwrap_or(None);
            if slot.is_none() {
                let slot_label = input_labels.get(slot_index).cloned().flatten();
                issues.push(DisconnectedNodeSlot {
                    node_id,
                    op_kind: op_kind.clone(),
                    direction: SlotDirection::Input,
                    slot_index,
                    slot_label,
                });
            }
        }

        let output_slots = node.output_slots().collect::<Vec<_>>();
        let output_labels = node.output_slot_labels().collect::<Vec<_>>();
        let output_len = output_slots.len().max(output_labels.len());
        for slot_index in 0..output_len {
            let slot = output_slots.get(slot_index).copied().unwrap_or(None);
            if slot.is_none() {
                let slot_label = output_labels.get(slot_index).cloned().flatten();
                issues.push(DisconnectedNodeSlot {
                    node_id,
                    op_kind: op_kind.clone(),
                    direction: SlotDirection::Output,
                    slot_index,
                    slot_label,
                });
            }
        }
    }

    issues
}

/// Observer API for instrumentation across graph execution and transformations.
pub trait Observer<G: Graph> {
    fn on_node_scheduled(&mut self, _path: &[GlobalId], _node: &G::AnyNode) {}
    fn on_node_executed(&mut self, _path: &[GlobalId], _node: &G::AnyNode) {}
    fn on_tensor_assigned(&mut self, _path: &[GlobalId], _tensor: &G::AnyLink) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[derive(Clone, Debug)]
    struct DummyLink {
        id: GlobalId,
    }

    impl Link for DummyLink {
        fn global_id(&self) -> GlobalId {
            self.id
        }
    }

    impl LinkMetadata for DummyLink {}

    #[derive(Clone, Debug)]
    struct DummyNode {
        id: GlobalId,
        op_kind: &'static str,
        input_slots: Vec<Option<GlobalId>>,
        output_slots: Vec<Option<GlobalId>>,
        input_labels: Vec<Option<String>>,
        output_labels: Vec<Option<String>>,
    }

    impl Node for DummyNode {
        type OpKind = &'static str;

        fn global_id(&self) -> GlobalId {
            self.id
        }

        fn op_kind(&self) -> Self::OpKind {
            self.op_kind
        }

        fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId> + '_> {
            Box::new(self.input_slots.iter().flatten().copied())
        }

        fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId> + '_> {
            Box::new(self.output_slots.iter().flatten().copied())
        }

        fn input_slots(&self) -> Box<dyn Iterator<Item = Option<GlobalId>> + '_> {
            Box::new(self.input_slots.iter().copied())
        }

        fn output_slots(&self) -> Box<dyn Iterator<Item = Option<GlobalId>> + '_> {
            Box::new(self.output_slots.iter().copied())
        }

        fn input_slot_labels(&self) -> Box<dyn Iterator<Item = Option<String>> + '_> {
            Box::new(self.input_labels.iter().cloned())
        }

        fn output_slot_labels(&self) -> Box<dyn Iterator<Item = Option<String>> + '_> {
            Box::new(self.output_labels.iter().cloned())
        }
    }

    impl NodeMetadata for DummyNode {}

    struct DummyGraph {
        id: GlobalId,
        nodes: HashMap<GlobalId, DummyNode>,
        links: HashMap<GlobalId, DummyLink>,
    }

    impl Default for DummyGraph {
        fn default() -> Self {
            Self {
                id: GlobalId(0),
                nodes: HashMap::new(),
                links: HashMap::new(),
            }
        }
    }

    impl Graph for DummyGraph {
        type Error = ();
        type AnyNode = DummyNode;
        type AnyLink = DummyLink;

        fn global_id(&self) -> GlobalId {
            self.id
        }

        fn node_ids(&self) -> impl Iterator<Item = GlobalId> {
            self.nodes.keys().copied()
        }

        fn inner_link_ids(&self) -> impl Iterator<Item = GlobalId> {
            self.links.keys().copied()
        }

        fn get_node_by_id(&self, id: &GlobalId) -> Option<&Self::AnyNode> {
            self.nodes.get(id)
        }

        fn get_link_by_id(&self, id: &GlobalId) -> Option<&Self::AnyLink> {
            self.links.get(id)
        }

        fn input_link_ids(&self) -> impl Iterator<Item = (GlobalId, GlobalId)> {
            core::iter::empty()
        }

        fn output_link_ids(&self) -> impl Iterator<Item = (GlobalId, GlobalId)> {
            core::iter::empty()
        }

        fn constant_link_ids(&self) -> impl Iterator<Item = GlobalId> {
            core::iter::empty()
        }
    }

    #[test]
    fn collect_disconnected_node_slots_reports_disconnected_with_labels() {
        let mut graph = DummyGraph {
            id: GlobalId(1),
            ..Default::default()
        };
        let node_id = GlobalId(2);
        graph.nodes.insert(
            node_id,
            DummyNode {
                id: node_id,
                op_kind: "dummy",
                input_slots: vec![Some(GlobalId(10)), None],
                output_slots: vec![None, Some(GlobalId(11))],
                input_labels: vec![Some("lhs".to_string()), Some("rhs".to_string())],
                output_labels: vec![Some("primary".to_string()), Some("secondary".to_string())],
            },
        );

        let issues = collect_disconnected_node_slots(&graph);
        assert_eq!(issues.len(), 2);
        assert_eq!(issues[0].direction, SlotDirection::Input);
        assert_eq!(issues[0].slot_index, 1);
        assert_eq!(issues[0].slot_label.as_deref(), Some("rhs"));
        assert_eq!(issues[1].direction, SlotDirection::Output);
        assert_eq!(issues[1].slot_index, 0);
        assert_eq!(issues[1].slot_label.as_deref(), Some("primary"));
    }
}
