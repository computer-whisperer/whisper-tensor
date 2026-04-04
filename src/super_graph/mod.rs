pub mod cache;
pub mod data;
pub mod links;
pub mod nodes;
pub mod observer;

use crate::graph::{GlobalId, Graph, Link, collect_disconnected_node_slots};
use crate::milli_graph::MilliOpGraphError;
use crate::model::ModelError;
use crate::super_graph::cache::SuperGraphCache;
use crate::super_graph::data::SuperGraphData;
pub use crate::super_graph::links::{
    SuperGraphAnyLink, SuperGraphAtomicLinkKind, SuperGraphLink, SuperGraphLinkInfo,
    SuperGraphLinkKind,
};
use crate::super_graph::nodes::{SuperGraphAnyNode, SuperGraphNode};
use crate::super_graph::observer::SuperGraphObserver;
use crate::symbolic_graph::SymbolicGraph;
use crate::tokenizer::TokenizerError;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::Instant;

#[derive(Debug, thiserror::Error)]
pub enum SuperGraphError {
    #[error(transparent)]
    ModelError(#[from] ModelError),
    #[error(transparent)]
    TokenizerError(#[from] TokenizerError),
    #[error(transparent)]
    MilliOpGraphError(#[from] MilliOpGraphError),
    #[error("Missing link{0}")]
    MissingLinkError(String),
    #[error("Invalid input: {0}")]
    InvalidInputError(String),
    #[error("Invalid graph structure: {0}")]
    InvalidGraph(String),
    #[error(transparent)]
    SymbolicEvalError(#[from] crate::symbolic_graph::ops::EvalError),
    #[error("Execution cancelled")]
    Cancelled,
}

pub type SuperGraphHash = u64;

pub struct SuperGraphContext<'short, 'model, 'p, P: crate::pool::Pool + 'p, T: SuperGraphObserver> {
    pub pool: &'p P,
    pub observer: &'short mut T,
    pub caches: Option<&'short mut SuperGraphCache>,
    pub symbolic_graphs: Vec<&'model SymbolicGraph>,
}

impl<'short, 'model, 'p, P: crate::pool::Pool + 'p, T: SuperGraphObserver>
    SuperGraphContext<'short, 'model, 'p, P, T>
{
    /// Construct a context with only the required fields; caches and
    /// symbolic graphs default to empty/None.
    pub fn new(pool: &'p P, observer: &'short mut T) -> Self {
        Self {
            pool,
            observer,
            caches: None,
            symbolic_graphs: vec![],
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraph {
    global_id: GlobalId,
    pub input_links: HashSet<SuperGraphAnyLink>,
    pub output_links: HashSet<SuperGraphAnyLink>,
    pub nodes: HashMap<GlobalId, SuperGraphAnyNode>,
    pub links_by_global_id: HashMap<GlobalId, SuperGraphLinkInfo>,
}

impl SuperGraph {
    pub fn run<'short, 'model, 'p, P: crate::pool::Pool + 'p, T: SuperGraphObserver>(
        &'short self,
        data: SuperGraphData<'p, 'model, P>,
        context: &mut SuperGraphContext<'short, 'model, 'p, P, T>,
    ) -> Result<SuperGraphData<'p, 'model, P>, SuperGraphError> {
        self.eval(&[], data, context)
    }

    pub fn eval<'a, 'b, 'p, P: crate::pool::Pool + 'p, T: SuperGraphObserver>(
        &'a self,
        node_path: &[GlobalId],
        data: SuperGraphData<'p, 'b, P>,
        context: &mut SuperGraphContext<'a, 'b, 'p, P, T>,
    ) -> Result<SuperGraphData<'p, 'b, P>, SuperGraphError> {
        if let Some(first_issue) = self.validate_structure().into_iter().next() {
            return Err(SuperGraphError::InvalidGraph(first_issue));
        }
        if let Some(first_issue) = collect_disconnected_node_slots(self).into_iter().next() {
            return Err(SuperGraphError::InvalidGraph(first_issue.describe()));
        }

        let mut data = data;

        let mut remaining_ops = self.nodes.keys().cloned().collect::<Vec<_>>();

        loop {
            if context.observer.should_cancel() {
                return Err(SuperGraphError::Cancelled);
            }
            let op_id_to_use = {
                let mut op_id_to_use = None;
                for op_id in &remaining_ops {
                    let op = self.nodes.get(op_id).unwrap();
                    let mut all_inputs_ready = true;
                    for input in SuperGraphNode::inputs(op) {
                        if !data.contains_link(&input) {
                            all_inputs_ready = false;
                            break;
                        }
                    }
                    if all_inputs_ready {
                        op_id_to_use = Some(op_id);
                        break;
                    }
                }
                op_id_to_use
            };
            if let Some(op_id) = op_id_to_use {
                let op_id = *op_id;
                let mut this_path = node_path.to_vec();
                let op = self.nodes.get(&op_id).unwrap();
                if context.observer.should_cancel() {
                    return Err(SuperGraphError::Cancelled);
                }
                let start_instant = Instant::now();
                op.eval(&this_path, &mut data, context)?;
                this_path.push(op.global_id());
                let end_instant = Instant::now();
                context.observer.on_node_executed(
                    &this_path,
                    &op.op_kind(),
                    start_instant,
                    end_instant,
                );
                remaining_ops.retain(|x| *x != op_id);
            } else {
                break;
            }
        }

        let output_data =
            data.into_selected(&self.output_links.iter().cloned().collect::<Vec<_>>())?;

        Ok(output_data)
    }

    pub fn get_all_links(&self) -> HashSet<SuperGraphAnyLink> {
        let mut links = HashSet::new();
        links.extend(self.input_links.iter().cloned());
        links.extend(self.output_links.iter().cloned());
        for node in self.nodes.values() {
            links.extend(node.inputs());
            links.extend(node.outputs());
        }
        links
    }

    /// Returns structural issues for editor/draft scenarios where graphs may be partially wired.
    /// This does not panic and can be called on intentionally invalid graphs.
    pub fn validate_structure(&self) -> Vec<String> {
        let mut issues = Vec::new();

        let mut sourced_links = HashSet::new();
        sourced_links.extend(self.input_links.iter().copied());
        for node in self.nodes.values() {
            for link in node.outputs() {
                if !sourced_links.insert(link) {
                    issues.push(format!("link {link:?} is sourced multiple times"));
                }
            }
        }

        let mut sinked_links = HashSet::new();
        sinked_links.extend(self.output_links.iter().copied());
        for node in self.nodes.values() {
            sinked_links.extend(node.inputs());
        }
        for link in sinked_links {
            if !sourced_links.contains(&link) {
                issues.push(format!("link {link:?} is sinked but never sourced"));
            }
        }

        for link in &sourced_links {
            if !self.links_by_global_id.contains_key(&link.global_id()) {
                issues.push(format!(
                    "missing link metadata entry for link {:?} ({})",
                    link,
                    link.global_id()
                ));
            }
        }

        issues
    }
}

pub struct SuperGraphBuilder {
    nodes: HashMap<GlobalId, SuperGraphAnyNode>,
    link_labels: HashMap<GlobalId, String>,
}

impl SuperGraphBuilder {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            link_labels: HashMap::new(),
        }
    }

    pub fn add_node(&mut self, node: SuperGraphAnyNode) -> GlobalId {
        let id = node.global_id();
        self.nodes.insert(id, node);
        id
    }

    pub fn build(
        self,
        rng: &mut impl Rng,
        input_links: &[SuperGraphAnyLink],
        output_links: &[SuperGraphAnyLink],
    ) -> SuperGraph {
        let Self { nodes, link_labels } = self;
        // Validate that all input and output links are present in the graph
        let mut sourced_links = HashSet::new();
        let mut sinked_links = HashSet::new();
        sinked_links.extend(output_links.iter().cloned());
        sourced_links.extend(input_links.iter().cloned());

        for node in nodes.values() {
            for link in node.outputs() {
                if !sourced_links.insert(link) {
                    panic!("Link {link:?} is sourced multiple times");
                }
            }
            sinked_links.extend(node.inputs());
        }

        for link in sinked_links {
            if !sourced_links.contains(&link) {
                panic!("Link {link:?} is not sourced");
            }
        }

        let links_by_global_id = sourced_links
            .iter()
            .map(|link| {
                let label = link_labels.get(&link.global_id()).cloned();
                (link.global_id(), SuperGraphLinkInfo::new(*link, label))
            })
            .collect::<HashMap<_, _>>();

        SuperGraph {
            global_id: GlobalId::new(rng),
            nodes,
            input_links: HashSet::from_iter(input_links.iter().cloned()),
            output_links: HashSet::from_iter(output_links.iter().cloned()),
            links_by_global_id,
        }
    }

    pub fn set_link_label(&mut self, link: SuperGraphLink, label: impl Into<String>) {
        self.link_labels.insert(link.global_id(), label.into());
    }

    pub fn new_tensor_link(&mut self, rng: &mut impl Rng) -> SuperGraphLink {
        SuperGraphLink::new(SuperGraphLinkKind::Tensor, rng)
    }

    pub fn new_model_link(&mut self, rng: &mut impl Rng) -> SuperGraphLink {
        SuperGraphLink::new(SuperGraphLinkKind::TensorMap, rng)
    }

    pub fn new_tokenizer_link(&mut self, rng: &mut impl Rng) -> SuperGraphLink {
        SuperGraphLink::new(SuperGraphLinkKind::Tokenizer, rng)
    }

    pub fn new_string_link(&mut self, rng: &mut impl Rng) -> SuperGraphLink {
        SuperGraphLink::new(SuperGraphLinkKind::String, rng)
    }

    pub fn new_hash_link(&mut self, rng: &mut impl Rng) -> SuperGraphLink {
        SuperGraphLink::new(SuperGraphLinkKind::Hash, rng)
    }

    pub fn new_image_link(&mut self, rng: &mut impl Rng) -> SuperGraphLink {
        SuperGraphLink::new(SuperGraphLinkKind::Image, rng)
    }

    pub fn new_audio_clip_link(&mut self, rng: &mut impl Rng) -> SuperGraphLink {
        SuperGraphLink::new(SuperGraphLinkKind::AudioClip, rng)
    }

    pub fn new_video_clip_link(&mut self, rng: &mut impl Rng) -> SuperGraphLink {
        SuperGraphLink::new(SuperGraphLinkKind::VideoClip, rng)
    }

    pub fn new_multimodal_item_link(&mut self, rng: &mut impl Rng) -> SuperGraphLink {
        SuperGraphLink::new(SuperGraphLinkKind::MultimodalItem, rng)
    }

    pub fn new_list_link(
        &mut self,
        item_kind: SuperGraphAtomicLinkKind,
        rng: &mut impl Rng,
    ) -> SuperGraphLink {
        SuperGraphLink::new(SuperGraphLinkKind::list(item_kind), rng)
    }
}

impl Default for SuperGraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl Link for SuperGraphAnyLink {
    fn global_id(&self) -> GlobalId {
        self.global_id()
    }
}

impl Graph for SuperGraph {
    type Error = ();
    type AnyNode = SuperGraphAnyNode;
    type AnyLink = SuperGraphLinkInfo;

    fn global_id(&self) -> GlobalId {
        self.global_id
    }

    fn node_ids(&self) -> impl Iterator<Item = GlobalId> {
        self.nodes.keys().cloned()
    }

    fn inner_link_ids(&self) -> impl Iterator<Item = GlobalId> {
        let mut links = HashSet::new();
        for node in self.nodes.values() {
            links.extend(node.inputs().map(|x| x.global_id()));
            links.extend(node.outputs().map(|x| x.global_id()));
        }
        links.into_iter()
    }

    fn get_node_by_id(&self, id: &GlobalId) -> Option<&Self::AnyNode> {
        self.nodes.get(id)
    }

    fn get_link_by_id(&self, id: &GlobalId) -> Option<&Self::AnyLink> {
        self.links_by_global_id.get(id)
    }

    fn input_link_ids(&self) -> impl Iterator<Item = (GlobalId, GlobalId)> {
        self.input_links
            .iter()
            .map(|x| (x.global_id(), x.global_id()))
    }

    fn output_link_ids(&self) -> impl Iterator<Item = (GlobalId, GlobalId)> {
        self.output_links
            .iter()
            .map(|x| (x.global_id(), x.global_id()))
    }

    fn constant_link_ids(&self) -> impl Iterator<Item = GlobalId> {
        core::iter::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::super_graph::data::SuperGraphData;

    #[test]
    fn validate_structure_reports_unsourced_output_link() {
        let mut rng = rand::rng();
        let mut builder = SuperGraphBuilder::new();
        let io_link = builder.new_tensor_link(&mut rng).to_any();
        let mut graph = builder.build(&mut rng, &[io_link], &[io_link]);

        let dangling_output = SuperGraphLink::new(SuperGraphLinkKind::Tensor, &mut rng).to_any();
        graph.output_links.insert(dangling_output);

        let issues = graph.validate_structure();
        assert!(
            issues
                .iter()
                .any(|issue| issue.contains("sinked but never sourced")),
            "expected unsourced-link validation issue, got: {issues:?}"
        );
    }

    #[test]
    fn eval_rejects_invalid_structure_before_execution() {
        let mut rng = rand::rng();
        let mut builder = SuperGraphBuilder::new();
        let io_link = builder.new_tensor_link(&mut rng).to_any();
        let mut graph = builder.build(&mut rng, &[io_link], &[io_link]);

        let dangling_output = SuperGraphLink::new(SuperGraphLinkKind::Tensor, &mut rng).to_any();
        graph.output_links.insert(dangling_output);

        use crate::pool::SystemPool;
        static POOL: SystemPool = SystemPool;
        let mut observer = ();
        let mut context = SuperGraphContext::new(&POOL, &mut observer);

        let result = graph.eval(&[], SuperGraphData::new(), &mut context);
        assert!(matches!(result, Err(SuperGraphError::InvalidGraph(_))));
    }
}
