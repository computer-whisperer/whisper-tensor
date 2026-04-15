use crate::DynRank;
use crate::graph::{
    GlobalId, Graph, Node, NodeMetadata, NodeSlotEditError, Property, PropertyValue, SlotDirection,
};
use crate::metadata::TokenizerInfo;
use crate::milli_graph::MilliOpGraph;
use crate::milli_graph::observer::MilliOpGraphObserver;
use crate::numeric_dtype::NumericDType;
use crate::numeric_scalar::NumericScalar;
use crate::numeric_tensor::NumericTensor as PoolNumericTensor;
use crate::phonemization::{text_to_kokoro_phonemes, text_to_piper_phonemes};
use crate::pool::Pool;
use crate::super_graph::data::{SuperGraphAudioClip, SuperGraphImage};
use crate::super_graph::links::{
    SuperGraphAnyLink, SuperGraphLink, SuperGraphLinkDouble, SuperGraphLinkKind,
    SuperGraphLinkTriple,
};
use crate::super_graph::observer::SuperGraphObserver;
use crate::super_graph::{
    SuperGraph, SuperGraphBuilder, SuperGraphContext, SuperGraphData, SuperGraphError,
};

use crate::tokenizer::{AnyTokenizer, Tokenizer};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use std::time::Instant;

pub trait SuperGraphNode {
    fn to_any(self) -> SuperGraphAnyNode;

    fn op_kind(&self) -> String;
    fn label(&self) -> Option<String> {
        None
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_>;
    fn outputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_>;
    fn input_slots(&self) -> Box<dyn Iterator<Item = Option<SuperGraphAnyLink>> + '_> {
        Box::new(self.inputs().map(Some))
    }
    fn output_slots(&self) -> Box<dyn Iterator<Item = Option<SuperGraphAnyLink>> + '_> {
        Box::new(self.outputs().map(Some))
    }
    fn set_input_slot(
        &mut self,
        slot_index: usize,
        _link: Option<GlobalId>,
    ) -> Result<(), NodeSlotEditError> {
        Err(NodeSlotEditError::unsupported(
            self.op_kind(),
            SlotDirection::Input,
            slot_index,
        ))
    }
    fn set_output_slot(
        &mut self,
        slot_index: usize,
        _link: Option<GlobalId>,
    ) -> Result<(), NodeSlotEditError> {
        Err(NodeSlotEditError::unsupported(
            self.op_kind(),
            SlotDirection::Output,
            slot_index,
        ))
    }
    fn global_id(&self) -> GlobalId;

    fn eval<'short, 'model, 'p, P: Pool + 'p, T: SuperGraphObserver>(
        &'short self,
        this_path: &[GlobalId],
        data: &mut SuperGraphData<'p, 'model, P>,
        context: &mut SuperGraphContext<'short, 'model, 'p, P, T>,
    ) -> Result<(), SuperGraphError>;
}

impl<T: SuperGraphNode> Node for T {
    type OpKind = String;

    fn global_id(&self) -> GlobalId {
        <Self as SuperGraphNode>::global_id(self)
    }

    fn op_kind(&self) -> Self::OpKind {
        <Self as SuperGraphNode>::op_kind(self)
    }

    fn label(&self) -> Option<String> {
        <Self as SuperGraphNode>::label(self)
    }

    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId> + '_> {
        Box::new(<Self as SuperGraphNode>::inputs(self).map(|x| x.global_id()))
    }

    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId> + '_> {
        Box::new(<Self as SuperGraphNode>::outputs(self).map(|x| x.global_id()))
    }

    fn input_slots(&self) -> Box<dyn Iterator<Item = Option<GlobalId>> + '_> {
        Box::new(
            <Self as SuperGraphNode>::input_slots(self).map(|x| x.map(|link| link.global_id())),
        )
    }

    fn output_slots(&self) -> Box<dyn Iterator<Item = Option<GlobalId>> + '_> {
        Box::new(
            <Self as SuperGraphNode>::output_slots(self).map(|x| x.map(|link| link.global_id())),
        )
    }

    fn set_input_slot(
        &mut self,
        slot_index: usize,
        link: Option<GlobalId>,
    ) -> Result<(), NodeSlotEditError> {
        <Self as SuperGraphNode>::set_input_slot(self, slot_index, link)
    }

    fn set_output_slot(
        &mut self,
        slot_index: usize,
        link: Option<GlobalId>,
    ) -> Result<(), NodeSlotEditError> {
        <Self as SuperGraphNode>::set_output_slot(self, slot_index, link)
    }
}

fn require_node_link(
    maybe_link: Option<SuperGraphLink>,
    node_kind: &str,
    field_name: &str,
) -> Result<SuperGraphLink, SuperGraphError> {
    maybe_link.ok_or_else(|| {
        SuperGraphError::MissingLinkError(format!(": {node_kind} missing link '{field_name}'"))
    })
}

fn set_slot_link(link: Option<GlobalId>, kind: SuperGraphLinkKind) -> Option<SuperGraphLink> {
    link.map(|global_id| SuperGraphLink::with_global_id(global_id, kind))
}

fn set_slot_link_like(
    link: Option<GlobalId>,
    like: Option<SuperGraphLink>,
    op_kind: &str,
    direction: SlotDirection,
    slot_index: usize,
) -> Result<Option<SuperGraphLink>, NodeSlotEditError> {
    let kind = like.map(|x| x.kind()).ok_or_else(|| {
        NodeSlotEditError::missing_slot_kind(op_kind.to_string(), direction, slot_index)
    })?;
    Ok(set_slot_link(link, kind))
}

fn tensor_bool_scalar<'p, P: Pool + 'p>(
    value: bool,
    pool: &'p P,
) -> Result<PoolNumericTensor<'p, DynRank, P>, SuperGraphError> {
    use crate::numeric_dtype::NumericDType;
    use crate::numeric_tensor::TensorLayout;
    let layout = TensorLayout::<DynRank>::row_major(vec![], NumericDType::BOOL);
    let buf = pool
        .allocate(layout.buffer_size_bytes())
        .map_err(|e| SuperGraphError::InvalidInputError(format!("allocation: {e}")))?;
    let mut t = PoolNumericTensor::from_parts(buf, layout);
    t.write_element(0, crate::numeric_scalar::NumericScalar::from_bool(value));
    Ok(t)
}

fn read_rank0_bool_tensor<P: Pool>(
    tensor: &PoolNumericTensor<'_, DynRank, P>,
    input_name: &str,
) -> Result<bool, SuperGraphError> {
    if tensor.numel() != 1 {
        return Err(SuperGraphError::InvalidInputError(format!(
            "{} must be a rank-0 bool tensor, got shape {:?}",
            input_name,
            tensor.shape()
        )));
    }
    Ok(tensor.read_element(0).is_nonzero())
}

fn alloc_err(e: crate::pool::AllocationError) -> SuperGraphError {
    SuperGraphError::InvalidInputError(format!("allocation: {e}"))
}

/// Stack N tensors along `axis`. Each input must have size 1 along
/// `axis`; all other dims must match. The output has size N along
/// `axis` and preserves all other dims. Used by `RNNCacheRead` to
/// assemble a batched state from per-row cached entries.
fn stack_along_axis<'p, P: Pool + 'p>(
    parts: &[&PoolNumericTensor<'p, DynRank, P>],
    axis: usize,
    pool: &'p P,
) -> Result<PoolNumericTensor<'p, DynRank, P>, SuperGraphError> {
    if parts.is_empty() {
        return Err(SuperGraphError::InvalidInputError(
            "stack_along_axis: empty parts".to_string(),
        ));
    }
    let row_shape = parts[0].shape();
    if axis >= row_shape.len() {
        return Err(SuperGraphError::InvalidInputError(format!(
            "stack_along_axis: axis {axis} out of range for shape {:?}",
            row_shape
        )));
    }
    if row_shape[axis] != 1 {
        return Err(SuperGraphError::InvalidInputError(format!(
            "stack_along_axis: each part must have size 1 at axis {axis}, got {}",
            row_shape[axis]
        )));
    }
    for (i, p) in parts.iter().enumerate().skip(1) {
        if p.shape() != row_shape {
            return Err(SuperGraphError::InvalidInputError(format!(
                "stack_along_axis: part {i} shape {:?} != part 0 shape {:?}",
                p.shape(),
                row_shape
            )));
        }
    }

    let n = parts.len() as u64;
    let mut out_shape = row_shape.clone();
    out_shape[axis] = n;
    let post_axis_size: u64 = out_shape.iter().skip(axis + 1).product();
    let dtype = parts[0].dtype();
    PoolNumericTensor::from_fn(out_shape, dtype, pool, |flat_idx| {
        let f = flat_idx as u64;
        let post_idx = f % post_axis_size;
        let batch_row = (f / post_axis_size) % n;
        let pre_idx = f / (post_axis_size * n);
        // Per-row size at `axis` is 1, so per-row flat =
        // pre_idx * post_axis_size + post_idx.
        let per_row_flat = pre_idx * post_axis_size + post_idx;
        parts[batch_row as usize].read_element(per_row_flat as usize)
    })
    .map_err(alloc_err)
}

fn check_rank0(shape: &[u64], input_name: &str) -> Result<(), SuperGraphError> {
    if shape.len() > 1 || (shape.len() == 1 && shape[0] != 1) {
        return Err(SuperGraphError::InvalidInputError(format!(
            "{} must be a scalar (rank-0 or rank-1 with 1 element), got rank={} shape={:?}",
            input_name,
            shape.len(),
            shape
        )));
    }
    Ok(())
}

fn read_rank0_i64_tensor<P: Pool>(
    tensor: &PoolNumericTensor<'_, DynRank, P>,
    input_name: &str,
) -> Result<i64, SuperGraphError> {
    check_rank0(tensor.shape().as_slice(), input_name)?;
    Ok(tensor.read_element(0).to_i64())
}

fn read_rank0_f64_tensor<P: Pool>(
    tensor: &PoolNumericTensor<'_, DynRank, P>,
    input_name: &str,
) -> Result<f64, SuperGraphError> {
    check_rank0(tensor.shape().as_slice(), input_name)?;
    Ok(tensor.read_element(0).to_f64())
}

fn parse_piper_phoneme_id_map(json: &str) -> Result<HashMap<char, Vec<i64>>, SuperGraphError> {
    let value: serde_json::Value = serde_json::from_str(json).map_err(|e| {
        SuperGraphError::InvalidInputError(format!("invalid Piper phoneme_id_map JSON: {e}"))
    })?;
    let obj = value.as_object().ok_or(SuperGraphError::InvalidInputError(
        "Piper phoneme_id_map is not an object".to_string(),
    ))?;
    let mut map = HashMap::new();
    for (key, val) in obj {
        if key.chars().count() != 1 {
            continue;
        }
        let ch = key.chars().next().unwrap();
        let ids = val
            .as_array()
            .ok_or(SuperGraphError::InvalidInputError(format!(
                "Piper phoneme_id_map[{key}] is not an array"
            )))?
            .iter()
            .map(|v| {
                v.as_i64().ok_or(SuperGraphError::InvalidInputError(format!(
                    "Piper phoneme_id_map[{key}] contains non-i64 value"
                )))
            })
            .collect::<Result<Vec<_>, _>>()?;
        map.insert(ch, ids);
    }
    Ok(map)
}

fn load_kokoro_vocab(info: &TokenizerInfo) -> Result<HashMap<char, u32>, SuperGraphError> {
    let path = match info {
        TokenizerInfo::HFTokenizerLocal(path) => path,
        _ => {
            return Err(SuperGraphError::InvalidInputError(
                "Kokoro phoneme tokenizer must be HFTokenizerLocal".to_string(),
            ));
        }
    };
    let json = std::fs::read_to_string(path).map_err(|e| {
        SuperGraphError::InvalidInputError(format!(
            "failed to read Kokoro tokenizer file {path}: {e}"
        ))
    })?;
    let value: serde_json::Value = serde_json::from_str(&json).map_err(|e| {
        SuperGraphError::InvalidInputError(format!(
            "invalid JSON in Kokoro tokenizer file {path}: {e}"
        ))
    })?;
    let vocab_obj =
        value["model"]["vocab"]
            .as_object()
            .ok_or(SuperGraphError::InvalidInputError(format!(
                "missing model.vocab object in Kokoro tokenizer file {path}"
            )))?;
    let mut vocab = HashMap::new();
    for (key, val) in vocab_obj {
        if key.chars().count() != 1 {
            continue;
        }
        let id = val
            .as_u64()
            .ok_or(SuperGraphError::InvalidInputError(format!(
                "non-u64 vocab id for key {key} in {path}"
            )))? as u32;
        vocab.insert(key.chars().next().unwrap(), id);
    }
    Ok(vocab)
}

fn build_f5_vocab(vocab_text: &str) -> HashMap<char, i32> {
    let mut map = HashMap::new();
    for (id, line) in vocab_text.lines().enumerate() {
        if line.chars().count() == 1 {
            map.insert(line.chars().next().unwrap(), id as i32);
        } else if line.is_empty() {
            // Line 0 is space in F5 vocab.
            map.insert(' ', id as i32);
        }
    }
    map
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelReference {
    idx: usize,
}

/// Compile-time upper bound for a symbolic dim group.
///
/// Sym dims have no inherent bound — at lowering they're just identities.
/// The compiled path needs a max so the placer can reserve a worst-case
/// footprint per atom for max-bound sym expansion. Each variant resolves
/// to a single u64 at the moment compilation runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SymDimBound {
    /// Compile-time hard cap. The plan handles any runtime dim ≤ `max`.
    /// Use this when you know the maximum exactly (e.g. interface
    /// `max_batch`).
    Fixed { max: u64 },
    /// Compile at `factor` × the runtime dim size that triggered
    /// compilation. A future call exceeding the resolved bound forces
    /// a cache miss → recompile at the new size. Useful when the dim
    /// grows over time and you want amortized recompile cost.
    Headroom { factor: u64 },
}

/// Sparse symbolic-dim declaration for a `SuperGraphNodeModelExecution`.
///
/// Each entry in `entries` pins a single `(tensor_input_name, dim_index)`
/// to a named group. Dims sharing a group name are bound to the same
/// symbolic scalar during lowering — two matmul inputs with `(A, 0)` and
/// `(B, 1)` both in group "K" produce a nano graph whose contraction
/// dim is symbolic and shared. Dims not listed stay concrete (bound to
/// the runtime input shape at eval time).
///
/// `group_bounds` carries a per-group compile-time max for the compiled
/// path. Lowered eval ignores it. Groups appearing in `entries` but not
/// in `group_bounds` cause the compiled path to error — every symbolic
/// dim must have a bound for the placer to size buffers.
///
/// The runtime input tensors still have concrete shapes; the override only
/// controls how `build_info_inputs` describes those inputs to the lowering
/// pipeline, which in turn determines what shows up as symbolic in the
/// resulting nano graph.
#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct SymbolicInputDims {
    /// `((tensor_input_name, dim_index), group_name)` entries. Using a Vec
    /// (rather than a HashMap) keeps the serialization order stable and makes
    /// it trivial to author by hand.
    pub entries: Vec<((String, usize), String)>,
    /// Per-group compile-time bound. Compiled mode requires every group
    /// referenced in `entries` to appear here. Lowered mode ignores
    /// the field entirely.
    #[serde(default)]
    pub group_bounds: HashMap<String, SymDimBound>,
}

impl SymbolicInputDims {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn push(
        &mut self,
        input_name: impl Into<String>,
        dim_index: usize,
        group: impl Into<String>,
    ) {
        self.entries
            .push(((input_name.into(), dim_index), group.into()));
    }

    /// Set the compile-time max for a group. Required for every group
    /// referenced in `entries` when using compiled mode.
    pub fn with_group_bound(mut self, group: impl Into<String>, bound: SymDimBound) -> Self {
        self.group_bounds.insert(group.into(), bound);
        self
    }

    /// Mutating variant of `with_group_bound`.
    pub fn set_group_bound(&mut self, group: impl Into<String>, bound: SymDimBound) {
        self.group_bounds.insert(group.into(), bound);
    }

    /// Lookup the group name, if any, assigned to a given (input_name, dim_index).
    pub fn group_for(&self, input_name: &str, dim_index: usize) -> Option<&str> {
        self.entries.iter().find_map(|((name, dim), group)| {
            if name == input_name && *dim == dim_index {
                Some(group.as_str())
            } else {
                None
            }
        })
    }
}

/// Reproduce the symbol-id derivation used by `lowered_eval::build_info_inputs`.
/// Same hash, same namespace prefix — two paths produce identical sids for
/// the same group name so the compile path can map group → GraphConstantId
/// via the lower's `symbol_id_map`.
#[cfg(feature = "x86_compile")]
fn group_name_to_symbol_id(group: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    "symbolic_input_dims:".hash(&mut h);
    group.hash(&mut h);
    h.finish()
}

/// Resolve `SymbolicInputDims::group_bounds` against runtime input shapes
/// and the lowered model's `symbol_id_map`, producing the per-GC max
/// overrides the placer needs.
///
/// Returns `Err` if any group referenced in `symbolic_dim_overrides` lacks
/// a configured bound, or if a Headroom group's input has no runtime view.
/// The caller treats `Err` as "fall back to symbolic eval" rather than a
/// hard panic — bounds misconfiguration is recoverable.
#[cfg(feature = "x86_compile")]
fn resolve_gc_max_overrides(
    sym_dims: &SymbolicInputDims,
    symbolic_dim_overrides: &HashMap<GlobalId, Vec<(usize, String)>>,
    user_input_view_map: &HashMap<GlobalId, crate::numeric_tensor::NumericTensorView<'_, DynRank>>,
    symbol_id_map: &HashMap<u64, crate::nano_graph::pattern::GraphConstantId>,
) -> Result<HashMap<crate::nano_graph::pattern::GraphConstantId, u64>, String> {
    let mut group_max_by_name: HashMap<String, u64> = HashMap::new();

    for (tensor_id, entries) in symbolic_dim_overrides {
        for (dim_idx, group_name) in entries {
            if group_max_by_name.contains_key(group_name) {
                continue;
            }
            let bound = sym_dims.group_bounds.get(group_name).ok_or_else(|| {
                format!(
                    "no SymDimBound configured for sym group {:?} \
                     (set via SymbolicInputDims::with_group_bound)",
                    group_name
                )
            })?;
            let max = match bound {
                SymDimBound::Fixed { max } => *max,
                SymDimBound::Headroom { factor } => {
                    let view = user_input_view_map.get(tensor_id).ok_or_else(|| {
                        format!(
                            "Headroom bound on group {:?} requires a runtime \
                             view for tensor {:?}, none supplied",
                            group_name, tensor_id
                        )
                    })?;
                    let shape = view.shape();
                    let dim_val = *shape.get(*dim_idx).ok_or_else(|| {
                        format!(
                            "Headroom bound on group {:?}: dim_idx {} out of \
                             range for tensor {:?} (rank {})",
                            group_name,
                            dim_idx,
                            tensor_id,
                            shape.len()
                        )
                    })?;
                    dim_val.saturating_mul(*factor)
                }
            };
            group_max_by_name.insert(group_name.clone(), max);
        }
    }

    let mut out = HashMap::new();
    for (group_name, max) in group_max_by_name {
        let sid = group_name_to_symbol_id(&group_name);
        // A group can be in symbolic_dim_overrides but not in the
        // lowered graph (lowering may have constant-folded the dim
        // away). Skip silently in that case — there's no GC to bind.
        if let Some(&gc) = symbol_id_map.get(&sid) {
            out.insert(gc, max);
        }
    }
    Ok(out)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeModelExecution {
    global_id: GlobalId,
    pub label: Option<String>,
    tensor_map: Option<SuperGraphLink>,
    pub symbolic_graph_id: usize, // Which graph (passed to
    tensor_inputs: Vec<(Option<SuperGraphLink>, String)>,
    tensor_outputs: Vec<(String, Option<SuperGraphLink>)>,
    /// Optional per-input dim symbolic overrides. Dims listed here are lowered
    /// as symbolic scalars (shared across entries with the same group name)
    /// instead of being bound to the runtime shape. Empty = current behavior.
    pub symbolic_input_dims: SymbolicInputDims,
}

impl SuperGraphNodeModelExecution {
    pub fn new(
        rng: &mut impl Rng,
        tensor_map: SuperGraphLink,
        symbolic_graph_id: usize,
        tensor_inputs: Vec<(SuperGraphLink, String)>,
        tensor_outputs: Vec<(String, SuperGraphLink)>,
    ) -> Self {
        Self {
            global_id: GlobalId::new(rng),
            label: None,
            tensor_map: Some(tensor_map),
            symbolic_graph_id,
            tensor_inputs: tensor_inputs
                .into_iter()
                .map(|(link, name)| (Some(link), name))
                .collect(),
            tensor_outputs: tensor_outputs
                .into_iter()
                .map(|(name, link)| (name, Some(link)))
                .collect(),
            symbolic_input_dims: SymbolicInputDims::default(),
        }
    }

    /// Builder-style setter for the symbolic-dim config.
    pub fn with_symbolic_input_dims(mut self, dims: SymbolicInputDims) -> Self {
        self.symbolic_input_dims = dims;
        self
    }
}

/// Adapts a `SuperGraphObserver` into a `SymbolicGraphObserver` by prepending
/// the node path to all forwarded events.
struct SymbolicGraphObserverWrapper<'a, T: SuperGraphObserver> {
    inner: &'a mut T,
    path: Vec<GlobalId>,
}

impl<'a, T: SuperGraphObserver> SymbolicGraphObserverWrapper<'a, T> {
    fn new(inner: &'a mut T, path: &[GlobalId]) -> Self {
        Self {
            inner,
            path: path.to_vec(),
        }
    }
}

impl<T: SuperGraphObserver> crate::symbolic_graph::observer::SymbolicGraphObserver
    for SymbolicGraphObserverWrapper<'_, T>
{
    fn on_op_executed(
        &mut self,
        node_path: &[GlobalId],
        start_instant: Instant,
        end_instant: Instant,
    ) {
        let full_path: Vec<GlobalId> = self.path.iter().chain(node_path.iter()).copied().collect();
        self.inner
            .on_node_executed(&full_path, "", start_instant, end_instant);
    }

    fn on_tensor_assigned(
        &mut self,
        tensor_path: &[GlobalId],
        tensor: &crate::numeric_tensor::NumericTensorView<'_, crate::tensor_rank::DynRank>,
    ) {
        let full_path: Vec<GlobalId> = self
            .path
            .iter()
            .chain(tensor_path.iter())
            .copied()
            .collect();
        self.inner.on_tensor_assigned(&full_path, tensor);
    }

    fn on_loading_weight(&mut self, path: &[GlobalId], weight_name: Option<String>) {
        let full_path: Vec<GlobalId> = self.path.iter().chain(path.iter()).copied().collect();
        self.inner.on_loading_weight(&full_path, weight_name);
    }

    fn should_cancel(&mut self) -> bool {
        self.inner.should_cancel()
    }
}

impl SuperGraphNode for SuperGraphNodeModelExecution {
    fn to_any(self) -> SuperGraphAnyNode {
        SuperGraphAnyNode::ModelExecution(self)
    }
    fn eval<'short, 'model, 'p, P: Pool + 'p, T: SuperGraphObserver>(
        &'short self,
        node_path: &[GlobalId],
        data: &mut SuperGraphData<'p, 'model, P>,
        context: &mut SuperGraphContext<'short, 'model, 'p, P, T>,
    ) -> Result<(), SuperGraphError> {
        let tensor_map = require_node_link(self.tensor_map, "ModelExecution", "tensor_map")?;
        let tensor_store = data.tensor_maps.get(&tensor_map).ok_or_else(|| {
            SuperGraphError::MissingLinkError(format!(
                ": ModelExecution missing tensor map value {:?}",
                tensor_map
            ))
        })?;

        let symbolic_graph = context.symbolic_graphs[self.symbolic_graph_id];

        let tensors_by_name = symbolic_graph.get_tensors_by_name();
        let input_views: Vec<(
            GlobalId,
            crate::numeric_tensor::NumericTensorView<'_, DynRank>,
        )> = self
            .tensor_inputs
            .iter()
            .filter_map(|(link, name)| {
                let link = (*link)?;
                let &tensor_id = tensors_by_name.get(name)?;
                let tensor = data.tensors.get(&link)?;
                Some((tensor_id, tensor.view()))
            })
            .collect();

        // Resolve the user-facing (input_name, dim) → group config into a
        // GlobalId-keyed map that matches what `build_info_inputs` expects.
        // Entries that don't resolve (unknown input name) are silently
        // dropped — the user's config referred to a tensor the model does
        // not expose, which is either a typo or stale config. Surfacing
        // this via an error would force every caller to handle a new
        // error variant; silently ignoring keeps the knob additive.
        let symbolic_dim_overrides: HashMap<GlobalId, Vec<(usize, String)>> = {
            let mut out: HashMap<GlobalId, Vec<(usize, String)>> = HashMap::new();
            for ((input_name, dim_idx), group) in &self.symbolic_input_dims.entries {
                if let Some(&tensor_id) = tensors_by_name.get(input_name) {
                    out.entry(tensor_id)
                        .or_default()
                        .push((*dim_idx, group.clone()));
                }
            }
            out
        };

        // Try the lowered or compiled eval path if configured.
        match &context.eval_options.model_eval_mode {
            crate::super_graph::ModelEvalMode::LoweredEval {
                inline_constant_threshold,
            } => {
                if let Some(results) = self.try_lowered_eval(
                    node_path,
                    symbolic_graph,
                    tensor_store,
                    &input_views,
                    *inline_constant_threshold,
                    &symbolic_dim_overrides,
                    context,
                )? {
                    self.insert_results(results, &tensors_by_name, data)?;
                    return Ok(());
                }
                // Fall through to symbolic eval if lowered path declined.
            }
            #[cfg(feature = "x86_compile")]
            crate::super_graph::ModelEvalMode::CompiledEval {
                inline_constant_threshold,
                compile_options,
            } => {
                let compile_options = compile_options.clone();
                if let Some(results) = self.try_compiled_eval(
                    node_path,
                    symbolic_graph,
                    tensor_store,
                    &input_views,
                    *inline_constant_threshold,
                    &compile_options,
                    &symbolic_dim_overrides,
                    context,
                )? {
                    self.insert_results(results, &tensors_by_name, data)?;
                    return Ok(());
                }
                // Fall through to symbolic eval if compiled path declined.
            }
            _ => {}
        }

        // Symbolic eval path (default).
        let view_map: HashMap<
            GlobalId,
            &crate::numeric_tensor::NumericTensorView<'_, crate::tensor_rank::DynRank>,
        > = input_views.iter().map(|(id, view)| (*id, view)).collect();

        let global_id = node_path
            .iter()
            .chain(core::iter::once(&self.global_id))
            .copied()
            .collect::<Vec<_>>();
        let mut observer = SymbolicGraphObserverWrapper::new(context.observer, &global_id);

        let results = symbolic_graph.pool_eval_with_store_observed(
            &view_map,
            tensor_store,
            context.pool,
            &mut observer,
        )?;

        self.insert_results(results, &tensors_by_name, data)?;
        Ok(())
    }

    fn op_kind(&self) -> String {
        "Model Execution".to_string()
    }
    fn label(&self) -> Option<String> {
        self.label.clone()
    }

    fn inputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        let ret = self
            .tensor_inputs
            .iter()
            .filter_map(|x| x.0.map(|link| link.to_any()));
        Box::new(ret.chain(self.tensor_map.map(|link| link.to_any())))
    }

    fn outputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(
            self.tensor_outputs
                .iter()
                .filter_map(|x| x.1.map(|link| link.to_any())),
        )
    }

    fn global_id(&self) -> GlobalId {
        self.global_id
    }
}

impl SuperGraphNodeModelExecution {
    /// Map eval results back to super graph data links by tensor name.
    fn insert_results<'p, 'model, P: Pool + 'p>(
        &self,
        results: HashMap<GlobalId, crate::numeric_tensor::NumericTensor<'p, DynRank, P>>,
        tensors_by_name: &HashMap<String, GlobalId>,
        data: &mut SuperGraphData<'p, 'model, P>,
    ) -> Result<(), SuperGraphError> {
        let tensors_by_id: HashMap<GlobalId, &str> = tensors_by_name
            .iter()
            .map(|(name, &id)| (id, name.as_str()))
            .collect();

        for (id, tensor) in results {
            if let Some(&name) = tensors_by_id.get(&id) {
                for (out_name, link) in &self.tensor_outputs {
                    if out_name == name {
                        let link = require_node_link(*link, "ModelExecution", "tensor_outputs")?;
                        data.tensors.insert(link, tensor);
                        break;
                    }
                }
            }
        }
        Ok(())
    }

    /// Attempt the lowered eval path. Returns:
    /// - `Ok(Some(results))` if lowered eval succeeded
    /// - `Ok(None)` if the graph can't be lowered (fall back to symbolic eval)
    /// - `Err(e)` on hard failure
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn try_lowered_eval<'short, 'model, 'p, P: Pool + 'p, T: SuperGraphObserver>(
        &self,
        node_path: &[GlobalId],
        symbolic_graph: &crate::symbolic_graph::SymbolicGraph,
        tensor_store: &crate::symbolic_graph::tensor_store::TensorStore,
        input_views: &[(
            GlobalId,
            crate::numeric_tensor::NumericTensorView<'_, DynRank>,
        )],
        inline_constant_threshold: u64,
        symbolic_dim_overrides: &HashMap<GlobalId, Vec<(usize, String)>>,
        context: &mut SuperGraphContext<'short, 'model, 'p, P, T>,
    ) -> Result<
        Option<HashMap<GlobalId, crate::numeric_tensor::NumericTensor<'p, DynRank, P>>>,
        SuperGraphError,
    > {
        use crate::super_graph::lowered_eval;

        // Gate: only flat graphs (no Scan/If/LSTM sub-graphs).
        if !lowered_eval::can_lower_symbolic_graph(symbolic_graph) {
            return Ok(None);
        }

        // Build user input view map (symbolic graph tensor ID → view).
        let user_input_view_map: HashMap<
            GlobalId,
            crate::numeric_tensor::NumericTensorView<'_, DynRank>,
        > = input_views
            .iter()
            .map(|(id, v)| {
                (
                    *id,
                    crate::numeric_tensor::NumericTensorView::new(v.data(), v.layout().clone()),
                )
            })
            .collect();

        // Build info_inputs for lowering with the threshold policy.
        let (info_inputs, user_input_ext_ids, weight_input_ext_ids) =
            lowered_eval::build_info_inputs(
                symbolic_graph,
                tensor_store,
                &user_input_view_map,
                inline_constant_threshold,
                symbolic_dim_overrides,
            );

        let info_hash = lowered_eval::hash_info_inputs(&info_inputs);

        // Check cache.
        let sym_graph_id = {
            use crate::graph::Graph;
            symbolic_graph.global_id()
        };

        let cache_hit = context
            .caches
            .as_ref()
            .and_then(|c| c.lowered_model_cache.get(&sym_graph_id))
            .map(|cached| cached.info_inputs_hash == info_hash)
            .unwrap_or(false);

        // Lower on cache miss, keeping the result alive for execution.
        let lowered_owned: Option<lowered_eval::CachedLoweredModel>;
        if !cache_hit {
            let t0 = Instant::now();
            let cached = match lowered_eval::lower_symbolic_graph(
                symbolic_graph,
                &info_inputs,
                info_hash,
                user_input_ext_ids,
                weight_input_ext_ids,
            ) {
                Some(c) => c,
                None => return Ok(None),
            };
            let dt = t0.elapsed();
            eprintln!(
                "[lowered_eval] lowered in {:.0}ms (cache miss)",
                dt.as_secs_f64() * 1e3,
            );

            if let Some(caches) = &mut context.caches {
                caches.lowered_model_cache.insert(sym_graph_id, cached);
                lowered_owned = None;
            } else {
                lowered_owned = Some(cached);
            }
        } else {
            lowered_owned = None;
        }

        // Borrow the cached lowered model and the loaded-tensor-cache slot in
        // a single split borrow so both can coexist for the eval call. The
        // `cached` ref borrows from `lowered_model_cache`; the `loaded_cache`
        // mut ref borrows from the disjoint `loaded_tensor_cache` field.
        use crate::super_graph::cache::LoadedTensorCache;
        let (cached, loaded_cache): (
            &lowered_eval::CachedLoweredModel,
            Option<&mut LoadedTensorCache>,
        ) = match (&lowered_owned, context.caches.as_deref_mut()) {
            (Some(owned), Some(caches)) => (owned, Some(&mut caches.loaded_tensor_cache)),
            (Some(owned), None) => (owned, None),
            (None, Some(caches)) => {
                let cached = caches
                    .lowered_model_cache
                    .get(&sym_graph_id)
                    .expect("just inserted or validated cache entry");
                let loaded = &mut caches.loaded_tensor_cache;
                (cached, Some(loaded))
            }
            (None, None) => unreachable!("lowered_owned must be Some when no caches are available"),
        };

        // TODO: collect only subscribed tensor IDs once observer subscriptions
        // are threaded through. For now, request no intermediates to avoid
        // liveness conflicts in pool_eval.
        let intermediate_ids: Vec<GlobalId> = Vec::new();

        let results = lowered_eval::execute_lowered(
            cached,
            symbolic_graph,
            tensor_store,
            &user_input_view_map,
            &intermediate_ids,
            context.pool,
            loaded_cache,
        )?;

        // Feed observer with intermediate tensors, using the same path
        // format as the symbolic eval path: [model_exec_node_id, tensor_id].
        let observer_path: Vec<GlobalId> = node_path
            .iter()
            .chain(core::iter::once(&self.global_id))
            .copied()
            .collect();
        for (&tensor_id, tensor) in &results {
            let full_path: Vec<GlobalId> = observer_path
                .iter()
                .chain(core::iter::once(&tensor_id))
                .copied()
                .collect();
            context
                .observer
                .on_tensor_assigned(&full_path, &tensor.view());
        }

        Ok(Some(results))
    }

    /// Attempt the compiled eval path. Returns:
    /// - `Ok(Some(results))` if compiled eval succeeded
    /// - `Ok(None)` if the graph can't be lowered/compiled (fall back)
    /// - `Err(e)` on hard failure
    #[cfg(feature = "x86_compile")]
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn try_compiled_eval<'short, 'model, 'p, P: Pool + 'p, T: SuperGraphObserver>(
        &self,
        node_path: &[GlobalId],
        symbolic_graph: &crate::symbolic_graph::SymbolicGraph,
        tensor_store: &crate::symbolic_graph::tensor_store::TensorStore,
        input_views: &[(
            GlobalId,
            crate::numeric_tensor::NumericTensorView<'_, DynRank>,
        )],
        inline_constant_threshold: u64,
        compile_options: &crate::compiler::CompileOptions,
        symbolic_dim_overrides: &HashMap<GlobalId, Vec<(usize, String)>>,
        context: &mut SuperGraphContext<'short, 'model, 'p, P, T>,
    ) -> Result<
        Option<HashMap<GlobalId, crate::numeric_tensor::NumericTensor<'p, DynRank, P>>>,
        SuperGraphError,
    > {
        use crate::super_graph::{compiled_eval, lowered_eval};

        // Gate: only flat graphs (no Scan/If/LSTM sub-graphs).
        if !lowered_eval::can_lower_symbolic_graph(symbolic_graph) {
            return Ok(None);
        }

        // Path used for all milestone events emitted by this ModelExecution
        // call. Built once and shared between direct observer calls and the
        // CompiledEvalObserverWrapper passed into compiled_eval.
        let observer_path: Vec<GlobalId> = node_path
            .iter()
            .chain(core::iter::once(&self.global_id))
            .copied()
            .collect();

        // Build user input view map.
        let user_input_view_map: HashMap<
            GlobalId,
            crate::numeric_tensor::NumericTensorView<'_, DynRank>,
        > = input_views
            .iter()
            .map(|(id, v)| {
                (
                    *id,
                    crate::numeric_tensor::NumericTensorView::new(v.data(), v.layout().clone()),
                )
            })
            .collect();

        let sym_graph_id = {
            use crate::graph::Graph;
            symbolic_graph.global_id()
        };

        let user_inputs_signature = lowered_eval::hash_user_input_views(
            &user_input_view_map,
            inline_constant_threshold,
            symbolic_dim_overrides,
        );

        // Fast path: reuse the previously computed info-inputs hash when the
        // user-input shape/dtype signature and inline-constant policy match.
        let mut info_hash_opt = context
            .caches
            .as_ref()
            .and_then(|c| c.compiled_info_signature_cache.get(&sym_graph_id))
            .filter(|sig| {
                sig.inline_constant_threshold == inline_constant_threshold
                    && sig.user_inputs_signature == user_inputs_signature
            })
            .map(|sig| sig.info_inputs_hash);

        // Build details are only needed when lowering/compiling misses cache.
        let mut info_inputs_opt: Option<
            HashMap<
                GlobalId,
                crate::tensor_info::TensorInfo<'static, 'static, crate::pool::SystemPool>,
            >,
        > = None;
        let mut user_input_ext_ids: Vec<GlobalId> = Vec::new();
        let mut weight_input_ext_ids: Vec<GlobalId> = Vec::new();

        let build_info_inputs = |context: &mut SuperGraphContext<'short, 'model, 'p, P, T>| {
            let t_info = Instant::now();
            let (info_inputs, users, weights) = lowered_eval::build_info_inputs(
                symbolic_graph,
                tensor_store,
                &user_input_view_map,
                inline_constant_threshold,
                symbolic_dim_overrides,
            );
            context.observer.on_compiled_milestone(
                &observer_path,
                "compiled.exec.info_inputs_build",
                None,
                t_info,
                Instant::now(),
            );
            let info_hash = lowered_eval::hash_info_inputs(&info_inputs);
            if let Some(caches) = &mut context.caches {
                caches.compiled_info_signature_cache.insert(
                    sym_graph_id,
                    crate::super_graph::cache::CompiledInfoSignature {
                        user_inputs_signature,
                        inline_constant_threshold,
                        info_inputs_hash: info_hash,
                    },
                );
            }
            (info_inputs, users, weights, info_hash)
        };

        if info_hash_opt.is_none() {
            let (info_inputs, users, weights, info_hash) = build_info_inputs(context);
            info_inputs_opt = Some(info_inputs);
            user_input_ext_ids = users;
            weight_input_ext_ids = weights;
            info_hash_opt = Some(info_hash);
        }
        let mut info_hash = info_hash_opt.expect("info hash should be available");

        // --- Ensure lowered model available (cached or owned) ---
        let mut lower_cache_hit = context
            .caches
            .as_ref()
            .and_then(|c| c.lowered_model_cache.get(&sym_graph_id))
            .map(|cached| cached.info_inputs_hash == info_hash)
            .unwrap_or(false);

        let mut compile_cache_hit = context
            .caches
            .as_ref()
            .and_then(|c| c.compiled_plan_cache.get(&sym_graph_id))
            .map(|cached| cached.info_inputs_hash == info_hash)
            .unwrap_or(false);

        // If the quick signature path produced a hash but either cache entry is
        // missing/mismatched, materialize full info_inputs now so we can lower
        // and/or compile below.
        if !(lower_cache_hit && compile_cache_hit) && info_inputs_opt.is_none() {
            let (info_inputs, users, weights, rebuilt_hash) = build_info_inputs(context);
            info_inputs_opt = Some(info_inputs);
            user_input_ext_ids = users;
            weight_input_ext_ids = weights;
            info_hash = rebuilt_hash;
            lower_cache_hit = context
                .caches
                .as_ref()
                .and_then(|c| c.lowered_model_cache.get(&sym_graph_id))
                .map(|cached| cached.info_inputs_hash == info_hash)
                .unwrap_or(false);
            compile_cache_hit = context
                .caches
                .as_ref()
                .and_then(|c| c.compiled_plan_cache.get(&sym_graph_id))
                .map(|cached| cached.info_inputs_hash == info_hash)
                .unwrap_or(false);
        }

        let owned_lower: Option<lowered_eval::CachedLoweredModel>;
        if !lower_cache_hit {
            let info_inputs = info_inputs_opt
                .as_ref()
                .expect("info_inputs must be available when lowering is needed");
            let t0 = Instant::now();
            let cached = match lowered_eval::lower_symbolic_graph(
                symbolic_graph,
                info_inputs,
                info_hash,
                user_input_ext_ids,
                weight_input_ext_ids,
            ) {
                Some(c) => c,
                None => return Ok(None),
            };
            context.observer.on_compiled_milestone(
                &observer_path,
                "compiled.lower",
                None,
                t0,
                Instant::now(),
            );

            if let Some(caches) = &mut context.caches {
                caches.lowered_model_cache.insert(sym_graph_id, cached);
                owned_lower = None;
            } else {
                owned_lower = Some(cached);
            }
        } else {
            owned_lower = None;
        }

        let lower_ref = if let Some(ref owned) = owned_lower {
            owned
        } else {
            context
                .caches
                .as_ref()
                .and_then(|c| c.lowered_model_cache.get(&sym_graph_id))
                .expect("lowered model available")
        };

        // Resolve per-group sym bounds → per-GraphConstantId max overrides.
        // Required for the placer to size sym groups at max-bound footprint.
        // Errors here surface as "fall back to symbolic eval" rather than
        // hard failures so an incomplete bound config doesn't take down
        // the call.
        let gc_max_overrides = match resolve_gc_max_overrides(
            &self.symbolic_input_dims,
            symbolic_dim_overrides,
            &user_input_view_map,
            &lower_ref.symbol_id_map,
        ) {
            Ok(m) => m,
            Err(e) => {
                eprintln!(
                    "[compiled_eval] sym-bound resolution failed; falling back \
                     to symbolic eval: {e}"
                );
                return Ok(None);
            }
        };

        // Compile cache hit also requires the compile-time bounds match
        // — a plan compiled for batch_max=2 isn't safe to reuse for
        // batch=4, regardless of info_inputs equality.
        let compile_bounds_hash = compiled_eval::hash_gc_max_overrides(&gc_max_overrides);
        let compile_cache_hit = compile_cache_hit
            && context
                .caches
                .as_ref()
                .and_then(|c| c.compiled_plan_cache.get(&sym_graph_id))
                .map(|cached| cached.compile_bounds_hash == compile_bounds_hash)
                .unwrap_or(false);

        // --- Ensure compiled plan available (cached or owned) ---
        //
        // Compile if needed, then re-borrow both from cache (or owned).
        // We must drop lower_ref before mutably borrowing the cache.
        let compile_needed = !compile_cache_hit;

        let owned_compiled: Option<compiled_eval::CachedCompiledPlan>;
        if compile_needed {
            let compiled = {
                let mut wrapper = compiled_eval::CompiledEvalObserverWrapper::new(
                    context.observer,
                    observer_path.clone(),
                    None,
                );
                match compiled_eval::compile_lowered_model(
                    lower_ref,
                    symbolic_graph,
                    &gc_max_overrides,
                    compile_options,
                    &mut wrapper,
                ) {
                    Some(c) => c,
                    None => return Ok(None),
                }
            };
            // Drop the immutable borrow before mutable insert.
            let _ = lower_ref;

            if let Some(caches) = &mut context.caches {
                caches.compiled_plan_cache.insert(sym_graph_id, compiled);
                owned_compiled = None;
            } else {
                owned_compiled = Some(compiled);
            }
        } else {
            let _ = lower_ref;
            owned_compiled = None;
        }

        // Re-borrow lowered, compiled, and loaded-tensor-cache from their
        // final locations in a single split borrow so all three coexist for
        // the eval call. Disjoint fields on the same SuperGraphCache.
        use crate::super_graph::cache::LoadedTensorCache;
        let (cached_lower, cached_compiled, loaded_cache): (
            &lowered_eval::CachedLoweredModel,
            &compiled_eval::CachedCompiledPlan,
            Option<&mut LoadedTensorCache>,
        ) = match (&owned_lower, &owned_compiled, context.caches.as_deref_mut()) {
            (Some(ol), Some(oc), Some(caches)) => (ol, oc, Some(&mut caches.loaded_tensor_cache)),
            (Some(ol), Some(oc), None) => (ol, oc, None),
            (Some(ol), None, Some(caches)) => {
                let cc = caches
                    .compiled_plan_cache
                    .get(&sym_graph_id)
                    .expect("compiled plan available");
                let loaded = &mut caches.loaded_tensor_cache;
                (ol, cc, Some(loaded))
            }
            (None, Some(oc), Some(caches)) => {
                let cl = caches
                    .lowered_model_cache
                    .get(&sym_graph_id)
                    .expect("lowered model available");
                let loaded = &mut caches.loaded_tensor_cache;
                (cl, oc, Some(loaded))
            }
            (None, None, Some(caches)) => {
                let cl = caches
                    .lowered_model_cache
                    .get(&sym_graph_id)
                    .expect("lowered model available");
                let cc = caches
                    .compiled_plan_cache
                    .get(&sym_graph_id)
                    .expect("compiled plan available");
                let loaded = &mut caches.loaded_tensor_cache;
                (cl, cc, Some(loaded))
            }
            _ => unreachable!("owned values must be Some when no caches are available"),
        };

        let results = {
            let mut wrapper = compiled_eval::CompiledEvalObserverWrapper::new(
                context.observer,
                observer_path.clone(),
                None,
            );
            compiled_eval::execute_compiled(
                cached_compiled,
                cached_lower,
                symbolic_graph,
                tensor_store,
                &user_input_view_map,
                context.pool,
                loaded_cache,
                &mut wrapper,
            )?
        };

        // Feed observer.
        for (&tensor_id, tensor) in &results {
            let full_path: Vec<GlobalId> = observer_path
                .iter()
                .chain(core::iter::once(&tensor_id))
                .copied()
                .collect();
            context
                .observer
                .on_tensor_assigned(&full_path, &tensor.view());
        }

        Ok(Some(results))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeTokenizerLoad {
    global_id: GlobalId,
    pub label: Option<String>,
    info: TokenizerInfo,
    output: Option<SuperGraphLink>,
}

impl SuperGraphNodeTokenizerLoad {
    pub fn new(_builder: &mut SuperGraphBuilder, info: TokenizerInfo, rng: &mut impl Rng) -> Self {
        Self {
            global_id: GlobalId::new(rng),
            label: None,
            info,
            output: Some(SuperGraphLink::new(SuperGraphLinkKind::Tokenizer, rng)),
        }
    }

    pub fn new_and_add(
        builder: &mut SuperGraphBuilder,
        info: TokenizerInfo,
        rng: &mut impl Rng,
    ) -> SuperGraphLink {
        let node = Self::new(builder, info, rng);
        let output = node.get_tokenizer_output();
        builder.add_node(node.to_any());
        output
    }

    pub fn get_tokenizer_output(&self) -> SuperGraphLink {
        self.output
            .expect("tokenizer load output link should be configured")
    }
}

impl SuperGraphNode for SuperGraphNodeTokenizerLoad {
    fn to_any(self) -> SuperGraphAnyNode {
        SuperGraphAnyNode::TokenizerLoad(self)
    }
    fn eval<'short, 'model, 'p, P: Pool + 'p, T: SuperGraphObserver>(
        &'short self,
        _this_path: &[GlobalId],
        data: &mut SuperGraphData<'p, 'model, P>,
        _context: &mut SuperGraphContext<'short, 'model, 'p, P, T>,
    ) -> Result<(), SuperGraphError> {
        let tokenizer = AnyTokenizer::from_tokenizer_info(&self.info);
        let output = require_node_link(self.output, "TokenizerLoad", "output")?;
        data.tokenizers.insert(output, tokenizer);
        Ok(())
    }

    fn op_kind(&self) -> String {
        "Tokenizer Load".to_string()
    }
    fn label(&self) -> Option<String> {
        self.label.clone()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(std::iter::empty())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(self.output.map(|link| link.to_any()).into_iter())
    }
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SuperGraphNodeTokenizerEncodeMode {
    Plain,
    ClipStyle {
        seq_len: usize,
        bos: u32,
        eos: u32,
        pad: u32,
    },
    RawPad {
        seq_len: usize,
        pad: u32,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeTokenizerEncode {
    global_id: GlobalId,
    pub label: Option<String>,
    tokenizer: Option<SuperGraphLink>,
    text_input: Option<SuperGraphLink>,
    tensor_output: Option<SuperGraphLink>,
    mode: SuperGraphNodeTokenizerEncodeMode,
}

impl SuperGraphNodeTokenizerEncode {
    pub fn new(
        _builder: &mut SuperGraphBuilder,
        tokenizer: SuperGraphLink,
        text_input: SuperGraphLink,
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            global_id: GlobalId::new(rng),
            label: None,
            tokenizer: Some(tokenizer),
            text_input: Some(text_input),
            tensor_output: Some(SuperGraphLink::new(SuperGraphLinkKind::Tensor, rng)),
            mode: SuperGraphNodeTokenizerEncodeMode::Plain,
        }
    }

    pub fn new_with_mode(
        _builder: &mut SuperGraphBuilder,
        tokenizer: SuperGraphLink,
        text_input: SuperGraphLink,
        mode: SuperGraphNodeTokenizerEncodeMode,
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            global_id: GlobalId::new(rng),
            label: None,
            tokenizer: Some(tokenizer),
            text_input: Some(text_input),
            tensor_output: Some(SuperGraphLink::new(SuperGraphLinkKind::Tensor, rng)),
            mode,
        }
    }

    pub fn new_and_add(
        builder: &mut SuperGraphBuilder,
        tokenizer: SuperGraphLink,
        text_input: SuperGraphLink,
        rng: &mut impl Rng,
    ) -> SuperGraphLink {
        let node = Self::new(builder, tokenizer, text_input, rng);
        let output = node.get_tensor_output();
        builder.add_node(node.to_any());
        output
    }

    pub fn new_with_mode_and_add(
        builder: &mut SuperGraphBuilder,
        tokenizer: SuperGraphLink,
        text_input: SuperGraphLink,
        mode: SuperGraphNodeTokenizerEncodeMode,
        rng: &mut impl Rng,
    ) -> SuperGraphLink {
        let node = Self::new_with_mode(builder, tokenizer, text_input, mode, rng);
        let output = node.get_tensor_output();
        builder.add_node(node.to_any());
        output
    }

    pub fn get_tensor_output(&self) -> SuperGraphLink {
        self.tensor_output
            .expect("tokenizer encode output link should be configured")
    }
}

impl SuperGraphNode for SuperGraphNodeTokenizerEncode {
    fn to_any(self) -> SuperGraphAnyNode {
        SuperGraphAnyNode::TokenizerEncode(self)
    }
    fn eval<'short, 'model, 'p, P: Pool + 'p, T: SuperGraphObserver>(
        &'short self,
        _this_path: &[GlobalId],
        data: &mut SuperGraphData<'p, 'model, P>,
        context: &mut SuperGraphContext<'short, 'model, 'p, P, T>,
    ) -> Result<(), SuperGraphError> {
        let tokenizer_link = require_node_link(self.tokenizer, "TokenizerEncode", "tokenizer")?;
        let text_input_link = require_node_link(self.text_input, "TokenizerEncode", "text_input")?;
        let tensor_output_link =
            require_node_link(self.tensor_output, "TokenizerEncode", "tensor_output")?;

        let text = data.strings.get(&text_input_link).ok_or_else(|| {
            SuperGraphError::MissingLinkError(format!(
                ": TokenizerEncode missing text_input {:?}",
                text_input_link
            ))
        })?;
        let tokenizer = data.tokenizers.get(&tokenizer_link).ok_or_else(|| {
            SuperGraphError::MissingLinkError(format!(
                ": TokenizerEncode missing tokenizer {:?}",
                tokenizer_link
            ))
        })?;
        let pool = context.pool;
        let output_tensor = match &self.mode {
            SuperGraphNodeTokenizerEncodeMode::Plain => {
                let tokens = tokenizer
                    .encode(text)
                    .iter()
                    .map(|x| *x as i64)
                    .collect::<Vec<_>>();
                let n = tokens.len() as u64;
                PoolNumericTensor::from_fn(vec![1, 1, n], NumericDType::I64, pool, |i| {
                    NumericScalar::from_i64(tokens[i])
                })
                .map_err(alloc_err)?
            }
            SuperGraphNodeTokenizerEncodeMode::ClipStyle {
                seq_len,
                bos,
                eos,
                pad,
            } => {
                let mut encoded = tokenizer.encode(text);
                if encoded.first() == Some(bos) {
                    encoded.remove(0);
                }
                if encoded.last() == Some(eos) {
                    encoded.pop();
                }
                let max_text_tokens = seq_len.saturating_sub(2);
                let mut ids: Vec<i32> = Vec::with_capacity(*seq_len);
                ids.push(*bos as i32);
                for &id in encoded.iter().take(max_text_tokens) {
                    ids.push(id as i32);
                }
                ids.push(*eos as i32);
                ids.resize(*seq_len, *pad as i32);
                PoolNumericTensor::from_fn(vec![1, *seq_len as u64], NumericDType::I32, pool, |i| {
                    NumericScalar::from_i32(ids[i])
                })
                .map_err(alloc_err)?
            }
            SuperGraphNodeTokenizerEncodeMode::RawPad { seq_len, pad } => {
                let encoded = tokenizer.encode(text);
                let mut ids: Vec<i32> =
                    encoded.iter().take(*seq_len).map(|&id| id as i32).collect();
                ids.resize(*seq_len, *pad as i32);
                PoolNumericTensor::from_fn(vec![1, *seq_len as u64], NumericDType::I32, pool, |i| {
                    NumericScalar::from_i32(ids[i])
                })
                .map_err(alloc_err)?
            }
        };
        data.tensors.insert(tensor_output_link, output_tensor);
        Ok(())
    }

    fn op_kind(&self) -> String {
        "Tokenizer Encode".to_string()
    }
    fn label(&self) -> Option<String> {
        self.label.clone()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(
            self.tokenizer
                .map(|link| link.to_any())
                .into_iter()
                .chain(self.text_input.map(|link| link.to_any())),
        )
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(self.tensor_output.map(|link| link.to_any()).into_iter())
    }
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeTokenizerDecode {
    global_id: GlobalId,
    pub label: Option<String>,
    tokenizer: Option<SuperGraphLink>,
    tensor_input: Option<SuperGraphLink>,
    text_output: Option<SuperGraphLink>,
}

impl SuperGraphNodeTokenizerDecode {
    pub fn new(
        _builder: &mut SuperGraphBuilder,
        tokenizer: SuperGraphLink,
        tensor_input: SuperGraphLink,
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            global_id: GlobalId::new(rng),
            label: None,
            tokenizer: Some(tokenizer),
            tensor_input: Some(tensor_input),
            text_output: Some(SuperGraphLink::new(SuperGraphLinkKind::String, rng)),
        }
    }
    pub fn get_string_output(&self) -> SuperGraphLink {
        self.text_output
            .expect("tokenizer decode output link should be configured")
    }
    pub fn new_and_add(
        builder: &mut SuperGraphBuilder,
        tokenizer: SuperGraphLink,
        tensor_input: SuperGraphLink,
        rng: &mut impl Rng,
    ) -> SuperGraphLink {
        let node = Self::new(builder, tokenizer, tensor_input, rng);
        let output = node.get_string_output();
        builder.add_node(node.to_any());
        output
    }
}

impl SuperGraphNode for SuperGraphNodeTokenizerDecode {
    fn to_any(self) -> SuperGraphAnyNode {
        SuperGraphAnyNode::TokenizerDecode(self)
    }
    fn eval<'short, 'model, 'p, P: Pool + 'p, T: SuperGraphObserver>(
        &'short self,
        _node_path: &[GlobalId],
        data: &mut SuperGraphData<'p, 'model, P>,
        _context: &mut SuperGraphContext<'short, 'model, 'p, P, T>,
    ) -> Result<(), SuperGraphError> {
        let tokenizer_link = require_node_link(self.tokenizer, "TokenizerDecode", "tokenizer")?;
        let tensor_input_link =
            require_node_link(self.tensor_input, "TokenizerDecode", "tensor_input")?;
        let text_output_link =
            require_node_link(self.text_output, "TokenizerDecode", "text_output")?;

        let tensor = data.tensors.get(&tensor_input_link).ok_or_else(|| {
            SuperGraphError::MissingLinkError(format!(
                ": TokenizerDecode missing tensor_input {:?}",
                tensor_input_link
            ))
        })?;
        let tokenizer = data.tokenizers.get(&tokenizer_link).ok_or_else(|| {
            SuperGraphError::MissingLinkError(format!(
                ": TokenizerDecode missing tokenizer {:?}",
                tokenizer_link
            ))
        })?;
        let output_values: Vec<u32> = (0..tensor.numel())
            .map(|i| tensor.read_element(i).to_i64() as u32)
            .collect();
        let text = tokenizer.decode(&output_values)?;
        data.strings.insert(text_output_link, text);
        Ok(())
    }
    fn op_kind(&self) -> String {
        "Tokenizer Decode".to_string()
    }
    fn label(&self) -> Option<String> {
        self.label.clone()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(
            self.tokenizer
                .map(|link| link.to_any())
                .into_iter()
                .chain(self.tensor_input.map(|link| link.to_any())),
        )
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(self.text_output.map(|link| link.to_any()).into_iter())
    }
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SuperGraphNodeTextToPhonemesMode {
    Piper { voice: String },
    Kokoro { voice: String },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeTextToPhonemes {
    global_id: GlobalId,
    pub label: Option<String>,
    text_input: Option<SuperGraphLink>,
    phonemes_output: Option<SuperGraphLink>,
    mode: SuperGraphNodeTextToPhonemesMode,
}

impl SuperGraphNodeTextToPhonemes {
    pub fn new(
        _builder: &mut SuperGraphBuilder,
        text_input: SuperGraphLink,
        mode: SuperGraphNodeTextToPhonemesMode,
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            global_id: GlobalId::new(rng),
            label: None,
            text_input: Some(text_input),
            phonemes_output: Some(SuperGraphLink::new(SuperGraphLinkKind::String, rng)),
            mode,
        }
    }

    pub fn new_and_add(
        builder: &mut SuperGraphBuilder,
        text_input: SuperGraphLink,
        mode: SuperGraphNodeTextToPhonemesMode,
        rng: &mut impl Rng,
    ) -> SuperGraphLink {
        let node = Self::new(builder, text_input, mode, rng);
        let output = node.get_phonemes_output();
        builder.add_node(node.to_any());
        output
    }

    pub fn get_phonemes_output(&self) -> SuperGraphLink {
        self.phonemes_output
            .expect("text-to-phonemes output link should be configured")
    }
}

impl SuperGraphNode for SuperGraphNodeTextToPhonemes {
    fn to_any(self) -> SuperGraphAnyNode {
        SuperGraphAnyNode::TextToPhonemes(self)
    }

    fn eval<'short, 'model, 'p, P: Pool + 'p, T: SuperGraphObserver>(
        &'short self,
        _this_path: &[GlobalId],
        data: &mut SuperGraphData<'p, 'model, P>,
        _context: &mut SuperGraphContext<'short, 'model, 'p, P, T>,
    ) -> Result<(), SuperGraphError> {
        let text_input_link = require_node_link(self.text_input, "TextToPhonemes", "text_input")?;
        let phonemes_output_link =
            require_node_link(self.phonemes_output, "TextToPhonemes", "phonemes_output")?;
        let text = data
            .strings
            .get(&text_input_link)
            .ok_or(SuperGraphError::MissingLinkError(format!(
                ": missing text input {:?}",
                text_input_link
            )))?;
        let phonemes = match &self.mode {
            SuperGraphNodeTextToPhonemesMode::Piper { voice } => {
                text_to_piper_phonemes(text, voice)
            }
            SuperGraphNodeTextToPhonemesMode::Kokoro { voice } => {
                text_to_kokoro_phonemes(text, voice)
            }
        }
        .map_err(SuperGraphError::InvalidInputError)?;
        data.strings.insert(phonemes_output_link, phonemes);
        Ok(())
    }

    fn op_kind(&self) -> String {
        "TextToPhonemes".to_string()
    }
    fn label(&self) -> Option<String> {
        self.label.clone()
    }

    fn inputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(self.text_input.map(|link| link.to_any()).into_iter())
    }

    fn outputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(self.phonemes_output.map(|link| link.to_any()).into_iter())
    }

    fn global_id(&self) -> GlobalId {
        self.global_id
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodePiperPhonemesToTensor {
    global_id: GlobalId,
    pub label: Option<String>,
    phonemes_input: Option<SuperGraphLink>,
    token_ids_output: Option<SuperGraphLink>,
    input_lengths_output: Option<SuperGraphLink>,
    phoneme_id_map_json: String,
}

impl SuperGraphNodePiperPhonemesToTensor {
    pub fn new(
        _builder: &mut SuperGraphBuilder,
        phonemes_input: SuperGraphLink,
        phoneme_id_map_json: String,
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            global_id: GlobalId::new(rng),
            label: None,
            phonemes_input: Some(phonemes_input),
            token_ids_output: Some(SuperGraphLink::new(SuperGraphLinkKind::Tensor, rng)),
            input_lengths_output: Some(SuperGraphLink::new(SuperGraphLinkKind::Tensor, rng)),
            phoneme_id_map_json,
        }
    }

    pub fn new_and_add(
        builder: &mut SuperGraphBuilder,
        phonemes_input: SuperGraphLink,
        phoneme_id_map_json: String,
        rng: &mut impl Rng,
    ) -> (SuperGraphLink, SuperGraphLink) {
        let node = Self::new(builder, phonemes_input, phoneme_id_map_json, rng);
        let token_ids = node.get_token_ids_output();
        let input_lengths = node.get_input_lengths_output();
        builder.add_node(node.to_any());
        (token_ids, input_lengths)
    }

    pub fn get_token_ids_output(&self) -> SuperGraphLink {
        self.token_ids_output
            .expect("piper token_ids output link should be configured")
    }

    pub fn get_input_lengths_output(&self) -> SuperGraphLink {
        self.input_lengths_output
            .expect("piper input_lengths output link should be configured")
    }
}

impl SuperGraphNode for SuperGraphNodePiperPhonemesToTensor {
    fn to_any(self) -> SuperGraphAnyNode {
        SuperGraphAnyNode::PiperPhonemesToTensor(self)
    }

    fn eval<'short, 'model, 'p, P: Pool + 'p, T: SuperGraphObserver>(
        &'short self,
        _this_path: &[GlobalId],
        data: &mut SuperGraphData<'p, 'model, P>,
        context: &mut SuperGraphContext<'short, 'model, 'p, P, T>,
    ) -> Result<(), SuperGraphError> {
        let phonemes_input_link = require_node_link(
            self.phonemes_input,
            "PiperPhonemesToTensor",
            "phonemes_input",
        )?;
        let token_ids_output_link = require_node_link(
            self.token_ids_output,
            "PiperPhonemesToTensor",
            "token_ids_output",
        )?;
        let input_lengths_output_link = require_node_link(
            self.input_lengths_output,
            "PiperPhonemesToTensor",
            "input_lengths_output",
        )?;
        let phonemes =
            data.strings
                .get(&phonemes_input_link)
                .ok_or(SuperGraphError::MissingLinkError(format!(
                    ": missing phoneme input {:?}",
                    phonemes_input_link
                )))?;
        let phoneme_id_map = parse_piper_phoneme_id_map(&self.phoneme_id_map_json)?;

        let mut token_ids: Vec<i64> = vec![1, 0];
        for ch in phonemes.chars() {
            if let Some(ids) = phoneme_id_map.get(&ch) {
                token_ids.extend(ids.iter().copied());
            }
            token_ids.push(0);
        }
        token_ids.push(2);

        let num_tokens = token_ids.len();
        let pool = context.pool;
        data.tensors.insert(
            token_ids_output_link,
            PoolNumericTensor::from_fn(vec![1, num_tokens as u64], NumericDType::I64, pool, |i| {
                NumericScalar::from_i64(token_ids[i])
            })
            .map_err(alloc_err)?,
        );
        data.tensors.insert(
            input_lengths_output_link,
            PoolNumericTensor::from_fn(vec![1], NumericDType::I64, pool, |_| {
                NumericScalar::from_i64(num_tokens as i64)
            })
            .map_err(alloc_err)?,
        );
        Ok(())
    }

    fn op_kind(&self) -> String {
        "PiperPhonemesToTensor".to_string()
    }
    fn label(&self) -> Option<String> {
        self.label.clone()
    }

    fn inputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(self.phonemes_input.map(|link| link.to_any()).into_iter())
    }

    fn outputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(
            self.token_ids_output
                .map(|link| link.to_any())
                .into_iter()
                .chain(self.input_lengths_output.map(|link| link.to_any())),
        )
    }

    fn global_id(&self) -> GlobalId {
        self.global_id
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeKokoroPhonemesToTensor {
    global_id: GlobalId,
    pub label: Option<String>,
    phonemes_input: Option<SuperGraphLink>,
    token_ids_output: Option<SuperGraphLink>,
    tokenizer: TokenizerInfo,
}

impl SuperGraphNodeKokoroPhonemesToTensor {
    pub fn new(
        _builder: &mut SuperGraphBuilder,
        phonemes_input: SuperGraphLink,
        tokenizer: TokenizerInfo,
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            global_id: GlobalId::new(rng),
            label: None,
            phonemes_input: Some(phonemes_input),
            token_ids_output: Some(SuperGraphLink::new(SuperGraphLinkKind::Tensor, rng)),
            tokenizer,
        }
    }

    pub fn new_and_add(
        builder: &mut SuperGraphBuilder,
        phonemes_input: SuperGraphLink,
        tokenizer: TokenizerInfo,
        rng: &mut impl Rng,
    ) -> SuperGraphLink {
        let node = Self::new(builder, phonemes_input, tokenizer, rng);
        let output = node.get_token_ids_output();
        builder.add_node(node.to_any());
        output
    }

    pub fn get_token_ids_output(&self) -> SuperGraphLink {
        self.token_ids_output
            .expect("kokoro token_ids output link should be configured")
    }
}

impl SuperGraphNode for SuperGraphNodeKokoroPhonemesToTensor {
    fn to_any(self) -> SuperGraphAnyNode {
        SuperGraphAnyNode::KokoroPhonemesToTensor(self)
    }

    fn eval<'short, 'model, 'p, P: Pool + 'p, T: SuperGraphObserver>(
        &'short self,
        _this_path: &[GlobalId],
        data: &mut SuperGraphData<'p, 'model, P>,
        context: &mut SuperGraphContext<'short, 'model, 'p, P, T>,
    ) -> Result<(), SuperGraphError> {
        let phonemes_input_link = require_node_link(
            self.phonemes_input,
            "KokoroPhonemesToTensor",
            "phonemes_input",
        )?;
        let token_ids_output_link = require_node_link(
            self.token_ids_output,
            "KokoroPhonemesToTensor",
            "token_ids_output",
        )?;
        let phonemes =
            data.strings
                .get(&phonemes_input_link)
                .ok_or(SuperGraphError::MissingLinkError(format!(
                    ": missing phoneme input {:?}",
                    phonemes_input_link
                )))?;
        let vocab = load_kokoro_vocab(&self.tokenizer)?;
        let mut token_ids: Vec<i64> = vec![0]; // BOS ($)
        for ch in phonemes.chars() {
            if let Some(&id) = vocab.get(&ch) {
                token_ids.push(id as i64);
            }
        }
        token_ids.push(0); // EOS ($)
        let num_tokens = token_ids.len();
        data.tensors.insert(
            token_ids_output_link,
            PoolNumericTensor::from_fn(
                vec![1, num_tokens as u64],
                NumericDType::I64,
                context.pool,
                |i| NumericScalar::from_i64(token_ids[i]),
            )
            .map_err(alloc_err)?,
        );
        Ok(())
    }

    fn op_kind(&self) -> String {
        "KokoroPhonemesToTensor".to_string()
    }
    fn label(&self) -> Option<String> {
        self.label.clone()
    }

    fn inputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(self.phonemes_input.map(|link| link.to_any()).into_iter())
    }

    fn outputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(self.token_ids_output.map(|link| link.to_any()).into_iter())
    }

    fn global_id(&self) -> GlobalId {
        self.global_id
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeF5TextToTensor {
    global_id: GlobalId,
    pub label: Option<String>,
    text_input: Option<SuperGraphLink>,
    token_ids_output: Option<SuperGraphLink>,
    vocab: String,
}

impl SuperGraphNodeF5TextToTensor {
    pub fn new(
        _builder: &mut SuperGraphBuilder,
        text_input: SuperGraphLink,
        vocab: String,
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            global_id: GlobalId::new(rng),
            label: None,
            text_input: Some(text_input),
            token_ids_output: Some(SuperGraphLink::new(SuperGraphLinkKind::Tensor, rng)),
            vocab,
        }
    }

    pub fn new_and_add(
        builder: &mut SuperGraphBuilder,
        text_input: SuperGraphLink,
        vocab: String,
        rng: &mut impl Rng,
    ) -> SuperGraphLink {
        let node = Self::new(builder, text_input, vocab, rng);
        let output = node.get_token_ids_output();
        builder.add_node(node.to_any());
        output
    }

    pub fn get_token_ids_output(&self) -> SuperGraphLink {
        self.token_ids_output
            .expect("f5 token_ids output link should be configured")
    }
}

impl SuperGraphNode for SuperGraphNodeF5TextToTensor {
    fn to_any(self) -> SuperGraphAnyNode {
        SuperGraphAnyNode::F5TextToTensor(self)
    }

    fn eval<'short, 'model, 'p, P: Pool + 'p, T: SuperGraphObserver>(
        &'short self,
        _this_path: &[GlobalId],
        data: &mut SuperGraphData<'p, 'model, P>,
        context: &mut SuperGraphContext<'short, 'model, 'p, P, T>,
    ) -> Result<(), SuperGraphError> {
        let text_input_link = require_node_link(self.text_input, "F5TextToTensor", "text_input")?;
        let token_ids_output_link =
            require_node_link(self.token_ids_output, "F5TextToTensor", "token_ids_output")?;
        let text = data
            .strings
            .get(&text_input_link)
            .ok_or(SuperGraphError::MissingLinkError(format!(
                ": missing text input {:?}",
                text_input_link
            )))?;
        let vocab_map = build_f5_vocab(&self.vocab);
        let mut token_ids: Vec<i32> = Vec::new();
        for ch in text.chars() {
            if let Some(&id) = vocab_map.get(&ch) {
                token_ids.push(id);
            }
        }
        let num_tokens = token_ids.len();
        data.tensors.insert(
            token_ids_output_link,
            PoolNumericTensor::from_fn(
                vec![1, num_tokens as u64],
                NumericDType::I32,
                context.pool,
                |i| NumericScalar::from_i32(token_ids[i]),
            )
            .map_err(alloc_err)?,
        );
        Ok(())
    }

    fn op_kind(&self) -> String {
        "F5TextToTensor".to_string()
    }
    fn label(&self) -> Option<String> {
        self.label.clone()
    }

    fn inputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(self.text_input.map(|link| link.to_any()).into_iter())
    }

    fn outputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(self.token_ids_output.map(|link| link.to_any()).into_iter())
    }

    fn global_id(&self) -> GlobalId {
        self.global_id
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeTensorToImage {
    global_id: GlobalId,
    pub label: Option<String>,
    tensor_input: Option<SuperGraphLink>,
    image_output: Option<SuperGraphLink>,
}

impl SuperGraphNodeTensorToImage {
    pub fn new(
        _builder: &mut SuperGraphBuilder,
        tensor_input: SuperGraphLink,
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            global_id: GlobalId::new(rng),
            label: None,
            tensor_input: Some(tensor_input),
            image_output: Some(SuperGraphLink::new(SuperGraphLinkKind::Image, rng)),
        }
    }

    pub fn new_and_add(
        builder: &mut SuperGraphBuilder,
        tensor_input: SuperGraphLink,
        rng: &mut impl Rng,
    ) -> SuperGraphLink {
        let node = Self::new(builder, tensor_input, rng);
        let output = node.get_image_output();
        builder.add_node(node.to_any());
        output
    }

    pub fn get_image_output(&self) -> SuperGraphLink {
        self.image_output
            .expect("tensor-to-image output link should be configured")
    }
}

impl SuperGraphNode for SuperGraphNodeTensorToImage {
    fn to_any(self) -> SuperGraphAnyNode {
        SuperGraphAnyNode::TensorToImage(self)
    }

    fn eval<'short, 'model, 'p, P: Pool + 'p, T: SuperGraphObserver>(
        &'short self,
        _node_path: &[GlobalId],
        data: &mut SuperGraphData<'p, 'model, P>,
        _context: &mut SuperGraphContext<'short, 'model, 'p, P, T>,
    ) -> Result<(), SuperGraphError> {
        let tensor_input_link =
            require_node_link(self.tensor_input, "TensorToImage", "tensor_input")?;
        let image_output_link =
            require_node_link(self.image_output, "TensorToImage", "image_output")?;
        let tensor =
            data.tensors
                .remove(&tensor_input_link)
                .ok_or(SuperGraphError::MissingLinkError(format!(
                    ": missing tensor input link {:?}",
                    tensor_input_link
                )))?;
        data.images
            .insert(image_output_link, SuperGraphImage::new(tensor));
        Ok(())
    }

    fn op_kind(&self) -> String {
        "TensorToImage".to_string()
    }
    fn label(&self) -> Option<String> {
        self.label.clone()
    }

    fn inputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(self.tensor_input.map(|link| link.to_any()).into_iter())
    }

    fn outputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(self.image_output.map(|link| link.to_any()).into_iter())
    }

    fn global_id(&self) -> GlobalId {
        self.global_id
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeTensorToAudioClip {
    global_id: GlobalId,
    pub label: Option<String>,
    tensor_input: Option<SuperGraphLink>,
    audio_output: Option<SuperGraphLink>,
    sample_rate_hz: u32,
}

impl SuperGraphNodeTensorToAudioClip {
    pub fn new(
        _builder: &mut SuperGraphBuilder,
        tensor_input: SuperGraphLink,
        sample_rate_hz: u32,
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            global_id: GlobalId::new(rng),
            label: None,
            tensor_input: Some(tensor_input),
            audio_output: Some(SuperGraphLink::new(SuperGraphLinkKind::AudioClip, rng)),
            sample_rate_hz,
        }
    }

    pub fn new_and_add(
        builder: &mut SuperGraphBuilder,
        tensor_input: SuperGraphLink,
        sample_rate_hz: u32,
        rng: &mut impl Rng,
    ) -> SuperGraphLink {
        let node = Self::new(builder, tensor_input, sample_rate_hz, rng);
        let output = node.get_audio_output();
        builder.add_node(node.to_any());
        output
    }

    pub fn get_audio_output(&self) -> SuperGraphLink {
        self.audio_output
            .expect("tensor-to-audio output link should be configured")
    }
}

impl SuperGraphNode for SuperGraphNodeTensorToAudioClip {
    fn to_any(self) -> SuperGraphAnyNode {
        SuperGraphAnyNode::TensorToAudioClip(self)
    }

    fn eval<'short, 'model, 'p, P: Pool + 'p, T: SuperGraphObserver>(
        &'short self,
        _node_path: &[GlobalId],
        data: &mut SuperGraphData<'p, 'model, P>,
        _context: &mut SuperGraphContext<'short, 'model, 'p, P, T>,
    ) -> Result<(), SuperGraphError> {
        let tensor_input_link =
            require_node_link(self.tensor_input, "TensorToAudioClip", "tensor_input")?;
        let audio_output_link =
            require_node_link(self.audio_output, "TensorToAudioClip", "audio_output")?;
        let tensor =
            data.tensors
                .remove(&tensor_input_link)
                .ok_or(SuperGraphError::MissingLinkError(format!(
                    ": missing tensor input link {:?}",
                    tensor_input_link
                )))?;
        data.audio_clips.insert(
            audio_output_link,
            SuperGraphAudioClip::new(tensor, self.sample_rate_hz),
        );
        Ok(())
    }

    fn op_kind(&self) -> String {
        "TensorToAudioClip".to_string()
    }
    fn label(&self) -> Option<String> {
        self.label.clone()
    }

    fn inputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(self.tensor_input.map(|link| link.to_any()).into_iter())
    }

    fn outputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(self.audio_output.map(|link| link.to_any()).into_iter())
    }

    fn global_id(&self) -> GlobalId {
        self.global_id
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeTensorToVideoClip {
    global_id: GlobalId,
    pub label: Option<String>,
    tensor_input: Option<SuperGraphLink>,
    video_output: Option<SuperGraphLink>,
    fps: f32,
}

impl SuperGraphNodeTensorToVideoClip {
    pub fn new(
        _builder: &mut SuperGraphBuilder,
        tensor_input: SuperGraphLink,
        fps: f32,
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            global_id: GlobalId::new(rng),
            label: None,
            tensor_input: Some(tensor_input),
            video_output: Some(SuperGraphLink::new(SuperGraphLinkKind::VideoClip, rng)),
            fps,
        }
    }

    pub fn new_and_add(
        builder: &mut SuperGraphBuilder,
        tensor_input: SuperGraphLink,
        fps: f32,
        rng: &mut impl Rng,
    ) -> SuperGraphLink {
        let node = Self::new(builder, tensor_input, fps, rng);
        let output = node.get_video_output();
        builder.add_node(node.to_any());
        output
    }

    pub fn get_video_output(&self) -> SuperGraphLink {
        self.video_output
            .expect("tensor-to-video output link should be configured")
    }
}

impl SuperGraphNode for SuperGraphNodeTensorToVideoClip {
    fn to_any(self) -> SuperGraphAnyNode {
        SuperGraphAnyNode::TensorToVideoClip(self)
    }

    fn eval<'short, 'model, 'p, P: Pool + 'p, T: SuperGraphObserver>(
        &'short self,
        _node_path: &[GlobalId],
        data: &mut SuperGraphData<'p, 'model, P>,
        _context: &mut SuperGraphContext<'short, 'model, 'p, P, T>,
    ) -> Result<(), SuperGraphError> {
        let tensor_input_link =
            require_node_link(self.tensor_input, "TensorToVideoClip", "tensor_input")?;
        let video_output_link =
            require_node_link(self.video_output, "TensorToVideoClip", "video_output")?;
        let tensor =
            data.tensors
                .remove(&tensor_input_link)
                .ok_or(SuperGraphError::MissingLinkError(format!(
                    ": missing tensor input link {:?}",
                    tensor_input_link
                )))?;
        data.video_clips.insert(
            video_output_link,
            super::data::SuperGraphVideoClip::new(tensor, self.fps),
        );
        Ok(())
    }

    fn op_kind(&self) -> String {
        "TensorToVideoClip".to_string()
    }
    fn label(&self) -> Option<String> {
        self.label.clone()
    }

    fn inputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(self.tensor_input.map(|link| link.to_any()).into_iter())
    }

    fn outputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(self.video_output.map(|link| link.to_any()).into_iter())
    }

    fn global_id(&self) -> GlobalId {
        self.global_id
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeVideoClipToTensor {
    global_id: GlobalId,
    pub label: Option<String>,
    video_input: Option<SuperGraphLink>,
    tensor_output: Option<SuperGraphLink>,
}

impl SuperGraphNodeVideoClipToTensor {
    pub fn new(
        _builder: &mut SuperGraphBuilder,
        video_input: SuperGraphLink,
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            global_id: GlobalId::new(rng),
            label: None,
            video_input: Some(video_input),
            tensor_output: Some(SuperGraphLink::new(SuperGraphLinkKind::Tensor, rng)),
        }
    }

    pub fn new_and_add(
        builder: &mut SuperGraphBuilder,
        video_input: SuperGraphLink,
        rng: &mut impl Rng,
    ) -> SuperGraphLink {
        let node = Self::new(builder, video_input, rng);
        let output = node.get_tensor_output();
        builder.add_node(node.to_any());
        output
    }

    pub fn get_tensor_output(&self) -> SuperGraphLink {
        self.tensor_output
            .expect("video-to-tensor output link should be configured")
    }
}

impl SuperGraphNode for SuperGraphNodeVideoClipToTensor {
    fn to_any(self) -> SuperGraphAnyNode {
        SuperGraphAnyNode::VideoClipToTensor(self)
    }

    fn eval<'short, 'model, 'p, P: Pool + 'p, T: SuperGraphObserver>(
        &'short self,
        _node_path: &[GlobalId],
        data: &mut SuperGraphData<'p, 'model, P>,
        context: &mut SuperGraphContext<'short, 'model, 'p, P, T>,
    ) -> Result<(), SuperGraphError> {
        let video_input_link =
            require_node_link(self.video_input, "VideoClipToTensor", "video_input")?;
        let tensor_output_link =
            require_node_link(self.tensor_output, "VideoClipToTensor", "tensor_output")?;
        let clip =
            data.video_clips
                .get(&video_input_link)
                .ok_or(SuperGraphError::MissingLinkError(format!(
                    ": missing video clip input link {:?}",
                    video_input_link
                )))?;
        let tensor_copy = clip
            .frames
            .to_tensor(context.pool)
            .map_err(|e| SuperGraphError::InvalidInputError(format!("allocation: {e}")))?;
        data.tensors.insert(tensor_output_link, tensor_copy);
        Ok(())
    }

    fn op_kind(&self) -> String {
        "VideoClipToTensor".to_string()
    }
    fn label(&self) -> Option<String> {
        self.label.clone()
    }

    fn inputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(self.video_input.map(|link| link.to_any()).into_iter())
    }

    fn outputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(self.tensor_output.map(|link| link.to_any()).into_iter())
    }

    fn global_id(&self) -> GlobalId {
        self.global_id
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeAudioClipToTensor {
    global_id: GlobalId,
    pub label: Option<String>,
    audio_input: Option<SuperGraphLink>,
    tensor_output: Option<SuperGraphLink>,
    expected_sample_rate_hz: Option<u32>,
}

impl SuperGraphNodeAudioClipToTensor {
    pub fn new(
        _builder: &mut SuperGraphBuilder,
        audio_input: SuperGraphLink,
        expected_sample_rate_hz: Option<u32>,
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            global_id: GlobalId::new(rng),
            label: None,
            audio_input: Some(audio_input),
            tensor_output: Some(SuperGraphLink::new(SuperGraphLinkKind::Tensor, rng)),
            expected_sample_rate_hz,
        }
    }

    pub fn new_and_add(
        builder: &mut SuperGraphBuilder,
        audio_input: SuperGraphLink,
        expected_sample_rate_hz: Option<u32>,
        rng: &mut impl Rng,
    ) -> SuperGraphLink {
        let node = Self::new(builder, audio_input, expected_sample_rate_hz, rng);
        let output = node.get_tensor_output();
        builder.add_node(node.to_any());
        output
    }

    pub fn get_tensor_output(&self) -> SuperGraphLink {
        self.tensor_output
            .expect("audio-to-tensor output link should be configured")
    }
}

impl SuperGraphNode for SuperGraphNodeAudioClipToTensor {
    fn to_any(self) -> SuperGraphAnyNode {
        SuperGraphAnyNode::AudioClipToTensor(self)
    }

    fn eval<'short, 'model, 'p, P: Pool + 'p, T: SuperGraphObserver>(
        &'short self,
        _node_path: &[GlobalId],
        data: &mut SuperGraphData<'p, 'model, P>,
        context: &mut SuperGraphContext<'short, 'model, 'p, P, T>,
    ) -> Result<(), SuperGraphError> {
        let audio_input_link =
            require_node_link(self.audio_input, "AudioClipToTensor", "audio_input")?;
        let tensor_output_link =
            require_node_link(self.tensor_output, "AudioClipToTensor", "tensor_output")?;
        let clip =
            data.audio_clips
                .get(&audio_input_link)
                .ok_or(SuperGraphError::MissingLinkError(format!(
                    ": missing audio input link {:?}",
                    audio_input_link
                )))?;
        if let Some(expected) = self.expected_sample_rate_hz
            && clip.sample_rate_hz != expected
        {
            return Err(SuperGraphError::InvalidInputError(format!(
                "audio sample rate mismatch for {:?}: expected {}, got {}",
                audio_input_link, expected, clip.sample_rate_hz
            )));
        }
        let tensor_copy = clip
            .samples
            .to_tensor(context.pool)
            .map_err(|e| SuperGraphError::InvalidInputError(format!("allocation: {e}")))?;
        data.tensors.insert(tensor_output_link, tensor_copy);
        Ok(())
    }

    fn op_kind(&self) -> String {
        "AudioClipToTensor".to_string()
    }
    fn label(&self) -> Option<String> {
        self.label.clone()
    }

    fn inputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(self.audio_input.map(|link| link.to_any()).into_iter())
    }

    fn outputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(self.tensor_output.map(|link| link.to_any()).into_iter())
    }

    fn global_id(&self) -> GlobalId {
        self.global_id
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeAudioToMelConfig {
    pub expected_sample_rate_hz: Option<u32>,
    pub n_fft: u32,
    pub hop_length: u32,
    pub center_padding: u32,
    pub max_samples: Option<u32>,
    pub drop_last_frame: bool,
    pub num_mel_bins: u32,
    pub mel_filters: Vec<f32>,
    pub log_floor: f32,
    pub clamp_dynamic_range: Option<f32>,
    pub normalize_add: Option<f32>,
    pub normalize_div: Option<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeAudioClipToMelSpectrogram {
    global_id: GlobalId,
    pub label: Option<String>,
    audio_input: Option<SuperGraphLink>,
    tensor_output: Option<SuperGraphLink>,
    config: SuperGraphNodeAudioToMelConfig,
}

impl SuperGraphNodeAudioClipToMelSpectrogram {
    pub fn new(
        _builder: &mut SuperGraphBuilder,
        audio_input: SuperGraphLink,
        config: SuperGraphNodeAudioToMelConfig,
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            global_id: GlobalId::new(rng),
            label: None,
            audio_input: Some(audio_input),
            tensor_output: Some(SuperGraphLink::new(SuperGraphLinkKind::Tensor, rng)),
            config,
        }
    }

    pub fn new_and_add(
        builder: &mut SuperGraphBuilder,
        audio_input: SuperGraphLink,
        config: SuperGraphNodeAudioToMelConfig,
        rng: &mut impl Rng,
    ) -> SuperGraphLink {
        let node = Self::new(builder, audio_input, config, rng);
        let output = node.get_tensor_output();
        builder.add_node(node.to_any());
        output
    }

    pub fn get_tensor_output(&self) -> SuperGraphLink {
        self.tensor_output
            .expect("audio-to-mel output link should be configured")
    }
}

impl SuperGraphNode for SuperGraphNodeAudioClipToMelSpectrogram {
    fn to_any(self) -> SuperGraphAnyNode {
        SuperGraphAnyNode::AudioClipToMelSpectrogram(self)
    }

    fn eval<'short, 'model, 'p, P: Pool + 'p, T: SuperGraphObserver>(
        &'short self,
        _node_path: &[GlobalId],
        data: &mut SuperGraphData<'p, 'model, P>,
        context: &mut SuperGraphContext<'short, 'model, 'p, P, T>,
    ) -> Result<(), SuperGraphError> {
        let audio_input_link =
            require_node_link(self.audio_input, "AudioClipToMelSpectrogram", "audio_input")?;
        let tensor_output_link = require_node_link(
            self.tensor_output,
            "AudioClipToMelSpectrogram",
            "tensor_output",
        )?;
        let clip =
            data.audio_clips
                .get(&audio_input_link)
                .ok_or(SuperGraphError::MissingLinkError(format!(
                    ": missing audio input link {:?}",
                    audio_input_link
                )))?;
        if let Some(expected) = self.config.expected_sample_rate_hz
            && clip.sample_rate_hz != expected
        {
            return Err(SuperGraphError::InvalidInputError(format!(
                "audio sample rate mismatch for {:?}: expected {}, got {}",
                audio_input_link, expected, clip.sample_rate_hz
            )));
        }
        if self.config.n_fft == 0 {
            return Err(SuperGraphError::InvalidInputError(
                "audio->mel n_fft must be > 0".to_string(),
            ));
        }
        if self.config.hop_length == 0 {
            return Err(SuperGraphError::InvalidInputError(
                "audio->mel hop_length must be > 0".to_string(),
            ));
        }
        if self.config.num_mel_bins == 0 {
            return Err(SuperGraphError::InvalidInputError(
                "audio->mel num_mel_bins must be > 0".to_string(),
            ));
        }
        if self.config.log_floor <= 0.0 {
            return Err(SuperGraphError::InvalidInputError(
                "audio->mel log_floor must be > 0".to_string(),
            ));
        }
        if let Some(dynamic) = self.config.clamp_dynamic_range
            && dynamic <= 0.0
        {
            return Err(SuperGraphError::InvalidInputError(
                "audio->mel clamp_dynamic_range must be > 0 when set".to_string(),
            ));
        }
        if (self.config.normalize_add.is_some() && self.config.normalize_div.is_none())
            || (self.config.normalize_add.is_none() && self.config.normalize_div.is_some())
        {
            return Err(SuperGraphError::InvalidInputError(
                "audio->mel normalize_add and normalize_div must be set together".to_string(),
            ));
        }
        if let Some(div) = self.config.normalize_div
            && div == 0.0
        {
            return Err(SuperGraphError::InvalidInputError(
                "audio->mel normalize_div must be non-zero".to_string(),
            ));
        }

        let n_fft = self.config.n_fft as usize;
        let hop_length = self.config.hop_length as usize;
        let center_padding = self.config.center_padding as usize;
        let num_mel_bins = self.config.num_mel_bins as usize;
        let n_freqs = n_fft / 2 + 1;
        if self.config.mel_filters.len() != num_mel_bins * n_freqs {
            return Err(SuperGraphError::InvalidInputError(format!(
                "audio->mel mel_filters length mismatch: expected {}, got {}",
                num_mel_bins * n_freqs,
                self.config.mel_filters.len()
            )));
        }

        let mut samples: Vec<f32> = (0..clip.samples.numel())
            .map(|i| clip.samples.read_element(i).to_f64() as f32)
            .collect();

        if let Some(max_samples) = self.config.max_samples {
            let max_samples = max_samples as usize;
            if samples.len() > max_samples {
                samples.truncate(max_samples);
            } else if samples.len() < max_samples {
                samples.resize(max_samples, 0.0);
            }
        }
        if samples.is_empty() {
            return Err(SuperGraphError::InvalidInputError(
                "audio->mel input has no samples".to_string(),
            ));
        }

        let mut padded = vec![0.0f32; center_padding + samples.len() + center_padding];
        padded[center_padding..center_padding + samples.len()].copy_from_slice(&samples);

        if padded.len() < n_fft {
            return Err(SuperGraphError::InvalidInputError(format!(
                "audio->mel padded input too short: len={} n_fft={}",
                padded.len(),
                n_fft
            )));
        }

        let stft_frames = (padded.len() - n_fft) / hop_length + 1;
        let num_frames = if self.config.drop_last_frame {
            stft_frames.saturating_sub(1)
        } else {
            stft_frames
        };
        if num_frames == 0 {
            return Err(SuperGraphError::InvalidInputError(
                "audio->mel produced zero frames".to_string(),
            ));
        }

        let window: Vec<f32> = (0..n_fft)
            .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / n_fft as f32).cos()))
            .collect();

        let mut cos_table = vec![0.0f32; n_freqs * n_fft];
        let mut sin_table = vec![0.0f32; n_freqs * n_fft];
        for k in 0..n_freqs {
            for n in 0..n_fft {
                let angle = -2.0 * std::f32::consts::PI * k as f32 * n as f32 / n_fft as f32;
                let idx = k * n_fft + n;
                cos_table[idx] = angle.cos();
                sin_table[idx] = angle.sin();
            }
        }

        let mut frame_buf = vec![0.0f32; n_fft];
        let mut magnitudes = vec![0.0f32; n_freqs * num_frames];
        for frame in 0..num_frames {
            let start = frame * hop_length;
            for i in 0..n_fft {
                frame_buf[i] = padded[start + i] * window[i];
            }
            for k in 0..n_freqs {
                let mut re = 0.0f32;
                let mut im = 0.0f32;
                let trig_base = k * n_fft;
                for n in 0..n_fft {
                    let x = frame_buf[n];
                    re += x * cos_table[trig_base + n];
                    im += x * sin_table[trig_base + n];
                }
                magnitudes[k * num_frames + frame] = re * re + im * im;
            }
        }

        let mut mel_spec = vec![0.0f32; num_mel_bins * num_frames];
        for mel_idx in 0..num_mel_bins {
            let filter_row = &self.config.mel_filters[mel_idx * n_freqs..(mel_idx + 1) * n_freqs];
            for frame in 0..num_frames {
                let mut sum = 0.0f32;
                for freq in 0..n_freqs {
                    sum += filter_row[freq] * magnitudes[freq * num_frames + frame];
                }
                mel_spec[mel_idx * num_frames + frame] = sum;
            }
        }

        let mut output = mel_spec
            .into_iter()
            .map(|x| x.max(self.config.log_floor).log10())
            .collect::<Vec<_>>();

        if let Some(dynamic) = self.config.clamp_dynamic_range {
            let max_val = output.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let min_val = max_val - dynamic;
            for x in &mut output {
                if *x < min_val {
                    *x = min_val;
                }
            }
        }

        if let (Some(add), Some(div)) = (self.config.normalize_add, self.config.normalize_div) {
            for x in &mut output {
                *x = (*x + add) / div;
            }
        }

        data.tensors.insert(
            tensor_output_link,
            PoolNumericTensor::from_fn(
                vec![1, num_mel_bins as u64, num_frames as u64],
                NumericDType::F32,
                context.pool,
                |i| NumericScalar::from_f32(output[i]),
            )
            .map_err(alloc_err)?,
        );
        Ok(())
    }

    fn op_kind(&self) -> String {
        "AudioClipToMelSpectrogram".to_string()
    }
    fn label(&self) -> Option<String> {
        self.label.clone()
    }

    fn inputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(self.audio_input.map(|link| link.to_any()).into_iter())
    }

    fn outputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(self.tensor_output.map(|link| link.to_any()).into_iter())
    }

    fn global_id(&self) -> GlobalId {
        self.global_id
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeMilliOpGraph {
    global_id: GlobalId,
    pub label: Option<String>,
    pub graph: MilliOpGraph,
}

impl SuperGraphNodeMilliOpGraph {
    pub fn new(graph: MilliOpGraph, rng: &mut impl Rng) -> Self {
        Self {
            graph,
            global_id: GlobalId::new(rng),
            label: None,
        }
    }
}

#[allow(dead_code)]
struct MilliOpGraphObserverWrapper<'a, T: SuperGraphObserver> {
    inner: &'a mut T,
    node_path: Vec<GlobalId>,
}

#[allow(dead_code)]
impl<'a, T: SuperGraphObserver> MilliOpGraphObserverWrapper<'a, T> {
    fn new(inner: &'a mut T, node_path: &[GlobalId]) -> Self {
        Self {
            inner,
            node_path: node_path.to_vec(),
        }
    }
}

impl<'a, T: SuperGraphObserver> MilliOpGraphObserver for MilliOpGraphObserverWrapper<'a, T> {
    fn on_tensor_assigned(
        &mut self,
        tensor_path: &[GlobalId],
        tensor: &crate::numeric_tensor::NumericTensorView<'_, crate::tensor_rank::DynRank>,
    ) {
        let tensor_path = self
            .node_path
            .iter()
            .chain(tensor_path.iter())
            .copied()
            .collect::<Vec<_>>();
        self.inner
            .on_tensor_assigned(tensor_path.as_slice(), tensor);
    }

    fn on_node_executed(
        &mut self,
        node_path: &[GlobalId],
        start_instant: Instant,
        end_instant: Instant,
    ) {
        let node_path = self
            .node_path
            .iter()
            .chain(node_path.iter())
            .copied()
            .collect::<Vec<_>>();
        self.inner
            .on_node_executed(node_path.as_slice(), "", start_instant, end_instant);
    }

    fn should_cancel(&mut self) -> bool {
        self.inner.should_cancel()
    }
}

impl SuperGraphNode for SuperGraphNodeMilliOpGraph {
    fn to_any(self) -> SuperGraphAnyNode {
        SuperGraphAnyNode::MilliOpGraph(self)
    }
    fn eval<'short, 'model, 'p, P: Pool + 'p, T: SuperGraphObserver>(
        &'short self,
        _node_path: &[GlobalId],
        data: &mut SuperGraphData<'p, 'model, P>,
        context: &mut SuperGraphContext<'short, 'model, 'p, P, T>,
    ) -> Result<(), SuperGraphError> {
        // Build input views from SuperGraphData.
        let input_views: Vec<_> = self
            .graph
            .get_inputs()
            .into_iter()
            .filter_map(|id| {
                let tensor = data.tensors.get(&SuperGraphLink::tensor(id))?;
                Some((id, tensor.view()))
            })
            .collect();
        let input_map: HashMap<
            GlobalId,
            &crate::numeric_tensor::NumericTensorView<'_, crate::tensor_rank::DynRank>,
        > = input_views.iter().map(|(id, view)| (*id, view)).collect();

        let results = self.graph.pool_eval(&input_map, context.pool)?;

        for (id, tensor) in results {
            data.tensors.insert(SuperGraphLink::tensor(id), tensor);
        }
        Ok(())
    }
    fn op_kind(&self) -> String {
        "MilliOpGraph".to_string()
    }
    fn label(&self) -> Option<String> {
        self.label.clone()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(
            self.graph
                .input_link_ids()
                .map(|(a, _b)| SuperGraphLink::tensor(a).to_any()),
        )
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(
            self.graph
                .output_link_ids()
                .map(|(a, _b)| SuperGraphLink::tensor(a).to_any()),
        )
    }
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeScan {
    global_id: GlobalId,
    pub label: Option<String>,
    inner_graph: SuperGraph,
    iteration_count: Option<SuperGraphLink>,
    simple_inputs: Vec<(Option<SuperGraphLink>, Option<SuperGraphLink>)>,
    state_links: Vec<(
        Option<SuperGraphLink>,
        Option<SuperGraphLink>,
        Option<SuperGraphLink>,
    )>,
    scan_inputs: Vec<(Option<SuperGraphLink>, Option<SuperGraphLink>, u32)>,
    scan_outputs: Vec<(Option<SuperGraphLink>, Option<SuperGraphLink>, u32)>,
    simple_outputs: Vec<(Option<SuperGraphLink>, Option<SuperGraphLink>)>,
}

impl SuperGraphNodeScan {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        inner_graph: SuperGraph,
        iteration_count: SuperGraphLink,
        simple_inputs: Vec<SuperGraphLinkDouble>,
        state_links: Vec<SuperGraphLinkTriple>,
        scan_inputs: Vec<(SuperGraphLink, SuperGraphLink, u32)>,
        scan_outputs: Vec<(SuperGraphLink, SuperGraphLink, u32)>,
        simple_outputs: Vec<SuperGraphLinkDouble>,
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            inner_graph,
            iteration_count: Some(iteration_count),
            simple_inputs: simple_inputs
                .into_iter()
                .map(|link| (Some(link.first()), Some(link.second())))
                .collect(),
            state_links: state_links
                .into_iter()
                .map(|link| (Some(link.first()), Some(link.second()), Some(link.third())))
                .collect(),
            scan_inputs: scan_inputs
                .into_iter()
                .map(|(a, b, c)| (Some(a), Some(b), c))
                .collect(),
            scan_outputs: scan_outputs
                .into_iter()
                .map(|(a, b, c)| (Some(a), Some(b), c))
                .collect(),
            simple_outputs: simple_outputs
                .into_iter()
                .map(|link| (Some(link.first()), Some(link.second())))
                .collect(),
            global_id: GlobalId::new(rng),
            label: None,
        }
    }
}

fn eval_scan<'short, 'model, 'p, P: Pool + 'p, T: SuperGraphObserver>(
    scan: &'short SuperGraphNodeScan,
    node_path: &[GlobalId],
    data: &mut SuperGraphData<'p, 'model, P>,
    context: &mut SuperGraphContext<'short, 'model, 'p, P, T>,
) -> Result<(), SuperGraphError> {
    let iteration_count_link = require_node_link(scan.iteration_count, "Scan", "iteration_count")?;
    let iteration_count_tensor =
        data.tensors
            .get(&iteration_count_link)
            .ok_or(SuperGraphError::MissingLinkError(format!(
                ": scan iteration_count {:?}",
                iteration_count_link
            )))?;
    let iteration_count: i64 = iteration_count_tensor.read_element(0).to_i64();
    if iteration_count < 0 {
        return Err(SuperGraphError::InvalidInputError(format!(
            "scan iteration_count must be non-negative, got {iteration_count}"
        )));
    }
    let iteration_count = iteration_count as u64;

    let node_path = node_path
        .iter()
        .chain(core::iter::once(&scan.global_id))
        .copied()
        .collect::<Vec<_>>();

    let simple_inputs = {
        let mut simple_inputs = SuperGraphData::new();
        for (outer, inner) in &scan.simple_inputs {
            let outer = require_node_link(*outer, "Scan", "simple_inputs.outer")?;
            let inner = require_node_link(*inner, "Scan", "simple_inputs.inner")?;
            simple_inputs.copy_link_from(data, outer, inner, context.pool)?;
        }
        simple_inputs
    };

    let mut state_values = {
        let mut state_values = SuperGraphData::new();
        for (outer, inner, _) in &scan.state_links {
            let outer = require_node_link(*outer, "Scan", "state_links.outer")?;
            let inner = require_node_link(*inner, "Scan", "state_links.inner")?;
            state_values.copy_link_from(data, outer, inner, context.pool)?;
        }
        state_values
    };

    let mut prev_iter_outputs: Option<SuperGraphData<'p, 'model, P>> = None;

    let mut output_scan_tensor_parts = HashMap::new();
    for (_inner, outer, scan_axis) in &scan.scan_outputs {
        let outer = require_node_link(*outer, "Scan", "scan_outputs.outer")?;
        output_scan_tensor_parts.insert(outer, (Vec::new(), *scan_axis as usize));
    }

    for i in 0..iteration_count {
        if context.observer.should_cancel() {
            return Err(SuperGraphError::Cancelled);
        }
        let iter_inputs = {
            let mut iter_inputs = SuperGraphData::new();
            // Copy simple inputs each iteration
            for (_outer, inner) in &scan.simple_inputs {
                let inner = require_node_link(*inner, "Scan", "simple_inputs.inner")?;
                iter_inputs.copy_link_from(&simple_inputs, inner, inner, context.pool)?;
            }
            // Copy state values
            for (_, inner, _) in &scan.state_links {
                let inner = require_node_link(*inner, "Scan", "state_links.inner")?;
                iter_inputs.copy_link_from(&state_values, inner, inner, context.pool)?;
            }
            for (outer, inner, scan_axis) in &scan.scan_inputs {
                let outer = require_node_link(*outer, "Scan", "scan_inputs.outer")?;
                let inner = require_node_link(*inner, "Scan", "scan_inputs.inner")?;
                let tensor = data
                    .tensors
                    .get(&outer)
                    .ok_or(SuperGraphError::MissingLinkError(format!(
                        ": scan_input outer={:?} inner={:?}",
                        outer, inner
                    )))?;
                let shape = tensor.shape();
                let axis = *scan_axis as usize;
                let mut slice_ranges: Vec<(u64, u64)> = Vec::new();
                for (j, &d) in shape.iter().enumerate() {
                    if j == axis {
                        slice_ranges.push((i, i + 1));
                    } else {
                        slice_ranges.push((0, d));
                    }
                }
                let sliced = tensor
                    .slice(&slice_ranges)
                    .map_err(|e| SuperGraphError::InvalidInputError(format!("scan slice: {e}")))?;
                // Squeeze the scan axis (size 1 after slicing) — elements unchanged
                let squeezed_shape: Vec<u64> = shape
                    .iter()
                    .enumerate()
                    .filter_map(|(j, &d)| if j == axis { None } else { Some(d) })
                    .collect();
                let squeezed = PoolNumericTensor::from_fn(
                    squeezed_shape,
                    sliced.dtype(),
                    context.pool,
                    |idx| sliced.read_element(idx),
                )
                .map_err(alloc_err)?;
                iter_inputs.tensors.insert(inner, squeezed);
            }
            iter_inputs
        };
        let iter_outputs = scan
            .inner_graph
            .eval(node_path.as_slice(), iter_inputs, context)?;

        for (inner, outer, _scan_axis) in &scan.scan_outputs {
            let inner = require_node_link(*inner, "Scan", "scan_outputs.inner")?;
            let outer = require_node_link(*outer, "Scan", "scan_outputs.outer")?;
            let tensor = iter_outputs
                .tensors
                .get(&inner)
                .ok_or(SuperGraphError::MissingLinkError(String::new()))?;
            let (tensors, _) = output_scan_tensor_parts
                .get_mut(&outer)
                .ok_or(SuperGraphError::MissingLinkError(String::new()))?;
            let tensor_copy = tensor
                .to_tensor(context.pool)
                .map_err(|e| SuperGraphError::InvalidInputError(format!("allocation: {e}")))?;
            tensors.push(tensor_copy);
        }

        state_values = {
            let mut state_values = SuperGraphData::new();
            for (_, inner, iter_output) in &scan.state_links {
                let iter_output =
                    require_node_link(*iter_output, "Scan", "state_links.iter_output")?;
                let inner = require_node_link(*inner, "Scan", "state_links.inner")?;
                state_values.copy_link_from(&iter_outputs, iter_output, inner, context.pool)?;
            }
            state_values
        };

        prev_iter_outputs = Some(iter_outputs);
    }

    let mut output_data = SuperGraphData::new();
    let prev_iter_outputs = if iteration_count == 0 {
        None
    } else {
        Some(
            prev_iter_outputs
                .as_ref()
                .ok_or(SuperGraphError::InvalidInputError(
                    "scan had iterations but no outputs were produced".to_string(),
                ))?,
        )
    };

    for (inner, outer) in &scan.simple_outputs {
        let inner = require_node_link(*inner, "Scan", "simple_outputs.inner")?;
        let outer = require_node_link(*outer, "Scan", "simple_outputs.outer")?;
        output_data.copy_link_from(
            prev_iter_outputs.ok_or(SuperGraphError::InvalidInputError(
                "scan simple_outputs are unavailable when iteration_count is 0".to_string(),
            ))?,
            inner,
            outer,
            context.pool,
        )?;
    }

    for (link, (parts, axis)) in output_scan_tensor_parts {
        if parts.is_empty() {
            return Err(SuperGraphError::InvalidInputError(format!(
                "scan output {:?} has no parts; iteration_count is likely 0",
                link
            )));
        }
        // Each part has shape [d0, ..., d{rank-1}]. Concatenating
        // unsqueezes at `axis` (size 1 in each part) and stacks N parts,
        // producing output [d0, ..., d{axis-1}, N, d{axis}, ..., d{rank-1}].
        // The flat index math is general for any `axis`: split the
        // output flat index into the contributions from axes < axis,
        // axis itself (= part index), and axes > axis.
        let part_shape = parts[0].shape();
        let n_parts = parts.len() as u64;
        let mut concat_shape: Vec<u64> = Vec::with_capacity(part_shape.len() + 1);
        for (j, &d) in part_shape.iter().enumerate() {
            if j == axis {
                concat_shape.push(n_parts);
            }
            concat_shape.push(d);
        }
        if axis >= part_shape.len() {
            concat_shape.push(n_parts);
        }
        // Row-major: post_axis_size is the product of all dims after
        // the inserted N axis in the output (== product of part dims at
        // positions >= axis).
        let post_axis_size: u64 = part_shape.iter().skip(axis).product();
        let concat =
            PoolNumericTensor::from_fn(concat_shape, parts[0].dtype(), context.pool, |flat_idx| {
                let f = flat_idx as u64;
                let post_idx = f % post_axis_size;
                let part_idx = (f / post_axis_size) % n_parts;
                let pre_idx = f / (post_axis_size * n_parts);
                let per_part_flat = pre_idx * post_axis_size + post_idx;
                parts[part_idx as usize].read_element(per_part_flat as usize)
            })
            .map_err(alloc_err)?;
        output_data.tensors.insert(link, concat);
    }

    data.extend_from(output_data);
    Ok(())
}

impl SuperGraphNode for SuperGraphNodeScan {
    fn to_any(self) -> SuperGraphAnyNode {
        SuperGraphAnyNode::Scan(self)
    }

    fn eval<'short, 'model, 'p, P: Pool + 'p, T: SuperGraphObserver>(
        &'short self,
        node_path: &[GlobalId],
        data: &mut SuperGraphData<'p, 'model, P>,
        context: &mut SuperGraphContext<'short, 'model, 'p, P, T>,
    ) -> Result<(), SuperGraphError> {
        eval_scan(self, node_path, data, context)
    }
    fn op_kind(&self) -> String {
        "Scan".to_string()
    }
    fn label(&self) -> Option<String> {
        self.label.clone()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        let mut inputs = Vec::new();
        if let Some(iteration_count) = self.iteration_count {
            inputs.push(iteration_count.to_any());
        }
        for (outer, _) in &self.simple_inputs {
            if let Some(outer) = outer {
                inputs.push(outer.to_any());
            }
        }
        for (outer, _, _) in &self.state_links {
            if let Some(outer) = outer {
                inputs.push(outer.to_any());
            }
        }
        for (input, _, _) in &self.scan_inputs {
            if let Some(input) = input {
                inputs.push(input.to_any());
            }
        }
        Box::new(inputs.into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        let mut outputs = Vec::new();
        for (_, outer) in &self.simple_outputs {
            if let Some(outer) = outer {
                outputs.push(outer.to_any());
            }
        }
        for (_, output, _) in &self.scan_outputs {
            if let Some(output) = output {
                outputs.push(output.to_any());
            }
        }
        Box::new(outputs.into_iter())
    }
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeReportProgress {
    global_id: GlobalId,
    pub label: Option<String>,
    tier_input: Option<SuperGraphLink>,
    numerator_input: Option<SuperGraphLink>,
    denominator_input: Option<SuperGraphLink>,
}

impl SuperGraphNodeReportProgress {
    pub fn new(
        tier_input: SuperGraphLink,
        numerator_input: SuperGraphLink,
        denominator_input: SuperGraphLink,
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            global_id: GlobalId::new(rng),
            label: None,
            tier_input: Some(tier_input),
            numerator_input: Some(numerator_input),
            denominator_input: Some(denominator_input),
        }
    }
}

impl SuperGraphNode for SuperGraphNodeReportProgress {
    fn to_any(self) -> SuperGraphAnyNode {
        SuperGraphAnyNode::ReportProgress(self)
    }

    fn eval<'short, 'model, 'p, P: Pool + 'p, T: SuperGraphObserver>(
        &'short self,
        node_path: &[GlobalId],
        data: &mut SuperGraphData<'p, 'model, P>,
        context: &mut SuperGraphContext<'short, 'model, 'p, P, T>,
    ) -> Result<(), SuperGraphError> {
        let tier_input = require_node_link(self.tier_input, "ReportProgress", "tier_input")?;
        let numerator_input =
            require_node_link(self.numerator_input, "ReportProgress", "numerator_input")?;
        let denominator_input = require_node_link(
            self.denominator_input,
            "ReportProgress",
            "denominator_input",
        )?;
        let tier = read_rank0_i64_tensor(
            data.tensors
                .get(&tier_input)
                .ok_or(SuperGraphError::MissingLinkError(format!(
                    ": report_progress tier_input {:?}",
                    tier_input
                )))?,
            "ReportProgress.tier_input",
        )?;
        let numerator = read_rank0_f64_tensor(
            data.tensors
                .get(&numerator_input)
                .ok_or(SuperGraphError::MissingLinkError(format!(
                    ": report_progress numerator_input {:?}",
                    numerator_input
                )))?,
            "ReportProgress.numerator_input",
        )?;
        let denominator = read_rank0_f64_tensor(
            data.tensors
                .get(&denominator_input)
                .ok_or(SuperGraphError::MissingLinkError(format!(
                    ": report_progress denominator_input {:?}",
                    denominator_input
                )))?,
            "ReportProgress.denominator_input",
        )?;
        let path = node_path
            .iter()
            .chain(core::iter::once(&self.global_id))
            .copied()
            .collect::<Vec<_>>();
        context
            .observer
            .on_progress(path.as_slice(), tier, numerator, denominator);
        Ok(())
    }

    fn op_kind(&self) -> String {
        "ReportProgress".to_string()
    }
    fn label(&self) -> Option<String> {
        self.label.clone()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(
            self.tier_input
                .map(|x| x.to_any())
                .into_iter()
                .chain(self.numerator_input.map(|x| x.to_any()))
                .chain(self.denominator_input.map(|x| x.to_any())),
        )
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(std::iter::empty())
    }
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeRNNCacheRead {
    global_id: GlobalId,
    pub label: Option<String>,
    key_input: Option<SuperGraphLink>,
    tokens_input: Option<SuperGraphLink>,
    tokens_output: Option<SuperGraphLink>,
    state_outputs: Vec<(String, Option<SuperGraphLink>)>,
    default_state_inputs: Vec<(String, Option<SuperGraphLink>)>,
    /// Axis of `tokens_input` that indexes the batch (one cache key per row).
    #[serde(default)]
    pub tokens_batch_axis: u32,
    /// Axis of `tokens_input` that indexes the sequence (the axis we trim
    /// when emitting `tokens_output` after a cache hit).
    #[serde(default = "default_seq_axis")]
    pub tokens_seq_axis: u32,
    /// Axis of each state tensor that indexes the batch. Per-row states
    /// stored in the cache have size 1 along this axis; on hit they are
    /// stacked along this axis to form the batched state.
    #[serde(default)]
    pub state_batch_axis: u32,
}

fn default_seq_axis() -> u32 {
    1
}

impl SuperGraphNodeRNNCacheRead {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        key_input: SuperGraphLink,
        tokens_input: SuperGraphLink,
        tokens_output: SuperGraphLink,
        state_outputs: Vec<(String, SuperGraphLink)>,
        default_state_inputs: Vec<(String, SuperGraphLink)>,
        tokens_batch_axis: u32,
        tokens_seq_axis: u32,
        state_batch_axis: u32,
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            global_id: GlobalId::new(rng),
            label: None,
            key_input: Some(key_input),
            tokens_input: Some(tokens_input),
            tokens_output: Some(tokens_output),
            state_outputs: state_outputs
                .into_iter()
                .map(|(name, link)| (name, Some(link)))
                .collect(),
            default_state_inputs: default_state_inputs
                .into_iter()
                .map(|(name, link)| (name, Some(link)))
                .collect(),
            tokens_batch_axis,
            tokens_seq_axis,
            state_batch_axis,
        }
    }
}

impl SuperGraphNode for SuperGraphNodeRNNCacheRead {
    fn to_any(self) -> SuperGraphAnyNode {
        SuperGraphAnyNode::RNNCacheRead(self)
    }

    fn eval<'short, 'model, 'p, P: Pool + 'p, T: SuperGraphObserver>(
        &'short self,
        _node_path: &[GlobalId],
        data: &mut SuperGraphData<'p, 'model, P>,
        context: &mut SuperGraphContext<'short, 'model, 'p, P, T>,
    ) -> Result<(), SuperGraphError> {
        let key_input_link = require_node_link(self.key_input, "RNNCacheRead", "key_input")?;
        let tokens_input_link =
            require_node_link(self.tokens_input, "RNNCacheRead", "tokens_input")?;
        let tokens_output_link =
            require_node_link(self.tokens_output, "RNNCacheRead", "tokens_output")?;
        let tokens_input = data
            .tensors
            .get(&tokens_input_link)
            .ok_or(SuperGraphError::MissingLinkError(String::new()))?;
        let tokens_shape = tokens_input.shape();
        let batch_axis = self.tokens_batch_axis as usize;
        let seq_axis = self.tokens_seq_axis as usize;
        if batch_axis >= tokens_shape.len() || seq_axis >= tokens_shape.len() {
            return Err(SuperGraphError::InvalidInputError(format!(
                "RNNCacheRead axes batch={batch_axis} seq={seq_axis} out of range \
                 for tokens shape {:?}",
                tokens_shape
            )));
        }
        if batch_axis == seq_axis {
            return Err(SuperGraphError::InvalidInputError(
                "RNNCacheRead tokens_batch_axis must differ from tokens_seq_axis".to_string(),
            ));
        }
        let batch = tokens_shape[batch_axis] as usize;
        let seq = tokens_shape[seq_axis] as usize;

        // Per-row token extraction: slice tokens_input at batch_axis to a single row
        // and read the elements as a flat sequence. For [batch, seq] inputs this
        // yields exactly seq tokens per row.
        let mut per_row_tokens: Vec<Vec<u32>> = Vec::with_capacity(batch);
        for r in 0..batch {
            let mut slice_ranges: Vec<(u64, u64)> = tokens_shape.iter().map(|&d| (0, d)).collect();
            slice_ranges[batch_axis] = (r as u64, r as u64 + 1);
            let row = tokens_input.slice(&slice_ranges).map_err(|e| {
                SuperGraphError::InvalidInputError(format!("RNNCacheRead row slice: {e}"))
            })?;
            let toks: Vec<u32> = (0..row.numel())
                .map(|i| row.read_element(i).to_i64() as u32)
                .collect();
            per_row_tokens.push(toks);
        }

        // Per-row longest-matched-prefix lookup, then min-L alignment so every
        // row resumes the scan from the same position.
        let mut chosen_l: usize = 0;
        let mut per_row_state: Option<Vec<HashMap<String, PoolNumericTensor<'p, DynRank, P>>>> =
            None;
        if let Some(caches) = &mut context.caches {
            let key_input = *data
                .hashes
                .get(&key_input_link)
                .ok_or(SuperGraphError::MissingLinkError(String::new()))?;
            if let Some(rnn_cache) = caches.rnn_cache.get(&key_input) {
                let mut matched_lens: Vec<usize> = vec![0; batch];
                for (r, toks) in per_row_tokens.iter().enumerate() {
                    for i in (1..=toks.len()).rev() {
                        if rnn_cache.contains_key(&toks[..i].to_vec()) {
                            matched_lens[r] = i;
                            break;
                        }
                    }
                }
                let l_candidate = matched_lens.iter().copied().min().unwrap_or(0);
                if l_candidate > 0 {
                    // Confirm every row also has an entry at *exactly* L
                    // tokens. Same-prompt batch hits this trivially; mixed
                    // prompts may not, in which case we skip the cache.
                    let mut row_states: Vec<HashMap<String, PoolNumericTensor<'p, DynRank, P>>> =
                        Vec::with_capacity(batch);
                    let mut all_have_l = true;
                    for toks in per_row_tokens.iter() {
                        let key = toks[..l_candidate].to_vec();
                        match rnn_cache.get(&key) {
                            Some(state) => {
                                let materialized: HashMap<
                                    String,
                                    PoolNumericTensor<'p, DynRank, P>,
                                > = state
                                    .iter()
                                    .map(|(k, v)| {
                                        v.to_tensor(context.pool).map(|t| (k.clone(), t)).map_err(
                                            |e| {
                                                SuperGraphError::InvalidInputError(format!(
                                                    "allocation: {e}"
                                                ))
                                            },
                                        )
                                    })
                                    .collect::<Result<_, _>>()?;
                                row_states.push(materialized);
                            }
                            None => {
                                all_have_l = false;
                                break;
                            }
                        }
                    }
                    if all_have_l {
                        chosen_l = l_candidate;
                        per_row_state = Some(row_states);
                    }
                }
            }
        }

        if chosen_l > 0 {
            let row_states = per_row_state.expect("per_row_state set when chosen_l > 0");
            let state_batch_axis = self.state_batch_axis as usize;

            // Materialize the trimmed tokens first while we still hold
            // the read borrow on `data.tensors`; after this the borrow
            // ends and we can mutate data freely.
            let trimmed_tensor = {
                let mut trim_ranges: Vec<(u64, u64)> =
                    tokens_shape.iter().map(|&d| (0, d)).collect();
                trim_ranges[seq_axis] = (chosen_l as u64, seq as u64);
                let trimmed = tokens_input.slice(&trim_ranges).map_err(|e| {
                    SuperGraphError::InvalidInputError(format!("RNNCacheRead seq trim: {e}"))
                })?;
                let mut trimmed_shape = tokens_shape.clone();
                trimmed_shape[seq_axis] = (seq - chosen_l) as u64;
                PoolNumericTensor::from_fn(trimmed_shape, trimmed.dtype(), context.pool, |i| {
                    trimmed.read_element(i)
                })
                .map_err(alloc_err)?
            };

            for (state_name, output_link) in self.state_outputs.iter() {
                let output_link = require_node_link(*output_link, "RNNCacheRead", "state_outputs")?;
                let per_row: Vec<&PoolNumericTensor<'p, DynRank, P>> = row_states
                    .iter()
                    .map(|s| {
                        s.get(state_name).ok_or_else(|| {
                            SuperGraphError::InvalidInputError(format!(
                                "RNNCacheRead: cached state missing entry for {state_name}"
                            ))
                        })
                    })
                    .collect::<Result<_, _>>()?;
                let stacked = stack_along_axis(&per_row, state_batch_axis, context.pool)?;
                data.tensors.insert(output_link, stacked);
            }

            data.tensors.insert(tokens_output_link, trimmed_tensor);
        } else {
            // Cache miss — emit defaults and pass tokens through unchanged.
            for (key, value) in self.default_state_inputs.iter() {
                let value_link = require_node_link(*value, "RNNCacheRead", "default_state_inputs")?;
                let value = data
                    .tensors
                    .get(&value_link)
                    .ok_or(SuperGraphError::MissingLinkError(String::new()))?;
                if let Some((_, output_link)) = self.state_outputs.iter().find(|x| x.0 == *key) {
                    let output_link =
                        require_node_link(*output_link, "RNNCacheRead", "state_outputs")?;
                    let value_copy = value.to_tensor(context.pool).map_err(|e| {
                        SuperGraphError::InvalidInputError(format!("allocation: {e}"))
                    })?;
                    data.tensors.insert(output_link, value_copy);
                }
            }
            let tokens_copy = data
                .tensors
                .get(&tokens_input_link)
                .ok_or(SuperGraphError::MissingLinkError(String::new()))?
                .to_tensor(context.pool)
                .map_err(|e| SuperGraphError::InvalidInputError(format!("allocation: {e}")))?;
            data.tensors.insert(tokens_output_link, tokens_copy);
        }
        Ok(())
    }

    fn op_kind(&self) -> String {
        "RNNCacheRead".to_string()
    }
    fn label(&self) -> Option<String> {
        self.label.clone()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(
            self.key_input
                .map(|x| x.to_any())
                .into_iter()
                .chain(self.tokens_input.map(|x| x.to_any()))
                .chain(
                    self.default_state_inputs
                        .iter()
                        .filter_map(|x| x.1.map(|link| link.to_any())),
                ),
        )
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(
            self.tokens_output.map(|x| x.to_any()).into_iter().chain(
                self.state_outputs
                    .iter()
                    .filter_map(|x| x.1.map(|link| link.to_any())),
            ),
        )
    }
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeRNNCacheWrite {
    global_id: GlobalId,
    pub label: Option<String>,
    key_input: Option<SuperGraphLink>,
    tokens_input: Option<SuperGraphLink>,
    state_inputs: Vec<(String, Option<SuperGraphLink>)>,
    /// Axis of `tokens_input` that indexes the batch (used to derive
    /// per-row token sequences as cache keys).
    #[serde(default)]
    pub tokens_batch_axis: u32,
    /// Axis of `tokens_input` that indexes the sequence (used to read
    /// each row's token sequence in order).
    #[serde(default = "default_seq_axis")]
    pub tokens_seq_axis: u32,
    /// Axis of each state tensor that indexes the batch. Per-row state
    /// stored in the cache is sliced along this axis.
    #[serde(default)]
    pub state_batch_axis: u32,
}

impl SuperGraphNodeRNNCacheWrite {
    pub fn new(
        key_input: SuperGraphLink,
        tokens_input: SuperGraphLink,
        state_inputs: Vec<(String, SuperGraphLink)>,
        tokens_batch_axis: u32,
        tokens_seq_axis: u32,
        state_batch_axis: u32,
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            global_id: GlobalId::new(rng),
            label: None,
            key_input: Some(key_input),
            tokens_input: Some(tokens_input),
            state_inputs: state_inputs
                .into_iter()
                .map(|(name, link)| (name, Some(link)))
                .collect(),
            tokens_batch_axis,
            tokens_seq_axis,
            state_batch_axis,
        }
    }
}

impl SuperGraphNode for SuperGraphNodeRNNCacheWrite {
    fn to_any(self) -> SuperGraphAnyNode {
        SuperGraphAnyNode::RNNCacheWrite(self)
    }

    fn eval<'short, 'model, 'p, P: Pool + 'p, T: SuperGraphObserver>(
        &'short self,
        _node_path: &[GlobalId],
        data: &mut SuperGraphData<'p, 'model, P>,
        context: &mut SuperGraphContext<'short, 'model, 'p, P, T>,
    ) -> Result<(), SuperGraphError> {
        let key_input_link = require_node_link(self.key_input, "RNNCacheWrite", "key_input")?;
        let tokens_input_link =
            require_node_link(self.tokens_input, "RNNCacheWrite", "tokens_input")?;
        let Some(caches) = &mut context.caches else {
            return Ok(());
        };
        let key_input = *data
            .hashes
            .get(&key_input_link)
            .ok_or(SuperGraphError::MissingLinkError(String::new()))?;
        let tokens_input = data
            .tensors
            .get(&tokens_input_link)
            .ok_or(SuperGraphError::MissingLinkError(String::new()))?;
        let tokens_shape = tokens_input.shape();
        let batch_axis = self.tokens_batch_axis as usize;
        let seq_axis = self.tokens_seq_axis as usize;
        if batch_axis >= tokens_shape.len() || seq_axis >= tokens_shape.len() {
            return Err(SuperGraphError::InvalidInputError(format!(
                "RNNCacheWrite axes batch={batch_axis} seq={seq_axis} out of range \
                 for tokens shape {:?}",
                tokens_shape
            )));
        }
        if batch_axis == seq_axis {
            return Err(SuperGraphError::InvalidInputError(
                "RNNCacheWrite tokens_batch_axis must differ from tokens_seq_axis".to_string(),
            ));
        }
        let batch = tokens_shape[batch_axis] as usize;

        // Extract per-row token sequences (cache keys).
        let mut per_row_tokens: Vec<Vec<u32>> = Vec::with_capacity(batch);
        for r in 0..batch {
            let mut slice_ranges: Vec<(u64, u64)> = tokens_shape.iter().map(|&d| (0, d)).collect();
            slice_ranges[batch_axis] = (r as u64, r as u64 + 1);
            let row = tokens_input.slice(&slice_ranges).map_err(|e| {
                SuperGraphError::InvalidInputError(format!("RNNCacheWrite row slice: {e}"))
            })?;
            let toks: Vec<u32> = (0..row.numel())
                .map(|i| row.read_element(i).to_i64() as u32)
                .collect();
            per_row_tokens.push(toks);
        }

        // For each state, slice per row along state_batch_axis and cache
        // as [..., 1 at state_batch_axis, ...]. On read this is stacked
        // back up along the same axis.
        let state_batch_axis = self.state_batch_axis as usize;
        let cache_pool = caches.pool.clone();
        let mut per_row_state: Vec<
            HashMap<
                String,
                crate::numeric_tensor::NumericTensor<'static, DynRank, crate::pool::ArcTrackedPool>,
            >,
        > = (0..batch).map(|_| HashMap::new()).collect();
        for (state_name, link_opt) in self.state_inputs.iter() {
            let link = require_node_link(*link_opt, "RNNCacheWrite", "state_inputs")?;
            let state_tensor = data
                .tensors
                .get(&link)
                .ok_or(SuperGraphError::MissingLinkError(String::new()))?;
            let state_shape = state_tensor.shape();
            if state_batch_axis >= state_shape.len() {
                return Err(SuperGraphError::InvalidInputError(format!(
                    "RNNCacheWrite state_batch_axis {state_batch_axis} out of range for \
                     state {state_name} shape {:?}",
                    state_shape
                )));
            }
            if state_shape[state_batch_axis] as usize != batch {
                return Err(SuperGraphError::InvalidInputError(format!(
                    "RNNCacheWrite state {state_name} has {} rows at axis {state_batch_axis}, \
                     tokens have {batch} rows",
                    state_shape[state_batch_axis]
                )));
            }
            for (r, per_row) in per_row_state.iter_mut().enumerate().take(batch) {
                let mut slice_ranges: Vec<(u64, u64)> =
                    state_shape.iter().map(|&d| (0, d)).collect();
                slice_ranges[state_batch_axis] = (r as u64, r as u64 + 1);
                let row_view = state_tensor.slice(&slice_ranges).map_err(|e| {
                    SuperGraphError::InvalidInputError(format!(
                        "RNNCacheWrite state row slice: {e}"
                    ))
                })?;
                // Materialize into a contiguous tensor so the cache owns
                // its own storage (the slice is a view over the eval pool).
                let owned_in_eval_pool = PoolNumericTensor::from_fn(
                    row_view.shape().to_vec(),
                    row_view.dtype(),
                    context.pool,
                    |i| row_view.read_element(i),
                )
                .map_err(alloc_err)?;
                let cached = owned_in_eval_pool
                    .to_arc_tracked_static(&cache_pool)
                    .map_err(|e| SuperGraphError::InvalidInputError(format!("allocation: {e}")))?;
                per_row.insert(state_name.clone(), cached);
            }
        }

        let entry = caches.rnn_cache.entry(key_input).or_default();
        for (toks, state_map) in per_row_tokens.into_iter().zip(per_row_state.into_iter()) {
            entry.insert(toks, state_map);
        }
        Ok(())
    }
    fn op_kind(&self) -> String {
        "RNNCacheWrite".to_string()
    }
    fn label(&self) -> Option<String> {
        self.label.clone()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(
            self.key_input
                .map(|x| x.to_any())
                .into_iter()
                .chain(self.tokens_input.map(|x| x.to_any()))
                .chain(
                    self.state_inputs
                        .iter()
                        .filter_map(|x| x.1.map(|link| link.to_any())),
                ),
        )
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(std::iter::empty())
    }
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeTensorCacheRead {
    global_id: GlobalId,
    pub label: Option<String>,
    key_input: Option<SuperGraphLink>,
    default_input: Option<SuperGraphLink>,
    value_output: Option<SuperGraphLink>,
    hit_output: Option<SuperGraphLink>,
}

impl SuperGraphNodeTensorCacheRead {
    pub fn new(
        key_input: SuperGraphLink,
        default_input: SuperGraphLink,
        value_output: SuperGraphLink,
        hit_output: SuperGraphLink,
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            global_id: GlobalId::new(rng),
            label: None,
            key_input: Some(key_input),
            default_input: Some(default_input),
            value_output: Some(value_output),
            hit_output: Some(hit_output),
        }
    }
}

impl SuperGraphNode for SuperGraphNodeTensorCacheRead {
    fn to_any(self) -> SuperGraphAnyNode {
        SuperGraphAnyNode::TensorCacheRead(self)
    }

    fn eval<'short, 'model, 'p, P: Pool + 'p, T: SuperGraphObserver>(
        &'short self,
        _node_path: &[GlobalId],
        data: &mut SuperGraphData<'p, 'model, P>,
        context: &mut SuperGraphContext<'short, 'model, 'p, P, T>,
    ) -> Result<(), SuperGraphError> {
        let key_input_link = require_node_link(self.key_input, "TensorCacheRead", "key_input")?;
        let default_input_link =
            require_node_link(self.default_input, "TensorCacheRead", "default_input")?;
        let value_output_link =
            require_node_link(self.value_output, "TensorCacheRead", "value_output")?;
        let hit_output_link = require_node_link(self.hit_output, "TensorCacheRead", "hit_output")?;
        let key_input = *data
            .hashes
            .get(&key_input_link)
            .ok_or(SuperGraphError::MissingLinkError(String::new()))?;
        let mut hit = false;
        if let Some(caches) = &mut context.caches
            && let Some(value) = caches.tensor_cache.get(&key_input)
        {
            let output = value
                .to_tensor(context.pool)
                .map_err(|e| SuperGraphError::InvalidInputError(format!("allocation: {e}")))?;
            data.tensors.insert(value_output_link, output);
            hit = true;
        }
        if !hit {
            let default_input = data
                .tensors
                .get(&default_input_link)
                .ok_or(SuperGraphError::MissingLinkError(String::new()))?
                .to_tensor(context.pool)
                .map_err(|e| SuperGraphError::InvalidInputError(format!("allocation: {e}")))?;
            data.tensors.insert(value_output_link, default_input);
        }
        data.tensors
            .insert(hit_output_link, tensor_bool_scalar(hit, context.pool)?);
        Ok(())
    }

    fn op_kind(&self) -> String {
        "TensorCacheRead".to_string()
    }
    fn label(&self) -> Option<String> {
        self.label.clone()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(
            self.key_input
                .map(|x| x.to_any())
                .into_iter()
                .chain(self.default_input.map(|x| x.to_any())),
        )
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(
            self.value_output
                .map(|x| x.to_any())
                .into_iter()
                .chain(self.hit_output.map(|x| x.to_any())),
        )
    }
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeTensorCacheWrite {
    global_id: GlobalId,
    pub label: Option<String>,
    key_input: Option<SuperGraphLink>,
    value_input: Option<SuperGraphLink>,
    write_enable_input: Option<SuperGraphLink>,
}

impl SuperGraphNodeTensorCacheWrite {
    pub fn new(
        key_input: SuperGraphLink,
        value_input: SuperGraphLink,
        write_enable_input: SuperGraphLink,
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            global_id: GlobalId::new(rng),
            label: None,
            key_input: Some(key_input),
            value_input: Some(value_input),
            write_enable_input: Some(write_enable_input),
        }
    }
}

impl SuperGraphNode for SuperGraphNodeTensorCacheWrite {
    fn to_any(self) -> SuperGraphAnyNode {
        SuperGraphAnyNode::TensorCacheWrite(self)
    }

    fn eval<'short, 'model, 'p, P: Pool + 'p, T: SuperGraphObserver>(
        &'short self,
        _node_path: &[GlobalId],
        data: &mut SuperGraphData<'p, 'model, P>,
        context: &mut SuperGraphContext<'short, 'model, 'p, P, T>,
    ) -> Result<(), SuperGraphError> {
        let key_input_link = require_node_link(self.key_input, "TensorCacheWrite", "key_input")?;
        let value_input_link =
            require_node_link(self.value_input, "TensorCacheWrite", "value_input")?;
        let write_enable_input_link = require_node_link(
            self.write_enable_input,
            "TensorCacheWrite",
            "write_enable_input",
        )?;
        let write_enable = read_rank0_bool_tensor(
            data.tensors
                .get(&write_enable_input_link)
                .ok_or(SuperGraphError::MissingLinkError(String::new()))?,
            "TensorCacheWrite.write_enable_input",
        )?;
        if write_enable && let Some(caches) = &mut context.caches {
            let key_input = *data
                .hashes
                .get(&key_input_link)
                .ok_or(SuperGraphError::MissingLinkError(String::new()))?;
            let cache_pool = caches.pool.clone();
            let value_input = data
                .tensors
                .get(&value_input_link)
                .ok_or(SuperGraphError::MissingLinkError(String::new()))?;
            let cached = value_input
                .to_arc_tracked_static(&cache_pool)
                .map_err(|e| SuperGraphError::InvalidInputError(format!("allocation: {e}")))?;
            caches.tensor_cache.insert(key_input, cached);
        }
        Ok(())
    }
    fn op_kind(&self) -> String {
        "TensorCacheWrite".to_string()
    }
    fn label(&self) -> Option<String> {
        self.label.clone()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(
            self.key_input
                .map(|x| x.to_any())
                .into_iter()
                .chain(self.value_input.map(|x| x.to_any()))
                .chain(self.write_enable_input.map(|x| x.to_any())),
        )
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(std::iter::empty())
    }
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeTensorPackCacheRead {
    global_id: GlobalId,
    pub label: Option<String>,
    key_input: Option<SuperGraphLink>,
    value_outputs: Vec<(String, Option<SuperGraphLink>)>,
    default_value_inputs: Vec<(String, Option<SuperGraphLink>)>,
    hit_output: Option<SuperGraphLink>,
}

impl SuperGraphNodeTensorPackCacheRead {
    pub fn new(
        key_input: SuperGraphLink,
        value_outputs: Vec<(String, SuperGraphLink)>,
        default_value_inputs: Vec<(String, SuperGraphLink)>,
        hit_output: SuperGraphLink,
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            global_id: GlobalId::new(rng),
            label: None,
            key_input: Some(key_input),
            value_outputs: value_outputs
                .into_iter()
                .map(|(name, link)| (name, Some(link)))
                .collect(),
            default_value_inputs: default_value_inputs
                .into_iter()
                .map(|(name, link)| (name, Some(link)))
                .collect(),
            hit_output: Some(hit_output),
        }
    }
}

impl SuperGraphNode for SuperGraphNodeTensorPackCacheRead {
    fn to_any(self) -> SuperGraphAnyNode {
        SuperGraphAnyNode::TensorPackCacheRead(self)
    }

    fn eval<'short, 'model, 'p, P: Pool + 'p, T: SuperGraphObserver>(
        &'short self,
        _node_path: &[GlobalId],
        data: &mut SuperGraphData<'p, 'model, P>,
        context: &mut SuperGraphContext<'short, 'model, 'p, P, T>,
    ) -> Result<(), SuperGraphError> {
        let key_input_link = require_node_link(self.key_input, "TensorPackCacheRead", "key_input")?;
        let hit_output_link =
            require_node_link(self.hit_output, "TensorPackCacheRead", "hit_output")?;
        let key_input = *data
            .hashes
            .get(&key_input_link)
            .ok_or(SuperGraphError::MissingLinkError(String::new()))?;
        // Collect default input links for miss path
        let default_input_links: Vec<(String, SuperGraphLink)> = self
            .default_value_inputs
            .iter()
            .map(|(name, input)| {
                let input_link =
                    require_node_link(*input, "TensorPackCacheRead", "default_value_inputs")?;
                Ok((name.clone(), input_link))
            })
            .collect::<Result<_, SuperGraphError>>()?;

        let mut hit = false;
        if let Some(caches) = &mut context.caches
            && let Some(cached_values) = caches.tensor_pack_cache.get(&key_input)
        {
            let full_hit = self
                .value_outputs
                .iter()
                .all(|(name, _)| cached_values.contains_key(name));
            if full_hit {
                for (name, output) in &self.value_outputs {
                    let output_link =
                        require_node_link(*output, "TensorPackCacheRead", "value_outputs")?;
                    let cached = cached_values.get(name).unwrap();
                    let pool_tensor = cached.to_tensor(context.pool).map_err(|e| {
                        SuperGraphError::InvalidInputError(format!("allocation: {e}"))
                    })?;
                    data.tensors.insert(output_link, pool_tensor);
                }
                hit = true;
            }
        }

        if !hit {
            for (name, output) in &self.value_outputs {
                let output_link =
                    require_node_link(*output, "TensorPackCacheRead", "value_outputs")?;
                let default_link = default_input_links.iter().find(|(n, _)| n == name).ok_or(
                    SuperGraphError::InvalidInputError(format!(
                        "TensorPackCacheRead missing default input for key '{name}'"
                    )),
                )?;
                let default_value = data
                    .tensors
                    .get(&default_link.1)
                    .ok_or(SuperGraphError::MissingLinkError(String::new()))?
                    .to_tensor(context.pool)
                    .map_err(|e| SuperGraphError::InvalidInputError(format!("allocation: {e}")))?;
                data.tensors.insert(output_link, default_value);
            }
        }

        data.tensors
            .insert(hit_output_link, tensor_bool_scalar(hit, context.pool)?);
        Ok(())
    }

    fn op_kind(&self) -> String {
        "TensorPackCacheRead".to_string()
    }
    fn label(&self) -> Option<String> {
        self.label.clone()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(
            self.key_input.map(|x| x.to_any()).into_iter().chain(
                self.default_value_inputs
                    .iter()
                    .filter_map(|(_, x)| x.map(|link| link.to_any())),
            ),
        )
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(
            self.value_outputs
                .iter()
                .filter_map(|(_, x)| x.map(|link| link.to_any()))
                .chain(self.hit_output.map(|x| x.to_any())),
        )
    }
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphNodeTensorPackCacheWrite {
    global_id: GlobalId,
    pub label: Option<String>,
    key_input: Option<SuperGraphLink>,
    value_inputs: Vec<(String, Option<SuperGraphLink>)>,
    write_enable_input: Option<SuperGraphLink>,
}

impl SuperGraphNodeTensorPackCacheWrite {
    pub fn new(
        key_input: SuperGraphLink,
        value_inputs: Vec<(String, SuperGraphLink)>,
        write_enable_input: SuperGraphLink,
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            global_id: GlobalId::new(rng),
            label: None,
            key_input: Some(key_input),
            value_inputs: value_inputs
                .into_iter()
                .map(|(name, link)| (name, Some(link)))
                .collect(),
            write_enable_input: Some(write_enable_input),
        }
    }
}

impl SuperGraphNode for SuperGraphNodeTensorPackCacheWrite {
    fn to_any(self) -> SuperGraphAnyNode {
        SuperGraphAnyNode::TensorPackCacheWrite(self)
    }

    fn eval<'short, 'model, 'p, P: Pool + 'p, T: SuperGraphObserver>(
        &'short self,
        _node_path: &[GlobalId],
        data: &mut SuperGraphData<'p, 'model, P>,
        context: &mut SuperGraphContext<'short, 'model, 'p, P, T>,
    ) -> Result<(), SuperGraphError> {
        let key_input_link =
            require_node_link(self.key_input, "TensorPackCacheWrite", "key_input")?;
        let write_enable_input_link = require_node_link(
            self.write_enable_input,
            "TensorPackCacheWrite",
            "write_enable_input",
        )?;
        let write_enable = read_rank0_bool_tensor(
            data.tensors
                .get(&write_enable_input_link)
                .ok_or(SuperGraphError::MissingLinkError(String::new()))?,
            "TensorPackCacheWrite.write_enable_input",
        )?;
        if write_enable && let Some(caches) = &mut context.caches {
            let key_input = *data
                .hashes
                .get(&key_input_link)
                .ok_or(SuperGraphError::MissingLinkError(String::new()))?;
            let cache_pool = caches.pool.clone();
            let values = self
                .value_inputs
                .iter()
                .map(|(name, input)| {
                    let input_link =
                        require_node_link(*input, "TensorPackCacheWrite", "value_inputs")?;
                    let tensor = data
                        .tensors
                        .get(&input_link)
                        .ok_or(SuperGraphError::MissingLinkError(String::new()))?;
                    let cached = tensor.to_arc_tracked_static(&cache_pool).map_err(|e| {
                        SuperGraphError::InvalidInputError(format!("allocation: {e}"))
                    })?;
                    Ok((name.clone(), cached))
                })
                .collect::<Result<
                    HashMap<
                        String,
                        crate::numeric_tensor::NumericTensor<
                            'static,
                            DynRank,
                            crate::pool::ArcTrackedPool,
                        >,
                    >,
                    SuperGraphError,
                >>()?;
            caches.tensor_pack_cache.insert(key_input, values);
        }
        Ok(())
    }
    fn op_kind(&self) -> String {
        "TensorPackCacheWrite".to_string()
    }
    fn label(&self) -> Option<String> {
        self.label.clone()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(
            self.key_input
                .map(|x| x.to_any())
                .into_iter()
                .chain(self.write_enable_input.map(|x| x.to_any()))
                .chain(
                    self.value_inputs
                        .iter()
                        .filter_map(|(_, x)| x.map(|link| link.to_any())),
                ),
        )
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_> {
        Box::new(std::iter::empty())
    }
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[allow(clippy::large_enum_variant)]
pub enum SuperGraphAnyNode {
    ModelExecution(SuperGraphNodeModelExecution),
    TokenizerEncode(SuperGraphNodeTokenizerEncode),
    TokenizerDecode(SuperGraphNodeTokenizerDecode),
    TokenizerLoad(SuperGraphNodeTokenizerLoad),
    TextToPhonemes(SuperGraphNodeTextToPhonemes),
    PiperPhonemesToTensor(SuperGraphNodePiperPhonemesToTensor),
    KokoroPhonemesToTensor(SuperGraphNodeKokoroPhonemesToTensor),
    F5TextToTensor(SuperGraphNodeF5TextToTensor),
    TensorToImage(SuperGraphNodeTensorToImage),
    TensorToAudioClip(SuperGraphNodeTensorToAudioClip),
    TensorToVideoClip(SuperGraphNodeTensorToVideoClip),
    AudioClipToTensor(SuperGraphNodeAudioClipToTensor),
    VideoClipToTensor(SuperGraphNodeVideoClipToTensor),
    AudioClipToMelSpectrogram(SuperGraphNodeAudioClipToMelSpectrogram),
    MilliOpGraph(SuperGraphNodeMilliOpGraph),
    Scan(SuperGraphNodeScan),
    ReportProgress(SuperGraphNodeReportProgress),
    RNNCacheWrite(SuperGraphNodeRNNCacheWrite),
    RNNCacheRead(SuperGraphNodeRNNCacheRead),
    TensorCacheRead(SuperGraphNodeTensorCacheRead),
    TensorCacheWrite(SuperGraphNodeTensorCacheWrite),
    TensorPackCacheRead(SuperGraphNodeTensorPackCacheRead),
    TensorPackCacheWrite(SuperGraphNodeTensorPackCacheWrite),
}

macro_rules! delegate {
    ($name:ident($($arg:ident: $ty:ty),*) -> $ret:ty) => {
        fn $name(&self, $($arg: $ty),*) -> $ret {
            match self {
                SuperGraphAnyNode::ModelExecution(x) => SuperGraphNode::$name(x,$($arg),*),
                SuperGraphAnyNode::TokenizerEncode(x) => SuperGraphNode::$name(x,$($arg),*),
                SuperGraphAnyNode::TokenizerDecode(x) => SuperGraphNode::$name(x,$($arg),*),
                SuperGraphAnyNode::TokenizerLoad(x) => SuperGraphNode::$name(x,$($arg),*),
                SuperGraphAnyNode::TextToPhonemes(x) => SuperGraphNode::$name(x,$($arg),*),
                SuperGraphAnyNode::PiperPhonemesToTensor(x) => SuperGraphNode::$name(x,$($arg),*),
                SuperGraphAnyNode::KokoroPhonemesToTensor(x) => SuperGraphNode::$name(x,$($arg),*),
                SuperGraphAnyNode::F5TextToTensor(x) => SuperGraphNode::$name(x,$($arg),*),
                SuperGraphAnyNode::TensorToImage(x) => SuperGraphNode::$name(x,$($arg),*),
                SuperGraphAnyNode::TensorToAudioClip(x) => SuperGraphNode::$name(x,$($arg),*),
                SuperGraphAnyNode::TensorToVideoClip(x) => SuperGraphNode::$name(x,$($arg),*),
                SuperGraphAnyNode::AudioClipToTensor(x) => SuperGraphNode::$name(x,$($arg),*),
                SuperGraphAnyNode::VideoClipToTensor(x) => SuperGraphNode::$name(x,$($arg),*),
                SuperGraphAnyNode::AudioClipToMelSpectrogram(x) => SuperGraphNode::$name(x,$($arg),*),
                SuperGraphAnyNode::MilliOpGraph(x) => SuperGraphNode::$name(x,$($arg),*),
                SuperGraphAnyNode::Scan(x) => SuperGraphNode::$name(x,$($arg),*),
                SuperGraphAnyNode::ReportProgress(x) => SuperGraphNode::$name(x,$($arg),*),
                SuperGraphAnyNode::RNNCacheRead(x) => SuperGraphNode::$name(x,$($arg),*),
                SuperGraphAnyNode::RNNCacheWrite(x) => SuperGraphNode::$name(x,$($arg),*),
                SuperGraphAnyNode::TensorCacheRead(x) => SuperGraphNode::$name(x,$($arg),*),
                SuperGraphAnyNode::TensorCacheWrite(x) => SuperGraphNode::$name(x,$($arg),*),
                SuperGraphAnyNode::TensorPackCacheRead(x) => SuperGraphNode::$name(x,$($arg),*),
                SuperGraphAnyNode::TensorPackCacheWrite(x) => SuperGraphNode::$name(x,$($arg),*),
            }
        }
    }
}

impl SuperGraphAnyNode {
    pub fn get_sub_graph(&self) -> Option<&SuperGraph> {
        match self {
            SuperGraphAnyNode::Scan(x) => Some(&x.inner_graph),
            _ => None,
        }
    }

    pub fn get_sub_graph_mut(&mut self) -> Option<&mut SuperGraph> {
        match self {
            SuperGraphAnyNode::Scan(x) => Some(&mut x.inner_graph),
            _ => None,
        }
    }
}

impl SuperGraphNode for SuperGraphAnyNode {
    fn to_any(self) -> SuperGraphAnyNode {
        self
    }

    fn eval<'short, 'model, 'p, P: Pool + 'p, T: SuperGraphObserver>(
        &'short self,
        node_path: &[GlobalId],
        data: &mut SuperGraphData<'p, 'model, P>,
        context: &mut SuperGraphContext<'short, 'model, 'p, P, T>,
    ) -> Result<(), SuperGraphError> {
        match self {
            SuperGraphAnyNode::ModelExecution(node) => node.eval(node_path, data, context),
            SuperGraphAnyNode::TokenizerEncode(node) => node.eval(node_path, data, context),
            SuperGraphAnyNode::TokenizerDecode(node) => node.eval(node_path, data, context),
            SuperGraphAnyNode::TokenizerLoad(node) => node.eval(node_path, data, context),
            SuperGraphAnyNode::TextToPhonemes(node) => node.eval(node_path, data, context),
            SuperGraphAnyNode::PiperPhonemesToTensor(node) => node.eval(node_path, data, context),
            SuperGraphAnyNode::KokoroPhonemesToTensor(node) => node.eval(node_path, data, context),
            SuperGraphAnyNode::F5TextToTensor(node) => node.eval(node_path, data, context),
            SuperGraphAnyNode::TensorToImage(node) => node.eval(node_path, data, context),
            SuperGraphAnyNode::TensorToAudioClip(node) => node.eval(node_path, data, context),
            SuperGraphAnyNode::TensorToVideoClip(node) => node.eval(node_path, data, context),
            SuperGraphAnyNode::AudioClipToTensor(node) => node.eval(node_path, data, context),
            SuperGraphAnyNode::VideoClipToTensor(node) => node.eval(node_path, data, context),
            SuperGraphAnyNode::AudioClipToMelSpectrogram(node) => {
                node.eval(node_path, data, context)
            }
            SuperGraphAnyNode::MilliOpGraph(node) => node.eval(node_path, data, context),
            SuperGraphAnyNode::Scan(node) => node.eval(node_path, data, context),
            SuperGraphAnyNode::ReportProgress(node) => node.eval(node_path, data, context),
            SuperGraphAnyNode::RNNCacheWrite(node) => node.eval(node_path, data, context),
            SuperGraphAnyNode::RNNCacheRead(node) => node.eval(node_path, data, context),
            SuperGraphAnyNode::TensorCacheRead(node) => node.eval(node_path, data, context),
            SuperGraphAnyNode::TensorCacheWrite(node) => node.eval(node_path, data, context),
            SuperGraphAnyNode::TensorPackCacheRead(node) => node.eval(node_path, data, context),
            SuperGraphAnyNode::TensorPackCacheWrite(node) => node.eval(node_path, data, context),
        }
    }

    delegate!(op_kind() -> String);
    delegate!(label() -> Option<String>);
    delegate!(inputs() -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_>);
    delegate!(outputs() -> Box<dyn Iterator<Item = SuperGraphAnyLink> + '_>);
    fn input_slots(&self) -> Box<dyn Iterator<Item = Option<SuperGraphAnyLink>> + '_> {
        let slots = match self {
            SuperGraphAnyNode::ModelExecution(node) => {
                let mut slots = node
                    .tensor_inputs
                    .iter()
                    .map(|(link, _)| link.map(|x| x.to_any()))
                    .collect::<Vec<_>>();
                slots.push(node.tensor_map.map(|x| x.to_any()));
                slots
            }
            SuperGraphAnyNode::TokenizerLoad(_) => Vec::new(),
            SuperGraphAnyNode::TokenizerEncode(node) => vec![
                node.tokenizer.map(|x| x.to_any()),
                node.text_input.map(|x| x.to_any()),
            ],
            SuperGraphAnyNode::TokenizerDecode(node) => vec![
                node.tokenizer.map(|x| x.to_any()),
                node.tensor_input.map(|x| x.to_any()),
            ],
            SuperGraphAnyNode::TextToPhonemes(node) => vec![node.text_input.map(|x| x.to_any())],
            SuperGraphAnyNode::PiperPhonemesToTensor(node) => {
                vec![node.phonemes_input.map(|x| x.to_any())]
            }
            SuperGraphAnyNode::KokoroPhonemesToTensor(node) => {
                vec![node.phonemes_input.map(|x| x.to_any())]
            }
            SuperGraphAnyNode::F5TextToTensor(node) => vec![node.text_input.map(|x| x.to_any())],
            SuperGraphAnyNode::TensorToImage(node) => vec![node.tensor_input.map(|x| x.to_any())],
            SuperGraphAnyNode::TensorToAudioClip(node) => {
                vec![node.tensor_input.map(|x| x.to_any())]
            }
            SuperGraphAnyNode::TensorToVideoClip(node) => {
                vec![node.tensor_input.map(|x| x.to_any())]
            }
            SuperGraphAnyNode::AudioClipToTensor(node) => {
                vec![node.audio_input.map(|x| x.to_any())]
            }
            SuperGraphAnyNode::VideoClipToTensor(node) => {
                vec![node.video_input.map(|x| x.to_any())]
            }
            SuperGraphAnyNode::AudioClipToMelSpectrogram(node) => {
                vec![node.audio_input.map(|x| x.to_any())]
            }
            SuperGraphAnyNode::MilliOpGraph(node) => node
                .graph
                .input_link_ids()
                .map(|(id, _)| Some(SuperGraphLink::tensor(id).to_any()))
                .collect::<Vec<_>>(),
            SuperGraphAnyNode::Scan(node) => {
                let mut slots = Vec::new();
                slots.push(node.iteration_count.map(|x| x.to_any()));
                slots.extend(
                    node.simple_inputs
                        .iter()
                        .map(|(outer, _)| outer.map(|x| x.to_any())),
                );
                slots.extend(
                    node.state_links
                        .iter()
                        .map(|(outer, _, _)| outer.map(|x| x.to_any())),
                );
                slots.extend(
                    node.scan_inputs
                        .iter()
                        .map(|(outer, _, _)| outer.map(|x| x.to_any())),
                );
                slots
            }
            SuperGraphAnyNode::ReportProgress(node) => vec![
                node.tier_input.map(|x| x.to_any()),
                node.numerator_input.map(|x| x.to_any()),
                node.denominator_input.map(|x| x.to_any()),
            ],
            SuperGraphAnyNode::RNNCacheWrite(node) => {
                let mut slots = vec![
                    node.key_input.map(|x| x.to_any()),
                    node.tokens_input.map(|x| x.to_any()),
                ];
                slots.extend(
                    node.state_inputs
                        .iter()
                        .map(|(_, link)| link.map(|x| x.to_any())),
                );
                slots
            }
            SuperGraphAnyNode::RNNCacheRead(node) => {
                let mut slots = vec![
                    node.key_input.map(|x| x.to_any()),
                    node.tokens_input.map(|x| x.to_any()),
                ];
                slots.extend(
                    node.default_state_inputs
                        .iter()
                        .map(|(_, link)| link.map(|x| x.to_any())),
                );
                slots
            }
            SuperGraphAnyNode::TensorCacheRead(node) => vec![
                node.key_input.map(|x| x.to_any()),
                node.default_input.map(|x| x.to_any()),
            ],
            SuperGraphAnyNode::TensorCacheWrite(node) => vec![
                node.key_input.map(|x| x.to_any()),
                node.value_input.map(|x| x.to_any()),
                node.write_enable_input.map(|x| x.to_any()),
            ],
            SuperGraphAnyNode::TensorPackCacheRead(node) => {
                let mut slots = vec![node.key_input.map(|x| x.to_any())];
                slots.extend(
                    node.default_value_inputs
                        .iter()
                        .map(|(_, link)| link.map(|x| x.to_any())),
                );
                slots
            }
            SuperGraphAnyNode::TensorPackCacheWrite(node) => {
                let mut slots = vec![
                    node.key_input.map(|x| x.to_any()),
                    node.write_enable_input.map(|x| x.to_any()),
                ];
                slots.extend(
                    node.value_inputs
                        .iter()
                        .map(|(_, link)| link.map(|x| x.to_any())),
                );
                slots
            }
        };
        Box::new(slots.into_iter())
    }
    fn output_slots(&self) -> Box<dyn Iterator<Item = Option<SuperGraphAnyLink>> + '_> {
        let slots = match self {
            SuperGraphAnyNode::ModelExecution(node) => node
                .tensor_outputs
                .iter()
                .map(|(_, link)| link.map(|x| x.to_any()))
                .collect::<Vec<_>>(),
            SuperGraphAnyNode::TokenizerLoad(node) => vec![node.output.map(|x| x.to_any())],
            SuperGraphAnyNode::TokenizerEncode(node) => {
                vec![node.tensor_output.map(|x| x.to_any())]
            }
            SuperGraphAnyNode::TokenizerDecode(node) => vec![node.text_output.map(|x| x.to_any())],
            SuperGraphAnyNode::TextToPhonemes(node) => {
                vec![node.phonemes_output.map(|x| x.to_any())]
            }
            SuperGraphAnyNode::PiperPhonemesToTensor(node) => vec![
                node.token_ids_output.map(|x| x.to_any()),
                node.input_lengths_output.map(|x| x.to_any()),
            ],
            SuperGraphAnyNode::KokoroPhonemesToTensor(node) => {
                vec![node.token_ids_output.map(|x| x.to_any())]
            }
            SuperGraphAnyNode::F5TextToTensor(node) => {
                vec![node.token_ids_output.map(|x| x.to_any())]
            }
            SuperGraphAnyNode::TensorToImage(node) => vec![node.image_output.map(|x| x.to_any())],
            SuperGraphAnyNode::TensorToAudioClip(node) => {
                vec![node.audio_output.map(|x| x.to_any())]
            }
            SuperGraphAnyNode::TensorToVideoClip(node) => {
                vec![node.video_output.map(|x| x.to_any())]
            }
            SuperGraphAnyNode::AudioClipToTensor(node) => {
                vec![node.tensor_output.map(|x| x.to_any())]
            }
            SuperGraphAnyNode::VideoClipToTensor(node) => {
                vec![node.tensor_output.map(|x| x.to_any())]
            }
            SuperGraphAnyNode::AudioClipToMelSpectrogram(node) => {
                vec![node.tensor_output.map(|x| x.to_any())]
            }
            SuperGraphAnyNode::MilliOpGraph(node) => node
                .graph
                .output_link_ids()
                .map(|(id, _)| Some(SuperGraphLink::tensor(id).to_any()))
                .collect::<Vec<_>>(),
            SuperGraphAnyNode::Scan(node) => {
                let mut slots = Vec::new();
                slots.extend(
                    node.simple_outputs
                        .iter()
                        .map(|(_, outer)| outer.map(|x| x.to_any())),
                );
                slots.extend(
                    node.scan_outputs
                        .iter()
                        .map(|(_, outer, _)| outer.map(|x| x.to_any())),
                );
                slots
            }
            SuperGraphAnyNode::ReportProgress(_) => Vec::new(),
            SuperGraphAnyNode::RNNCacheWrite(_) => Vec::new(),
            SuperGraphAnyNode::RNNCacheRead(node) => {
                let mut slots = vec![node.tokens_output.map(|x| x.to_any())];
                slots.extend(
                    node.state_outputs
                        .iter()
                        .map(|(_, link)| link.map(|x| x.to_any())),
                );
                slots
            }
            SuperGraphAnyNode::TensorCacheRead(node) => vec![
                node.value_output.map(|x| x.to_any()),
                node.hit_output.map(|x| x.to_any()),
            ],
            SuperGraphAnyNode::TensorCacheWrite(_) => Vec::new(),
            SuperGraphAnyNode::TensorPackCacheRead(node) => {
                let mut slots = node
                    .value_outputs
                    .iter()
                    .map(|(_, link)| link.map(|x| x.to_any()))
                    .collect::<Vec<_>>();
                slots.push(node.hit_output.map(|x| x.to_any()));
                slots
            }
            SuperGraphAnyNode::TensorPackCacheWrite(_) => Vec::new(),
        };
        Box::new(slots.into_iter())
    }
    fn set_input_slot(
        &mut self,
        slot_index: usize,
        link: Option<GlobalId>,
    ) -> Result<(), NodeSlotEditError> {
        match self {
            SuperGraphAnyNode::ModelExecution(node) => {
                let slot_count = node.tensor_inputs.len() + 1;
                if slot_index < node.tensor_inputs.len() {
                    node.tensor_inputs[slot_index].0 =
                        set_slot_link(link, SuperGraphLinkKind::Tensor);
                    Ok(())
                } else if slot_index == node.tensor_inputs.len() {
                    node.tensor_map = set_slot_link(link, SuperGraphLinkKind::TensorMap);
                    Ok(())
                } else {
                    Err(NodeSlotEditError::invalid_slot_index(
                        "Model Execution".to_string(),
                        SlotDirection::Input,
                        slot_index,
                        slot_count,
                    ))
                }
            }
            SuperGraphAnyNode::TokenizerLoad(_) => Err(NodeSlotEditError::invalid_slot_index(
                "Tokenizer Load".to_string(),
                SlotDirection::Input,
                slot_index,
                0,
            )),
            SuperGraphAnyNode::TokenizerEncode(node) => match slot_index {
                0 => {
                    node.tokenizer = set_slot_link(link, SuperGraphLinkKind::Tokenizer);
                    Ok(())
                }
                1 => {
                    node.text_input = set_slot_link(link, SuperGraphLinkKind::String);
                    Ok(())
                }
                _ => Err(NodeSlotEditError::invalid_slot_index(
                    "Tokenizer Encode".to_string(),
                    SlotDirection::Input,
                    slot_index,
                    2,
                )),
            },
            SuperGraphAnyNode::TokenizerDecode(node) => match slot_index {
                0 => {
                    node.tokenizer = set_slot_link(link, SuperGraphLinkKind::Tokenizer);
                    Ok(())
                }
                1 => {
                    node.tensor_input = set_slot_link(link, SuperGraphLinkKind::Tensor);
                    Ok(())
                }
                _ => Err(NodeSlotEditError::invalid_slot_index(
                    "Tokenizer Decode".to_string(),
                    SlotDirection::Input,
                    slot_index,
                    2,
                )),
            },
            SuperGraphAnyNode::TextToPhonemes(node) => match slot_index {
                0 => {
                    node.text_input = set_slot_link(link, SuperGraphLinkKind::String);
                    Ok(())
                }
                _ => Err(NodeSlotEditError::invalid_slot_index(
                    "TextToPhonemes".to_string(),
                    SlotDirection::Input,
                    slot_index,
                    1,
                )),
            },
            SuperGraphAnyNode::PiperPhonemesToTensor(node) => match slot_index {
                0 => {
                    node.phonemes_input = set_slot_link(link, SuperGraphLinkKind::String);
                    Ok(())
                }
                _ => Err(NodeSlotEditError::invalid_slot_index(
                    "PiperPhonemesToTensor".to_string(),
                    SlotDirection::Input,
                    slot_index,
                    1,
                )),
            },
            SuperGraphAnyNode::KokoroPhonemesToTensor(node) => match slot_index {
                0 => {
                    node.phonemes_input = set_slot_link(link, SuperGraphLinkKind::String);
                    Ok(())
                }
                _ => Err(NodeSlotEditError::invalid_slot_index(
                    "KokoroPhonemesToTensor".to_string(),
                    SlotDirection::Input,
                    slot_index,
                    1,
                )),
            },
            SuperGraphAnyNode::F5TextToTensor(node) => match slot_index {
                0 => {
                    node.text_input = set_slot_link(link, SuperGraphLinkKind::String);
                    Ok(())
                }
                _ => Err(NodeSlotEditError::invalid_slot_index(
                    "F5TextToTensor".to_string(),
                    SlotDirection::Input,
                    slot_index,
                    1,
                )),
            },
            SuperGraphAnyNode::TensorToImage(node) => match slot_index {
                0 => {
                    node.tensor_input = set_slot_link(link, SuperGraphLinkKind::Tensor);
                    Ok(())
                }
                _ => Err(NodeSlotEditError::invalid_slot_index(
                    "TensorToImage".to_string(),
                    SlotDirection::Input,
                    slot_index,
                    1,
                )),
            },
            SuperGraphAnyNode::TensorToAudioClip(node) => match slot_index {
                0 => {
                    node.tensor_input = set_slot_link(link, SuperGraphLinkKind::Tensor);
                    Ok(())
                }
                _ => Err(NodeSlotEditError::invalid_slot_index(
                    "TensorToAudioClip".to_string(),
                    SlotDirection::Input,
                    slot_index,
                    1,
                )),
            },
            SuperGraphAnyNode::TensorToVideoClip(node) => match slot_index {
                0 => {
                    node.tensor_input = set_slot_link(link, SuperGraphLinkKind::Tensor);
                    Ok(())
                }
                _ => Err(NodeSlotEditError::invalid_slot_index(
                    "TensorToVideoClip".to_string(),
                    SlotDirection::Input,
                    slot_index,
                    1,
                )),
            },
            SuperGraphAnyNode::AudioClipToTensor(node) => match slot_index {
                0 => {
                    node.audio_input = set_slot_link(link, SuperGraphLinkKind::AudioClip);
                    Ok(())
                }
                _ => Err(NodeSlotEditError::invalid_slot_index(
                    "AudioClipToTensor".to_string(),
                    SlotDirection::Input,
                    slot_index,
                    1,
                )),
            },
            SuperGraphAnyNode::VideoClipToTensor(node) => match slot_index {
                0 => {
                    node.video_input = set_slot_link(link, SuperGraphLinkKind::VideoClip);
                    Ok(())
                }
                _ => Err(NodeSlotEditError::invalid_slot_index(
                    "VideoClipToTensor".to_string(),
                    SlotDirection::Input,
                    slot_index,
                    1,
                )),
            },
            SuperGraphAnyNode::AudioClipToMelSpectrogram(node) => match slot_index {
                0 => {
                    node.audio_input = set_slot_link(link, SuperGraphLinkKind::AudioClip);
                    Ok(())
                }
                _ => Err(NodeSlotEditError::invalid_slot_index(
                    "AudioClipToMelSpectrogram".to_string(),
                    SlotDirection::Input,
                    slot_index,
                    1,
                )),
            },
            SuperGraphAnyNode::MilliOpGraph(_) => Err(NodeSlotEditError::unsupported(
                "MilliOpGraph".to_string(),
                SlotDirection::Input,
                slot_index,
            )),
            SuperGraphAnyNode::Scan(node) => {
                let mut remaining = slot_index;
                if remaining == 0 {
                    node.iteration_count = set_slot_link(link, SuperGraphLinkKind::Tensor);
                    return Ok(());
                }
                remaining -= 1;

                if remaining < node.simple_inputs.len() {
                    let (outer, inner) = &mut node.simple_inputs[remaining];
                    let like = (*inner).or(*outer);
                    *outer =
                        set_slot_link_like(link, like, "Scan", SlotDirection::Input, slot_index)?;
                    return Ok(());
                }
                remaining -= node.simple_inputs.len();

                if remaining < node.state_links.len() {
                    let (outer, inner, iter_output) = &mut node.state_links[remaining];
                    let like = (*inner).or(*iter_output).or(*outer);
                    *outer =
                        set_slot_link_like(link, like, "Scan", SlotDirection::Input, slot_index)?;
                    return Ok(());
                }
                remaining -= node.state_links.len();

                if remaining < node.scan_inputs.len() {
                    let (outer, inner, _) = &mut node.scan_inputs[remaining];
                    let like = (*inner).or(*outer);
                    *outer =
                        set_slot_link_like(link, like, "Scan", SlotDirection::Input, slot_index)?;
                    return Ok(());
                }

                let slot_count =
                    1 + node.simple_inputs.len() + node.state_links.len() + node.scan_inputs.len();
                Err(NodeSlotEditError::invalid_slot_index(
                    "Scan".to_string(),
                    SlotDirection::Input,
                    slot_index,
                    slot_count,
                ))
            }
            SuperGraphAnyNode::ReportProgress(node) => match slot_index {
                0 => {
                    node.tier_input = set_slot_link(link, SuperGraphLinkKind::Tensor);
                    Ok(())
                }
                1 => {
                    node.numerator_input = set_slot_link(link, SuperGraphLinkKind::Tensor);
                    Ok(())
                }
                2 => {
                    node.denominator_input = set_slot_link(link, SuperGraphLinkKind::Tensor);
                    Ok(())
                }
                _ => Err(NodeSlotEditError::invalid_slot_index(
                    "ReportProgress".to_string(),
                    SlotDirection::Input,
                    slot_index,
                    3,
                )),
            },
            SuperGraphAnyNode::RNNCacheWrite(node) => {
                let slot_count = 2 + node.state_inputs.len();
                if slot_index == 0 {
                    node.key_input = set_slot_link(link, SuperGraphLinkKind::Hash);
                    Ok(())
                } else if slot_index == 1 {
                    node.tokens_input = set_slot_link(link, SuperGraphLinkKind::Tensor);
                    Ok(())
                } else {
                    let i = slot_index - 2;
                    if i < node.state_inputs.len() {
                        node.state_inputs[i].1 = set_slot_link(link, SuperGraphLinkKind::Tensor);
                        Ok(())
                    } else {
                        Err(NodeSlotEditError::invalid_slot_index(
                            "RNNCacheWrite".to_string(),
                            SlotDirection::Input,
                            slot_index,
                            slot_count,
                        ))
                    }
                }
            }
            SuperGraphAnyNode::RNNCacheRead(node) => {
                let slot_count = 2 + node.default_state_inputs.len();
                if slot_index == 0 {
                    node.key_input = set_slot_link(link, SuperGraphLinkKind::Hash);
                    Ok(())
                } else if slot_index == 1 {
                    node.tokens_input = set_slot_link(link, SuperGraphLinkKind::Tensor);
                    Ok(())
                } else {
                    let i = slot_index - 2;
                    if i < node.default_state_inputs.len() {
                        node.default_state_inputs[i].1 =
                            set_slot_link(link, SuperGraphLinkKind::Tensor);
                        Ok(())
                    } else {
                        Err(NodeSlotEditError::invalid_slot_index(
                            "RNNCacheRead".to_string(),
                            SlotDirection::Input,
                            slot_index,
                            slot_count,
                        ))
                    }
                }
            }
            SuperGraphAnyNode::TensorCacheRead(node) => match slot_index {
                0 => {
                    node.key_input = set_slot_link(link, SuperGraphLinkKind::Hash);
                    Ok(())
                }
                1 => {
                    node.default_input = set_slot_link(link, SuperGraphLinkKind::Tensor);
                    Ok(())
                }
                _ => Err(NodeSlotEditError::invalid_slot_index(
                    "TensorCacheRead".to_string(),
                    SlotDirection::Input,
                    slot_index,
                    2,
                )),
            },
            SuperGraphAnyNode::TensorCacheWrite(node) => match slot_index {
                0 => {
                    node.key_input = set_slot_link(link, SuperGraphLinkKind::Hash);
                    Ok(())
                }
                1 => {
                    node.value_input = set_slot_link(link, SuperGraphLinkKind::Tensor);
                    Ok(())
                }
                2 => {
                    node.write_enable_input = set_slot_link(link, SuperGraphLinkKind::Tensor);
                    Ok(())
                }
                _ => Err(NodeSlotEditError::invalid_slot_index(
                    "TensorCacheWrite".to_string(),
                    SlotDirection::Input,
                    slot_index,
                    3,
                )),
            },
            SuperGraphAnyNode::TensorPackCacheRead(node) => {
                let slot_count = 1 + node.default_value_inputs.len();
                if slot_index == 0 {
                    node.key_input = set_slot_link(link, SuperGraphLinkKind::Hash);
                    Ok(())
                } else {
                    let i = slot_index - 1;
                    if i < node.default_value_inputs.len() {
                        node.default_value_inputs[i].1 =
                            set_slot_link(link, SuperGraphLinkKind::Tensor);
                        Ok(())
                    } else {
                        Err(NodeSlotEditError::invalid_slot_index(
                            "TensorPackCacheRead".to_string(),
                            SlotDirection::Input,
                            slot_index,
                            slot_count,
                        ))
                    }
                }
            }
            SuperGraphAnyNode::TensorPackCacheWrite(node) => {
                let slot_count = 2 + node.value_inputs.len();
                if slot_index == 0 {
                    node.key_input = set_slot_link(link, SuperGraphLinkKind::Hash);
                    Ok(())
                } else if slot_index == 1 {
                    node.write_enable_input = set_slot_link(link, SuperGraphLinkKind::Tensor);
                    Ok(())
                } else {
                    let i = slot_index - 2;
                    if i < node.value_inputs.len() {
                        node.value_inputs[i].1 = set_slot_link(link, SuperGraphLinkKind::Tensor);
                        Ok(())
                    } else {
                        Err(NodeSlotEditError::invalid_slot_index(
                            "TensorPackCacheWrite".to_string(),
                            SlotDirection::Input,
                            slot_index,
                            slot_count,
                        ))
                    }
                }
            }
        }
    }
    fn set_output_slot(
        &mut self,
        slot_index: usize,
        link: Option<GlobalId>,
    ) -> Result<(), NodeSlotEditError> {
        match self {
            SuperGraphAnyNode::ModelExecution(node) => {
                let slot_count = node.tensor_outputs.len();
                if slot_index < node.tensor_outputs.len() {
                    node.tensor_outputs[slot_index].1 =
                        set_slot_link(link, SuperGraphLinkKind::Tensor);
                    Ok(())
                } else {
                    Err(NodeSlotEditError::invalid_slot_index(
                        "Model Execution".to_string(),
                        SlotDirection::Output,
                        slot_index,
                        slot_count,
                    ))
                }
            }
            SuperGraphAnyNode::TokenizerLoad(node) => match slot_index {
                0 => {
                    node.output = set_slot_link(link, SuperGraphLinkKind::Tokenizer);
                    Ok(())
                }
                _ => Err(NodeSlotEditError::invalid_slot_index(
                    "Tokenizer Load".to_string(),
                    SlotDirection::Output,
                    slot_index,
                    1,
                )),
            },
            SuperGraphAnyNode::TokenizerEncode(node) => match slot_index {
                0 => {
                    node.tensor_output = set_slot_link(link, SuperGraphLinkKind::Tensor);
                    Ok(())
                }
                _ => Err(NodeSlotEditError::invalid_slot_index(
                    "Tokenizer Encode".to_string(),
                    SlotDirection::Output,
                    slot_index,
                    1,
                )),
            },
            SuperGraphAnyNode::TokenizerDecode(node) => match slot_index {
                0 => {
                    node.text_output = set_slot_link(link, SuperGraphLinkKind::String);
                    Ok(())
                }
                _ => Err(NodeSlotEditError::invalid_slot_index(
                    "Tokenizer Decode".to_string(),
                    SlotDirection::Output,
                    slot_index,
                    1,
                )),
            },
            SuperGraphAnyNode::TextToPhonemes(node) => match slot_index {
                0 => {
                    node.phonemes_output = set_slot_link(link, SuperGraphLinkKind::String);
                    Ok(())
                }
                _ => Err(NodeSlotEditError::invalid_slot_index(
                    "TextToPhonemes".to_string(),
                    SlotDirection::Output,
                    slot_index,
                    1,
                )),
            },
            SuperGraphAnyNode::PiperPhonemesToTensor(node) => match slot_index {
                0 => {
                    node.token_ids_output = set_slot_link(link, SuperGraphLinkKind::Tensor);
                    Ok(())
                }
                1 => {
                    node.input_lengths_output = set_slot_link(link, SuperGraphLinkKind::Tensor);
                    Ok(())
                }
                _ => Err(NodeSlotEditError::invalid_slot_index(
                    "PiperPhonemesToTensor".to_string(),
                    SlotDirection::Output,
                    slot_index,
                    2,
                )),
            },
            SuperGraphAnyNode::KokoroPhonemesToTensor(node) => match slot_index {
                0 => {
                    node.token_ids_output = set_slot_link(link, SuperGraphLinkKind::Tensor);
                    Ok(())
                }
                _ => Err(NodeSlotEditError::invalid_slot_index(
                    "KokoroPhonemesToTensor".to_string(),
                    SlotDirection::Output,
                    slot_index,
                    1,
                )),
            },
            SuperGraphAnyNode::F5TextToTensor(node) => match slot_index {
                0 => {
                    node.token_ids_output = set_slot_link(link, SuperGraphLinkKind::Tensor);
                    Ok(())
                }
                _ => Err(NodeSlotEditError::invalid_slot_index(
                    "F5TextToTensor".to_string(),
                    SlotDirection::Output,
                    slot_index,
                    1,
                )),
            },
            SuperGraphAnyNode::TensorToImage(node) => match slot_index {
                0 => {
                    node.image_output = set_slot_link(link, SuperGraphLinkKind::Image);
                    Ok(())
                }
                _ => Err(NodeSlotEditError::invalid_slot_index(
                    "TensorToImage".to_string(),
                    SlotDirection::Output,
                    slot_index,
                    1,
                )),
            },
            SuperGraphAnyNode::TensorToAudioClip(node) => match slot_index {
                0 => {
                    node.audio_output = set_slot_link(link, SuperGraphLinkKind::AudioClip);
                    Ok(())
                }
                _ => Err(NodeSlotEditError::invalid_slot_index(
                    "TensorToAudioClip".to_string(),
                    SlotDirection::Output,
                    slot_index,
                    1,
                )),
            },
            SuperGraphAnyNode::TensorToVideoClip(node) => match slot_index {
                0 => {
                    node.video_output = set_slot_link(link, SuperGraphLinkKind::VideoClip);
                    Ok(())
                }
                _ => Err(NodeSlotEditError::invalid_slot_index(
                    "TensorToVideoClip".to_string(),
                    SlotDirection::Output,
                    slot_index,
                    1,
                )),
            },
            SuperGraphAnyNode::AudioClipToTensor(node) => match slot_index {
                0 => {
                    node.tensor_output = set_slot_link(link, SuperGraphLinkKind::Tensor);
                    Ok(())
                }
                _ => Err(NodeSlotEditError::invalid_slot_index(
                    "AudioClipToTensor".to_string(),
                    SlotDirection::Output,
                    slot_index,
                    1,
                )),
            },
            SuperGraphAnyNode::VideoClipToTensor(node) => match slot_index {
                0 => {
                    node.tensor_output = set_slot_link(link, SuperGraphLinkKind::Tensor);
                    Ok(())
                }
                _ => Err(NodeSlotEditError::invalid_slot_index(
                    "VideoClipToTensor".to_string(),
                    SlotDirection::Output,
                    slot_index,
                    1,
                )),
            },
            SuperGraphAnyNode::AudioClipToMelSpectrogram(node) => match slot_index {
                0 => {
                    node.tensor_output = set_slot_link(link, SuperGraphLinkKind::Tensor);
                    Ok(())
                }
                _ => Err(NodeSlotEditError::invalid_slot_index(
                    "AudioClipToMelSpectrogram".to_string(),
                    SlotDirection::Output,
                    slot_index,
                    1,
                )),
            },
            SuperGraphAnyNode::MilliOpGraph(_) => Err(NodeSlotEditError::unsupported(
                "MilliOpGraph".to_string(),
                SlotDirection::Output,
                slot_index,
            )),
            SuperGraphAnyNode::Scan(node) => {
                let mut remaining = slot_index;
                if remaining < node.simple_outputs.len() {
                    let (inner, outer) = &mut node.simple_outputs[remaining];
                    let like = (*inner).or(*outer);
                    *outer =
                        set_slot_link_like(link, like, "Scan", SlotDirection::Output, slot_index)?;
                    return Ok(());
                }
                remaining -= node.simple_outputs.len();

                if remaining < node.scan_outputs.len() {
                    let (inner, outer, _) = &mut node.scan_outputs[remaining];
                    let like = (*inner).or(*outer);
                    *outer =
                        set_slot_link_like(link, like, "Scan", SlotDirection::Output, slot_index)?;
                    return Ok(());
                }

                let slot_count = node.simple_outputs.len() + node.scan_outputs.len();
                Err(NodeSlotEditError::invalid_slot_index(
                    "Scan".to_string(),
                    SlotDirection::Output,
                    slot_index,
                    slot_count,
                ))
            }
            SuperGraphAnyNode::ReportProgress(_) => Err(NodeSlotEditError::invalid_slot_index(
                "ReportProgress".to_string(),
                SlotDirection::Output,
                slot_index,
                0,
            )),
            SuperGraphAnyNode::RNNCacheWrite(_) => Err(NodeSlotEditError::invalid_slot_index(
                "RNNCacheWrite".to_string(),
                SlotDirection::Output,
                slot_index,
                0,
            )),
            SuperGraphAnyNode::RNNCacheRead(node) => {
                let slot_count = 1 + node.state_outputs.len();
                if slot_index == 0 {
                    node.tokens_output = set_slot_link(link, SuperGraphLinkKind::Tensor);
                    Ok(())
                } else {
                    let i = slot_index - 1;
                    if i < node.state_outputs.len() {
                        node.state_outputs[i].1 = set_slot_link(link, SuperGraphLinkKind::Tensor);
                        Ok(())
                    } else {
                        Err(NodeSlotEditError::invalid_slot_index(
                            "RNNCacheRead".to_string(),
                            SlotDirection::Output,
                            slot_index,
                            slot_count,
                        ))
                    }
                }
            }
            SuperGraphAnyNode::TensorCacheRead(node) => match slot_index {
                0 => {
                    node.value_output = set_slot_link(link, SuperGraphLinkKind::Tensor);
                    Ok(())
                }
                1 => {
                    node.hit_output = set_slot_link(link, SuperGraphLinkKind::Tensor);
                    Ok(())
                }
                _ => Err(NodeSlotEditError::invalid_slot_index(
                    "TensorCacheRead".to_string(),
                    SlotDirection::Output,
                    slot_index,
                    2,
                )),
            },
            SuperGraphAnyNode::TensorCacheWrite(_) => Err(NodeSlotEditError::invalid_slot_index(
                "TensorCacheWrite".to_string(),
                SlotDirection::Output,
                slot_index,
                0,
            )),
            SuperGraphAnyNode::TensorPackCacheRead(node) => {
                let slot_count = node.value_outputs.len() + 1;
                if slot_index < node.value_outputs.len() {
                    node.value_outputs[slot_index].1 =
                        set_slot_link(link, SuperGraphLinkKind::Tensor);
                    Ok(())
                } else if slot_index == node.value_outputs.len() {
                    node.hit_output = set_slot_link(link, SuperGraphLinkKind::Tensor);
                    Ok(())
                } else {
                    Err(NodeSlotEditError::invalid_slot_index(
                        "TensorPackCacheRead".to_string(),
                        SlotDirection::Output,
                        slot_index,
                        slot_count,
                    ))
                }
            }
            SuperGraphAnyNode::TensorPackCacheWrite(_) => {
                Err(NodeSlotEditError::invalid_slot_index(
                    "TensorPackCacheWrite".to_string(),
                    SlotDirection::Output,
                    slot_index,
                    0,
                ))
            }
        }
    }
    delegate!(global_id() -> GlobalId);
}

impl NodeMetadata for SuperGraphAnyNode {
    fn parameters(&self) -> Vec<Property> {
        match self {
            SuperGraphAnyNode::ModelExecution(node) => {
                vec![Property::new(
                    "symbolic_graph_id",
                    PropertyValue::Int(node.symbolic_graph_id as i64),
                )]
            }
            SuperGraphAnyNode::Scan(node) => {
                vec![Property::new(
                    "num_scan_inputs",
                    PropertyValue::Int(node.scan_inputs.len() as i64),
                )]
            }
            _ => Vec::new(),
        }
    }

    fn has_subgraph(&self) -> bool {
        matches!(self, SuperGraphAnyNode::Scan(_))
    }
}
