//! Generic schedule-intent recovery from whitewashed nano-op pools.
//!
//! This module is intentionally pattern-agnostic at the op level. It groups
//! stores by reconstructed expression families and infers high-level intent
//! (pointwise vs additive reduction) from structure and index behavior.

use crate::compiler::common::v1_frontend::nano_op::{
    NanoExpandError, NanoOp, NanoOpExpander, NanoValue, ScalarBinOp, ScalarUnaryOp,
};
use crate::graph::GlobalId;
use crate::milli_graph::MilliOpGraph;
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error(transparent)]
    Expand(#[from] NanoExpandError),
    #[error("Duplicate definition for nano value {0:?}")]
    DuplicateDefinition(NanoValue),
    #[error("Missing definition for nano value {0:?}")]
    MissingDefinition(NanoValue),
    #[error("Missing shape for tensor {0}")]
    MissingShape(GlobalId),
}

#[derive(Debug, Clone, Default)]
pub struct ScheduleStats {
    pub total_nano_ops: usize,
    pub total_stores: usize,
    pub grouped_families: usize,
    pub pointwise_families: usize,
    pub additive_reduction_families: usize,
    pub unknown_families: usize,
}

#[derive(Debug, Clone)]
pub struct ScheduleIR {
    pub loops: Vec<RecoveredLoop>,
    pub stats: ScheduleStats,
}

#[derive(Debug, Clone)]
pub struct RecoveredLoop {
    pub output_tensor: GlobalId,
    pub output_shape: Vec<usize>,
    pub covered_outputs: usize,
    pub load_tensors: Vec<GlobalId>,
    pub intent: LoopIntent,
}

#[derive(Debug, Clone)]
pub enum LoopIntent {
    Pointwise { contiguous_output: bool },
    AdditiveReduction(ReductionIntent),
    Unknown,
}

#[derive(Debug, Clone)]
pub struct ReductionIntent {
    pub terms: usize,
    pub canonical_term: ReductionTermPattern,
    pub term_load_tensors: Vec<GlobalId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ReductionTermPattern {
    Load(usize),
    Literal(u64),
    Bin {
        op: ReductionBinOp,
        a: Box<ReductionTermPattern>,
        b: Box<ReductionTermPattern>,
    },
    Unary {
        op: ReductionUnaryOp,
        input: Box<ReductionTermPattern>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReductionBinOp {
    Add,
    Sub,
    Mul,
    Div,
    Max,
    Min,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReductionUnaryOp {
    Neg,
    Abs,
    Exp,
    Ln,
    Sqrt,
    Reciprocal,
    Tanh,
    Floor,
    Ceil,
}

#[derive(Debug, Clone)]
pub struct PipelineArtifacts {
    pub ordered_nano_ops: Vec<NanoOp>,
    pub whitewashed_nano_ops: Vec<NanoOp>,
    pub schedule: ScheduleIR,
}

pub fn build_from_graph(
    graph: &MilliOpGraph,
    shapes: &HashMap<GlobalId, Vec<usize>>,
) -> Result<PipelineArtifacts, PipelineError> {
    let mut expander = NanoOpExpander::new(shapes.clone());
    let ordered_nano_ops = expander.expand_iter(graph).collect::<Result<Vec<_>, _>>()?;
    let whitewashed_nano_ops = whitewash_pool_order(ordered_nano_ops.clone());
    let schedule = analyze_from_pool(&whitewashed_nano_ops, shapes)?;

    Ok(PipelineArtifacts {
        ordered_nano_ops,
        whitewashed_nano_ops,
        schedule,
    })
}

pub fn analyze_from_pool(
    nano_ops: &[NanoOp],
    shapes: &HashMap<GlobalId, Vec<usize>>,
) -> Result<ScheduleIR, PipelineError> {
    let defs = build_defs(nano_ops)?;
    let mut expr_cache = HashMap::new();
    let mut raw_groups: HashMap<GroupKey, StoreGroup> = HashMap::new();
    let mut total_stores = 0usize;

    for op in nano_ops {
        let (output_tensor, out_flat_index, src) = match op {
            NanoOp::Store {
                tensor,
                flat_index,
                src,
            } => (*tensor, *flat_index, *src),
            _ => continue,
        };

        total_stores += 1;
        let expr = build_expr(src, &defs, &mut expr_cache)?;

        let mut load_tensors = Vec::new();
        let pattern = pattern_expr(&expr, &mut load_tensors);
        let mut load_indices = Vec::new();
        collect_load_indices(&expr, &mut load_indices);

        let key = GroupKey {
            output_tensor,
            pattern_hash: stable_hash(&pattern),
            load_tensors_hash: stable_hash(&load_tensors),
        };

        let entry = raw_groups.entry(key).or_insert_with(|| StoreGroup {
            output_tensor,
            pattern: pattern.clone(),
            load_tensors: load_tensors.clone(),
            instances: Vec::new(),
        });

        // Hash collision fallback.
        if entry.pattern != pattern || entry.load_tensors != load_tensors {
            let mut alt_key = key;
            alt_key.pattern_hash ^= 0x9E37_79B9_7F4A_7C15;
            raw_groups
                .entry(alt_key)
                .or_insert_with(|| StoreGroup {
                    output_tensor,
                    pattern,
                    load_tensors,
                    instances: Vec::new(),
                })
                .instances
                .push(StoreInstance {
                    out_flat_index,
                    load_indices,
                });
            continue;
        }

        entry.instances.push(StoreInstance {
            out_flat_index,
            load_indices,
        });
    }

    let mut loops = Vec::new();
    let mut grouped: Vec<_> = raw_groups.into_values().collect();
    grouped.sort_by_key(|g| g.output_tensor);

    let mut pointwise = 0usize;
    let mut reductions = 0usize;
    let mut unknown = 0usize;

    for group in grouped {
        let recovered = classify_group(&group, shapes)?;
        match &recovered.intent {
            LoopIntent::Pointwise { .. } => pointwise += 1,
            LoopIntent::AdditiveReduction(_) => reductions += 1,
            LoopIntent::Unknown => unknown += 1,
        }
        loops.push(recovered);
    }

    let stats = ScheduleStats {
        total_nano_ops: nano_ops.len(),
        total_stores,
        grouped_families: loops.len(),
        pointwise_families: pointwise,
        additive_reduction_families: reductions,
        unknown_families: unknown,
    };

    Ok(ScheduleIR { loops, stats })
}

#[derive(Debug, Clone)]
enum Expr {
    Load {
        tensor: GlobalId,
        flat_index: usize,
    },
    Literal {
        value_bits: u64,
    },
    Bin {
        op: ScalarBinOp,
        a: Box<Expr>,
        b: Box<Expr>,
    },
    Unary {
        op: ScalarUnaryOp,
        input: Box<Expr>,
    },
}

#[derive(Debug, Clone)]
struct StoreInstance {
    out_flat_index: usize,
    load_indices: Vec<usize>,
}

#[derive(Debug, Clone)]
struct StoreGroup {
    output_tensor: GlobalId,
    pattern: PatternExpr,
    load_tensors: Vec<GlobalId>,
    instances: Vec<StoreInstance>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum PatternExpr {
    Load(usize),
    Literal(u64),
    Bin {
        op: BinTag,
        a: Box<PatternExpr>,
        b: Box<PatternExpr>,
    },
    Unary {
        op: UnaryTag,
        input: Box<PatternExpr>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum BinTag {
    Add,
    Sub,
    Mul,
    Div,
    Max,
    Min,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum UnaryTag {
    Neg,
    Abs,
    Exp,
    Ln,
    Sqrt,
    Reciprocal,
    Tanh,
    Floor,
    Ceil,
}

impl From<ScalarBinOp> for BinTag {
    fn from(value: ScalarBinOp) -> Self {
        match value {
            ScalarBinOp::Add => BinTag::Add,
            ScalarBinOp::Sub => BinTag::Sub,
            ScalarBinOp::Mul => BinTag::Mul,
            ScalarBinOp::Div => BinTag::Div,
            ScalarBinOp::Max => BinTag::Max,
            ScalarBinOp::Min => BinTag::Min,
        }
    }
}

impl From<ScalarUnaryOp> for UnaryTag {
    fn from(value: ScalarUnaryOp) -> Self {
        match value {
            ScalarUnaryOp::Neg => UnaryTag::Neg,
            ScalarUnaryOp::Abs => UnaryTag::Abs,
            ScalarUnaryOp::Exp => UnaryTag::Exp,
            ScalarUnaryOp::Ln => UnaryTag::Ln,
            ScalarUnaryOp::Sqrt => UnaryTag::Sqrt,
            ScalarUnaryOp::Reciprocal => UnaryTag::Reciprocal,
            ScalarUnaryOp::Tanh => UnaryTag::Tanh,
            ScalarUnaryOp::Floor => UnaryTag::Floor,
            ScalarUnaryOp::Ceil => UnaryTag::Ceil,
        }
    }
}

impl From<BinTag> for ReductionBinOp {
    fn from(value: BinTag) -> Self {
        match value {
            BinTag::Add => ReductionBinOp::Add,
            BinTag::Sub => ReductionBinOp::Sub,
            BinTag::Mul => ReductionBinOp::Mul,
            BinTag::Div => ReductionBinOp::Div,
            BinTag::Max => ReductionBinOp::Max,
            BinTag::Min => ReductionBinOp::Min,
        }
    }
}

impl From<UnaryTag> for ReductionUnaryOp {
    fn from(value: UnaryTag) -> Self {
        match value {
            UnaryTag::Neg => ReductionUnaryOp::Neg,
            UnaryTag::Abs => ReductionUnaryOp::Abs,
            UnaryTag::Exp => ReductionUnaryOp::Exp,
            UnaryTag::Ln => ReductionUnaryOp::Ln,
            UnaryTag::Sqrt => ReductionUnaryOp::Sqrt,
            UnaryTag::Reciprocal => ReductionUnaryOp::Reciprocal,
            UnaryTag::Tanh => ReductionUnaryOp::Tanh,
            UnaryTag::Floor => ReductionUnaryOp::Floor,
            UnaryTag::Ceil => ReductionUnaryOp::Ceil,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct GroupKey {
    output_tensor: GlobalId,
    pattern_hash: u64,
    load_tensors_hash: u64,
}

fn stable_hash<T: Hash>(value: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

/// Deterministic, structure-destroying reorder for pool-style recovery tests.
fn whitewash_pool_order(mut ops: Vec<NanoOp>) -> Vec<NanoOp> {
    if ops.len() < 3 {
        return ops;
    }
    let len = ops.len();
    ops.rotate_left(len / 3);
    for chunk in ops.chunks_mut(7) {
        chunk.reverse();
    }
    ops
}

fn build_defs(nano_ops: &[NanoOp]) -> Result<HashMap<NanoValue, NanoOp>, PipelineError> {
    let mut defs = HashMap::new();
    for op in nano_ops {
        if let Some(dst) = op.dst() {
            if defs.contains_key(&dst) {
                return Err(PipelineError::DuplicateDefinition(dst));
            }
            defs.insert(dst, op.clone());
        }
    }
    Ok(defs)
}

fn build_expr(
    value: NanoValue,
    defs: &HashMap<NanoValue, NanoOp>,
    cache: &mut HashMap<NanoValue, Expr>,
) -> Result<Expr, PipelineError> {
    if let Some(expr) = cache.get(&value) {
        return Ok(expr.clone());
    }

    let op = defs
        .get(&value)
        .ok_or(PipelineError::MissingDefinition(value))?;

    let expr = match op {
        NanoOp::Load {
            tensor, flat_index, ..
        } => Expr::Load {
            tensor: *tensor,
            flat_index: *flat_index,
        },
        NanoOp::Literal { value, .. } => Expr::Literal {
            value_bits: value.to_bits(),
        },
        NanoOp::BinOp { op, a, b, .. } => {
            let a_expr = build_expr(*a, defs, cache)?;
            let b_expr = build_expr(*b, defs, cache)?;
            Expr::Bin {
                op: *op,
                a: Box::new(a_expr),
                b: Box::new(b_expr),
            }
        }
        NanoOp::UnaryOp { op, input, .. } => {
            let input_expr = build_expr(*input, defs, cache)?;
            Expr::Unary {
                op: *op,
                input: Box::new(input_expr),
            }
        }
        NanoOp::Store { .. } => return Err(PipelineError::MissingDefinition(value)),
    };

    cache.insert(value, expr.clone());
    Ok(expr)
}

fn pattern_expr(expr: &Expr, load_tensors: &mut Vec<GlobalId>) -> PatternExpr {
    match expr {
        Expr::Load { tensor, .. } => {
            let slot = load_tensors.len();
            load_tensors.push(*tensor);
            PatternExpr::Load(slot)
        }
        Expr::Literal { value_bits } => PatternExpr::Literal(*value_bits),
        Expr::Bin { op, a, b } => PatternExpr::Bin {
            op: (*op).into(),
            a: Box::new(pattern_expr(a, load_tensors)),
            b: Box::new(pattern_expr(b, load_tensors)),
        },
        Expr::Unary { op, input } => PatternExpr::Unary {
            op: (*op).into(),
            input: Box::new(pattern_expr(input, load_tensors)),
        },
    }
}

fn collect_load_indices(expr: &Expr, out: &mut Vec<usize>) {
    match expr {
        Expr::Load { flat_index, .. } => out.push(*flat_index),
        Expr::Literal { .. } => {}
        Expr::Bin { a, b, .. } => {
            collect_load_indices(a, out);
            collect_load_indices(b, out);
        }
        Expr::Unary { input, .. } => collect_load_indices(input, out),
    }
}

fn classify_group(
    group: &StoreGroup,
    shapes: &HashMap<GlobalId, Vec<usize>>,
) -> Result<RecoveredLoop, PipelineError> {
    let output_shape = shapes
        .get(&group.output_tensor)
        .cloned()
        .ok_or(PipelineError::MissingShape(group.output_tensor))?;
    let output_elements = output_shape.iter().product::<usize>();

    let intent = if let Some(contiguous_output) = is_simple_pointwise(group, output_elements) {
        LoopIntent::Pointwise { contiguous_output }
    } else if let Some(reduction) = try_classify_additive_reduction(group) {
        LoopIntent::AdditiveReduction(reduction)
    } else {
        LoopIntent::Unknown
    };

    Ok(RecoveredLoop {
        output_tensor: group.output_tensor,
        output_shape,
        covered_outputs: group.instances.len(),
        load_tensors: group.load_tensors.clone(),
        intent,
    })
}

fn is_simple_pointwise(group: &StoreGroup, output_elements: usize) -> Option<bool> {
    if group.instances.len() < 2 {
        return None;
    }

    let mut inst = group.instances.clone();
    inst.sort_by_key(|i| i.out_flat_index);

    let out_step = (inst[1].out_flat_index as isize) - (inst[0].out_flat_index as isize);
    if !inst
        .windows(2)
        .all(|w| (w[1].out_flat_index as isize) - (w[0].out_flat_index as isize) == out_step)
    {
        return None;
    }

    let num_slots = group.load_tensors.len();
    for slot in 0..num_slots {
        if inst[0].load_indices.len() <= slot || inst[1].load_indices.len() <= slot {
            return None;
        }

        let step = (inst[1].load_indices[slot] as isize) - (inst[0].load_indices[slot] as isize);
        if !inst.windows(2).all(|w| {
            (w[1].load_indices[slot] as isize) - (w[0].load_indices[slot] as isize) == step
        }) {
            return None;
        }
    }

    let contiguous_output = inst[0].out_flat_index == 0
        && out_step == 1
        && inst.len() == output_elements
        && inst.iter().enumerate().all(|(i, x)| x.out_flat_index == i);

    Some(contiguous_output)
}

fn try_classify_additive_reduction(group: &StoreGroup) -> Option<ReductionIntent> {
    let mut terms = Vec::new();
    collect_add_terms(&group.pattern, &mut terms);
    if terms.len() < 2 {
        return None;
    }

    let mut first_tensors = Vec::new();
    let mut first_slot_map = HashMap::new();
    let first_pattern =
        canonicalize_term(terms[0], group, &mut first_slot_map, &mut first_tensors)?;

    for term in terms.iter().skip(1) {
        let mut tensors = Vec::new();
        let mut slot_map = HashMap::new();
        let pattern = canonicalize_term(term, group, &mut slot_map, &mut tensors)?;
        if pattern != first_pattern || tensors != first_tensors {
            return None;
        }
    }

    Some(ReductionIntent {
        terms: terms.len(),
        canonical_term: first_pattern,
        term_load_tensors: first_tensors,
    })
}

fn collect_add_terms<'a>(expr: &'a PatternExpr, out: &mut Vec<&'a PatternExpr>) {
    match expr {
        PatternExpr::Bin {
            op: BinTag::Add,
            a,
            b,
        } => {
            collect_add_terms(a, out);
            collect_add_terms(b, out);
        }
        PatternExpr::Literal(v) if *v == 0f64.to_bits() => {}
        _ => out.push(expr),
    }
}

fn canonicalize_term(
    expr: &PatternExpr,
    group: &StoreGroup,
    slot_map: &mut HashMap<usize, usize>,
    tensors: &mut Vec<GlobalId>,
) -> Option<ReductionTermPattern> {
    match expr {
        PatternExpr::Load(slot) => {
            let idx = if let Some(&idx) = slot_map.get(slot) {
                idx
            } else {
                let idx = tensors.len();
                tensors.push(*group.load_tensors.get(*slot)?);
                slot_map.insert(*slot, idx);
                idx
            };
            Some(ReductionTermPattern::Load(idx))
        }
        PatternExpr::Literal(bits) => Some(ReductionTermPattern::Literal(*bits)),
        PatternExpr::Bin { op, a, b } => Some(ReductionTermPattern::Bin {
            op: (*op).into(),
            a: Box::new(canonicalize_term(a, group, slot_map, tensors)?),
            b: Box::new(canonicalize_term(b, group, slot_map, tensors)?),
        }),
        PatternExpr::Unary { op, input } => Some(ReductionTermPattern::Unary {
            op: (*op).into(),
            input: Box::new(canonicalize_term(input, group, slot_map, tensors)?),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GlobalId;
    use crate::milli_graph::ops::{MatMul, SimpleBinary};

    #[test]
    fn test_classifies_pointwise_add_as_pointwise() {
        let mut rng = wyrand::WyRand::new(11);
        let ext_a = GlobalId::new(&mut rng);
        let ext_b = GlobalId::new(&mut rng);
        let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b], &mut rng);
        let a = input_map[&ext_a];
        let b = input_map[&ext_b];
        let c = SimpleBinary::add(&mut graph, a, b, &mut rng);

        let mut shapes = HashMap::new();
        shapes.insert(a, vec![16]);
        shapes.insert(b, vec![16]);
        shapes.insert(c, vec![16]);

        let artifacts = build_from_graph(&graph, &shapes).unwrap();
        assert_eq!(artifacts.schedule.loops.len(), 1);
        assert_eq!(artifacts.schedule.stats.pointwise_families, 1);
        assert_eq!(artifacts.schedule.stats.additive_reduction_families, 0);

        match &artifacts.schedule.loops[0].intent {
            LoopIntent::Pointwise { contiguous_output } => assert!(*contiguous_output),
            _ => panic!("expected pointwise intent"),
        }
    }

    #[test]
    fn test_classifies_matmul_as_additive_reduction_without_matmul_match() {
        let mut rng = wyrand::WyRand::new(12);
        let ext_a = GlobalId::new(&mut rng);
        let ext_b = GlobalId::new(&mut rng);
        let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b], &mut rng);
        let a = input_map[&ext_a];
        let b = input_map[&ext_b];
        let c = MatMul::push_new(&mut graph, a, b, &mut rng);

        let mut shapes = HashMap::new();
        shapes.insert(a, vec![4, 8]);
        shapes.insert(b, vec![8, 3]);
        shapes.insert(c, vec![4, 3]);

        let artifacts = build_from_graph(&graph, &shapes).unwrap();
        assert_eq!(artifacts.schedule.loops.len(), 1);
        assert_eq!(artifacts.schedule.stats.additive_reduction_families, 1);
        assert_eq!(artifacts.schedule.stats.pointwise_families, 0);

        let loop0 = &artifacts.schedule.loops[0];
        match &loop0.intent {
            LoopIntent::AdditiveReduction(red) => {
                assert_eq!(red.terms, 8);
                assert_eq!(red.term_load_tensors.len(), 2);
                assert_ne!(red.term_load_tensors[0], red.term_load_tensors[1]);
            }
            _ => panic!("expected additive reduction intent"),
        }
    }
}
