//! Recover loop crystals from an unordered nano-op pool.
//!
//! This pass:
//! 1) Rebuilds expression trees per store from SSA value definitions.
//! 2) Groups structurally-identical stores into loop candidates.
//! 3) Recognizes matmul reductions from nano expressions (no milli-op hints).
//! 4) Topologically orders recovered loops and applies v3 loop fusion.

use crate::compiler::attempts::v1_scalar_crystal::crystal::{CrystalLoop, CrystalOp, LoopBodyOp};
use crate::compiler::attempts::v1_scalar_crystal::nano_op::{
    NanoOp, NanoValue, ScalarBinOp, ScalarUnaryOp,
};
use crate::compiler::attempts::v3_nano_fusion::fusion as v3_fusion;
use crate::graph::GlobalId;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

#[derive(Debug, thiserror::Error)]
pub enum GrowthError {
    #[error("Duplicate definition for nano value {0:?}")]
    DuplicateDefinition(NanoValue),
    #[error("Missing definition for nano value {0:?}")]
    MissingDefinition(NanoValue),
}

#[derive(Debug, Clone)]
pub struct MatMulCrystal {
    pub a: GlobalId,
    pub b: GlobalId,
    pub c: GlobalId,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    /// Flat output indices in C covered by this crystal.
    pub covered_outputs: Vec<usize>,
}

#[derive(Debug, Clone, Default)]
pub struct PoolGrowthStats {
    pub total_nano_ops: usize,
    pub total_stores: usize,
    pub recovered_loop_groups: usize,
    pub recovered_matmul_groups: usize,
    pub ordered_loops: usize,
    pub fused_loops: usize,
    pub fused_pairs: usize,
    pub eliminated_loads: usize,
}

#[derive(Debug, Clone)]
pub struct PoolGrowthPlan {
    /// Loops recovered from pool grouping, before fusion.
    pub recovered_loops: Vec<CrystalOp>,
    /// Recovered loops ordered by tensor dependency.
    pub ordered_loops: Vec<CrystalOp>,
    /// Ordered loops after v3-style absorption/fusion.
    pub fused_loops: Vec<CrystalOp>,
    /// Matmul crystals recognized directly from nano expressions.
    pub matmul_crystals: Vec<MatMulCrystal>,
    pub stats: PoolGrowthStats,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct GroupKey {
    output_tensor: GlobalId,
    pattern_hash: u64,
    load_tensors_hash: u64,
}

pub fn grow_from_pool(
    nano_ops: &[NanoOp],
    tensor_shapes: &HashMap<GlobalId, Vec<usize>>,
) -> Result<PoolGrowthPlan, GrowthError> {
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

        // Collision safety: verify full structural match.
        if entry.pattern != pattern || entry.load_tensors != load_tensors {
            // Extremely unlikely hash collision; isolate by synthetic key perturbation.
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

    let mut recovered_loops = Vec::new();
    let mut matmul_crystals = Vec::new();

    for group in raw_groups.values() {
        if let Some(mm) = try_match_matmul(group, tensor_shapes)? {
            matmul_crystals.push(mm);
            continue;
        }

        if let Some(loop_op) = build_elementwise_loop(group) {
            recovered_loops.push(CrystalOp::Loop(loop_op));
        }
    }

    let ordered_loops = topo_order_loops(&recovered_loops);
    let (fused_loops, fuse_stats) = v3_fusion::fuse_with_stats(&ordered_loops);

    let stats = PoolGrowthStats {
        total_nano_ops: nano_ops.len(),
        total_stores,
        recovered_loop_groups: recovered_loops.len(),
        recovered_matmul_groups: matmul_crystals.len(),
        ordered_loops: ordered_loops.len(),
        fused_loops: fused_loops
            .iter()
            .filter(|op| matches!(op, CrystalOp::Loop(_)))
            .count(),
        fused_pairs: fuse_stats.fused_pairs,
        eliminated_loads: fuse_stats.eliminated_loads,
    };

    Ok(PoolGrowthPlan {
        recovered_loops,
        ordered_loops,
        fused_loops,
        matmul_crystals,
        stats,
    })
}

fn build_defs(nano_ops: &[NanoOp]) -> Result<HashMap<NanoValue, NanoOp>, GrowthError> {
    let mut defs = HashMap::new();
    for op in nano_ops {
        if let Some(dst) = op.dst() {
            if defs.contains_key(&dst) {
                return Err(GrowthError::DuplicateDefinition(dst));
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
) -> Result<Expr, GrowthError> {
    if let Some(expr) = cache.get(&value) {
        return Ok(expr.clone());
    }

    let op = defs
        .get(&value)
        .ok_or(GrowthError::MissingDefinition(value))?;

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
        NanoOp::Store { .. } => return Err(GrowthError::MissingDefinition(value)),
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

fn build_elementwise_loop(group: &StoreGroup) -> Option<CrystalLoop> {
    if group.instances.len() < 2 {
        return None;
    }

    let mut inst = group.instances.clone();
    inst.sort_by_key(|i| i.out_flat_index);

    let out_base = inst[0].out_flat_index;
    let out_step = (inst[1].out_flat_index as isize) - (inst[0].out_flat_index as isize);
    if !inst
        .windows(2)
        .all(|w| (w[1].out_flat_index as isize) - (w[0].out_flat_index as isize) == out_step)
    {
        return None;
    }

    let num_slots = group.load_tensors.len();
    let mut load_base = vec![0usize; num_slots];
    let mut load_step = vec![0isize; num_slots];
    for slot in 0..num_slots {
        if inst[0].load_indices.len() <= slot || inst[1].load_indices.len() <= slot {
            return None;
        }
        load_base[slot] = inst[0].load_indices[slot];
        load_step[slot] =
            (inst[1].load_indices[slot] as isize) - (inst[0].load_indices[slot] as isize);
        if !inst.windows(2).all(|w| {
            (w[1].load_indices[slot] as isize) - (w[0].load_indices[slot] as isize)
                == load_step[slot]
        }) {
            return None;
        }
    }

    let mut body = Vec::new();
    let value_ref = emit_pattern_body(
        &group.pattern,
        &group.load_tensors,
        &load_base,
        &load_step,
        &mut body,
    );

    body.push(LoopBodyOp::Store {
        tensor: group.output_tensor,
        base_index: out_base,
        step: out_step,
        value_ref,
    });

    Some(CrystalLoop {
        nano_ops: Vec::new(),
        count: inst.len(),
        body_len: body.len(),
        body,
    })
}

fn emit_pattern_body(
    pattern: &PatternExpr,
    load_tensors: &[GlobalId],
    load_base: &[usize],
    load_step: &[isize],
    body: &mut Vec<LoopBodyOp>,
) -> usize {
    match pattern {
        PatternExpr::Load(slot) => {
            let idx = body.len();
            body.push(LoopBodyOp::Load {
                tensor: load_tensors[*slot],
                base_index: load_base[*slot],
                step: load_step[*slot],
            });
            idx
        }
        PatternExpr::Literal(value_bits) => {
            let idx = body.len();
            body.push(LoopBodyOp::Literal {
                value: f64::from_bits(*value_bits),
            });
            idx
        }
        PatternExpr::Bin { op, a, b } => {
            let a_ref = emit_pattern_body(a, load_tensors, load_base, load_step, body);
            let b_ref = emit_pattern_body(b, load_tensors, load_base, load_step, body);
            let idx = body.len();
            body.push(LoopBodyOp::BinOp {
                op: match op {
                    BinTag::Add => ScalarBinOp::Add,
                    BinTag::Sub => ScalarBinOp::Sub,
                    BinTag::Mul => ScalarBinOp::Mul,
                    BinTag::Div => ScalarBinOp::Div,
                    BinTag::Max => ScalarBinOp::Max,
                    BinTag::Min => ScalarBinOp::Min,
                },
                a_ref,
                b_ref,
            });
            idx
        }
        PatternExpr::Unary { op, input } => {
            let input_ref = emit_pattern_body(input, load_tensors, load_base, load_step, body);
            let idx = body.len();
            body.push(LoopBodyOp::UnaryOp {
                op: match op {
                    UnaryTag::Neg => ScalarUnaryOp::Neg,
                    UnaryTag::Abs => ScalarUnaryOp::Abs,
                    UnaryTag::Exp => ScalarUnaryOp::Exp,
                    UnaryTag::Ln => ScalarUnaryOp::Ln,
                    UnaryTag::Sqrt => ScalarUnaryOp::Sqrt,
                    UnaryTag::Reciprocal => ScalarUnaryOp::Reciprocal,
                    UnaryTag::Tanh => ScalarUnaryOp::Tanh,
                    UnaryTag::Floor => ScalarUnaryOp::Floor,
                    UnaryTag::Ceil => ScalarUnaryOp::Ceil,
                },
                input_ref,
            });
            idx
        }
    }
}

fn try_match_matmul(
    group: &StoreGroup,
    shapes: &HashMap<GlobalId, Vec<usize>>,
) -> Result<Option<MatMulCrystal>, GrowthError> {
    let terms = match parse_sum_of_products(&group.pattern) {
        Some(t) if !t.is_empty() => t,
        _ => return Ok(None),
    };

    let (a_tensor, b_tensor, orientation) = {
        let (lhs_slot, rhs_slot) = terms[0];
        let lhs_tensor = group.load_tensors[lhs_slot];
        let rhs_tensor = group.load_tensors[rhs_slot];
        if looks_like_matmul_tensors(lhs_tensor, rhs_tensor, group.output_tensor, shapes) {
            (lhs_tensor, rhs_tensor, 0u8)
        } else if looks_like_matmul_tensors(rhs_tensor, lhs_tensor, group.output_tensor, shapes) {
            (rhs_tensor, lhs_tensor, 1u8)
        } else {
            return Ok(None);
        }
    };

    let a_shape = match get_shape_2d(a_tensor, shapes) {
        Some(s) => s,
        None => return Ok(None),
    };
    let b_shape = match get_shape_2d(b_tensor, shapes) {
        Some(s) => s,
        None => return Ok(None),
    };
    let c_shape = match get_shape_2d(group.output_tensor, shapes) {
        Some(s) => s,
        None => return Ok(None),
    };
    let (m, k) = (a_shape[0], a_shape[1]);
    let (k2, n) = (b_shape[0], b_shape[1]);
    if k != k2 || c_shape != [m, n] {
        return Ok(None);
    }
    if terms.len() != k {
        return Ok(None);
    }

    let mut covered_outputs = Vec::new();
    for inst in &group.instances {
        let c_idx = inst.out_flat_index;
        if c_idx >= m * n {
            return Ok(None);
        }
        let row = c_idx / n;
        let col = c_idx % n;

        let mut seen = vec![false; k];
        for (lhs_slot, rhs_slot) in &terms {
            let (a_slot, b_slot) = if orientation == 0 {
                (*lhs_slot, *rhs_slot)
            } else {
                (*rhs_slot, *lhs_slot)
            };

            let a_idx = inst.load_indices[a_slot];
            let b_idx = inst.load_indices[b_slot];

            if a_idx < row * k || a_idx >= (row + 1) * k {
                return Ok(None);
            }
            let kk = a_idx - row * k;
            if kk >= k || b_idx != kk * n + col {
                return Ok(None);
            }
            seen[kk] = true;
        }
        if seen.iter().any(|s| !*s) {
            return Ok(None);
        }
        covered_outputs.push(c_idx);
    }

    covered_outputs.sort_unstable();
    covered_outputs.dedup();

    Ok(Some(MatMulCrystal {
        a: a_tensor,
        b: b_tensor,
        c: group.output_tensor,
        m,
        n,
        k,
        covered_outputs,
    }))
}

fn looks_like_matmul_tensors(
    a: GlobalId,
    b: GlobalId,
    c: GlobalId,
    shapes: &HashMap<GlobalId, Vec<usize>>,
) -> bool {
    let a_shape = match get_shape_2d(a, shapes) {
        Some(s) => s,
        None => return false,
    };
    let b_shape = match get_shape_2d(b, shapes) {
        Some(s) => s,
        None => return false,
    };
    let c_shape = match get_shape_2d(c, shapes) {
        Some(s) => s,
        None => return false,
    };
    a_shape[1] == b_shape[0] && c_shape == [a_shape[0], b_shape[1]]
}

fn get_shape_2d(tensor: GlobalId, shapes: &HashMap<GlobalId, Vec<usize>>) -> Option<[usize; 2]> {
    let shape = shapes.get(&tensor)?;
    if shape.len() != 2 {
        return None;
    }
    Some([shape[0], shape[1]])
}

/// Parses expression as sum of products:
///   term := load(slot) * load(slot)
///   expr := term (+ term)+, optionally with literal 0 as additive identity.
fn parse_sum_of_products(pattern: &PatternExpr) -> Option<Vec<(usize, usize)>> {
    fn gather(expr: &PatternExpr, terms: &mut Vec<(usize, usize)>) -> Option<()> {
        match expr {
            PatternExpr::Bin {
                op: BinTag::Add,
                a,
                b,
            } => {
                gather(a, terms)?;
                gather(b, terms)?;
                Some(())
            }
            PatternExpr::Literal(v) if *v == 0f64.to_bits() => Some(()),
            PatternExpr::Bin {
                op: BinTag::Mul,
                a,
                b,
            } => {
                if let (PatternExpr::Load(sa), PatternExpr::Load(sb)) = (&**a, &**b) {
                    terms.push((*sa, *sb));
                    Some(())
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    let mut terms = Vec::new();
    gather(pattern, &mut terms)?;
    Some(terms)
}

fn topo_order_loops(crystal_ops: &[CrystalOp]) -> Vec<CrystalOp> {
    let loops: Vec<CrystalLoop> = crystal_ops
        .iter()
        .filter_map(|op| match op {
            CrystalOp::Loop(l) => Some(l.clone()),
            CrystalOp::Scalar(_) => None,
        })
        .collect();

    let mut producer: HashMap<GlobalId, usize> = HashMap::new();
    for (i, loop_op) in loops.iter().enumerate() {
        for body_op in &loop_op.body {
            if let LoopBodyOp::Store { tensor, .. } = body_op {
                producer.insert(*tensor, i);
            }
        }
    }

    let mut indeg = vec![0usize; loops.len()];
    let mut out_edges: Vec<HashSet<usize>> = vec![HashSet::new(); loops.len()];

    for (consumer_idx, loop_op) in loops.iter().enumerate() {
        for body_op in &loop_op.body {
            if let LoopBodyOp::Load { tensor, .. } = body_op {
                if let Some(&prod_idx) = producer.get(tensor) {
                    if prod_idx != consumer_idx && out_edges[prod_idx].insert(consumer_idx) {
                        indeg[consumer_idx] += 1;
                    }
                }
            }
        }
    }

    let mut ready: BTreeMap<GlobalId, Vec<usize>> = BTreeMap::new();
    for (i, loop_op) in loops.iter().enumerate() {
        if indeg[i] == 0 {
            let key = loop_primary_output(loop_op).unwrap_or(GlobalId(0));
            ready.entry(key).or_default().push(i);
        }
    }

    let mut queue = VecDeque::new();
    for (_, bucket) in ready {
        for i in bucket {
            queue.push_back(i);
        }
    }

    let mut ordered_idx = Vec::with_capacity(loops.len());
    while let Some(i) = queue.pop_front() {
        ordered_idx.push(i);
        for &next in &out_edges[i] {
            indeg[next] -= 1;
            if indeg[next] == 0 {
                queue.push_back(next);
            }
        }
    }

    if ordered_idx.len() != loops.len() {
        // Cycle should not happen in this dataflow, but keep deterministic fallback.
        let mut remaining: Vec<usize> = (0..loops.len())
            .filter(|i| !ordered_idx.contains(i))
            .collect();
        remaining.sort_by_key(|i| loop_primary_output(&loops[*i]).unwrap_or(GlobalId(0)));
        ordered_idx.extend(remaining);
    }

    ordered_idx
        .into_iter()
        .map(|i| CrystalOp::Loop(loops[i].clone()))
        .collect()
}

fn loop_primary_output(loop_op: &CrystalLoop) -> Option<GlobalId> {
    loop_op.body.iter().find_map(|op| match op {
        LoopBodyOp::Store { tensor, .. } => Some(*tensor),
        _ => None,
    })
}

fn stable_hash<T: std::hash::Hash>(value: &T) -> u64 {
    use std::hash::Hasher;
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::attempts::v1_scalar_crystal::nano_op::NanoOpExpander;
    use crate::graph::GlobalId;
    use crate::milli_graph::MilliOpGraph;
    use crate::milli_graph::ops::{SimpleBinary, SimpleUnaryOp};

    fn shuffle_ops(mut ops: Vec<NanoOp>) -> Vec<NanoOp> {
        // Deterministic "whitewash": rotate + reverse chunks.
        let len = ops.len();
        ops.rotate_left(len / 3);
        for chunk in ops.chunks_mut(7) {
            chunk.reverse();
        }
        ops
    }

    #[test]
    fn test_grow_from_shuffled_elementwise_chain() {
        let mut rng = wyrand::WyRand::new(11);
        let ext_a = GlobalId::new(&mut rng);
        let ext_b = GlobalId::new(&mut rng);
        let ext_c = GlobalId::new(&mut rng);
        let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b, ext_c], &mut rng);
        let a = input_map[&ext_a];
        let b = input_map[&ext_b];
        let c = input_map[&ext_c];

        let mul = SimpleBinary::mul(&mut graph, a, b, &mut rng);
        let add = SimpleBinary::add(&mut graph, mul, c, &mut rng);
        let neg = SimpleUnaryOp::neg(&mut graph, add, &mut rng);

        let mut shapes = HashMap::new();
        for &t in &[a, b, c, mul, add, neg] {
            shapes.insert(t, vec![16]);
        }

        let mut expander = NanoOpExpander::new(shapes.clone());
        let ordered_ops = expander.expand(&graph).unwrap();
        let pool_ops = shuffle_ops(ordered_ops);

        let plan = grow_from_pool(&pool_ops, &shapes).unwrap();

        assert_eq!(plan.stats.recovered_loop_groups, 3);
        assert_eq!(plan.stats.recovered_matmul_groups, 0);
        assert_eq!(plan.stats.ordered_loops, 3);
        assert_eq!(plan.stats.fused_pairs, 2);
        assert_eq!(plan.stats.eliminated_loads, 2);

        let fused_loop_count = plan
            .fused_loops
            .iter()
            .filter(|op| matches!(op, CrystalOp::Loop(_)))
            .count();
        assert_eq!(fused_loop_count, 1);
    }

    #[test]
    fn test_grow_recovers_matmul_from_shuffled_pool() {
        let a = GlobalId(1001);
        let b = GlobalId(1002);
        let c = GlobalId(1003);
        let mut next_value = 0u32;
        let m = 2usize;
        let n = 2usize;
        let k = 3usize;

        let mut ops = Vec::new();
        for row in 0..m {
            for col in 0..n {
                let mut acc = NanoValue(next_value);
                next_value += 1;
                ops.push(NanoOp::Literal {
                    dst: acc,
                    value: 0.0,
                });
                for kk in 0..k {
                    let va = NanoValue(next_value);
                    next_value += 1;
                    ops.push(NanoOp::Load {
                        dst: va,
                        tensor: a,
                        flat_index: row * k + kk,
                    });
                    let vb = NanoValue(next_value);
                    next_value += 1;
                    ops.push(NanoOp::Load {
                        dst: vb,
                        tensor: b,
                        flat_index: kk * n + col,
                    });
                    let vm = NanoValue(next_value);
                    next_value += 1;
                    ops.push(NanoOp::BinOp {
                        dst: vm,
                        op: ScalarBinOp::Mul,
                        a: va,
                        b: vb,
                    });
                    let va2 = NanoValue(next_value);
                    next_value += 1;
                    ops.push(NanoOp::BinOp {
                        dst: va2,
                        op: ScalarBinOp::Add,
                        a: acc,
                        b: vm,
                    });
                    acc = va2;
                }
                ops.push(NanoOp::Store {
                    tensor: c,
                    flat_index: row * n + col,
                    src: acc,
                });
            }
        }
        let pool_ops = shuffle_ops(ops);

        let mut shapes = HashMap::new();
        shapes.insert(a, vec![m, k]);
        shapes.insert(b, vec![k, n]);
        shapes.insert(c, vec![m, n]);

        let plan = grow_from_pool(&pool_ops, &shapes).unwrap();
        assert_eq!(plan.matmul_crystals.len(), 1);
        assert_eq!(plan.stats.recovered_matmul_groups, 1);

        let mm = &plan.matmul_crystals[0];
        assert_eq!(mm.a, a);
        assert_eq!(mm.b, b);
        assert_eq!(mm.c, c);
        assert_eq!((mm.m, mm.n, mm.k), (m, n, k));
        assert_eq!(mm.covered_outputs, vec![0, 1, 2, 3]);
    }
}
