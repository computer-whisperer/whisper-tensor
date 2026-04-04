use crate::graph::{GlobalId, Node, Property, PropertyValue};
use crate::milli_graph::{self, MilliLoweringContext, MilliOpGraph, ops_helpers};
use crate::numeric_dtype::NumericDType;
use crate::onnx::AttributeProto;
use crate::symbolic_graph::ops::{EvalError, Operation};
use crate::symbolic_graph::{ONNXDecodingError, query_attribute_string};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// ONNX Einsum operator.
///
/// Evaluates Einstein summation convention on the inputs.
/// Lowered to milli ops via transpose + matmul decomposition.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EinsumOperation {
    global_id: GlobalId,
    inputs: Vec<GlobalId>,
    output: GlobalId,
    equation: String,
}

impl EinsumOperation {
    pub fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.is_empty() {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Einsum"));
        }
        if outputs.is_empty() {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Einsum"));
        }

        let equation = query_attribute_string(attributes, "equation")
            .ok_or(ONNXDecodingError::MissingField("equation"))?;

        let input_ids: Vec<GlobalId> = inputs.iter().filter_map(|x| *x).collect();

        Ok(Self {
            global_id: GlobalId::new(rng),
            inputs: input_ids,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Einsum"))?,
            equation,
        })
    }
}

impl Node for EinsumOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "Einsum".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId> + '_> {
        Box::new(self.inputs.iter().copied())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId> + '_> {
        Box::new([self.output].into_iter())
    }
}

// ── Equation parsing ─────────────────────────────────────────────────────────

fn parse_equation(eq: &str) -> Result<(Vec<Vec<char>>, Vec<char>), EvalError> {
    let eq = eq.replace(' ', "");
    let (lhs, rhs) = if let Some((l, r)) = eq.split_once("->") {
        (l, Some(r))
    } else {
        (eq.as_str(), None)
    };

    let input_subs: Vec<Vec<char>> = lhs.split(',').map(parse_subscript).collect();

    let output_sub = if let Some(r) = rhs {
        parse_subscript(r)
    } else {
        let mut counts: HashMap<char, usize> = HashMap::new();
        for sub in &input_subs {
            for &c in sub {
                *counts.entry(c).or_default() += 1;
            }
        }
        let mut out: Vec<char> = counts
            .into_iter()
            .filter(|&(_, count)| count == 1)
            .map(|(c, _)| c)
            .collect();
        out.sort();
        out
    };

    Ok((input_subs, output_sub))
}

fn parse_subscript(s: &str) -> Vec<char> {
    let mut result = Vec::new();
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if i + 2 < chars.len() && chars[i] == '.' && chars[i + 1] == '.' && chars[i + 2] == '.' {
            result.push('\u{2026}');
            i += 3;
        } else {
            result.push(chars[i]);
            i += 1;
        }
    }
    result
}

fn has_ellipsis(input_subs: &[Vec<char>], output_sub: &[char]) -> bool {
    let e = '\u{2026}';
    input_subs.iter().any(|s| s.contains(&e)) || output_sub.contains(&e)
}

fn has_diagonal(input_subs: &[Vec<char>]) -> bool {
    for sub in input_subs {
        let mut seen = std::collections::HashSet::new();
        for &c in sub {
            if !seen.insert(c) {
                return true;
            }
        }
    }
    false
}

// ── Batch diagonal: ...ii -> ...i ─────────────────────────────────────────────
//
// Extract diagonal of the last 2 dims, preserving all batch dims.
// Fully dynamic — works for any number of batch dims.

fn lower_batch_diagonal(graph: &mut MilliOpGraph, input: GlobalId, rng: &mut impl Rng) -> GlobalId {
    use milli_graph::ops as mops;

    let shape = mops::Shape::push_new(graph, input, rng);
    let rank = ops_helpers::rank(graph, input, rng);
    let one = ops_helpers::scalar_const(graph, 1i64, rng);
    let two = ops_helpers::scalar_const(graph, 2i64, rng);
    let zero = ops_helpers::scalar_const(graph, 0i64, rng);
    let neg_one = ops_helpers::scalar_const(graph, -1i64, rng);

    // dim = shape[-1] (size of the diagonal dimension)
    let dim = mops::Gather::push_new(graph, shape, neg_one, 0, rng);

    // batch_shape = shape[:-2]
    let rank_minus_2 = mops::SimpleBinary::sub(graph, rank, two, rng);
    let batch_shape = mops::Slice::push_new(graph, shape, zero, rank_minus_2, None, None, rng);

    // Flatten to 3D: [-1, dim, dim]
    let shape_3d = mops::Concat::push_new(graph, vec![neg_one, dim, dim], 0, rng);
    let flat = mops::Reshape::push_new(graph, input, shape_3d, false, rng);

    // indices = Range(0, dim) → [dim], reshape to [1, dim, 1]
    let shape_1 = mops::Constant::from_vec(graph, vec![1i64], rng);
    let dim_1d = mops::Reshape::push_new(graph, dim, shape_1, false, rng);
    let arange = mops::Range::push_new(graph, zero, dim_1d, one, rng);
    let idx_shape = mops::Concat::push_new(graph, vec![one, dim, one], 0, rng);
    let idx_3d = mops::Reshape::push_new(graph, arange, idx_shape, false, rng);

    // Expand indices to [N, dim, 1] to match flat's batch dim.
    let flat_shape = mops::Shape::push_new(graph, flat, rng);
    let n_val = mops::Gather::push_new(graph, flat_shape, zero, 0, rng);
    let expand_shape = mops::Concat::push_new(graph, vec![n_val, dim, one], 0, rng);
    let idx_expanded = mops::Expand::push_new(graph, idx_3d, expand_shape, rng);

    // GatherElements(flat, idx_expanded, axis=2) → [N, dim, 1]
    let gathered = mops::GatherElements::push_new(graph, flat, idx_expanded, 2, rng);

    // Squeeze axis 2 → [N, dim]
    let axes_2 = mops::Constant::from_vec(graph, vec![2i64], rng);
    let squeezed = mops::Squeeze::push_new(graph, gathered, axes_2, rng);

    // Reshape back to [..., dim]
    let out_shape = mops::Concat::push_new(graph, vec![batch_shape, dim], 0, rng);
    mops::Reshape::push_new(graph, squeezed, out_shape, false, rng)
}

// ── Single-input lowering (transpose + reduce) ──────────────────────────────

fn lower_single_input(
    graph: &mut MilliOpGraph,
    input: GlobalId,
    input_sub: &[char],
    output_sub: &[char],
    rng: &mut impl Rng,
) -> GlobalId {
    let mut perm = Vec::new();
    let mut contract_axes = Vec::new();

    for &out_label in output_sub {
        let pos = input_sub
            .iter()
            .position(|&c| c == out_label)
            .unwrap_or_else(|| panic!("Einsum: output label '{out_label}' not in input"));
        perm.push(pos);
    }

    for (pos, &label) in input_sub.iter().enumerate() {
        if !output_sub.contains(&label) {
            perm.push(pos);
            contract_axes.push(output_sub.len() + contract_axes.len());
        }
    }

    let is_identity = perm.iter().enumerate().all(|(i, &p)| p == i);
    let current = if !is_identity {
        let perm_i64: Vec<i64> = perm.iter().map(|&p| p as i64).collect();
        milli_graph::ops::Transpose::push_new(graph, input, Some(perm_i64), rng)
    } else {
        input
    };

    if contract_axes.is_empty() {
        current
    } else {
        let axes_vec: Vec<i64> = contract_axes.iter().map(|&a| a as i64).collect();
        let axes_id = milli_graph::ops::Constant::from_vec(graph, axes_vec, rng);
        milli_graph::ops::ReduceSum::push_new(graph, current, Some(axes_id), false, false, rng)
    }
}

// ── Two-input lowering (transpose + reshape + matmul) ────────────────────────

fn lower_two_input(
    graph: &mut MilliOpGraph,
    a_id: GlobalId,
    b_id: GlobalId,
    a_sub: &[char],
    b_sub: &[char],
    output_sub: &[char],
    rng: &mut impl Rng,
) -> GlobalId {
    // Classify labels
    let mut batch = Vec::new();
    let mut free_a = Vec::new();
    let mut free_b = Vec::new();
    let mut contract = Vec::new();
    let mut reduce_a = Vec::new(); // in A only, not in output — pre-reduce
    let mut reduce_b = Vec::new(); // in B only, not in output — pre-reduce

    let mut all_labels = Vec::new();
    for &c in a_sub.iter().chain(b_sub.iter()) {
        if !all_labels.contains(&c) {
            all_labels.push(c);
        }
    }

    let a_set: std::collections::HashSet<char> = a_sub.iter().copied().collect();
    let b_set: std::collections::HashSet<char> = b_sub.iter().copied().collect();
    let out_set: std::collections::HashSet<char> = output_sub.iter().copied().collect();

    for &label in &all_labels {
        let in_a = a_set.contains(&label);
        let in_b = b_set.contains(&label);
        let in_out = out_set.contains(&label);
        match (in_a, in_b, in_out) {
            (true, true, true) => batch.push(label),
            (true, true, false) => contract.push(label),
            (true, false, true) => free_a.push(label),
            (false, true, true) => free_b.push(label),
            (true, false, false) => reduce_a.push(label),
            (false, true, false) => reduce_b.push(label),
            _ => panic!("Einsum: output label '{label}' not found in any input"),
        }
    }

    // Pre-reduce labels that appear in only one input and not in the output.
    // These can't be handled by matmul contraction — sum them out first.
    let (mut a_current, mut a_sub_current) = (a_id, a_sub.to_vec());
    if !reduce_a.is_empty() {
        let axes: Vec<i64> = reduce_a
            .iter()
            .map(|label| a_sub_current.iter().position(|c| c == label).unwrap() as i64)
            .collect();
        let axes_id = milli_graph::ops::Constant::from_vec(graph, axes, rng);
        a_current = milli_graph::ops::ReduceSum::push_new(
            graph,
            a_current,
            Some(axes_id),
            false,
            false,
            rng,
        );
        a_sub_current.retain(|c| !reduce_a.contains(c));
    }

    let (mut b_current, mut b_sub_current) = (b_id, b_sub.to_vec());
    if !reduce_b.is_empty() {
        let axes: Vec<i64> = reduce_b
            .iter()
            .map(|label| b_sub_current.iter().position(|c| c == label).unwrap() as i64)
            .collect();
        let axes_id = milli_graph::ops::Constant::from_vec(graph, axes, rng);
        b_current = milli_graph::ops::ReduceSum::push_new(
            graph,
            b_current,
            Some(axes_id),
            false,
            false,
            rng,
        );
        b_sub_current.retain(|c| !reduce_b.contains(c));
    }

    // Build transpose permutations
    // A target: [batch..., free_A..., contract...]
    let a_perm: Vec<i64> = batch
        .iter()
        .chain(free_a.iter())
        .chain(contract.iter())
        .map(|label| a_sub_current.iter().position(|c| c == label).unwrap() as i64)
        .collect();

    // B target: [batch..., contract..., free_B...]
    let b_perm: Vec<i64> = batch
        .iter()
        .chain(contract.iter())
        .chain(free_b.iter())
        .map(|label| b_sub_current.iter().position(|c| c == label).unwrap() as i64)
        .collect();

    let a_t = if a_perm.iter().enumerate().all(|(i, &p)| p == i as i64) {
        a_current
    } else {
        milli_graph::ops::Transpose::push_new(graph, a_current, Some(a_perm), rng)
    };

    let b_t = if b_perm.iter().enumerate().all(|(i, &p)| p == i as i64) {
        b_current
    } else {
        milli_graph::ops::Transpose::push_new(graph, b_current, Some(b_perm), rng)
    };

    let n_batch = batch.len();
    let n_free_a = free_a.len();
    let n_free_b = free_b.len();
    let n_contract = contract.len();

    let needs_reshape = n_free_a > 1 || n_free_b > 1 || n_contract > 1;

    let (a_mm, b_mm) = if needs_reshape {
        let a_shape = milli_graph::ops::Shape::push_new(graph, a_t, rng);
        let b_shape = milli_graph::ops::Shape::push_new(graph, b_t, rng);

        let a_new = build_collapsed_shape(graph, a_shape, n_batch, n_free_a, n_contract, rng);
        let b_new = build_collapsed_shape(graph, b_shape, n_batch, n_contract, n_free_b, rng);

        (
            milli_graph::ops::Reshape::push_new(graph, a_t, a_new, false, rng),
            milli_graph::ops::Reshape::push_new(graph, b_t, b_new, false, rng),
        )
    } else {
        (a_t, b_t)
    };

    // Cast to F32, matmul, cast back
    let a_f32 = milli_graph::ops::Cast::push_new(graph, a_mm, NumericDType::F32, rng);
    let b_f32 = milli_graph::ops::Cast::push_new(graph, b_mm, NumericDType::F32, rng);

    let mm = milli_graph::ops::MatMul::push_new_default_precision(
        graph,
        a_f32,
        b_f32,
        NumericDType::F32,
        rng,
    );

    let mm_cast = milli_graph::ops::CastLike::push_new(graph, mm, a_id, rng);

    // Reshape back if collapsed
    let after_mm = if needs_reshape {
        let a_orig_shape = milli_graph::ops::Shape::push_new(graph, a_t, rng);
        let b_orig_shape = milli_graph::ops::Shape::push_new(graph, b_t, rng);
        let expand_shape = build_expanded_shape(
            graph,
            a_orig_shape,
            b_orig_shape,
            n_batch,
            n_free_a,
            n_contract,
            n_free_b,
            rng,
        );
        milli_graph::ops::Reshape::push_new(graph, mm_cast, expand_shape, false, rng)
    } else {
        mm_cast
    };

    // Final transpose to match output label order
    let result_labels: Vec<char> = batch
        .iter()
        .chain(free_a.iter())
        .chain(free_b.iter())
        .copied()
        .collect();

    let final_perm: Vec<i64> = output_sub
        .iter()
        .map(|label| {
            result_labels
                .iter()
                .position(|c| c == label)
                .unwrap_or_else(|| panic!("Einsum: output label '{label}' not in result"))
                as i64
        })
        .collect();

    if final_perm.iter().enumerate().all(|(i, &p)| p == i as i64) {
        after_mm
    } else {
        milli_graph::ops::Transpose::push_new(graph, after_mm, Some(final_perm), rng)
    }
}

// ── Shape helpers for dynamic reshape ────────────────────────────────────────

/// Build [batch_dim0, ..., prod(group1), prod(group2)] shape tensor.
fn build_collapsed_shape(
    graph: &mut MilliOpGraph,
    shape_id: GlobalId,
    n_batch: usize,
    n_group1: usize,
    n_group2: usize,
    rng: &mut impl Rng,
) -> GlobalId {
    let mut parts: Vec<GlobalId> = Vec::new();

    // Batch dims kept individually
    for i in 0..n_batch {
        let idx_id = milli_graph::ops::Constant::from_vec(graph, vec![i as i64], rng);
        let dim = milli_graph::ops::Gather::push_new(graph, shape_id, idx_id, 0, rng);
        parts.push(dim);
    }

    if n_group1 > 0 {
        let prod = gather_and_product(graph, shape_id, n_batch, n_group1, rng);
        let axes = milli_graph::ops::Constant::from_vec(graph, vec![0i64], rng);
        parts.push(milli_graph::ops::Unsqueeze::push_new(
            graph, prod, axes, rng,
        ));
    }

    if n_group2 > 0 {
        let prod = gather_and_product(graph, shape_id, n_batch + n_group1, n_group2, rng);
        let axes = milli_graph::ops::Constant::from_vec(graph, vec![0i64], rng);
        parts.push(milli_graph::ops::Unsqueeze::push_new(
            graph, prod, axes, rng,
        ));
    }

    if parts.len() == 1 {
        parts.into_iter().next().unwrap()
    } else {
        milli_graph::ops::Concat::push_new(graph, parts, 0, rng)
    }
}

/// Gather dims at [offset..offset+count) from shape tensor and multiply them together.
fn gather_and_product(
    graph: &mut MilliOpGraph,
    shape_id: GlobalId,
    offset: usize,
    count: usize,
    rng: &mut impl Rng,
) -> GlobalId {
    assert!(count > 0);
    if count == 1 {
        let idx = milli_graph::ops::Constant::new_scalar(graph, offset as i64, rng);
        return milli_graph::ops::Gather::push_new(graph, shape_id, idx, 0, rng);
    }
    let indices: Vec<i64> = (offset..offset + count).map(|i| i as i64).collect();
    let idx_id = milli_graph::ops::Constant::from_vec(graph, indices, rng);
    let gathered = milli_graph::ops::Gather::push_new(graph, shape_id, idx_id, 0, rng);

    let mut product = {
        let i0 = milli_graph::ops::Constant::new_scalar(graph, 0i64, rng);
        milli_graph::ops::Gather::push_new(graph, gathered, i0, 0, rng)
    };
    for i in 1..count {
        let idx = milli_graph::ops::Constant::new_scalar(graph, i as i64, rng);
        let dim = milli_graph::ops::Gather::push_new(graph, gathered, idx, 0, rng);
        product = milli_graph::ops::SimpleBinary::mul(graph, product, dim, rng);
    }
    product
}

/// Build expanded shape: [batch_dims..., free_A_dims..., free_B_dims...]
#[allow(clippy::too_many_arguments)]
fn build_expanded_shape(
    graph: &mut MilliOpGraph,
    a_shape_id: GlobalId,
    b_shape_id: GlobalId,
    n_batch: usize,
    n_free_a: usize,
    n_contract: usize,
    n_free_b: usize,
    rng: &mut impl Rng,
) -> GlobalId {
    let mut parts: Vec<GlobalId> = Vec::new();

    if n_batch > 0 {
        let idx: Vec<i64> = (0..n_batch).map(|i| i as i64).collect();
        let idx_id = milli_graph::ops::Constant::from_vec(graph, idx, rng);
        parts.push(milli_graph::ops::Gather::push_new(
            graph, a_shape_id, idx_id, 0, rng,
        ));
    }

    if n_free_a > 0 {
        let idx: Vec<i64> = (n_batch..n_batch + n_free_a).map(|i| i as i64).collect();
        let idx_id = milli_graph::ops::Constant::from_vec(graph, idx, rng);
        parts.push(milli_graph::ops::Gather::push_new(
            graph, a_shape_id, idx_id, 0, rng,
        ));
    }

    if n_free_b > 0 {
        let start = n_batch + n_contract;
        let idx: Vec<i64> = (start..start + n_free_b).map(|i| i as i64).collect();
        let idx_id = milli_graph::ops::Constant::from_vec(graph, idx, rng);
        parts.push(milli_graph::ops::Gather::push_new(
            graph, b_shape_id, idx_id, 0, rng,
        ));
    }

    if parts.len() == 1 {
        parts.into_iter().next().unwrap()
    } else {
        milli_graph::ops::Concat::push_new(graph, parts, 0, rng)
    }
}

// ── Operation impl ───────────────────────────────────────────────────────────

impl Operation for EinsumOperation {
    fn parameters(&self) -> Vec<Property> {
        vec![Property::new(
            "equation",
            PropertyValue::String(self.equation.clone()),
        )]
    }

    fn is_differentiable(&self) -> bool {
        false
    }

    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);

        let (input_subs, output_sub) =
            parse_equation(&self.equation).expect("Einsum: failed to parse equation");

        assert_eq!(
            input_subs.len(),
            self.inputs.len(),
            "Einsum: equation has {} inputs but operation has {}",
            input_subs.len(),
            self.inputs.len()
        );

        // Special case: batch diagonal "...ii -> ...i" (single input).
        // Extract diagonal of the last 2 dims, preserving batch dims.
        if self.inputs.len() == 1
            && has_ellipsis(&input_subs, &output_sub)
            && has_diagonal(&input_subs)
        {
            // Check pattern: input ends in [X, X] (same label twice),
            // output ends in [X] (that label once), both have ellipsis prefix.
            let isub = &input_subs[0];
            let e = '\u{2026}'; // ellipsis char
            let is_batch_diag = isub.len() >= 3
                && isub[0] == e
                && isub[isub.len() - 1] == isub[isub.len() - 2]
                && isub[isub.len() - 1] != e
                && output_sub.len() >= 2
                && output_sub[0] == e
                && output_sub[output_sub.len() - 1] == isub[isub.len() - 1];

            if is_batch_diag {
                let inp = input_map[&self.inputs[0]];
                let result = lower_batch_diagonal(&mut graph, inp, rng);
                let mut output_map = HashMap::new();
                output_map.insert(result, self.output);
                graph.set_output_map(output_map);
                return graph;
            }
        }

        if has_ellipsis(&input_subs, &output_sub) {
            panic!(
                "Einsum: ellipsis ('...') requires runtime rank not available during \
                 milli lowering. Equation: '{}'",
                self.equation
            );
        }
        if has_diagonal(&input_subs) {
            panic!(
                "Einsum: diagonal labels (e.g. 'ii->i') not supported. Equation: '{}'",
                self.equation
            );
        }

        let result = if self.inputs.len() == 1 {
            let inp = input_map[&self.inputs[0]];
            lower_single_input(&mut graph, inp, &input_subs[0], &output_sub, rng)
        } else if self.inputs.len() == 2 {
            let a = input_map[&self.inputs[0]];
            let b = input_map[&self.inputs[1]];
            lower_two_input(
                &mut graph,
                a,
                b,
                &input_subs[0],
                &input_subs[1],
                &output_sub,
                rng,
            )
        } else {
            // >2 inputs: chain pairwise left to right
            let mut current = input_map[&self.inputs[0]];
            let mut current_sub = input_subs[0].clone();

            for i in 1..self.inputs.len() {
                let next = input_map[&self.inputs[i]];
                let next_sub = &input_subs[i];

                let intermediate_out = if i == self.inputs.len() - 1 {
                    output_sub.clone()
                } else {
                    // Keep labels needed by future inputs or the final output
                    let mut future: std::collections::HashSet<char> =
                        output_sub.iter().copied().collect();
                    for sub in &input_subs[(i + 1)..] {
                        for &c in sub {
                            future.insert(c);
                        }
                    }
                    let mut intermediate = Vec::new();
                    for &c in current_sub.iter().chain(next_sub.iter()) {
                        if future.contains(&c) && !intermediate.contains(&c) {
                            intermediate.push(c);
                        }
                    }
                    intermediate
                };

                current = lower_two_input(
                    &mut graph,
                    current,
                    next,
                    &current_sub,
                    next_sub,
                    &intermediate_out,
                    rng,
                );
                current_sub = intermediate_out;
            }
            current
        };

        let mut output_map = HashMap::new();
        output_map.insert(result, self.output);
        graph.set_output_map(output_map);
        graph
    }
}
