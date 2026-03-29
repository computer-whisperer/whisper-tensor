use crate::graph::{GlobalId, Node, Property, PropertyValue};
use crate::milli_graph::ops as milli_ops;
use crate::milli_graph::{MilliLoweringContext, MilliOpGraph};
use crate::numeric_dtype::NumericDType;
use crate::onnx;
use crate::symbolic_graph::ops::Operation;
use crate::symbolic_graph::{
    ONNXDecodingError, query_attribute_int, query_attribute_ints, query_attribute_string,
};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
enum AutoPad {
    NotSet,
    SameUpper,
    SameLower,
    Valid,
}

/// ONNX ConvTranspose (transposed / deconvolution).
///
/// Lowered to milli ops by decomposing into:
/// 1. Weight transpose + spatial flip
/// 2. Input dilation (insert zeros between elements for stride > 1)
/// 3. Explicit padding
/// 4. Standard Conv with stride=1
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConvTransposeOperation {
    global_id: GlobalId,
    input: GlobalId,
    weight: GlobalId,
    bias: Option<GlobalId>,
    output: GlobalId,
    auto_pad: AutoPad,
    dilations: Vec<i64>,
    group: i64,
    kernel_shape: Vec<i64>,
    output_padding: Vec<i64>,
    output_shape: Vec<i64>,
    pads: Vec<i64>,
    strides: Vec<i64>,
}

impl ConvTransposeOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() < 2 || inputs.len() > 3 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("ConvTranspose"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("ConvTranspose"));
        }

        let auto_pad = match query_attribute_string(attributes, "auto_pad").as_deref() {
            Some("SAME_UPPER") => AutoPad::SameUpper,
            Some("SAME_LOWER") => AutoPad::SameLower,
            Some("VALID") => AutoPad::Valid,
            _ => AutoPad::NotSet,
        };

        Ok(Self {
            global_id: GlobalId::new(rng),
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("ConvTranspose"))?,
            weight: inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs("ConvTranspose"))?,
            bias: inputs.get(2).and_then(|x| *x),
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("ConvTranspose"))?,
            auto_pad,
            dilations: query_attribute_ints(attributes, "dilations").unwrap_or_default(),
            group: query_attribute_int(attributes, "group").unwrap_or(1),
            kernel_shape: query_attribute_ints(attributes, "kernel_shape").unwrap_or_default(),
            output_padding: query_attribute_ints(attributes, "output_padding").unwrap_or_default(),
            output_shape: query_attribute_ints(attributes, "output_shape").unwrap_or_default(),
            pads: query_attribute_ints(attributes, "pads").unwrap_or_default(),
            strides: query_attribute_ints(attributes, "strides").unwrap_or_default(),
        })
    }

    fn infer_nd(&self) -> Option<usize> {
        if !self.kernel_shape.is_empty() {
            Some(self.kernel_shape.len())
        } else if !self.strides.is_empty() {
            Some(self.strides.len())
        } else if !self.dilations.is_empty() {
            Some(self.dilations.len())
        } else if !self.pads.is_empty() {
            Some(self.pads.len() / 2)
        } else if !self.output_padding.is_empty() {
            Some(self.output_padding.len())
        } else {
            None
        }
    }
}

impl Node for ConvTransposeOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "ConvTranspose".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        let mut v = vec![self.input, self.weight];
        if let Some(b) = self.bias {
            v.push(b);
        }
        Box::new(v.into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}

fn const_i64_vec(graph: &mut MilliOpGraph, values: Vec<i64>, rng: &mut impl Rng) -> GlobalId {
    milli_ops::Constant::from_vec(graph, values, rng)
}

fn const_i64_scalar(graph: &mut MilliOpGraph, value: i64, rng: &mut impl Rng) -> GlobalId {
    milli_ops::Constant::new_scalar(graph, value, rng)
}

impl Operation for ConvTransposeOperation {
    fn parameters(&self) -> Vec<Property> {
        let mut params = Vec::new();
        params.push(Property::new("group", PropertyValue::Int(self.group)));
        if !self.kernel_shape.is_empty() {
            params.push(Property::new(
                "kernel_shape",
                PropertyValue::IntList(self.kernel_shape.clone()),
            ));
        }
        if !self.strides.is_empty() {
            params.push(Property::new(
                "strides",
                PropertyValue::IntList(self.strides.clone()),
            ));
        }
        if !self.pads.is_empty() {
            params.push(Property::new(
                "pads",
                PropertyValue::IntList(self.pads.clone()),
            ));
        }
        params
    }

    fn is_differentiable(&self) -> bool {
        false
    }

    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        assert!(
            matches!(self.auto_pad, AutoPad::NotSet | AutoPad::Valid),
            "ConvTranspose: auto_pad SAME_UPPER/SAME_LOWER not yet supported"
        );
        assert!(
            self.output_shape.is_empty(),
            "ConvTranspose: output_shape attribute not yet supported"
        );

        let nd = self.infer_nd().expect(
            "ConvTranspose: cannot infer spatial dims — provide kernel_shape, strides, or dilations",
        );

        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);

        let x_in = input_map[&self.input];
        let w_in = input_map[&self.weight];

        // Cast to F32
        let x_f32 = milli_ops::Cast::push_new(&mut graph, x_in, NumericDType::F32, rng);
        let w_f32 = milli_ops::Cast::push_new(&mut graph, w_in, NumericDType::F32, rng);

        // Per-axis parameters with defaults
        let kernel: Vec<i64> = (0..nd)
            .map(|i| self.kernel_shape.get(i).copied().unwrap_or(0))
            .collect();
        let stride: Vec<i64> = (0..nd)
            .map(|i| self.strides.get(i).copied().unwrap_or(1))
            .collect();
        let dilation: Vec<i64> = (0..nd)
            .map(|i| self.dilations.get(i).copied().unwrap_or(1))
            .collect();
        let output_pad: Vec<i64> = (0..nd)
            .map(|i| self.output_padding.get(i).copied().unwrap_or(0))
            .collect();

        // ─── Step 1: Weight rearrangement ────────────────────────────────────
        let w_conv = build_weight(&mut graph, w_f32, self.group, nd, rng);

        // ─── Step 2: Input dilation ──────────────────────────────────────────
        let mut current = x_f32;
        for (i, &s) in stride.iter().enumerate() {
            if s > 1 {
                current = dilate_spatial_dim(&mut graph, current, i, nd, s, rng);
            }
        }

        // ─── Step 3: Padding ─────────────────────────────────────────────────
        let has_kernel_attr = kernel.iter().all(|&k| k > 0);
        let padded = if has_kernel_attr {
            let orig_pb: Vec<i64> = (0..nd)
                .map(|i| self.pads.get(i).copied().unwrap_or(0))
                .collect();
            let orig_pe: Vec<i64> = (0..nd)
                .map(|i| self.pads.get(nd + i).copied().unwrap_or(0))
                .collect();
            let new_pb: Vec<i64> = (0..nd)
                .map(|i| dilation[i] * (kernel[i] - 1) - orig_pb[i])
                .collect();
            let new_pe: Vec<i64> = (0..nd)
                .map(|i| dilation[i] * (kernel[i] - 1) - orig_pe[i] + output_pad[i])
                .collect();

            if new_pb.iter().all(|&p| p == 0) && new_pe.iter().all(|&p| p == 0) {
                current
            } else {
                let rank = 2 + nd;
                let mut pv = vec![0i64; 2 * rank];
                for i in 0..nd {
                    pv[2 + i] = new_pb[i];
                    pv[rank + 2 + i] = new_pe[i];
                }
                let pads_id = const_i64_vec(&mut graph, pv, rng);
                let zero_val = milli_ops::Constant::new_scalar(&mut graph, 0.0f32, rng);
                milli_ops::Pad::push_new(
                    &mut graph,
                    current,
                    pads_id,
                    Some(zero_val),
                    None,
                    milli_ops::PadMode::Constant,
                    rng,
                )
            }
        } else {
            build_dynamic_padding(
                &mut graph,
                current,
                w_f32,
                nd,
                &dilation,
                &output_pad,
                &self.pads,
                rng,
            )
        };

        // ─── Step 4: Conv with stride=1 ──────────────────────────────────────
        let conv_bias = self.bias.map(|id| {
            let b_in = input_map[&id];
            milli_ops::Cast::push_new(&mut graph, b_in, NumericDType::F32, rng)
        });

        let conv_out = milli_ops::Conv::push_new(
            &mut graph,
            padded,
            w_conv,
            conv_bias,
            milli_ops::ConvAutoPad::NotSet,
            dilation,
            self.group,
            if has_kernel_attr {
                self.kernel_shape.clone()
            } else {
                vec![]
            },
            vec![],
            vec![1; nd],
            rng,
        );

        let result = milli_ops::CastLike::push_new(&mut graph, conv_out, x_in, rng);

        let mut output_map = HashMap::new();
        output_map.insert(result, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

/// W: [C_in, C_out/g, K...] → [C_out, C_in/g, K_reversed...]
fn build_weight(
    graph: &mut MilliOpGraph,
    w: GlobalId,
    group: i64,
    nd: usize,
    rng: &mut impl Rng,
) -> GlobalId {
    let w_transposed = if group == 1 {
        let perm: Vec<i64> = [1, 0].into_iter().chain(2..2 + nd as i64).collect();
        milli_ops::Transpose::push_new(graph, w, Some(perm), rng)
    } else {
        let w_shape = milli_ops::Shape::push_new(graph, w, rng);
        let idx_0 = const_i64_scalar(graph, 0, rng);
        let idx_1 = const_i64_scalar(graph, 1, rng);
        let g_const = const_i64_scalar(graph, group, rng);
        let c_in = milli_ops::Gather::push_new(graph, w_shape, idx_0, 0, rng);
        let c_in_per_g = milli_ops::SimpleBinary::div(graph, c_in, g_const, rng);
        let c_out_per_g = milli_ops::Gather::push_new(graph, w_shape, idx_1, 0, rng);
        let minus_one = const_i64_scalar(graph, -1, rng);

        let grouped_shape = milli_ops::Concat::push_new(
            graph,
            vec![g_const, c_in_per_g, c_out_per_g, minus_one],
            0,
            rng,
        );
        let w_grouped = milli_ops::Reshape::push_new(graph, w, grouped_shape, false, rng);
        let w_swapped =
            milli_ops::Transpose::push_new(graph, w_grouped, Some(vec![0, 2, 1, 3]), rng);

        let c_out = milli_ops::SimpleBinary::mul(graph, c_out_per_g, g_const, rng);
        let slice_start = const_i64_vec(graph, vec![2], rng);
        let slice_end = const_i64_vec(graph, vec![i64::MAX], rng);
        let spatial_shape =
            milli_ops::Slice::push_new(graph, w_shape, slice_start, slice_end, None, None, rng);
        let final_shape =
            milli_ops::Concat::push_new(graph, vec![c_out, c_in_per_g, spatial_shape], 0, rng);
        milli_ops::Reshape::push_new(graph, w_swapped, final_shape, false, rng)
    };

    // Reverse spatial dims via Slice with step=-1
    if nd == 0 {
        return w_transposed;
    }
    let starts = const_i64_vec(graph, vec![i64::MAX; nd], rng);
    let ends = const_i64_vec(graph, vec![i64::MIN; nd], rng);
    let steps = const_i64_vec(graph, vec![-1i64; nd], rng);
    let axes: Vec<i64> = (2..2 + nd as i64).collect();
    let axes_id = const_i64_vec(graph, axes, rng);
    milli_ops::Slice::push_new(
        graph,
        w_transposed,
        starts,
        ends,
        Some(steps),
        Some(axes_id),
        rng,
    )
}

/// Insert (stride-1) zeros between elements along spatial dim `dim_idx`.
fn dilate_spatial_dim(
    graph: &mut MilliOpGraph,
    input: GlobalId,
    dim_idx: usize,
    nd: usize,
    s: i64,
    rng: &mut impl Rng,
) -> GlobalId {
    let spatial_axis = 2 + dim_idx;
    let rank_after = 2 + nd + 1;

    // 1. Unsqueeze at spatial_axis+1
    let unsq_axes = const_i64_vec(graph, vec![spatial_axis as i64 + 1], rng);
    let unsqueezed = milli_ops::Unsqueeze::push_new(graph, input, unsq_axes, rng);

    // 2. Pad with (0, s-1) zeros along the new dim
    let mut pv = vec![0i64; 2 * rank_after];
    pv[rank_after + spatial_axis + 1] = s - 1;
    let pads_id = const_i64_vec(graph, pv, rng);
    let zero_val = milli_ops::Constant::new_scalar(graph, 0.0f32, rng);
    let padded = milli_ops::Pad::push_new(
        graph,
        unsqueezed,
        pads_id,
        Some(zero_val),
        None,
        milli_ops::PadMode::Constant,
        rng,
    );

    // 3. Reshape: merge dims [spatial_axis, spatial_axis+1]
    let padded_shape = milli_ops::Shape::push_new(graph, padded, rng);

    let prefix_start = const_i64_vec(graph, vec![0], rng);
    let prefix_end = const_i64_vec(graph, vec![spatial_axis as i64], rng);
    let prefix = milli_ops::Slice::push_new(
        graph,
        padded_shape,
        prefix_start,
        prefix_end,
        None,
        None,
        rng,
    );

    let idx_sa = const_i64_scalar(graph, spatial_axis as i64, rng);
    let idx_sa1 = const_i64_scalar(graph, spatial_axis as i64 + 1, rng);
    let dim_l = milli_ops::Gather::push_new(graph, padded_shape, idx_sa, 0, rng);
    let dim_s = milli_ops::Gather::push_new(graph, padded_shape, idx_sa1, 0, rng);
    let merged = milli_ops::SimpleBinary::mul(graph, dim_l, dim_s, rng);

    let suffix_start = const_i64_vec(graph, vec![spatial_axis as i64 + 2], rng);
    let suffix_end = const_i64_vec(graph, vec![rank_after as i64], rng);
    let suffix = milli_ops::Slice::push_new(
        graph,
        padded_shape,
        suffix_start,
        suffix_end,
        None,
        None,
        rng,
    );

    let new_shape = milli_ops::Concat::push_new(graph, vec![prefix, merged, suffix], 0, rng);
    let reshaped = milli_ops::Reshape::push_new(graph, padded, new_shape, false, rng);

    // 4. Slice to trim trailing zeros: keep [0, L*s - (s-1))
    let s_minus_1 = const_i64_scalar(graph, s - 1, rng);
    let dilated_end = milli_ops::SimpleBinary::sub(graph, merged, s_minus_1, rng);
    let slice_starts = const_i64_vec(graph, vec![0], rng);
    let slice_axes = const_i64_vec(graph, vec![spatial_axis as i64], rng);
    milli_ops::Slice::push_new(
        graph,
        reshaped,
        slice_starts,
        dilated_end,
        None,
        Some(slice_axes),
        rng,
    )
}

/// Dynamic padding when kernel_shape not in attributes.
#[allow(clippy::too_many_arguments)]
fn build_dynamic_padding(
    graph: &mut MilliOpGraph,
    input: GlobalId,
    w: GlobalId,
    nd: usize,
    dilation: &[i64],
    output_pad: &[i64],
    orig_pads: &[i64],
    rng: &mut impl Rng,
) -> GlobalId {
    let w_shape = milli_ops::Shape::push_new(graph, w, rng);
    let s2 = const_i64_vec(graph, vec![2], rng);
    let s2nd = const_i64_vec(graph, vec![2 + nd as i64], rng);
    let kernel_dims = milli_ops::Slice::push_new(graph, w_shape, s2, s2nd, None, None, rng);

    let ones = const_i64_vec(graph, vec![1i64; nd], rng);
    let km1 = milli_ops::SimpleBinary::sub(graph, kernel_dims, ones, rng);

    let dil_t = const_i64_vec(graph, dilation.to_vec(), rng);
    let dkm1 = milli_ops::SimpleBinary::mul(graph, dil_t, km1, rng);

    let orig_pb: Vec<i64> = (0..nd)
        .map(|i| orig_pads.get(i).copied().unwrap_or(0))
        .collect();
    let orig_pe: Vec<i64> = (0..nd)
        .map(|i| orig_pads.get(nd + i).copied().unwrap_or(0))
        .collect();
    let pb_t = const_i64_vec(graph, orig_pb, rng);
    let pe_t = const_i64_vec(graph, orig_pe, rng);
    let opad_t = const_i64_vec(graph, output_pad.to_vec(), rng);

    let new_pb = milli_ops::SimpleBinary::sub(graph, dkm1, pb_t, rng);
    let new_pe_no_op = milli_ops::SimpleBinary::sub(graph, dkm1, pe_t, rng);
    let new_pe = milli_ops::SimpleBinary::add(graph, new_pe_no_op, opad_t, rng);

    let zeros_2 = const_i64_vec(graph, vec![0, 0], rng);
    let pad_tensor =
        milli_ops::Concat::push_new(graph, vec![zeros_2, new_pb, zeros_2, new_pe], 0, rng);

    let zero_val = milli_ops::Constant::new_scalar(graph, 0.0f32, rng);
    milli_ops::Pad::push_new(
        graph,
        input,
        pad_tensor,
        Some(zero_val),
        None,
        milli_ops::PadMode::Constant,
        rng,
    )
}
