use crate::graph::{GlobalId, Node, Property, PropertyValue};
use crate::milli_graph::ops as milli_ops;
use crate::milli_graph::{MilliLoweringContext, MilliOpGraph, ops_helpers};
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
/// 1. Weight transpose (swap C_in/C_out) + spatial flip
/// 2. Input dilation (Dilate milli op — insert zeros for stride > 1)
/// 3. Explicit padding (computed from kernel shape, dilation, pads, output_padding)
/// 4. Standard Conv with stride=1
///
/// All shape-dependent logic is built dynamically using Shape/Slice/Concat/etc.
/// so no tensor ranks or shapes need to be known at graph construction time.
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
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);

        let x_in = input_map[&self.input];
        let w_in = input_map[&self.weight];

        // Cast to F32 for computation.
        let x_f32 = milli_ops::Cast::push_new(&mut graph, x_in, NumericDType::F32, rng);
        let w_f32 = milli_ops::Cast::push_new(&mut graph, w_in, NumericDType::F32, rng);

        // ─── Step 1: Weight rearrangement ────────────────────────────────────
        // W: [C_in, C_out/g, K...] → [C_out, C_in/g, K_reversed...]
        let w_conv = build_weight(&mut graph, w_f32, self.group, rng);

        // ─── Step 2: Input dilation ──────────────────────────────────────────
        // Dilate inserts stride[i]-1 zeros between elements along spatial axes.
        // If strides is empty, all strides default to 1 → no dilation needed.
        let dilated = if self.strides.is_empty() || self.strides.iter().all(|&s| s <= 1) {
            x_f32
        } else {
            milli_ops::Dilate::push_new(&mut graph, x_f32, self.strides.clone(), rng)
        };

        // ─── Step 3: Padding ─────────────────────────────────────────────────
        // pad_begin[i] = effective_kernel[i] - 1 - orig_pads_begin[i]
        // pad_end[i]   = effective_kernel[i] - 1 - orig_pads_end[i] + output_padding[i]
        // where effective_kernel[i] = (kernel[i] - 1) * dilation[i] + 1
        //
        // All computed dynamically from weight shape.
        let padded = build_padding(
            &mut graph,
            dilated,
            w_f32,
            &self.dilations,
            &self.pads,
            &self.output_padding,
            &self.output_shape,
            &self.auto_pad,
            x_f32,
            &self.strides,
            rng,
        );

        // ─── Step 4: Conv with stride=1 ──────────────────────────────────────
        let conv_bias = self.bias.map(|id| {
            let b_in = input_map[&id];
            milli_ops::Cast::push_new(&mut graph, b_in, NumericDType::F32, rng)
        });

        // Conv with stride=1. Pass dilations and kernel_shape from attributes
        // (empty means "infer from weight"). Strides is always all-1s
        // (the ConvTranspose strides were handled by Dilate).
        let conv_out = milli_ops::Conv::push_new(
            &mut graph,
            padded,
            w_conv,
            conv_bias,
            milli_ops::ConvAutoPad::NotSet,
            self.dilations.clone(),
            self.group,
            self.kernel_shape.clone(),
            vec![], // pads: none (baked into explicit Pad above)
            vec![], // strides: all 1s (empty = default)
            rng,
        );

        let result = milli_ops::CastLike::push_new(&mut graph, conv_out, x_in, rng);

        let mut output_map = HashMap::new();
        output_map.insert(result, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

// ─── Weight rearrangement ────────────────────────────────────────────────────
//
// W: [C_in, C_out/g, K...] → [C_out, C_in/g, K_reversed...]
//
// Done without knowing nd by:
// 1. Reshape to 3D: [dim0, dim1, product_of_spatial]
// 2. Transpose [1, 0, 2]
// 3. Reshape back to [dim1, dim0, *spatial_dims]
// 4. Reverse spatial dims using dynamic axes

fn build_weight(graph: &mut MilliOpGraph, w: GlobalId, group: i64, rng: &mut impl Rng) -> GlobalId {
    let w_transposed = if group == 1 {
        swap_axes_01(graph, w, rng)
    } else {
        build_weight_grouped(graph, w, group, rng)
    };

    // Reverse spatial dims: Slice with step=-1 on axes [2, 3, ..., rank-1].
    reverse_spatial_dims(graph, w_transposed, rng)
}

/// Swap axes 0 and 1 of a tensor with arbitrary rank.
/// Reshape [d0, d1, *rest] → [d0, d1, -1] → Transpose [1,0,2] → Reshape [d1, d0, *rest].
fn swap_axes_01(graph: &mut MilliOpGraph, t: GlobalId, rng: &mut impl Rng) -> GlobalId {
    let shape = milli_ops::Shape::push_new(graph, t, rng);
    let idx0 = ops_helpers::scalar_const(graph, 0i64, rng);
    let idx1 = ops_helpers::scalar_const(graph, 1i64, rng);
    let d0 = milli_ops::Gather::push_new(graph, shape, idx0, 0, rng);
    let d1 = milli_ops::Gather::push_new(graph, shape, idx1, 0, rng);
    let minus_one = ops_helpers::scalar_const(graph, -1i64, rng);

    // Reshape to 3D: [d0, d1, -1]
    let shape_3d = milli_ops::Concat::push_new(graph, vec![d0, d1, minus_one], 0, rng);
    let t_3d = milli_ops::Reshape::push_new(graph, t, shape_3d, false, rng);

    // Transpose [1, 0, 2]
    let swapped_3d = milli_ops::Transpose::push_new(graph, t_3d, Some(vec![1, 0, 2]), rng);

    // Reshape back: [d1, d0, *spatial_dims]
    let two = ops_helpers::scalar_const(graph, 2i64, rng);
    let rank = ops_helpers::rank(graph, t, rng);
    let spatial_shape = milli_ops::Slice::push_new(graph, shape, two, rank, None, None, rng);
    let out_shape = milli_ops::Concat::push_new(graph, vec![d1, d0, spatial_shape], 0, rng);
    milli_ops::Reshape::push_new(graph, swapped_3d, out_shape, false, rng)
}

/// Grouped weight rearrangement:
/// [C_in, C_out/g, K...] → reshape to [g, C_in/g, C_out/g, -1]
/// → Transpose [0, 2, 1, 3] → reshape to [C_out, C_in/g, *spatial]
fn build_weight_grouped(
    graph: &mut MilliOpGraph,
    w: GlobalId,
    group: i64,
    rng: &mut impl Rng,
) -> GlobalId {
    let w_shape = milli_ops::Shape::push_new(graph, w, rng);
    let idx_0 = ops_helpers::scalar_const(graph, 0i64, rng);
    let idx_1 = ops_helpers::scalar_const(graph, 1i64, rng);
    let g_const = ops_helpers::scalar_const(graph, group, rng);
    let c_in = milli_ops::Gather::push_new(graph, w_shape, idx_0, 0, rng);
    let c_in_per_g = milli_ops::SimpleBinary::div(graph, c_in, g_const, rng);
    let c_out_per_g = milli_ops::Gather::push_new(graph, w_shape, idx_1, 0, rng);
    let minus_one = ops_helpers::scalar_const(graph, -1i64, rng);

    // Reshape to [g, C_in/g, C_out/g, -1]
    let grouped_shape = milli_ops::Concat::push_new(
        graph,
        vec![g_const, c_in_per_g, c_out_per_g, minus_one],
        0,
        rng,
    );
    let w_grouped = milli_ops::Reshape::push_new(graph, w, grouped_shape, false, rng);

    // Transpose [0, 2, 1, 3] — swap c_in_per_g and c_out_per_g
    let w_swapped = milli_ops::Transpose::push_new(graph, w_grouped, Some(vec![0, 2, 1, 3]), rng);

    // Reshape to [C_out, C_in/g, *spatial_dims]
    let c_out = milli_ops::SimpleBinary::mul(graph, c_out_per_g, g_const, rng);
    let two = ops_helpers::scalar_const(graph, 2i64, rng);
    let rank = ops_helpers::rank(graph, w, rng);
    let spatial_shape = milli_ops::Slice::push_new(graph, w_shape, two, rank, None, None, rng);
    let final_shape =
        milli_ops::Concat::push_new(graph, vec![c_out, c_in_per_g, spatial_shape], 0, rng);
    milli_ops::Reshape::push_new(graph, w_swapped, final_shape, false, rng)
}

/// Reverse all spatial dims (axes [2, ..., rank-1]) using Slice with step=-1.
/// Axes, starts, ends, and steps are all built dynamically from the tensor's rank.
fn reverse_spatial_dims(graph: &mut MilliOpGraph, t: GlobalId, rng: &mut impl Rng) -> GlobalId {
    let two = ops_helpers::scalar_const(graph, 2i64, rng);
    let one = ops_helpers::scalar_const(graph, 1i64, rng);
    let rank = ops_helpers::rank(graph, t, rng);

    // nd = rank - 2 (number of spatial dims)
    let nd = milli_ops::SimpleBinary::sub(graph, rank, two, rng);

    // axes = [2, 3, ..., rank-1]
    let axes = milli_ops::Range::push_new(graph, two, rank, one, rng);

    // starts = ConstantOfShape(nd, value=i64::MAX)
    let starts = milli_ops::ConstantOfShape::push_new(
        graph,
        crate::numeric_scalar::NumericScalar::from_i64(i64::MAX),
        nd,
        rng,
    );
    // ends = ConstantOfShape(nd, value=i64::MIN)
    let ends = milli_ops::ConstantOfShape::push_new(
        graph,
        crate::numeric_scalar::NumericScalar::from_i64(i64::MIN),
        nd,
        rng,
    );
    // steps = ConstantOfShape(nd, value=-1)
    let steps = milli_ops::ConstantOfShape::push_new(
        graph,
        crate::numeric_scalar::NumericScalar::from_i64(-1),
        nd,
        rng,
    );

    milli_ops::Slice::push_new(graph, t, starts, ends, Some(steps), Some(axes), rng)
}

// ─── Padding ─────────────────────────────────────────────────────────────────
//
// Computes pad amounts dynamically from the weight shape:
//   kernel_shape = Shape(w)[2:]
//   effective_kernel = (kernel_shape - 1) * dilation + 1
//   pad_begin = effective_kernel - 1 - orig_pads_begin
//   pad_end   = effective_kernel - 1 - orig_pads_end + output_padding
//
// When output_shape is provided, pads are computed from:
//   total_pad = stride * (input_size - 1) + output_padding + effective_kernel - output_shape
//   pad_begin = total_pad / 2, pad_end = total_pad - pad_begin

#[allow(clippy::too_many_arguments)]
fn build_padding(
    graph: &mut MilliOpGraph,
    input: GlobalId,
    w: GlobalId,
    dilations_attr: &[i64],
    pads_attr: &[i64],
    output_padding_attr: &[i64],
    output_shape_attr: &[i64],
    auto_pad: &AutoPad,
    original_input: GlobalId,
    strides_attr: &[i64],
    rng: &mut impl Rng,
) -> GlobalId {
    let w_shape = milli_ops::Shape::push_new(graph, w, rng);
    let two = ops_helpers::scalar_const(graph, 2i64, rng);
    let one = ops_helpers::scalar_const(graph, 1i64, rng);
    let rank = ops_helpers::rank(graph, w, rng);

    // kernel_shape = Shape(w)[2:]  (spatial kernel dims)
    let kernel_dims = milli_ops::Slice::push_new(graph, w_shape, two, rank, None, None, rng);

    // nd = rank - 2
    let nd = milli_ops::SimpleBinary::sub(graph, rank, two, rng);

    // dilation vector: from attribute or default all-1s
    let dilation_vec = if dilations_attr.is_empty() {
        milli_ops::ConstantOfShape::push_new(
            graph,
            crate::numeric_scalar::NumericScalar::from_i64(1),
            nd,
            rng,
        )
    } else {
        milli_ops::Constant::from_vec(graph, dilations_attr.to_vec(), rng)
    };

    // effective_kernel = (kernel - 1) * dilation + 1
    let km1 = milli_ops::SimpleBinary::sub(graph, kernel_dims, one, rng);
    let dkm1 = milli_ops::SimpleBinary::mul(graph, dilation_vec, km1, rng);
    let eff_kernel = milli_ops::SimpleBinary::add(graph, dkm1, one, rng);

    // eff_kernel - 1 = dkm1
    // (reuse dkm1 directly since eff_kernel - 1 = dilation * (kernel - 1))

    // output_padding vector: from attribute or default all-0s
    let output_pad_vec = if output_padding_attr.is_empty() {
        milli_ops::ConstantOfShape::push_new(
            graph,
            crate::numeric_scalar::NumericScalar::from_i64(0),
            nd,
            rng,
        )
    } else {
        milli_ops::Constant::from_vec(graph, output_padding_attr.to_vec(), rng)
    };

    // First compute ONNX pads (onnx_pb, onnx_pe), then convert to
    // decomposition padding:
    //   decomp_pb = dkm1 - onnx_pb
    //   decomp_pe = dkm1 - onnx_pe + output_padding

    let stride_vec = if strides_attr.is_empty() {
        milli_ops::ConstantOfShape::push_new(
            graph,
            crate::numeric_scalar::NumericScalar::from_i64(1),
            nd,
            rng,
        )
    } else {
        milli_ops::Constant::from_vec(graph, strides_attr.to_vec(), rng)
    };

    let (onnx_pb, onnx_pe) = if !output_shape_attr.is_empty() {
        // output_shape provided: compute ONNX pads from desired output shape.
        // total_onnx_pad = stride*(input-1) + output_padding + eff_kernel - output_shape
        let input_shape = milli_ops::Shape::push_new(graph, original_input, rng);
        let input_spatial =
            milli_ops::Slice::push_new(graph, input_shape, two, rank, None, None, rng);

        let out_shape_vec = milli_ops::Constant::from_vec(graph, output_shape_attr.to_vec(), rng);

        let in_m1 = milli_ops::SimpleBinary::sub(graph, input_spatial, one, rng);
        let s_times_in_m1 = milli_ops::SimpleBinary::mul(graph, stride_vec, in_m1, rng);
        let total = milli_ops::SimpleBinary::add(graph, s_times_in_m1, output_pad_vec, rng);
        let total = milli_ops::SimpleBinary::add(graph, total, eff_kernel, rng);
        let total_pad = milli_ops::SimpleBinary::sub(graph, total, out_shape_vec, rng);

        let two_val = ops_helpers::scalar_const(graph, 2i64, rng);
        let pb = milli_ops::SimpleBinary::div(graph, total_pad, two_val, rng);
        let pe = milli_ops::SimpleBinary::sub(graph, total_pad, pb, rng);
        (pb, pe)
    } else if matches!(auto_pad, AutoPad::SameUpper | AutoPad::SameLower) {
        // SAME: output = input * stride.
        // total_onnx_pad = stride*(input-1) + eff_kernel - input*stride
        //                = eff_kernel - stride
        let total_pad = milli_ops::SimpleBinary::sub(graph, eff_kernel, stride_vec, rng);
        let zero_vec = milli_ops::ConstantOfShape::push_new(
            graph,
            crate::numeric_scalar::NumericScalar::from_i64(0),
            nd,
            rng,
        );
        let total_pad = milli_ops::SimpleBinary::max(graph, total_pad, zero_vec, rng);

        let two_val = ops_helpers::scalar_const(graph, 2i64, rng);
        let half = milli_ops::SimpleBinary::div(graph, total_pad, two_val, rng);
        let other_half = milli_ops::SimpleBinary::sub(graph, total_pad, half, rng);
        if matches!(auto_pad, AutoPad::SameUpper) {
            // SameUpper: less at begin, more at end
            (half, other_half)
        } else {
            (other_half, half)
        }
    } else {
        // NotSet / Valid: ONNX pads from attribute (default all-0s).
        let n_spatial_known = pads_attr.len() / 2;
        let pb = if n_spatial_known > 0 {
            milli_ops::Constant::from_vec(graph, pads_attr[..n_spatial_known].to_vec(), rng)
        } else {
            milli_ops::ConstantOfShape::push_new(
                graph,
                crate::numeric_scalar::NumericScalar::from_i64(0),
                nd,
                rng,
            )
        };
        let pe = if n_spatial_known > 0 {
            milli_ops::Constant::from_vec(graph, pads_attr[n_spatial_known..].to_vec(), rng)
        } else {
            milli_ops::ConstantOfShape::push_new(
                graph,
                crate::numeric_scalar::NumericScalar::from_i64(0),
                nd,
                rng,
            )
        };
        (pb, pe)
    };

    // Convert ONNX pads to decomposition padding:
    //   decomp_pb = dkm1 - onnx_pb
    //   decomp_pe = dkm1 - onnx_pe + output_padding
    let pad_begin = milli_ops::SimpleBinary::sub(graph, dkm1, onnx_pb, rng);
    let pad_end_no_op = milli_ops::SimpleBinary::sub(graph, dkm1, onnx_pe, rng);
    let pad_end = milli_ops::SimpleBinary::add(graph, pad_end_no_op, output_pad_vec, rng);

    // Build ONNX-format pad tensor: [0, 0, pb0, pb1, ..., 0, 0, pe0, pe1, ...]
    let zeros_2 = milli_ops::Constant::from_vec(graph, vec![0i64, 0], rng);
    let pad_tensor =
        milli_ops::Concat::push_new(graph, vec![zeros_2, pad_begin, zeros_2, pad_end], 0, rng);

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
