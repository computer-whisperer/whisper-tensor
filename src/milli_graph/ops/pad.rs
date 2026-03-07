use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::dtype::DType;
use crate::graph::GlobalId;
use crate::milli_graph::MilliOpGraphError;
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::numeric_tensor::NumericTensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use typenum::P1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PadMode {
    Constant,
    Reflect,
    Edge,
    Wrap,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pad {
    global_id: GlobalId,
    output: GlobalId,
    data: GlobalId,
    pads: GlobalId,
    constant_value: Option<GlobalId>,
    axes: Option<GlobalId>,
    mode: PadMode,
}

impl Pad {
    pub fn push_new(
        graph: &mut crate::milli_graph::MilliOpGraph,
        data: GlobalId,
        pads: GlobalId,
        constant_value: Option<GlobalId>,
        axes: Option<GlobalId>,
        mode: PadMode,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        let output = graph.get_new_tensor_id(rng);
        let node = Self {
            output,
            data,
            pads,
            constant_value,
            axes,
            mode,
            global_id: GlobalId::new(rng),
        };
        graph.push_op(AnyMilliOp::Pad(node));
        output
    }

    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl rand::Rng) {
        self.global_id = GlobalId::new(rng);
        super::remap(&mut self.output, map);
        super::remap(&mut self.data, map);
        super::remap(&mut self.pads, map);
        super::remap_opt(&mut self.constant_value, map);
        super::remap_opt(&mut self.axes, map);
    }
}

impl crate::graph::Node for Pad {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "Pad".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId> + '_> {
        let mut res = vec![self.data, self.pads];
        if let Some(cv) = self.constant_value {
            res.push(cv);
        }
        if let Some(ax) = self.axes {
            res.push(ax);
        }
        Box::new(res.into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId> + '_> {
        Box::new([self.output].into_iter())
    }
}

/// Layout info for N-dimensional index decomposition.
struct NdLayout {
    strides: Vec<usize>,
}

impl NdLayout {
    fn from_shape(shape: &[u64]) -> Self {
        let rank = shape.len();
        let mut strides = vec![1usize; rank];
        for i in (0..rank.saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1] as usize;
        }
        Self { strides }
    }
}

impl MilliOp for Pad {
    fn eval(
        &self,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>, MilliOpGraphError>
    {
        let data = &inputs[&self.data];
        let rank = data.rank();
        let shape = data.shape();

        // Parse pads tensor
        let pads_raw: Vec<i64> = inputs[&self.pads]
            .cast(DType::I64, backend)?
            .try_to_rank::<P1>()?
            .try_into()?;

        // Parse axes (optional)
        let axes: Vec<usize> = if let Some(axes_id) = self.axes {
            let raw: Vec<i64> = inputs[&axes_id]
                .cast(DType::I64, backend)?
                .try_to_rank::<P1>()?
                .try_into()?;
            raw.iter()
                .map(|&a| {
                    if a < 0 {
                        (rank as i64 + a) as usize
                    } else {
                        a as usize
                    }
                })
                .collect()
        } else {
            (0..rank).collect()
        };

        // Build per-axis (begin_pad, end_pad)
        let num_axes = axes.len();
        if pads_raw.len() != 2 * num_axes {
            return Err(MilliOpGraphError::InvalidInput(format!(
                "Pad: expected pads length {}, got {}",
                2 * num_axes,
                pads_raw.len()
            )));
        }
        let mut begin_pads = vec![0i64; rank];
        for (i, &axis) in axes.iter().enumerate() {
            begin_pads[axis] = pads_raw[i];
        }

        let mut end_pads = vec![0i64; rank];
        for (i, &axis) in axes.iter().enumerate() {
            end_pads[axis] = pads_raw[num_axes + i];
        }
        let out_shape: Vec<u64> = (0..rank)
            .map(|i| (shape[i] as i64 + begin_pads[i] + end_pads[i]) as u64)
            .collect();

        // Cast data to f32 for manipulation, remember original dtype
        let orig_dtype = data.dtype();
        let data_f32 = data.cast(DType::F32, backend)?;

        // Get constant value (default 0.0)
        let const_val: f32 = if let Some(cv_id) = self.constant_value {
            let cv = inputs[&cv_id].cast(DType::F32, backend)?;
            let cv = cv.reshape(vec![1u64], backend)?;
            let v: Vec<f32> = cv.try_to_rank::<P1>()?.try_into()?;
            v[0]
        } else {
            0.0
        };

        // Get flat data
        let data_flat: Vec<f32> = data_f32.to_ndarray()?.flatten().try_into()?;

        // Compute strides for input and output
        let in_layout = NdLayout::from_shape(&shape);
        let out_layout = NdLayout::from_shape(&out_shape);
        let out_total: usize = out_shape.iter().map(|&s| s as usize).product();

        let mut out_flat = vec![const_val; out_total];

        match self.mode {
            PadMode::Constant => {
                // Copy input data into output at offset position
                let in_total: usize = shape.iter().map(|&s| s as usize).product();
                for (flat_idx, &val) in data_flat.iter().enumerate().take(in_total) {
                    let mut remaining = flat_idx;
                    let mut out_offset = 0;
                    for (d, &bp) in begin_pads.iter().enumerate().take(rank) {
                        let coord = remaining / in_layout.strides[d];
                        remaining %= in_layout.strides[d];
                        out_offset += (coord as i64 + bp) as usize * out_layout.strides[d];
                    }
                    out_flat[out_offset] = val;
                }
            }
            PadMode::Edge => {
                fill_pad(
                    &data_flat, &mut out_flat, &shape, &out_shape,
                    &begin_pads, &in_layout, &out_layout, rank,
                    |idx, dim| idx.clamp(0, dim as i64 - 1) as usize,
                );
            }
            PadMode::Reflect => {
                fill_pad(
                    &data_flat, &mut out_flat, &shape, &out_shape,
                    &begin_pads, &in_layout, &out_layout, rank,
                    |idx, dim| reflect_index(idx, dim as usize),
                );
            }
            PadMode::Wrap => {
                fill_pad(
                    &data_flat, &mut out_flat, &shape, &out_shape,
                    &begin_pads, &in_layout, &out_layout, rank,
                    |idx, dim| ((idx % dim as i64 + dim as i64) % dim as i64) as usize,
                );
            }
        }

        let output = NumericTensor::<DynRank>::from_vec_shape(
            out_flat,
            out_shape.iter().map(|&s| s as usize).collect(),
        )
        .map_err(|e| MilliOpGraphError::InvalidInput(format!("Pad output creation failed: {e}")))?;
        let output = output.cast(orig_dtype, backend)?;

        Ok(Box::new([(self.output, output)].into_iter()))
    }
}

/// Fill entire output by mapping each output coordinate back to an input coordinate.
#[allow(clippy::too_many_arguments)]
fn fill_pad(
    data: &[f32],
    out: &mut [f32],
    in_shape: &[u64],
    out_shape: &[u64],
    begin_pads: &[i64],
    in_layout: &NdLayout,
    out_layout: &NdLayout,
    rank: usize,
    index_fn: impl Fn(i64, u64) -> usize,
) {
    let total: usize = out_shape.iter().map(|&s| s as usize).product();
    for (flat_idx, out_val) in out.iter_mut().enumerate().take(total) {
        let mut remaining = flat_idx;
        let mut in_flat = 0;
        for d in 0..rank {
            let out_coord = remaining / out_layout.strides[d];
            remaining %= out_layout.strides[d];
            let in_coord = index_fn(out_coord as i64 - begin_pads[d], in_shape[d]);
            in_flat += in_coord * in_layout.strides[d];
        }
        *out_val = data[in_flat];
    }
}

/// Reflect index: maps negative or out-of-bounds indices via reflection.
fn reflect_index(idx: i64, dim: usize) -> usize {
    if dim <= 1 {
        return 0;
    }
    let period = 2 * (dim as i64 - 1);
    let mut i = ((idx % period) + period) % period;
    if i >= dim as i64 {
        i = period - i;
    }
    i as usize
}
