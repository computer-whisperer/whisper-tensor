use rand::Rng;
use crate::graph::{GlobalId, Node};
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError};
use crate::nano_graph::lower::{DimKind, NanoLoweringContext, TensorAtomMap};
use crate::nano_graph::ops::ScalarOp;
use crate::nano_graph::pattern::{AtomId, GroupInput};
use crate::pool::Pool;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Where {
    global_id: GlobalId,
    pub(crate) label: Option<String>,
    output: GlobalId,
    condition: GlobalId,
    x: GlobalId,
    y: GlobalId,
}

impl Where {
    pub fn push_new(
        graph: &mut MilliOpGraph,
        condition: GlobalId,
        x: GlobalId,
        y: GlobalId,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        Self::push_new_with_label(graph, condition, x, y, None, rng)
    }

    pub fn push_new_with_label(
        graph: &mut MilliOpGraph,
        condition: GlobalId,
        x: GlobalId,
        y: GlobalId,
        label: Option<String>,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        let output = graph.get_new_tensor_id(rng);
        let node = Self {
            global_id: GlobalId::new(rng),
            label,
            output,
            condition,
            x,
            y,
        };
        graph.push_op(AnyMilliOp::Where(node));
        output
    }
}

impl Where {
    pub fn lower_to_nano<'p, P: crate::pool::Pool + 'p>(
        &self,
        ctx: &mut crate::nano_graph::NanoLoweringContext<'_, 'p, P>,
    ) -> crate::milli_graph::ops::LowerResult {
        let all_infos = ctx.all_infos;
        let mut inputs_iter = Node::inputs(self);
        let cond_id = inputs_iter.next().unwrap();
        let x_id = inputs_iter.next().unwrap();
        let y_id = inputs_iter.next().unwrap();
        let out_id = Node::outputs(self).next().unwrap();

        let (Some(cond_map), Some(x_map), Some(y_map)) = (
            ctx.tensor_map.get(&cond_id).cloned(),
            ctx.tensor_map.get(&x_id).cloned(),
            ctx.tensor_map.get(&y_id).cloned(),
        ) else {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        };
        let Some(out_info) = all_infos.get(&out_id) else {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        };

        let Some(dims) = ctx.classify_dims(out_info) else {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        };
        let count: u64 = dims.iter().filter_map(|d| match d { DimKind::Known { size, .. } => Some(*size), _ => None }).product::<u64>().max(1);
        let sym_dims: Vec<_> = dims.iter().filter_map(|d| match d { DimKind::Sym { gc, .. } => Some(*gc), _ => None }).collect();

        let dt = NanoLoweringContext::ndt(out_info);
        let out_tmp = TensorAtomMap::simple(
            AtomId(0),
            count,
            dt,
            dims.clone(),
        );

        let cond_info = all_infos.get(&cond_id);
        let x_info = all_infos.get(&x_id);
        let y_info = all_infos.get(&y_id);
        let input_cond =
            ctx.compute_input_ref(&out_tmp, &cond_map, out_info, cond_info.unwrap_or(out_info));
        let input_x = ctx.compute_input_ref(&out_tmp, &x_map, out_info, x_info.unwrap_or(out_info));
        let input_y = ctx.compute_input_ref(&out_tmp, &y_map, out_info, y_info.unwrap_or(out_info));

        let base_id = ctx.nano.push_group(
            count,
            dt,
            ScalarOp::Select,
            sym_dims.clone(),
            vec![
                GroupInput::identity(input_cond, sym_dims.len()),
                GroupInput::identity(input_x, sym_dims.len()),
                GroupInput::identity(input_y, sym_dims.len()),
            ],
        );

        ctx.tensor_map.insert(
            out_id,
            TensorAtomMap::simple(base_id, count, dt, dims),
        );
        crate::milli_graph::ops::LowerResult::Lowered
    }

    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl rand::Rng) {
        self.global_id = GlobalId::new(rng);
        super::remap(&mut self.output, map);
        super::remap(&mut self.condition, map);
        super::remap(&mut self.x, map);
        super::remap(&mut self.y, map);
    }
}

impl Node for Where {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> String {
        "Where".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new([self.condition, self.x, self.y].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new([self.output].into_iter())
    }
}

impl MilliOp for Where {
    fn infer<'a, 'p, P: Pool + 'p>(
        &self,
        known_inputs: &HashMap<GlobalId, crate::tensor_info::TensorInfo<'a, 'p, P>>,
        rng: &mut impl Rng,
        pool: &'p P,
    ) -> Result<Vec<(GlobalId, crate::tensor_info::TensorInfo<'a, 'p, P>)>, MilliOpGraphError>
    where
        'p: 'a,
    {
        use crate::tensor_info::TensorInfo;

        let cond_info = known_inputs
            .get(&self.condition)
            .ok_or(MilliOpGraphError::UnableToInfer)?;
        let x_info = known_inputs
            .get(&self.x)
            .ok_or(MilliOpGraphError::UnableToInfer)?;
        let y_info = known_inputs
            .get(&self.y)
            .ok_or(MilliOpGraphError::UnableToInfer)?;

        let out_dtype = x_info.dtype();

        // Compute symbolic output info for the hint.
        let out_info = if let (Some(c_ranked), Some(x_ranked), Some(y_ranked)) = (
            cond_info.as_ranked(),
            x_info.as_ranked(),
            y_info.as_ranked(),
        ) && let Ok(out_dims) = super::infer_multidirectional_broadcasting_shape(
            &[c_ranked.shape(), x_ranked.shape(), y_ranked.shape()],
            rng,
        ) {
            TensorInfo::from_dtype_and_shape_scalars(out_dtype, &out_dims)
        } else {
            let c_shape = cond_info.shape(rng);
            let x_shape = x_info.shape(rng);
            let y_shape = y_info.shape(rng);
            let out_rank = super::infer_multidirectional_broadcasting_rank(
                &[c_shape, x_shape, y_shape],
                rng,
            )?;
            let first_elem = crate::scalar_info::ScalarInfo::Symbolic(
                crate::symbolic_scalar::SymbolicScalar::new(out_dtype, rng),
            );
            TensorInfo::new_from_first_element_and_rank(first_elem, out_rank, rng)
        };

        // If all concrete, try constant fold with output hints.
        if let Some(results) = super::constant_fold(
            self,
            known_inputs,
            &[(self.output, out_info.clone_with_pool(pool))],
            pool,
        ) {
            return Ok(results);
        }

        Ok(vec![(self.output, out_info)])
    }

    fn eval_new<'p, P2: crate::pool::Pool + 'p>(
        &self,
        inputs: &[crate::numeric_tensor::NumericTensorView<'_, crate::tensor_rank::DynRank>],
        pool: &'p P2,
    ) -> Result<
        Vec<crate::numeric_tensor::NumericTensor<'p, crate::tensor_rank::DynRank, P2>>,
        crate::nano_graph::pool_eval::PoolEvalError,
    > {
        use crate::numeric_tensor::{NumericTensor, TensorLayout};
        use crate::tensor_rank::DynRank;

        let cond = &inputs[0];
        let x = &inputs[1];
        let y = &inputs[2];
        let cond_shape = cond.shape();
        let x_shape = x.shape();
        let y_shape = y.shape();
        let dtype = x.dtype();

        // Compute broadcast output shape across all three inputs
        let out_rank = cond_shape.len().max(x_shape.len()).max(y_shape.len());
        let mut pc = vec![1u64; out_rank];
        let mut px = vec![1u64; out_rank];
        let mut py = vec![1u64; out_rank];
        for (i, &d) in cond_shape.iter().enumerate() {
            pc[out_rank - cond_shape.len() + i] = d;
        }
        for (i, &d) in x_shape.iter().enumerate() {
            px[out_rank - x_shape.len() + i] = d;
        }
        for (i, &d) in y_shape.iter().enumerate() {
            py[out_rank - y_shape.len() + i] = d;
        }
        let mut output_shape = vec![0u64; out_rank];
        for i in 0..out_rank {
            output_shape[i] = pc[i].max(px[i]).max(py[i]);
        }

        let out_numel: usize = output_shape.iter().product::<u64>() as usize;

        // Compute strides
        let mut out_strides = vec![1usize; out_rank];
        for i in (0..out_rank.saturating_sub(1)).rev() {
            out_strides[i] = out_strides[i + 1] * output_shape[i + 1] as usize;
        }
        let mut c_strides = vec![1usize; out_rank];
        for i in (0..out_rank.saturating_sub(1)).rev() {
            c_strides[i] = c_strides[i + 1] * pc[i + 1] as usize;
        }
        let mut x_strides = vec![1usize; out_rank];
        for i in (0..out_rank.saturating_sub(1)).rev() {
            x_strides[i] = x_strides[i + 1] * px[i + 1] as usize;
        }
        let mut y_strides = vec![1usize; out_rank];
        for i in (0..out_rank.saturating_sub(1)).rev() {
            y_strides[i] = y_strides[i + 1] * py[i + 1] as usize;
        }

        let layout = TensorLayout::<DynRank>::row_major(output_shape.clone(), dtype);
        let buf = pool
            .allocate(layout.buffer_size_bytes())
            .map_err(crate::nano_graph::pool_eval::PoolEvalError::Allocation)?;
        let mut out = NumericTensor::from_parts(buf, layout);

        for out_flat in 0..out_numel {
            let mut rem = out_flat;
            let mut c_flat = 0usize;
            let mut x_flat = 0usize;
            let mut y_flat = 0usize;
            for d in 0..out_rank {
                let coord = rem / out_strides[d];
                rem %= out_strides[d];
                c_flat += (if pc[d] == 1 { 0 } else { coord }) * c_strides[d];
                x_flat += (if px[d] == 1 { 0 } else { coord }) * x_strides[d];
                y_flat += (if py[d] == 1 { 0 } else { coord }) * y_strides[d];
            }
            let cv = cond.read_element(c_flat);
            let result = if cv.is_nonzero() {
                x.read_element(x_flat)
            } else {
                y.read_element(y_flat)
            };
            out.write_element(out_flat, result);
        }

        Ok(vec![out])
    }

    fn lower_to_nano<'p, P: crate::pool::Pool + 'p>(
        &self,
        ctx: &mut crate::nano_graph::NanoLoweringContext<'_, 'p, P>,
    ) -> crate::milli_graph::ops::LowerResult {
        Where::lower_to_nano(self, ctx)
    }
}
