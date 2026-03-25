use crate::pool::Pool;
use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::graph::{GlobalId, Node};
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError};
use crate::nano_graph::lower::{NanoLoweringContext, TensorAtomMap};
use crate::nano_graph::ops::ScalarOp;
use crate::nano_graph::pattern::AtomId;
use crate::migration::numeric_tensor::NumericTensor;
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
    pub fn lower_to_nano(&self, ctx: &mut crate::nano_graph::NanoLoweringContext) {
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
            ctx.lower_as_boundary_named(self, "Where");
            return;
        };
        let Some(out_info) = all_infos.get(&out_id) else {
            ctx.lower_as_boundary_named(self, "Where");
            return;
        };

        let Some((layout, known_dims, sym_dims, count)) = ctx.classify_dims(out_info) else {
            ctx.lower_as_boundary_named(self, "Where");
            return;
        };
        let count = count.max(1);
        let strides = TensorAtomMap::compute_strides(&known_dims);

        let dt = NanoLoweringContext::ndt(out_info);
        let out_tmp = TensorAtomMap::simple(
            AtomId(0),
            count,
            dt,
            layout.clone(),
            strides.clone(),
            sym_dims.clone(),
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
            vec![input_cond, input_x, input_y],
        );

        ctx.tensor_map.insert(
            out_id,
            TensorAtomMap::simple(base_id, count, dt, layout, strides, sym_dims),
        );
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
    fn infer<'p, P: Pool + 'p>(
        &self,
        known_inputs: &HashMap<GlobalId, crate::tensor_info::TensorInfo<'p, P>>,
        symbolic_resolver: &mut crate::symbolic_scalar::SymbolicResolver,
        pool: &'p P,
    ) -> Result<
        Vec<(GlobalId, crate::tensor_info::TensorInfo<'p, P>)>,
        MilliOpGraphError,
    > {
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

        // If all concrete, fall back to eval.
        if let Some(results) = super::constant_fold(self, known_inputs, pool) {
            return Ok(results);
        }

        let out_dtype = x_info.dtype();

        // Try per-dim broadcast shape inference.
        if let (Some(c_ranked), Some(x_ranked), Some(y_ranked)) = (
            cond_info.as_ranked(),
            x_info.as_ranked(),
            y_info.as_ranked(),
        ) && let Ok(out_dims) = super::infer_multidirectional_broadcasting_shape(
            &[c_ranked.shape(), x_ranked.shape(), y_ranked.shape()],
            symbolic_resolver,
        ) {
            let out_info = TensorInfo::from_dtype_and_shape_scalars(out_dtype, &out_dims);
            return Ok(vec![((self.output, out_info))]);
        }

        // Fallback: rank-only inference.
        let c_shape = cond_info.shape(symbolic_resolver);
        let x_shape = x_info.shape(symbolic_resolver);
        let y_shape = y_info.shape(symbolic_resolver);
        let out_rank = super::infer_multidirectional_broadcasting_rank(
            &[c_shape, x_shape, y_shape],
            symbolic_resolver,
        )?;
        let first_elem = crate::scalar_info::ScalarInfo::Symbolic(
            crate::symbolic_scalar::SymbolicScalar::new(out_dtype, symbolic_resolver),
        );
        let out_info =
            TensorInfo::new_from_first_element_and_rank(first_elem, out_rank, symbolic_resolver);
        Ok(vec![((self.output, out_info))])
    }

    fn eval(
        &self,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
        _config: &super::MilliEvalConfig,
        backend: &mut EvalBackend,
    ) -> Result<Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>, MilliOpGraphError>
    {
        let out = inputs[&self.condition].where_op(&inputs[&self.x], &inputs[&self.y], backend)?;
        Ok(Box::new([(self.output, out)].into_iter()))
    }
}
