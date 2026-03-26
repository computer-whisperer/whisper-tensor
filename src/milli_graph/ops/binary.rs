use crate::pool::Pool;
use super::AccumulationMode;
use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::dtype::DType;
use crate::milli_graph::MilliOpGraphError;
use crate::milli_graph::ops::MilliOp;
use crate::migration::numeric_tensor::NumericTensor;
use crate::numeric_dtype::NumericDType;
use crate::scalar_info::ScalarInfoTyped;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) enum WhichSimpleBinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Modulo(Option<bool>),
    And,
    Or,
    Xor,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    Equal,
    Greater,
    GreaterOrEqual,
    Less,
    LessOrEqual,
    Max,
    Min,
    BitShiftLeft,
    BitShiftRight,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleBinary {
    global_id: GlobalId,
    pub(crate) label: Option<String>,
    output: GlobalId,
    which_op: WhichSimpleBinaryOp,
    a: GlobalId,
    b: GlobalId,
}

use crate::graph::{GlobalId, Node};
use crate::milli_graph::MilliOpGraph;
use crate::milli_graph::ops::AnyMilliOp;

impl SimpleBinary {
    pub(crate) fn which_op(&self) -> &WhichSimpleBinaryOp {
        &self.which_op
    }

    fn push_new(
        graph: &mut MilliOpGraph,
        a: GlobalId,
        b: GlobalId,
        which_op: WhichSimpleBinaryOp,
        label: Option<String>,
        rng: &mut impl Rng,
    ) -> GlobalId {
        let output = graph.get_new_tensor_id(rng);
        let node = Self {
            global_id: GlobalId::new(rng),
            label,
            output,
            which_op,
            a,
            b,
        };
        graph.push_op(AnyMilliOp::SimpleBinary(node));
        output
    }
    pub fn add(graph: &mut MilliOpGraph, a: GlobalId, b: GlobalId, rng: &mut impl Rng) -> GlobalId {
        Self::push_new(graph, a, b, WhichSimpleBinaryOp::Add, None, rng)
    }

    pub fn sub(graph: &mut MilliOpGraph, a: GlobalId, b: GlobalId, rng: &mut impl Rng) -> GlobalId {
        Self::push_new(graph, a, b, WhichSimpleBinaryOp::Sub, None, rng)
    }

    pub fn mul(graph: &mut MilliOpGraph, a: GlobalId, b: GlobalId, rng: &mut impl Rng) -> GlobalId {
        Self::push_new(graph, a, b, WhichSimpleBinaryOp::Mul, None, rng)
    }

    pub fn div(graph: &mut MilliOpGraph, a: GlobalId, b: GlobalId, rng: &mut impl Rng) -> GlobalId {
        Self::push_new(graph, a, b, WhichSimpleBinaryOp::Div, None, rng)
    }

    pub fn modulo(
        graph: &mut MilliOpGraph,
        a: GlobalId,
        b: GlobalId,
        fmod: Option<bool>,
        rng: &mut impl Rng,
    ) -> GlobalId {
        Self::push_new(graph, a, b, WhichSimpleBinaryOp::Modulo(fmod), None, rng)
    }
    pub fn and(graph: &mut MilliOpGraph, a: GlobalId, b: GlobalId, rng: &mut impl Rng) -> GlobalId {
        Self::push_new(graph, a, b, WhichSimpleBinaryOp::And, None, rng)
    }

    pub fn or(graph: &mut MilliOpGraph, a: GlobalId, b: GlobalId, rng: &mut impl Rng) -> GlobalId {
        Self::push_new(graph, a, b, WhichSimpleBinaryOp::Or, None, rng)
    }

    pub fn xor(graph: &mut MilliOpGraph, a: GlobalId, b: GlobalId, rng: &mut impl Rng) -> GlobalId {
        Self::push_new(graph, a, b, WhichSimpleBinaryOp::Xor, None, rng)
    }

    pub fn bitwise_and(
        graph: &mut MilliOpGraph,
        a: GlobalId,
        b: GlobalId,
        rng: &mut impl Rng,
    ) -> GlobalId {
        Self::push_new(graph, a, b, WhichSimpleBinaryOp::BitwiseAnd, None, rng)
    }

    pub fn bitwise_or(
        graph: &mut MilliOpGraph,
        a: GlobalId,
        b: GlobalId,
        rng: &mut impl Rng,
    ) -> GlobalId {
        Self::push_new(graph, a, b, WhichSimpleBinaryOp::BitwiseOr, None, rng)
    }

    pub fn bitwise_xor(
        graph: &mut MilliOpGraph,
        a: GlobalId,
        b: GlobalId,
        rng: &mut impl Rng,
    ) -> GlobalId {
        Self::push_new(graph, a, b, WhichSimpleBinaryOp::BitwiseXor, None, rng)
    }

    pub fn equal(
        graph: &mut MilliOpGraph,
        a: GlobalId,
        b: GlobalId,
        rng: &mut impl Rng,
    ) -> GlobalId {
        Self::push_new(graph, a, b, WhichSimpleBinaryOp::Equal, None, rng)
    }

    pub fn greater(
        graph: &mut MilliOpGraph,
        a: GlobalId,
        b: GlobalId,
        rng: &mut impl Rng,
    ) -> GlobalId {
        Self::push_new(graph, a, b, WhichSimpleBinaryOp::Greater, None, rng)
    }

    pub fn greater_or_equal(
        graph: &mut MilliOpGraph,
        a: GlobalId,
        b: GlobalId,
        rng: &mut impl Rng,
    ) -> GlobalId {
        Self::push_new(graph, a, b, WhichSimpleBinaryOp::GreaterOrEqual, None, rng)
    }

    pub fn less(
        graph: &mut MilliOpGraph,
        a: GlobalId,
        b: GlobalId,
        rng: &mut impl Rng,
    ) -> GlobalId {
        Self::push_new(graph, a, b, WhichSimpleBinaryOp::Less, None, rng)
    }

    pub fn less_or_equal(
        graph: &mut MilliOpGraph,
        a: GlobalId,
        b: GlobalId,
        rng: &mut impl Rng,
    ) -> GlobalId {
        Self::push_new(graph, a, b, WhichSimpleBinaryOp::LessOrEqual, None, rng)
    }

    pub fn max(graph: &mut MilliOpGraph, a: GlobalId, b: GlobalId, rng: &mut impl Rng) -> GlobalId {
        Self::push_new(graph, a, b, WhichSimpleBinaryOp::Max, None, rng)
    }

    pub fn min(graph: &mut MilliOpGraph, a: GlobalId, b: GlobalId, rng: &mut impl Rng) -> GlobalId {
        Self::push_new(graph, a, b, WhichSimpleBinaryOp::Min, None, rng)
    }

    pub fn bitshift_left(
        graph: &mut MilliOpGraph,
        a: GlobalId,
        b: GlobalId,
        rng: &mut impl Rng,
    ) -> GlobalId {
        Self::push_new(graph, a, b, WhichSimpleBinaryOp::BitShiftLeft, None, rng)
    }

    pub fn bitshift_right(
        graph: &mut MilliOpGraph,
        a: GlobalId,
        b: GlobalId,
        rng: &mut impl Rng,
    ) -> GlobalId {
        Self::push_new(graph, a, b, WhichSimpleBinaryOp::BitShiftRight, None, rng)
    }
}

impl SimpleBinary {
    pub fn lower_to_nano(&self, ctx: &mut crate::nano_graph::NanoLoweringContext) -> crate::milli_graph::ops::LowerResult {
        use crate::nano_graph::lower::TensorAtomMap;
        use crate::nano_graph::ops::{ScalarBinOp, ScalarOp};
        use crate::nano_graph::pattern::AtomId;

        let all_infos = ctx.all_infos;
        let mut inputs_iter = Node::inputs(self);
        let a_id = inputs_iter.next().unwrap();
        let b_id = inputs_iter.next().unwrap();
        let out_id = Node::outputs(self).next().unwrap();

        let (Some(a_map), Some(b_map)) = (
            ctx.tensor_map.get(&a_id).cloned(),
            ctx.tensor_map.get(&b_id).cloned(),
        ) else {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        };
        let Some(out_info) = all_infos.get(&out_id) else {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        };

        let out_dt = crate::nano_graph::NanoLoweringContext::ndt(out_info);
        // Comparison ops output BOOL but compute in input precision.
        // Use input dtype for compute_dtype when output is BOOL.
        let input_dt = all_infos
            .get(&a_id)
            .map(|i| crate::nano_graph::NanoLoweringContext::ndt(i))
            .unwrap_or(out_dt);
        let compute_dt = if out_dt == NumericDType::BOOL {
            input_dt
        } else {
            out_dt
        };
        let scalar_op = match self.which_op() {
            WhichSimpleBinaryOp::Add => ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: compute_dt,
            },
            WhichSimpleBinaryOp::Sub => ScalarOp::Binary {
                op: ScalarBinOp::Sub,
                compute_dtype: compute_dt,
            },
            WhichSimpleBinaryOp::Mul => ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: compute_dt,
            },
            WhichSimpleBinaryOp::Div => ScalarOp::Binary {
                op: ScalarBinOp::Div,
                compute_dtype: compute_dt,
            },
            WhichSimpleBinaryOp::Max => ScalarOp::Binary {
                op: ScalarBinOp::Max,
                compute_dtype: compute_dt,
            },
            WhichSimpleBinaryOp::Min => ScalarOp::Binary {
                op: ScalarBinOp::Min,
                compute_dtype: compute_dt,
            },
            WhichSimpleBinaryOp::Modulo(_) => ScalarOp::Binary {
                op: ScalarBinOp::Mod,
                compute_dtype: compute_dt,
            },
            WhichSimpleBinaryOp::Equal => ScalarOp::Binary {
                op: ScalarBinOp::Equal,
                compute_dtype: compute_dt,
            },
            WhichSimpleBinaryOp::Greater => ScalarOp::Binary {
                op: ScalarBinOp::Greater,
                compute_dtype: compute_dt,
            },
            WhichSimpleBinaryOp::GreaterOrEqual => ScalarOp::Binary {
                op: ScalarBinOp::GreaterOrEqual,
                compute_dtype: compute_dt,
            },
            WhichSimpleBinaryOp::Less => ScalarOp::Binary {
                op: ScalarBinOp::Less,
                compute_dtype: compute_dt,
            },
            WhichSimpleBinaryOp::LessOrEqual => ScalarOp::Binary {
                op: ScalarBinOp::LessOrEqual,
                compute_dtype: compute_dt,
            },
            WhichSimpleBinaryOp::And => ScalarOp::Binary {
                op: ScalarBinOp::And,
                compute_dtype: compute_dt,
            },
            WhichSimpleBinaryOp::Or => ScalarOp::Binary {
                op: ScalarBinOp::Or,
                compute_dtype: compute_dt,
            },
            WhichSimpleBinaryOp::Xor | WhichSimpleBinaryOp::BitwiseXor => ScalarOp::Binary {
                op: ScalarBinOp::Xor,
                compute_dtype: compute_dt,
            },
            WhichSimpleBinaryOp::BitwiseAnd => ScalarOp::Binary {
                op: ScalarBinOp::And,
                compute_dtype: compute_dt,
            },
            WhichSimpleBinaryOp::BitwiseOr => ScalarOp::Binary {
                op: ScalarBinOp::Or,
                compute_dtype: compute_dt,
            },
            WhichSimpleBinaryOp::BitShiftLeft | WhichSimpleBinaryOp::BitShiftRight => {
                return crate::milli_graph::ops::LowerResult::Unsupported;
            }
        };

        let Some((layout, known_dims, sym_dims, count)) = ctx.classify_dims(out_info) else {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        };
        let count = count.max(1);
        let strides = TensorAtomMap::compute_strides(&known_dims);

        let out_tmp = TensorAtomMap::simple(
            AtomId(0),
            count,
            out_dt,
            layout.clone(),
            strides.clone(),
            sym_dims.clone(),
        );

        let a_info = all_infos.get(&a_id);
        let b_info = all_infos.get(&b_id);
        let input_a = ctx.compute_input_ref(&out_tmp, &a_map, out_info, a_info.unwrap_or(out_info));
        let input_b = ctx.compute_input_ref(&out_tmp, &b_map, out_info, b_info.unwrap_or(out_info));

        let base_id = ctx.nano.push_group(
            count,
            out_dt,
            scalar_op,
            sym_dims.clone(),
            vec![input_a, input_b],
        );

        ctx.tensor_map.insert(
            out_id,
            TensorAtomMap::simple(base_id, count, out_dt, layout, strides, sym_dims),
        );
        crate::milli_graph::ops::LowerResult::Lowered
    }

    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl Rng) {
        self.global_id = GlobalId::new(rng);
        super::remap(&mut self.output, map);
        super::remap(&mut self.a, map);
        super::remap(&mut self.b, map);
    }
}

impl MilliOp for SimpleBinary {
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

        let a_info = known_inputs
            .get(&self.a)
            .ok_or(MilliOpGraphError::UnableToInfer)?;
        let b_info = known_inputs
            .get(&self.b)
            .ok_or(MilliOpGraphError::UnableToInfer)?;

        // Determine output dtype: comparison ops produce Bool, others match input.
        let out_dtype = match self.which_op {
            WhichSimpleBinaryOp::Equal
            | WhichSimpleBinaryOp::Greater
            | WhichSimpleBinaryOp::GreaterOrEqual
            | WhichSimpleBinaryOp::Less
            | WhichSimpleBinaryOp::LessOrEqual
            | WhichSimpleBinaryOp::And
            | WhichSimpleBinaryOp::Or
            | WhichSimpleBinaryOp::Xor => NumericDType::Bool,
            _ => a_info.dtype(),
        };

        // Compute symbolic output info for the hint.
        let out_info = {
            let a_ranked = a_info.as_ranked();
            let b_ranked = b_info.as_ranked();
            if let (Some(a_ranked), Some(b_ranked)) = (a_ranked, b_ranked) {
                let a_dims = a_ranked.shape();
                let b_dims = b_ranked.shape();
                if let Ok(out_dims) = super::infer_multidirectional_broadcasting_shape(
                    &[a_dims.clone(), b_dims.clone()],
                    symbolic_resolver,
                ) {
                    TensorInfo::from_dtype_and_shape_scalars(out_dtype, &out_dims)
                } else {
                    let a_shape = a_info.shape(symbolic_resolver);
                    let b_shape = b_info.shape(symbolic_resolver);
                    let out_rank = super::infer_multidirectional_broadcasting_rank(
                        &[a_shape, b_shape],
                        symbolic_resolver,
                    )?;
                    let first_elem = crate::scalar_info::ScalarInfo::Symbolic(
                        crate::symbolic_scalar::SymbolicScalar::new(out_dtype, symbolic_resolver),
                    );
                    TensorInfo::new_from_first_element_and_rank(first_elem, out_rank, symbolic_resolver)
                }
            } else {
                let a_shape = a_info.shape(symbolic_resolver);
                let b_shape = b_info.shape(symbolic_resolver);
                let out_rank = super::infer_multidirectional_broadcasting_rank(
                    &[a_shape, b_shape],
                    symbolic_resolver,
                )?;
                let first_elem = crate::scalar_info::ScalarInfo::Symbolic(
                    crate::symbolic_scalar::SymbolicScalar::new(out_dtype, symbolic_resolver),
                );
                TensorInfo::new_from_first_element_and_rank(first_elem, out_rank, symbolic_resolver)
            }
        };

        // If both inputs are concrete, try constant fold with output hints.
        if let Some(results) = super::constant_fold(self, known_inputs, &[(self.output, out_info.clone_with_pool(pool))], pool) {
            return Ok(results);
        }

        Ok(vec![((self.output, out_info))])
    }

    fn eval(
        &self,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
        _config: &super::MilliEvalConfig,
        backend: &mut EvalBackend,
    ) -> Result<Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>, MilliOpGraphError>
    {
        let a = &inputs[&self.a];
        let b = &inputs[&self.b];
        let out = match self.which_op {
            WhichSimpleBinaryOp::Add => NumericTensor::<DynRank>::add(a, b, backend)?,
            WhichSimpleBinaryOp::Sub => NumericTensor::<DynRank>::sub(a, b, backend)?,
            WhichSimpleBinaryOp::Mul => NumericTensor::<DynRank>::mul(a, b, backend)?,
            WhichSimpleBinaryOp::Div => NumericTensor::<DynRank>::div(a, b, backend)?,
            WhichSimpleBinaryOp::Modulo(fmod) => {
                let is_float =
                    [DType::F64, DType::F32, DType::BF16, DType::F16].contains(&a.dtype());
                let fmod = if is_float {
                    true
                } else {
                    fmod.unwrap_or(false)
                };
                if fmod {
                    NumericTensor::<DynRank>::fmod(a, b, backend)?
                } else {
                    NumericTensor::<DynRank>::imod(a, b, backend)?
                }
            }
            WhichSimpleBinaryOp::And => NumericTensor::<DynRank>::and(a, b, backend)?,
            WhichSimpleBinaryOp::Or => NumericTensor::<DynRank>::or(a, b, backend)?,
            WhichSimpleBinaryOp::Xor => NumericTensor::<DynRank>::xor(a, b, backend)?,
            WhichSimpleBinaryOp::BitwiseAnd => {
                NumericTensor::<DynRank>::bitwise_and(a, b, backend)?
            }
            WhichSimpleBinaryOp::BitwiseOr => NumericTensor::<DynRank>::bitwise_or(a, b, backend)?,
            WhichSimpleBinaryOp::BitwiseXor => {
                NumericTensor::<DynRank>::bitwise_xor(a, b, backend)?
            }
            WhichSimpleBinaryOp::Equal => NumericTensor::<DynRank>::equal(a, b, backend)?,
            WhichSimpleBinaryOp::Greater => NumericTensor::<DynRank>::greater(a, b, backend)?,
            WhichSimpleBinaryOp::GreaterOrEqual => {
                NumericTensor::<DynRank>::greater_or_equal(a, b, backend)?
            }
            WhichSimpleBinaryOp::Less => NumericTensor::<DynRank>::less(a, b, backend)?,
            WhichSimpleBinaryOp::LessOrEqual => {
                NumericTensor::<DynRank>::less_or_equal(a, b, backend)?
            }
            WhichSimpleBinaryOp::Max => NumericTensor::<DynRank>::max(a, b, backend)?,
            WhichSimpleBinaryOp::Min => NumericTensor::<DynRank>::min(a, b, backend)?,
            WhichSimpleBinaryOp::BitShiftLeft => {
                NumericTensor::<DynRank>::bitshift_left(a, b, backend)?
            }
            WhichSimpleBinaryOp::BitShiftRight => {
                NumericTensor::<DynRank>::bitshift_right(a, b, backend)?
            }
        };
        Ok(Box::new([(self.output, out)].into_iter()))
    }

    fn backward(
        &self,
        output_grads: &HashMap<GlobalId, GlobalId>,
        graph: &mut MilliOpGraph,
        rng: &mut impl Rng,
    ) -> Option<HashMap<GlobalId, GlobalId>> {
        let grad_output = *output_grads.get(&self.output)?;

        // Compute per-input gradients as (input_id, grad_tensor_id) pairs.
        // Using a Vec instead of HashMap so that when self.a == self.b,
        // both contributions are preserved and summed below.
        let pairs: Vec<(GlobalId, GlobalId)> = match self.which_op {
            // d/da(a+b) = 1, d/db(a+b) = 1
            WhichSimpleBinaryOp::Add => {
                vec![(self.a, grad_output), (self.b, grad_output)]
            }
            // d/da(a-b) = 1, d/db(a-b) = -1
            WhichSimpleBinaryOp::Sub => {
                let neg_grad = super::SimpleUnaryOp::neg(graph, grad_output, rng);
                vec![(self.a, grad_output), (self.b, neg_grad)]
            }
            // d/da(a*b) = b, d/db(a*b) = a
            WhichSimpleBinaryOp::Mul => {
                let grad_a = SimpleBinary::mul(graph, grad_output, self.b, rng);
                let grad_b = SimpleBinary::mul(graph, grad_output, self.a, rng);
                vec![(self.a, grad_a), (self.b, grad_b)]
            }
            // d/da(a/b) = 1/b, d/db(a/b) = -a/b^2
            WhichSimpleBinaryOp::Div => {
                let grad_a = SimpleBinary::div(graph, grad_output, self.b, rng);
                let b_sq = SimpleBinary::mul(graph, self.b, self.b, rng);
                let a_over_b_sq = SimpleBinary::div(graph, self.a, b_sq, rng);
                let neg = super::SimpleUnaryOp::neg(graph, a_over_b_sq, rng);
                let grad_b = SimpleBinary::mul(graph, grad_output, neg, rng);
                vec![(self.a, grad_a), (self.b, grad_b)]
            }
            _ => return None,
        };

        // Reduce gradients to match input shapes (un-broadcast)
        // and accumulate: if self.a == self.b, sum both gradient contributions
        let mut result = HashMap::new();
        for (input_id, grad_id) in pairs {
            let shape = super::Shape::push_new(graph, input_id, rng);
            let reduced = super::SumTo::push_new(graph, grad_id, shape, rng);
            result
                .entry(input_id)
                .and_modify(|existing: &mut GlobalId| {
                    *existing = SimpleBinary::add(graph, *existing, reduced, rng);
                })
                .or_insert(reduced);
        }
        Some(result)
    }

    fn lower_to_nano(&self, ctx: &mut crate::nano_graph::NanoLoweringContext) -> crate::milli_graph::ops::LowerResult {
        SimpleBinary::lower_to_nano(self, ctx)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pow {
    global_id: GlobalId,
    pub(crate) label: Option<String>,
    output: GlobalId,
    a: GlobalId,
    b: GlobalId,
}

impl Pow {
    pub fn push_new(
        graph: &mut MilliOpGraph,
        a: GlobalId,
        b: GlobalId,
        rng: &mut impl Rng,
    ) -> GlobalId {
        Self::push_new_with_label(graph, a, b, None, rng)
    }

    pub fn push_new_with_label(
        graph: &mut MilliOpGraph,
        a: GlobalId,
        b: GlobalId,
        label: Option<String>,
        rng: &mut impl Rng,
    ) -> GlobalId {
        let output = graph.get_new_tensor_id(rng);
        let node = Self {
            output,
            a,
            b,
            global_id: GlobalId::new(rng),
            label,
        };
        graph.push_op(AnyMilliOp::Pow(node));
        output
    }
}

impl Pow {
    pub fn lower_to_nano(&self, ctx: &mut crate::nano_graph::NanoLoweringContext) -> crate::milli_graph::ops::LowerResult {
        use crate::nano_graph::lower::TensorAtomMap;
        use crate::nano_graph::ops::{ScalarBinOp, ScalarOp};
        use crate::nano_graph::pattern::AtomId;

        let all_infos = ctx.all_infos;
        let mut inputs_iter = Node::inputs(self);
        let a_id = inputs_iter.next().unwrap();
        let b_id = inputs_iter.next().unwrap();
        let out_id = Node::outputs(self).next().unwrap();

        let (Some(a_map), Some(b_map)) = (
            ctx.tensor_map.get(&a_id).cloned(),
            ctx.tensor_map.get(&b_id).cloned(),
        ) else {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        };
        let Some(out_info) = all_infos.get(&out_id) else {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        };

        let Some((layout, known_dims, sym_dims, count)) = ctx.classify_dims(out_info) else {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        };
        let count = count.max(1);
        let strides = TensorAtomMap::compute_strides(&known_dims);

        let dt = crate::nano_graph::NanoLoweringContext::ndt(out_info);
        let out_tmp = TensorAtomMap::simple(
            AtomId(0),
            count,
            dt,
            layout.clone(),
            strides.clone(),
            sym_dims.clone(),
        );

        let a_info = all_infos.get(&a_id);
        let b_info = all_infos.get(&b_id);
        let input_a = ctx.compute_input_ref(&out_tmp, &a_map, out_info, a_info.unwrap_or(out_info));
        let input_b = ctx.compute_input_ref(&out_tmp, &b_map, out_info, b_info.unwrap_or(out_info));

        let base_id = ctx.nano.push_group(
            count,
            dt,
            ScalarOp::Binary {
                op: ScalarBinOp::Pow,
                compute_dtype: dt,
            },
            sym_dims.clone(),
            vec![input_a, input_b],
        );

        ctx.tensor_map.insert(
            out_id,
            TensorAtomMap::simple(base_id, count, dt, layout, strides, sym_dims),
        );
        crate::milli_graph::ops::LowerResult::Lowered
    }

    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl Rng) {
        self.global_id = GlobalId::new(rng);
        super::remap(&mut self.output, map);
        super::remap(&mut self.a, map);
        super::remap(&mut self.b, map);
    }
}

impl MilliOp for Pow {
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

        let a_info = known_inputs
            .get(&self.a)
            .ok_or(MilliOpGraphError::UnableToInfer)?;
        let b_info = known_inputs
            .get(&self.b)
            .ok_or(MilliOpGraphError::UnableToInfer)?;

        // Output dtype = input dtype.
        let out_dtype = a_info.dtype();

        // Compute symbolic output info for the hint.
        let out_info = {
            let a_ranked = a_info.as_ranked();
            let b_ranked = b_info.as_ranked();
            if let (Some(a_ranked), Some(b_ranked)) = (a_ranked, b_ranked) {
                let a_dims = a_ranked.shape();
                let b_dims = b_ranked.shape();
                if let Ok(out_dims) = super::infer_multidirectional_broadcasting_shape(
                    &[a_dims.clone(), b_dims.clone()],
                    symbolic_resolver,
                ) {
                    TensorInfo::from_dtype_and_shape_scalars(out_dtype, &out_dims)
                } else {
                    let a_shape = a_info.shape(symbolic_resolver);
                    let b_shape = b_info.shape(symbolic_resolver);
                    let out_rank = super::infer_multidirectional_broadcasting_rank(
                        &[a_shape, b_shape],
                        symbolic_resolver,
                    )?;
                    let first_elem = crate::scalar_info::ScalarInfo::Symbolic(
                        crate::symbolic_scalar::SymbolicScalar::new(out_dtype, symbolic_resolver),
                    );
                    TensorInfo::new_from_first_element_and_rank(first_elem, out_rank, symbolic_resolver)
                }
            } else {
                let a_shape = a_info.shape(symbolic_resolver);
                let b_shape = b_info.shape(symbolic_resolver);
                let out_rank = super::infer_multidirectional_broadcasting_rank(
                    &[a_shape, b_shape],
                    symbolic_resolver,
                )?;
                let first_elem = crate::scalar_info::ScalarInfo::Symbolic(
                    crate::symbolic_scalar::SymbolicScalar::new(out_dtype, symbolic_resolver),
                );
                TensorInfo::new_from_first_element_and_rank(first_elem, out_rank, symbolic_resolver)
            }
        };

        // If both inputs are concrete, try constant fold with output hints.
        if let Some(results) = super::constant_fold(self, known_inputs, &[(self.output, out_info.clone_with_pool(pool))], pool) {
            return Ok(results);
        }

        Ok(vec![((self.output, out_info))])
    }

    fn eval(
        &self,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
        _config: &super::MilliEvalConfig,
        backend: &mut EvalBackend,
    ) -> Result<Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>, MilliOpGraphError>
    {
        let out = NumericTensor::<DynRank>::pow(&inputs[&self.a], &inputs[&self.b], backend)?;
        Ok(Box::new([(self.output, out)].into_iter()))
    }

    fn lower_to_nano(&self, ctx: &mut crate::nano_graph::NanoLoweringContext) -> crate::milli_graph::ops::LowerResult {
        Pow::lower_to_nano(self, ctx)
    }
}

fn default_f32() -> NumericDType {
    NumericDType::F32
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatMul {
    global_id: GlobalId,
    pub(crate) label: Option<String>,
    output: GlobalId,
    a: GlobalId,
    b: GlobalId,
    /// Expected dtype of the input tensors (e.g. BF16).
    #[serde(default = "default_f32")]
    input_dtype: NumericDType,
    /// Precision of the A[m,k] * B[k,n] products before accumulation.
    /// When wider than input_dtype, the full-precision product is kept
    /// (e.g. BF16 inputs → F32 products, matching tensor-core behavior).
    #[serde(default = "default_f32")]
    product_dtype: NumericDType,
    /// Precision for accumulating (summing) the products across K.
    #[serde(default = "default_f32")]
    accumulate_dtype: NumericDType,
    /// Dtype of the output tensor.
    #[serde(default = "default_f32")]
    output_dtype: NumericDType,
    /// Accumulation order for the contraction (K) dimension.
    #[serde(default)]
    accumulation_mode: AccumulationMode,
}

impl MatMul {
    pub fn input_dtype(&self) -> NumericDType {
        self.input_dtype
    }
    pub fn product_dtype(&self) -> NumericDType {
        self.product_dtype
    }
    pub fn accumulate_dtype(&self) -> NumericDType {
        self.accumulate_dtype
    }
    pub fn output_dtype(&self) -> NumericDType {
        self.output_dtype
    }
    pub fn accumulation_mode(&self) -> AccumulationMode {
        self.accumulation_mode
    }

    #[allow(clippy::too_many_arguments)]
    pub fn push_new(
        graph: &mut MilliOpGraph,
        a: GlobalId,
        b: GlobalId,
        input_dtype: NumericDType,
        product_dtype: NumericDType,
        accumulate_dtype: NumericDType,
        output_dtype: NumericDType,
        rng: &mut impl Rng,
    ) -> GlobalId {
        Self::push_new_with_label(
            graph,
            a,
            b,
            input_dtype,
            product_dtype,
            accumulate_dtype,
            output_dtype,
            None,
            rng,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn push_new_with_label(
        graph: &mut MilliOpGraph,
        a: GlobalId,
        b: GlobalId,
        input_dtype: NumericDType,
        product_dtype: NumericDType,
        accumulate_dtype: NumericDType,
        output_dtype: NumericDType,
        label: Option<String>,
        rng: &mut impl Rng,
    ) -> GlobalId {
        let output = graph.get_new_tensor_id(rng);
        let node = Self {
            output,
            a,
            b,
            input_dtype,
            product_dtype,
            accumulate_dtype,
            output_dtype,
            accumulation_mode: AccumulationMode::default(),
            global_id: GlobalId::new(rng),
            label,
        };
        graph.push_op(AnyMilliOp::MatMul(node));
        output
    }

    /// Standard precision convention: BF16/F16 inputs produce F32 products,
    /// accumulate in F32, output matches input dtype. All other types use
    /// native precision throughout.
    pub fn push_new_default_precision(
        graph: &mut MilliOpGraph,
        a: GlobalId,
        b: GlobalId,
        input_dtype: NumericDType,
        rng: &mut impl Rng,
    ) -> GlobalId {
        let (prod_dt, acc_dt, out_dt) = Self::default_precision_for(input_dtype);
        Self::push_new(graph, a, b, input_dtype, prod_dt, acc_dt, out_dt, rng)
    }

    /// Returns (product_dtype, accumulate_dtype, output_dtype) for the standard
    /// precision convention given an input dtype.
    pub fn default_precision_for(input_dtype: NumericDType) -> (NumericDType, NumericDType, NumericDType) {
        match input_dtype {
            NumericDType::BF16 | NumericDType::F16 => (NumericDType::F32, NumericDType::F32, input_dtype),
            _ => (input_dtype, input_dtype, input_dtype),
        }
    }
}

impl MatMul {
    pub fn lower_to_nano(&self, ctx: &mut crate::nano_graph::NanoLoweringContext) -> crate::milli_graph::ops::LowerResult {
        use crate::nano_graph::lower::DimKind;
        use crate::nano_graph::lower::TensorAtomMap;
        use crate::nano_graph::ops::{ReduceKind, ScalarBinOp, ScalarOp};
        use crate::nano_graph::pattern::{AtomId, InputRef};

        let all_infos = ctx.all_infos;
        let mut inputs_iter = Node::inputs(self);
        let a_id = inputs_iter.next().unwrap();
        let b_id = inputs_iter.next().unwrap();
        let out_id = Node::outputs(self).next().unwrap();

        let (Some(a_map), Some(b_map)) = (
            ctx.tensor_map.get(&a_id).cloned(),
            ctx.tensor_map.get(&b_id).cloned(),
        ) else {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        };
        let Some(out_info) = all_infos.get(&out_id) else {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        };

        // MatMul operates on the last 2 tensor dims: A=[...,M,K] @ B=[...,K,N].
        // We look at the full layout (not just known dims) to handle symbolic M.
        // Requirements:
        //   - K (last dim of A, second-to-last of B): must be Known, must match
        //   - N (last dim of B): must be Known
        //   - M (second-to-last of A): can be Known or Symbolic
        //   - Known batch dims must match between A and B
        let a_layout = &a_map.layout;
        let b_layout = &b_map.layout;

        if a_layout.len() < 2 || b_layout.len() < 2 {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        }

        // Extract K from last dim of A and second-to-last of B.
        let k = match (&a_layout[a_layout.len() - 1], &b_layout[b_layout.len() - 2]) {
            (DimKind::Known(ka), DimKind::Known(kb)) if ka == kb && *ka > 0 => *ka,
            _ => {
                return crate::milli_graph::ops::LowerResult::Unsupported;
            }
        };

        // Extract N from last dim of B.
        let n = match &b_layout[b_layout.len() - 1] {
            DimKind::Known(n) if *n > 0 => *n,
            _ => {
                return crate::milli_graph::ops::LowerResult::Unsupported;
            }
        };

        // M from second-to-last of A: can be known or symbolic.
        let m_known: Option<u64> = match &a_layout[a_layout.len() - 2] {
            DimKind::Known(m) => Some(*m),
            DimKind::Symbolic(_) => None,
        };

        // Extract known batch dims from A and B (everything except last 2).
        let a_batch_layout = &a_layout[..a_layout.len() - 2];
        let b_batch_layout = &b_layout[..b_layout.len() - 2];

        let a_batch_known: Vec<u64> = a_batch_layout
            .iter()
            .filter_map(|d| {
                if let DimKind::Known(s) = d {
                    Some(*s)
                } else {
                    None
                }
            })
            .collect();
        let b_batch_known: Vec<u64> = b_batch_layout
            .iter()
            .filter_map(|d| {
                if let DimKind::Known(s) = d {
                    Some(*s)
                } else {
                    None
                }
            })
            .collect();

        // B can have fewer batch dims (broadcasting). If B has batch dims, they must match.
        if !b_batch_known.is_empty() && a_batch_known != b_batch_known {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        }

        let batch_known_product: u64 = a_batch_known.iter().product::<u64>().max(1);

        // Classify output dims.
        let Some((out_layout, out_known_dims, out_sym_dims, out_count)) =
            ctx.classify_dims(out_info)
        else {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        };
        let out_count = out_count.max(1);

        if out_count > 64_000_000 {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        }

        // Use the MatMul's explicit dtype fields for nano group precision.
        // product_dtype: precision of A*B products (Mul groups).
        // accumulate_dtype: precision for summing products (ReduceSum groups).
        // output_dtype: final output precision (Identity cast-back if needed).
        let product_dtype = self.product_dtype();
        let accumulate_dtype = self.accumulate_dtype();
        let out_dtype = self.output_dtype();

        // Use A and B's actual physical strides (may be non-row-major after Transpose).
        let a_known_dims: Vec<u64> = a_layout
            .iter()
            .filter_map(|d| {
                if let DimKind::Known(s) = d {
                    Some(*s)
                } else {
                    None
                }
            })
            .collect();
        let a_strides = &a_map.known_strides;

        let b_known_dims: Vec<u64> = b_layout
            .iter()
            .filter_map(|d| {
                if let DimKind::Known(s) = d {
                    Some(*s)
                } else {
                    None
                }
            })
            .collect();
        let b_strides = &b_map.known_strides;

        // A's K dim is the last known dim. B's K dim is second-to-last known dim.
        // B's N dim is the last known dim.
        let a_k_known_idx = a_known_dims.len() - 1;
        let _b_k_known_idx = b_known_dims.len() - 2;

        // The number of row groups: batch_known_product * (M if known, else 1).
        let m_groups = m_known.unwrap_or(1);
        let num_row_groups = (batch_known_product * m_groups) as usize;
        let n_u64 = n;

        // For each row group, compute the A and B base offsets within their atoms.
        // A's known dims: [...batch_known, (M if known), K]
        // B's known dims: [...batch_known, K, N]
        let a_m_known_idx = if m_known.is_some() {
            Some(a_known_dims.len() - 2) // M is second-to-last known dim
        } else {
            None
        };

        // Merged matmul Mul groups: one per row instead of K per row.
        // Each merged group has K*N atoms. Atom j in the merged group computes:
        //   A[m, j/N] * B[j/N, j%N]
        //
        // Input 0: StridedBroadcast { base: A[m,0], stride: 1, repeat: N }
        //   → each block of N atoms broadcasts the same A element
        // Input 1: Affine { base: B[0,0], stride: 1 }
        //   → B is row-major, so B[k,n] = B_base + k*N + n = B_base + j
        //
        // The ReduceSum groups still have count=N and use SymAffine with
        // stride_k = N to hop between k-blocks within the merged Mul group.
        let mut mul_base_id = None;
        let k_u64 = k;
        let merged_mul_count = k_u64 * n_u64;

        for g in 0..num_row_groups {
            let m_idx = if m_known.is_some() {
                g as u64 % m_groups
            } else {
                0
            };
            let batch_idx = if m_known.is_some() {
                g as u64 / m_groups
            } else {
                g as u64
            };

            // Compute A's base offset: set batch indices + m_idx, K=0.
            let mut a_offset = 0u64;
            let mut batch_rem = batch_idx;
            let a_batch_strides = TensorAtomMap::compute_strides(&a_batch_known);
            for (i, &stride) in a_batch_strides.iter().enumerate() {
                if stride > 0 {
                    let idx = batch_rem / stride;
                    batch_rem %= stride;
                    a_offset += idx * a_strides[i];
                }
            }
            if let Some(m_ki) = a_m_known_idx {
                a_offset += m_idx * a_strides[m_ki];
            }

            // Compute B's base offset: set batch indices, K=0, N=0.
            let mut b_offset = 0u64;
            if !b_batch_known.is_empty() {
                let mut batch_rem_b = batch_idx;
                let b_batch_strides = TensorAtomMap::compute_strides(&b_batch_known);
                for (i, &stride) in b_batch_strides.iter().enumerate() {
                    if stride > 0 {
                        let idx = batch_rem_b / stride;
                        batch_rem_b %= stride;
                        b_offset += idx * b_strides[i];
                    }
                }
            }

            // B[0, 0] — base of B for this batch
            let b_base = b_map.base_id.offset(b_offset);

            // Input 0: A elements for this row, each repeated N times.
            // For each k, we need A[batch, m, k]. With physical strides,
            // offset = a_offset + k * a_k_stride.
            // For segmented A (from Concat), use atom_id_for_element to resolve.
            let a_k_stride = a_strides[a_k_known_idx];
            let input_a = if a_map.segments.is_empty() {
                // Simple A: StridedBroadcast with physical k-stride.
                let a_row_base = a_map.base_id.offset(a_offset);
                InputRef::strided_broadcast(a_row_base, a_k_stride as i64, n_u64)
            } else {
                // Segmented A: build explicit per-atom mapping.
                // Atom j in the merged group reads A[m, j/N], repeated N times.
                let a_rowmajor = TensorAtomMap::compute_strides(&a_known_dims);
                let mut ids = Vec::with_capacity(merged_mul_count as usize);
                for j in 0..merged_mul_count {
                    let k_idx = j / n_u64;
                    // Recover batch and m indices from a_offset using physical strides,
                    // then set K to k_idx and compute flat logical index.
                    let mut indices = vec![0u64; a_known_dims.len()];
                    let mut remaining = a_offset;
                    for d in 0..a_known_dims.len() {
                        if d == a_k_known_idx {
                            indices[d] = 0;
                        } else if a_strides[d] > 0 {
                            indices[d] = remaining / a_strides[d];
                            remaining %= a_strides[d];
                        }
                    }
                    indices[a_k_known_idx] = k_idx;
                    // Convert to flat logical index (row-major)
                    let flat_a: u64 = indices
                        .iter()
                        .zip(a_rowmajor.iter())
                        .map(|(&idx, &stride)| idx * stride)
                        .sum();
                    ids.push(a_map.atom_id_for_element(flat_a));
                }
                crate::nano_graph::NanoLoweringContext::compress_explicit(ids)
            };

            // Input 1: B elements for all K*N positions in the merged group.
            // Atom j reads B[k=j/N, n=j%N].
            // With physical strides: offset = k * b_k_stride + n * b_n_stride.
            // If B is row-major (b_k_stride=N, b_n_stride=1), this simplifies to Affine{stride:1}.
            let b_k_known_idx = b_known_dims.len() - 2;
            let b_n_known_idx = b_known_dims.len() - 1;
            let b_k_stride = b_strides[b_k_known_idx];
            let b_n_stride = b_strides[b_n_known_idx];
            let input_b = if b_k_stride == n_u64 && b_n_stride == 1 {
                // Row-major B: simple affine.
                InputRef::affine(b_base, 1)
            } else {
                // Non-row-major B: build explicit mapping.
                let mut ids = Vec::with_capacity(merged_mul_count as usize);
                for j in 0..merged_mul_count {
                    let k_idx = j / n_u64;
                    let n_idx = j % n_u64;
                    let offset = k_idx * b_k_stride + n_idx * b_n_stride;
                    ids.push(b_base.offset(offset));
                }
                crate::nano_graph::NanoLoweringContext::compress_explicit(ids)
            };

            let base = ctx.nano.push_group(
                merged_mul_count,
                product_dtype,
                ScalarOp::Binary {
                    op: ScalarBinOp::Mul,
                    compute_dtype: product_dtype,
                },
                out_sym_dims.clone(),
                vec![input_a, input_b],
            );

            if mul_base_id.is_none() {
                mul_base_id = Some(base);
            }
        }

        let mul_base = mul_base_id.unwrap();

        // ReduceSum: one group per row, each with N atoms.
        // reduce_stride = N so that stepping k hops between k-blocks within the
        // merged Mul group. Input stride = 1 for consecutive output atoms.
        let mut reduce_base_id = None;

        for g in 0..num_row_groups {
            let row_mul_base = AtomId(mul_base.0 + (g as u64) * merged_mul_count);

            let base = ctx.nano.push_group(
                n_u64,
                out_dtype,
                ScalarOp::Reduce {
                    kind: ReduceKind::Sum,
                    reduce_count: k_u64,
                    reduce_stride: n_u64 as i64,
                    compute_dtype: accumulate_dtype,
                },
                out_sym_dims.clone(),
                vec![InputRef::affine(row_mul_base, 1)],
            );

            if reduce_base_id.is_none() {
                reduce_base_id = Some(base);
            }
        }

        let base_id = reduce_base_id.unwrap();

        ctx.tensor_map.insert(
            out_id,
            TensorAtomMap::simple(
                base_id,
                out_count,
                out_dtype,
                out_layout,
                TensorAtomMap::compute_strides(&out_known_dims),
                out_sym_dims,
            ),
        );
        crate::milli_graph::ops::LowerResult::Lowered
    }

    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl Rng) {
        self.global_id = GlobalId::new(rng);
        super::remap(&mut self.output, map);
        super::remap(&mut self.a, map);
        super::remap(&mut self.b, map);
        // input_dtype, product_dtype, accumulate_dtype, output_dtype are preserved as-is
    }
}

impl MilliOp for MatMul {
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

        let a_info = known_inputs
            .get(&self.a)
            .ok_or(MilliOpGraphError::UnableToInfer)?;
        let b_info = known_inputs
            .get(&self.b)
            .ok_or(MilliOpGraphError::UnableToInfer)?;

        // MatMul output dtype comes from the struct's explicit field.
        let out_dtype = self.output_dtype;

        // Compute output info: try per-dim shape inference first, then rank-only fallback.
        // A[...,M,K] @ B[...,K,N] -> [...,M,N]
        // Batch dims are broadcast, last two follow matmul rules.
        let out_info = if let (Some(a_ranked), Some(b_ranked)) = (a_info.as_ranked(), b_info.as_ranked()) {
            let a_dims = a_ranked.shape();
            let b_dims = b_ranked.shape();
            let a_rank = a_dims.len();
            let b_rank = b_dims.len();

            let out_dims: Option<Vec<ScalarInfoTyped<u64>>> = if a_rank >= 1 && b_rank >= 1 {
                if a_rank >= 2 && b_rank >= 2 {
                    // Standard case: batch broadcast + [M,K]@[K,N]->[M,N]
                    let a_batch = &a_dims[..a_rank - 2];
                    let b_batch = &b_dims[..b_rank - 2];
                    let batch = super::infer_multidirectional_broadcasting_shape(
                        &[a_batch.to_vec(), b_batch.to_vec()],
                        symbolic_resolver,
                    )
                    .ok();
                    batch.map(|mut out| {
                        out.push(a_dims[a_rank - 2].clone()); // M
                        out.push(b_dims[b_rank - 1].clone()); // N
                        out
                    })
                } else if a_rank == 1 && b_rank >= 2 {
                    // vector @ matrix: [K] @ [...,K,N] -> [...,N]
                    let mut out = b_dims[..b_rank - 2].to_vec();
                    out.push(b_dims[b_rank - 1].clone()); // N
                    Some(out)
                } else if a_rank >= 2 && b_rank == 1 {
                    // matrix @ vector: [...,M,K] @ [K] -> [...,M]
                    let mut out = a_dims[..a_rank - 2].to_vec();
                    out.push(a_dims[a_rank - 2].clone()); // M
                    Some(out)
                } else {
                    // both rank 1: dot product -> scalar []
                    Some(vec![])
                }
            } else {
                None
            };

            if let Some(out_dims) = out_dims {
                TensorInfo::from_dtype_and_shape_scalars(out_dtype, &out_dims)
            } else {
                // Ranked inputs but couldn't compute shape — fall through to rank-only
                let first_elem = crate::scalar_info::ScalarInfo::Symbolic(
                    crate::symbolic_scalar::SymbolicScalar::new(out_dtype, symbolic_resolver),
                );
                TensorInfo::new_from_first_element_and_rank(first_elem, a_info.rank(), symbolic_resolver)
            }
        } else {
            // Fallback: rank-only inference.
            let out_rank = match (a_info.rank(), b_info.rank()) {
                (
                    crate::scalar_info::ScalarInfoTyped::Numeric(a_rank),
                    crate::scalar_info::ScalarInfoTyped::Numeric(b_rank),
                ) => {
                    let out_r = if a_rank >= 2 && b_rank >= 2 {
                        a_rank.max(b_rank)
                    } else if a_rank == 1 && b_rank >= 2 {
                        b_rank - 1
                    } else if a_rank >= 2 && b_rank == 1 {
                        a_rank - 1
                    } else {
                        0
                    };
                    crate::scalar_info::ScalarInfoTyped::Numeric(out_r)
                }
                _ => crate::scalar_info::ScalarInfoTyped::Symbolic(
                    crate::symbolic_scalar::SymbolicScalarTyped::new(symbolic_resolver),
                ),
            };

            let first_elem = crate::scalar_info::ScalarInfo::Symbolic(
                crate::symbolic_scalar::SymbolicScalar::new(out_dtype, symbolic_resolver),
            );
            TensorInfo::new_from_first_element_and_rank(first_elem, out_rank, symbolic_resolver)
        };

        // If both inputs are concrete, try constant fold via nano+pool_eval path.
        if let Some(results) = super::constant_fold(self, known_inputs, &[(self.output, out_info.clone_with_pool(pool))], pool) {
            return Ok(results);
        }

        Ok(vec![((self.output, out_info))])
    }

    fn eval(
        &self,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
        config: &super::MilliEvalConfig,
        backend: &mut EvalBackend,
    ) -> Result<Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>, MilliOpGraphError>
    {
        let a_input = &inputs[&self.a];
        let b_input = &inputs[&self.b];
        let accumulate_legacy = self.accumulate_dtype.to_legacy();
        let accumulate_dtype = if accumulate_legacy != a_input.dtype() {
            Some(accumulate_legacy)
        } else {
            None
        };
        let mode = if config.relaxed_accumulation {
            AccumulationMode::Pairwise // allows BLAS
        } else {
            self.accumulation_mode
        };
        let out = NumericTensor::<DynRank>::matmul(
            a_input,
            b_input,
            accumulate_dtype,
            self.output_dtype.to_legacy(),
            mode,
            backend,
        )?;
        Ok(Box::new([(self.output, out)].into_iter()))
    }

    fn backward(
        &self,
        output_grads: &HashMap<GlobalId, GlobalId>,
        graph: &mut crate::milli_graph::MilliOpGraph,
        rng: &mut impl Rng,
    ) -> Option<HashMap<GlobalId, GlobalId>> {
        let grad_output = *output_grads.get(&self.output)?;
        // d/dA (A @ B) = grad @ B^T
        let b_t = super::Transpose::push_new(graph, self.b, Some(vec![-1, -2]), rng);
        let grad_a = MatMul::push_new(
            graph,
            grad_output,
            b_t,
            self.input_dtype,
            self.product_dtype,
            self.accumulate_dtype,
            self.output_dtype,
            rng,
        );
        // d/dB (A @ B) = A^T @ grad
        let a_t = super::Transpose::push_new(graph, self.a, Some(vec![-1, -2]), rng);
        let grad_b = MatMul::push_new(
            graph,
            a_t,
            grad_output,
            self.input_dtype,
            self.product_dtype,
            self.accumulate_dtype,
            self.output_dtype,
            rng,
        );

        // Reduce gradients to match input shapes (un-broadcast batch dims)
        // and accumulate if self.a == self.b (self-matmul)
        let mut result = HashMap::new();
        for (input_id, grad_id) in [(self.a, grad_a), (self.b, grad_b)] {
            let shape = super::Shape::push_new(graph, input_id, rng);
            let reduced = super::SumTo::push_new(graph, grad_id, shape, rng);
            result
                .entry(input_id)
                .and_modify(|existing: &mut GlobalId| {
                    *existing = SimpleBinary::add(graph, *existing, reduced, rng);
                })
                .or_insert(reduced);
        }
        Some(result)
    }

    fn lower_to_nano(&self, ctx: &mut crate::nano_graph::NanoLoweringContext) -> crate::milli_graph::ops::LowerResult {
        MatMul::lower_to_nano(self, ctx)
    }
}

impl Node for SimpleBinary {
    type OpKind = String;
    fn op_kind(&self) -> Self::OpKind {
        match self.which_op {
            WhichSimpleBinaryOp::Add => "Add",
            WhichSimpleBinaryOp::Sub => "Sub",
            WhichSimpleBinaryOp::Mul => "Mul",
            WhichSimpleBinaryOp::Div => "Div",
            WhichSimpleBinaryOp::Modulo(_) => "Modulo",
            WhichSimpleBinaryOp::And => "And",
            WhichSimpleBinaryOp::Or => "Or",
            WhichSimpleBinaryOp::Xor => "Xor",
            WhichSimpleBinaryOp::BitwiseAnd => "Bitwise And",
            WhichSimpleBinaryOp::BitwiseOr => "Bitwise Or",
            WhichSimpleBinaryOp::BitwiseXor => "Bitwise Xor",
            WhichSimpleBinaryOp::Equal => "Equal",
            WhichSimpleBinaryOp::Greater => "Greater",
            WhichSimpleBinaryOp::GreaterOrEqual => "Greater or Equal",
            WhichSimpleBinaryOp::Less => "Less",
            WhichSimpleBinaryOp::LessOrEqual => "Less or Equal",
            WhichSimpleBinaryOp::Max => "Max",
            WhichSimpleBinaryOp::Min => "Min",
            WhichSimpleBinaryOp::BitShiftLeft => "BitShiftLeft",
            WhichSimpleBinaryOp::BitShiftRight => "BitShiftRight",
        }
        .to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.a, self.b].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.output].into_iter())
    }
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
}

impl Node for Pow {
    type OpKind = String;
    fn op_kind(&self) -> Self::OpKind {
        "Pow".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.a, self.b].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.output].into_iter())
    }
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
}

impl Node for MatMul {
    type OpKind = String;
    fn op_kind(&self) -> Self::OpKind {
        "MatMul".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.a, self.b].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.output].into_iter())
    }
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
}
