use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::dtype::DType;
use crate::graph::GlobalId;
use crate::milli_graph::MilliOpGraphError;
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::migration::numeric_tensor::NumericTensor;
use rand::{Rng, RngExt};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Generates a tensor of random normal values with the same shape as the input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomNormalLike {
    global_id: GlobalId,
    pub(crate) label: Option<String>,
    output: GlobalId,
    input: GlobalId,
    dtype: Option<DType>,
    mean: f32,
    scale: f32,
    seed: Option<f32>,
}

impl RandomNormalLike {
    pub fn push_new(
        graph: &mut crate::milli_graph::MilliOpGraph,
        input: GlobalId,
        dtype: Option<DType>,
        mean: f32,
        scale: f32,
        seed: Option<f32>,
        rng: &mut impl Rng,
    ) -> GlobalId {
        Self::push_new_with_label(graph, input, dtype, mean, scale, seed, None, rng)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn push_new_with_label(
        graph: &mut crate::milli_graph::MilliOpGraph,
        input: GlobalId,
        dtype: Option<DType>,
        mean: f32,
        scale: f32,
        seed: Option<f32>,
        label: Option<String>,
        rng: &mut impl Rng,
    ) -> GlobalId {
        let output = graph.get_new_tensor_id(rng);
        let node = Self {
            global_id: GlobalId::new(rng),
            label,
            output,
            input,
            dtype,
            mean,
            scale,
            seed,
        };
        graph.push_op(AnyMilliOp::RandomNormalLike(node));
        output
    }

    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl Rng) {
        self.global_id = GlobalId::new(rng);
        super::remap(&mut self.output, map);
        super::remap(&mut self.input, map);
    }
}

impl crate::graph::Node for RandomNormalLike {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "RandomNormalLike".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId> + '_> {
        Box::new(std::iter::once(self.input))
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId> + '_> {
        Box::new(std::iter::once(self.output))
    }
}

/// Box-Muller transform: generate a standard normal sample from two uniform samples.
fn box_muller(rng: &mut impl Rng) -> (f32, f32) {
    let u1: f32 = rng.random_range(f32::EPSILON..1.0);
    let u2: f32 = rng.random_range(0.0f32..std::f32::consts::TAU);
    let r = (-2.0f32 * u1.ln()).sqrt();
    (r * u2.cos(), r * u2.sin())
}

fn fill_normal(rng: &mut impl Rng, total: usize, mean: f32, scale: f32) -> Vec<f32> {
    let mut values = Vec::with_capacity(total);
    while values.len() + 1 < total {
        let (z1, z2) = box_muller(rng);
        values.push(z1 * scale + mean);
        values.push(z2 * scale + mean);
    }
    if values.len() < total {
        let (z1, _) = box_muller(rng);
        values.push(z1 * scale + mean);
    }
    values
}

impl MilliOp for RandomNormalLike {
    fn infer<'p, P: crate::pool::Pool + 'p>(
        &self,
        known_inputs: &HashMap<GlobalId, crate::tensor_info::TensorInfo<'p, P>>,
        symbolic_resolver: &mut crate::symbolic_scalar::SymbolicResolver,
        _pool: &'p P,
    ) -> Result<Vec<(GlobalId, crate::tensor_info::TensorInfo<'p, P>)>, MilliOpGraphError> {
        use crate::numeric_dtype::NumericDType;
        use crate::scalar_info::{ScalarInfo, ScalarInfoTyped};
        use crate::symbolic_scalar::SymbolicScalar;
        use crate::tensor_info::TensorInfo;

        let input_info = known_inputs
            .get(&self.input)
            .ok_or(MilliOpGraphError::UnableToInfer)?;
        let out_dtype = self
            .dtype
            .map(|dt| NumericDType::from_legacy(dt).unwrap())
            .unwrap_or_else(|| input_info.dtype());

        if let Some(ranked) = input_info.as_ranked() {
            let dims = ranked.shape();
            return Ok(vec![(
                self.output,
                TensorInfo::from_dtype_and_shape_scalars(out_dtype, &dims),
            )]);
        }

        // Fallback: unknown shape
        let first = ScalarInfo::Symbolic(SymbolicScalar::new(out_dtype, symbolic_resolver));
        Ok(vec![(
            self.output,
            TensorInfo::new_from_first_element_and_rank(
                first,
                input_info.rank(),
                symbolic_resolver,
            ),
        )])
    }

    fn eval(
        &self,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
        _config: &super::MilliEvalConfig,
        backend: &mut EvalBackend,
    ) -> Result<Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>, MilliOpGraphError>
    {
        let input = &inputs[&self.input];
        let shape = input.shape();
        let total: usize = shape.iter().map(|&s| s as usize).product();

        let values = if let Some(seed) = self.seed {
            use rand::SeedableRng;
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed.to_bits() as u64);
            fill_normal(&mut rng, total, self.mean, self.scale)
        } else {
            let mut rng = rand::rng();
            fill_normal(&mut rng, total, self.mean, self.scale)
        };

        let shape_usize: Vec<usize> = shape.iter().map(|&s| s as usize).collect();
        let out = NumericTensor::<DynRank>::from_vec_shape(values, shape_usize).map_err(|e| {
            MilliOpGraphError::InvalidInput(format!("RandomNormalLike output creation failed: {e}"))
        })?;

        // Cast to target dtype if specified, otherwise use input's dtype
        let target_dtype = self.dtype.unwrap_or_else(|| input.dtype());
        let out = out.cast(target_dtype, backend)?;

        Ok(Box::new([(self.output, out)].into_iter()))
    }

    fn eval_new<'p, P2: crate::pool::Pool + 'p>(
        &self,
        inputs: &[crate::numeric_tensor::NumericTensorView<'_, crate::tensor_rank::DynRank>],
        pool: &'p P2,
    ) -> Result<Vec<crate::numeric_tensor::NumericTensor<'p, crate::tensor_rank::DynRank, P2>>, crate::nano_graph::pool_eval::PoolEvalError> {
        use crate::numeric_scalar::NumericScalar;
        use crate::numeric_tensor::{NumericTensor, TensorLayout};
        use crate::tensor_rank::DynRank;

        let input = &inputs[0];
        let shape = input.shape().clone();
        let numel: usize = shape.iter().product::<u64>() as usize;

        let out_dtype = self.dtype
            .and_then(|dt| crate::numeric_dtype::NumericDType::from_legacy(dt))
            .unwrap_or_else(|| input.dtype());

        let layout = TensorLayout::<DynRank>::row_major(shape, out_dtype);
        let buf = pool.allocate(layout.buffer_size_bytes())
            .map_err(crate::nano_graph::pool_eval::PoolEvalError::Allocation)?;
        let mut out = NumericTensor::from_parts(buf, layout);

        // Generate random normal values using fill_normal helper
        let values = if let Some(seed) = self.seed {
            use rand::SeedableRng;
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed.to_bits() as u64);
            fill_normal(&mut rng, numel, self.mean, self.scale)
        } else {
            let mut rng = rand::rng();
            fill_normal(&mut rng, numel, self.mean, self.scale)
        };

        for (i, &v) in values.iter().enumerate() {
            out.write_element(i, NumericScalar::from_f64(v as f64).cast_to(out_dtype));
        }

        Ok(vec![out])
    }
}
