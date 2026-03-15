use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::dtype::DType;
use crate::graph::GlobalId;
use crate::milli_graph::MilliOpGraphError;
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::numeric_tensor::NumericTensor;
use rand::{Rng, RngExt};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Generates a tensor of random normal values with the same shape as the input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomNormalLike {
    global_id: GlobalId,
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
        let output = graph.get_new_tensor_id(rng);
        let node = Self {
            global_id: GlobalId::new(rng),
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
    fn eval(
        &self,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
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
}
