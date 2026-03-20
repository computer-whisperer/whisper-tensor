//! Shared helpers for building VideoGenerationInterface supergraphs.

use rand::Rng;
use whisper_tensor::backends::ndarray_backend::NDArrayNumericTensor;
use whisper_tensor::dtype::DType;
use whisper_tensor::milli_graph::MilliOpGraph;
use whisper_tensor::milli_graph::ops::{Cast, Constant, SimpleBinary};
use whisper_tensor::super_graph::nodes::{SuperGraphNode, SuperGraphNodeMilliOpGraph};
use whisper_tensor::super_graph::{SuperGraphBuilder, SuperGraphLink};

/// Build a MilliOpGraph node that casts a tensor to a target dtype.
pub fn build_cast_node(
    builder: &mut SuperGraphBuilder,
    rng: &mut impl Rng,
    input: SuperGraphLink,
    dtype: DType,
) -> SuperGraphLink {
    let output = builder.new_tensor_link(rng);
    let (mut mg, input_map) = MilliOpGraph::new(std::iter::once(input.global_id()), rng);
    let inp = *input_map.get(&input.global_id()).unwrap();
    let casted =
        Cast::push_new_with_label(&mut mg, inp, dtype, Some(format!("cast_to_{dtype:?}")), rng);
    mg.set_output_map(std::iter::once((casted, output.global_id())));
    let mut node = SuperGraphNodeMilliOpGraph::new(mg, rng);
    node.label = Some(format!("cast_to_{dtype:?}"));
    builder.add_node(node.to_any());
    output
}

/// Build a MilliOpGraph node that creates a zero tensor matching the shape/dtype of the input.
pub fn build_zeros_like(
    builder: &mut SuperGraphBuilder,
    rng: &mut impl Rng,
    input: SuperGraphLink,
    output: SuperGraphLink,
) {
    let (mut mg, input_map) = MilliOpGraph::new(std::iter::once(input.global_id()), rng);
    let inp = *input_map.get(&input.global_id()).unwrap();
    let zero = Constant::push_new(
        &mut mg,
        NDArrayNumericTensor::from_vec_shape(vec![0.0f32], &vec![1]).unwrap(),
        rng,
    );
    let zero_cast = whisper_tensor::milli_graph::ops::CastLike::push_new(&mut mg, zero, inp, rng);
    let zeros = SimpleBinary::mul(&mut mg, inp, zero_cast, rng);
    mg.set_output_map(std::iter::once((zeros, output.global_id())));
    let mut node = SuperGraphNodeMilliOpGraph::new(mg, rng);
    node.label = Some("zeros_like".to_string());
    builder.add_node(node.to_any());
}

/// Build the progress tier initialization node (emits constant 0).
pub fn build_progress_init(
    builder: &mut SuperGraphBuilder,
    rng: &mut impl Rng,
    progress_tier_link: SuperGraphLink,
    label: &str,
) {
    let (mut mg, _) = MilliOpGraph::new(std::iter::empty(), rng);
    let tier = Constant::push_new_with_label(
        &mut mg,
        NDArrayNumericTensor::from_vec_shape(vec![0i64], &vec![1]).unwrap(),
        Some("progress_tier_zero".to_string()),
        rng,
    );
    mg.set_output_map(std::iter::once((tier, progress_tier_link.global_id())));
    let mut node = SuperGraphNodeMilliOpGraph::new(mg, rng);
    node.label = Some(label.to_string());
    builder.add_node(node.to_any());
}

/// Build the input prep node: cast latent to model_dtype, reshape timestep to [1].
pub fn build_input_prep(
    builder: &mut SuperGraphBuilder,
    rng: &mut impl Rng,
    inner_latent_in: SuperGraphLink,
    inner_timestep: SuperGraphLink,
    cast_latent: SuperGraphLink,
    cast_timestep: SuperGraphLink,
    model_dtype: DType,
    label: &str,
) {
    let (mut mg, input_map) = MilliOpGraph::new(
        [inner_latent_in.global_id(), inner_timestep.global_id()],
        rng,
    );
    let lat_in = *input_map.get(&inner_latent_in.global_id()).unwrap();
    let ts_in = *input_map.get(&inner_timestep.global_id()).unwrap();

    let lat_cast = Cast::push_new(&mut mg, lat_in, model_dtype, rng);
    let ts_shape = Constant::push_new(
        &mut mg,
        NDArrayNumericTensor::from_vec_shape(vec![1i64], &vec![1]).unwrap(),
        rng,
    );
    let ts_reshaped =
        whisper_tensor::milli_graph::ops::Reshape::push_new(&mut mg, ts_in, ts_shape, false, rng);

    mg.set_output_map(vec![
        (lat_cast, cast_latent.global_id()),
        (ts_reshaped, cast_timestep.global_id()),
    ]);
    let mut node = SuperGraphNodeMilliOpGraph::new(mg, rng);
    node.label = Some(label.to_string());
    builder.add_node(node.to_any());
}

/// Build the step counter increment node.
pub fn build_step_increment(
    builder: &mut SuperGraphBuilder,
    rng: &mut impl Rng,
    step_in: SuperGraphLink,
    step_out: SuperGraphLink,
) {
    let (mut mg, input_map) =
        MilliOpGraph::new(std::iter::once(step_in.global_id()), rng);
    let s_in = *input_map.get(&step_in.global_id()).unwrap();
    let one = Constant::push_new(
        &mut mg,
        NDArrayNumericTensor::from_vec_shape(vec![1i64], &vec![1]).unwrap(),
        rng,
    );
    let s_next = SimpleBinary::add(&mut mg, s_in, one, rng);
    mg.set_output_map(std::iter::once((s_next, step_out.global_id())));
    let mut node = SuperGraphNodeMilliOpGraph::new(mg, rng);
    node.label = Some("step_increment".to_string());
    builder.add_node(node.to_any());
}
