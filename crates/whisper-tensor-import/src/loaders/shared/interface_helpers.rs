//! Shared helpers for building image/video generation interface supergraphs.

use rand::Rng;
use whisper_tensor::dtype::DType;
use whisper_tensor::milli_graph::MilliOpGraph;
use whisper_tensor::milli_graph::ops::{
    ArgMax, Cast, Constant, SimpleBinary, SimpleUnaryOp, Unsqueeze,
};
use whisper_tensor::numeric_dtype::NumericDType;
use whisper_tensor::super_graph::links::{SuperGraphLinkDouble, SuperGraphLinkTriple};
use whisper_tensor::super_graph::nodes::{
    SuperGraphNode, SuperGraphNodeMilliOpGraph, SuperGraphNodeModelExecution,
    SuperGraphNodeReportProgress, SuperGraphNodeScan,
};
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
    let casted = Cast::push_new_with_label(
        &mut mg,
        inp,
        NumericDType::from_legacy(dtype).unwrap(),
        Some(format!("cast_to_{dtype:?}")),
        rng,
    );
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
    let zero = Constant::from_vec(&mut mg, vec![0.0f32], rng);
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
    let tier = Constant::from_vec_with_label(
        &mut mg,
        vec![0i64],
        Some("progress_tier_zero".to_string()),
        rng,
    );
    mg.set_output_map(std::iter::once((tier, progress_tier_link.global_id())));
    let mut node = SuperGraphNodeMilliOpGraph::new(mg, rng);
    node.label = Some(label.to_string());
    builder.add_node(node.to_any());
}

/// Build the input prep node: cast latent to model_dtype, reshape timestep to [1].
#[allow(clippy::too_many_arguments)]
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

    let lat_cast = Cast::push_new(
        &mut mg,
        lat_in,
        NumericDType::from_legacy(model_dtype).unwrap(),
        rng,
    );
    let ts_shape = Constant::from_vec(&mut mg, vec![1i64], rng);
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
    let (mut mg, input_map) = MilliOpGraph::new(std::iter::once(step_in.global_id()), rng);
    let s_in = *input_map.get(&step_in.global_id()).unwrap();
    let one = Constant::from_vec(&mut mg, vec![1i64], rng);
    let s_next = SimpleBinary::add(&mut mg, s_in, one, rng);
    mg.set_output_map(std::iter::once((s_next, step_out.global_id())));
    let mut node = SuperGraphNodeMilliOpGraph::new(mg, rng);
    node.label = Some("step_increment".to_string());
    builder.add_node(node.to_any());
}

/// Build a MilliOpGraph that computes ArgMax(input_ids, axis=1) to find EOS indices.
pub(crate) fn build_eos_indices_node(
    builder: &mut SuperGraphBuilder,
    rng: &mut impl Rng,
    input_ids: SuperGraphLink,
) -> SuperGraphLink {
    let eos_indices = builder.new_tensor_link(rng);
    let (mut mg, input_map) = MilliOpGraph::new(std::iter::once(input_ids.global_id()), rng);
    let ids_in = *input_map.get(&input_ids.global_id()).unwrap();
    // EOS (49407) is the max token in CLIP vocab, so argmax finds its position
    let argmax = ArgMax::push_new_with_label(
        &mut mg,
        ids_in,
        1,
        false,
        false,
        Some("eos_indices.argmax".to_string()),
        rng,
    );
    mg.set_output_map(std::iter::once((argmax, eos_indices.global_id())));
    let mut node = SuperGraphNodeMilliOpGraph::new(mg, rng);
    node.label = Some("eos_indices".to_string());
    builder.add_node(node.to_any());
    eos_indices
}

/// Build the Euler-discrete CFG denoising scan loop.
///
/// Returns the final latent link (F32, after Euler integration).
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_denoising_loop(
    builder: &mut SuperGraphBuilder,
    rng: &mut impl Rng,
    unet_weights: SuperGraphLink,
    cond_context: SuperGraphLink,
    uncond_context: SuperGraphLink,
    cond_y: Option<SuperGraphLink>,
    uncond_y: Option<SuperGraphLink>,
    guidance_scale_input: SuperGraphLink,
    initial_latent_input: SuperGraphLink,
    timesteps_input: SuperGraphLink,
    dt_input: SuperGraphLink,
    sigmas_input: SuperGraphLink,
    iteration_count_input: SuperGraphLink,
    model_dtype: DType,
    unet_model_index: usize,
) -> SuperGraphLink {
    let outer_final_latent = builder.new_tensor_link(rng);
    let progress_tier_link = builder.new_tensor_link(rng);
    builder.set_link_label(outer_final_latent, "latent_final");
    builder.set_link_label(progress_tier_link, "progress_tier");

    {
        let (mut mg, _) = MilliOpGraph::new(std::iter::empty(), rng);
        let tier = Constant::from_vec_with_label(
            &mut mg,
            vec![0i64],
            Some("progress_tier_zero".to_string()),
            rng,
        );
        mg.set_output_map(std::iter::once((tier, progress_tier_link.global_id())));
        let mut node = SuperGraphNodeMilliOpGraph::new(mg, rng);
        node.label = Some("denoise_progress_init".to_string());
        builder.add_node(node.to_any());
    }

    let mut inner_builder = SuperGraphBuilder::new();

    // Inner links
    let inner_unet_weights = inner_builder.new_model_link(rng);
    let inner_cond_context = inner_builder.new_tensor_link(rng);
    let inner_uncond_context = inner_builder.new_tensor_link(rng);
    let inner_guidance_scale = inner_builder.new_tensor_link(rng);
    let inner_latent_in = inner_builder.new_tensor_link(rng);
    let inner_latent_out = inner_builder.new_tensor_link(rng);
    let inner_timestep = inner_builder.new_tensor_link(rng);
    let inner_dt = inner_builder.new_tensor_link(rng);
    let inner_progress_tier = inner_builder.new_tensor_link(rng);
    let inner_total_steps = inner_builder.new_tensor_link(rng);
    let inner_step_in = inner_builder.new_tensor_link(rng);
    let inner_step_out = inner_builder.new_tensor_link(rng);
    let inner_sigma = inner_builder.new_tensor_link(rng);
    inner_builder.set_link_label(inner_unet_weights, "unet_weights");
    inner_builder.set_link_label(inner_cond_context, "context_conditional");
    inner_builder.set_link_label(inner_uncond_context, "context_unconditional");
    inner_builder.set_link_label(inner_guidance_scale, "guidance_scale");
    inner_builder.set_link_label(inner_latent_in, "latent_in");
    inner_builder.set_link_label(inner_latent_out, "latent_out");
    inner_builder.set_link_label(inner_timestep, "timestep");
    inner_builder.set_link_label(inner_dt, "dt");
    inner_builder.set_link_label(inner_progress_tier, "progress_tier");
    inner_builder.set_link_label(inner_total_steps, "total_steps");
    inner_builder.set_link_label(inner_step_in, "step_in");
    inner_builder.set_link_label(inner_step_out, "step_out");
    inner_builder.set_link_label(inner_sigma, "sigma");

    // Optional ADM conditioning links
    let inner_cond_y = cond_y.as_ref().map(|_| inner_builder.new_tensor_link(rng));
    let inner_uncond_y = uncond_y
        .as_ref()
        .map(|_| inner_builder.new_tensor_link(rng));
    if let Some(cy) = inner_cond_y {
        inner_builder.set_link_label(cy, "adm_conditional");
    }
    if let Some(uy) = inner_uncond_y {
        inner_builder.set_link_label(uy, "adm_unconditional");
    }

    // Inner node 1: Prep — scale latent by 1/sqrt(sigma^2+1), cast to model_dtype, reshape timestep
    let cast_latent = inner_builder.new_tensor_link(rng);
    let cast_timestep = inner_builder.new_tensor_link(rng);
    inner_builder.set_link_label(cast_latent, "latent_model_dtype");
    inner_builder.set_link_label(cast_timestep, "timestep_model_dtype");
    {
        let (mut mg, input_map) = MilliOpGraph::new(
            [
                inner_latent_in.global_id(),
                inner_timestep.global_id(),
                inner_sigma.global_id(),
            ],
            rng,
        );
        let lat_in = *input_map.get(&inner_latent_in.global_id()).unwrap();
        let ts_in = *input_map.get(&inner_timestep.global_id()).unwrap();
        let sigma_in = *input_map.get(&inner_sigma.global_id()).unwrap();

        // scale = 1 / sqrt(sigma^2 + 1)
        let sigma_sq = SimpleBinary::mul(&mut mg, sigma_in, sigma_in, rng);
        let one = Constant::from_vec_with_label(
            &mut mg,
            vec![1.0f32],
            Some("sigma.one".to_string()),
            rng,
        );
        let sigma_sq_plus_1 = SimpleBinary::add(&mut mg, sigma_sq, one, rng);
        let sqrt_val = SimpleUnaryOp::sqrt(&mut mg, sigma_sq_plus_1, rng);
        let inv_scale = SimpleBinary::div(&mut mg, one, sqrt_val, rng);
        let scaled_lat = SimpleBinary::mul(&mut mg, lat_in, inv_scale, rng);

        let lat_cast = Cast::push_new_with_label(
            &mut mg,
            scaled_lat,
            NumericDType::from_legacy(model_dtype).unwrap(),
            Some("latent.cast_model_dtype".to_string()),
            rng,
        );
        let ts_cast = Cast::push_new_with_label(
            &mut mg,
            ts_in,
            NumericDType::from_legacy(model_dtype).unwrap(),
            Some("timestep.cast_model_dtype".to_string()),
            rng,
        );
        let zero_axis = Constant::from_vec_with_label(
            &mut mg,
            vec![0i64],
            Some("timestep.unsqueeze_axis".to_string()),
            rng,
        );
        let ts_reshaped = Unsqueeze::push_new_with_label(
            &mut mg,
            ts_cast,
            zero_axis,
            Some("timestep.reshape_for_unet".to_string()),
            rng,
        );

        mg.set_output_map([
            (lat_cast, cast_latent.global_id()),
            (ts_reshaped, cast_timestep.global_id()),
        ]);
        let mut node = SuperGraphNodeMilliOpGraph::new(mg, rng);
        node.label = Some("unet_input_prep".to_string());
        inner_builder.add_node(node.to_any());
    }

    // Inner node 2: UNet unconditional
    let uncond_noise = inner_builder.new_tensor_link(rng);
    inner_builder.set_link_label(uncond_noise, "noise_unconditional");
    {
        let mut inputs = vec![
            (cast_latent, "sample".to_string()),
            (cast_timestep, "timestep".to_string()),
            (inner_uncond_context, "encoder_hidden_states".to_string()),
        ];
        if let Some(uy) = inner_uncond_y {
            inputs.push((uy, "y".to_string()));
        }
        let mut node = SuperGraphNodeModelExecution::new(
            rng,
            inner_unet_weights,
            unet_model_index,
            inputs,
            vec![("out_sample".to_string(), uncond_noise)],
        );
        node.label = Some("unet_unconditional".to_string());
        inner_builder.add_node(node.to_any());
    }

    // Inner node 3: UNet conditional
    let cond_noise = inner_builder.new_tensor_link(rng);
    inner_builder.set_link_label(cond_noise, "noise_conditional");
    {
        let mut inputs = vec![
            (cast_latent, "sample".to_string()),
            (cast_timestep, "timestep".to_string()),
            (inner_cond_context, "encoder_hidden_states".to_string()),
        ];
        if let Some(cy) = inner_cond_y {
            inputs.push((cy, "y".to_string()));
        }
        let mut node = SuperGraphNodeModelExecution::new(
            rng,
            inner_unet_weights,
            unet_model_index,
            inputs,
            vec![("out_sample".to_string(), cond_noise)],
        );
        node.label = Some("unet_conditional".to_string());
        inner_builder.add_node(node.to_any());
    }

    // Inner node 4: CFG + Euler step
    {
        let (mut mg, input_map) = MilliOpGraph::new(
            [
                uncond_noise.global_id(),
                cond_noise.global_id(),
                inner_latent_in.global_id(),
                inner_guidance_scale.global_id(),
                inner_dt.global_id(),
            ],
            rng,
        );
        let uncond_in = *input_map.get(&uncond_noise.global_id()).unwrap();
        let cond_in = *input_map.get(&cond_noise.global_id()).unwrap();
        let lat_in = *input_map.get(&inner_latent_in.global_id()).unwrap();
        let gs_in = *input_map.get(&inner_guidance_scale.global_id()).unwrap();
        let dt_in = *input_map.get(&inner_dt.global_id()).unwrap();

        // Cast noises to f32
        let uncond_f32 = Cast::push_new_with_label(
            &mut mg,
            uncond_in,
            NumericDType::F32,
            Some("noise_uncond.cast_f32".to_string()),
            rng,
        );
        let cond_f32 = Cast::push_new_with_label(
            &mut mg,
            cond_in,
            NumericDType::F32,
            Some("noise_cond.cast_f32".to_string()),
            rng,
        );

        // CFG: uncond + scale * (cond - uncond)
        let diff = SimpleBinary::sub(&mut mg, cond_f32, uncond_f32, rng);
        let scaled = SimpleBinary::mul(&mut mg, diff, gs_in, rng);
        let guided = SimpleBinary::add(&mut mg, uncond_f32, scaled, rng);

        // Euler step: latent + guided * dt
        let step = SimpleBinary::mul(&mut mg, guided, dt_in, rng);
        let latent_next = SimpleBinary::add(&mut mg, lat_in, step, rng);

        mg.set_output_map(std::iter::once((latent_next, inner_latent_out.global_id())));
        let mut node = SuperGraphNodeMilliOpGraph::new(mg, rng);
        node.label = Some("cfg_euler_step".to_string());
        inner_builder.add_node(node.to_any());
    }

    {
        let (mut mg, input_map) =
            MilliOpGraph::new(std::iter::once(inner_step_in.global_id()), rng);
        let step_in = *input_map.get(&inner_step_in.global_id()).unwrap();
        let one =
            Constant::from_vec_with_label(&mut mg, vec![1i64], Some("step.one".to_string()), rng);
        let step_next = SimpleBinary::add(&mut mg, step_in, one, rng);
        mg.set_output_map(std::iter::once((step_next, inner_step_out.global_id())));
        let mut node = SuperGraphNodeMilliOpGraph::new(mg, rng);
        node.label = Some("step_increment".to_string());
        inner_builder.add_node(node.to_any());
    }

    let mut report = SuperGraphNodeReportProgress::new(
        inner_progress_tier,
        inner_step_out,
        inner_total_steps,
        rng,
    );
    report.label = Some("denoise_progress".to_string());
    inner_builder.add_node(report.to_any());

    // Build inner graph
    let mut inner_inputs: Vec<_> = vec![
        inner_unet_weights.to_any(),
        inner_cond_context.to_any(),
        inner_uncond_context.to_any(),
        inner_guidance_scale.to_any(),
        inner_progress_tier.to_any(),
        inner_total_steps.to_any(),
        inner_latent_in.to_any(),
        inner_step_in.to_any(),
        inner_timestep.to_any(),
        inner_dt.to_any(),
        inner_sigma.to_any(),
    ];
    if let Some(cy) = inner_cond_y {
        inner_inputs.push(cy.to_any());
    }
    if let Some(uy) = inner_uncond_y {
        inner_inputs.push(uy.to_any());
    }
    let inner_outputs: Vec<_> = vec![inner_latent_out.to_any(), inner_step_out.to_any()];
    let inner_graph = inner_builder.build(rng, &inner_inputs, &inner_outputs);

    // Create scan node
    let mut simple_inputs = vec![
        SuperGraphLinkDouble::new(unet_weights, inner_unet_weights),
        SuperGraphLinkDouble::new(cond_context, inner_cond_context),
        SuperGraphLinkDouble::new(uncond_context, inner_uncond_context),
        SuperGraphLinkDouble::new(guidance_scale_input, inner_guidance_scale),
    ];
    if let (Some(cy_outer), Some(cy_inner)) = (cond_y, inner_cond_y) {
        simple_inputs.push(SuperGraphLinkDouble::new(cy_outer, cy_inner));
    }
    if let (Some(uy_outer), Some(uy_inner)) = (uncond_y, inner_uncond_y) {
        simple_inputs.push(SuperGraphLinkDouble::new(uy_outer, uy_inner));
    }
    simple_inputs.push(SuperGraphLinkDouble::new(
        progress_tier_link,
        inner_progress_tier,
    ));
    simple_inputs.push(SuperGraphLinkDouble::new(
        iteration_count_input,
        inner_total_steps,
    ));

    let mut scan_node = SuperGraphNodeScan::new(
        inner_graph,
        iteration_count_input,
        simple_inputs,
        // state_links: (initial, inner_in, inner_out)
        vec![
            SuperGraphLinkTriple::new(initial_latent_input, inner_latent_in, inner_latent_out),
            SuperGraphLinkTriple::new(progress_tier_link, inner_step_in, inner_step_out),
        ],
        // scan_inputs: (outer, inner, axis)
        vec![
            (timesteps_input, inner_timestep, 0),
            (dt_input, inner_dt, 0),
            (sigmas_input, inner_sigma, 0),
        ],
        // scan_outputs: none
        vec![],
        // simple_outputs: final latent
        vec![SuperGraphLinkDouble::new(
            inner_latent_out,
            outer_final_latent,
        )],
        rng,
    );
    scan_node.label = Some("denoising_scan".to_string());
    builder.add_node(scan_node.to_any());

    outer_final_latent
}

/// Build the VAE decode node (scale latent + decode).
pub(crate) fn build_vae_decode(
    builder: &mut SuperGraphBuilder,
    rng: &mut impl Rng,
    latent: SuperGraphLink,
    vae_weights: SuperGraphLink,
    vae_model_index: usize,
    vae_scale_factor: f32,
    model_dtype: DType,
) -> SuperGraphLink {
    // Scale latent by 1/vae_scale_factor and cast to model_dtype
    let scaled_latent = builder.new_tensor_link(rng);
    {
        let (mut mg, input_map) = MilliOpGraph::new(std::iter::once(latent.global_id()), rng);
        let lat_in = *input_map.get(&latent.global_id()).unwrap();

        let scale = Constant::from_vec_with_label(
            &mut mg,
            vec![1.0f32 / vae_scale_factor],
            Some("vae.inv_scale".to_string()),
            rng,
        );
        let scaled = SimpleBinary::mul(&mut mg, lat_in, scale, rng);
        let scaled_cast = Cast::push_new_with_label(
            &mut mg,
            scaled,
            NumericDType::from_legacy(model_dtype).unwrap(),
            Some("vae_latent.cast_model_dtype".to_string()),
            rng,
        );

        mg.set_output_map(std::iter::once((scaled_cast, scaled_latent.global_id())));
        let mut node = SuperGraphNodeMilliOpGraph::new(mg, rng);
        node.label = Some("vae_latent_prepare".to_string());
        builder.add_node(node.to_any());
    }

    // VAE decoder
    let image_output = builder.new_tensor_link(rng);
    let mut node = SuperGraphNodeModelExecution::new(
        rng,
        vae_weights,
        vae_model_index,
        vec![(scaled_latent, "latent_sample".to_string())],
        vec![("sample".to_string(), image_output)],
    );
    node.label = Some("vae_decode".to_string());
    builder.add_node(node.to_any());

    image_output
}

/// Build a VAE decode node with optional latent shift and custom IO names.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_vae_decode_with_shift(
    builder: &mut SuperGraphBuilder,
    rng: &mut impl Rng,
    latent: SuperGraphLink,
    vae_weights: SuperGraphLink,
    vae_model_index: usize,
    vae_scale_factor: f32,
    vae_shift_factor: f32,
    model_dtype: DType,
    vae_input_name: &str,
    vae_output_name: &str,
) -> SuperGraphLink {
    // Scale latent by 1/vae_scale_factor, optionally shift, then cast to model dtype.
    let scaled_latent = builder.new_tensor_link(rng);
    {
        let (mut mg, input_map) = MilliOpGraph::new(std::iter::once(latent.global_id()), rng);
        let lat_in = *input_map.get(&latent.global_id()).unwrap();

        let inv_scale = Constant::from_vec_with_label(
            &mut mg,
            vec![1.0f32 / vae_scale_factor],
            Some("vae.inv_scale".to_string()),
            rng,
        );
        let scaled = SimpleBinary::mul(&mut mg, lat_in, inv_scale, rng);
        let shifted = if vae_shift_factor != 0.0 {
            let shift = Constant::from_vec_with_label(
                &mut mg,
                vec![vae_shift_factor],
                Some("vae.shift".to_string()),
                rng,
            );
            SimpleBinary::add(&mut mg, scaled, shift, rng)
        } else {
            scaled
        };
        let casted = Cast::push_new_with_label(
            &mut mg,
            shifted,
            NumericDType::from_legacy(model_dtype).unwrap(),
            Some("vae_latent.cast_model_dtype".to_string()),
            rng,
        );

        mg.set_output_map(std::iter::once((casted, scaled_latent.global_id())));
        let mut node = SuperGraphNodeMilliOpGraph::new(mg, rng);
        node.label = Some("vae_latent_prepare".to_string());
        builder.add_node(node.to_any());
    }

    let image_output = builder.new_tensor_link(rng);
    let mut node = SuperGraphNodeModelExecution::new(
        rng,
        vae_weights,
        vae_model_index,
        vec![(scaled_latent, vae_input_name.to_string())],
        vec![(vae_output_name.to_string(), image_output)],
    );
    node.label = Some("vae_decode".to_string());
    builder.add_node(node.to_any());

    image_output
}

/// Build the Flux rectified flow denoising scan loop.
///
/// No CFG — single DiT forward pass per step. No input scaling.
/// DiT inputs: latent_sample, timestep, clip_pooled, t5_hidden_states
/// Returns the final latent link (F32, after Euler integration).
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_flux_denoising_loop(
    builder: &mut SuperGraphBuilder,
    rng: &mut impl Rng,
    dit_weights: SuperGraphLink,
    clip_pooled: SuperGraphLink,
    t5_hidden: SuperGraphLink,
    initial_latent_input: SuperGraphLink,
    timesteps_input: SuperGraphLink,
    dt_input: SuperGraphLink,
    _sigmas_input: SuperGraphLink,
    iteration_count_input: SuperGraphLink,
    guidance_input: Option<SuperGraphLink>,
    model_dtype: DType,
    dit_model_index: usize,
) -> SuperGraphLink {
    let outer_final_latent = builder.new_tensor_link(rng);
    let progress_tier_link = builder.new_tensor_link(rng);
    builder.set_link_label(outer_final_latent, "latent_final");
    builder.set_link_label(progress_tier_link, "progress_tier");

    {
        let (mut mg, _) = MilliOpGraph::new(std::iter::empty(), rng);
        let tier = Constant::from_vec_with_label(
            &mut mg,
            vec![0i64],
            Some("progress_tier_zero".to_string()),
            rng,
        );
        mg.set_output_map(std::iter::once((tier, progress_tier_link.global_id())));
        let mut node = SuperGraphNodeMilliOpGraph::new(mg, rng);
        node.label = Some("flux_progress_init".to_string());
        builder.add_node(node.to_any());
    }

    let mut inner_builder = SuperGraphBuilder::new();

    // Inner links
    let inner_dit_weights = inner_builder.new_model_link(rng);
    let inner_clip_pooled = inner_builder.new_tensor_link(rng);
    let inner_t5_hidden = inner_builder.new_tensor_link(rng);
    let inner_latent_in = inner_builder.new_tensor_link(rng);
    let inner_latent_out = inner_builder.new_tensor_link(rng);
    let inner_timestep = inner_builder.new_tensor_link(rng);
    let inner_dt = inner_builder.new_tensor_link(rng);
    let inner_guidance = guidance_input.map(|_| inner_builder.new_tensor_link(rng));
    let inner_progress_tier = inner_builder.new_tensor_link(rng);
    let inner_total_steps = inner_builder.new_tensor_link(rng);
    let inner_step_in = inner_builder.new_tensor_link(rng);
    let inner_step_out = inner_builder.new_tensor_link(rng);
    inner_builder.set_link_label(inner_dit_weights, "dit_weights");
    inner_builder.set_link_label(inner_clip_pooled, "clip_pooled");
    inner_builder.set_link_label(inner_t5_hidden, "t5_hidden_states");
    inner_builder.set_link_label(inner_latent_in, "latent_in");
    inner_builder.set_link_label(inner_latent_out, "latent_out");
    inner_builder.set_link_label(inner_timestep, "timestep");
    inner_builder.set_link_label(inner_dt, "dt");
    inner_builder.set_link_label(inner_progress_tier, "progress_tier");
    inner_builder.set_link_label(inner_total_steps, "total_steps");
    inner_builder.set_link_label(inner_step_in, "step_in");
    inner_builder.set_link_label(inner_step_out, "step_out");

    // Inner node 1: Prep — cast latent to model_dtype, reshape timestep (and guidance)
    let cast_latent = inner_builder.new_tensor_link(rng);
    let cast_timestep = inner_builder.new_tensor_link(rng);
    let cast_guidance = inner_guidance.map(|_| inner_builder.new_tensor_link(rng));
    inner_builder.set_link_label(cast_latent, "latent_model_dtype");
    inner_builder.set_link_label(cast_timestep, "timestep_reshaped");
    if let Some(ig) = inner_guidance {
        inner_builder.set_link_label(ig, "guidance");
    }
    if let Some(cg) = cast_guidance {
        inner_builder.set_link_label(cg, "guidance_reshaped");
    }
    {
        let mut input_ids = vec![inner_latent_in.global_id(), inner_timestep.global_id()];
        if let Some(ig) = inner_guidance {
            input_ids.push(ig.global_id());
        }
        let (mut mg, input_map) = MilliOpGraph::new(input_ids, rng);
        let lat_in = *input_map.get(&inner_latent_in.global_id()).unwrap();
        let ts_in = *input_map.get(&inner_timestep.global_id()).unwrap();

        // No sigma scaling for Flux (rectified flow operates directly on latents)
        let lat_cast = Cast::push_new_with_label(
            &mut mg,
            lat_in,
            NumericDType::from_legacy(model_dtype).unwrap(),
            Some("latent.cast_model_dtype".to_string()),
            rng,
        );

        // Reshape timestep from scalar to [1, 1]
        let ts_shape = Constant::from_vec_with_label(
            &mut mg,
            vec![1i64, 1],
            Some("timestep.shape_1x1".to_string()),
            rng,
        );
        let ts_reshaped = whisper_tensor::milli_graph::ops::Reshape::push_new_with_label(
            &mut mg,
            ts_in,
            ts_shape,
            false,
            Some("timestep.reshape_1x1".to_string()),
            rng,
        );

        let mut outputs = vec![
            (lat_cast, cast_latent.global_id()),
            (ts_reshaped, cast_timestep.global_id()),
        ];

        // Reshape guidance from scalar to [1, 1] (same as timestep)
        if let (Some(ig), Some(cg)) = (inner_guidance, cast_guidance) {
            let g_in = *input_map.get(&ig.global_id()).unwrap();
            let g_shape = Constant::from_vec_with_label(
                &mut mg,
                vec![1i64, 1],
                Some("guidance.shape_1x1".to_string()),
                rng,
            );
            let g_reshaped = whisper_tensor::milli_graph::ops::Reshape::push_new_with_label(
                &mut mg,
                g_in,
                g_shape,
                false,
                Some("guidance.reshape_1x1".to_string()),
                rng,
            );
            outputs.push((g_reshaped, cg.global_id()));
        }

        mg.set_output_map(outputs);
        let mut node = SuperGraphNodeMilliOpGraph::new(mg, rng);
        node.label = Some("dit_input_prep".to_string());
        inner_builder.add_node(node.to_any());
    }

    // Inner node 2: DiT forward pass
    let dit_output = inner_builder.new_tensor_link(rng);
    inner_builder.set_link_label(dit_output, "velocity_raw");
    let mut dit_inputs = vec![
        (cast_latent, "latent_sample".to_string()),
        (cast_timestep, "timestep".to_string()),
        (inner_clip_pooled, "clip_pooled".to_string()),
        (inner_t5_hidden, "t5_hidden_states".to_string()),
    ];
    if let Some(cg) = cast_guidance {
        dit_inputs.push((cg, "guidance".to_string()));
    }
    let mut node = SuperGraphNodeModelExecution::new(
        rng,
        inner_dit_weights,
        dit_model_index,
        dit_inputs,
        vec![("out_sample".to_string(), dit_output)],
    );
    node.label = Some("dit_forward".to_string());
    inner_builder.add_node(node.to_any());

    // Inner node 3: Euler step — latent_new = latent + velocity * dt
    {
        let (mut mg, input_map) = MilliOpGraph::new(
            [
                dit_output.global_id(),
                inner_latent_in.global_id(),
                inner_dt.global_id(),
            ],
            rng,
        );
        let velocity = *input_map.get(&dit_output.global_id()).unwrap();
        let lat_in = *input_map.get(&inner_latent_in.global_id()).unwrap();
        let dt_in = *input_map.get(&inner_dt.global_id()).unwrap();

        // Cast velocity to f32
        let velocity_f32 = Cast::push_new_with_label(
            &mut mg,
            velocity,
            NumericDType::F32,
            Some("velocity.cast_f32".to_string()),
            rng,
        );

        // Euler step: latent + velocity * dt
        let step = SimpleBinary::mul(&mut mg, velocity_f32, dt_in, rng);
        let latent_next = SimpleBinary::add(&mut mg, lat_in, step, rng);

        mg.set_output_map(std::iter::once((latent_next, inner_latent_out.global_id())));
        let mut node = SuperGraphNodeMilliOpGraph::new(mg, rng);
        node.label = Some("euler_step".to_string());
        inner_builder.add_node(node.to_any());
    }

    {
        let (mut mg, input_map) =
            MilliOpGraph::new(std::iter::once(inner_step_in.global_id()), rng);
        let step_in = *input_map.get(&inner_step_in.global_id()).unwrap();
        let one =
            Constant::from_vec_with_label(&mut mg, vec![1i64], Some("step.one".to_string()), rng);
        let step_next = SimpleBinary::add(&mut mg, step_in, one, rng);
        mg.set_output_map(std::iter::once((step_next, inner_step_out.global_id())));
        let mut node = SuperGraphNodeMilliOpGraph::new(mg, rng);
        node.label = Some("step_increment".to_string());
        inner_builder.add_node(node.to_any());
    }

    let mut report = SuperGraphNodeReportProgress::new(
        inner_progress_tier,
        inner_step_out,
        inner_total_steps,
        rng,
    );
    report.label = Some("flux_progress".to_string());
    inner_builder.add_node(report.to_any());

    // Build inner graph
    let mut inner_inputs: Vec<_> = vec![
        inner_dit_weights.to_any(),
        inner_clip_pooled.to_any(),
        inner_t5_hidden.to_any(),
        inner_progress_tier.to_any(),
        inner_total_steps.to_any(),
        inner_latent_in.to_any(),
        inner_step_in.to_any(),
        inner_timestep.to_any(),
        inner_dt.to_any(),
    ];
    if let Some(ig) = inner_guidance {
        inner_inputs.push(ig.to_any());
    }
    let inner_outputs: Vec<_> = vec![inner_latent_out.to_any(), inner_step_out.to_any()];
    let inner_graph = inner_builder.build(rng, &inner_inputs, &inner_outputs);

    // Create scan node
    let mut simple_inputs = vec![
        SuperGraphLinkDouble::new(dit_weights, inner_dit_weights),
        SuperGraphLinkDouble::new(clip_pooled, inner_clip_pooled),
        SuperGraphLinkDouble::new(t5_hidden, inner_t5_hidden),
    ];
    if let (Some(outer_g), Some(ig)) = (guidance_input, inner_guidance) {
        simple_inputs.push(SuperGraphLinkDouble::new(outer_g, ig));
    }
    simple_inputs.push(SuperGraphLinkDouble::new(
        progress_tier_link,
        inner_progress_tier,
    ));
    simple_inputs.push(SuperGraphLinkDouble::new(
        iteration_count_input,
        inner_total_steps,
    ));

    let mut scan_node = SuperGraphNodeScan::new(
        inner_graph,
        iteration_count_input,
        simple_inputs,
        // state_links: latent carried across iterations
        vec![
            SuperGraphLinkTriple::new(initial_latent_input, inner_latent_in, inner_latent_out),
            SuperGraphLinkTriple::new(progress_tier_link, inner_step_in, inner_step_out),
        ],
        // scan_inputs: timestep and dt scanned along axis 0
        vec![
            (timesteps_input, inner_timestep, 0),
            (dt_input, inner_dt, 0),
        ],
        // scan_outputs: none
        vec![],
        // simple_outputs: final latent
        vec![SuperGraphLinkDouble::new(
            inner_latent_out,
            outer_final_latent,
        )],
        rng,
    );
    scan_node.label = Some("flux_denoise_scan".to_string());
    builder.add_node(scan_node.to_any());

    outer_final_latent
}

/// Build the Flux VAE decode node.
///
/// Flux VAE scaling: latent_for_vae = latent / 0.3611 + 0.1159
pub(crate) fn build_flux_vae_decode(
    builder: &mut SuperGraphBuilder,
    rng: &mut impl Rng,
    latent: SuperGraphLink,
    vae_weights: SuperGraphLink,
    vae_model_index: usize,
) -> SuperGraphLink {
    // Scale latent: x / 0.3611 + 0.1159, cast to F32 for VAE
    let scaled_latent = builder.new_tensor_link(rng);
    {
        let (mut mg, input_map) = MilliOpGraph::new(std::iter::once(latent.global_id()), rng);
        let lat_in = *input_map.get(&latent.global_id()).unwrap();

        // Cast to F32 first (denoising loop outputs in model_dtype which may be BF16)
        let f32_lat = Cast::push_new_with_label(
            &mut mg,
            lat_in,
            NumericDType::F32,
            Some("latent.cast_f32".to_string()),
            rng,
        );

        let inv_scale = Constant::from_vec_with_label(
            &mut mg,
            vec![1.0f32 / 0.3611],
            Some("flux_vae.inv_scale".to_string()),
            rng,
        );
        let shift = Constant::from_vec_with_label(
            &mut mg,
            vec![0.1159f32],
            Some("flux_vae.shift".to_string()),
            rng,
        );
        let scaled = SimpleBinary::mul(&mut mg, f32_lat, inv_scale, rng);
        let shifted = SimpleBinary::add(&mut mg, scaled, shift, rng);

        mg.set_output_map(std::iter::once((shifted, scaled_latent.global_id())));
        let mut node = SuperGraphNodeMilliOpGraph::new(mg, rng);
        node.label = Some("flux_vae_latent_prepare".to_string());
        builder.add_node(node.to_any());
    }

    // VAE decoder
    let image_output = builder.new_tensor_link(rng);
    let mut node = SuperGraphNodeModelExecution::new(
        rng,
        vae_weights,
        vae_model_index,
        vec![(scaled_latent, "latent_sample".to_string())],
        vec![("sample".to_string(), image_output)],
    );
    node.label = Some("flux_vae_decode".to_string());
    builder.add_node(node.to_any());

    image_output
}

/// Build an SD3-style rectified flow denoising loop with CFG.
///
/// No sigma scaling. One unconditional and one conditional transformer pass per step.
/// Transformer inputs are provided by name to support different ONNX export variants.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_sd3_denoising_loop(
    builder: &mut SuperGraphBuilder,
    rng: &mut impl Rng,
    transformer_weights: SuperGraphLink,
    cond_context: SuperGraphLink,
    uncond_context: SuperGraphLink,
    cond_pooled: SuperGraphLink,
    uncond_pooled: SuperGraphLink,
    guidance_scale_input: SuperGraphLink,
    initial_latent_input: SuperGraphLink,
    timesteps_input: SuperGraphLink,
    dt_input: SuperGraphLink,
    _sigmas_input: SuperGraphLink,
    iteration_count_input: SuperGraphLink,
    model_dtype: DType,
    transformer_model_index: usize,
    transformer_latent_input_name: &str,
    transformer_timestep_input_name: &str,
    transformer_context_input_name: &str,
    transformer_pooled_input_name: &str,
    transformer_output_name: &str,
) -> SuperGraphLink {
    let outer_final_latent = builder.new_tensor_link(rng);
    let progress_tier_link = builder.new_tensor_link(rng);
    builder.set_link_label(outer_final_latent, "latent_final");
    builder.set_link_label(progress_tier_link, "progress_tier");

    {
        let (mut mg, _) = MilliOpGraph::new(std::iter::empty(), rng);
        let tier = Constant::from_vec_with_label(
            &mut mg,
            vec![0i64],
            Some("progress_tier_zero".to_string()),
            rng,
        );
        mg.set_output_map(std::iter::once((tier, progress_tier_link.global_id())));
        let mut node = SuperGraphNodeMilliOpGraph::new(mg, rng);
        node.label = Some("sd3_progress_init".to_string());
        builder.add_node(node.to_any());
    }

    let mut inner_builder = SuperGraphBuilder::new();

    // Inner links
    let inner_transformer_weights = inner_builder.new_model_link(rng);
    let inner_cond_context = inner_builder.new_tensor_link(rng);
    let inner_uncond_context = inner_builder.new_tensor_link(rng);
    let inner_cond_pooled = inner_builder.new_tensor_link(rng);
    let inner_uncond_pooled = inner_builder.new_tensor_link(rng);
    let inner_guidance_scale = inner_builder.new_tensor_link(rng);
    let inner_latent_in = inner_builder.new_tensor_link(rng);
    let inner_latent_out = inner_builder.new_tensor_link(rng);
    let inner_timestep = inner_builder.new_tensor_link(rng);
    let inner_dt = inner_builder.new_tensor_link(rng);
    let inner_progress_tier = inner_builder.new_tensor_link(rng);
    let inner_total_steps = inner_builder.new_tensor_link(rng);
    let inner_step_in = inner_builder.new_tensor_link(rng);
    let inner_step_out = inner_builder.new_tensor_link(rng);
    inner_builder.set_link_label(inner_transformer_weights, "transformer_weights");
    inner_builder.set_link_label(inner_cond_context, "context_conditional");
    inner_builder.set_link_label(inner_uncond_context, "context_unconditional");
    inner_builder.set_link_label(inner_cond_pooled, "pooled_conditional");
    inner_builder.set_link_label(inner_uncond_pooled, "pooled_unconditional");
    inner_builder.set_link_label(inner_guidance_scale, "guidance_scale");
    inner_builder.set_link_label(inner_latent_in, "latent_in");
    inner_builder.set_link_label(inner_latent_out, "latent_out");
    inner_builder.set_link_label(inner_timestep, "timestep");
    inner_builder.set_link_label(inner_dt, "dt");
    inner_builder.set_link_label(inner_progress_tier, "progress_tier");
    inner_builder.set_link_label(inner_total_steps, "total_steps");
    inner_builder.set_link_label(inner_step_in, "step_in");
    inner_builder.set_link_label(inner_step_out, "step_out");

    // Inner node 1: cast latent/timestep to model_dtype and reshape timestep to [1]
    let cast_latent = inner_builder.new_tensor_link(rng);
    let cast_timestep = inner_builder.new_tensor_link(rng);
    inner_builder.set_link_label(cast_latent, "latent_model_dtype");
    inner_builder.set_link_label(cast_timestep, "timestep_model_dtype");
    {
        let (mut mg, input_map) = MilliOpGraph::new(
            [inner_latent_in.global_id(), inner_timestep.global_id()],
            rng,
        );
        let lat_in = *input_map.get(&inner_latent_in.global_id()).unwrap();
        let ts_in = *input_map.get(&inner_timestep.global_id()).unwrap();

        let lat_cast = Cast::push_new_with_label(
            &mut mg,
            lat_in,
            NumericDType::from_legacy(model_dtype).unwrap(),
            Some("latent.cast_model_dtype".to_string()),
            rng,
        );
        let ts_cast = Cast::push_new_with_label(
            &mut mg,
            ts_in,
            NumericDType::from_legacy(model_dtype).unwrap(),
            Some("timestep.cast_model_dtype".to_string()),
            rng,
        );
        let zero_axis = Constant::from_vec_with_label(
            &mut mg,
            vec![0i64],
            Some("timestep.unsqueeze_axis".to_string()),
            rng,
        );
        let ts_reshaped = Unsqueeze::push_new_with_label(
            &mut mg,
            ts_cast,
            zero_axis,
            Some("timestep.reshape_for_transformer".to_string()),
            rng,
        );

        mg.set_output_map([
            (lat_cast, cast_latent.global_id()),
            (ts_reshaped, cast_timestep.global_id()),
        ]);
        let mut node = SuperGraphNodeMilliOpGraph::new(mg, rng);
        node.label = Some("transformer_input_prep".to_string());
        inner_builder.add_node(node.to_any());
    }

    // Inner node 2: Transformer unconditional
    let uncond_pred = inner_builder.new_tensor_link(rng);
    inner_builder.set_link_label(uncond_pred, "prediction_unconditional");
    let mut uncond_node = SuperGraphNodeModelExecution::new(
        rng,
        inner_transformer_weights,
        transformer_model_index,
        vec![
            (cast_latent, transformer_latent_input_name.to_string()),
            (cast_timestep, transformer_timestep_input_name.to_string()),
            (
                inner_uncond_context,
                transformer_context_input_name.to_string(),
            ),
            (
                inner_uncond_pooled,
                transformer_pooled_input_name.to_string(),
            ),
        ],
        vec![(transformer_output_name.to_string(), uncond_pred)],
    );
    uncond_node.label = Some("transformer_unconditional".to_string());
    inner_builder.add_node(uncond_node.to_any());

    // Inner node 3: Transformer conditional
    let cond_pred = inner_builder.new_tensor_link(rng);
    inner_builder.set_link_label(cond_pred, "prediction_conditional");
    let mut cond_node = SuperGraphNodeModelExecution::new(
        rng,
        inner_transformer_weights,
        transformer_model_index,
        vec![
            (cast_latent, transformer_latent_input_name.to_string()),
            (cast_timestep, transformer_timestep_input_name.to_string()),
            (
                inner_cond_context,
                transformer_context_input_name.to_string(),
            ),
            (inner_cond_pooled, transformer_pooled_input_name.to_string()),
        ],
        vec![(transformer_output_name.to_string(), cond_pred)],
    );
    cond_node.label = Some("transformer_conditional".to_string());
    inner_builder.add_node(cond_node.to_any());

    // Inner node 4: CFG + Euler step
    {
        let (mut mg, input_map) = MilliOpGraph::new(
            [
                uncond_pred.global_id(),
                cond_pred.global_id(),
                inner_latent_in.global_id(),
                inner_guidance_scale.global_id(),
                inner_dt.global_id(),
            ],
            rng,
        );
        let uncond_in = *input_map.get(&uncond_pred.global_id()).unwrap();
        let cond_in = *input_map.get(&cond_pred.global_id()).unwrap();
        let lat_in = *input_map.get(&inner_latent_in.global_id()).unwrap();
        let gs_in = *input_map.get(&inner_guidance_scale.global_id()).unwrap();
        let dt_in = *input_map.get(&inner_dt.global_id()).unwrap();

        let uncond_f32 = Cast::push_new_with_label(
            &mut mg,
            uncond_in,
            NumericDType::F32,
            Some("pred_uncond.cast_f32".to_string()),
            rng,
        );
        let cond_f32 = Cast::push_new_with_label(
            &mut mg,
            cond_in,
            NumericDType::F32,
            Some("pred_cond.cast_f32".to_string()),
            rng,
        );
        let diff = SimpleBinary::sub(&mut mg, cond_f32, uncond_f32, rng);
        let scaled = SimpleBinary::mul(&mut mg, diff, gs_in, rng);
        let guided = SimpleBinary::add(&mut mg, uncond_f32, scaled, rng);
        let step = SimpleBinary::mul(&mut mg, guided, dt_in, rng);
        let latent_next = SimpleBinary::add(&mut mg, lat_in, step, rng);

        mg.set_output_map(std::iter::once((latent_next, inner_latent_out.global_id())));
        let mut node = SuperGraphNodeMilliOpGraph::new(mg, rng);
        node.label = Some("cfg_euler_step".to_string());
        inner_builder.add_node(node.to_any());
    }

    {
        let (mut mg, input_map) =
            MilliOpGraph::new(std::iter::once(inner_step_in.global_id()), rng);
        let step_in = *input_map.get(&inner_step_in.global_id()).unwrap();
        let one =
            Constant::from_vec_with_label(&mut mg, vec![1i64], Some("step.one".to_string()), rng);
        let step_next = SimpleBinary::add(&mut mg, step_in, one, rng);
        mg.set_output_map(std::iter::once((step_next, inner_step_out.global_id())));
        let mut node = SuperGraphNodeMilliOpGraph::new(mg, rng);
        node.label = Some("step_increment".to_string());
        inner_builder.add_node(node.to_any());
    }

    let mut report = SuperGraphNodeReportProgress::new(
        inner_progress_tier,
        inner_step_out,
        inner_total_steps,
        rng,
    );
    report.label = Some("sd3_progress".to_string());
    inner_builder.add_node(report.to_any());

    let inner_inputs: Vec<_> = vec![
        inner_transformer_weights.to_any(),
        inner_cond_context.to_any(),
        inner_uncond_context.to_any(),
        inner_cond_pooled.to_any(),
        inner_uncond_pooled.to_any(),
        inner_guidance_scale.to_any(),
        inner_progress_tier.to_any(),
        inner_total_steps.to_any(),
        inner_latent_in.to_any(),
        inner_step_in.to_any(),
        inner_timestep.to_any(),
        inner_dt.to_any(),
    ];
    let inner_outputs: Vec<_> = vec![inner_latent_out.to_any(), inner_step_out.to_any()];
    let inner_graph = inner_builder.build(rng, &inner_inputs, &inner_outputs);

    let mut scan_node = SuperGraphNodeScan::new(
        inner_graph,
        iteration_count_input,
        vec![
            SuperGraphLinkDouble::new(transformer_weights, inner_transformer_weights),
            SuperGraphLinkDouble::new(cond_context, inner_cond_context),
            SuperGraphLinkDouble::new(uncond_context, inner_uncond_context),
            SuperGraphLinkDouble::new(cond_pooled, inner_cond_pooled),
            SuperGraphLinkDouble::new(uncond_pooled, inner_uncond_pooled),
            SuperGraphLinkDouble::new(guidance_scale_input, inner_guidance_scale),
            SuperGraphLinkDouble::new(progress_tier_link, inner_progress_tier),
            SuperGraphLinkDouble::new(iteration_count_input, inner_total_steps),
        ],
        vec![
            SuperGraphLinkTriple::new(initial_latent_input, inner_latent_in, inner_latent_out),
            SuperGraphLinkTriple::new(progress_tier_link, inner_step_in, inner_step_out),
        ],
        vec![
            (timesteps_input, inner_timestep, 0),
            (dt_input, inner_dt, 0),
        ],
        vec![],
        vec![SuperGraphLinkDouble::new(
            inner_latent_out,
            outer_final_latent,
        )],
        rng,
    );
    scan_node.label = Some("sd3_denoise_scan".to_string());
    builder.add_node(scan_node.to_any());

    outer_final_latent
}

/// Build a single-text-encoder CFG interface for SD 1.5 / SD 2.
///
/// Model weights order: [text_encoder, unet, vae_decoder]
pub(crate) fn build_single_te_cfg_interface(
    rng: &mut impl Rng,
    tokenizer: whisper_tensor::metadata::TokenizerInfo,
    model_dtype: DType,
    vae_scale_factor: f32,
) -> whisper_tensor::interfaces::ImageGenerationInterface {
    use whisper_tensor::interfaces::{ImageGenerationInterface, SchedulerType};
    use whisper_tensor::super_graph::nodes::{
        SuperGraphNodeTensorToImage, SuperGraphNodeTokenizerEncode,
        SuperGraphNodeTokenizerEncodeMode, SuperGraphNodeTokenizerLoad,
    };

    let mut builder = SuperGraphBuilder::new();

    // Create input links
    let positive_prompt_input = builder.new_string_link(rng);
    let negative_prompt_input = builder.new_string_link(rng);
    let initial_latent_input = builder.new_tensor_link(rng);
    let timesteps_input = builder.new_tensor_link(rng);
    let dt_input = builder.new_tensor_link(rng);
    let sigmas_input = builder.new_tensor_link(rng);
    let iteration_count_input = builder.new_tensor_link(rng);
    let guidance_scale_input = builder.new_tensor_link(rng);
    let te_weights = builder.new_model_link(rng);
    let unet_weights = builder.new_model_link(rng);
    let vae_weights = builder.new_model_link(rng);
    builder.set_link_label(positive_prompt_input, "prompt_positive");
    builder.set_link_label(negative_prompt_input, "prompt_negative");
    builder.set_link_label(initial_latent_input, "latent_initial");
    builder.set_link_label(timesteps_input, "timesteps");
    builder.set_link_label(dt_input, "dt");
    builder.set_link_label(sigmas_input, "sigmas");
    builder.set_link_label(iteration_count_input, "iteration_count");
    builder.set_link_label(guidance_scale_input, "guidance_scale");
    builder.set_link_label(te_weights, "text_encoder_weights");
    builder.set_link_label(unet_weights, "unet_weights");
    builder.set_link_label(vae_weights, "vae_decoder_weights");

    // Prompt tokenization inside the supergraph.
    let tokenizer_link = SuperGraphNodeTokenizerLoad::new_and_add(&mut builder, tokenizer, rng);
    let cond_ids_input = SuperGraphNodeTokenizerEncode::new_with_mode_and_add(
        &mut builder,
        tokenizer_link,
        positive_prompt_input,
        SuperGraphNodeTokenizerEncodeMode::ClipStyle {
            seq_len: 77,
            bos: 49406,
            eos: 49407,
            pad: 0,
        },
        rng,
    );
    let negative_cond_ids_input = SuperGraphNodeTokenizerEncode::new_with_mode_and_add(
        &mut builder,
        tokenizer_link,
        negative_prompt_input,
        SuperGraphNodeTokenizerEncodeMode::ClipStyle {
            seq_len: 77,
            bos: 49406,
            eos: 49407,
            pad: 0,
        },
        rng,
    );

    // Text encoder: conditional -> F32, cast to model_dtype
    let cond_hidden_f32 = builder.new_tensor_link(rng);
    let mut cond_te = SuperGraphNodeModelExecution::new(
        rng,
        te_weights,
        0,
        vec![(cond_ids_input, "input_ids".to_string())],
        vec![("last_hidden_state".to_string(), cond_hidden_f32)],
    );
    cond_te.label = Some("text_encoder_conditional".to_string());
    builder.add_node(cond_te.to_any());
    let cond_context = build_cast_node(&mut builder, rng, cond_hidden_f32, model_dtype);

    // Text encoder: unconditional -> F32, cast to model_dtype
    let uncond_hidden_f32 = builder.new_tensor_link(rng);
    let mut uncond_te = SuperGraphNodeModelExecution::new(
        rng,
        te_weights,
        0,
        vec![(negative_cond_ids_input, "input_ids".to_string())],
        vec![("last_hidden_state".to_string(), uncond_hidden_f32)],
    );
    uncond_te.label = Some("text_encoder_unconditional".to_string());
    builder.add_node(uncond_te.to_any());
    let uncond_context = build_cast_node(&mut builder, rng, uncond_hidden_f32, model_dtype);

    // Denoising loop
    let final_latent = build_denoising_loop(
        &mut builder,
        rng,
        unet_weights,
        cond_context,
        uncond_context,
        None, // no ADM
        None,
        guidance_scale_input,
        initial_latent_input,
        timesteps_input,
        dt_input,
        sigmas_input,
        iteration_count_input,
        model_dtype,
        1, // unet model index
    );

    // VAE decode + wrap tensor into Image
    let decoded_image_tensor = build_vae_decode(
        &mut builder,
        rng,
        final_latent,
        vae_weights,
        2,
        vae_scale_factor,
        model_dtype,
    );
    let image_output =
        SuperGraphNodeTensorToImage::new_and_add(&mut builder, decoded_image_tensor, rng);
    builder.set_link_label(image_output, "image_output");

    // Build outer graph
    let model_weights = vec![te_weights, unet_weights, vae_weights];
    let input_links: Vec<_> = vec![
        positive_prompt_input.to_any(),
        negative_prompt_input.to_any(),
        initial_latent_input.to_any(),
        timesteps_input.to_any(),
        dt_input.to_any(),
        sigmas_input.to_any(),
        iteration_count_input.to_any(),
        guidance_scale_input.to_any(),
        te_weights.to_any(),
        unet_weights.to_any(),
        vae_weights.to_any(),
    ];
    let output_links: Vec<_> = vec![image_output.to_any()];
    let super_graph = builder.build(rng, &input_links, &output_links);

    ImageGenerationInterface {
        super_graph,
        positive_prompt_input,
        negative_prompt_input: Some(negative_prompt_input),
        initial_latent_input,
        timesteps_input,
        dt_input,
        sigmas_input,
        iteration_count_input,
        guidance_scale_input: Some(guidance_scale_input),
        model_weights,
        image_output,
        scheduler: SchedulerType::EulerDiscrete,
        latent_channels: 4,
    }
}
