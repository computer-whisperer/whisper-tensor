use super::onnx_bytes_to_model;
use std::path::Path;
use whisper_tensor::backends::ndarray_backend::NDArrayNumericTensor;
use whisper_tensor::dtype::DType;
use whisper_tensor::interfaces::{TTSInputConfig, TextToSpeechInterface};
use whisper_tensor::loader::{LoadedInterface, LoaderError, LoaderOutput};
use whisper_tensor::milli_graph::MilliOpGraph;
use whisper_tensor::milli_graph::ops::{Cast, Constant, SimpleBinary};
use whisper_tensor::super_graph::SuperGraphBuilder;
use whisper_tensor::super_graph::links::{
    SuperGraphLink, SuperGraphLinkDouble, SuperGraphLinkTriple,
};
use whisper_tensor::super_graph::nodes::{
    SuperGraphNode, SuperGraphNodeF5TextToTensor, SuperGraphNodeMilliOpGraph,
    SuperGraphNodeModelExecution, SuperGraphNodeScan, SuperGraphNodeTensorToAudioClip,
};

const NFE_STEPS: u32 = 32;
const SAMPLE_RATE: u32 = 24000;

/// Load F5-TTS from a directory containing the 3 ONNX components and vocab.
pub fn load_f5_tts(dir: &Path) -> Result<LoaderOutput, LoaderError> {
    let preprocess_path = dir.join("F5_Preprocess.onnx");
    let transformer_path = dir.join("F5_Transformer.onnx");
    let decode_path = dir.join("F5_Decode.onnx");

    for (name, path) in [
        ("F5_Preprocess.onnx", &preprocess_path),
        ("F5_Transformer.onnx", &transformer_path),
        ("F5_Decode.onnx", &decode_path),
    ] {
        if !path.exists() {
            return Err(LoaderError::LoadFailed(anyhow::anyhow!(
                "{} not found in {}",
                name,
                dir.display()
            )));
        }
    }

    let base_dir = Some(dir);

    let preprocess_data =
        crate::load_onnx_file(&preprocess_path).map_err(|e| LoaderError::LoadFailed(e.into()))?;
    let (_, preprocess_out) = onnx_bytes_to_model(&preprocess_data, "f5-preprocess", base_dir)?;

    let transformer_data =
        crate::load_onnx_file(&transformer_path).map_err(|e| LoaderError::LoadFailed(e.into()))?;
    let (_, transformer_out) = onnx_bytes_to_model(&transformer_data, "f5-transformer", base_dir)?;

    let decode_data =
        crate::load_onnx_file(&decode_path).map_err(|e| LoaderError::LoadFailed(e.into()))?;
    let (_, decode_out) = onnx_bytes_to_model(&decode_data, "f5-decode", base_dir)?;

    let vocab_path = dir.join("vocab.txt");
    let vocab = if vocab_path.exists() {
        std::fs::read_to_string(&vocab_path).map_err(|e| LoaderError::LoadFailed(e.into()))?
    } else {
        return Err(LoaderError::LoadFailed(anyhow::anyhow!(
            "vocab.txt not found in {}",
            dir.display()
        )));
    };

    let mut rng = rand::rng();
    let interface = build_f5_supergraph(&vocab, &mut rng);

    let mut models = Vec::new();
    models.extend(preprocess_out.models);
    models.extend(transformer_out.models);
    models.extend(decode_out.models);

    Ok(LoaderOutput {
        models,
        interfaces: vec![LoadedInterface {
            name: "f5-TextToSpeech".to_string(),
            interface: interface.to_any(),
        }],
    })
}

/// Build the F5-TTS SuperGraph.
///
/// Pipeline:
/// 1. Preprocess (model 0): audio + text_ids + max_duration -> noise, rope, conditioning, ref_signal_len
/// 2. Scan loop (model 1): 31 iterations of Transformer + Euler step
/// 3. Decode (model 2): final denoised + ref_signal_len -> audio waveform
fn build_f5_supergraph(vocab: &str, rng: &mut impl rand::Rng) -> TextToSpeechInterface {
    let mut builder = SuperGraphBuilder::new();

    // External input links
    let text_input_link = builder.new_string_link(rng);
    let ref_audio_link = builder.new_tensor_link(rng);
    let max_duration_link = builder.new_tensor_link(rng);
    let preprocess_weights = builder.new_model_link(rng);
    let transformer_weights = builder.new_model_link(rng);
    let decode_weights = builder.new_model_link(rng);
    let time_steps_link = builder.new_tensor_link(rng);
    let iteration_count_link = builder.new_tensor_link(rng);
    let audio_tensor_output_link = builder.new_tensor_link(rng);

    let text_ids_link = SuperGraphNodeF5TextToTensor::new_and_add(
        &mut builder,
        text_input_link,
        vocab.to_string(),
        rng,
    );

    // --- Node 1: Preprocess ---
    let noise_link = builder.new_tensor_link(rng);
    let rope_cos_link = builder.new_tensor_link(rng);
    let rope_sin_link = builder.new_tensor_link(rng);
    let cat_mel_text_link = builder.new_tensor_link(rng);
    let cat_mel_text_drop_link = builder.new_tensor_link(rng);
    let qk_rotated_empty_link = builder.new_tensor_link(rng);
    let ref_signal_len_link = builder.new_tensor_link(rng);

    builder.add_node(
        SuperGraphNodeModelExecution::new(
            rng,
            preprocess_weights,
            0, // model index 0
            vec![
                (ref_audio_link, "audio".to_string()),
                (text_ids_link, "text_ids".to_string()),
                (max_duration_link, "max_duration".to_string()),
            ],
            vec![
                ("noise".to_string(), noise_link),
                ("rope_cos".to_string(), rope_cos_link),
                ("rope_sin".to_string(), rope_sin_link),
                ("cat_mel_text".to_string(), cat_mel_text_link),
                ("cat_mel_text_drop".to_string(), cat_mel_text_drop_link),
                ("qk_rotated_empty".to_string(), qk_rotated_empty_link),
                ("ref_signal_len".to_string(), ref_signal_len_link),
            ],
        )
        .to_any(),
    );

    // --- Node 2: Scan loop (ODE denoising) ---
    let final_denoised_link = build_f5_denoising_loop(
        &mut builder,
        rng,
        transformer_weights,
        noise_link,
        rope_cos_link,
        rope_sin_link,
        cat_mel_text_link,
        cat_mel_text_drop_link,
        qk_rotated_empty_link,
        time_steps_link,
        iteration_count_link,
    );

    // --- Node 3: Decode ---
    builder.add_node(
        SuperGraphNodeModelExecution::new(
            rng,
            decode_weights,
            2, // model index 2
            vec![
                (final_denoised_link, "denoised".to_string()),
                (ref_signal_len_link, "ref_signal_len".to_string()),
            ],
            vec![("output_audio".to_string(), audio_tensor_output_link)],
        )
        .to_any(),
    );

    let audio_output_link = SuperGraphNodeTensorToAudioClip::new_and_add(
        &mut builder,
        audio_tensor_output_link,
        SAMPLE_RATE,
        rng,
    );

    // Build the top-level SuperGraph
    let sg_inputs = vec![
        text_input_link.to_any(),
        ref_audio_link.to_any(),
        max_duration_link.to_any(),
        preprocess_weights.to_any(),
        transformer_weights.to_any(),
        decode_weights.to_any(),
        time_steps_link.to_any(),
        iteration_count_link.to_any(),
    ];
    let sg_outputs = vec![audio_output_link.to_any()];
    let super_graph = builder.build(rng, &sg_inputs, &sg_outputs);

    TextToSpeechInterface {
        super_graph,
        text_input_link,
        model_weights: vec![preprocess_weights, transformer_weights, decode_weights],
        audio_output_link,
        sample_rate: SAMPLE_RATE,
        input_config: TTSInputConfig::F5 {
            ref_audio_link,
            max_duration_link,
            time_steps_link,
            iteration_count_link,
            nfe_steps: NFE_STEPS,
        },
    }
}

/// Build the inner Scan loop for the F5-TTS ODE denoising.
///
/// Each iteration: Transformer predicts denoised signal, Euler step updates noise.
/// noise_next = noise + dt * (denoised - noise), where dt = 1/NFE_STEPS.
#[allow(clippy::too_many_arguments)]
fn build_f5_denoising_loop(
    builder: &mut SuperGraphBuilder,
    rng: &mut impl rand::Rng,
    transformer_weights: SuperGraphLink,
    initial_noise: SuperGraphLink,
    rope_cos: SuperGraphLink,
    rope_sin: SuperGraphLink,
    cat_mel_text: SuperGraphLink,
    cat_mel_text_drop: SuperGraphLink,
    qk_rotated_empty: SuperGraphLink,
    time_steps_input: SuperGraphLink,
    iteration_count_input: SuperGraphLink,
) -> SuperGraphLink {
    let outer_final_denoised = builder.new_tensor_link(rng);

    let mut inner_builder = SuperGraphBuilder::new();

    // Inner links for simple inputs (constant across iterations)
    let inner_transformer_weights = inner_builder.new_model_link(rng);
    let inner_rope_cos = inner_builder.new_tensor_link(rng);
    let inner_rope_sin = inner_builder.new_tensor_link(rng);
    let inner_cat_mel_text = inner_builder.new_tensor_link(rng);
    let inner_cat_mel_text_drop = inner_builder.new_tensor_link(rng);
    let inner_qk_rotated_empty = inner_builder.new_tensor_link(rng);

    // Inner links for state (noise carried between iterations)
    let inner_noise_in = inner_builder.new_tensor_link(rng);
    let inner_noise_out = inner_builder.new_tensor_link(rng);

    // Inner link for scan input (time_step per iteration)
    let inner_time_step = inner_builder.new_tensor_link(rng);

    // --- Inner Node A: Transformer model execution ---
    let raw_denoised = inner_builder.new_tensor_link(rng);
    inner_builder.add_node(
        SuperGraphNodeModelExecution::new(
            rng,
            inner_transformer_weights,
            1, // model index 1
            vec![
                (inner_noise_in, "noise".to_string()),
                (inner_rope_cos, "rope_cos".to_string()),
                (inner_rope_sin, "rope_sin".to_string()),
                (inner_cat_mel_text, "cat_mel_text".to_string()),
                (inner_cat_mel_text_drop, "cat_mel_text_drop".to_string()),
                (inner_qk_rotated_empty, "qk_rotated_empty".to_string()),
                (inner_time_step, "time_step".to_string()),
            ],
            vec![("denoised".to_string(), raw_denoised)],
        )
        .to_any(),
    );

    // --- Inner Node B: Euler step ---
    // noise_next = noise + dt * (denoised - noise)
    //            = (1 - dt) * noise + dt * denoised
    // where dt = 1.0 / NFE_STEPS
    {
        let (mut mg, input_map) =
            MilliOpGraph::new([inner_noise_in.global_id(), raw_denoised.global_id()], rng);
        let noise_in = *input_map.get(&inner_noise_in.global_id()).unwrap();
        let denoised_in = *input_map.get(&raw_denoised.global_id()).unwrap();

        // dt = 1/32 as f16 (matching model dtype)
        let dt_val = 1.0f32 / NFE_STEPS as f32;
        let dt = Constant::push_new(
            &mut mg,
            NDArrayNumericTensor::from_vec_shape(vec![dt_val], &vec![1]).unwrap(),
            rng,
        );
        let dt = Cast::push_new(&mut mg, dt, DType::F16, rng);

        // one_minus_dt = 1 - dt
        let one = Constant::push_new(
            &mut mg,
            NDArrayNumericTensor::from_vec_shape(vec![1.0f32], &vec![1]).unwrap(),
            rng,
        );
        let one = Cast::push_new(&mut mg, one, DType::F16, rng);
        let one_minus_dt = SimpleBinary::sub(&mut mg, one, dt, rng);

        // noise_next = one_minus_dt * noise + dt * denoised
        let scaled_noise = SimpleBinary::mul(&mut mg, one_minus_dt, noise_in, rng);
        let scaled_denoised = SimpleBinary::mul(&mut mg, dt, denoised_in, rng);
        let noise_next = SimpleBinary::add(&mut mg, scaled_noise, scaled_denoised, rng);

        mg.set_output_map(std::iter::once((noise_next, inner_noise_out.global_id())));
        inner_builder.add_node(SuperGraphNodeMilliOpGraph::new(mg, rng).to_any());
    }

    // Build inner graph
    let inner_inputs = vec![
        inner_transformer_weights.to_any(),
        inner_rope_cos.to_any(),
        inner_rope_sin.to_any(),
        inner_cat_mel_text.to_any(),
        inner_cat_mel_text_drop.to_any(),
        inner_qk_rotated_empty.to_any(),
        inner_noise_in.to_any(),
        inner_time_step.to_any(),
    ];
    let inner_outputs = vec![inner_noise_out.to_any()];
    let inner_graph = inner_builder.build(rng, &inner_inputs, &inner_outputs);

    // Create scan node
    let scan_node = SuperGraphNodeScan::new(
        inner_graph,
        iteration_count_input,
        // simple_inputs: constant per iteration
        vec![
            SuperGraphLinkDouble::new(transformer_weights, inner_transformer_weights),
            SuperGraphLinkDouble::new(rope_cos, inner_rope_cos),
            SuperGraphLinkDouble::new(rope_sin, inner_rope_sin),
            SuperGraphLinkDouble::new(cat_mel_text, inner_cat_mel_text),
            SuperGraphLinkDouble::new(cat_mel_text_drop, inner_cat_mel_text_drop),
            SuperGraphLinkDouble::new(qk_rotated_empty, inner_qk_rotated_empty),
        ],
        // state_links: noise carried between iterations
        vec![SuperGraphLinkTriple::new(
            initial_noise,
            inner_noise_in,
            inner_noise_out,
        )],
        // scan_inputs: time_step sliced from [nfe_steps-1] tensor
        vec![(time_steps_input, inner_time_step, 0)],
        // scan_outputs: none (we only care about final state)
        vec![],
        // simple_outputs: final noise state
        vec![SuperGraphLinkDouble::new(
            inner_noise_out,
            outer_final_denoised,
        )],
        rng,
    );
    builder.add_node(scan_node.to_any());

    outer_final_denoised
}
