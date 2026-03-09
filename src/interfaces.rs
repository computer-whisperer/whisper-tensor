use crate::backends::ModelLoadedTensorCache;
use crate::backends::eval_backend::EvalBackend;
use crate::backends::ndarray_backend::NDArrayNumericTensor;
use crate::compiler::CompiledProgram;
use crate::dtype::DType;
use crate::metadata::TokenizerInfo;
use crate::milli_graph::MilliOpGraph;
use crate::milli_graph::ops::{
    ArgMax, Cast, Concat as MilliConcat, Constant, Pad, PadMode, SimpleBinary, SimpleUnaryOp,
    Unsqueeze,
};
use crate::model::Model;
use crate::numeric_tensor::NumericTensor;
use crate::super_graph::cache::{SuperGraphCache, SuperGraphTensorCache};
use crate::super_graph::data::SuperGraphData;
use crate::super_graph::links::{
    SuperGraphLink, SuperGraphLinkDouble, SuperGraphLinkHash, SuperGraphLinkTensor,
    SuperGraphLinkTensorMap, SuperGraphLinkTriple,
};
use crate::super_graph::nodes::{
    SuperGraphNode, SuperGraphNodeMilliOpGraph, SuperGraphNodeModelExecution, SuperGraphNodeScan,
};
use crate::super_graph::{SuperGraph, SuperGraphBuilder, SuperGraphContext, SuperGraphError};
use crate::tensor_rank::DynRank;
use crate::tokenizer::{AnyTokenizer, Tokenizer};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum AnyInterface {
    TextInferenceTokensInLogitOutInterface(TextInferenceTokensInLogitOutInterface),
    ImageGenerationInterface(ImageGenerationInterface),
    TextToSpeechInterface(TextToSpeechInterface),
    PiperInterface(PiperInterface),
}

impl AnyInterface {
    pub fn name(&self) -> String {
        match self {
            AnyInterface::TextInferenceTokensInLogitOutInterface(_) => {
                "TextInferenceTokensInLogitsOut".to_string()
            }
            AnyInterface::ImageGenerationInterface(_) => "ImageGeneration".to_string(),
            AnyInterface::TextToSpeechInterface(_) => "TextToSpeech".to_string(),
            AnyInterface::PiperInterface(_) => "Piper".to_string(),
        }
    }

    pub fn get_super_graph(&self) -> &SuperGraph {
        match self {
            AnyInterface::TextInferenceTokensInLogitOutInterface(x) => &x.super_graph,
            AnyInterface::ImageGenerationInterface(x) => &x.super_graph,
            AnyInterface::TextToSpeechInterface(x) => &x.super_graph,
            AnyInterface::PiperInterface(x) => &x.super_graph,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TextInferenceTokensInLogitOutInterface {
    pub cache_key_input_link: SuperGraphLinkHash,
    pub token_context_input_link: SuperGraphLinkTensor,
    pub model_input_link: SuperGraphLinkTensorMap,
    pub logit_output_link: SuperGraphLinkTensor,
    pub super_graph: SuperGraph,
    pub tokenizer: TokenizerInfo,
}

impl TextInferenceTokensInLogitOutInterface {
    #[allow(clippy::too_many_arguments)]
    pub fn run_string_in_string_out(
        &self,
        model: &Model,
        compiled_model: Option<&CompiledProgram>,
        text_in: String,
        tokenizer_cache: &mut HashMap<TokenizerInfo, Arc<AnyTokenizer>>,
        tensor_cache: Option<&mut ModelLoadedTensorCache>,
        super_graph_caches: Option<&mut SuperGraphCache>,
        backend: &mut EvalBackend,
    ) -> Result<String, SuperGraphError> {
        let tokenizer = {
            if let Some(x) = tokenizer_cache.get(&self.tokenizer) {
                x.clone()
            } else {
                let x = Arc::new(AnyTokenizer::from_tokenizer_info(&self.tokenizer));
                tokenizer_cache.insert(self.tokenizer.clone(), x.clone());
                x
            }
        };
        let tokens = tokenizer.encode(text_in.as_str());
        let tokens_tensor = NumericTensor::from_vec(tokens.clone()).to_dyn_rank();

        let super_graph_data = {
            let mut super_graph_data = SuperGraphData::new();
            super_graph_data
                .tensor_maps
                .insert(self.model_input_link, model.get_tensor_store());
            super_graph_data
                .tensors
                .insert(self.token_context_input_link, tokens_tensor);
            super_graph_data.hashes.insert(self.cache_key_input_link, 0);
            super_graph_data
        };
        let super_graph_output = {
            let mut observer = ();
            let mut super_graph_tensor_cache = SuperGraphTensorCache::new();
            if let Some(tensor_cache) = &tensor_cache {
                super_graph_tensor_cache
                    .caches
                    .push((model.get_tensor_store(), (*tensor_cache).clone()))
            }
            let compiled_models = {
                let mut compiled_models = Vec::new();
                if let Some(compiled_model) = compiled_model {
                    compiled_models.push((model, compiled_model));
                }
                compiled_models
            };
            let mut context = SuperGraphContext {
                observer: &mut observer,
                eval_backend: backend,
                super_graph_tensor_cache: &mut super_graph_tensor_cache,
                caches: super_graph_caches,
                symbolic_graphs: vec![model.get_symbolic_graph()],
                use_compiled_models: compiled_model.is_some(),
                compiled_models: Some(compiled_models),
            };
            let res = self.super_graph.run(super_graph_data, &mut context)?;
            if let Some(tensor_cache) = tensor_cache {
                *tensor_cache = context.super_graph_tensor_cache.caches.remove(0).1
            }
            res
        };
        let logits = super_graph_output
            .tensors
            .get(&self.logit_output_link)
            .unwrap();
        let logits_shape = logits.shape();
        // Select last position
        let logits = logits.slice(
            &[logits_shape[0] - 1..logits_shape[0], 0..logits_shape[1]],
            backend,
        )?;
        let logits = logits.squeeze(0)?;
        let token_id = logits.argmax(0, true, false, backend)?;

        let token_id: u32 = token_id.first_element().into();
        let token_str = tokenizer.decode(&[token_id])?;
        Ok(token_str)
    }

    pub fn get_tokenizer(&self) -> &TokenizerInfo {
        &self.tokenizer
    }

    pub fn to_any(self) -> AnyInterface {
        AnyInterface::TextInferenceTokensInLogitOutInterface(self)
    }
}

// ============================================================================
// Image Generation Interface
// ============================================================================

/// Scheduler type for the denoising loop.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SchedulerType {
    /// Euler discrete scheduler (SD 1.5, SD 2, SDXL).
    /// Initial noise is scaled by init_sigma.
    EulerDiscrete,
    /// Rectified flow scheduler (Flux).
    /// No noise scaling.
    RectifiedFlow,
}

/// How to encode a text prompt into token IDs for a specific input slot.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum PromptEncoding {
    /// CLIP-style: prepend BOS, append EOS, pad to seq_len with pad token.
    ClipStyle { bos: u32, eos: u32, pad: u32 },
    /// Raw encode and pad to seq_len (e.g. T5 SentencePiece).
    RawPad { pad: u32 },
}

/// A single prompt input slot: tokenizer, target link, sequence length, and encoding style.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PromptInput {
    pub tokenizer: TokenizerInfo,
    pub link: SuperGraphLinkTensor,
    pub seq_len: usize,
    pub encoding: PromptEncoding,
}

impl PromptInput {
    /// Tokenize a text prompt according to this input's encoding style.
    pub fn tokenize(&self, tokenizer: &dyn Tokenizer, text: &str) -> Vec<i32> {
        match &self.encoding {
            PromptEncoding::ClipStyle { bos, eos, pad } => {
                let mut encoded = tokenizer.encode(text);
                // Strip BOS/EOS if the tokenizer's post-processor already added them
                if encoded.first() == Some(bos) {
                    encoded.remove(0);
                }
                if encoded.last() == Some(eos) {
                    encoded.pop();
                }
                let mut ids = Vec::with_capacity(self.seq_len);
                ids.push(*bos as i32);
                let max_text_tokens = self.seq_len.saturating_sub(2);
                for &id in encoded.iter().take(max_text_tokens) {
                    ids.push(id as i32);
                }
                ids.push(*eos as i32);
                ids.resize(self.seq_len, *pad as i32);
                ids
            }
            PromptEncoding::RawPad { pad } => {
                let encoded = tokenizer.encode(text);
                let mut ids: Vec<i32> = encoded
                    .iter()
                    .take(self.seq_len)
                    .map(|&id| id as i32)
                    .collect();
                ids.resize(self.seq_len, *pad as i32);
                ids
            }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImageGenerationInterface {
    pub super_graph: SuperGraph,
    // Prompt inputs
    pub positive_prompts: Vec<PromptInput>,
    pub negative_prompts: Option<Vec<PromptInput>>,
    // Latent / scheduler inputs
    pub initial_latent_input: SuperGraphLinkTensor,
    pub timesteps_input: SuperGraphLinkTensor,
    pub dt_input: SuperGraphLinkTensor,
    pub sigmas_input: SuperGraphLinkTensor,
    pub iteration_count_input: SuperGraphLinkTensor,
    pub guidance_scale_input: Option<SuperGraphLinkTensor>,
    // Model weight maps (in order matching loader's model_ids)
    pub model_weights: Vec<SuperGraphLinkTensorMap>,
    // Output
    pub image_output: SuperGraphLinkTensor,
    /// Scheduler type for the denoising loop.
    pub scheduler: SchedulerType,
    /// Number of latent channels (4 for SD/SDXL, 16 for Flux).
    pub latent_channels: usize,
}

/// Helper: build a MilliOpGraph node that casts a tensor to a target dtype.
fn build_cast_node(
    builder: &mut SuperGraphBuilder,
    rng: &mut impl Rng,
    input: SuperGraphLinkTensor,
    dtype: DType,
) -> SuperGraphLinkTensor {
    let output = builder.new_tensor_link(rng);
    let (mut mg, input_map) = MilliOpGraph::new(std::iter::once(input.global_id()), rng);
    let inp = *input_map.get(&input.global_id()).unwrap();
    let casted = Cast::push_new(&mut mg, inp, dtype, rng);
    mg.set_output_map(std::iter::once((casted, output.global_id())));
    builder.add_node(SuperGraphNodeMilliOpGraph::new(mg, rng).to_any());
    output
}

/// Helper: build the denoising scan loop.
///
/// Returns the final latent link (F32, after Euler integration).
#[allow(clippy::too_many_arguments)]
fn build_denoising_loop(
    builder: &mut SuperGraphBuilder,
    rng: &mut impl Rng,
    unet_weights: SuperGraphLinkTensorMap,
    cond_context: SuperGraphLinkTensor,
    uncond_context: SuperGraphLinkTensor,
    cond_y: Option<SuperGraphLinkTensor>,
    uncond_y: Option<SuperGraphLinkTensor>,
    guidance_scale_input: SuperGraphLinkTensor,
    initial_latent_input: SuperGraphLinkTensor,
    timesteps_input: SuperGraphLinkTensor,
    dt_input: SuperGraphLinkTensor,
    sigmas_input: SuperGraphLinkTensor,
    iteration_count_input: SuperGraphLinkTensor,
    model_dtype: DType,
    unet_model_index: usize,
) -> SuperGraphLinkTensor {
    let outer_final_latent = builder.new_tensor_link(rng);

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
    let inner_sigma = inner_builder.new_tensor_link(rng);

    // Optional ADM conditioning links
    let inner_cond_y = cond_y.as_ref().map(|_| inner_builder.new_tensor_link(rng));
    let inner_uncond_y = uncond_y
        .as_ref()
        .map(|_| inner_builder.new_tensor_link(rng));

    // Inner node 1: Prep — scale latent by 1/sqrt(sigma²+1), cast to model_dtype, reshape timestep
    let cast_latent = inner_builder.new_tensor_link(rng);
    let cast_timestep = inner_builder.new_tensor_link(rng);
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
        let one = Constant::push_new(
            &mut mg,
            NDArrayNumericTensor::from_vec_shape(vec![1.0f32], &vec![1]).unwrap(),
            rng,
        );
        let sigma_sq_plus_1 = SimpleBinary::add(&mut mg, sigma_sq, one, rng);
        let sqrt_val = SimpleUnaryOp::sqrt(&mut mg, sigma_sq_plus_1, rng);
        let inv_scale = SimpleBinary::div(&mut mg, one, sqrt_val, rng);
        let scaled_lat = SimpleBinary::mul(&mut mg, lat_in, inv_scale, rng);

        let lat_cast = Cast::push_new(&mut mg, scaled_lat, model_dtype, rng);
        let ts_cast = Cast::push_new(&mut mg, ts_in, model_dtype, rng);
        let zero_axis = Constant::push_new(
            &mut mg,
            NDArrayNumericTensor::from_vec_shape(vec![0i64], &vec![1]).unwrap(),
            rng,
        );
        let ts_reshaped = Unsqueeze::push_new(&mut mg, ts_cast, zero_axis, rng);

        mg.set_output_map([
            (lat_cast, cast_latent.global_id()),
            (ts_reshaped, cast_timestep.global_id()),
        ]);
        inner_builder.add_node(SuperGraphNodeMilliOpGraph::new(mg, rng).to_any());
    }

    // Inner node 2: UNet unconditional
    let uncond_noise = inner_builder.new_tensor_link(rng);
    {
        let mut inputs = vec![
            (cast_latent, "sample".to_string()),
            (cast_timestep, "timestep".to_string()),
            (inner_uncond_context, "encoder_hidden_states".to_string()),
        ];
        if let Some(uy) = inner_uncond_y {
            inputs.push((uy, "y".to_string()));
        }
        inner_builder.add_node(
            SuperGraphNodeModelExecution::new(
                rng,
                inner_unet_weights,
                unet_model_index,
                inputs,
                vec![("out_sample".to_string(), uncond_noise)],
            )
            .to_any(),
        );
    }

    // Inner node 3: UNet conditional
    let cond_noise = inner_builder.new_tensor_link(rng);
    {
        let mut inputs = vec![
            (cast_latent, "sample".to_string()),
            (cast_timestep, "timestep".to_string()),
            (inner_cond_context, "encoder_hidden_states".to_string()),
        ];
        if let Some(cy) = inner_cond_y {
            inputs.push((cy, "y".to_string()));
        }
        inner_builder.add_node(
            SuperGraphNodeModelExecution::new(
                rng,
                inner_unet_weights,
                unet_model_index,
                inputs,
                vec![("out_sample".to_string(), cond_noise)],
            )
            .to_any(),
        );
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
        let uncond_f32 = Cast::push_new(&mut mg, uncond_in, DType::F32, rng);
        let cond_f32 = Cast::push_new(&mut mg, cond_in, DType::F32, rng);

        // CFG: uncond + scale * (cond - uncond)
        let diff = SimpleBinary::sub(&mut mg, cond_f32, uncond_f32, rng);
        let scaled = SimpleBinary::mul(&mut mg, diff, gs_in, rng);
        let guided = SimpleBinary::add(&mut mg, uncond_f32, scaled, rng);

        // Euler step: latent + guided * dt
        let step = SimpleBinary::mul(&mut mg, guided, dt_in, rng);
        let latent_next = SimpleBinary::add(&mut mg, lat_in, step, rng);

        mg.set_output_map(std::iter::once((latent_next, inner_latent_out.global_id())));
        inner_builder.add_node(SuperGraphNodeMilliOpGraph::new(mg, rng).to_any());
    }

    // Build inner graph
    let mut inner_inputs: Vec<_> = vec![
        inner_unet_weights.to_any(),
        inner_cond_context.to_any(),
        inner_uncond_context.to_any(),
        inner_guidance_scale.to_any(),
        inner_latent_in.to_any(),
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
    let inner_outputs: Vec<_> = vec![inner_latent_out.to_any()];
    let inner_graph = inner_builder.build(rng, &inner_inputs, &inner_outputs);

    // Create scan node
    let mut simple_inputs = vec![
        SuperGraphLinkDouble::TensorMap(unet_weights, inner_unet_weights),
        SuperGraphLinkDouble::Tensor(cond_context, inner_cond_context),
        SuperGraphLinkDouble::Tensor(uncond_context, inner_uncond_context),
        SuperGraphLinkDouble::Tensor(guidance_scale_input, inner_guidance_scale),
    ];
    if let (Some(cy_outer), Some(cy_inner)) = (cond_y, inner_cond_y) {
        simple_inputs.push(SuperGraphLinkDouble::Tensor(cy_outer, cy_inner));
    }
    if let (Some(uy_outer), Some(uy_inner)) = (uncond_y, inner_uncond_y) {
        simple_inputs.push(SuperGraphLinkDouble::Tensor(uy_outer, uy_inner));
    }

    let scan_node = SuperGraphNodeScan::new(
        inner_graph,
        iteration_count_input,
        simple_inputs,
        // state_links: (initial, inner_in, inner_out)
        vec![SuperGraphLinkTriple::Tensor(
            initial_latent_input,
            inner_latent_in,
            inner_latent_out,
        )],
        // scan_inputs: (outer, inner, axis)
        vec![
            (timesteps_input, inner_timestep, 0),
            (dt_input, inner_dt, 0),
            (sigmas_input, inner_sigma, 0),
        ],
        // scan_outputs: none
        vec![],
        // simple_outputs: final latent
        vec![SuperGraphLinkDouble::Tensor(
            inner_latent_out,
            outer_final_latent,
        )],
        rng,
    );
    builder.add_node(scan_node.to_any());

    outer_final_latent
}

/// Helper: build the VAE decode node (scale latent + decode).
fn build_vae_decode(
    builder: &mut SuperGraphBuilder,
    rng: &mut impl Rng,
    latent: SuperGraphLinkTensor,
    vae_weights: SuperGraphLinkTensorMap,
    vae_model_index: usize,
    vae_scale_factor: f32,
    model_dtype: DType,
) -> SuperGraphLinkTensor {
    // Scale latent by 1/vae_scale_factor and cast to model_dtype
    let scaled_latent = builder.new_tensor_link(rng);
    {
        let (mut mg, input_map) = MilliOpGraph::new(std::iter::once(latent.global_id()), rng);
        let lat_in = *input_map.get(&latent.global_id()).unwrap();

        let scale = Constant::push_new(
            &mut mg,
            NDArrayNumericTensor::from_vec_shape(vec![1.0f32 / vae_scale_factor], &vec![1])
                .unwrap(),
            rng,
        );
        let scaled = SimpleBinary::mul(&mut mg, lat_in, scale, rng);
        let scaled_cast = Cast::push_new(&mut mg, scaled, model_dtype, rng);

        mg.set_output_map(std::iter::once((scaled_cast, scaled_latent.global_id())));
        builder.add_node(SuperGraphNodeMilliOpGraph::new(mg, rng).to_any());
    }

    // VAE decoder
    let image_output = builder.new_tensor_link(rng);
    builder.add_node(
        SuperGraphNodeModelExecution::new(
            rng,
            vae_weights,
            vae_model_index,
            vec![(scaled_latent, "latent_sample".to_string())],
            vec![("sample".to_string(), image_output)],
        )
        .to_any(),
    );

    image_output
}

/// Helper: build a MilliOpGraph that computes ArgMax(input_ids, axis=1) → eos_indices.
fn build_eos_indices_node(
    builder: &mut SuperGraphBuilder,
    rng: &mut impl Rng,
    input_ids: SuperGraphLinkTensor,
) -> SuperGraphLinkTensor {
    let eos_indices = builder.new_tensor_link(rng);
    let (mut mg, input_map) = MilliOpGraph::new(std::iter::once(input_ids.global_id()), rng);
    let ids_in = *input_map.get(&input_ids.global_id()).unwrap();
    // EOS (49407) is the max token in CLIP vocab, so argmax finds its position
    let argmax = ArgMax::push_new(&mut mg, ids_in, 1, false, false, rng);
    mg.set_output_map(std::iter::once((argmax, eos_indices.global_id())));
    builder.add_node(SuperGraphNodeMilliOpGraph::new(mg, rng).to_any());
    eos_indices
}

/// Helper: build the Flux rectified flow denoising scan loop.
///
/// No CFG — single DiT forward pass per step. No input scaling.
/// DiT inputs: latent_sample, timestep, clip_pooled, t5_hidden_states
/// Returns the final latent link (F32, after Euler integration).
#[allow(clippy::too_many_arguments)]
fn build_flux_denoising_loop(
    builder: &mut SuperGraphBuilder,
    rng: &mut impl Rng,
    dit_weights: SuperGraphLinkTensorMap,
    clip_pooled: SuperGraphLinkTensor,
    t5_hidden: SuperGraphLinkTensor,
    initial_latent_input: SuperGraphLinkTensor,
    timesteps_input: SuperGraphLinkTensor,
    dt_input: SuperGraphLinkTensor,
    _sigmas_input: SuperGraphLinkTensor,
    iteration_count_input: SuperGraphLinkTensor,
    guidance_input: Option<SuperGraphLinkTensor>,
    model_dtype: DType,
    dit_model_index: usize,
) -> SuperGraphLinkTensor {
    let outer_final_latent = builder.new_tensor_link(rng);

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

    // Inner node 1: Prep — cast latent to model_dtype, reshape timestep (and guidance)
    let cast_latent = inner_builder.new_tensor_link(rng);
    let cast_timestep = inner_builder.new_tensor_link(rng);
    let cast_guidance = inner_guidance.map(|_| inner_builder.new_tensor_link(rng));
    {
        let mut input_ids = vec![inner_latent_in.global_id(), inner_timestep.global_id()];
        if let Some(ig) = inner_guidance {
            input_ids.push(ig.global_id());
        }
        let (mut mg, input_map) = MilliOpGraph::new(input_ids, rng);
        let lat_in = *input_map.get(&inner_latent_in.global_id()).unwrap();
        let ts_in = *input_map.get(&inner_timestep.global_id()).unwrap();

        // No sigma scaling for Flux (rectified flow operates directly on latents)
        let lat_cast = Cast::push_new(&mut mg, lat_in, model_dtype, rng);

        // Reshape timestep from scalar to [1, 1]
        let ts_shape = Constant::push_new(
            &mut mg,
            NDArrayNumericTensor::from_vec_shape(vec![1i64, 1], &vec![2]).unwrap(),
            rng,
        );
        let ts_reshaped =
            crate::milli_graph::ops::Reshape::push_new(&mut mg, ts_in, ts_shape, false, rng);

        let mut outputs = vec![
            (lat_cast, cast_latent.global_id()),
            (ts_reshaped, cast_timestep.global_id()),
        ];

        // Reshape guidance from scalar to [1, 1] (same as timestep)
        if let (Some(ig), Some(cg)) = (inner_guidance, cast_guidance) {
            let g_in = *input_map.get(&ig.global_id()).unwrap();
            let g_shape = Constant::push_new(
                &mut mg,
                NDArrayNumericTensor::from_vec_shape(vec![1i64, 1], &vec![2]).unwrap(),
                rng,
            );
            let g_reshaped =
                crate::milli_graph::ops::Reshape::push_new(&mut mg, g_in, g_shape, false, rng);
            outputs.push((g_reshaped, cg.global_id()));
        }

        mg.set_output_map(outputs);
        inner_builder.add_node(SuperGraphNodeMilliOpGraph::new(mg, rng).to_any());
    }

    // Inner node 2: DiT forward pass
    let dit_output = inner_builder.new_tensor_link(rng);
    let mut dit_inputs = vec![
        (cast_latent, "latent_sample".to_string()),
        (cast_timestep, "timestep".to_string()),
        (inner_clip_pooled, "clip_pooled".to_string()),
        (inner_t5_hidden, "t5_hidden_states".to_string()),
    ];
    if let Some(cg) = cast_guidance {
        dit_inputs.push((cg, "guidance".to_string()));
    }
    inner_builder.add_node(
        SuperGraphNodeModelExecution::new(
            rng,
            inner_dit_weights,
            dit_model_index,
            dit_inputs,
            vec![("out_sample".to_string(), dit_output)],
        )
        .to_any(),
    );

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
        let velocity_f32 = Cast::push_new(&mut mg, velocity, DType::F32, rng);

        // Euler step: latent + velocity * dt
        let step = SimpleBinary::mul(&mut mg, velocity_f32, dt_in, rng);
        let latent_next = SimpleBinary::add(&mut mg, lat_in, step, rng);

        mg.set_output_map(std::iter::once((latent_next, inner_latent_out.global_id())));
        inner_builder.add_node(SuperGraphNodeMilliOpGraph::new(mg, rng).to_any());
    }

    // Build inner graph
    let mut inner_inputs: Vec<_> = vec![
        inner_dit_weights.to_any(),
        inner_clip_pooled.to_any(),
        inner_t5_hidden.to_any(),
        inner_latent_in.to_any(),
        inner_timestep.to_any(),
        inner_dt.to_any(),
    ];
    if let Some(ig) = inner_guidance {
        inner_inputs.push(ig.to_any());
    }
    let inner_outputs: Vec<_> = vec![inner_latent_out.to_any()];
    let inner_graph = inner_builder.build(rng, &inner_inputs, &inner_outputs);

    // Create scan node
    let mut simple_inputs = vec![
        SuperGraphLinkDouble::TensorMap(dit_weights, inner_dit_weights),
        SuperGraphLinkDouble::Tensor(clip_pooled, inner_clip_pooled),
        SuperGraphLinkDouble::Tensor(t5_hidden, inner_t5_hidden),
    ];
    if let (Some(outer_g), Some(ig)) = (guidance_input, inner_guidance) {
        simple_inputs.push(SuperGraphLinkDouble::Tensor(outer_g, ig));
    }

    let scan_node = SuperGraphNodeScan::new(
        inner_graph,
        iteration_count_input,
        simple_inputs,
        // state_links: latent carried across iterations
        vec![SuperGraphLinkTriple::Tensor(
            initial_latent_input,
            inner_latent_in,
            inner_latent_out,
        )],
        // scan_inputs: timestep and dt scanned along axis 0
        vec![
            (timesteps_input, inner_timestep, 0),
            (dt_input, inner_dt, 0),
        ],
        // scan_outputs: none
        vec![],
        // simple_outputs: final latent
        vec![SuperGraphLinkDouble::Tensor(
            inner_latent_out,
            outer_final_latent,
        )],
        rng,
    );
    builder.add_node(scan_node.to_any());

    outer_final_latent
}

/// Helper: build the Flux VAE decode node.
///
/// Flux VAE scaling: latent_for_vae = latent / 0.3611 + 0.1159
fn build_flux_vae_decode(
    builder: &mut SuperGraphBuilder,
    rng: &mut impl Rng,
    latent: SuperGraphLinkTensor,
    vae_weights: SuperGraphLinkTensorMap,
    vae_model_index: usize,
) -> SuperGraphLinkTensor {
    // Scale latent: x / 0.3611 + 0.1159, cast to F32 for VAE
    let scaled_latent = builder.new_tensor_link(rng);
    {
        let (mut mg, input_map) = MilliOpGraph::new(std::iter::once(latent.global_id()), rng);
        let lat_in = *input_map.get(&latent.global_id()).unwrap();

        // Cast to F32 first (denoising loop outputs in model_dtype which may be BF16)
        let f32_lat = Cast::push_new(&mut mg, lat_in, DType::F32, rng);

        let inv_scale = Constant::push_new(
            &mut mg,
            NDArrayNumericTensor::from_vec_shape(vec![1.0f32 / 0.3611], &vec![1]).unwrap(),
            rng,
        );
        let shift = Constant::push_new(
            &mut mg,
            NDArrayNumericTensor::from_vec_shape(vec![0.1159f32], &vec![1]).unwrap(),
            rng,
        );
        let scaled = SimpleBinary::mul(&mut mg, f32_lat, inv_scale, rng);
        let shifted = SimpleBinary::add(&mut mg, scaled, shift, rng);

        mg.set_output_map(std::iter::once((shifted, scaled_latent.global_id())));
        builder.add_node(SuperGraphNodeMilliOpGraph::new(mg, rng).to_any());
    }

    // VAE decoder
    let image_output = builder.new_tensor_link(rng);
    builder.add_node(
        SuperGraphNodeModelExecution::new(
            rng,
            vae_weights,
            vae_model_index,
            vec![(scaled_latent, "latent_sample".to_string())],
            vec![("sample".to_string(), image_output)],
        )
        .to_any(),
    );

    image_output
}

impl ImageGenerationInterface {
    /// Build interface for single-text-encoder models with CFG (SD 1.5, SD 2).
    ///
    /// Model weights order: [text_encoder, unet, vae_decoder]
    pub fn new_single_te_cfg(
        rng: &mut impl Rng,
        tokenizer: TokenizerInfo,
        model_dtype: DType,
        vae_scale_factor: f32,
    ) -> Self {
        let mut builder = SuperGraphBuilder::new();

        // Create input links
        let cond_ids_input = builder.new_tensor_link(rng);
        let negative_cond_ids_input = builder.new_tensor_link(rng);
        let initial_latent_input = builder.new_tensor_link(rng);
        let timesteps_input = builder.new_tensor_link(rng);
        let dt_input = builder.new_tensor_link(rng);
        let sigmas_input = builder.new_tensor_link(rng);
        let iteration_count_input = builder.new_tensor_link(rng);
        let guidance_scale_input = builder.new_tensor_link(rng);
        let te_weights = builder.new_model_link(rng);
        let unet_weights = builder.new_model_link(rng);
        let vae_weights = builder.new_model_link(rng);

        // Text encoder: conditional → F32, cast to model_dtype
        let cond_hidden_f32 = builder.new_tensor_link(rng);
        builder.add_node(
            SuperGraphNodeModelExecution::new(
                rng,
                te_weights,
                0,
                vec![(cond_ids_input, "input_ids".to_string())],
                vec![("last_hidden_state".to_string(), cond_hidden_f32)],
            )
            .to_any(),
        );
        let cond_context = build_cast_node(&mut builder, rng, cond_hidden_f32, model_dtype);

        // Text encoder: unconditional → F32, cast to model_dtype
        let uncond_hidden_f32 = builder.new_tensor_link(rng);
        builder.add_node(
            SuperGraphNodeModelExecution::new(
                rng,
                te_weights,
                0,
                vec![(negative_cond_ids_input, "input_ids".to_string())],
                vec![("last_hidden_state".to_string(), uncond_hidden_f32)],
            )
            .to_any(),
        );
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

        // VAE decode
        let image_output = build_vae_decode(
            &mut builder,
            rng,
            final_latent,
            vae_weights,
            2,
            vae_scale_factor,
            model_dtype,
        );

        // Build outer graph
        let model_weights = vec![te_weights, unet_weights, vae_weights];
        let input_links: Vec<_> = vec![
            cond_ids_input.to_any(),
            negative_cond_ids_input.to_any(),
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

        let clip_prompt = PromptInput {
            tokenizer,
            link: cond_ids_input,
            seq_len: 77,
            encoding: PromptEncoding::ClipStyle {
                bos: 49406,
                eos: 49407,
                pad: 0,
            },
        };
        let clip_neg_prompt = PromptInput {
            tokenizer: clip_prompt.tokenizer.clone(),
            link: negative_cond_ids_input,
            seq_len: 77,
            encoding: clip_prompt.encoding.clone(),
        };

        Self {
            super_graph,
            positive_prompts: vec![clip_prompt],
            negative_prompts: Some(vec![clip_neg_prompt]),
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

    /// Build interface for SDXL (dual text encoders + ADM conditioning + CFG).
    ///
    /// Model weights order: [text_encoder_1, text_encoder_2, unet, vae_decoder]
    pub fn new_sdxl(rng: &mut impl Rng, tokenizer: TokenizerInfo, model_dtype: DType) -> Self {
        let mut builder = SuperGraphBuilder::new();

        // Create input links
        let cond_ids_input = builder.new_tensor_link(rng);
        let negative_cond_ids_input = builder.new_tensor_link(rng);
        let initial_latent_input = builder.new_tensor_link(rng);
        let timesteps_input = builder.new_tensor_link(rng);
        let dt_input = builder.new_tensor_link(rng);
        let sigmas_input = builder.new_tensor_link(rng);
        let iteration_count_input = builder.new_tensor_link(rng);
        let guidance_scale_input = builder.new_tensor_link(rng);
        let te1_weights = builder.new_model_link(rng);
        let te2_weights = builder.new_model_link(rng);
        let unet_weights = builder.new_model_link(rng);
        let vae_weights = builder.new_model_link(rng);

        // --- Conditional path ---

        // TE1 conditional: cond_ids → hidden1 [1, 77, 768] (F32)
        let cond_hidden1_f32 = builder.new_tensor_link(rng);
        builder.add_node(
            SuperGraphNodeModelExecution::new(
                rng,
                te1_weights,
                0,
                vec![(cond_ids_input, "input_ids".to_string())],
                vec![("last_hidden_state".to_string(), cond_hidden1_f32)],
            )
            .to_any(),
        );

        // Compute eos_indices for conditional
        let cond_eos = build_eos_indices_node(&mut builder, rng, cond_ids_input);

        // TE2 conditional: cond_ids + eos → penultimate [1, 77, 1280] + pooled [1, 1280] (F32)
        let cond_penult2_f32 = builder.new_tensor_link(rng);
        let cond_pooled_f32 = builder.new_tensor_link(rng);
        builder.add_node(
            SuperGraphNodeModelExecution::new(
                rng,
                te2_weights,
                1,
                vec![
                    (cond_ids_input, "input_ids".to_string()),
                    (cond_eos, "eos_indices".to_string()),
                ],
                vec![
                    ("penultimate_hidden_state".to_string(), cond_penult2_f32),
                    ("pooled_output".to_string(), cond_pooled_f32),
                ],
            )
            .to_any(),
        );

        // Concat hidden1 + penult2 → context [1, 77, 2048], pad pooled → y [1, 2816]
        // Then cast both to model_dtype
        let cond_context = builder.new_tensor_link(rng);
        let cond_y = builder.new_tensor_link(rng);
        {
            let (mut mg, input_map) = MilliOpGraph::new(
                [
                    cond_hidden1_f32.global_id(),
                    cond_penult2_f32.global_id(),
                    cond_pooled_f32.global_id(),
                ],
                rng,
            );
            let h1 = *input_map.get(&cond_hidden1_f32.global_id()).unwrap();
            let p2 = *input_map.get(&cond_penult2_f32.global_id()).unwrap();
            let pooled = *input_map.get(&cond_pooled_f32.global_id()).unwrap();

            // Concat along last dim: [1,77,768] + [1,77,1280] → [1,77,2048]
            let ctx = MilliConcat::push_new(&mut mg, vec![h1, p2], -1, rng);
            let ctx_cast = Cast::push_new(&mut mg, ctx, model_dtype, rng);

            // Pad pooled [1,1280] → [1,2816] (add 1536 zeros on right of dim 1)
            let pads = Constant::push_new(
                &mut mg,
                NDArrayNumericTensor::from_vec_shape(vec![0i64, 0, 0, 1536], &vec![4]).unwrap(),
                rng,
            );
            let padded = Pad::push_new(&mut mg, pooled, pads, None, None, PadMode::Constant, rng);
            let y_cast = Cast::push_new(&mut mg, padded, model_dtype, rng);

            mg.set_output_map([
                (ctx_cast, cond_context.global_id()),
                (y_cast, cond_y.global_id()),
            ]);
            builder.add_node(SuperGraphNodeMilliOpGraph::new(mg, rng).to_any());
        }

        // --- Unconditional path ---

        // TE1 unconditional
        let uncond_hidden1_f32 = builder.new_tensor_link(rng);
        builder.add_node(
            SuperGraphNodeModelExecution::new(
                rng,
                te1_weights,
                0,
                vec![(negative_cond_ids_input, "input_ids".to_string())],
                vec![("last_hidden_state".to_string(), uncond_hidden1_f32)],
            )
            .to_any(),
        );

        // Compute eos_indices for unconditional
        let uncond_eos = build_eos_indices_node(&mut builder, rng, negative_cond_ids_input);

        // TE2 unconditional
        let uncond_penult2_f32 = builder.new_tensor_link(rng);
        let uncond_pooled_f32 = builder.new_tensor_link(rng);
        builder.add_node(
            SuperGraphNodeModelExecution::new(
                rng,
                te2_weights,
                1,
                vec![
                    (negative_cond_ids_input, "input_ids".to_string()),
                    (uncond_eos, "eos_indices".to_string()),
                ],
                vec![
                    ("penultimate_hidden_state".to_string(), uncond_penult2_f32),
                    ("pooled_output".to_string(), uncond_pooled_f32),
                ],
            )
            .to_any(),
        );

        // Concat + pad for unconditional
        let uncond_context = builder.new_tensor_link(rng);
        let uncond_y = builder.new_tensor_link(rng);
        {
            let (mut mg, input_map) = MilliOpGraph::new(
                [
                    uncond_hidden1_f32.global_id(),
                    uncond_penult2_f32.global_id(),
                    uncond_pooled_f32.global_id(),
                ],
                rng,
            );
            let h1 = *input_map.get(&uncond_hidden1_f32.global_id()).unwrap();
            let p2 = *input_map.get(&uncond_penult2_f32.global_id()).unwrap();
            let pooled = *input_map.get(&uncond_pooled_f32.global_id()).unwrap();

            let ctx = MilliConcat::push_new(&mut mg, vec![h1, p2], -1, rng);
            let ctx_cast = Cast::push_new(&mut mg, ctx, model_dtype, rng);

            let pads = Constant::push_new(
                &mut mg,
                NDArrayNumericTensor::from_vec_shape(vec![0i64, 0, 0, 1536], &vec![4]).unwrap(),
                rng,
            );
            let padded = Pad::push_new(&mut mg, pooled, pads, None, None, PadMode::Constant, rng);
            let y_cast = Cast::push_new(&mut mg, padded, model_dtype, rng);

            mg.set_output_map([
                (ctx_cast, uncond_context.global_id()),
                (y_cast, uncond_y.global_id()),
            ]);
            builder.add_node(SuperGraphNodeMilliOpGraph::new(mg, rng).to_any());
        }

        // --- Denoising loop ---
        let final_latent = build_denoising_loop(
            &mut builder,
            rng,
            unet_weights,
            cond_context,
            uncond_context,
            Some(cond_y), // ADM conditioning
            Some(uncond_y),
            guidance_scale_input,
            initial_latent_input,
            timesteps_input,
            dt_input,
            sigmas_input,
            iteration_count_input,
            model_dtype,
            2, // unet model index (te1=0, te2=1, unet=2)
        );

        // VAE decode (SDXL uses 0.13025 scale factor)
        let image_output = build_vae_decode(
            &mut builder,
            rng,
            final_latent,
            vae_weights,
            3,
            0.13025,
            model_dtype,
        );

        // Build outer graph
        let model_weights = vec![te1_weights, te2_weights, unet_weights, vae_weights];
        let input_links: Vec<_> = vec![
            cond_ids_input.to_any(),
            negative_cond_ids_input.to_any(),
            initial_latent_input.to_any(),
            timesteps_input.to_any(),
            dt_input.to_any(),
            sigmas_input.to_any(),
            iteration_count_input.to_any(),
            guidance_scale_input.to_any(),
            te1_weights.to_any(),
            te2_weights.to_any(),
            unet_weights.to_any(),
            vae_weights.to_any(),
        ];
        let output_links: Vec<_> = vec![image_output.to_any()];
        let super_graph = builder.build(rng, &input_links, &output_links);

        let clip_prompt = PromptInput {
            tokenizer,
            link: cond_ids_input,
            seq_len: 77,
            encoding: PromptEncoding::ClipStyle {
                bos: 49406,
                eos: 49407,
                pad: 0,
            },
        };
        let clip_neg_prompt = PromptInput {
            tokenizer: clip_prompt.tokenizer.clone(),
            link: negative_cond_ids_input,
            seq_len: 77,
            encoding: clip_prompt.encoding.clone(),
        };

        Self {
            super_graph,
            positive_prompts: vec![clip_prompt],
            negative_prompts: Some(vec![clip_neg_prompt]),
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

    /// Pre-compute the Euler discrete scheduler parameters.
    /// Returns (timestep_values, dt_values, sigma_values, initial_sigma).
    pub fn compute_euler_schedule(
        num_inference_steps: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, f32) {
        let num_train_timesteps = 1000;
        let beta_start: f32 = 0.00085;
        let beta_end: f32 = 0.012;

        // Linear beta schedule (scaled_linear: sqrt of linear interpolation)
        let betas: Vec<f32> = (0..num_train_timesteps)
            .map(|i| {
                let t = i as f32 / (num_train_timesteps - 1) as f32;
                let b = beta_start.sqrt() + t * (beta_end.sqrt() - beta_start.sqrt());
                b * b
            })
            .collect();

        let mut alphas_cumprod = Vec::with_capacity(num_train_timesteps);
        let mut cumprod = 1.0f32;
        for &beta in &betas {
            cumprod *= 1.0 - beta;
            alphas_cumprod.push(cumprod);
        }

        // Evenly spaced timesteps (descending)
        let step_ratio = num_train_timesteps / num_inference_steps;
        let timesteps: Vec<usize> = (0..num_inference_steps)
            .rev()
            .map(|i| i * step_ratio + step_ratio - 1)
            .collect();

        // Compute sigmas from alphas_cumprod
        let sigmas: Vec<f32> = timesteps
            .iter()
            .map(|&t| ((1.0 - alphas_cumprod[t]) / alphas_cumprod[t]).sqrt())
            .collect();

        let init_sigma = sigmas[0];

        // Timestep values as f32
        let timestep_values: Vec<f32> = timesteps.iter().map(|&t| t as f32).collect();

        // dt[i] = sigma[i+1] - sigma[i], with sigma[num_steps] = 0
        let dt_values: Vec<f32> = (0..num_inference_steps)
            .map(|i| {
                let sigma_next = if i + 1 < num_inference_steps {
                    sigmas[i + 1]
                } else {
                    0.0
                };
                sigma_next - sigmas[i]
            })
            .collect();

        (timestep_values, dt_values, sigmas, init_sigma)
    }

    /// Pre-compute the rectified flow scheduler parameters for Flux Schnell.
    ///
    /// Flux uses flow matching: timesteps go from 1.0 (noise) to 0.0 (clean).
    /// Returns (timestep_values, dt_values, sigma_values).
    /// For Flux, sigma values equal timestep values (used for interface compatibility).
    pub fn compute_flux_schedule(num_inference_steps: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        // Linearly spaced timesteps from 1.0 to near 0.0
        // sigmas[i] = 1.0 - i/(N), giving [1.0, 1-1/N, ..., 1/N]
        // Final sigma (after last step) is 0.0
        let sigmas: Vec<f32> = (0..num_inference_steps)
            .map(|i| 1.0 - i as f32 / num_inference_steps as f32)
            .collect();

        let timestep_values = sigmas.clone();

        // dt[i] = sigma[i+1] - sigma[i], with sigma[num_steps] = 0
        let dt_values: Vec<f32> = (0..num_inference_steps)
            .map(|i| {
                let sigma_next = if i + 1 < num_inference_steps {
                    sigmas[i + 1]
                } else {
                    0.0
                };
                sigma_next - sigmas[i]
            })
            .collect();

        (timestep_values, dt_values, sigmas)
    }

    /// Build interface for Flux (CLIP-L + T5-XXL + DiT + VAE, rectified flow).
    ///
    /// Model weights order: [clip_l, t5_xxl, dit, vae_decoder]
    ///
    /// When `has_guidance` is true (Flux Dev), the DiT expects a guidance input
    /// and `guidance_scale_input` is populated. When false (Schnell), no guidance.
    #[allow(clippy::too_many_arguments)]
    pub fn new_flux(
        rng: &mut impl Rng,
        clip_tokenizer: TokenizerInfo,
        t5_tokenizer: TokenizerInfo,
        model_dtype: DType,
        has_guidance: bool,
    ) -> Self {
        let mut builder = SuperGraphBuilder::new();

        // Create input links
        let cond_ids_input = builder.new_tensor_link(rng); // CLIP token IDs
        let t5_ids_input = builder.new_tensor_link(rng); // T5 token IDs
        let initial_latent_input = builder.new_tensor_link(rng);
        let timesteps_input = builder.new_tensor_link(rng);
        let dt_input = builder.new_tensor_link(rng);
        let sigmas_input = builder.new_tensor_link(rng);
        let iteration_count_input = builder.new_tensor_link(rng);
        let guidance_scale_link = if has_guidance {
            Some(builder.new_tensor_link(rng))
        } else {
            None
        };
        let clip_weights = builder.new_model_link(rng);
        let t5_weights = builder.new_model_link(rng);
        let dit_weights = builder.new_model_link(rng);
        let vae_weights = builder.new_model_link(rng);

        // --- CLIP-L: input_ids + eos_indices → pooled_output [1, 768] ---
        let clip_pooled_f32 = builder.new_tensor_link(rng);
        let clip_eos = build_eos_indices_node(&mut builder, rng, cond_ids_input);
        builder.add_node(
            SuperGraphNodeModelExecution::new(
                rng,
                clip_weights,
                0,
                vec![
                    (cond_ids_input, "input_ids".to_string()),
                    (clip_eos, "eos_indices".to_string()),
                ],
                vec![("pooled_output".to_string(), clip_pooled_f32)],
            )
            .to_any(),
        );
        let clip_pooled = build_cast_node(&mut builder, rng, clip_pooled_f32, model_dtype);

        // --- T5-XXL: input_ids → hidden_states [1, seq, 4096] ---
        let t5_hidden_f32 = builder.new_tensor_link(rng);
        builder.add_node(
            SuperGraphNodeModelExecution::new(
                rng,
                t5_weights,
                1,
                vec![(t5_ids_input, "input_ids".to_string())],
                vec![("hidden_states".to_string(), t5_hidden_f32)],
            )
            .to_any(),
        );
        let t5_hidden = build_cast_node(&mut builder, rng, t5_hidden_f32, model_dtype);

        // --- Denoising loop (rectified flow) ---
        let final_latent = build_flux_denoising_loop(
            &mut builder,
            rng,
            dit_weights,
            clip_pooled,
            t5_hidden,
            initial_latent_input,
            timesteps_input,
            dt_input,
            sigmas_input,
            iteration_count_input,
            guidance_scale_link,
            model_dtype,
            2, // dit model index
        );

        // --- VAE decode ---
        // Flux VAE: latent / 0.3611 + 0.1159
        let image_output = build_flux_vae_decode(&mut builder, rng, final_latent, vae_weights, 3);

        // Build outer graph
        let model_weights = vec![clip_weights, t5_weights, dit_weights, vae_weights];
        let mut input_links: Vec<_> = vec![
            cond_ids_input.to_any(),
            t5_ids_input.to_any(),
            initial_latent_input.to_any(),
            timesteps_input.to_any(),
            dt_input.to_any(),
            sigmas_input.to_any(),
            iteration_count_input.to_any(),
        ];
        if let Some(gl) = guidance_scale_link {
            input_links.push(gl.to_any());
        }
        input_links.extend([
            clip_weights.to_any(),
            t5_weights.to_any(),
            dit_weights.to_any(),
            vae_weights.to_any(),
        ]);
        let output_links: Vec<_> = vec![image_output.to_any()];
        let super_graph = builder.build(rng, &input_links, &output_links);

        let clip_prompt = PromptInput {
            tokenizer: clip_tokenizer,
            link: cond_ids_input,
            seq_len: 77,
            encoding: PromptEncoding::ClipStyle {
                bos: 49406,
                eos: 49407,
                pad: 0,
            },
        };
        let t5_prompt = PromptInput {
            tokenizer: t5_tokenizer,
            link: t5_ids_input,
            seq_len: 256,
            encoding: PromptEncoding::RawPad { pad: 0 },
        };

        Self {
            super_graph,
            positive_prompts: vec![clip_prompt, t5_prompt],
            negative_prompts: None,
            initial_latent_input,
            timesteps_input,
            dt_input,
            sigmas_input,
            iteration_count_input,
            guidance_scale_input: guidance_scale_link,
            model_weights,
            image_output,
            scheduler: SchedulerType::RectifiedFlow,
            latent_channels: 16,
        }
    }

    /// Run the full image generation pipeline.
    ///
    /// `prompt_tokens` contains pre-tokenized tensors keyed by SuperGraphLinkTensor,
    /// one entry per prompt input slot (positive and negative).
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        models: &[&Model],
        prompt_tokens: HashMap<SuperGraphLinkTensor, NumericTensor<DynRank>>,
        initial_noise: Vec<f32>,
        latent_shape: Vec<usize>,
        num_inference_steps: usize,
        guidance_scale: f32,
        backend: &mut EvalBackend,
    ) -> Result<NumericTensor<DynRank>, SuperGraphError> {
        assert_eq!(
            models.len(),
            self.model_weights.len(),
            "Expected {} models, got {}",
            self.model_weights.len(),
            models.len()
        );

        // Compute schedule and prepare latent based on scheduler type
        let (timestep_values, dt_values, sigma_values, latent_tensor) = match &self.scheduler {
            SchedulerType::EulerDiscrete => {
                let (ts, dt, sigmas, init_sigma) =
                    Self::compute_euler_schedule(num_inference_steps);
                let scaled: Vec<f32> = initial_noise.iter().map(|&x| x * init_sigma).collect();
                let lat = NumericTensor::<DynRank>::from_vec_shape(scaled, latent_shape).unwrap();
                (ts, dt, sigmas, lat)
            }
            SchedulerType::RectifiedFlow => {
                let (ts, dt, sigmas) = Self::compute_flux_schedule(num_inference_steps);
                let lat =
                    NumericTensor::<DynRank>::from_vec_shape(initial_noise, latent_shape).unwrap();
                (ts, dt, sigmas, lat)
            }
        };

        let timesteps_tensor =
            NumericTensor::<DynRank>::from_vec_shape(timestep_values, vec![num_inference_steps])
                .unwrap();
        let dt_tensor =
            NumericTensor::<DynRank>::from_vec_shape(dt_values, vec![num_inference_steps]).unwrap();
        let sigmas_tensor =
            NumericTensor::<DynRank>::from_vec_shape(sigma_values, vec![num_inference_steps])
                .unwrap();
        let iter_count =
            NumericTensor::<DynRank>::from_vec_shape(vec![num_inference_steps as i64], vec![1])
                .unwrap();

        // Pack data
        let mut data = SuperGraphData::new();
        for (link, tensor) in prompt_tokens {
            data.tensors.insert(link, tensor);
        }
        data.tensors
            .insert(self.initial_latent_input, latent_tensor);
        data.tensors.insert(self.timesteps_input, timesteps_tensor);
        data.tensors.insert(self.dt_input, dt_tensor);
        data.tensors.insert(self.sigmas_input, sigmas_tensor);
        data.tensors.insert(self.iteration_count_input, iter_count);
        if let Some(gs_link) = self.guidance_scale_input {
            let guidance =
                NumericTensor::<DynRank>::from_vec_shape(vec![guidance_scale], vec![]).unwrap();
            data.tensors.insert(gs_link, guidance);
        }
        for (weight_link, model) in self.model_weights.iter().zip(models.iter()) {
            data.tensor_maps
                .insert(*weight_link, model.get_tensor_store());
        }

        // Run
        let mut observer = ();
        let mut tensor_cache = SuperGraphTensorCache::new();
        let symbolic_graphs: Vec<_> = models.iter().map(|m| m.get_symbolic_graph()).collect();
        let mut context = SuperGraphContext {
            observer: &mut observer,
            eval_backend: backend,
            super_graph_tensor_cache: &mut tensor_cache,
            caches: None,
            symbolic_graphs,
            use_compiled_models: false,
            compiled_models: None,
        };

        let result = self.super_graph.run(data, &mut context)?;
        let image = result.tensors.get(&self.image_output).unwrap().clone();
        Ok(image)
    }

    pub fn to_any(self) -> AnyInterface {
        AnyInterface::ImageGenerationInterface(self)
    }
}

// ============================================================================
// Text-to-Speech Interface
// ============================================================================

/// Interface for text-to-speech models (e.g. Kokoro).
///
/// The model takes phoneme token IDs, a style embedding, and a speed scalar,
/// and produces an audio waveform.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TextToSpeechInterface {
    pub super_graph: SuperGraph,
    /// Phoneme token IDs input (shape depends on model; Kokoro uses [1, seq_len]).
    pub input_ids_link: SuperGraphLinkTensor,
    /// Style/voice embedding input (e.g. [1, 1, 256] for Kokoro).
    pub style_link: SuperGraphLinkTensor,
    /// Speed control scalar input (e.g. [1]).
    pub speed_link: SuperGraphLinkTensor,
    /// Model weights.
    pub model_weights_link: SuperGraphLinkTensorMap,
    /// Output audio waveform (e.g. [1, audio_length]).
    pub audio_output_link: SuperGraphLinkTensor,
    /// Sample rate of the output audio in Hz.
    pub sample_rate: u32,
    /// Tokenizer for phoneme encoding.
    pub tokenizer: TokenizerInfo,
}

impl TextToSpeechInterface {
    pub fn to_any(self) -> AnyInterface {
        AnyInterface::TextToSpeechInterface(self)
    }
}

/// Interface for Piper VITS TTS models.
///
/// Piper models take phoneme IDs, input lengths, and scale parameters,
/// with an optional speaker ID for multi-speaker models.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PiperInterface {
    pub super_graph: SuperGraph,
    /// Phoneme ID sequence input [1, phoneme_count] i64.
    pub input_link: SuperGraphLinkTensor,
    /// Input sequence length [1] i64.
    pub input_lengths_link: SuperGraphLinkTensor,
    /// Scale parameters [3] f32: [noise_scale, length_scale, noise_scale_w].
    pub scales_link: SuperGraphLinkTensor,
    /// Speaker ID [1] i64 (only used for multi-speaker models).
    pub speaker_id_link: Option<SuperGraphLinkTensor>,
    /// Model weights.
    pub model_weights_link: SuperGraphLinkTensorMap,
    /// Output audio waveform [1, 1, time] f32.
    pub audio_output_link: SuperGraphLinkTensor,
    /// Sample rate of the output audio in Hz (from config).
    pub sample_rate: u32,
    /// Number of speakers (1 = single-speaker).
    pub num_speakers: u32,
    /// Phoneme ID map: IPA character → list of token IDs (from config).
    /// Stored as JSON string for serialization.
    pub phoneme_id_map_json: String,
    /// eSpeak voice code (e.g. "en-us") for phonemization.
    pub espeak_voice: String,
}

impl PiperInterface {
    pub fn to_any(self) -> AnyInterface {
        AnyInterface::PiperInterface(self)
    }
}
