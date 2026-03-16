use whisper_tensor::dtype::DType;
use whisper_tensor::loader::LoaderError;
use whisper_tensor::symbolic_graph::SymbolicGraph;

pub(super) struct ClipEncoderIo {
    pub(super) input: String,
    pub(super) eos_input: Option<String>,
    pub(super) hidden_output: String,
    pub(super) pooled_output: String,
}

pub(super) struct T5EncoderIo {
    pub(super) input: String,
    pub(super) hidden_output: String,
}

pub(super) struct TransformerIo {
    pub(super) latent_input: String,
    pub(super) timestep_input: String,
    pub(super) context_input: String,
    pub(super) pooled_input: String,
    pub(super) output: String,
}

pub(super) struct VaeDecoderIo {
    pub(super) input: String,
    pub(super) output: String,
}

pub(super) fn infer_clip_io(
    graph: &SymbolicGraph,
    model_name: &str,
) -> Result<ClipEncoderIo, LoaderError> {
    let (inputs, outputs) = list_io_names(graph);
    let input = pick_name(&inputs, &["input_ids"], "input_ids", model_name)?;
    let eos_input = inputs
        .iter()
        .find(|name| name.as_str() == "eos_indices" || name.as_str() == "eos_index")
        .or_else(|| {
            inputs
                .iter()
                .find(|name| name.to_ascii_lowercase().contains("eos"))
        })
        .and_then(|name| {
            let tid = graph.get_tensors_by_name().get(name).copied()?;
            let info = graph.get_tensor_info(tid)?;
            let shape = info.shape.as_ref()?;
            if shape.len() == 1 {
                Some(name.clone())
            } else {
                None
            }
        });
    let hidden_output = pick_name_with_rank(
        graph,
        &outputs,
        &["last_hidden_state", "hidden_states", "prompt_embeds"],
        Some(3),
        "hidden output",
        model_name,
    )?;
    let pooled_output = pick_name_with_rank(
        graph,
        &outputs,
        &[
            "text_embeds",
            "pooled_output",
            "pooler_output",
            "pooled_text_embeds",
        ],
        Some(2),
        "pooled output",
        model_name,
    )?;
    Ok(ClipEncoderIo {
        input,
        eos_input,
        hidden_output,
        pooled_output,
    })
}

pub(super) fn infer_t5_io(
    graph: &SymbolicGraph,
    model_name: &str,
) -> Result<T5EncoderIo, LoaderError> {
    let (inputs, outputs) = list_io_names(graph);
    let input = pick_name(&inputs, &["input_ids"], "input_ids", model_name)?;
    let hidden_output = pick_name_with_rank(
        graph,
        &outputs,
        &[
            "last_hidden_state",
            "hidden_states",
            "encoder_hidden_states",
        ],
        Some(3),
        "hidden output",
        model_name,
    )?;
    Ok(T5EncoderIo {
        input,
        hidden_output,
    })
}

pub(super) fn infer_transformer_io(
    graph: &SymbolicGraph,
    model_name: &str,
) -> Result<TransformerIo, LoaderError> {
    let (inputs, outputs) = list_io_names(graph);
    let latent_input = pick_name(
        &inputs,
        &["hidden_states", "latent_sample", "sample"],
        "latent input",
        model_name,
    )?;
    let timestep_input = pick_name(
        &inputs,
        &["timestep", "timesteps", "t"],
        "timestep input",
        model_name,
    )?;
    let context_input = pick_name(
        &inputs,
        &["encoder_hidden_states", "prompt_embeds", "context"],
        "encoder_hidden_states input",
        model_name,
    )?;
    let pooled_input = pick_name(
        &inputs,
        &["pooled_projections", "pooled_prompt_embeds", "text_embeds"],
        "pooled projections input",
        model_name,
    )?;
    let output = pick_name_with_rank(
        graph,
        &outputs,
        &["sample", "out_sample", "model_output"],
        Some(4),
        "model output",
        model_name,
    )?;
    Ok(TransformerIo {
        latent_input,
        timestep_input,
        context_input,
        pooled_input,
        output,
    })
}

pub(super) fn infer_vae_io(
    graph: &SymbolicGraph,
    model_name: &str,
) -> Result<VaeDecoderIo, LoaderError> {
    let (inputs, outputs) = list_io_names(graph);
    let input = pick_name(
        &inputs,
        &["latent_sample", "latent", "sample"],
        "latent input",
        model_name,
    )?;
    let output = pick_name_with_rank(
        graph,
        &outputs,
        &["sample", "image", "decoded"],
        Some(4),
        "image output",
        model_name,
    )?;
    Ok(VaeDecoderIo { input, output })
}

pub(super) fn infer_model_dtype(graph: &SymbolicGraph, io: &TransformerIo) -> Option<DType> {
    let tid = graph.get_tensors_by_name().get(&io.latent_input).copied()?;
    graph.get_tensor_info(tid)?.dtype
}

pub(super) fn infer_t5_seq_len(graph: &SymbolicGraph, io: &T5EncoderIo) -> Option<usize> {
    let tid = graph.get_tensors_by_name().get(&io.input).copied()?;
    let info = graph.get_tensor_info(tid)?;
    let shape = info.shape.as_ref()?;
    shape
        .last()
        .and_then(|dim| dim.as_numeric().copied())
        .map(|v| v as usize)
}

pub(super) fn infer_latent_channels(graph: &SymbolicGraph, io: &TransformerIo) -> Option<usize> {
    let tid = graph.get_tensors_by_name().get(&io.latent_input).copied()?;
    let info = graph.get_tensor_info(tid)?;
    let shape = info.shape.as_ref()?;
    if shape.len() < 2 {
        return None;
    }
    shape[1].as_numeric().copied().map(|v| v as usize)
}

fn list_io_names(graph: &SymbolicGraph) -> (Vec<String>, Vec<String>) {
    let inputs = graph
        .get_inputs()
        .into_iter()
        .filter_map(|id| graph.get_tensor_name(id).map(str::to_string))
        .collect::<Vec<_>>();
    let outputs = graph
        .get_outputs()
        .into_iter()
        .filter_map(|id| graph.get_tensor_name(id).map(str::to_string))
        .collect::<Vec<_>>();
    (inputs, outputs)
}

fn pick_name(
    available: &[String],
    candidates: &[&str],
    what: &str,
    model_name: &str,
) -> Result<String, LoaderError> {
    for candidate in candidates {
        if let Some(found) = available.iter().find(|x| x.as_str() == *candidate) {
            return Ok(found.clone());
        }
    }
    for candidate in candidates {
        let lower_candidate = candidate.to_ascii_lowercase();
        if let Some(found) = available
            .iter()
            .find(|x| x.to_ascii_lowercase().contains(&lower_candidate))
        {
            return Ok(found.clone());
        }
    }
    Err(LoaderError::LoadFailed(anyhow::anyhow!(
        "Could not infer {what} for {model_name}. Available names: {:?}",
        available
    )))
}

fn pick_name_with_rank(
    graph: &SymbolicGraph,
    available: &[String],
    candidates: &[&str],
    rank: Option<usize>,
    what: &str,
    model_name: &str,
) -> Result<String, LoaderError> {
    for candidate in candidates {
        if let Some(found) = available.iter().find(|x| x.as_str() == *candidate)
            && rank_matches(graph, found, rank)
        {
            return Ok(found.clone());
        }
    }
    for candidate in candidates {
        let lower_candidate = candidate.to_ascii_lowercase();
        if let Some(found) = available.iter().find(|x| {
            x.to_ascii_lowercase().contains(&lower_candidate) && rank_matches(graph, x, rank)
        }) {
            return Ok(found.clone());
        }
    }
    if let Some(expected_rank) = rank
        && let Some(found) = available
            .iter()
            .find(|x| rank_matches(graph, x, Some(expected_rank)))
    {
        return Ok(found.clone());
    }
    Err(LoaderError::LoadFailed(anyhow::anyhow!(
        "Could not infer {what} for {model_name}. Available names: {:?}",
        available
    )))
}

fn rank_matches(graph: &SymbolicGraph, name: &str, rank: Option<usize>) -> bool {
    let Some(expected_rank) = rank else {
        return true;
    };
    let Some(tid) = graph.get_tensors_by_name().get(name).copied() else {
        return false;
    };
    let Some(info) = graph.get_tensor_info(tid) else {
        return false;
    };
    let Some(shape) = &info.shape else {
        return false;
    };
    shape.len() == expected_rank
}
