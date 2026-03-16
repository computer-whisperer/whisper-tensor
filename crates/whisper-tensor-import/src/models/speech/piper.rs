use super::onnx_bytes_to_model;
use std::path::{Path, PathBuf};
use whisper_tensor::interfaces::{TTSInputConfig, TextToSpeechInterface};
use whisper_tensor::loader::{LoadedInterface, LoaderError, LoaderOutput};
use whisper_tensor::super_graph::SuperGraphBuilder;
use whisper_tensor::super_graph::nodes::{
    SuperGraphNode, SuperGraphNodeModelExecution, SuperGraphNodePiperPhonemesToTensor,
    SuperGraphNodeTensorToAudioClip, SuperGraphNodeTextToPhonemes,
    SuperGraphNodeTextToPhonemesMode,
};

/// Load Piper VITS TTS from a single `.onnx` file or a directory containing it.
pub fn load_piper(path: &Path) -> Result<LoaderOutput, LoaderError> {
    let (onnx_path, config_path) = if path.is_dir() {
        let onnx = find_onnx_in_dir(path)?;
        let cfg = find_piper_config(&onnx);
        (onnx, cfg)
    } else {
        let cfg = find_piper_config(path);
        (path.to_path_buf(), cfg)
    };

    let config_path = config_path.ok_or_else(|| {
        LoaderError::LoadFailed(anyhow::anyhow!(
            "Piper config (.onnx.json) not found for {}",
            onnx_path.display()
        ))
    })?;

    let config_json =
        std::fs::read_to_string(&config_path).map_err(|e| LoaderError::LoadFailed(e.into()))?;
    let config_value: serde_json::Value =
        serde_json::from_str(&config_json).map_err(|e| LoaderError::LoadFailed(e.into()))?;

    let sample_rate = config_value["audio"]["sample_rate"]
        .as_u64()
        .unwrap_or(22050) as u32;
    let num_speakers = config_value["num_speakers"].as_u64().unwrap_or(1) as u32;
    let espeak_voice = config_value["espeak"]["voice"]
        .as_str()
        .unwrap_or("en-us")
        .to_string();

    let phoneme_id_map_json = config_value["phoneme_id_map"].to_string();

    let base_dir = onnx_path.parent();
    let onnx_data =
        crate::load_onnx_file(&onnx_path).map_err(|e| LoaderError::LoadFailed(e.into()))?;
    let (_, mut output) = onnx_bytes_to_model(&onnx_data, "piper", base_dir)?;

    let mut rng = rand::rng();
    let interface = build_piper_supergraph(
        sample_rate,
        num_speakers,
        espeak_voice,
        phoneme_id_map_json,
        &mut rng,
    );

    output.interfaces.push(LoadedInterface {
        name: "piper-TextToSpeech".to_string(),
        interface: interface.to_any(),
    });

    Ok(output)
}

/// Detect whether a path looks like a Piper model.
pub fn is_piper_model(path: &Path) -> bool {
    if path.is_file() {
        path.extension().is_some_and(|e| e == "onnx") && find_piper_config(path).is_some()
    } else if path.is_dir() {
        find_onnx_in_dir(path)
            .ok()
            .is_some_and(|onnx| find_piper_config(&onnx).is_some())
    } else {
        false
    }
}

fn find_onnx_in_dir(dir: &Path) -> Result<PathBuf, LoaderError> {
    let entries: Vec<_> = std::fs::read_dir(dir)
        .map_err(|e| LoaderError::LoadFailed(e.into()))?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "onnx"))
        .collect();

    match entries.len() {
        0 => Err(LoaderError::LoadFailed(anyhow::anyhow!(
            "No .onnx file found in {}",
            dir.display()
        ))),
        1 => Ok(entries[0].path()),
        _ => {
            let mut paths: Vec<_> = entries.into_iter().map(|e| e.path()).collect();
            paths.sort();
            Ok(paths[0].clone())
        }
    }
}

fn find_piper_config(onnx_path: &Path) -> Option<PathBuf> {
    let config_path = onnx_path.with_extension("onnx.json");
    if config_path.exists() {
        return Some(config_path);
    }
    let alt = PathBuf::from(format!("{}.json", onnx_path.display()));
    if alt.exists() {
        return Some(alt);
    }
    None
}

fn build_piper_supergraph(
    sample_rate: u32,
    num_speakers: u32,
    espeak_voice: String,
    phoneme_id_map_json: String,
    rng: &mut impl rand::Rng,
) -> TextToSpeechInterface {
    let mut builder = SuperGraphBuilder::new();

    let text_input_link = builder.new_string_link(rng);
    let scales_link = builder.new_tensor_link(rng);
    let model_weights_link = builder.new_model_link(rng);
    let audio_tensor_output_link = builder.new_tensor_link(rng);
    builder.set_link_label(text_input_link, "text");
    builder.set_link_label(scales_link, "scales");
    builder.set_link_label(model_weights_link, "model_weights");
    builder.set_link_label(audio_tensor_output_link, "raw_audio");

    let phonemes_link = SuperGraphNodeTextToPhonemes::new_and_add(
        &mut builder,
        text_input_link,
        SuperGraphNodeTextToPhonemesMode::Piper {
            voice: espeak_voice.clone(),
        },
        rng,
    );
    let (input_link, input_lengths_link) = SuperGraphNodePiperPhonemesToTensor::new_and_add(
        &mut builder,
        phonemes_link,
        phoneme_id_map_json,
        rng,
    );

    let mut tensor_inputs = vec![
        (input_link, "input".to_string()),
        (input_lengths_link, "input_lengths".to_string()),
        (scales_link, "scales".to_string()),
    ];

    let speaker_id_link = if num_speakers > 1 {
        let sid_link = builder.new_tensor_link(rng);
        builder.set_link_label(sid_link, "speaker_id");
        tensor_inputs.push((sid_link, "sid".to_string()));
        Some(sid_link)
    } else {
        None
    };

    let tensor_outputs = vec![("output".to_string(), audio_tensor_output_link)];

    builder.add_node(
        SuperGraphNodeModelExecution::new(
            rng,
            model_weights_link,
            0,
            tensor_inputs,
            tensor_outputs,
        )
        .to_any(),
    );

    let audio_output_link = SuperGraphNodeTensorToAudioClip::new_and_add(
        &mut builder,
        audio_tensor_output_link,
        sample_rate,
        rng,
    );
    builder.set_link_label(audio_output_link, "audio_output");

    let mut sg_inputs = vec![
        text_input_link.to_any(),
        scales_link.to_any(),
        model_weights_link.to_any(),
    ];
    if let Some(sid) = speaker_id_link {
        sg_inputs.push(sid.to_any());
    }
    let sg_outputs = vec![audio_output_link.to_any()];
    let super_graph = builder.build(rng, &sg_inputs, &sg_outputs);

    TextToSpeechInterface {
        super_graph,
        text_input_link,
        model_weights: vec![model_weights_link],
        audio_output_link,
        sample_rate,
        input_config: TTSInputConfig::Piper {
            scales_link,
            speaker_id_link,
            num_speakers,
        },
    }
}
