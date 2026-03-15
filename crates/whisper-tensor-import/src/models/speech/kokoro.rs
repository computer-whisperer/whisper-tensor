use super::onnx_bytes_to_model;
use std::path::Path;
use whisper_tensor::interfaces::{KokoroVoiceEmbedding, TTSInputConfig, TextToSpeechInterface};
use whisper_tensor::loader::{LoadedInterface, LoaderError, LoaderOutput};
use whisper_tensor::metadata::TokenizerInfo;
use whisper_tensor::super_graph::SuperGraphBuilder;
use whisper_tensor::super_graph::nodes::{
    SuperGraphNode, SuperGraphNodeKokoroPhonemesToTensor, SuperGraphNodeModelExecution,
    SuperGraphNodeTensorToAudioClip, SuperGraphNodeTextToPhonemes,
    SuperGraphNodeTextToPhonemesMode,
};

/// Load Kokoro TTS from a model directory and variant name.
pub fn load_kokoro(dir: &Path, variant: &str) -> Result<LoaderOutput, LoaderError> {
    let onnx_path = dir.join("onnx").join(format!("{variant}.onnx"));
    if !onnx_path.exists() {
        return Err(LoaderError::LoadFailed(anyhow::anyhow!(
            "ONNX model not found at {}",
            onnx_path.display()
        )));
    }
    let onnx_data =
        crate::load_onnx_file(&onnx_path).map_err(|e| LoaderError::LoadFailed(e.into()))?;

    let (_, mut output) = onnx_bytes_to_model(&onnx_data, "kokoro", Some(dir))?;

    let tokenizer_path = dir.join("tokenizer.json");
    let tokenizer = if tokenizer_path.exists() {
        TokenizerInfo::HFTokenizerLocal(tokenizer_path.to_string_lossy().to_string())
    } else {
        return Err(LoaderError::LoadFailed(anyhow::anyhow!(
            "tokenizer.json not found in {}",
            dir.display()
        )));
    };

    let mut rng = rand::rng();
    let voices = load_kokoro_voices(dir)?;
    let default_voice = pick_default_voice(&voices);
    let interface = build_kokoro_supergraph(tokenizer, voices, default_voice, &mut rng);

    output.interfaces.push(LoadedInterface {
        name: "kokoro-TextToSpeech".to_string(),
        interface: interface.to_any(),
    });

    Ok(output)
}

/// Build a SuperGraph for the Kokoro TTS model.
///
/// The graph converts raw text to Kokoro phoneme IDs, then runs model execution
/// with style and speed conditioning to produce the waveform output.
fn build_kokoro_supergraph(
    tokenizer: TokenizerInfo,
    voices: Vec<KokoroVoiceEmbedding>,
    default_voice: Option<String>,
    rng: &mut impl rand::Rng,
) -> TextToSpeechInterface {
    let mut builder = SuperGraphBuilder::new();

    // External input links
    let text_input_link = builder.new_string_link(rng);
    let style_link = builder.new_tensor_link(rng);
    let speed_link = builder.new_tensor_link(rng);
    let model_weights_link = builder.new_model_link(rng);
    let audio_tensor_output_link = builder.new_tensor_link(rng);
    builder.set_link_label(text_input_link, "text");
    builder.set_link_label(style_link, "style");
    builder.set_link_label(speed_link, "speed");
    builder.set_link_label(model_weights_link, "model_weights");
    builder.set_link_label(audio_tensor_output_link, "raw_audio");

    let phonemes_link = SuperGraphNodeTextToPhonemes::new_and_add(
        &mut builder,
        text_input_link,
        SuperGraphNodeTextToPhonemesMode::Kokoro {
            voice: "en-us".to_string(),
        },
        rng,
    );
    let input_ids_link = SuperGraphNodeKokoroPhonemesToTensor::new_and_add(
        &mut builder,
        phonemes_link,
        tokenizer,
        rng,
    );

    // Model execution: wire named inputs/outputs to the ONNX graph
    let tensor_inputs = vec![
        (input_ids_link, "input_ids".to_string()),
        (style_link, "style".to_string()),
        (speed_link, "speed".to_string()),
    ];
    let tensor_outputs = vec![("waveform".to_string(), audio_tensor_output_link)];

    builder.add_node(
        SuperGraphNodeModelExecution::new(
            rng,
            model_weights_link,
            0, // model index
            tensor_inputs,
            tensor_outputs,
        )
        .to_any(),
    );

    let audio_output_link = SuperGraphNodeTensorToAudioClip::new_and_add(
        &mut builder,
        audio_tensor_output_link,
        24000,
        rng,
    );
    builder.set_link_label(audio_output_link, "audio_output");

    // Build the SuperGraph
    let sg_inputs = vec![
        text_input_link.to_any(),
        style_link.to_any(),
        speed_link.to_any(),
        model_weights_link.to_any(),
    ];
    let sg_outputs = vec![audio_output_link.to_any()];
    let super_graph = builder.build(rng, &sg_inputs, &sg_outputs);

    TextToSpeechInterface {
        super_graph,
        text_input_link,
        model_weights: vec![model_weights_link],
        audio_output_link,
        sample_rate: 24000,
        input_config: TTSInputConfig::Kokoro {
            style_link,
            speed_link,
            voices,
            default_voice,
        },
    }
}

fn load_kokoro_voices(dir: &Path) -> Result<Vec<KokoroVoiceEmbedding>, LoaderError> {
    let voices_dir = dir.join("voices");
    if !voices_dir.is_dir() {
        return Err(LoaderError::LoadFailed(anyhow::anyhow!(
            "voices directory not found in {}",
            dir.display()
        )));
    }

    let mut voices = Vec::new();
    let entries = std::fs::read_dir(&voices_dir).map_err(|e| LoaderError::LoadFailed(e.into()))?;
    for entry in entries {
        let entry = entry.map_err(|e| LoaderError::LoadFailed(e.into()))?;
        let path = entry.path();
        if !path.is_file() || path.extension().is_none_or(|ext| ext != "bin") {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        let bytes = std::fs::read(&path).map_err(|e| {
            LoaderError::LoadFailed(anyhow::anyhow!(
                "failed to read voice embedding {}: {}",
                path.display(),
                e
            ))
        })?;
        voices.push(KokoroVoiceEmbedding {
            name: stem.to_string(),
            style_table_le_bytes: bytes,
        });
    }

    voices.sort_by(|a, b| a.name.cmp(&b.name));
    if voices.is_empty() {
        return Err(LoaderError::LoadFailed(anyhow::anyhow!(
            "no .bin voice embeddings found in {}",
            voices_dir.display()
        )));
    }
    Ok(voices)
}

fn pick_default_voice(voices: &[KokoroVoiceEmbedding]) -> Option<String> {
    if voices.iter().any(|v| v.name == "af") {
        return Some("af".to_string());
    }
    if voices.iter().any(|v| v.name == "af_heart") {
        return Some("af_heart".to_string());
    }
    voices.first().map(|v| v.name.clone())
}
