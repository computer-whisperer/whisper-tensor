use super::onnx_bytes_to_model;
use whisper_tensor::interfaces::PiperInterface;
use whisper_tensor::loader::*;
use whisper_tensor::super_graph::SuperGraphBuilder;
use whisper_tensor::super_graph::links::SuperGraphLink;
use whisper_tensor::super_graph::nodes::{SuperGraphNode, SuperGraphNodeModelExecution};

/// Loader for Piper VITS TTS models (ONNX format).
///
/// Expects either:
/// - A single `.onnx` file (with `.onnx.json` config alongside it), or
/// - A directory containing the `.onnx` and `.onnx.json` files.
pub struct PiperLoader;

impl Loader for PiperLoader {
    fn name(&self) -> &str {
        "Piper TTS"
    }

    fn description(&self) -> &str {
        "Load a Piper VITS text-to-speech model"
    }

    fn config_schema(&self) -> Vec<ConfigField> {
        vec![ConfigField {
            key: "path".to_string(),
            label: "Model Path".to_string(),
            description: "Path to the Piper .onnx file or directory containing it".to_string(),
            field_type: ConfigFieldType::FilePath,
            required: true,
            default: None,
        }]
    }

    fn load(&self, config: ConfigValues) -> Result<LoaderOutput, LoaderError> {
        let path = require_path(&config, "path")?;

        // Find the ONNX file and its config
        let (onnx_path, config_path) = if path.is_dir() {
            let onnx = find_onnx_in_dir(&path)?;
            let cfg = find_piper_config(&onnx);
            (onnx, cfg)
        } else {
            let cfg = find_piper_config(&path);
            (path, cfg)
        };

        let config_path = config_path.ok_or_else(|| {
            LoaderError::LoadFailed(anyhow::anyhow!(
                "Piper config (.onnx.json) not found for {}",
                onnx_path.display()
            ))
        })?;

        // Parse the Piper config
        let config_json = std::fs::read_to_string(&config_path)
            .map_err(|e| LoaderError::LoadFailed(e.into()))?;
        let config_value: serde_json::Value = serde_json::from_str(&config_json)
            .map_err(|e| LoaderError::LoadFailed(e.into()))?;

        let sample_rate = config_value["audio"]["sample_rate"]
            .as_u64()
            .unwrap_or(22050) as u32;
        let num_speakers = config_value["num_speakers"].as_u64().unwrap_or(1) as u32;
        let espeak_voice = config_value["espeak"]["voice"]
            .as_str()
            .unwrap_or("en-us")
            .to_string();

        // Extract phoneme_id_map as JSON string
        let phoneme_id_map_json = config_value["phoneme_id_map"].to_string();

        // Load the ONNX model
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
            name: "piper-Piper".to_string(),
            interface: interface.to_any(),
        });

        Ok(output)
    }
}

/// Detect whether a path is a Piper model (single .onnx file with .onnx.json).
pub fn is_piper_model(path: &std::path::Path) -> bool {
    if path.is_file() {
        path.extension().is_some_and(|e| e == "onnx") && find_piper_config(path).is_some()
    } else if path.is_dir() {
        find_onnx_in_dir(path).ok().is_some_and(|onnx| find_piper_config(&onnx).is_some())
    } else {
        false
    }
}

fn find_onnx_in_dir(dir: &std::path::Path) -> Result<std::path::PathBuf, LoaderError> {
    // Look for .onnx files directly in the directory
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
            // Multiple ONNX files — pick the first one alphabetically
            let mut paths: Vec<_> = entries.into_iter().map(|e| e.path()).collect();
            paths.sort();
            Ok(paths[0].clone())
        }
    }
}

fn find_piper_config(onnx_path: &std::path::Path) -> Option<std::path::PathBuf> {
    // Piper config is <name>.onnx.json alongside the .onnx file
    let config_path = onnx_path.with_extension("onnx.json");
    if config_path.exists() {
        return Some(config_path);
    }
    // Also try just appending .json
    let alt = std::path::PathBuf::from(format!("{}.json", onnx_path.display()));
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
) -> PiperInterface {
    let mut builder = SuperGraphBuilder::new();

    let input_link = builder.new_tensor_link(rng);
    let input_lengths_link = builder.new_tensor_link(rng);
    let scales_link = builder.new_tensor_link(rng);
    let model_weights_link = builder.new_model_link(rng);
    let audio_output_link = builder.new_tensor_link(rng);

    let mut tensor_inputs = vec![
        (input_link, "input".to_string()),
        (input_lengths_link, "input_lengths".to_string()),
        (scales_link, "scales".to_string()),
    ];

    let speaker_id_link = if num_speakers > 1 {
        let sid_link = builder.new_tensor_link(rng);
        tensor_inputs.push((sid_link, "sid".to_string()));
        Some(sid_link)
    } else {
        None
    };

    let tensor_outputs = vec![("output".to_string(), audio_output_link)];

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

    let mut sg_inputs = vec![
        input_link.to_any(),
        input_lengths_link.to_any(),
        scales_link.to_any(),
        model_weights_link.to_any(),
    ];
    if let Some(sid) = speaker_id_link {
        sg_inputs.push(sid.to_any());
    }
    let sg_outputs = vec![audio_output_link.to_any()];
    let super_graph = builder.build(rng, &sg_inputs, &sg_outputs);

    PiperInterface {
        super_graph,
        input_link,
        input_lengths_link,
        scales_link,
        speaker_id_link,
        model_weights_link,
        audio_output_link,
        sample_rate,
        num_speakers,
        phoneme_id_map_json,
        espeak_voice,
    }
}
