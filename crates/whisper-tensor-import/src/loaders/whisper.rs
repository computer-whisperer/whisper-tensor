use std::sync::Arc;
use whisper_tensor::interfaces::SpeechToTextInterface;
use whisper_tensor::loader::*;
use whisper_tensor::metadata::TokenizerInfo;
use whisper_tensor::model::Model;

use crate::models::speech::whisper::WhisperConfig;
use crate::models::speech::whisper_stt::{
    build_whisper_supergraph, load_forced_decoder_ids, load_mel_filters,
};

/// Loader for Whisper speech-to-text models (HuggingFace safetensors format).
///
/// Expects a directory containing:
/// - `model.safetensors` (or sharded safetensors)
/// - `config.json`
/// - `tokenizer.json`
pub struct WhisperLoader;

impl Loader for WhisperLoader {
    fn name(&self) -> &str {
        "Whisper"
    }

    fn description(&self) -> &str {
        "Load a Whisper speech-to-text model (HuggingFace format)"
    }

    fn config_schema(&self) -> Vec<ConfigField> {
        vec![ConfigField {
            key: "path".to_string(),
            label: "Model Directory".to_string(),
            description: "Path to the Whisper model directory (containing model.safetensors, config.json, tokenizer.json)".to_string(),
            field_type: ConfigFieldType::FilePath,
            required: true,
            default: None,
        }]
    }

    fn load(&self, config: ConfigValues) -> Result<LoaderOutput, LoaderError> {
        let dir = require_path(&config, "path")?;

        // Parse config.json
        let config_path = dir.join("config.json");
        let config_json: serde_json::Value = {
            let data = std::fs::read_to_string(&config_path)
                .map_err(|e| LoaderError::LoadFailed(e.into()))?;
            serde_json::from_str(&data).map_err(|e| LoaderError::LoadFailed(e.into()))?
        };

        let whisper_config = WhisperConfig {
            d_model: config_json["d_model"].as_u64().unwrap_or(768) as usize,
            encoder_layers: config_json["encoder_layers"].as_u64().unwrap_or(12) as usize,
            encoder_attention_heads: config_json["encoder_attention_heads"]
                .as_u64()
                .unwrap_or(12) as usize,
            encoder_ffn_dim: config_json["encoder_ffn_dim"].as_u64().unwrap_or(3072) as usize,
            decoder_layers: config_json["decoder_layers"].as_u64().unwrap_or(12) as usize,
            decoder_attention_heads: config_json["decoder_attention_heads"]
                .as_u64()
                .unwrap_or(12) as usize,
            decoder_ffn_dim: config_json["decoder_ffn_dim"].as_u64().unwrap_or(3072) as usize,
            num_mel_bins: config_json["num_mel_bins"].as_u64().unwrap_or(80) as usize,
            max_source_positions: config_json["max_source_positions"].as_u64().unwrap_or(1500)
                as usize,
            max_target_positions: config_json["max_target_positions"].as_u64().unwrap_or(448)
                as usize,
            vocab_size: config_json["vocab_size"].as_u64().unwrap_or(51864) as usize,
        };

        let decoder_start_token_id = config_json["decoder_start_token_id"]
            .as_u64()
            .unwrap_or(50257) as u32;
        let eos_token_id = config_json["eos_token_id"].as_u64().unwrap_or(50256) as u32;
        let decoder_prefix_token_ids = load_forced_decoder_ids(&dir, decoder_start_token_id);
        let max_decode_steps = whisper_config.max_target_positions.max(1) as u32;
        let mel_filters = load_mel_filters(&dir, whisper_config.num_mel_bins);

        // Load safetensors weights
        let safetensors_path = dir.join("model.safetensors");
        if !safetensors_path.exists() {
            return Err(LoaderError::LoadFailed(anyhow::anyhow!(
                "model.safetensors not found in {}",
                dir.display()
            )));
        }

        let origin_path =
            std::fs::canonicalize(&safetensors_path).unwrap_or_else(|_| safetensors_path.clone());

        use crate::onnx_graph::WeightStorageStrategy;
        use crate::onnx_graph::weights::SafetensorsWeightManager;
        use memmap2::Mmap;

        let file = std::fs::File::open(&safetensors_path)
            .map_err(|e| LoaderError::LoadFailed(e.into()))?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| LoaderError::LoadFailed(e.into()))?;
        let wm = SafetensorsWeightManager::new(vec![Arc::new(mmap)])
            .map_err(|e| LoaderError::LoadFailed(e.into()))?;

        let storage = WeightStorageStrategy::OriginReference;

        // Build encoder ONNX model
        eprintln!("Building Whisper encoder...");
        let encoder_onnx = crate::models::speech::whisper::build_encoder(
            &wm,
            &whisper_config,
            storage.clone(),
            &origin_path,
        )
        .map_err(LoaderError::LoadFailed)?;

        // Build decoder ONNX model
        eprintln!("Building Whisper decoder...");
        let decoder_onnx = crate::models::speech::whisper::build_decoder(
            &wm,
            &whisper_config,
            storage,
            &origin_path,
        )
        .map_err(LoaderError::LoadFailed)?;

        let base_dir = dir.as_path();

        let mut rng = rand::rng();
        let encoder_model = Model::new_from_onnx(&encoder_onnx, &mut rng, Some(base_dir))
            .map_err(|e| LoaderError::LoadFailed(e.into()))?;
        let encoder_model = Arc::new(encoder_model);

        let decoder_model = Model::new_from_onnx(&decoder_onnx, &mut rng, Some(base_dir))
            .map_err(|e| LoaderError::LoadFailed(e.into()))?;
        let decoder_model = Arc::new(decoder_model);

        // Build unified STT supergraph.
        let tokenizer_path = dir.join("tokenizer.json");
        let tokenizer = if tokenizer_path.exists() {
            TokenizerInfo::HFTokenizerLocal(tokenizer_path.to_string_lossy().to_string())
        } else {
            return Err(LoaderError::LoadFailed(anyhow::anyhow!(
                "tokenizer.json not found in {}",
                dir.display()
            )));
        };

        let whisper_sg = build_whisper_supergraph(
            &whisper_config,
            mel_filters,
            decoder_prefix_token_ids.as_slice(),
            eos_token_id,
            max_decode_steps,
            &mut rng,
        );

        let interface = SpeechToTextInterface {
            super_graph: whisper_sg.super_graph,
            audio_input_link: whisper_sg.audio_link,
            encoder_weights_link: whisper_sg.encoder_weights_link,
            decoder_weights_link: whisper_sg.decoder_weights_link,
            output_token_link: whisper_sg.output_token_link,
            tokenizer,
            sample_rate: 16000,
            num_mel_bins: whisper_config.num_mel_bins as u32,
            decoder_start_token_id,
            eos_token_id,
            decoder_prefix_token_ids,
            max_decode_steps,
        };

        let models = vec![
            LoadedModel {
                name: "whisper-encoder".to_string(),
                model: encoder_model,
            },
            LoadedModel {
                name: "whisper-decoder".to_string(),
                model: decoder_model,
            },
        ];

        Ok(LoaderOutput {
            models,
            interfaces: vec![LoadedInterface {
                name: "whisper-SpeechToText".to_string(),
                interface: interface.to_any(),
            }],
        })
    }
}
