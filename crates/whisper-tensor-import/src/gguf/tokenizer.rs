use super::GgufFile;
use serde_json::{Value, json};

/// Pre-tokenizer type, determined from GGUF metadata.
enum PreTokenizerType {
    /// Standard GPT-2 pre-tokenizer (ByteLevel with use_regex=true).
    Gpt2,
    /// Llama-3 style: Split with regex + ByteLevel.
    Llama3,
    /// Qwen2 style: Split with regex + ByteLevel (single-digit number matching).
    Qwen2,
}

const LLAMA3_REGEX: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

const QWEN2_REGEX: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Synthesize a tokenizer.json string from GGUF embedded tokenizer metadata.
///
/// Supports:
/// - BPE tokenizers (`tokenizer.ggml.model = "gpt2"`) — Llama-3, Mistral, Qwen, etc.
/// - Unigram/SentencePiece tokenizers (`tokenizer.ggml.model = "llama"`) — Phi-3, Llama-2, etc.
///
/// Returns `None` if the required metadata is missing or the tokenizer model
/// type is unsupported.
pub fn synthesize_tokenizer_json(gguf: &GgufFile) -> Option<String> {
    let model_type = gguf.get_metadata("tokenizer.ggml.model")?.as_str()?;
    match model_type {
        "gpt2" => synthesize_bpe(gguf),
        "llama" => synthesize_unigram(gguf),
        _ => None,
    }
}

/// Extract common tokenizer arrays from GGUF.
struct TokenizerArrays<'a> {
    tokens: &'a [super::GgufMetadataValue],
    token_types: Vec<u32>,
}

impl<'a> TokenizerArrays<'a> {
    fn from_gguf(gguf: &'a GgufFile) -> Option<Self> {
        let tokens = match gguf.get_metadata("tokenizer.ggml.tokens")? {
            super::GgufMetadataValue::Array(arr) => arr.as_slice(),
            _ => return None,
        };

        let token_types: Vec<u32> = gguf
            .get_metadata("tokenizer.ggml.token_type")
            .and_then(|v| match v {
                super::GgufMetadataValue::Array(arr) => Some(
                    arr.iter()
                        .map(|x| x.as_u64_coerce().unwrap_or(1) as u32)
                        .collect(),
                ),
                _ => None,
            })
            .unwrap_or_default();

        Some(Self {
            tokens,
            token_types,
        })
    }

    /// Collect special (control) tokens as added_tokens entries.
    /// Token type 3 = CONTROL in GGUF's llm_token_type enum.
    fn added_tokens(&self) -> Vec<Value> {
        let mut added = Vec::new();
        for (id, token_val) in self.tokens.iter().enumerate() {
            let token_type = self.token_types.get(id).copied().unwrap_or(1);
            if token_type == 3
                && let Some(content) = token_val.as_str()
            {
                added.push(json!({
                    "id": id,
                    "content": content,
                    "single_word": false,
                    "lstrip": false,
                    "rstrip": false,
                    "normalized": false,
                    "special": true
                }));
            }
        }
        added
    }
}

/// Synthesize a BPE (GPT-2 style) tokenizer.json.
fn synthesize_bpe(gguf: &GgufFile) -> Option<String> {
    let arrays = TokenizerArrays::from_gguf(gguf)?;

    let merges_array = match gguf.get_metadata("tokenizer.ggml.merges")? {
        super::GgufMetadataValue::Array(arr) => arr,
        _ => return None,
    };

    // Determine pre-tokenizer variant from tokenizer.ggml.pre (if present)
    let pre_type = match gguf
        .get_metadata("tokenizer.ggml.pre")
        .and_then(|v| v.as_str())
    {
        Some("llama3" | "llama-bpe") => PreTokenizerType::Llama3,
        Some("qwen2") => PreTokenizerType::Qwen2,
        Some("gpt-2" | "gpt2") => PreTokenizerType::Gpt2,
        None => PreTokenizerType::Gpt2,
        Some(_) => PreTokenizerType::Gpt2,
    };

    // Build vocab: token_string → token_id
    let mut vocab = serde_json::Map::new();
    for (id, token_val) in arrays.tokens.iter().enumerate() {
        if let Some(token_str) = token_val.as_str() {
            vocab.insert(token_str.to_string(), json!(id));
        }
    }

    // Build merges list
    let merges: Vec<Value> = merges_array
        .iter()
        .filter_map(|v| v.as_str().map(|s| json!(s)))
        .collect();

    let added_tokens = arrays.added_tokens();

    let pre_tokenizer = match pre_type {
        PreTokenizerType::Gpt2 => json!({
            "type": "ByteLevel",
            "add_prefix_space": false,
            "trim_offsets": true,
            "use_regex": true
        }),
        PreTokenizerType::Llama3 | PreTokenizerType::Qwen2 => {
            let regex = match pre_type {
                PreTokenizerType::Llama3 => LLAMA3_REGEX,
                PreTokenizerType::Qwen2 => QWEN2_REGEX,
                _ => unreachable!(),
            };
            json!({
                "type": "Sequence",
                "pretokenizers": [
                    {
                        "type": "Split",
                        "pattern": { "Regex": regex },
                        "behavior": "Isolated",
                        "invert": false
                    },
                    {
                        "type": "ByteLevel",
                        "add_prefix_space": false,
                        "trim_offsets": true,
                        "use_regex": false
                    }
                ]
            })
        }
    };

    let tokenizer_json = json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added_tokens,
        "normalizer": null,
        "pre_tokenizer": pre_tokenizer,
        "post_processor": null,
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": true,
            "trim_offsets": true,
            "use_regex": true
        },
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": null,
            "continuing_subword_prefix": null,
            "end_of_word_suffix": null,
            "fuse_unk": false,
            "byte_fallback": false,
            "ignore_merges": false,
            "vocab": vocab,
            "merges": merges
        }
    });

    Some(serde_json::to_string(&tokenizer_json).unwrap())
}

/// Synthesize a Unigram (SentencePiece) tokenizer.json.
fn synthesize_unigram(gguf: &GgufFile) -> Option<String> {
    let arrays = TokenizerArrays::from_gguf(gguf)?;

    // Scores array (log-probabilities for Unigram model)
    let scores: Vec<f64> = gguf
        .get_metadata("tokenizer.ggml.scores")
        .and_then(|v| match v {
            super::GgufMetadataValue::Array(arr) => Some(
                arr.iter()
                    .map(|x| x.as_f32_coerce().unwrap_or(0.0) as f64)
                    .collect(),
            ),
            _ => None,
        })
        .unwrap_or_default();

    let unk_id = gguf
        .get_metadata("tokenizer.ggml.unknown_token_id")
        .and_then(|v| v.as_u64_coerce())
        .unwrap_or(0);

    // Build vocab: [[token_string, score], ...]
    let vocab: Vec<Value> = arrays
        .tokens
        .iter()
        .enumerate()
        .filter_map(|(id, token_val)| {
            let token_str = token_val.as_str()?;
            let score = scores.get(id).copied().unwrap_or(0.0);
            Some(json!([token_str, score]))
        })
        .collect();

    let added_tokens = arrays.added_tokens();

    // Check if the model has byte fallback tokens (<0x00> through <0xFF>)
    // Token type 6 = BYTE in GGUF's llm_token_type enum
    let has_byte_fallback = arrays.token_types.contains(&6);

    let tokenizer_json = json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added_tokens,
        "normalizer": null,
        "pre_tokenizer": {
            "type": "Metaspace",
            "replacement": "\u{2581}",
            "prepend_scheme": "always",
            "split": true
        },
        "post_processor": null,
        "decoder": {
            "type": "Sequence",
            "decoders": [
                {
                    "type": "Replace",
                    "pattern": { "String": "\u{2581}" },
                    "content": " "
                },
                {
                    "type": "ByteFallback"
                },
                {
                    "type": "Fuse"
                },
                {
                    "type": "Strip",
                    "content": " ",
                    "start": 1,
                    "stop": 0
                }
            ]
        },
        "model": {
            "type": "Unigram",
            "unk_id": unk_id,
            "byte_fallback": has_byte_fallback,
            "vocab": vocab
        }
    });

    Some(serde_json::to_string(&tokenizer_json).unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn synthesize_llama3_tokenizer() {
        let path = Path::new("/ceph/public/neural_models/llms/Llama-3-8B.Q4_0.gguf");
        if !path.exists() {
            eprintln!("Skipping: GGUF file not found");
            return;
        }
        let gguf = GgufFile::open(path).unwrap();
        let json = synthesize_tokenizer_json(&gguf).expect("should synthesize tokenizer");

        // Verify the JSON is valid and has the expected structure
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["version"], "1.0");
        assert_eq!(parsed["model"]["type"], "BPE");

        let vocab = parsed["model"]["vocab"].as_object().unwrap();
        assert_eq!(vocab.len(), 128256, "Llama-3 has 128256 tokens");

        let merges = parsed["model"]["merges"].as_array().unwrap();
        assert!(!merges.is_empty(), "should have merge rules");

        let added = parsed["added_tokens"].as_array().unwrap();
        assert!(!added.is_empty(), "should have special tokens");

        // Verify BOS/EOS are in the added tokens
        let bos_id = gguf
            .get_metadata("tokenizer.ggml.bos_token_id")
            .and_then(|v| v.as_u64_coerce())
            .unwrap();
        assert!(
            added.iter().any(|t| t["id"] == bos_id),
            "BOS token should be in added_tokens"
        );
    }

    #[test]
    fn synthesize_phi3_tokenizer() {
        let path =
            Path::new("/ceph/public/neural_models/llms/Phi-3-medium-128k-instruct-Q5_K_S.gguf");
        if !path.exists() {
            eprintln!("Skipping: GGUF file not found");
            return;
        }
        let gguf = GgufFile::open(path).unwrap();
        let json = synthesize_tokenizer_json(&gguf).expect("should synthesize tokenizer");

        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["version"], "1.0");
        assert_eq!(parsed["model"]["type"], "Unigram");

        let vocab = parsed["model"]["vocab"].as_array().unwrap();
        assert_eq!(vocab.len(), 32064, "Phi-3 has 32064 tokens");

        // Each vocab entry is [token_string, score]
        let first = vocab[0].as_array().unwrap();
        assert!(first[0].is_string());
        assert!(first[1].is_number());

        let added = parsed["added_tokens"].as_array().unwrap();
        assert!(!added.is_empty(), "should have special tokens");

        println!(
            "Phi-3 synthesized tokenizer: {} vocab, {} added tokens",
            vocab.len(),
            added.len()
        );
    }
}
