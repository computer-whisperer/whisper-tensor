use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum TokenizerInfo {
    HFTokenizer(String),
    HFTokenizerLocal(String),
    /// In-memory tokenizer.json content (e.g. synthesized from GGUF metadata).
    HFTokenizerJson(String),
    RWKVWorld,
}
