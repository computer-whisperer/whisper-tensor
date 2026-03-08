use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelInputType {
    TokenID(usize), // Tokenizer ID number
    PreviousInternal(usize),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputMetadata {
    pub model_input_type: ModelInputType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelOutputType {
    TokenID(usize), // Tokenizer ID number
    NextInternal(usize),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputMetadata {
    pub model_output_type: ModelOutputType,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum TokenizerInfo {
    HFTokenizer(String),
    RWKVWorld,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub tokenizer_infos: Vec<TokenizerInfo>,
    pub max_token_batch: Option<usize>,
}
