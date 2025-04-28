use std::str::Utf8Error;

#[derive(Debug, thiserror::Error)]
pub enum TokenizerError {
    #[cfg(feature = "tokenizers")]
    #[error(transparent)]
    TokenizersError(#[from] tokenizers::Error),
    #[error(transparent)]
    Utf8Error(#[from] Utf8Error),
}

pub trait Tokenizer {
    fn encode(&self, text: &str) -> Vec<u32>;
    fn decode(&self, tokens: &[u32]) -> Result<String, TokenizerError>;
}

#[cfg(feature = "tokenizers")]
impl Tokenizer for tokenizers::Tokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        tokenizers::TokenizerImpl::encode(self, text, false).unwrap().get_ids().to_vec()
    }
    fn decode(&self, tokens: &[u32]) -> Result<String, TokenizerError> {
        Ok(tokenizers::TokenizerImpl::decode(self, tokens, false)?)
    }
}

#[cfg(feature = "rwkv-tokenizer")]
impl Tokenizer for rwkv_tokenizer::WorldTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        self.encode(text).into_iter().map(|x| x as u32).collect()
    }
    fn decode(&self, tokens: &[u32]) -> Result<String, TokenizerError> {
        Ok(self.decode(tokens.iter().map(|x| *x as u16).collect())?)
    }
}