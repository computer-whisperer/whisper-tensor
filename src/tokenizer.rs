use std::str::Utf8Error;

#[derive(Debug, thiserror::Error)]
pub enum TokenizerError {
    #[cfg(feature = "tokenizers")]
    #[error(transparent)]
    TokenizersError(#[from] tokenizers::Error),
    #[error(transparent)]
    Utf8Error(#[from] Utf8Error),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum AnyTokenizer {
    #[cfg(feature = "tokenizers")]
    Tokenizers(tokenizers::Tokenizer),
    #[cfg(feature = "rwkv-tokenizer")]
    Rwkv(rwkv_tokenizer::WorldTokenizer),
}

impl Tokenizer for AnyTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        match self {
            #[cfg(feature = "tokenizers")]
            AnyTokenizer::Tokenizers(x) => <_ as Tokenizer>::encode(x, text),
            #[cfg(feature = "rwkv-tokenizer")]
            AnyTokenizer::Rwkv(x) => <_ as Tokenizer>::encode(x, text),
            //_ => unimplemented!()
        }
    }
    fn decode(&self, tokens: &[u32]) -> Result<String, TokenizerError> {
        match self {
            #[cfg(feature = "tokenizers")]
            AnyTokenizer::Tokenizers(x) => <_ as Tokenizer>::decode(x, tokens),
            #[cfg(feature = "rwkv-tokenizer")]
            AnyTokenizer::Rwkv(x) => <_ as Tokenizer>::decode(x, tokens),
            //_ => unimplemented!()
        }
    }
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