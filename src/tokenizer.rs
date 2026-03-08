use crate::metadata::TokenizerInfo;
use std::str::Utf8Error;

#[derive(Debug, thiserror::Error)]
pub enum TokenizerError {
    #[cfg(feature = "tokenizers")]
    #[error(transparent)]
    TokenizersError(#[from] tokenizers::Error),
    #[error(transparent)]
    Utf8Error(#[from] Utf8Error),
}

#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
#[non_exhaustive]
pub enum AnyTokenizer {
    #[cfg(feature = "tokenizers")]
    Tokenizers(tokenizers::Tokenizer),
    #[cfg(feature = "rwkv-tokenizer")]
    Rwkv(rwkv_tokenizer::WorldTokenizer),
}

impl AnyTokenizer {
    pub fn from_tokenizer_info(info: &TokenizerInfo) -> AnyTokenizer {
        match &info {
            #[allow(unused_variables)]
            TokenizerInfo::HFTokenizer(name) => {
                #[cfg(all(feature = "tokenizers", feature = "http"))]
                {
                    AnyTokenizer::Tokenizers(
                        tokenizers::Tokenizer::from_pretrained(name, None).unwrap(),
                    )
                }
                #[cfg(not(all(feature = "tokenizers", feature = "http")))]
                {
                    panic!("Huggingface tokenizer not supported")
                }
            }
            TokenizerInfo::RWKVWorld => {
                #[cfg(feature = "rwkv-tokenizer")]
                {
                    AnyTokenizer::Rwkv(rwkv_tokenizer::WorldTokenizer::new(None).unwrap())
                }
                #[cfg(not(feature = "rwkv-tokenizer"))]
                {
                    panic!("RWKV tokenizer not supported")
                }
            }
        }
    }
}

impl Tokenizer for AnyTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        match self {
            #[cfg(feature = "tokenizers")]
            AnyTokenizer::Tokenizers(x) => <_ as Tokenizer>::encode(x, text),
            #[cfg(feature = "rwkv-tokenizer")]
            AnyTokenizer::Rwkv(x) => <_ as Tokenizer>::encode(x, text),
            #[cfg(not(all(feature = "tokenizers", feature = "rwkv-tokenizer")))]
            _ => unreachable!(),
        }
    }
    fn decode(&self, tokens: &[u32]) -> Result<String, TokenizerError> {
        match self {
            #[cfg(feature = "tokenizers")]
            AnyTokenizer::Tokenizers(x) => <_ as Tokenizer>::decode(x, tokens),
            #[cfg(feature = "rwkv-tokenizer")]
            AnyTokenizer::Rwkv(x) => <_ as Tokenizer>::decode(x, tokens),
            #[cfg(not(all(feature = "tokenizers", feature = "rwkv-tokenizer")))]
            _ => unreachable!(),
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
        tokenizers::TokenizerImpl::encode(self, text, false)
            .unwrap()
            .get_ids()
            .to_vec()
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
