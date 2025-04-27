pub trait Tokenizer { 
    fn encode(&self, text: &str) -> Vec<u32>;
    fn decode(&self, tokens: &[u32]) -> String;
}

#[cfg(feature = "tokenizers")]
impl Tokenizer for tokenizers::Tokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        tokenizers::TokenizerImpl::encode(self, text, false).unwrap().get_ids().to_vec()
    }
    fn decode(&self, tokens: &[u32]) -> String {
        tokenizers::TokenizerImpl::decode(self, tokens, false).unwrap()
    }
}