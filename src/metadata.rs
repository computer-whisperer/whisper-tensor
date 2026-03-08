use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum TokenizerInfo {
    HFTokenizer(String),
    RWKVWorld,
}
