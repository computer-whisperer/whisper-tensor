use crate::DynRank;
use crate::model::Model;
use crate::numeric_tensor::NumericTensor;
use crate::super_graph::SuperGraphError;
use crate::super_graph::links::{
    SuperGraphAnyLink, SuperGraphLinkDouble, SuperGraphLinkHash, SuperGraphLinkModel,
    SuperGraphLinkString, SuperGraphLinkTensor, SuperGraphLinkTokenizer,
};
use crate::tokenizer::AnyTokenizer;
use std::collections::HashMap;

#[derive(Clone, Default)]
pub struct SuperGraphData<'models> {
    pub tensors: HashMap<SuperGraphLinkTensor, NumericTensor<DynRank>>,
    pub strings: HashMap<SuperGraphLinkString, String>,
    pub tokenizers: HashMap<SuperGraphLinkTokenizer, AnyTokenizer>,
    pub models: HashMap<SuperGraphLinkModel, &'models Model>,
    pub hashes: HashMap<SuperGraphLinkHash, u64>,
}

impl<'models> SuperGraphData<'models> {
    pub fn new() -> Self {
        Self {
            tensors: HashMap::new(),
            strings: HashMap::new(),
            tokenizers: HashMap::new(),
            models: HashMap::new(),
            hashes: HashMap::new(),
        }
    }

    pub fn select(&self, links: &[SuperGraphAnyLink]) -> Result<Self, SuperGraphError> {
        let mut tensors = HashMap::new();
        let mut strings = HashMap::new();
        let mut tokenizers = HashMap::new();
        let mut models = HashMap::new();
        let mut hashes = HashMap::new();
        for link in links {
            match link {
                SuperGraphAnyLink::Tensor(t) => {
                    tensors.insert(
                        t.clone(),
                        self.tensors
                            .get(t)
                            .ok_or(SuperGraphError::MissingLinkError())?
                            .clone(),
                    );
                }
                SuperGraphAnyLink::String(s) => {
                    strings.insert(
                        s.clone(),
                        self.strings
                            .get(s)
                            .ok_or(SuperGraphError::MissingLinkError())?
                            .clone(),
                    );
                }
                SuperGraphAnyLink::Tokenizer(t) => {
                    tokenizers.insert(
                        t.clone(),
                        self.tokenizers
                            .get(t)
                            .ok_or(SuperGraphError::MissingLinkError())?
                            .clone(),
                    );
                }
                SuperGraphAnyLink::Model(m) => {
                    models.insert(
                        m.clone(),
                        *self
                            .models
                            .get(m)
                            .ok_or(SuperGraphError::MissingLinkError())?,
                    );
                }
                SuperGraphAnyLink::Hash(h) => {
                    hashes.insert(
                        h.clone(),
                        *self.hashes
                            .get(h)
                            .ok_or(SuperGraphError::MissingLinkError())?,
                    );
                }
            }
        }
        Ok(Self {
            tensors,
            strings,
            tokenizers,
            models,
            hashes,
        })
    }

    pub fn remap(&self, map: Vec<SuperGraphLinkDouble>) -> Result<Self, SuperGraphError> {
        let mut new_data = Self::new();

        for link in map {
            match link {
                SuperGraphLinkDouble::Tensor(input, output) => {
                    new_data.tensors.insert(
                        output,
                        self.tensors
                            .get(&input)
                            .ok_or(SuperGraphError::MissingLinkError())?
                            .clone(),
                    );
                }
                SuperGraphLinkDouble::String(input, output) => {
                    new_data.strings.insert(
                        output,
                        self.strings
                            .get(&input)
                            .ok_or(SuperGraphError::MissingLinkError())?
                            .clone(),
                    );
                }
                SuperGraphLinkDouble::Tokenizer(input, output) => {
                    new_data.tokenizers.insert(
                        output,
                        self.tokenizers
                            .get(&input)
                            .ok_or(SuperGraphError::MissingLinkError())?
                            .clone(),
                    );
                }
                SuperGraphLinkDouble::Model(input, output) => {
                    new_data.models.insert(
                        output,
                        self.models
                            .get(&input)
                            .ok_or(SuperGraphError::MissingLinkError())?,
                    );
                }
                SuperGraphLinkDouble::Hash(input, output) => {
                    new_data.hashes.insert(
                        output,
                        *self.hashes
                            .get(&input)
                            .ok_or(SuperGraphError::MissingLinkError())?,
                    );
                }
            }
        }

        Ok(new_data)
    }

    pub fn extend(&mut self, other: &Self) {
        self.tensors
            .extend(other.tensors.iter().map(|(a, b)| (a.clone(), b.clone())));
        self.strings
            .extend(other.strings.iter().map(|(a, b)| (a.clone(), b.clone())));
        self.tokenizers
            .extend(other.tokenizers.iter().map(|(a, b)| (a.clone(), b.clone())));
        self.models
            .extend(other.models.iter().map(|(a, b)| (a.clone(), *b)));
        self.hashes
            .extend(other.hashes.iter().map(|(a, b)| (a.clone(), *b)));
    }
}
