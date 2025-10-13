use rand::Rng;
use serde::{Deserialize, Serialize};
use crate::graph::GlobalId;

pub trait SuperGraphLink {
    fn id(&self) -> SuperGraphLinkId;
    fn to_any(&self) -> SuperGraphAnyLink;
    fn global_id(&self) -> GlobalId;
}

impl<T: SuperGraphLink> From<&T> for SuperGraphAnyLink {
    fn from(value: &T) -> Self {
        value.to_any()
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct SuperGraphLinkId(pub(crate) u32);

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct SuperGraphLinkString(SuperGraphLinkId, GlobalId);
impl SuperGraphLinkString {
    pub fn new(id: SuperGraphLinkId, rng: &mut impl Rng) -> Self {
        Self(id, GlobalId::new(rng))
    }
}
impl SuperGraphLink for SuperGraphLinkString {
    fn id(&self) -> SuperGraphLinkId {
        self.0
    }
    fn to_any(&self) -> SuperGraphAnyLink {
        SuperGraphAnyLink::String(*self)
    }
    fn global_id(&self) -> GlobalId {
        self.1
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct SuperGraphLinkTensor(SuperGraphLinkId, GlobalId);
impl SuperGraphLinkTensor {
    pub fn new(id: SuperGraphLinkId, rng: &mut impl Rng) -> Self {
        Self(id, GlobalId::new(rng))
    }
}
impl SuperGraphLink for SuperGraphLinkTensor {
    fn id(&self) -> SuperGraphLinkId {
        self.0
    }
    fn to_any(&self) -> SuperGraphAnyLink {
        SuperGraphAnyLink::Tensor(*self)
    }
    fn global_id(&self) -> GlobalId {
        self.1
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct SuperGraphLinkTensorMap(SuperGraphLinkId, GlobalId);
impl SuperGraphLinkTensorMap {
    pub fn new(id: SuperGraphLinkId, rng: &mut impl Rng) -> Self {
        Self(id, GlobalId::new(rng))
    }
}
impl SuperGraphLink for SuperGraphLinkTensorMap {
    fn id(&self) -> SuperGraphLinkId {
        self.0
    }
    fn to_any(&self) -> SuperGraphAnyLink {
        SuperGraphAnyLink::TensorMap(*self)
    }
    fn global_id(&self) -> GlobalId {
        self.1
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct SuperGraphLinkTokenizer(SuperGraphLinkId, GlobalId);
impl SuperGraphLinkTokenizer {
    pub fn new(id: SuperGraphLinkId, rng: &mut impl Rng) -> Self {
        Self(id, GlobalId::new(rng))
    }
}
impl SuperGraphLink for SuperGraphLinkTokenizer {
    fn id(&self) -> SuperGraphLinkId {
        self.0
    }
    fn to_any(&self) -> SuperGraphAnyLink {
        SuperGraphAnyLink::Tokenizer(*self)
    }
    fn global_id(&self) -> GlobalId {
        self.1
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct SuperGraphLinkHash(SuperGraphLinkId, GlobalId);
impl SuperGraphLinkHash {
    pub fn new(id: SuperGraphLinkId, rng: &mut impl Rng) -> Self {
        Self(id, GlobalId::new(rng))
    }
}
impl SuperGraphLink for SuperGraphLinkHash {
    fn id(&self) -> SuperGraphLinkId {
        self.0
    }
    fn to_any(&self) -> SuperGraphAnyLink {
        SuperGraphAnyLink::Hash(*self)
    }
    fn global_id(&self) -> GlobalId {
        self.1
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum SuperGraphAnyLink {
    Tensor(SuperGraphLinkTensor),
    String(SuperGraphLinkString),
    TensorMap(SuperGraphLinkTensorMap),
    Tokenizer(SuperGraphLinkTokenizer),
    Hash(SuperGraphLinkHash),
}

impl SuperGraphAnyLink {
    #[allow(dead_code)]
    pub(crate) fn id(&self) -> SuperGraphLinkId {
        match self {
            SuperGraphAnyLink::Tensor(link) => link.id(),
            SuperGraphAnyLink::String(link) => link.id(),
            SuperGraphAnyLink::TensorMap(link) => link.id(),
            SuperGraphAnyLink::Tokenizer(link) => link.id(),
            SuperGraphAnyLink::Hash(link) => link.id(),
        }
    }

    pub(crate) fn global_id(&self) -> GlobalId {
        match self {
            SuperGraphAnyLink::Tensor(link) => link.global_id(),
            SuperGraphAnyLink::String(link) => link.global_id(),
            SuperGraphAnyLink::TensorMap(link) => link.global_id(),
            SuperGraphAnyLink::Tokenizer(link) => link.global_id(),
            SuperGraphAnyLink::Hash(link) => link.global_id(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum SuperGraphLinkDouble {
    Tensor(SuperGraphLinkTensor, SuperGraphLinkTensor),
    String(SuperGraphLinkString, SuperGraphLinkString),
    TensorMap(SuperGraphLinkTensorMap, SuperGraphLinkTensorMap),
    Tokenizer(SuperGraphLinkTokenizer, SuperGraphLinkTokenizer),
    Hash(SuperGraphLinkHash, SuperGraphLinkHash),
}

impl SuperGraphLinkDouble {
    pub fn first(&self) -> SuperGraphAnyLink {
        match self {
            SuperGraphLinkDouble::Tensor(t1, _) => SuperGraphAnyLink::Tensor(*t1),
            SuperGraphLinkDouble::String(s1, _) => SuperGraphAnyLink::String(*s1),
            SuperGraphLinkDouble::TensorMap(m1, _) => SuperGraphAnyLink::TensorMap(*m1),
            SuperGraphLinkDouble::Tokenizer(t1, _) => SuperGraphAnyLink::Tokenizer(*t1),
            SuperGraphLinkDouble::Hash(h1, _) => SuperGraphAnyLink::Hash(*h1),
        }
    }
    pub fn second(&self) -> SuperGraphAnyLink {
        match self {
            SuperGraphLinkDouble::Tensor(_, t2) => SuperGraphAnyLink::Tensor(*t2),
            SuperGraphLinkDouble::String(_, s2) => SuperGraphAnyLink::String(*s2),
            SuperGraphLinkDouble::TensorMap(_, m2) => SuperGraphAnyLink::TensorMap(*m2),
            SuperGraphLinkDouble::Tokenizer(_, t2) => SuperGraphAnyLink::Tokenizer(*t2),
            SuperGraphLinkDouble::Hash(_, h2) => SuperGraphAnyLink::Hash(*h2),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum SuperGraphLinkTriple {
    Tensor(
        SuperGraphLinkTensor,
        SuperGraphLinkTensor,
        SuperGraphLinkTensor,
    ),
    String(
        SuperGraphLinkString,
        SuperGraphLinkString,
        SuperGraphLinkString,
    ),
    TensorMap(
        SuperGraphLinkTensorMap,
        SuperGraphLinkTensorMap,
        SuperGraphLinkTensorMap,
    ),
    Tokenizer(
        SuperGraphLinkTokenizer,
        SuperGraphLinkTokenizer,
        SuperGraphLinkTokenizer,
    ),
    Hash(SuperGraphLinkHash, SuperGraphLinkHash, SuperGraphLinkHash),
}

impl SuperGraphLinkTriple {
    pub fn first(&self) -> SuperGraphAnyLink {
        match self {
            SuperGraphLinkTriple::Tensor(t1, _, _) => SuperGraphAnyLink::Tensor(*t1),
            SuperGraphLinkTriple::String(s1, _, _) => SuperGraphAnyLink::String(*s1),
            SuperGraphLinkTriple::TensorMap(m1, _, _) => SuperGraphAnyLink::TensorMap(*m1),
            SuperGraphLinkTriple::Tokenizer(t1, _, _) => SuperGraphAnyLink::Tokenizer(*t1),
            SuperGraphLinkTriple::Hash(h1, _, _) => SuperGraphAnyLink::Hash(*h1),
        }
    }
    pub fn second(&self) -> SuperGraphAnyLink {
        match self {
            SuperGraphLinkTriple::Tensor(_, t2, _) => SuperGraphAnyLink::Tensor(*t2),
            SuperGraphLinkTriple::String(_, s2, _) => SuperGraphAnyLink::String(*s2),
            SuperGraphLinkTriple::TensorMap(_, m2, _) => SuperGraphAnyLink::TensorMap(*m2),
            SuperGraphLinkTriple::Tokenizer(_, t2, _) => SuperGraphAnyLink::Tokenizer(*t2),
            SuperGraphLinkTriple::Hash(_, h2, _) => SuperGraphAnyLink::Hash(*h2),
        }
    }
    pub fn third(&self) -> SuperGraphAnyLink {
        match self {
            SuperGraphLinkTriple::Tensor(_, _, t3) => SuperGraphAnyLink::Tensor(*t3),
            SuperGraphLinkTriple::String(_, _, s3) => SuperGraphAnyLink::String(*s3),
            SuperGraphLinkTriple::TensorMap(_, _, m3) => SuperGraphAnyLink::TensorMap(*m3),
            SuperGraphLinkTriple::Tokenizer(_, _, t3) => SuperGraphAnyLink::Tokenizer(*t3),
            SuperGraphLinkTriple::Hash(_, _, h3) => SuperGraphAnyLink::Hash(*h3),
        }
    }
}
