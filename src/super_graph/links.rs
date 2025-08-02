use serde::{Deserialize, Serialize};

pub trait SuperGraphLink {
    fn id(&self) -> SuperGraphLinkId;
    fn to_any(&self) -> SuperGraphAnyLink;
}

impl<T: SuperGraphLink> From<&T> for SuperGraphAnyLink {
    fn from(value: &T) -> Self {
        value.to_any()
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct SuperGraphLinkId(pub(crate) u32);

#[derive(Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct SuperGraphLinkString(SuperGraphLinkId);
impl SuperGraphLinkString {
    pub fn new(id: SuperGraphLinkId) -> Self {
        Self(id)
    }
}
impl SuperGraphLink for SuperGraphLinkString {
    fn id(&self) -> SuperGraphLinkId {
        self.0.clone()
    }
    fn to_any(&self) -> SuperGraphAnyLink {
        SuperGraphAnyLink::String(self.clone())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct SuperGraphLinkTensor(SuperGraphLinkId);
impl SuperGraphLinkTensor {
    pub fn new(id: SuperGraphLinkId) -> Self {
        Self(id)
    }
}
impl SuperGraphLink for SuperGraphLinkTensor {
    fn id(&self) -> SuperGraphLinkId {
        self.0.clone()
    }
    fn to_any(&self) -> SuperGraphAnyLink {
        SuperGraphAnyLink::Tensor(self.clone())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct SuperGraphLinkModel(SuperGraphLinkId);
impl SuperGraphLinkModel {
    pub fn new(id: SuperGraphLinkId) -> Self {
        Self(id)
    }
}
impl SuperGraphLink for SuperGraphLinkModel {
    fn id(&self) -> SuperGraphLinkId {
        self.0.clone()
    }
    fn to_any(&self) -> SuperGraphAnyLink {
        SuperGraphAnyLink::Model(self.clone())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct SuperGraphLinkTokenizer(SuperGraphLinkId);
impl SuperGraphLinkTokenizer {
    pub fn new(id: SuperGraphLinkId) -> Self {
        Self(id)
    }
}
impl SuperGraphLink for SuperGraphLinkTokenizer {
    fn id(&self) -> SuperGraphLinkId {
        self.0.clone()
    }
    fn to_any(&self) -> SuperGraphAnyLink {
        SuperGraphAnyLink::Tokenizer(self.clone())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct SuperGraphLinkHash(SuperGraphLinkId);
impl SuperGraphLinkHash {
    pub fn new(id: SuperGraphLinkId) -> Self {
        Self(id)
    }
}
impl SuperGraphLink for SuperGraphLinkHash {
    fn id(&self) -> SuperGraphLinkId {
        self.0.clone()
    }
    fn to_any(&self) -> SuperGraphAnyLink {
        SuperGraphAnyLink::Hash(self.clone())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum SuperGraphAnyLink {
    Tensor(SuperGraphLinkTensor),
    String(SuperGraphLinkString),
    Model(SuperGraphLinkModel),
    Tokenizer(SuperGraphLinkTokenizer),
    Hash(SuperGraphLinkHash),
}

impl SuperGraphAnyLink {
    #[allow(dead_code)]
    fn id(&self) -> SuperGraphLinkId {
        match self {
            SuperGraphAnyLink::Tensor(link) => link.id(),
            SuperGraphAnyLink::String(link) => link.id(),
            SuperGraphAnyLink::Model(link) => link.id(),
            SuperGraphAnyLink::Tokenizer(link) => link.id(),
            SuperGraphAnyLink::Hash(link) => link.id(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum SuperGraphLinkDouble {
    Tensor(SuperGraphLinkTensor, SuperGraphLinkTensor),
    String(SuperGraphLinkString, SuperGraphLinkString),
    Model(SuperGraphLinkModel, SuperGraphLinkModel),
    Tokenizer(SuperGraphLinkTokenizer, SuperGraphLinkTokenizer),
    Hash(SuperGraphLinkHash, SuperGraphLinkHash),
}

impl SuperGraphLinkDouble {
    pub fn first(&self) -> SuperGraphAnyLink {
        match self {
            SuperGraphLinkDouble::Tensor(t1, _) => SuperGraphAnyLink::Tensor(t1.clone()),
            SuperGraphLinkDouble::String(s1, _) => SuperGraphAnyLink::String(s1.clone()),
            SuperGraphLinkDouble::Model(m1, _) => SuperGraphAnyLink::Model(m1.clone()),
            SuperGraphLinkDouble::Tokenizer(t1, _) => SuperGraphAnyLink::Tokenizer(t1.clone()),
            SuperGraphLinkDouble::Hash(h1, _) => SuperGraphAnyLink::Hash(h1.clone()),
        }
    }
    pub fn second(&self) -> SuperGraphAnyLink {
        match self {
            SuperGraphLinkDouble::Tensor(_, t2) => SuperGraphAnyLink::Tensor(t2.clone()),
            SuperGraphLinkDouble::String(_, s2) => SuperGraphAnyLink::String(s2.clone()),
            SuperGraphLinkDouble::Model(_, m2) => SuperGraphAnyLink::Model(m2.clone()),
            SuperGraphLinkDouble::Tokenizer(_, t2) => SuperGraphAnyLink::Tokenizer(t2.clone()),
            SuperGraphLinkDouble::Hash(_, h2) => SuperGraphAnyLink::Hash(h2.clone()),
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
    Model(
        SuperGraphLinkModel,
        SuperGraphLinkModel,
        SuperGraphLinkModel,
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
            SuperGraphLinkTriple::Tensor(t1, _, _) => SuperGraphAnyLink::Tensor(t1.clone()),
            SuperGraphLinkTriple::String(s1, _, _) => SuperGraphAnyLink::String(s1.clone()),
            SuperGraphLinkTriple::Model(m1, _, _) => SuperGraphAnyLink::Model(m1.clone()),
            SuperGraphLinkTriple::Tokenizer(t1, _, _) => SuperGraphAnyLink::Tokenizer(t1.clone()),
            SuperGraphLinkTriple::Hash(h1, _, _) => SuperGraphAnyLink::Hash(h1.clone()),
        }
    }
    pub fn second(&self) -> SuperGraphAnyLink {
        match self {
            SuperGraphLinkTriple::Tensor(_, t2, _) => SuperGraphAnyLink::Tensor(t2.clone()),
            SuperGraphLinkTriple::String(_, s2, _) => SuperGraphAnyLink::String(s2.clone()),
            SuperGraphLinkTriple::Model(_, m2, _) => SuperGraphAnyLink::Model(m2.clone()),
            SuperGraphLinkTriple::Tokenizer(_, t2, _) => SuperGraphAnyLink::Tokenizer(t2.clone()),
            SuperGraphLinkTriple::Hash(_, h2, _) => SuperGraphAnyLink::Hash(h2.clone()),
        }
    }
    pub fn third(&self) -> SuperGraphAnyLink {
        match self {
            SuperGraphLinkTriple::Tensor(_, _, t3) => SuperGraphAnyLink::Tensor(t3.clone()),
            SuperGraphLinkTriple::String(_, _, s3) => SuperGraphAnyLink::String(s3.clone()),
            SuperGraphLinkTriple::Model(_, _, m3) => SuperGraphAnyLink::Model(m3.clone()),
            SuperGraphLinkTriple::Tokenizer(_, _, t3) => SuperGraphAnyLink::Tokenizer(t3.clone()),
            SuperGraphLinkTriple::Hash(_, _, h3) => SuperGraphAnyLink::Hash(h3.clone()),
        }
    }
}
