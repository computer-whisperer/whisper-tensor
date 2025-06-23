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
pub struct SuperGraphLinkId (pub(crate) u32);

#[derive(Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct SuperGraphLinkString (SuperGraphLinkId);
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
pub struct SuperGraphLinkTensor (SuperGraphLinkId);
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
pub struct SuperGraphLinkModel (SuperGraphLinkId);
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
pub struct SuperGraphLinkTokenizer (SuperGraphLinkId);
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
pub enum SuperGraphAnyLink {
    Tensor(SuperGraphLinkTensor),
    String(SuperGraphLinkString),
    Model(SuperGraphLinkModel),
    Tokenizer(SuperGraphLinkTokenizer)
}

impl SuperGraphAnyLink {
    #[allow(dead_code)]
    fn id(&self) -> SuperGraphLinkId {
        match self {
            SuperGraphAnyLink::Tensor(link) => link.id(),
            SuperGraphAnyLink::String(link) => link.id(),
            SuperGraphAnyLink::Model(link) => link.id(),
            SuperGraphAnyLink::Tokenizer(link) => link.id(),
        }
    }
}