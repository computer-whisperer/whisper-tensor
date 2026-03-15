use crate::graph::{GlobalId, Link, LinkMetadata, Property, PropertyValue};
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum SuperGraphAtomicLinkKind {
    Tensor,
    String,
    TensorMap,
    Tokenizer,
    Hash,
    Image,
    AudioClip,
    MultimodalItem,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum SuperGraphLinkKind {
    Tensor,
    String,
    TensorMap,
    Tokenizer,
    Hash,
    Image,
    AudioClip,
    MultimodalItem,
    List(SuperGraphAtomicLinkKind),
}

impl SuperGraphLinkKind {
    pub fn list(item_kind: SuperGraphAtomicLinkKind) -> Self {
        SuperGraphLinkKind::List(item_kind)
    }

    pub fn list_item_kind(&self) -> Option<SuperGraphAtomicLinkKind> {
        match self {
            SuperGraphLinkKind::List(item_kind) => Some(*item_kind),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            SuperGraphLinkKind::Tensor => "Tensor",
            SuperGraphLinkKind::String => "String",
            SuperGraphLinkKind::TensorMap => "TensorMap",
            SuperGraphLinkKind::Tokenizer => "Tokenizer",
            SuperGraphLinkKind::Hash => "Hash",
            SuperGraphLinkKind::Image => "Image",
            SuperGraphLinkKind::AudioClip => "AudioClip",
            SuperGraphLinkKind::MultimodalItem => "MultimodalItem",
            SuperGraphLinkKind::List(SuperGraphAtomicLinkKind::Tensor) => "TensorList",
            SuperGraphLinkKind::List(SuperGraphAtomicLinkKind::String) => "StringList",
            SuperGraphLinkKind::List(SuperGraphAtomicLinkKind::TensorMap) => "TensorMapList",
            SuperGraphLinkKind::List(SuperGraphAtomicLinkKind::Tokenizer) => "TokenizerList",
            SuperGraphLinkKind::List(SuperGraphAtomicLinkKind::Hash) => "HashList",
            SuperGraphLinkKind::List(SuperGraphAtomicLinkKind::Image) => "ImageList",
            SuperGraphLinkKind::List(SuperGraphAtomicLinkKind::AudioClip) => "AudioClipList",
            SuperGraphLinkKind::List(SuperGraphAtomicLinkKind::MultimodalItem) => {
                "MultimodalItemList"
            }
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct SuperGraphLink {
    global_id: GlobalId,
    kind: SuperGraphLinkKind,
}

impl SuperGraphLink {
    pub fn new(kind: SuperGraphLinkKind, rng: &mut impl Rng) -> Self {
        Self {
            global_id: GlobalId::new(rng),
            kind,
        }
    }

    pub fn with_global_id(global_id: GlobalId, kind: SuperGraphLinkKind) -> Self {
        Self { global_id, kind }
    }

    pub fn tensor(global_id: GlobalId) -> Self {
        Self::with_global_id(global_id, SuperGraphLinkKind::Tensor)
    }

    pub fn image(global_id: GlobalId) -> Self {
        Self::with_global_id(global_id, SuperGraphLinkKind::Image)
    }

    pub fn audio_clip(global_id: GlobalId) -> Self {
        Self::with_global_id(global_id, SuperGraphLinkKind::AudioClip)
    }

    pub fn multimodal_item(global_id: GlobalId) -> Self {
        Self::with_global_id(global_id, SuperGraphLinkKind::MultimodalItem)
    }

    pub fn list(global_id: GlobalId, item_kind: SuperGraphAtomicLinkKind) -> Self {
        Self::with_global_id(global_id, SuperGraphLinkKind::list(item_kind))
    }

    pub fn kind(&self) -> SuperGraphLinkKind {
        self.kind
    }

    pub fn to_any(&self) -> SuperGraphAnyLink {
        *self
    }

    pub fn global_id(&self) -> GlobalId {
        self.global_id
    }
}

pub type SuperGraphAnyLink = SuperGraphLink;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraphLinkInfo {
    link: SuperGraphLink,
    label: Option<String>,
}

impl SuperGraphLinkInfo {
    pub fn new(link: SuperGraphLink, label: Option<String>) -> Self {
        Self { link, label }
    }

    pub fn link(&self) -> SuperGraphLink {
        self.link
    }

    pub fn kind(&self) -> SuperGraphLinkKind {
        self.link.kind()
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct SuperGraphLinkDouble {
    first: SuperGraphLink,
    second: SuperGraphLink,
}

impl SuperGraphLinkDouble {
    pub fn new(first: SuperGraphLink, second: SuperGraphLink) -> Self {
        assert_eq!(
            first.kind(),
            second.kind(),
            "SuperGraphLinkDouble kind mismatch: {:?} vs {:?}",
            first.kind(),
            second.kind()
        );
        Self { first, second }
    }

    pub fn first(&self) -> SuperGraphAnyLink {
        self.first
    }

    pub fn second(&self) -> SuperGraphAnyLink {
        self.second
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct SuperGraphLinkTriple {
    first: SuperGraphLink,
    second: SuperGraphLink,
    third: SuperGraphLink,
}

impl SuperGraphLinkTriple {
    pub fn new(first: SuperGraphLink, second: SuperGraphLink, third: SuperGraphLink) -> Self {
        assert_eq!(
            first.kind(),
            second.kind(),
            "SuperGraphLinkTriple kind mismatch: {:?} vs {:?}",
            first.kind(),
            second.kind()
        );
        assert_eq!(
            second.kind(),
            third.kind(),
            "SuperGraphLinkTriple kind mismatch: {:?} vs {:?}",
            second.kind(),
            third.kind()
        );
        Self {
            first,
            second,
            third,
        }
    }

    pub fn first(&self) -> SuperGraphAnyLink {
        self.first
    }

    pub fn second(&self) -> SuperGraphAnyLink {
        self.second
    }

    pub fn third(&self) -> SuperGraphAnyLink {
        self.third
    }
}

impl LinkMetadata for SuperGraphAnyLink {
    fn properties(&self) -> Vec<Property> {
        vec![Property::new(
            "link_type",
            PropertyValue::String(self.kind().as_str().to_string()),
        )]
    }
}

impl Link for SuperGraphLinkInfo {
    fn global_id(&self) -> GlobalId {
        self.link.global_id()
    }

    fn label(&self) -> Option<String> {
        self.label.clone()
    }
}

impl LinkMetadata for SuperGraphLinkInfo {
    fn properties(&self) -> Vec<Property> {
        vec![Property::new(
            "link_type",
            PropertyValue::String(self.kind().as_str().to_string()),
        )]
    }
}
