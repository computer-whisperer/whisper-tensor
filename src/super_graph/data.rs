use crate::DynRank;
use crate::numeric_tensor::NumericTensor;
use crate::super_graph::links::{
    SuperGraphAnyLink, SuperGraphAtomicLinkKind, SuperGraphLink, SuperGraphLinkDouble,
    SuperGraphLinkKind,
};
use crate::super_graph::{SuperGraphError, SuperGraphHash};
use crate::symbolic_graph::tensor_store::TensorStore;
use crate::tokenizer::AnyTokenizer;
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct SuperGraphImage {
    pub tensor: NumericTensor<DynRank>,
}

impl SuperGraphImage {
    pub fn new(tensor: NumericTensor<DynRank>) -> Self {
        Self { tensor }
    }
}

impl From<NumericTensor<DynRank>> for SuperGraphImage {
    fn from(value: NumericTensor<DynRank>) -> Self {
        Self::new(value)
    }
}

#[derive(Clone, Debug)]
pub struct SuperGraphAudioClip {
    pub samples: NumericTensor<DynRank>,
    pub sample_rate_hz: u32,
}

impl SuperGraphAudioClip {
    pub fn new(samples: NumericTensor<DynRank>, sample_rate_hz: u32) -> Self {
        Self {
            samples,
            sample_rate_hz,
        }
    }
}

#[derive(Clone, Debug)]
pub enum SuperGraphMultimodalItem {
    Text(String),
    Image(SuperGraphImage),
    AudioClip(SuperGraphAudioClip),
}

#[derive(Clone)]
pub enum SuperGraphListValue<'models> {
    Tensor(Vec<NumericTensor<DynRank>>),
    String(Vec<String>),
    Tokenizer(Vec<AnyTokenizer>),
    TensorMap(Vec<&'models TensorStore>),
    Hash(Vec<SuperGraphHash>),
    Image(Vec<SuperGraphImage>),
    AudioClip(Vec<SuperGraphAudioClip>),
    MultimodalItem(Vec<SuperGraphMultimodalItem>),
}

impl<'models> SuperGraphListValue<'models> {
    pub fn item_kind(&self) -> SuperGraphAtomicLinkKind {
        match self {
            SuperGraphListValue::Tensor(_) => SuperGraphAtomicLinkKind::Tensor,
            SuperGraphListValue::String(_) => SuperGraphAtomicLinkKind::String,
            SuperGraphListValue::Tokenizer(_) => SuperGraphAtomicLinkKind::Tokenizer,
            SuperGraphListValue::TensorMap(_) => SuperGraphAtomicLinkKind::TensorMap,
            SuperGraphListValue::Hash(_) => SuperGraphAtomicLinkKind::Hash,
            SuperGraphListValue::Image(_) => SuperGraphAtomicLinkKind::Image,
            SuperGraphListValue::AudioClip(_) => SuperGraphAtomicLinkKind::AudioClip,
            SuperGraphListValue::MultimodalItem(_) => SuperGraphAtomicLinkKind::MultimodalItem,
        }
    }
}

#[derive(Clone, Default)]
pub struct SuperGraphData<'models> {
    pub tensors: HashMap<SuperGraphLink, NumericTensor<DynRank>>,
    pub strings: HashMap<SuperGraphLink, String>,
    pub tokenizers: HashMap<SuperGraphLink, AnyTokenizer>,
    pub tensor_maps: HashMap<SuperGraphLink, &'models TensorStore>,
    pub hashes: HashMap<SuperGraphLink, SuperGraphHash>,
    pub images: HashMap<SuperGraphLink, SuperGraphImage>,
    pub audio_clips: HashMap<SuperGraphLink, SuperGraphAudioClip>,
    pub multimodal_items: HashMap<SuperGraphLink, SuperGraphMultimodalItem>,
    pub lists: HashMap<SuperGraphLink, SuperGraphListValue<'models>>,
}

impl<'models> SuperGraphData<'models> {
    pub fn new() -> Self {
        Self {
            tensors: HashMap::new(),
            strings: HashMap::new(),
            tokenizers: HashMap::new(),
            tensor_maps: HashMap::new(),
            hashes: HashMap::new(),
            images: HashMap::new(),
            audio_clips: HashMap::new(),
            multimodal_items: HashMap::new(),
            lists: HashMap::new(),
        }
    }

    pub fn contains_link(&self, link: &SuperGraphLink) -> bool {
        match link.kind() {
            SuperGraphLinkKind::Tensor => self.tensors.contains_key(link),
            SuperGraphLinkKind::String => self.strings.contains_key(link),
            SuperGraphLinkKind::Tokenizer => self.tokenizers.contains_key(link),
            SuperGraphLinkKind::TensorMap => self.tensor_maps.contains_key(link),
            SuperGraphLinkKind::Hash => self.hashes.contains_key(link),
            SuperGraphLinkKind::Image => self.images.contains_key(link),
            SuperGraphLinkKind::AudioClip => self.audio_clips.contains_key(link),
            SuperGraphLinkKind::MultimodalItem => self.multimodal_items.contains_key(link),
            SuperGraphLinkKind::List(_) => self.lists.contains_key(link),
        }
    }

    pub fn copy_link_from(
        &mut self,
        source: &Self,
        input: SuperGraphLink,
        output: SuperGraphLink,
    ) -> Result<(), SuperGraphError> {
        if input.kind() != output.kind() {
            return Err(SuperGraphError::InvalidInputError(format!(
                "link kind mismatch while copying {:?} -> {:?}",
                input, output
            )));
        }

        match input.kind() {
            SuperGraphLinkKind::Tensor => {
                let value = source
                    .tensors
                    .get(&input)
                    .ok_or(SuperGraphError::MissingLinkError(format!(
                        ": missing tensor link {:?}",
                        input
                    )))?;
                self.tensors.insert(output, value.clone());
            }
            SuperGraphLinkKind::String => {
                let value = source
                    .strings
                    .get(&input)
                    .ok_or(SuperGraphError::MissingLinkError(format!(
                        ": missing string link {:?}",
                        input
                    )))?;
                self.strings.insert(output, value.clone());
            }
            SuperGraphLinkKind::Tokenizer => {
                let value =
                    source
                        .tokenizers
                        .get(&input)
                        .ok_or(SuperGraphError::MissingLinkError(format!(
                            ": missing tokenizer link {:?}",
                            input
                        )))?;
                self.tokenizers.insert(output, value.clone());
            }
            SuperGraphLinkKind::TensorMap => {
                let value =
                    source
                        .tensor_maps
                        .get(&input)
                        .ok_or(SuperGraphError::MissingLinkError(format!(
                            ": missing tensor_map link {:?}",
                            input
                        )))?;
                self.tensor_maps.insert(output, *value);
            }
            SuperGraphLinkKind::Hash => {
                let value = source
                    .hashes
                    .get(&input)
                    .ok_or(SuperGraphError::MissingLinkError(format!(
                        ": missing hash link {:?}",
                        input
                    )))?;
                self.hashes.insert(output, *value);
            }
            SuperGraphLinkKind::Image => {
                let value = source
                    .images
                    .get(&input)
                    .ok_or(SuperGraphError::MissingLinkError(format!(
                        ": missing image link {:?}",
                        input
                    )))?;
                self.images.insert(output, value.clone());
            }
            SuperGraphLinkKind::AudioClip => {
                let value =
                    source
                        .audio_clips
                        .get(&input)
                        .ok_or(SuperGraphError::MissingLinkError(format!(
                            ": missing audio clip link {:?}",
                            input
                        )))?;
                self.audio_clips.insert(output, value.clone());
            }
            SuperGraphLinkKind::MultimodalItem => {
                let value = source.multimodal_items.get(&input).ok_or(
                    SuperGraphError::MissingLinkError(format!(
                        ": missing multimodal item link {:?}",
                        input
                    )),
                )?;
                self.multimodal_items.insert(output, value.clone());
            }
            SuperGraphLinkKind::List(item_kind) => {
                let value = source
                    .lists
                    .get(&input)
                    .ok_or(SuperGraphError::MissingLinkError(format!(
                        ": missing list link {:?}",
                        input
                    )))?;
                if value.item_kind() != item_kind {
                    return Err(SuperGraphError::InvalidInputError(format!(
                        "list item kind mismatch for {:?}: link declares {:?}, value contains {:?}",
                        input,
                        item_kind,
                        value.item_kind()
                    )));
                }
                self.lists.insert(output, value.clone());
            }
        }

        Ok(())
    }

    pub fn select(&self, links: &[SuperGraphAnyLink]) -> Result<Self, SuperGraphError> {
        let mut selected = Self::new();
        for link in links {
            selected.copy_link_from(self, *link, *link)?;
        }
        Ok(selected)
    }

    pub fn remap(&self, map: Vec<SuperGraphLinkDouble>) -> Result<Self, SuperGraphError> {
        let mut new_data = Self::new();

        for link in map {
            new_data.copy_link_from(self, link.first(), link.second())?;
        }

        Ok(new_data)
    }

    pub fn extend(&mut self, other: &Self) {
        self.tensors
            .extend(other.tensors.iter().map(|(a, b)| (*a, b.clone())));
        self.strings
            .extend(other.strings.iter().map(|(a, b)| (*a, b.clone())));
        self.tokenizers
            .extend(other.tokenizers.iter().map(|(a, b)| (*a, b.clone())));
        self.tensor_maps
            .extend(other.tensor_maps.iter().map(|(a, b)| (*a, *b)));
        self.hashes
            .extend(other.hashes.iter().map(|(a, b)| (*a, *b)));
        self.images
            .extend(other.images.iter().map(|(a, b)| (*a, b.clone())));
        self.audio_clips
            .extend(other.audio_clips.iter().map(|(a, b)| (*a, b.clone())));
        self.multimodal_items
            .extend(other.multimodal_items.iter().map(|(a, b)| (*a, b.clone())));
        self.lists
            .extend(other.lists.iter().map(|(a, b)| (*a, b.clone())));
    }
}
