use crate::numeric_tensor::NumericTensor;
use crate::pool::Pool;
use crate::super_graph::links::{
    SuperGraphAnyLink, SuperGraphAtomicLinkKind, SuperGraphLink, SuperGraphLinkDouble,
    SuperGraphLinkKind,
};
use crate::super_graph::{SuperGraphError, SuperGraphHash};
use crate::symbolic_graph::tensor_store::TensorStore;
use crate::tensor_rank::DynRank;
use crate::tokenizer::AnyTokenizer;
use std::collections::HashMap;

#[derive(Debug)]
pub struct SuperGraphImage<'p, P: Pool + 'p> {
    pub tensor: NumericTensor<'p, DynRank, P>,
}

impl<'p, P: Pool + 'p> SuperGraphImage<'p, P> {
    pub fn new(tensor: NumericTensor<'p, DynRank, P>) -> Self {
        Self { tensor }
    }
}

#[derive(Debug)]
pub struct SuperGraphAudioClip<'p, P: Pool + 'p> {
    pub samples: NumericTensor<'p, DynRank, P>,
    pub sample_rate_hz: u32,
}

impl<'p, P: Pool + 'p> SuperGraphAudioClip<'p, P> {
    pub fn new(samples: NumericTensor<'p, DynRank, P>, sample_rate_hz: u32) -> Self {
        Self {
            samples,
            sample_rate_hz,
        }
    }
}

#[derive(Debug)]
pub struct SuperGraphVideoClip<'p, P: Pool + 'p> {
    pub frames: NumericTensor<'p, DynRank, P>,
    pub fps: f32,
}

impl<'p, P: Pool + 'p> SuperGraphVideoClip<'p, P> {
    pub fn new(frames: NumericTensor<'p, DynRank, P>, fps: f32) -> Self {
        Self { frames, fps }
    }
}

#[derive(Debug)]
pub enum SuperGraphMultimodalItem<'p, P: Pool + 'p> {
    Text(String),
    Image(SuperGraphImage<'p, P>),
    AudioClip(SuperGraphAudioClip<'p, P>),
    VideoClip(SuperGraphVideoClip<'p, P>),
}

pub enum SuperGraphListValue<'p, 'models, P: Pool + 'p> {
    Tensor(Vec<NumericTensor<'p, DynRank, P>>),
    String(Vec<String>),
    Tokenizer(Vec<AnyTokenizer>),
    TensorMap(Vec<&'models TensorStore>),
    Hash(Vec<SuperGraphHash>),
    Image(Vec<SuperGraphImage<'p, P>>),
    AudioClip(Vec<SuperGraphAudioClip<'p, P>>),
    VideoClip(Vec<SuperGraphVideoClip<'p, P>>),
    MultimodalItem(Vec<SuperGraphMultimodalItem<'p, P>>),
}

impl<'p, 'models, P: Pool + 'p> SuperGraphListValue<'p, 'models, P> {
    pub fn item_kind(&self) -> SuperGraphAtomicLinkKind {
        match self {
            SuperGraphListValue::Tensor(_) => SuperGraphAtomicLinkKind::Tensor,
            SuperGraphListValue::String(_) => SuperGraphAtomicLinkKind::String,
            SuperGraphListValue::Tokenizer(_) => SuperGraphAtomicLinkKind::Tokenizer,
            SuperGraphListValue::TensorMap(_) => SuperGraphAtomicLinkKind::TensorMap,
            SuperGraphListValue::Hash(_) => SuperGraphAtomicLinkKind::Hash,
            SuperGraphListValue::Image(_) => SuperGraphAtomicLinkKind::Image,
            SuperGraphListValue::AudioClip(_) => SuperGraphAtomicLinkKind::AudioClip,
            SuperGraphListValue::VideoClip(_) => SuperGraphAtomicLinkKind::VideoClip,
            SuperGraphListValue::MultimodalItem(_) => SuperGraphAtomicLinkKind::MultimodalItem,
        }
    }
}

pub struct SuperGraphData<'p, 'models, P: Pool + 'p> {
    pub tensors: HashMap<SuperGraphLink, NumericTensor<'p, DynRank, P>>,
    pub strings: HashMap<SuperGraphLink, String>,
    pub tokenizers: HashMap<SuperGraphLink, AnyTokenizer>,
    pub tensor_maps: HashMap<SuperGraphLink, &'models TensorStore>,
    pub hashes: HashMap<SuperGraphLink, SuperGraphHash>,
    pub images: HashMap<SuperGraphLink, SuperGraphImage<'p, P>>,
    pub audio_clips: HashMap<SuperGraphLink, SuperGraphAudioClip<'p, P>>,
    pub video_clips: HashMap<SuperGraphLink, SuperGraphVideoClip<'p, P>>,
    pub multimodal_items: HashMap<SuperGraphLink, SuperGraphMultimodalItem<'p, P>>,
    pub lists: HashMap<SuperGraphLink, SuperGraphListValue<'p, 'models, P>>,
}

impl<'p, 'models, P: Pool + 'p> Default for SuperGraphData<'p, 'models, P> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'p, 'models, P: Pool + 'p> SuperGraphData<'p, 'models, P> {
    pub fn new() -> Self {
        Self {
            tensors: HashMap::new(),
            strings: HashMap::new(),
            tokenizers: HashMap::new(),
            tensor_maps: HashMap::new(),
            hashes: HashMap::new(),
            images: HashMap::new(),
            audio_clips: HashMap::new(),
            video_clips: HashMap::new(),
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
            SuperGraphLinkKind::VideoClip => self.video_clips.contains_key(link),
            SuperGraphLinkKind::MultimodalItem => self.multimodal_items.contains_key(link),
            SuperGraphLinkKind::List(_) => self.lists.contains_key(link),
        }
    }

    /// Move a link's value from `source` into `self`. Removes from source.
    pub fn take_link_from(
        &mut self,
        source: &mut Self,
        input: SuperGraphLink,
        output: SuperGraphLink,
    ) -> Result<(), SuperGraphError> {
        if input.kind() != output.kind() {
            return Err(SuperGraphError::InvalidInputError(format!(
                "link kind mismatch while taking {:?} -> {:?}",
                input, output
            )));
        }

        match input.kind() {
            SuperGraphLinkKind::Tensor => {
                let value =
                    source
                        .tensors
                        .remove(&input)
                        .ok_or(SuperGraphError::MissingLinkError(format!(
                            ": missing tensor link {:?}",
                            input
                        )))?;
                self.tensors.insert(output, value);
            }
            SuperGraphLinkKind::String => {
                let value =
                    source
                        .strings
                        .remove(&input)
                        .ok_or(SuperGraphError::MissingLinkError(format!(
                            ": missing string link {:?}",
                            input
                        )))?;
                self.strings.insert(output, value);
            }
            SuperGraphLinkKind::Tokenizer => {
                let value =
                    source
                        .tokenizers
                        .remove(&input)
                        .ok_or(SuperGraphError::MissingLinkError(format!(
                            ": missing tokenizer link {:?}",
                            input
                        )))?;
                self.tokenizers.insert(output, value);
            }
            SuperGraphLinkKind::TensorMap => {
                let value =
                    source
                        .tensor_maps
                        .remove(&input)
                        .ok_or(SuperGraphError::MissingLinkError(format!(
                            ": missing tensor_map link {:?}",
                            input
                        )))?;
                self.tensor_maps.insert(output, value);
            }
            SuperGraphLinkKind::Hash => {
                let value =
                    source
                        .hashes
                        .remove(&input)
                        .ok_or(SuperGraphError::MissingLinkError(format!(
                            ": missing hash link {:?}",
                            input
                        )))?;
                self.hashes.insert(output, value);
            }
            SuperGraphLinkKind::Image => {
                let value =
                    source
                        .images
                        .remove(&input)
                        .ok_or(SuperGraphError::MissingLinkError(format!(
                            ": missing image link {:?}",
                            input
                        )))?;
                self.images.insert(output, value);
            }
            SuperGraphLinkKind::AudioClip => {
                let value =
                    source
                        .audio_clips
                        .remove(&input)
                        .ok_or(SuperGraphError::MissingLinkError(format!(
                            ": missing audio clip link {:?}",
                            input
                        )))?;
                self.audio_clips.insert(output, value);
            }
            SuperGraphLinkKind::VideoClip => {
                let value =
                    source
                        .video_clips
                        .remove(&input)
                        .ok_or(SuperGraphError::MissingLinkError(format!(
                            ": missing video clip link {:?}",
                            input
                        )))?;
                self.video_clips.insert(output, value);
            }
            SuperGraphLinkKind::MultimodalItem => {
                let value = source.multimodal_items.remove(&input).ok_or(
                    SuperGraphError::MissingLinkError(format!(
                        ": missing multimodal item link {:?}",
                        input
                    )),
                )?;
                self.multimodal_items.insert(output, value);
            }
            SuperGraphLinkKind::List(item_kind) => {
                let value =
                    source
                        .lists
                        .remove(&input)
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
                self.lists.insert(output, value);
            }
        }

        Ok(())
    }

    /// Copy a tensor from `source` into `self` by allocating a new copy on `pool`.
    /// Does not remove from source — use for data that must survive (e.g., Scan simple_inputs).
    /// Only supports tensor and non-tensor (Copy/Clone) link kinds.
    pub fn copy_link_from(
        &mut self,
        source: &Self,
        input: SuperGraphLink,
        output: SuperGraphLink,
        pool: &'p P,
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
                let copy = value
                    .view()
                    .to_tensor(pool)
                    .map_err(|e| SuperGraphError::InvalidInputError(format!("allocation: {e}")))?;
                self.tensors.insert(output, copy);
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
                let copy =
                    value.tensor.view().to_tensor(pool).map_err(|e| {
                        SuperGraphError::InvalidInputError(format!("allocation: {e}"))
                    })?;
                self.images.insert(output, SuperGraphImage::new(copy));
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
                let copy =
                    value.samples.view().to_tensor(pool).map_err(|e| {
                        SuperGraphError::InvalidInputError(format!("allocation: {e}"))
                    })?;
                self.audio_clips
                    .insert(output, SuperGraphAudioClip::new(copy, value.sample_rate_hz));
            }
            SuperGraphLinkKind::VideoClip => {
                let value =
                    source
                        .video_clips
                        .get(&input)
                        .ok_or(SuperGraphError::MissingLinkError(format!(
                            ": missing video clip link {:?}",
                            input
                        )))?;
                let copy =
                    value.frames.view().to_tensor(pool).map_err(|e| {
                        SuperGraphError::InvalidInputError(format!("allocation: {e}"))
                    })?;
                self.video_clips
                    .insert(output, SuperGraphVideoClip::new(copy, value.fps));
            }
            SuperGraphLinkKind::MultimodalItem => {
                let value = source.multimodal_items.get(&input).ok_or(
                    SuperGraphError::MissingLinkError(format!(
                        ": missing multimodal item link {:?}",
                        input
                    )),
                )?;
                let copied = match value {
                    SuperGraphMultimodalItem::Text(s) => SuperGraphMultimodalItem::Text(s.clone()),
                    SuperGraphMultimodalItem::Image(img) => {
                        let t = img.tensor.view().to_tensor(pool).map_err(|e| {
                            SuperGraphError::InvalidInputError(format!("allocation: {e}"))
                        })?;
                        SuperGraphMultimodalItem::Image(SuperGraphImage::new(t))
                    }
                    SuperGraphMultimodalItem::AudioClip(clip) => {
                        let t = clip.samples.view().to_tensor(pool).map_err(|e| {
                            SuperGraphError::InvalidInputError(format!("allocation: {e}"))
                        })?;
                        SuperGraphMultimodalItem::AudioClip(SuperGraphAudioClip::new(
                            t,
                            clip.sample_rate_hz,
                        ))
                    }
                    SuperGraphMultimodalItem::VideoClip(clip) => {
                        let t = clip.frames.view().to_tensor(pool).map_err(|e| {
                            SuperGraphError::InvalidInputError(format!("allocation: {e}"))
                        })?;
                        SuperGraphMultimodalItem::VideoClip(SuperGraphVideoClip::new(t, clip.fps))
                    }
                };
                self.multimodal_items.insert(output, copied);
            }
            SuperGraphLinkKind::List(_) => {
                // Lists contain owned tensors/media that would each need deep-copying.
                // This is not yet needed by any caller — Scan simple_inputs are never lists.
                return Err(SuperGraphError::InvalidInputError(
                    "copy_link_from not supported for List (use take_link_from instead)"
                        .to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Consume self, extracting only the requested links. All other data is dropped.
    pub fn into_selected(mut self, links: &[SuperGraphAnyLink]) -> Result<Self, SuperGraphError> {
        let mut selected = Self::new();
        for &link in links {
            selected.take_link_from(&mut self, link, link)?;
        }
        Ok(selected)
    }

    /// Remap links from source keys to destination keys, moving data.
    pub fn into_remapped(
        mut self,
        map: Vec<SuperGraphLinkDouble>,
    ) -> Result<Self, SuperGraphError> {
        let mut new_data = Self::new();
        for link in map {
            new_data.take_link_from(&mut self, link.first(), link.second())?;
        }
        Ok(new_data)
    }

    /// Merge all entries from `other` into `self`, consuming `other`.
    pub fn extend_from(&mut self, other: Self) {
        self.tensors.extend(other.tensors);
        self.strings.extend(other.strings);
        self.tokenizers.extend(other.tokenizers);
        self.tensor_maps.extend(other.tensor_maps);
        self.hashes.extend(other.hashes);
        self.images.extend(other.images);
        self.audio_clips.extend(other.audio_clips);
        self.video_clips.extend(other.video_clips);
        self.multimodal_items.extend(other.multimodal_items);
        self.lists.extend(other.lists);
    }
}
