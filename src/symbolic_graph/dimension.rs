use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub(crate) struct UnknownDimensionId (usize);

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub(crate) struct UnknownDimension {
    id: UnknownDimensionId,
    offset: i64
}

impl UnknownDimension {
    pub fn try_test_eq(&self, other: &UnknownDimension) -> Option<bool> {
        if self.id == other.id {
            Some(self.offset == other.offset)
        } else {
            None
        }
    }
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub(crate) enum Dimension {
    Known(usize),
    Unknown(UnknownDimension),
}

impl Dimension {
    pub fn try_test_eq(&self, other: &Dimension) -> Option<bool> {
        match self {
            Dimension::Known(x) => {
                match other {
                    Dimension::Known(y) => Some(x == y),
                    _ => None
                }
            }
            Dimension::Unknown(x) => {
                match other {
                    Dimension::Unknown(y) => x.try_test_eq(y),
                    _ => None
                }
            }
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct DimensionResolver {
    next_unknown_dim_id: usize
}

impl DimensionResolver {
    pub fn new() -> Self {
        DimensionResolver { next_unknown_dim_id: 0 }
    }
    pub fn update_last_assigned(&mut self, dim_id: UnknownDimension) {
        if dim_id.id.0 >= self.next_unknown_dim_id {
            self.next_unknown_dim_id = dim_id.id.0 + 1;
        }
    }
    pub fn new_unknown(&mut self) -> UnknownDimension {
        let id = UnknownDimensionId(self.next_unknown_dim_id);
        self.next_unknown_dim_id += 1;
        UnknownDimension { id, offset: 0 }
    }
}
