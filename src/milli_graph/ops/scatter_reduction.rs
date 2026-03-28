use serde::{Deserialize, Serialize};

/// Reduction mode for ScatterElements and ScatterND.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScatterReduction {
    None,
    Add,
    Mul,
    Max,
    Min,
}

impl ScatterReduction {
    /// Apply this reduction to combine `existing` and `update` values.
    pub fn apply(&self, existing: f32, update: f32) -> f32 {
        match self {
            ScatterReduction::None => update,
            ScatterReduction::Add => existing + update,
            ScatterReduction::Mul => existing * update,
            ScatterReduction::Max => existing.max(update),
            ScatterReduction::Min => existing.min(update),
        }
    }
}
