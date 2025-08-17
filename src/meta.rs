use serde::{Deserialize, Serialize};
use crate::milli_graph::MilliOpGraphTensorId;
use crate::super_graph::SuperGraphTensorPath;
use crate::symbolic_graph::SymbolicGraphTensorPath;

#[derive(Debug, Clone, Serialize, Deserialize, Hash, Eq, PartialEq)]
pub enum AnyTensorPath {
    SymbolicGraphTensorPath(SymbolicGraphTensorPath),
    SuperGraphTensorPath(SuperGraphTensorPath),
    MilliOpGraphTensorPath(MilliOpGraphTensorId)
}