use crate::milli_graph::MilliOpGraphNodePath;
use crate::super_graph::SuperGraphNodePath;
use crate::symbolic_graph::SymbolicGraphNodePath;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum AnyNodePath {
    MilliOpGraphNodePath(MilliOpGraphNodePath),
    SuperGraphNodePath(SuperGraphNodePath),
    SymbolicGraphNodePath(SymbolicGraphNodePath),
}

pub enum AnyGraphPath {
    SuperGraphGraphPath(SuperGraphNodePath),
    SymbolicGraphGraphPath,
}
