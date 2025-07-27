use std::sync::Arc;
use serde::{Deserialize, Serialize};
use crate::model::Model;
use crate::super_graph::links::SuperGraphLinkTensor;
use crate::super_graph::SuperGraph;

/*
#[derive(Clone, Serialize, Deserialize)]
struct SuperGraphLLMTokenLogitInterface {
    token_input: SuperGraphLinkTensor,
    logit_output: SuperGraphLinkTensor,
    model: Arc<Model>,
    super_graph: SuperGraph,
}*/
