use crate::onnx_graph::WeightStorageStrategy;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use whisper_tensor::backends::ndarray_backend::NDArrayNumericTensor;
use whisper_tensor::dtype::DType;
use whisper_tensor::interfaces::TextInferenceTokensInLogitOutInterface;
use whisper_tensor::loader::{LoadedModel, LoaderError, LoaderOutput};
use whisper_tensor::metadata::TokenizerInfo;
use whisper_tensor::milli_graph::MilliOpGraph;
use whisper_tensor::milli_graph::ops::{Cast, Constant, Shape, Squeeze, Unsqueeze};
use whisper_tensor::model::Model;
use whisper_tensor::super_graph::SuperGraphBuilder;
use whisper_tensor::super_graph::links::{
    SuperGraphLink, SuperGraphLinkDouble, SuperGraphLinkKind, SuperGraphLinkTriple,
};
use whisper_tensor::super_graph::nodes::{
    SuperGraphNode, SuperGraphNodeMilliOpGraph, SuperGraphNodeModelExecution,
    SuperGraphNodeRNNCacheRead, SuperGraphNodeRNNCacheWrite, SuperGraphNodeScan,
};

/// Helper: load ONNX bytes into a Model (no interface detection).
/// The calling loader is responsible for building any interfaces.
pub(super) fn onnx_bytes_to_model(
    onnx_data: &[u8],
    model_name: &str,
    base_dir: Option<&Path>,
) -> Result<(Arc<Model>, LoaderOutput), LoaderError> {
    let mut rng = rand::rng();
    let model = Model::new_from_onnx(onnx_data, &mut rng, base_dir)
        .map_err(|e| LoaderError::LoadFailed(e.into()))?;
    let model = Arc::new(model);

    let output = LoaderOutput {
        models: vec![LoadedModel {
            name: model_name.to_string(),
            model: model.clone(),
        }],
        interfaces: vec![],
    };

    Ok((model, output))
}

/// Get the default weight storage strategy for loaders.
pub(super) fn default_storage() -> WeightStorageStrategy {
    WeightStorageStrategy::OriginReference
}

/// Build a SuperGraph for an RNN-style / KV-cache model (processes one token at a time
/// with recurrent state). Used by TransformersLoader, Rwkv7Loader, and GgufLoader.
///
/// `state_pairs` are `(input_name, output_name)` pairs for state tensors.
/// The SuperGraph wraps the model in a Scan loop with RNN cache read/write.
pub(super) fn build_rnn_supergraph(
    tokenizer: TokenizerInfo,
    token_input_name: &str,
    logit_output_name: &str,
    state_pairs: &[(String, String)],
    symbolic_graph: &whisper_tensor::symbolic_graph::SymbolicGraph,
    rng: &mut impl rand::Rng,
) -> TextInferenceTokensInLogitOutInterface {
    let mut super_graph_builder = SuperGraphBuilder::new();
    let token_context_input_link = super_graph_builder.new_tensor_link(rng);
    let model_input_link = super_graph_builder.new_model_link(rng);
    let cache_key = super_graph_builder.new_hash_link(rng);

    let state_ids: Vec<usize> = (0..state_pairs.len()).collect();

    let state_init_links: Vec<(usize, SuperGraphLink)> = state_ids
        .iter()
        .map(|&id| (id, super_graph_builder.new_tensor_link(rng)))
        .collect();

    // Cache read
    let (post_cache_tokens_input, post_cache_state_init_links) = {
        let post_cache_state_init_links: Vec<(usize, SuperGraphLink)> = state_ids
            .iter()
            .map(|&id| (id, super_graph_builder.new_tensor_link(rng)))
            .collect();
        let post_cache_tokens = super_graph_builder.new_tensor_link(rng);
        let node = SuperGraphNodeRNNCacheRead::new(
            cache_key,
            token_context_input_link,
            post_cache_tokens,
            post_cache_state_init_links
                .iter()
                .map(|(id, link)| (id.to_string(), *link))
                .collect(),
            state_init_links
                .iter()
                .map(|(id, link)| (id.to_string(), *link))
                .collect(),
            rng,
        );
        super_graph_builder.add_node(node.to_any());
        (post_cache_tokens, post_cache_state_init_links)
    };

    // Loop count from token sequence length
    let loop_count_link = {
        let loop_count_link = SuperGraphLink::new(SuperGraphLinkKind::Tensor, rng);
        let (mut milli_graph, input_map) =
            MilliOpGraph::new(std::iter::once(post_cache_tokens_input.global_id()), rng);
        let milli_op_graph_input = *input_map.get(&post_cache_tokens_input.global_id()).unwrap();
        let shape_out = Shape::push_new(&mut milli_graph, milli_op_graph_input, rng);
        milli_graph.set_output_map(std::iter::once((shape_out, loop_count_link.global_id())));
        let node = SuperGraphNodeMilliOpGraph::new(milli_graph, rng);
        super_graph_builder.add_node(node.to_any());
        loop_count_link
    };

    // State initialization (zeros matching model input shapes/dtypes)
    {
        let (mut milli_graph, _) = MilliOpGraph::new(std::iter::empty(), rng);
        let mut output_map = HashMap::new();
        let mut output_order = vec![];
        for (id, link) in &state_init_links {
            let input_name = &state_pairs[*id].0;
            let input_tensor_id = *symbolic_graph
                .get_tensors_by_name()
                .get(input_name)
                .expect("state pair input tensor not found in graph");
            let input_tensor_info = symbolic_graph
                .get_tensor_info(input_tensor_id)
                .expect("state pair input tensor has no info");
            let input_tensor_shape: Vec<u64> = input_tensor_info
                .shape
                .clone()
                .expect("state pair input tensor has no shape")
                .iter()
                .map(|x| x.as_numeric().copied().unwrap_or(0))
                .collect();
            let input_tensor_dtype = input_tensor_info
                .dtype
                .expect("state pair input tensor has no dtype");
            let num_elements = input_tensor_shape.iter().product::<u64>();
            let input_tensor = NDArrayNumericTensor::from_vec_shape(
                vec![0.0; num_elements as usize],
                &input_tensor_shape,
            )
            .unwrap()
            .cast(input_tensor_dtype)
            .unwrap();
            let input_tensor_tid = Constant::push_new(&mut milli_graph, input_tensor, rng);
            output_map.insert(input_tensor_tid, link.global_id());
            output_order.push(link.global_id());
        }
        milli_graph.set_output_map_ordered(output_map, output_order);
        let node = SuperGraphNodeMilliOpGraph::new(milli_graph, rng);
        super_graph_builder.add_node(node.to_any());
    }

    // Build scan sub-graph
    let mut sub_builder = SuperGraphBuilder::new();
    let sub_model_input_link = sub_builder.new_model_link(rng);
    let sub_token_input = sub_builder.new_tensor_link(rng);

    let state_input_links: HashMap<usize, SuperGraphLink> = state_ids
        .iter()
        .map(|&id| (id, sub_builder.new_tensor_link(rng)))
        .collect();
    let state_output_links: HashMap<usize, SuperGraphLink> = state_ids
        .iter()
        .map(|&id| (id, sub_builder.new_tensor_link(rng)))
        .collect();
    let final_state_output_links: Vec<(usize, SuperGraphLink)> = state_ids
        .iter()
        .map(|&id| (id, super_graph_builder.new_tensor_link(rng)))
        .collect();

    // Input processing in sub-graph: cast dtype + unsqueeze
    let adjusted_token_context = {
        let input_tensor_id = *symbolic_graph
            .get_tensors_by_name()
            .get(token_input_name)
            .expect("token input tensor not found in graph");
        let input_tensor_info = symbolic_graph
            .get_tensor_info(input_tensor_id)
            .expect("token input tensor has no info");
        let input_tensor_rank = input_tensor_info
            .shape
            .clone()
            .expect("token input tensor has no shape")
            .len();
        let input_tensor_dtype = input_tensor_info
            .dtype
            .expect("token input tensor has no dtype");

        let (mut milli_graph, input_map) =
            MilliOpGraph::new(std::iter::once(sub_token_input.global_id()), rng);
        let milli_op_graph_input = *input_map.get(&sub_token_input.global_id()).unwrap();
        let mut x = Cast::push_new(
            &mut milli_graph,
            milli_op_graph_input,
            input_tensor_dtype,
            rng,
        );
        let zero_tid = Constant::push_new(
            &mut milli_graph,
            NDArrayNumericTensor::from_vec_shape(vec![0i64], &vec![1]).unwrap(),
            rng,
        );
        for _ in 0..input_tensor_rank {
            x = Unsqueeze::push_new(&mut milli_graph, x, zero_tid, rng);
        }

        let processed_input_link = sub_builder.new_tensor_link(rng);
        milli_graph.set_output_map(std::iter::once((x, processed_input_link.global_id())));
        let node = SuperGraphNodeMilliOpGraph::new(milli_graph, rng);
        sub_builder.add_node(node.to_any());
        processed_input_link
    };

    // Model execution in sub-graph
    let sub_logit_output = sub_builder.new_tensor_link(rng);
    {
        let mut tensor_inputs = vec![(adjusted_token_context, token_input_name.to_string())];
        for (id, pair) in state_pairs.iter().enumerate() {
            tensor_inputs.push((*state_input_links.get(&id).unwrap(), pair.0.clone()));
        }
        let mut tensor_outputs = vec![(logit_output_name.to_string(), sub_logit_output)];
        for (id, pair) in state_pairs.iter().enumerate() {
            tensor_outputs.push((pair.1.clone(), *state_output_links.get(&id).unwrap()));
        }
        sub_builder.add_node(
            SuperGraphNodeModelExecution::new(
                rng,
                sub_model_input_link,
                0,
                tensor_inputs,
                tensor_outputs,
            )
            .to_any(),
        );
    }

    // Output processing in sub-graph: squeeze + cast to F32
    let processed_logit_output_link = {
        let output_tensor_id = *symbolic_graph
            .get_tensors_by_name()
            .get(logit_output_name)
            .expect("logit output tensor not found in graph");
        let output_tensor_info = symbolic_graph
            .get_tensor_info(output_tensor_id)
            .expect("logit output tensor has no info");
        let output_tensor_rank = output_tensor_info
            .shape
            .clone()
            .expect("logit output tensor has no shape")
            .len();

        let (mut milli_graph, input_map) =
            MilliOpGraph::new(std::iter::once(sub_logit_output.global_id()), rng);
        let milli_op_graph_input = *input_map.get(&sub_logit_output.global_id()).unwrap();
        let mut x = milli_op_graph_input;
        let zero_tid = Constant::push_new(
            &mut milli_graph,
            NDArrayNumericTensor::from_vec_shape(vec![0i64], &vec![1]).unwrap(),
            rng,
        );
        for _ in 0..(output_tensor_rank - 1) {
            x = Squeeze::push_new(&mut milli_graph, x, zero_tid, rng);
        }
        x = Cast::push_new(&mut milli_graph, x, DType::F32, rng);

        let processed_logit_output_link = sub_builder.new_tensor_link(rng);
        milli_graph.set_output_map(std::iter::once((
            x,
            processed_logit_output_link.global_id(),
        )));
        let node = SuperGraphNodeMilliOpGraph::new(milli_graph, rng);
        sub_builder.add_node(node.to_any());
        processed_logit_output_link
    };

    // Build scan state links
    let state_links: Vec<SuperGraphLinkTriple> = post_cache_state_init_links
        .iter()
        .map(|(id, init_link)| {
            SuperGraphLinkTriple::new(
                *init_link,
                *state_input_links.get(id).unwrap(),
                *state_output_links.get(id).unwrap(),
            )
        })
        .collect();

    let outer_logit_output_link = super_graph_builder.new_tensor_link(rng);

    let mut input_links = vec![sub_model_input_link.to_any(), sub_token_input.to_any()];
    for &id in &state_ids {
        input_links.push(state_input_links.get(&id).unwrap().to_any());
    }
    let mut output_links = vec![processed_logit_output_link.to_any()];
    for id in &state_ids {
        output_links.push(state_output_links.get(id).unwrap().to_any());
    }

    let sub_graph_inner = sub_builder.build(rng, &input_links, &output_links);

    let final_state_outputs: Vec<SuperGraphLinkDouble> = final_state_output_links
        .iter()
        .map(|(id, link)| SuperGraphLinkDouble::new(*state_output_links.get(id).unwrap(), *link))
        .collect();

    let scan_node = SuperGraphNodeScan::new(
        sub_graph_inner,
        loop_count_link,
        vec![SuperGraphLinkDouble::new(
            model_input_link,
            sub_model_input_link,
        )],
        state_links,
        vec![(post_cache_tokens_input, sub_token_input, 0)],
        vec![(processed_logit_output_link, outer_logit_output_link, 0)],
        final_state_outputs.clone(),
        rng,
    );
    super_graph_builder.add_node(scan_node.to_any());

    // Cache write
    {
        let node = SuperGraphNodeRNNCacheWrite::new(
            cache_key,
            token_context_input_link,
            final_state_output_links
                .iter()
                .map(|(id, link)| (id.to_string(), *link))
                .collect(),
            rng,
        );
        super_graph_builder.add_node(node.to_any());
    }

    let super_graph_inputs = vec![
        cache_key.to_any(),
        model_input_link.to_any(),
        token_context_input_link.to_any(),
    ];
    let super_graph_outputs = vec![outer_logit_output_link.to_any()];
    let super_graph = super_graph_builder.build(
        rng,
        super_graph_inputs.as_slice(),
        super_graph_outputs.as_slice(),
    );

    TextInferenceTokensInLogitOutInterface {
        tokenizer,
        model_input_link,
        token_context_input_link,
        logit_output_link: outer_logit_output_link,
        super_graph,
        cache_key_input_link: cache_key,
    }
}
