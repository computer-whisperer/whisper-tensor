use crate::onnx_graph::WeightStorageStrategy;
use std::collections::HashMap;
use whisper_tensor::interfaces::TextInferenceTokensInLogitOutInterface;
use whisper_tensor::metadata::TokenizerInfo;
use whisper_tensor::milli_graph::MilliOpGraph;
use whisper_tensor::milli_graph::ops::{
    Cast, Concat, Constant, Expand, Shape, SimpleBinary, Slice, Squeeze, Unsqueeze,
};
use whisper_tensor::numeric_dtype::NumericDType;
use whisper_tensor::super_graph::SuperGraphBuilder;
use whisper_tensor::super_graph::links::{
    SuperGraphLink, SuperGraphLinkDouble, SuperGraphLinkKind, SuperGraphLinkTriple,
};
use whisper_tensor::super_graph::nodes::{
    SuperGraphNode, SuperGraphNodeMilliOpGraph, SuperGraphNodeModelExecution,
    SuperGraphNodeRNNCacheRead, SuperGraphNodeRNNCacheWrite, SuperGraphNodeReportProgress,
    SuperGraphNodeScan, SymDimBound, SymbolicInputDims,
};

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
    super_graph_builder.set_link_label(cache_key, "cache_key");
    super_graph_builder.set_link_label(model_input_link, "model_weights");
    super_graph_builder.set_link_label(token_context_input_link, token_input_name);

    // Tokens flow through this supergraph as [batch, seq] per the
    // TextInferenceTokensInLogitOutInterface contract. The scan
    // iterates along axis 1 (seq), and the cache nodes key per row
    // along axis 0.
    const TOKENS_BATCH_AXIS: u32 = 0;
    const TOKENS_SEQ_AXIS: u32 = 1;
    const STATE_BATCH_AXIS: u32 = 0;

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
        let mut node = SuperGraphNodeRNNCacheRead::new(
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
            TOKENS_BATCH_AXIS,
            TOKENS_SEQ_AXIS,
            STATE_BATCH_AXIS,
            rng,
        );
        node.label = Some("cache_read".to_string());
        super_graph_builder.add_node(node.to_any());
        (post_cache_tokens, post_cache_state_init_links)
    };

    // Loop count: Shape(post_cache_tokens) is a rank-1 tensor
    // [batch, seq]. Scan reads iteration count from element 0 of its
    // input, so we slice the seq element out to a [1] tensor.
    let loop_count_link = {
        let loop_count_link = SuperGraphLink::new(SuperGraphLinkKind::Tensor, rng);
        let (mut milli_graph, input_map) =
            MilliOpGraph::new(std::iter::once(post_cache_tokens_input.global_id()), rng);
        let milli_op_graph_input = *input_map.get(&post_cache_tokens_input.global_id()).unwrap();
        let shape_out = Shape::push_new_with_label(
            &mut milli_graph,
            milli_op_graph_input,
            Some("token_shape".to_string()),
            rng,
        );
        let seq_start = Constant::from_vec_with_label(
            &mut milli_graph,
            vec![TOKENS_SEQ_AXIS as i64],
            Some("loop_count.seq_start".to_string()),
            rng,
        );
        let seq_end = Constant::from_vec_with_label(
            &mut milli_graph,
            vec![TOKENS_SEQ_AXIS as i64 + 1],
            Some("loop_count.seq_end".to_string()),
            rng,
        );
        let seq_slice = Slice::push_new_with_label(
            &mut milli_graph,
            shape_out,
            seq_start,
            seq_end,
            None,
            None,
            Some("loop_count.seq_slice".to_string()),
            rng,
        );
        milli_graph.set_output_map(std::iter::once((seq_slice, loop_count_link.global_id())));
        let mut node = SuperGraphNodeMilliOpGraph::new(milli_graph, rng);
        node.label = Some("loop_count_from_shape".to_string());
        super_graph_builder.add_node(node.to_any());
        loop_count_link
    };

    let progress_tier_link = {
        let progress_tier_link = super_graph_builder.new_tensor_link(rng);
        let (mut milli_graph, _) = MilliOpGraph::new(std::iter::empty(), rng);
        let tier_zero = Constant::from_vec_with_label(
            &mut milli_graph,
            vec![0i64],
            Some("progress_tier_zero".to_string()),
            rng,
        );
        milli_graph.set_output_map(std::iter::once((tier_zero, progress_tier_link.global_id())));
        let mut node = SuperGraphNodeMilliOpGraph::new(milli_graph, rng);
        node.label = Some("progress_init".to_string());
        super_graph_builder.add_node(node.to_any());
        progress_tier_link
    };

    // State initialization: runtime-batch-aware zeros.
    //
    // Each state tensor's declared shape is [batch_sym, d1, d2, ...]
    // (batch along STATE_BATCH_AXIS = 0). We can't bake a constant of
    // that shape because batch isn't known at supergraph build time.
    // Instead, read batch from the live tokens tensor, compose the
    // full target shape, and Expand a [1, d1, d2, ...] zero template
    // to it. For batch=1 this degenerates to a one-row state; for
    // batch=N it materializes N stacked zero rows.
    {
        let (mut milli_graph, input_map) =
            MilliOpGraph::new(std::iter::once(token_context_input_link.global_id()), rng);
        let tokens_in = *input_map
            .get(&token_context_input_link.global_id())
            .unwrap();

        let tokens_shape = Shape::push_new_with_label(
            &mut milli_graph,
            tokens_in,
            Some("state_init.tokens_shape".to_string()),
            rng,
        );
        let batch_start = Constant::from_vec_with_label(
            &mut milli_graph,
            vec![TOKENS_BATCH_AXIS as i64],
            Some("state_init.batch_start".to_string()),
            rng,
        );
        let batch_end = Constant::from_vec_with_label(
            &mut milli_graph,
            vec![TOKENS_BATCH_AXIS as i64 + 1],
            Some("state_init.batch_end".to_string()),
            rng,
        );
        // [1]-tensor of dtype i64 carrying the current batch size.
        let batch_piece = Slice::push_new_with_label(
            &mut milli_graph,
            tokens_shape,
            batch_start,
            batch_end,
            None,
            None,
            Some("state_init.batch_piece".to_string()),
            rng,
        );

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
            let declared_shape = input_tensor_info
                .shape
                .clone()
                .expect("state pair input tensor has no shape");
            if declared_shape.is_empty() {
                panic!(
                    "state pair {input_name} has rank 0; expected batch axis at position \
                     {STATE_BATCH_AXIS}"
                );
            }
            // Non-batch dims: concrete → use that size; symbolic → use
            // 0. Zero matches the convention for KV-cache sequence
            // axes (empty cache at init, grows with each scan step)
            // and degenerates to an empty-along-that-axis tensor,
            // which is the correct initial shape for autoregressive
            // models that concat new entries along the symbolic axis.
            let other_dims: Vec<i64> = declared_shape
                .iter()
                .enumerate()
                .filter_map(|(i, s)| {
                    if i == STATE_BATCH_AXIS as usize {
                        None
                    } else {
                        Some(s.as_numeric().copied().unwrap_or(0) as i64)
                    }
                })
                .collect();
            let input_tensor_ndt = input_tensor_info
                .dtype
                .expect("state pair input tensor has no dtype")
                .expect_numeric("state pair input tensor dtype");

            let other_dims_piece = Constant::from_vec_with_label(
                &mut milli_graph,
                other_dims.clone(),
                Some(format!("state_init.other_dims_{}", input_name)),
                rng,
            );
            let target_shape = Concat::push_new_with_label(
                &mut milli_graph,
                vec![batch_piece, other_dims_piece],
                0,
                Some(format!("state_init.target_shape_{}", input_name)),
                rng,
            );

            // Zero template: [1, d1, d2, ...] (batch axis = 1). Expand
            // broadcasts the leading 1 to the runtime batch size.
            let template_shape: Vec<u64> = std::iter::once(1)
                .chain(other_dims.iter().map(|&d| d as u64))
                .collect();
            let zero_template = {
                use whisper_tensor::DynRank;
                use whisper_tensor::numeric_scalar::NumericScalar;
                use whisper_tensor::numeric_tensor::NumericTensor;
                use whisper_tensor::pool::SystemPool;
                use whisper_tensor::symbolic_graph::InlineConstantTensor;
                let tensor = NumericTensor::<DynRank, SystemPool>::from_fn(
                    template_shape,
                    input_tensor_ndt,
                    &SystemPool,
                    |_| NumericScalar::zero(input_tensor_ndt),
                )
                .expect("system pool allocation for state init constant");
                let ict = InlineConstantTensor(std::sync::Arc::new(tensor));
                Constant::push_new_pool(
                    &mut milli_graph,
                    ict,
                    Some(format!("state_init.zero_template_{}", input_name)),
                    rng,
                )
            };
            let state_init_out = Expand::push_new_with_label(
                &mut milli_graph,
                zero_template,
                target_shape,
                Some(format!("state_init.expand_{}", input_name)),
                rng,
            );

            output_map.insert(state_init_out, link.global_id());
            output_order.push(link.global_id());
        }
        milli_graph.set_output_map_ordered(output_map, output_order);
        let mut node = SuperGraphNodeMilliOpGraph::new(milli_graph, rng);
        node.label = Some("state_zero_init".to_string());
        super_graph_builder.add_node(node.to_any());
    }

    // Build scan sub-graph
    let mut sub_builder = SuperGraphBuilder::new();
    let sub_model_input_link = sub_builder.new_model_link(rng);
    let sub_token_input = sub_builder.new_tensor_link(rng);
    let sub_progress_tier = sub_builder.new_tensor_link(rng);
    let sub_total_steps = sub_builder.new_tensor_link(rng);
    let sub_step_in = sub_builder.new_tensor_link(rng);
    let sub_step_out = sub_builder.new_tensor_link(rng);
    sub_builder.set_link_label(sub_model_input_link, "model_weights");
    sub_builder.set_link_label(sub_token_input, token_input_name);
    sub_builder.set_link_label(sub_progress_tier, "progress_tier");
    sub_builder.set_link_label(sub_total_steps, "total_steps");
    sub_builder.set_link_label(sub_step_in, "step_in");
    sub_builder.set_link_label(sub_step_out, "step_out");

    let state_input_links: HashMap<usize, SuperGraphLink> = state_ids
        .iter()
        .map(|&id| (id, sub_builder.new_tensor_link(rng)))
        .collect();
    let state_output_links: HashMap<usize, SuperGraphLink> = state_ids
        .iter()
        .map(|&id| (id, sub_builder.new_tensor_link(rng)))
        .collect();
    for &id in &state_ids {
        let pair = &state_pairs[id];
        sub_builder.set_link_label(
            *state_input_links.get(&id).unwrap(),
            format!("state_in_{}", pair.0),
        );
        sub_builder.set_link_label(
            *state_output_links.get(&id).unwrap(),
            format!("state_out_{}", pair.1),
        );
    }
    let final_state_output_links: Vec<(usize, SuperGraphLink)> = state_ids
        .iter()
        .map(|&id| (id, super_graph_builder.new_tensor_link(rng)))
        .collect();
    for (id, link) in &final_state_output_links {
        super_graph_builder.set_link_label(*link, format!("state_out_{}", state_pairs[*id].1));
    }

    // Input processing in sub-graph: cast dtype + unsqueeze.
    //
    // Per-iter `sub_token_input` is rank-1 [batch] (the seq axis was
    // sliced+squeezed by the scan). Cast to the model's expected
    // dtype, then unsqueeze at axis 1 (input_tensor_rank - 1) times so
    // the model sees [batch, 1, ...] of its declared rank.
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
            .expect("token input tensor has no dtype")
            .expect_numeric("token input tensor dtype");

        let (mut milli_graph, input_map) =
            MilliOpGraph::new(std::iter::once(sub_token_input.global_id()), rng);
        let milli_op_graph_input = *input_map.get(&sub_token_input.global_id()).unwrap();
        let mut x = Cast::push_new_with_label(
            &mut milli_graph,
            milli_op_graph_input,
            input_tensor_dtype,
            Some("token.cast_dtype".to_string()),
            rng,
        );
        // Unsqueeze repeatedly at axis 1 to insert size-1 dims after
        // the batch dim, until we reach the model's declared rank.
        let one_axis = Constant::from_vec_with_label(
            &mut milli_graph,
            vec![1i64],
            Some("token.unsqueeze_axis".to_string()),
            rng,
        );
        for _ in 0..input_tensor_rank.saturating_sub(1) {
            x = Unsqueeze::push_new_with_label(
                &mut milli_graph,
                x,
                one_axis,
                Some("token.add_seq_dim".to_string()),
                rng,
            );
        }

        let processed_input_link = sub_builder.new_tensor_link(rng);
        sub_builder.set_link_label(processed_input_link, "token_preprocessed");
        milli_graph.set_output_map(std::iter::once((x, processed_input_link.global_id())));
        let mut node = SuperGraphNodeMilliOpGraph::new(milli_graph, rng);
        node.label = Some("token_preprocess".to_string());
        sub_builder.add_node(node.to_any());
        processed_input_link
    };

    // Model execution in sub-graph
    let sub_logit_output = sub_builder.new_tensor_link(rng);
    sub_builder.set_link_label(sub_logit_output, "raw_logits");
    {
        let mut tensor_inputs = vec![(adjusted_token_context, token_input_name.to_string())];
        for (id, pair) in state_pairs.iter().enumerate() {
            tensor_inputs.push((*state_input_links.get(&id).unwrap(), pair.0.clone()));
        }
        let mut tensor_outputs = vec![(logit_output_name.to_string(), sub_logit_output)];
        for (id, pair) in state_pairs.iter().enumerate() {
            tensor_outputs.push((pair.1.clone(), *state_output_links.get(&id).unwrap()));
        }
        // Tag every input's batch axis with a shared group so lowering
        // keeps the batch dim symbolic. Tokens arrive as
        // [batch, 1, ...] per-iter (Scan slices axis 1) and states as
        // [batch, ...]; batch is axis 0 for both (see
        // TOKENS_BATCH_AXIS / STATE_BATCH_AXIS above). Sharing group
        // "batch" forces the lowered nano graph to resolve all these
        // dims against one symbolic, exercising sym-dim lowering
        // end-to-end instead of baking the concrete batch value in.
        let mut symbolic_dims = SymbolicInputDims::new();
        symbolic_dims.push(token_input_name, TOKENS_BATCH_AXIS as usize, "batch");
        for pair in state_pairs.iter() {
            symbolic_dims.push(pair.0.clone(), STATE_BATCH_AXIS as usize, "batch");
        }
        // Default compile-time bound for "batch": 2× the runtime size
        // that triggers compilation. A larger batch on a later call
        // forces a cache miss → recompile at the new size. Callers
        // wanting a hard cap can override post-construction with
        // `set_group_bound("batch", SymDimBound::Fixed { max })`.
        symbolic_dims.set_group_bound("batch", SymDimBound::Headroom { factor: 2 });

        let mut node = SuperGraphNodeModelExecution::new(
            rng,
            sub_model_input_link,
            0,
            tensor_inputs,
            tensor_outputs,
        )
        .with_symbolic_input_dims(symbolic_dims);
        node.label = Some("decoder_forward".to_string());
        sub_builder.add_node(node.to_any());
    }

    // Output processing in sub-graph: squeeze the inner seq=1 dims
    // and cast to F32. Per-iter logits start as [batch, 1, ..., vocab]
    // (output_tensor_rank); we squeeze (rank - 2) axes at position 1
    // to leave rank-2 [batch, vocab].
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
        let one_axis = Constant::from_vec_with_label(
            &mut milli_graph,
            vec![1i64],
            Some("logits.squeeze_axis".to_string()),
            rng,
        );
        for _ in 0..output_tensor_rank.saturating_sub(2) {
            x = Squeeze::push_new_with_label(
                &mut milli_graph,
                x,
                one_axis,
                Some("logits.remove_seq_dim".to_string()),
                rng,
            );
        }
        x = Cast::push_new_with_label(
            &mut milli_graph,
            x,
            NumericDType::F32,
            Some("logits.cast_f32".to_string()),
            rng,
        );

        let processed_logit_output_link = sub_builder.new_tensor_link(rng);
        sub_builder.set_link_label(processed_logit_output_link, logit_output_name);
        milli_graph.set_output_map(std::iter::once((
            x,
            processed_logit_output_link.global_id(),
        )));
        let mut node = SuperGraphNodeMilliOpGraph::new(milli_graph, rng);
        node.label = Some("logits_postprocess".to_string());
        sub_builder.add_node(node.to_any());
        processed_logit_output_link
    };

    {
        let (mut milli_graph, input_map) =
            MilliOpGraph::new(std::iter::once(sub_step_in.global_id()), rng);
        let step_in = *input_map.get(&sub_step_in.global_id()).unwrap();
        let one = Constant::from_vec_with_label(
            &mut milli_graph,
            vec![1i64],
            Some("step.one".to_string()),
            rng,
        );
        let step_next = SimpleBinary::add(&mut milli_graph, step_in, one, rng);
        milli_graph.set_output_map(std::iter::once((step_next, sub_step_out.global_id())));
        let mut node = SuperGraphNodeMilliOpGraph::new(milli_graph, rng);
        node.label = Some("step_increment".to_string());
        sub_builder.add_node(node.to_any());
    }

    let mut report =
        SuperGraphNodeReportProgress::new(sub_progress_tier, sub_step_out, sub_total_steps, rng);
    report.label = Some("decode_progress".to_string());
    sub_builder.add_node(report.to_any());

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
        .chain(std::iter::once(SuperGraphLinkTriple::new(
            progress_tier_link,
            sub_step_in,
            sub_step_out,
        )))
        .collect();

    let outer_logit_output_link = super_graph_builder.new_tensor_link(rng);
    super_graph_builder.set_link_label(outer_logit_output_link, logit_output_name);

    let mut input_links = vec![
        sub_model_input_link.to_any(),
        sub_token_input.to_any(),
        sub_progress_tier.to_any(),
        sub_total_steps.to_any(),
        sub_step_in.to_any(),
    ];
    for &id in &state_ids {
        input_links.push(state_input_links.get(&id).unwrap().to_any());
    }
    let mut output_links = vec![processed_logit_output_link.to_any(), sub_step_out.to_any()];
    for id in &state_ids {
        output_links.push(state_output_links.get(id).unwrap().to_any());
    }

    let sub_graph_inner = sub_builder.build(rng, &input_links, &output_links);

    let final_state_outputs: Vec<SuperGraphLinkDouble> = final_state_output_links
        .iter()
        .map(|(id, link)| SuperGraphLinkDouble::new(*state_output_links.get(id).unwrap(), *link))
        .collect();

    let mut scan_node = SuperGraphNodeScan::new(
        sub_graph_inner,
        loop_count_link,
        vec![
            SuperGraphLinkDouble::new(model_input_link, sub_model_input_link),
            SuperGraphLinkDouble::new(progress_tier_link, sub_progress_tier),
            SuperGraphLinkDouble::new(loop_count_link, sub_total_steps),
        ],
        state_links,
        // Slice tokens along the seq axis (axis 1 of [batch, seq])
        // → per-iter [batch] (the scan eval squeezes the sliced axis).
        vec![(post_cache_tokens_input, sub_token_input, TOKENS_SEQ_AXIS)],
        // Per-iter output is [batch, vocab]; insert the seq axis back
        // at position 1 so scan accumulates into [batch, seq, vocab].
        vec![(processed_logit_output_link, outer_logit_output_link, 1)],
        final_state_outputs.clone(),
        rng,
    );
    scan_node.label = Some("token_decode_scan".to_string());
    super_graph_builder.add_node(scan_node.to_any());

    // Cache write — uses the original [batch, seq] tokens for keys.
    {
        let mut node = SuperGraphNodeRNNCacheWrite::new(
            cache_key,
            token_context_input_link,
            final_state_output_links
                .iter()
                .map(|(id, link)| (id.to_string(), *link))
                .collect(),
            TOKENS_BATCH_AXIS,
            TOKENS_SEQ_AXIS,
            STATE_BATCH_AXIS,
            rng,
        );
        node.label = Some("cache_write".to_string());
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
        // Per-row cache + runtime-batch state init + scan along the
        // seq axis lets this supergraph handle arbitrary batch sizes.
        max_batch: u64::MAX,
        max_seq: u64::MAX,
    }
}
