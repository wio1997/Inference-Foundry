"""Opt-in whole KDA decode capture using the existing SDK implementation.

The root SDK owns graph inputs, metadata buffers and physical state. This only
removes KDA's eager break for non-speculative pure decode without KV transfer.
Prefill and unsupported metadata retain the original decorated implementation.
"""
from collections import Counter


def install_kda_decode_capture(config, runner, *, max_requests):
    import torch
    from vllm.forward_context import get_forward_context
    from vllm_ascend.models.glm5next.kda import Glm5NextLinearAttention
    from vllm_ascend.ops.gdn_attn_builder import AscendGDNAttentionMetadataBuilder
    if config.kv_transfer_config is not None or config.speculative_config is not None:
        raise ValueError('KDA decode capture does not support KV transfer or speculation')
    cls = Glm5NextLinearAttention
    original = cls._forward
    unwrapped = getattr(original, '__wrapped__', None)
    if unwrapped is None:
        raise RuntimeError('expected pinned SDK eager-break wrapper')
    counts = Counter()
    layers = set()
    stable_builders = set()
    stable_pointer_pairs = set()
    observed = set()
    capture_phase = False
    original_metadata = runner._should_build_dummy_attn_metadata
    original_warmup = runner._warmup_and_capture
    original_determine = runner._determine_batch_execution_and_padding
    import inspect
    determine_signature = inspect.signature(original_determine)
    exact_shapes = frozenset(config.compilation_config.cudagraph_capture_sizes)
    original_builder_build = AscendGDNAttentionMetadataBuilder.build

    def build_attention_metadata(builder, *args, **kwargs):
        metadata = original_builder_build(builder, *args, **kwargs)
        pure_decode = (metadata.num_prefills == 0 and metadata.num_decodes > 0
                       and metadata.num_actual_tokens == metadata.num_decodes
                       and metadata.spec_sequence_masks is None
                       and metadata.num_decodes <= max_requests)
        if not pure_decode:
            return metadata
        # The SDK normally materializes these buffers only for FULL graphs.
        # In align mode its block-table selection can instead allocate a new
        # tensor each step. A PIECEWISE graph would then keep the capture-time
        # state IDs. Reuse the SDK's own materialization and attach routines for
        # both capture and replay, without changing its model/state math.
        fields = (
            ('non_spec_state_indices_tensor', metadata.non_spec_state_indices_tensor, max_requests),
            ('non_spec_query_start_loc', metadata.non_spec_query_start_loc, max_requests + 1),
        )
        for name, values, capacity in fields:
            if values is None or values.ndim != 1:
                raise RuntimeError('unexpected pinned SDK decode metadata: ' + name)
            buffer = getattr(builder, name, None)
            if buffer is None:
                setattr(builder, name, values.new_empty((capacity,)))
            elif (buffer.ndim != 1 or buffer.shape[0] < capacity
                  or buffer.dtype != values.dtype or buffer.device != values.device):
                raise RuntimeError('incompatible SDK stable decode buffer: ' + name)
        builder._pad_decode_metadata(metadata, metadata.num_decodes)
        builder._attach_non_spec_decode_metadata(metadata, metadata.non_spec_state_indices_tensor)
        stable_pointer_pairs.add((metadata.non_spec_state_indices_tensor.data_ptr(),
                                  metadata.non_spec_query_start_loc.data_ptr()))
        stable_builders.add(id(builder))
        counts['stable_decode_metadata_builds'] += 1
        return metadata

    AscendGDNAttentionMetadataBuilder.build = build_attention_metadata

    def build_metadata(force_attention=False, is_profile=False, cudagraph_runtime_mode=None):
        if capture_phase and not is_profile:
            counts['metadata_dummy_runs'] += 1
            return True
        return original_metadata(force_attention, is_profile, cudagraph_runtime_mode)

    def warmup_and_capture(desc, cudagraph_runtime_mode, *args, **kwargs):
        nonlocal capture_phase
        from vllm.config.compilation import CUDAGraphMode
        if cudagraph_runtime_mode != CUDAGraphMode.PIECEWISE:
            return original_warmup(desc, cudagraph_runtime_mode, *args, **kwargs)
        if not 1 <= desc.num_tokens <= max_requests:
            raise ValueError('KDA capture requires one-token-per-request dummy batch')
        capture_phase = True
        try:
            # The SDK's documented metadata hook is otherwise false for
            # PIECEWISE. Use its existing builder for warmup AND capture,
            # including state IDs, block tables and persistent device buffers.
            return original_warmup(desc, cudagraph_runtime_mode, *args, **kwargs)
        finally:
            capture_phase = False

    def determine(*args, **kwargs):
        if capture_phase:
            return original_determine(*args, **kwargs)
        values = determine_signature.bind(*args, **kwargs)
        tokens = values.arguments['num_tokens']
        if tokens <= max_requests:
            requests = values.arguments['num_reqs']
            max_query = values.arguments['max_num_scheduled_tokens']
            # PIECEWISE does not provide FULL's padded GDN state buffers.
            # Replay the captured KDA only for exact pure-decode shapes;
            # use the SDK's existing force_eager path for virtual padding
            # and small prefill/mixed batches. No user state is guessed.
            done_prefill = (runner.input_batch.num_computed_tokens_cpu[:requests]
                            >= runner.input_batch.num_prompt_tokens[:requests]).all()
            if tokens not in exact_shapes or tokens != requests or max_query != 1 or not done_prefill:
                values.arguments['force_eager'] = True
                counts['nonexact_or_prefill_eager_batches'] += 1
        return original_determine(*values.args, **values.kwargs)

    runner._determine_batch_execution_and_padding = determine
    runner._should_build_dummy_attn_metadata = build_metadata
    runner._warmup_and_capture = warmup_and_capture

    def forward(layer, *args, **kwargs):
        if torch.npu.is_current_stream_capturing():
            metadata = get_forward_context().attn_metadata
            item = metadata.get(layer.prefix) if isinstance(metadata, dict) else None
            if len(observed) < 16:
                observed.add((item is None, getattr(item, 'num_prefills', None),
                              getattr(item, 'num_decodes', None), getattr(item, 'num_actual_tokens', None)))
            eligible = (item is not None and item.num_prefills == 0
                        and item.num_decodes > 0
                        and item.num_actual_tokens == item.num_decodes
                        and item.spec_sequence_masks is None)
            if eligible:
                pointers = (item.non_spec_state_indices_tensor.data_ptr(),
                            item.non_spec_query_start_loc.data_ptr())
                if (pointers not in stable_pointer_pairs
                        or item.non_spec_decode_metadata.causal_conv1d.cache_indices.data_ptr() != pointers[0]):
                    raise RuntimeError('captured KDA metadata does not use stable SDK buffers')
                counts['stable_metadata_capture_calls'] += 1
                counts['captured_decode_calls'] += 1
                layers.add(layer.prefix)
                return unwrapped(layer, *args, **kwargs)
            counts['retained_eager_breaks'] += 1
        else:
            counts['ordinary_calls'] += 1
        return original(layer, *args, **kwargs)

    cls._forward = forward

    class Control:
        closed = False

        def snapshot(self):
            return {'closed': self.closed, 'counts': dict(counts), 'captured_layers': len(layers),
                    'stable_decode_builders': len(stable_builders),
                    'observed_capture_metadata': [list(v) for v in observed],
                    'scope': 'same SDK KDA math; pure decode graph coverage only;no KV transfer/speculation'}

        def close(self):
            counts['close_calls'] += 1
            if self.closed:
                return
            if cls._forward is not forward:
                raise RuntimeError('KDA capture hook ownership changed')
            cls._forward = original
            if runner._should_build_dummy_attn_metadata is not build_metadata or runner._warmup_and_capture is not warmup_and_capture:
                raise RuntimeError('KDA metadata hook ownership changed')
            runner._should_build_dummy_attn_metadata = original_metadata
            runner._warmup_and_capture = original_warmup
            if runner._determine_batch_execution_and_padding is not determine:
                raise RuntimeError('KDA dispatch hook ownership changed')
            runner._determine_batch_execution_and_padding = original_determine
            if AscendGDNAttentionMetadataBuilder.build is not build_attention_metadata:
                raise RuntimeError('KDA builder hook ownership changed')
            AscendGDNAttentionMetadataBuilder.build = original_builder_build
            self.closed = True

    return Control()
