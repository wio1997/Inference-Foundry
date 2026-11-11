"""Guard FULL replay shapes and observe unchanged SDK math and metadata.

The g61 failure occurred in a padded convolution replay. This bounded
candidate uses FULL only for exact, uniform decode batches; other batches
use the SDK's force_eager path. Both benchmark arms must use this contract.
"""
from collections import Counter


def install_full_graph_observer(config, runner, *, max_requests):
    import torch
    from vllm.compilation.breakable_cudagraph import is_breakable_cudagraph_enabled
    from vllm.forward_context import get_forward_context
    from vllm.config.compilation import CUDAGraphMode
    from vllm_ascend.models.glm5next.kda import Glm5NextLinearAttention
    from vllm_ascend.attention.sfa_v1 import AscendSFAImpl
    if is_breakable_cudagraph_enabled():
        raise ValueError('FULL experiment requires the ordinary ACLGraphWrapper')
    if config.speculative_config is not None or config.kv_transfer_config is not None:
        raise ValueError('non-speculative, resident FULL observation only')
    counts = Counter()
    kda_layers, sfa_layers = set(), set()
    original_kda, original_sfa = Glm5NextLinearAttention._forward, AscendSFAImpl.forward
    import inspect
    original_determine = runner._determine_batch_execution_and_padding
    original_warmup = runner._warmup_and_capture
    signature = inspect.signature(original_determine)
    exact_shapes = frozenset(config.compilation_config.cudagraph_capture_sizes)
    capture_phase = False

    def warmup(*args, **kwargs):
        nonlocal capture_phase
        capture_phase = True
        try:
            return original_warmup(*args, **kwargs)
        finally:
            capture_phase = False

    def determine(*args, **kwargs):
        if capture_phase:
            return original_determine(*args, **kwargs)
        values = signature.bind(*args, **kwargs)
        tokens = values.arguments['num_tokens']
        if tokens <= max_requests:
            requests = values.arguments['num_reqs']
            max_query = values.arguments['max_num_scheduled_tokens']
            done_prefill = (runner.input_batch.num_computed_tokens_cpu[:requests]
                >= runner.input_batch.num_prompt_tokens[:requests]).all()
            if tokens not in exact_shapes or tokens != requests or max_query != 1 or not done_prefill:
                values.arguments['force_eager'] = True
                counts['nonexact_or_prefill_eager_batches'] += 1
        return original_determine(*values.args, **values.kwargs)

    runner._determine_batch_execution_and_padding = determine
    runner._warmup_and_capture = warmup

    def kda(layer, *args, **kwargs):
        context = get_forward_context()
        if torch.npu.is_current_stream_capturing():
            metadata = context.attn_metadata
            item = metadata.get(layer.prefix) if isinstance(metadata, dict) else None
            if (context.cudagraph_runtime_mode != CUDAGraphMode.FULL or item is None
                    or item.num_prefills != 0 or item.num_decodes < 1
                    or item.num_actual_tokens != item.num_decodes
                    or item.num_decodes > max_requests):
                raise RuntimeError('captured KDA did not receive ordinary FULL decode metadata')
            counts['captured_decode_calls'] += 1
            kda_layers.add(layer.prefix)
        else:
            counts['ordinary_kda_calls'] += 1
        return original_kda(layer, *args, **kwargs)

    def sfa(impl, *args, **kwargs):
        if torch.npu.is_current_stream_capturing():
            context = get_forward_context()
            metadata = kwargs.get('attn_metadata', args[3] if len(args) > 3 else None)
            if context.cudagraph_runtime_mode != CUDAGraphMode.FULL or metadata is None:
                raise RuntimeError('captured SFA has no FULL metadata')
            name = kwargs.get('layer_name', args[0] if args else impl.layer_name)
            counts['captured_sfa_calls'] += 1
            sfa_layers.add(name)
        else:
            counts['ordinary_sfa_calls'] += 1
        return original_sfa(impl, *args, **kwargs)

    Glm5NextLinearAttention._forward, AscendSFAImpl.forward = kda, sfa

    class Control:
        closed = False

        def snapshot(self):
            model = runner.model
            return {'closed': self.closed, 'counts': dict(counts),
                    'captured_layers': len(kda_layers), 'captured_sfa_layers': len(sfa_layers),
                    'model_wrapper': type(model).__module__ + '.' + type(model).__qualname__,
                    'original_kda_has_eager_wrapper': hasattr(original_kda, '__wrapped__'),
                    'exact_decode_only': True,
                    'scope': 'SDK FULL with exact decode selection;no metadata or compute patch'}

        def close(self):
            if self.closed:
                return
            if Glm5NextLinearAttention._forward is not kda or AscendSFAImpl.forward is not sfa:
                raise RuntimeError('FULL observer ownership changed')
            if runner._determine_batch_execution_and_padding is not determine or runner._warmup_and_capture is not warmup:
                raise RuntimeError('FULL replay selection ownership changed')
            Glm5NextLinearAttention._forward, AscendSFAImpl.forward = original_kda, original_sfa
            runner._determine_batch_execution_and_padding = original_determine
            runner._warmup_and_capture = original_warmup
            counts['close_calls'] += 1
            self.closed = True

    return Control()
