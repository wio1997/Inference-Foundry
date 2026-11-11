"""Observe the unmodified SDK FULL path; no math or metadata replacement."""
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
                    'scope': 'SDK original FULL execution;observer only;no metadata or compute patch'}

        def close(self):
            if self.closed:
                return
            if Glm5NextLinearAttention._forward is not kda or AscendSFAImpl.forward is not sfa:
                raise RuntimeError('FULL observer ownership changed')
            Glm5NextLinearAttention._forward, AscendSFAImpl.forward = original_kda, original_sfa
            counts['close_calls'] += 1
            self.closed = True

    return Control()
