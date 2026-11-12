"""Count real speculative KDA calls without replacing model computation."""
from collections import Counter
from types import MethodType


def install_mtp_observer(config, runner, *, max_requests):
    from vllm.forward_context import get_forward_context
    from vllm_ascend.models.glm5next.kda import Glm5NextLinearAttention
    spec = config.speculative_config
    if spec is None or spec.num_speculative_tokens not in (1, 5) or config.kv_transfer_config is not None:
        raise ValueError('resident MTP requires a supported explicit draft width')
    original = Glm5NextLinearAttention._forward
    counts = Counter()

    def forward(layer, *args, **kwargs):
        context = get_forward_context()
        metadata = context.attn_metadata
        item = metadata.get(layer.prefix) if isinstance(metadata, dict) else None
        counts['kda_calls'] += 1
        if item is not None and getattr(item, 'num_spec_decodes', 0) > 0:
            counts['speculative_kda_calls'] += 1
        return original(layer, *args, **kwargs)

    import torch
    draft_model = runner.drafter.model
    if callable(getattr(draft_model, 'unwrap', None)):
        draft_model = draft_model.unwrap()
    original_draft = draft_model.forward
    def draft_forward(model, *args, **kwargs):
        counts['captured_mtp_forward_calls' if torch.npu.is_current_stream_capturing()
               else 'ordinary_mtp_forward_calls'] += 1
        return original_draft(*args, **kwargs)
    bound_draft = MethodType(draft_forward, draft_model)
    draft_model.forward = bound_draft
    Glm5NextLinearAttention._forward = forward

    # Observe the exact runnable used by the fixed Ascend proposer. An entry
    # must already own a captured graph and the runtime mode must match; the
    # SDK then takes its existing replay branch. No model tensor is inspected.
    from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
    original_graph_call = ACLGraphWrapper.__call__
    def graph_call(wrapper, *args, **kwargs):
        replay = False
        if wrapper is getattr(runner.drafter, '_runnable', None):
            context = get_forward_context()
            entry = wrapper.concrete_aclgraph_entries.get(context.batch_descriptor)
            replay = (context.cudagraph_runtime_mode == wrapper.runtime_mode
                      and entry is not None and entry.aclgraph is not None)
        result = original_graph_call(wrapper, *args, **kwargs)
        if replay:
            counts['draft_graph_replay_calls'] += 1
        return result
    ACLGraphWrapper.__call__ = graph_call

    class Control:
        closed = False

        def snapshot(self):
            drafter = runner.drafter
            model = getattr(drafter, 'model', None)
            if callable(getattr(model, 'unwrap', None)):
                model = model.unwrap()
            return {'closed': self.closed, 'counts': dict(counts),
                    'method': spec.method, 'num_speculative_tokens': spec.num_speculative_tokens,
                    'draft_enforce_eager': spec.enforce_eager,
                    'draft_model_config_enforce_eager': spec.draft_model_config.enforce_eager,
                    'draft_use_cuda_graph': bool(drafter.use_cuda_graph),
                    'draft_graph_entries': len(getattr(getattr(drafter, '_runnable', None), 'concrete_aclgraph_entries', {})),
                    'draft_dispatch_mode': str(drafter.cudagraph_dispatcher.cudagraph_mode),
                    'norm_contract': getattr(model, '_glm53_mtp_norm_contract', None),
                    'drafter_type': type(drafter).__module__ + '.' + type(drafter).__qualname__,
                    'draft_model_type': type(model).__module__ + '.' + type(model).__qualname__,
                    'scope': 'unscored SDK MTP compatibility;original model and metadata'}

        def close(self):
            if self.closed:
                return
            if Glm5NextLinearAttention._forward is not forward:
                raise RuntimeError('MTP observer ownership changed')
            if draft_model.forward is not bound_draft:
                raise RuntimeError('MTP draft forward ownership changed')
            if ACLGraphWrapper.__call__ is not graph_call:
                raise RuntimeError('MTP graph observer ownership changed')
            ACLGraphWrapper.__call__ = original_graph_call
            draft_model.forward = original_draft
            Glm5NextLinearAttention._forward = original
            counts['close_calls'] += 1
            self.closed = True

    return Control()
