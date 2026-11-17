"""Initialize the existing SGLang DeepEP buffer before legacy AICPU metadata.

G38/G40 reproduce QLI metadata error507018 when private metadata precedes
first DeepEP communication. G42 passes all16 ranks by reversing only that
order. No model forward or serving graph is changed by this startup warmup.
"""
import os
import torch

def initialize_before_private_metadata(model_runner):
    if getattr(model_runner,'is_draft_worker',False):
        return
    from sglang.srt.runtime_context import get_resources
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPDispatcher
    resources=get_resources().buffers
    if resources.get('v41_a3_deepep_initialized_before_metadata',False):
        return
    dispatchers=[]
    for module in model_runner.model.modules():
        owner=getattr(module,'dispatcher',None)
        if isinstance(owner,DeepEPDispatcher):
            dispatchers.append(owner)
        # The image's FusedMoE owns a MaybeTboDeepEPDispatcher wrapper.
        dispatchers.extend(x for x in getattr(owner,'_inners',[]) if isinstance(x,DeepEPDispatcher))
    if not dispatchers:
        raise RuntimeError('V41/A3 startup requires the actual native DeepEPDispatcher')
    impl=dispatchers[0]._low_latency_dispatcher
    H,E,K,C=impl.hidden_size,impl.num_experts,impl.router_topk,impl.num_max_dispatch_tokens_per_rank
    exact_heads = os.environ.get('SGLANG_V41_A3_EXACT_HEADS') == '1'
    expected_capacity = 32 if exact_heads else 16
    warm_tokens = 24 if exact_heads else 12
    assert (H,E,K,C,impl.group.size())==(5120,384,6,expected_capacity,16)
    assert impl.use_fp8 and not impl.use_mxfp8 and not impl.use_mxfp4
    from deep_ep.utils import _resolve_quant_mode
    assert _resolve_quant_mode(True,False,False)=='int8'
    print(f'V41_A3_DEEPEP_STARTUP before_first_private_metadata begin H={H} E={E} K={K} C={C} T={warm_tokens}',flush=True)
    with torch.inference_mode():
        buffer=impl._get_buffer()
        device=torch.device('npu',torch.npu.current_device());rank=impl.group.rank();T=warm_tokens
        x=torch.full((T,H),rank+1,dtype=torch.bfloat16,device=device)
        ids=((torch.arange(T*K,device=device).reshape(T,K)+rank*K)%E).long()
        weights=torch.full((T,K),1/K,dtype=torch.float32,device=device)
        recv,count,handle,event,hook=buffer.low_latency_dispatch(x,ids,C,E,use_fp8=True,
            topk_weights=weights,async_finish=False,return_recv_hook=False)
        assert isinstance(recv,tuple) and recv[0].dtype==torch.int8
        y=(recv[0].float()*recv[1].reshape(-1,1)).to(torch.bfloat16).contiguous()
        combined,event,hook=buffer.low_latency_combine(y,ids,weights,handle,
            async_finish=False,return_recv_hook=False,use_mxfp8=False)
        torch.npu.synchronize()
        assert combined.shape==x.shape and torch.equal(combined,x)
    resources['v41_a3_deepep_initialized_before_metadata']=True
    print('V41_A3_DEEPEP_STARTUP before_first_private_metadata identity_passed actual_native_buffer_reused',flush=True)
