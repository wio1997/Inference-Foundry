"""Candidate: materialize native process-wide LL communication before model capture.

Does not change model math, request state, graph requirement, dispatch strategy,
or weights. Full model acceptance remains required; this is not a proven fix.
"""
import json
import logging
import os
from pathlib import Path

def prepare_native_LL_resources():
    import torch
    from sglang.srt.runtime_context import get_parallel,get_resources
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPBuffer,DeepEPMode
    from sglang.srt.environ import envs
    state=get_resources().buffers
    if state.get('glm53_LL_resource_prewarm_g22'):
        return
    p=get_parallel()
    assert p.moe_ep_size==p.tp_size==8 and p.pp_size==1
    assert not torch.npu.is_current_stream_capturing(), 'Communication prewarm must precede actual capture'
    group=p.moe_ep_group.device_group
    cap=envs.SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get()
    assert cap>=64
    device=torch.npu.current_device();current=torch.npu.current_stream(device);default=torch.npu.default_stream(device)
    default.wait_stream(current)
    report=dict(pid=os.getpid(),physical_device=device,phase='preparing',scope='Synthetic T64 native INT8/BF16 dispatch+identitycombine on default stream, same process-wide AUTO buffer and native EP group; no model KV touched',before_free_bytes=torch.npu.mem_get_info()[0],capacity=cap)
    directory=Path(os.environ['GLM53_LOAD_LEDGER_DIR']);directory.mkdir(parents=True,exist_ok=True)
    path=directory/f'LL_resource_prewarm_g22.pid{os.getpid()}.json'
    path.write_text(json.dumps(report,indent=2))
    with torch.npu.stream(default):
        buffer=DeepEPBuffer.get_deepep_buffer(group,4096,2,DeepEPMode.AUTO,cap,288)
        assert buffer.group is group and buffer.low_latency_strategy.get_name()=='default'
        x=torch.ones((64,4096),dtype=torch.bfloat16,device=f'npu:{device}')
        ids=(torch.arange(64,device=x.device,dtype=torch.int32)[:,None]*8+torch.arange(8,device=x.device,dtype=torch.int32)[None,:]+p.moe_ep_rank*64).remainder(288)
        weights=torch.full((64,8),0.125,dtype=torch.float32,device=x.device)
        for int8_wire in [True,False]:
            recv,counts,handle,event,hook=buffer.low_latency_dispatch(x,ids,cap,288,use_fp8=int8_wire,use_mxfp4=False,use_mxfp8=False,topk_weights=weights,async_finish=False,return_recv_hook=False)
            if int8_wire:
                q,scale=recv;expert=(q.float()*scale.reshape(q.shape[:-1]+(1,))).bfloat16()
            else:
                expert=recv
            output,event,hook=buffer.low_latency_combine(expert,ids,weights,handle,async_finish=False,return_recv_hook=False,use_mxfp8=False)
            torch.npu.synchronize()
            assert output.shape==x.shape and torch.isfinite(output).all().item()
            torch.testing.assert_close(output,x,rtol=.03,atol=.03)
    torch.npu.synchronize();current.wait_stream(default)
    state['glm53_LL_resource_prewarm_g22']=True
    report.update(phase='passed',wire_modes=['native_INT8','BF16'],after_free_bytes=torch.npu.mem_get_info()[0],native_buffer_nvl=buffer.num_nvl_bytes,native_buffer_rdma=buffer.num_rdma_bytes)
    path.write_text(json.dumps(report,indent=2));logging.getLogger(__name__).info('GLM native LL resource prewarm before model capture: %s',report)
