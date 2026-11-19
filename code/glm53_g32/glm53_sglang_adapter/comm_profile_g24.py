"""Materialize native communicator resources before the native KV sizing pass."""
import json
import logging
import os
from pathlib import Path

def prepare_before_KV_profile():
    import torch
    import torch.distributed as dist
    from sglang.srt.runtime_context import get_parallel,get_resources
    from glm53_sglang_adapter.comm_prewarm_g22 import prepare_native_LL_resources
    resources=get_resources().buffers
    if resources.get('glm53_communication_budget_g24'):
        return
    p=get_parallel()
    assert p.tp_size==p.moe_ep_size==p.launch_world_size==8 and p.pp_size==1
    assert not torch.npu.is_current_stream_capturing()
    directory=Path(os.environ['GLM53_LOAD_LEDGER_DIR']);directory.mkdir(parents=True,exist_ok=True)
    path=directory/f'comm_budget_g24.pid{os.getpid()}.json'
    report=dict(pid=os.getpid(),device=torch.npu.current_device(),phase='before_KV_profile',groups=[],HCCL_BUFFSIZE=os.environ.get('HCCL_BUFFSIZE'),free_bytes_before=torch.npu.mem_get_info()[0],scope='Live native WORLD/TP/EP and process-wide native LL workspaces charged through free memory before native KV automatic sizing. No KV/request/model state touched.')
    path.write_text(json.dumps(report,indent=2))
    default=torch.npu.default_stream();current=torch.npu.current_stream()
    default.wait_stream(current)
    seen=set()
    with torch.npu.stream(default):
        for label,coordinator in [('WORLD',p.world_group),('TP',p.tp_group),('EP',p.moe_ep_group)]:
            group=coordinator.device_group
            if id(group) in seen:
                report['groups'].append(dict(group=label,aliased=True));continue
            seen.add(id(group));before=torch.npu.mem_get_info()[0]
            value=torch.ones(1,device='npu',dtype=torch.float32)
            dist.all_reduce(value,group=group);torch.npu.synchronize()
            assert value.item()==dist.get_world_size(group)
            # AscendTransferEngine initializes WORLD with all_gather before DEVICE_RDMA.
            values=[torch.empty_like(value) for _ in range(dist.get_world_size(group))]
            dist.all_gather(values,value,group=group);torch.npu.synchronize()
            report['groups'].append(dict(group=label,size=dist.get_world_size(group),free_before=before,free_after=torch.npu.mem_get_info()[0],all_reduce_and_gather_completed=True))
            del value,values
            path.write_text(json.dumps(report,indent=2))
        prepare_native_LL_resources()
    torch.npu.synchronize();current.wait_stream(default)
    torch.npu.empty_cache()
    resources['glm53_communication_budget_g24']=True
    report.update(phase='passed_before_native_KV_profile',free_bytes_after=torch.npu.mem_get_info()[0],native_LL_prewarm=True)
    path.write_text(json.dumps(report,indent=2))
    logging.getLogger(__name__).info('GLM communication-before-KV budget: %s',report)
