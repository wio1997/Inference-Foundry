#!/usr/bin/env python3
"""Run579 all8 HCCL Graph completion/event gate; no Bound or TPS measurement."""
from __future__ import annotations
import datetime,hashlib,json,os,time
from pathlib import Path
import torch
import torch.distributed as dist
import torch_npu

ROOT=Path('/data/wio/Inference_Foundry')
TAG=os.environ['EXTREME_RUN579_TAG']
assert TAG=='RUN579-EVENT-GATE-A'
OUT=ROOT/'evidence/20260928_loop080_bound/run579/event_gate'
RANK=int(os.environ['RANK'])
LOCAL=int(os.environ['LOCAL_RANK'])
WORLD=int(os.environ['WORLD_SIZE'])
assert WORLD==8 and 0<=RANK<8 and LOCAL==RANK
OUT.mkdir(parents=True,exist_ok=True)
torch.npu.set_device(LOCAL)
dist.init_process_group('hccl',rank=RANK,world_size=WORLD,
                        timeout=datetime.timedelta(seconds=180))
count=12*4096
source=torch.full((count,),RANK+1,dtype=torch.bfloat16,device=f'npu:{LOCAL}')
output=torch.empty((count*WORLD,),dtype=torch.bfloat16,device=f'npu:{LOCAL}')
comm=torch.npu.Stream()
join=torch.npu.Stream()
torch.npu.synchronize()
for _ in range(2):
    dist.all_gather_into_tensor(output,source,group=dist.group.WORLD)
torch.npu.synchronize()
dist.barrier()
torch.npu.synchronize()
graph=torch.npu.NPUGraph()
with torch.npu.graph(graph,stream=comm):
    dist.all_gather_into_tensor(output,source,group=dist.group.WORLD)
torch.npu.synchronize()
rows=[]
for generation in (11,21,31):
    with torch.npu.stream(comm):
        source.fill_(generation+RANK)
        output.fill_(-17)
        graph.replay()
        branch_done=torch.npu.Event(enable_timing=True)
        branch_done.record()
    with torch.npu.stream(join):
        join.wait_event(branch_done)
        expected_blocks=(torch.arange(WORLD,device=f'npu:{LOCAL}',
                            dtype=torch.bfloat16)+generation).repeat_interleave(count)
        mismatch=(output!=expected_blocks).to(torch.int32).sum()
        total=output.float().sum()
        end=torch.npu.Event(enable_timing=True)
        end.record()
    end.synchronize()
    mismatches=int(mismatch.item())
    got=float(total.item())
    expected=float(count*sum(generation+r for r in range(WORLD)))
    if mismatches!=0 or got!=expected:
        raise RuntimeError(f'rank{RANK} generation{generation} stale/reordered output: mismatches={mismatches} sum={got} expected={expected}')
    rows.append({'generation':generation,'mismatches':mismatches,'sum':got,'expected':expected})
    dist.barrier()
(OUT/f'rank{RANK}.json').write_text(json.dumps({
    'status':'fresh_hccl_graph_join_pass','run_tag':TAG,'rank':RANK,'world':WORLD,
    'collective':'BF16 TP8 AllGather [12,4096] -> [96,4096]',
    'input_bytes_per_rank':count*2,'output_bytes_per_rank':count*WORLD*2,
    'graph_capture_stream':'private comm','consumer_stream':'private join',
    'consumer_depends_on':'join.wait_event(branch_done) then device sum then end.synchronize',
    'generations':rows,'not_timed':True,
    'strict_bound_endpoint':None},indent=2)+'\n')
dist.barrier()
dist.destroy_process_group()
