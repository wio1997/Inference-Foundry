#!/usr/bin/env python3
"""No-service TP8 collective capacity diagnostic for the Run152 payload pair."""
import json, os, statistics, time
import torch
import torch.distributed as dist
import torch_npu

rank=int(os.environ['RANK']); local=int(os.environ['LOCAL_RANK']); world=int(os.environ['WORLD_SIZE'])
assert world==8
torch.npu.set_device(local)
dist.init_process_group('hccl',rank=rank,world_size=world)
bf=torch.zeros(49152,dtype=torch.bfloat16,device=f'npu:{local}')
fp=torch.zeros(3072,dtype=torch.float32,device=f'npu:{local}')
out_bf=torch.empty(world*bf.numel(),dtype=bf.dtype,device=bf.device)
out_fp=torch.empty(world*fp.numel(),dtype=fp.dtype,device=fp.device)
def chain(n):
    for _ in range(n):
        dist.all_gather_into_tensor(out_bf,bf)
        dist.all_gather_into_tensor(out_fp,fp)
for _ in range(8): chain(40)
torch.npu.synchronize(); dist.barrier(); torch.npu.synchronize()
samples=[]
for _ in range(30):
    start=torch.npu.Event(enable_timing=True); end=torch.npu.Event(enable_timing=True)
    start.record(); chain(40); end.record(); end.synchronize()
    samples.append(start.elapsed_time(end))
rows=[None]*world
dist.all_gather_object(rows,dict(rank=rank,samples_ms=samples,median_ms=statistics.median(samples),
                                 min_ms=min(samples),max_ms=max(samples)))
if rank==0:
    latest=[max(row['samples_ms'][i] for row in rows) for i in range(len(samples))]
    print(json.dumps(dict(world=world,chain_pairs=40,bf16_elements=49152,fp32_elements=3072,
                          repeats=30,ranks=rows,latest_rank_chain_ms=latest,
                          latest_rank_median_ms=statistics.median(latest),
                          per_pair_latest_median_ms=statistics.median(latest)/40,
                          note='Synthetic zeros, no interleaved compute, no per-call barrier; capacity diagnostic only'),indent=2),flush=True)
dist.destroy_process_group()
