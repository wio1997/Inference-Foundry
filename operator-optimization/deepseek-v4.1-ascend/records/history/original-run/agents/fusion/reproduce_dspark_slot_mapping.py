#!/usr/bin/env python3
"""Model the frozen 1-D DSpark cache-slot conversion for small integer examples."""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parents[2]
p='sources/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/dspark.py'
h=hashlib.sha256((root/p).read_bytes()).hexdigest()
block=128
slots=[0,127,128,-1,-25,4097]
def map_slot(x):
 if x<0:return [-1,-1]
 x=max(x,0);return [x//block,x%block]
result=[map_slot(x) for x in slots]
assert result==[[0,0],[0,127],[1,0],[-1,-1],[-1,-1],[32,1]]
print(json.dumps({'source_sha256':h,'example_block_size':block,'input_physical_slots':slots,
                  'two_dimensional_slots':result,'limitations':['Small examples only: integer conversion overflow, source tensor dtype/device, alias and kernel execution are not verified.',
                 'Negative slots mean skipped cache writes only under the cited scatter_cache_sk contract.',
                 'Examples do not prove actual slots in the captured 169 execution.'],'no_device_work':True},indent=2))
