#!/usr/bin/env python3
"""Read-only fixed EP8 trace lookup for layer-0 TensorMove flow; no device execution."""
from pathlib import Path
from decimal import Decimal
import hashlib,json,ijson
p=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces/dp1_pp0_tp0_dcp0_ep8_rank0_737_20261029023525648_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json')
wanted='53ec20cf6d8b8f79b67a7f2f4ddaabf823b4c0305e88e2dbf9cf62d02e7c6f23'
h=hashlib.sha256()
with p.open('rb') as f:
 for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
assert h.hexdigest()==wanted
host=Decimal('1793241327169187.740');dev=Decimal('1793241328465293.999');flow='1793241328465293999'
contains=[];near=[];ends=[];flows=[];count=0
with p.open('rb') as f:
 for e in ijson.items(f,'item'):
  count+=1
  if not isinstance(e,dict): continue
  ts=e.get('ts'); ph=e.get('ph')
  if ts is None:continue
  t=Decimal(str(ts));pid=str(e.get('pid'));tid=str(e.get('tid'))
  if str(e.get('id'))==flow and e.get('name')=='torch_to_npu':flows.append(e)
  if ph=='X' and pid=='737' and tid=='737' and e.get('dur') is not None:
   dur=Decimal(str(e['dur']))
   if t<=host<=t+dur:contains.append(e)
   if abs(t-host)<=Decimal('3'):near.append(e)
  if ph=='X' and pid=='755144' and tid=='47' and t==dev:ends.append(e)
print(json.dumps({'trace_sha256':wanted,'total_events':count,'host_start':str(host),'device_start':str(dev),'flow':flow,'flows':flows,'host_containing':contains,'host_near_3us':near,'device_exact':ends},default=str,indent=2))
