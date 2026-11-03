#!/usr/bin/env python3
"""Read one existing EP9 CANN analyze export, compare one fixed recorded interval."""
from collections import Counter
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
import json
p=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces/dp1_pp0_tp1_dcp0_ep9_rank1_752_20261029023525649_ascend_pt/PROF_000001_20261029023525664_00000752DCCBBRNB/analyze/communication.json')
b=p.read_bytes();x=json.loads(b,parse_float=Decimal); assert isinstance(x,dict)
a=Decimal('1793241333127617.287');z=Decimal('1793241333401182.441'); valid=invalid=0;intersections=[];named=[];types=Counter();before=[];after=[]
for name,val in x.items():
 assert isinstance(val,dict)
 types[name.split('__')[0]]+=1
 if any(w in name.lower() for w in ('dispatchffncombine','mc2','expert_token_nums')):named.append(name)
 t=val.get('Communication Time Info',{});s=t.get('Start Timestamp(us)');d=t.get('Elapse Time(ms)')
 try:
  s=Decimal(s);d=Decimal(d);assert s.is_finite() and d.is_finite() and d>=0
 except Exception:
  invalid+=1;continue
 valid+=1;end=s+d*1000
 if s<z and end>a:intersections.append({'name':name,'start_us':str(s),'duration_ms':str(d),'overlap_us':str(min(end,z)-max(s,a))})
 if end<=a:before.append((a-end,name))
 if s>=z:after.append((s-z,name))
print(json.dumps({'input_sha256_observed':sha256(b).hexdigest(),'ep':9,'source':'existing PROF analyze/communication.json, not a new export','target':{'name':'DispatchFFNCombine','digest_prefix':'8c18c293','layer':0,'stream':47,'task':46755,'globalTaskId':4709,'device_start_us':str(a),'device_end_us':str(z)},'entries':len(x),'valid_time_entries':valid,'invalid_time_entries':invalid,'name_prefix_counts':dict(types),'names_containing_target_or_expert':named,'recorded_interval_intersections':intersections,'nearest_before_gap_us':str(min(before)[0]) if before else None,'nearest_after_gap_us':str(min(after)[0]) if after else None,'limitations':['comparison is numeric exported timestamps; clock alignment and producer semantics not independently bound','absence of intersection does not prove no HcclShmem/internal transfer, event, other track or unexported work','source of analyzer JSON and original instrumentation/version/coverage not proven by file existence','cannot infer waiting, network activity, causal bottleneck, or output performance']},indent=2,ensure_ascii=False))
