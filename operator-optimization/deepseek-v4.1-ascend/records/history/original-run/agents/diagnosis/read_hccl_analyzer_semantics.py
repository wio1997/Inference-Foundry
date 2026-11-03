#!/usr/bin/env python3
"""Read four fixed installed CANN sources for HCCL naming and time semantics."""
import hashlib,json,re
from pathlib import Path
root=Path('/usr/local/Ascend/cann-9.2.0/tools/profiler/profiler_tool/analysis')
paths=[root/'analyzer/communication_analyzer.py',root/'msparser/cluster/communication_parser.py',root/'msparser/cluster/meta_parser.py',root/'msmodel/cluster_info/communication_model.py']
expected={str(paths[1]):'481c2634fcf0dfcfe016bb8a2dff69f720c59e409523e5e4dc1de28f33b98a89',str(paths[2]):'d60959f6e2deba9cdd80b433952174bc503dccbce9ebdfe8216e40fe2c53ab66'}
pat=re.compile(r'IDLE_TIME|ELAPSE_TIME|WAIT_TIME|TRANSIT_TIME|event\.group_name|event\.op_name|op_name_dict|group_name|SYNCHRONIZATION_TIME',re.I)
out=[]
for p in paths:
 if not p.is_file():out.append({'path':str(p),'missing':True});continue
 raw=p.read_bytes();digest=hashlib.sha256(raw).hexdigest()
 if str(p) in expected and digest!=expected[str(p)]:raise RuntimeError('source mismatch '+str(p))
 lines=raw.decode('utf8').splitlines();hits=[i for i,s in enumerate(lines) if pat.search(s)]
 chosen=sorted({j for i in hits for j in range(max(0,i-4),min(len(lines),i+5))})
 out.append({'path':str(p),'sha256':digest,'hit_count':len(hits),'lines':[{'n':i+1,'text':lines[i][:240]} for i in chosen[:260]],'truncated':len(chosen)>260})
print(json.dumps({'scope':'Four fixed CANN profiler Python sources, bounded matching lines with context','files':out,
 'limits':['Source is installed image version, but exact runtime exporter selection and binary implementation need independent evidence.',
 'Matching group labels are not per-collective identity; source context must establish uniqueness and reuse.',
 'Reader and aggregator source does not prove raw hardware subtask capture completeness.']},ensure_ascii=False,indent=2))
