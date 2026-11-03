#!/usr/bin/env python3
"""Read fixed same-image HCCL task selection and timing source ranges."""
import hashlib,json
from pathlib import Path
root=Path('/usr/local/Ascend/cann-9.2.0/tools/profiler/profiler_tool/analysis')
files=[(root/'msparser/cluster/communication_parser.py','481c2634fcf0dfcfe016bb8a2dff69f720c59e409523e5e4dc1de28f33b98a89',[(180,310)]),
 (root/'msmodel/cluster_info/communication_model.py','33066a6591ef6a3f5d9472ad8a512c5936a2231c36c58e35e04d14abb37321e9',[(40,105)]),
 (root/'common_func/ms_constant/str_constant.py','2887ff7c99d3701088f9f51e213bffe65fdb33cc6dbb7a33c7cdee63f7ae3824',[(400,425)])]
out=[]
for p,expected,ranges in files:
 raw=p.read_bytes();actual=hashlib.sha256(raw).hexdigest()
 if actual!=expected:raise RuntimeError('source mismatch: '+str(p))
 lines=raw.decode('utf8').splitlines();out.append({'path':str(p),'sha256':actual,
  'sections':[{'start':a,'end':min(b,len(lines)),'text':'\n'.join(lines[a-1:b])} for a,b in ranges]})
print(json.dumps({'scope':'Three known installed source files, only HCCL task-selection and section functions',
 'files':out,'limits':['This source defines a specific analyzer; its use in the captured JSON requires provenance.',
 'Absence of task rows in exported SQLite is not automatically absence in raw hccl.db.',
 'No runtime work or new profiling.']},ensure_ascii=False,indent=2))
