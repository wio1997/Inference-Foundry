#!/usr/bin/env python3
"""Read the version-bound implementation behind exact profiler clock/thread hits."""
import hashlib,json
from pathlib import Path
root=Path('/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/profiler/analysis')
targets=[('prof_view/prof_db_parse/_fwk_api_db_parser.py','8a8c558bee4995efde7c4f91fc4afc7fc03697b8d0d33e970e88b487fd8df39f',[(1,150),(151,310)]),('prof_parse/_cann_file_parser.py','fb47549af5aa1547461884d43150107eeca9131869959495bee8e5403fc9d4a3',[(255,292)]),('prof_common_func/_constant.py','4643479478fb31a8cce3b5838edb4ec61bb159d04846de99ec62f383dede5b31',[(135,160),(350,410),(510,555)])]
out=[]
for rel,expected,ranges in targets:
 p=root/rel;raw=p.read_bytes();digest=hashlib.sha256(raw).hexdigest()
 if digest!=expected:raise RuntimeError('source mismatch '+rel)
 lines=raw.decode().splitlines();out.append({'path':str(p),'sha256':digest,'sections':[{'start':a,'end':min(b,len(lines)),'text':'\n'.join(lines[a-1:b])} for a,b in ranges]})
print(json.dumps({'scope':'Three same-image files, fixed ranges; no runtime work','files':out,'limits':['Installed source does not by itself prove exact producer/version for the frozen export.','No raw requests, no NPU use.']},ensure_ascii=False,indent=2))
