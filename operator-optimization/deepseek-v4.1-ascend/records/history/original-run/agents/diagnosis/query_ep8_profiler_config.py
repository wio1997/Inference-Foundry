#!/usr/bin/env python3
"""Read selected profiler metadata from one fixed frozen EP8 file; no runtime access."""
import argparse,json,hashlib,re
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--trace-root',required=True);a=ap.parse_args();p=Path(a.trace_root)/'dp1_pp0_tp0_dcp0_ep8_rank0_727_20261028020238580_ascend_pt'/'profiler_info_8.json'; b=p.read_bytes(); h=hashlib.sha256(b).hexdigest()
if len(b)>100_000 or h!='a249b01a52dc6d9cbbd34953f2573b52d1d9f7280ff2d561b228b4a9417eb544':raise RuntimeError('profile info mismatch')
x=json.loads(b); matched=[];allowed={'profiler_level','aic_metrics','ai_core_metrics','cann_version','torch_npu_version','pytorch_version','version','export_type'}
def visit(obj,path,depth=0):
 if depth>8:raise RuntimeError('depth bound')
 if isinstance(obj,dict):
  if len(obj)>100:raise RuntimeError('dictionary bound')
  for k,v in obj.items():
   key=str(k)
   if key.lower() in allowed:
    valid=(v is None or type(v) in (int,float,bool) or (isinstance(v,str) and re.fullmatch(r'[A-Za-z0-9_.+, -]{1,128}',v)))
    matched.append({'path':'.'.join(path+[key]),'value':v if valid else None,'unsupported_type':not bool(valid)})
   if isinstance(v,(dict,list)):visit(v,path+[key],depth+1)
 elif isinstance(obj,list):
  if len(obj)>100:raise RuntimeError('array bound')
  for i,v in enumerate(obj):
   if isinstance(v,(dict,list)):visit(v,path+[str(i)],depth+1)
visit(x,[])
print(json.dumps({'scope':'single EP8 profiler_info_8.json, whitelist only','input_sha256':h,'fields':matched,
 'limits':['session metadata not per replay','version/config metadata does not certify compiled exporter arithmetic or PMU mapping','missing labels only apply to specified exact whitelist'], 'no_device_work':True},indent=2))
