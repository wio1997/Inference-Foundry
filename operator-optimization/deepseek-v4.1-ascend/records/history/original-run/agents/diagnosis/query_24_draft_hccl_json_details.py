#!/usr/bin/env python3
"""Inspect indexed communication JSON entries with fixed stage138 name hashes."""
import argparse,hashlib,json,re
from collections import defaultdict
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--task-root',required=True);ap.add_argument('--trace-root',required=True);a=ap.parse_args()
T=Path(a.task_root);R=Path(a.trace_root);sample=T/'agents/diagnosis/24-draft-alltoall-identity.json';inventory=T/'evidence/profiler-export-inventory.json'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
if sha(sample)!='bc572249e448c5470d3a9c59a08fa6f872c7e9674b78b37e33e926bad1ddf9ef':raise RuntimeError('stage138 hash mismatch')
if sha(inventory)!='ff0f4c8a7e46a48ca53f376dc749406fdced163de9cdca000c7681ff41b592f7':raise RuntimeError('inventory hash mismatch')
d=json.loads(sample.read_text());byrank=defaultdict(list)
for x in d['samples']:byrank[x['rank']].append(x)
if len(d['samples'])!=24 or len(byrank)!=8:raise RuntimeError('unexpected fixed sample set')
expected={r['rank']:next((f['bytes'] for f in r['files'] if f['path']=='ASCEND_PROFILER_OUTPUT/communication.json'),None) for r in json.loads(inventory.read_text())['exports']}
source_files=[('/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/profiler/analysis/prof_view/_communication_parser.py','4e941d9b7c18996687b3fcb2e9a7b46dae9dfb0acc44a06c7474fa9d101766c0'),('/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/profiler/analysis/prof_view/prof_db_parse/_trace_step_time_db_parser.py','7058eeef718b899657cd0738722e46e4d544c2dc9520375f22c21592130c5719')]
source=[];pattern=re.compile(r'HCCL_PATTERN|communication_op|COMMUNICATION_OP|Wait Time|Transit Time|Synchronization Time|Communication Time Info',re.I)
for filename,expected_sha in source_files:
 p=Path(filename)
 if sha(p)!=expected_sha:raise RuntimeError('unexpected source file: '+filename)
 lines=p.read_text().splitlines();hit=[{'line':i,'text':v.strip()[:220]} for i,v in enumerate(lines,1) if pattern.search(v)]
 source.append({'path':filename,'sha256':expected_sha,'matches':hit[:60],'truncated':len(hit)>60})
metric_whitelist={'wait time(ms)','transit time(ms)','synchronization time(ms)','idle time(ms)','wait time ratio','synchronization time ratio','transit size(mb)','transit size(bytes)','bandwidth(gb/s)','size(mb)','large packet ratio','transit time ratio'}
def digest(x):return hashlib.sha256(x.encode()).hexdigest()
def metrics(obj,path='',depth=0):
 if depth>6:return []
 out=[]
 if isinstance(obj,dict):
  for k,v in obj.items():
   if not isinstance(k,str):continue
   if k.lower() in metric_whitelist and type(v) in (int,float):out.append({'metric':k,'value':v})
   if isinstance(v,(dict,list)):out.extend(metrics(v,path,depth+1))
 elif isinstance(obj,list):
  for v in obj[:100]:out.extend(metrics(v,path,depth+1))
 return out
outputs=[];files=[]
for rank,targets in sorted(byrank.items()):
 p=R/rank/'ASCEND_PROFILER_OUTPUT/communication.json';st=p.stat()
 if st.st_size!=expected.get(rank):raise RuntimeError('file size differs from inventory: '+rank)
 raw_sha=sha(p);doc=json.loads(p.read_text());matches={x['op_name_sha256']:[] for x in targets}
 # Do not return key text: compare whole label and the left/right parts around @.
 def walk(node,depth=0):
  if depth>12:return
  if isinstance(node,dict):
   for k,v in node.items():
    if not isinstance(k,str):continue
    left,sep,right=k.partition('@');full=digest(k);lhs=digest(left);rhs=digest(right) if sep else None
    for x in targets:
     if full==x['op_name_sha256'] or lhs==x['op_name_sha256']:
      arr=matches[x['op_name_sha256']]
      if len(arr)<21:arr.append({'full_key_match':full==x['op_name_sha256'],'prefix_name_match':lhs==x['op_name_sha256'],
        'suffix_group_match':rhs==x['group_name_sha256'] if sep else None,'has_group_suffix':bool(sep),
        'numeric_metrics':metrics(v),'node_type':type(v).__name__})
    if isinstance(v,(dict,list)):walk(v,depth+1)
  elif isinstance(node,list):
   for v in node[:2000]:walk(v,depth+1)
 walk(doc)
 if (st.st_size,st.st_mtime_ns,st.st_ino)!=(p.stat().st_size,p.stat().st_mtime_ns,p.stat().st_ino):raise RuntimeError('JSON changed while read')
 files.append({'rank':rank,'size':st.st_size,'sha256':raw_sha,'identity_limit':'first recorded hash; frozen inventory covers size/path only'})
 for x in targets:
  v=matches[x['op_name_sha256']]
  outputs.append({'rank':rank,'opId_local':x['opId_local'],'op_name_sha256':x['op_name_sha256'],
   'group_name_sha256':x['group_name_sha256'],'matching_json_entries':v[:20], 'truncated':len(v)>20})
print(json.dumps({'scope':'8 indexed communication JSONs, 24 fixed rank/op hashes; parser source snippets',
 'sample_sha256':sha(sample),'inventory_sha256':sha(inventory),'source_review':source,'files':files,'samples':outputs,
 'limits':['JSON key name/hash and group suffix matching are candidates only; uniqueness, replay and exporter semantics remain unproven.',
 'Only whitelisted numeric duration/bandwidth labels are output; not all JSON metrics, names, request contents or timestamps are returned.',
 'Metrics may be aggregate, sampled, truncated, overlapping or use different profiler version semantics; no pure-network or causal attribution.',
 'JSON file size/path verified, hash recorded now but not against an earlier frozen content hash.',
 'Recursive array caps and depth caps can omit data; no match is not proof of no communication.']},ensure_ascii=False,indent=2))
