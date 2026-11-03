#!/usr/bin/env python3
"""Exactly 48 prior layer0 MC2 samples: static exported shapes and launch metadata."""
import argparse,csv,hashlib,json
from collections import Counter,defaultdict
from decimal import Decimal
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--task-root',required=True);ap.add_argument('--trace-root',required=True);a=ap.parse_args()
T=Path(a.task_root);R=Path(a.trace_root);prior=T/'agents/diagnosis/16rank-mc2-other-stream.json'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
if sha(prior)!='d423aa676cf7deb950321d3d32cf77845579589ac7ae4626714a82ec034f40cb':raise RuntimeError('prior mismatch')
x=json.loads(prior.read_text());samples=x['samples'];ranks=defaultdict(list)
for q in samples:ranks[q['rank']].append(q)
if len(samples)!=48 or len(ranks)!=16 or any(len(v)!=3 for v in ranks.values()):raise RuntimeError('sample dimensions changed')
fields=['Input Shapes','Input Data Types','Input Formats','Output Shapes','Output Data Types','Output Formats','Block Num','Mix Block Num','Accelerator Core','OP State'];out=[];issues=[]
for rank,targets in sorted(ranks.items()):
 p=R/rank/'ASCEND_PROFILER_OUTPUT/kernel_details.csv'
 if sha(p)!=x['kernel_sha256_by_rank'][rank]:raise RuntimeError('CSV hash mismatch '+rank)
 found={q['run']:[] for q in targets}
 with p.open(newline='') as f:
  for row in csv.DictReader(f):
   if row['Type']!='DispatchFFNCombine':continue
   st=int(Decimal(row['Start Time(us)'].strip())*1000);en=st+int(Decimal(row['Duration(us)'].strip())*1000)
   for q in targets:
    if st==q['mc2_start_ns'] and en==q['mc2_end_ns'] and row['Model ID']==q['model'] and row['Stream ID']==q['main_stream']:
     found[q['run']].append({'fields':{k:row.get(k) for k in fields},'kernel_name_sha256':hashlib.sha256(row['Name'].encode()).hexdigest(),'task_id':row['Task ID']})
 for q in targets:
  matches=found[q['run']]
  if len(matches)!=1:issues.append({'rank':rank,'run':q['run'],'matches':len(matches)});continue
  out.append({'rank':rank,'dp':rank.split('_')[0],'run':q['run'],'duration_us':q['mc2_duration_ns']/1000,**matches[0]})
groups={}
for dp in ('dp0','dp1'):
 subset=[r for r in out if r['dp']==dp];grouped=Counter(json.dumps({'fields':r['fields'],'kernel_name_sha256':r['kernel_name_sha256']},sort_keys=True,ensure_ascii=False) for r in subset)
 groups[dp]=[{'count':v,'signature':json.loads(k)} for k,v in sorted(grouped.items())]
print(json.dumps({'scope':'48 frozen layer0 same-rank exact MC2 rows; full static shapes, dtypes, formats and launch metadata only','prior_sha256':sha(prior),'samples':out,'signature_counts_by_dp':groups,'issues':issues,'limits':['Signatures are exported physical metadata only; no tensor identity, expert-routing values, valid token masks, capacity/padding contents or weight residency.', 'Equal metadata and forward ordinal do not establish same request, same collective, peer ranks or equal dynamic work.', 'CSV task is fused: no internal dispatch/compute/combine timings and no inference of causality, pure communication duration or speedup.', 'Durations computed within each device, no comparison of cross-rank absolute timestamps; rank samples from one capture not independent runs.', 'No NPU work.']},ensure_ascii=False,indent=2))
