#!/usr/bin/env python3
"""Extract only saved 80 task rows' shape/dtype/layout from frozen EP8 CSV."""
import argparse,csv,hashlib,json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--task-root',required=True);p.add_argument('--trace-root',required=True);p.add_argument('--anchors-json',required=True);a=p.parse_args()
T,R=Path(a.task_root),Path(a.trace_root); anchor_sha='a20fd5761c9ca88b5161ea624c267c248342a21d230bb69a7f83ee589fdc0666'
assert hashlib.sha256(a.anchors_json.encode()).hexdigest()==anchor_sha
x=json.loads(a.anchors_json);rank='dp1_pp0_tp0_dcp0_ep8_rank0_727_20261028020238580_ascend_pt';assert x['rank']==rank
manifest=T/'evidence/TRACE-EXPORT-SHA256.json'
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
assert sha(manifest)=='45ac6c7d22cc20601df2592ce0d78406df24a612f4aa56045566354d9ff46139'
expected={r['rank']:r['kernel_csv_sha256'] for r in json.loads(manifest.read_text())['exports']}
path=R/rank/'ASCEND_PROFILER_OUTPUT/kernel_details.csv';assert path.stat().st_size<=50*1024*1024 and sha(path)==x['csv_sha256']==expected[rank]
targets={}
for g in x['intervals']:
 assert g['ordinal'] in (6,7) and g['gap_index'] in (1,2,3,4) and len(g['events'])==10
 for e in g['events']:
  assert e['stream_category']=='main' and not e['crosses_start'] and not e['crosses_end']
  start=g['start_ns']+e['start_rel_ns'];end=g['start_ns']+e['end_rel_ns'];assert g['start_ns']<=start<end<=g['end_ns']
  key=(e['type'],e['model'],e['stream'],e['task_id'],start,end);assert key not in targets
  targets[key]={'ordinal':g['ordinal'],'gap_index':g['gap_index'],'start_ns':start,'end_ns':end,
                'start_rel_ns':e['start_rel_ns'],'end_rel_ns':e['end_rel_ns']}
assert len(targets)==80
hits=defaultdict(list);fields=['Name','Type','Task ID','Model ID','Stream ID','Input Shapes','Input Data Types','Input Formats','Output Shapes','Output Data Types','Output Formats','Accelerator Core']
with path.open(newline='') as f:
 reader=csv.DictReader(f);assert {'Type','Model ID','Stream ID','Task ID','Start Time(us)','Duration(us)'}.issubset(reader.fieldnames or ())
 missing=[v for v in fields if v not in (reader.fieldnames or ())]
 for z in reader:
  if z['Type'] not in {'Cast','ViewCopy','Slice','GatherV2','MatMulV2','Add'}:continue
  start=int(Decimal(z['Start Time(us)'].strip())*1000);end=start+int(Decimal(z['Duration(us)'].strip())*1000)
  key=(z['Type'],z['Model ID'].strip(),z['Stream ID'].strip(),z['Task ID'].strip(),start,end)
  if key in targets:
   hits[key].append({'anchor':targets[key],'fields':{k:z.get(k) for k in fields},'missing_columns':missing})
rows=[]
for key,target in sorted(targets.items(),key=lambda z:(z[1]['ordinal'],z[1]['gap_index'],z[1]['start_ns'])):
 matches=hits[key];assert len(matches)<=1,'nonunique exact task';rows.append({'type':key[0],'matches':len(matches),'match':matches[0] if matches else None})
assert len(rows)==80
print(json.dumps({'scope':'80 saved EP8 stage242 tasks only; shape/dtype/layout and system names, no token or weight data',
 'rank':rank,'anchors_sha256':anchor_sha,'csv_sha256':expected[rank],'manifest_sha256':sha(manifest),'rows':rows,
 'limits':['Exact CSV row matches do not prove source call, tensor/buffer identity, memory copies, concrete input values, request, model role or data dependencies.',
           'Shape/dtype/format are exported physical metadata; equal shapes do not prove same data or valid elements.',
           'No raw token/prompt/weight values, binary/tiling reads, NPU execution or profiling.']},ensure_ascii=False,indent=2))
