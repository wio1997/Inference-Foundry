#!/usr/bin/env python3
"""Query existing EP7 CSV around two predetermined draft brackets, no device work."""
import argparse,csv,hashlib,json
from collections import Counter
from decimal import Decimal
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--task-root',required=True);ap.add_argument('--csv',required=True);a=ap.parse_args()
t=Path(a.task_root);csvpath=Path(a.csv).resolve(strict=True);prior=t/'agents/diagnosis/draft-query-metadata-ep7.json'
rank='dp0_pp0_tp7_dcp0_ep7_rank7_4869_20261028020238581_ascend_pt'
if (csvpath.name,csvpath.parent.name,csvpath.parent.parent.name)!=('kernel_details.csv','ASCEND_PROFILER_OUTPUT',rank):raise RuntimeError('unexpected rank/path')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
if sha(csvpath)!='9aa04e4cbbe7edc2160befcec0b133fdaf3b3e2571a1d2f1d7c92c6998b9bdc1':raise RuntimeError('CSV hash differs')
if sha(prior)!='91bd06a12087247de6265ae5c5db915af8f86973d9ff78089b22e973b9545cf0':raise RuntimeError('window evidence differs')
w=[x['window'] for x in json.loads(prior.read_text())['outputs']]
assert len(w)==2 and {x['label'] for x in w}=={'v3_first_unclassified_no_exported_smla','first_strict_recorded_smla'}
lookback_ns=20_000_000
want={'_build_dspark_swa_indices_kernel','copy_and_expand_dflash_and_dspark_inputs_kernel','GreaterEqual','ReduceSum','SparseFlashMlaMetadata'}
found={x['label']:[] for x in w}; builders=[];counts=Counter()
with csvpath.open(newline='') as f:
 for z in csv.DictReader(f):
  if z['Type'] not in want:continue
  start=int(Decimal(z['Start Time(us)'].strip())*1000);end=start+int(Decimal(z['Duration(us)'].strip())*1000)
  rec={'type':z['Type'],'model':z['Model ID'],'stream':z['Stream ID'],'task_id':z['Task ID'],
       'start_ns':start,'end_ns':end,'input_shapes':z['Input Shapes'],'input_dtypes':z['Input Data Types'],
       'output_shapes':z['Output Shapes'],'output_dtypes':z['Output Data Types']}
  counts[z['Type']]+=1
  if z['Type']=='_build_dspark_swa_indices_kernel':
   if len(builders)>=100:raise RuntimeError('builder cap exceeded')
   builders.append(rec)
  for x in w:
   if start<x['block_end_ns'] and end>x['start_ns']-lookback_ns:
    found[x['label']].append(rec)
    if len(found[x['label']])>500:raise RuntimeError('window output cap exceeded')
summary=[]
for x in w:
 rows=sorted(found[x['label']],key=lambda r:r['start_ns']);same_model=[r for r in rows if r['model']==x['model']]
 summary.append({'label':x['label'],'anchor_start_ns':x['start_ns'],'anchor_end_ns':x['block_end_ns'],
   'lookback_ns':lookback_ns,'draft_model':x['model'],'draft_stream':x['stream'],
   'same_model_counts':dict(Counter(r['type'] for r in same_model)),
   'all_in_window_counts':dict(Counter(r['type'] for r in rows)),
   'events':[dict(r,start_rel_ns=r['start_ns']-x['start_ns'],end_rel_ns=r['end_ns']-x['start_ns']) for r in rows]})
print(json.dumps({'scope':'Existing EP7 CSV; exact operator-name records, two preselected draft brackets and preceding 20ms',
 'csv_sha256':sha(csvpath),'prior_sha256':sha(prior),'counts_whole_csv':dict(counts),
 'all_builder_kernel_records':builders,'windows':summary,
 'limitations':['20ms lookback is arbitrary; being inside the window does not establish association with draft step, call, consumer, request or stream readiness.',
 'Same names and shapes do not prove values, tensor storage identity, compute provenance or causal dependency.',
 'Selected type list may miss compiler-fused or alternate implementations; no matching row does not prove operation absent.',
 'Absolute times all from same rank; nonnumeric stream/model fields must remain unattributed.',
 'Only saved profiling data are read; no NPU execution or tensor values.']},ensure_ascii=False,indent=2))
