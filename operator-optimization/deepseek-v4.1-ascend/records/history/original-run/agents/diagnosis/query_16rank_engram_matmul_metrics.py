#!/usr/bin/env python3
"""Read 16 saved CSVs for 96 predetermined target BF16 matmul events."""
import argparse,csv,hashlib,json,math,statistics
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--task-root',required=True);ap.add_argument('--trace-root',required=True);a=ap.parse_args()
T=Path(a.task_root);R=Path(a.trace_root);wp=T/'captured-layer-attribution/module-call-metrics.json';ep=T/'agents/diagnosis/engram-boundary-controls.json';mp=T/'evidence/TRACE-EXPORT-SHA256.json'
W=json.loads(wp.read_text());W={(x['rank'],x['run_in_model'],x['layer']):x for x in W if x['run_in_model'] in (1,2,3) and x['layer'] in (0,13)}
S=[x for x in json.loads(ep.read_text())['samples'] if x['after_layer'] in (0,13)];assert len(S)==96
expected={x['rank']:x['kernel_csv_sha256'] for x in json.loads(mp.read_text())['exports']};assert len(expected)==16
targets=defaultdict(dict);issues=[]
for r in S:
 e=[z for z in r['events'] if z['type']=='MatMulV2' and z['input_shapes']=='"3,6144;25600,6144"' and z['input_dtypes']=='DT_BF16;DT_BF16']
 if len(e)!=1:issues.append({'rank':r['rank'],'run':r['run'],'after_layer':r['after_layer'],'reason':'source_event_count','count':len(e)});continue
 w=W[(r['rank'],r['run'],r['after_layer'])];z=e[0];start=w['window_end_ns']+z['event_start_rel_ns'];end=w['window_end_ns']+z['event_end_rel_ns']
 key=(str(z['model']),str(z['stream']),start,end)
 if key in targets[r['rank']]:issues.append({'rank':r['rank'],'reason':'duplicate_target_key'});continue
 targets[r['rank']][key]={'rank':r['rank'],'run':r['run'],'after_layer':r['after_layer'],'model':str(z['model']),'stream':str(z['stream']),
   'start_ns':start,'end_ns':end}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
fields=['Name','Type','Model ID','Task ID','Stream ID','Start Time(us)','Duration(us)','Wait Time(us)','Block Num',
 'Input Shapes','Input Data Types','Output Shapes','Output Data Types','aicore_time(us)','aic_total_cycles',
 'aic_mac_time(us)','aic_mac_ratio','aic_mte1_time(us)','aic_mte1_ratio','aic_mte2_time(us)','aic_mte2_ratio','aic_fixpipe_ratio','cube_utilization(%)']
numeric_fields=('aic_mac_ratio','aic_mte2_ratio','aic_total_cycles')
actual={};samples=[]
for rank in sorted(expected):
 p=R/rank/'ASCEND_PROFILER_OUTPUT/kernel_details.csv'
 try: actual[rank]=sha(p)
 except OSError as ex:issues.append({'rank':rank,'reason':'csv_unreadable','error':str(ex)});continue
 if actual[rank]!=expected[rank]:issues.append({'rank':rank,'reason':'csv_sha256_mismatch'});continue
 found=defaultdict(list)
 with p.open() as f:
  reader=csv.DictReader(f);missing=[k for k in fields if k not in reader.fieldnames]
  if missing:issues.append({'rank':rank,'reason':'missing_columns','columns':missing})
  for z in reader:
   if z.get('Type')!='MatMulV2' or z.get('Input Shapes')!='"3,6144;25600,6144"' or z.get('Input Data Types')!='DT_BF16;DT_BF16':continue
   try:start=int(Decimal(z['Start Time(us)'].strip())*1000);end=start+int(Decimal(z['Duration(us)'].strip())*1000)
   except Exception as ex:issues.append({'rank':rank,'reason':'invalid_time_field','error':str(ex)});continue
   key=(z.get('Model ID'),z.get('Stream ID'),start,end)
   if key in targets[rank]:found[key].append({k:z.get(k) for k in fields})
 for key,t in targets[rank].items():
  rows=found[key]
  if len(rows)!=1:issues.append({'rank':rank,'run':t['run'],'after_layer':t['after_layer'],'reason':'csv_match_count','count':len(rows)});continue
  z=rows[0];metrics={}
  for field in numeric_fields:
   try:
    v=float(z[field]);assert math.isfinite(v)
    if field.endswith('_ratio'):assert 0<=v<=1
    elif v<=0:raise ValueError('nonpositive cycles')
    metrics[field]=v
   except (KeyError,TypeError,ValueError,AssertionError):
    metrics[field]=None;issues.append({'rank':rank,'run':t['run'],'after_layer':t['after_layer'],'reason':'invalid_or_missing_metric','field':field,'raw':z.get(field)})
  samples.append({'target':t,'csv_fields':z,'numeric_metrics':metrics})
assert len(expected)==16
summ=[]
for after in (0,13):
 for run in (1,2,3):
  group=[x for x in samples if x['target']['after_layer']==after and x['target']['run']==run]
  row={'after_layer':after,'run':run,'matched_samples':len(group)}
  for metric in numeric_fields:
   values=[x['numeric_metrics'][metric] for x in group if x['numeric_metrics'][metric] is not None]
   row[metric]={'n':len(values),'min':min(values) if values else None,'median':statistics.median(values) if values else None,'max':max(values) if values else None}
  summ.append(row)
out={'scope':'16 rank x 3 profile target samples x before layers 1 and 14, exact timestamp/model/stream and BF16 shape',
 'status':'COMPLETE' if not issues and len(samples)==96 else 'INCOMPLETE',
 'window_sha256':sha(wp),'interval_sha256':sha(ep),'manifest_sha256':sha(mp),'kernel_sha256_by_rank':actual,
 'issues':issues,'samples':samples,'summary':summ,
 'limits':['Statistics are per task exported ratios for this source-attributed sample, not whole-chip utilization or end-to-end speedup.',
  'MTE2 ratio is not measured HBM bandwidth or proof of saturation; MAC and MTE pipelines can overlap.',
  'Same model/stream/shape/time associates CSV rows to selected event, not tensor or Engram function identity.',
  'Different ranks remain independent clocks; only per-rank durations and distributions are compared.',
  'Missing/mismatched data are reported, never converted to zero.']}
print(json.dumps(out,ensure_ascii=False,indent=2))
