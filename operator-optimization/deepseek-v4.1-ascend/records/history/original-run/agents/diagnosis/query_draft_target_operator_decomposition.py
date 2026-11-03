#!/usr/bin/env python3
"""Decompose frozen positive AIC-cycle proxies by CSV Type; no NPU execution."""
import argparse,csv,hashlib,json
from decimal import Decimal,InvalidOperation
from collections import defaultdict
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--trace-root',required=True);a=ap.parse_args();x=json.loads(FROZEN_INPUT);root=Path(a.trace_root);by=defaultdict(list)
for w in x['windows']:by[w['rank']].append(w)
if len(by)!=15 or len(x['windows'])!=207:raise RuntimeError('scope mismatch')
def decimal(v):
 try:
  x=Decimal(v);return x if x.is_finite() else None
 except (InvalidOperation,TypeError):return None
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
operator=defaultdict(lambda:{'rows':0,'den':Decimal(0),'num':Decimal(0),'skipped':0});windows=[];rank_target=defaultdict(lambda:{'den':Decimal(0),'num':Decimal(0),'skipped':0})
for rank,ws in sorted(by.items()):
 p=root/rank/'ASCEND_PROFILER_OUTPUT/kernel_details.csv'
 if sha(p)!=x['csv_sha256_by_rank'][rank]:raise RuntimeError('CSV changed')
 for key in {(w['model'],w['stream']) for w in ws}:
  group=sorted([w for w in ws if (w['model'],w['stream'])==key],key=lambda w:w['start'])
  if any(a['end']>b['start'] for a,b in zip(group,group[1:])):raise RuntimeError('overlapping matched windows')
 work=[{'den':Decimal(0),'num':Decimal(0),'skipped':0,'positive':0,'mc2':0} for w in ws]
 with p.open(newline='') as f:
  reader=csv.DictReader(f)
  required={'Type','Model ID','Stream ID','Task ID','Start Time(us)','Duration(us)','aic_total_cycles','aic_mac_ratio'}
  if not required.issubset(reader.fieldnames or ()):raise RuntimeError('missing columns')
  for z in reader:
   if 'metadata' in z['Type'].lower():continue
   start=int(Decimal(z['Start Time(us)'].strip())*1000);end=start+int(Decimal(z['Duration(us)'].strip())*1000)
   for i,w in enumerate(ws):
    if z['Model ID']!=w['model'] or z['Stream ID']!=w['stream'] or start<w['start'] or end>w['end']:continue
    name=z['Type'];
    if len(name)>128:raise RuntimeError('unexpected type')
    c=decimal(z['aic_total_cycles']);r=decimal(z['aic_mac_ratio']);g=operator[(w['kind'],name)];q=work[i]
    if c is None or c<=0:q['skipped']+=1;g['skipped']+=1;continue
    if r is None or not Decimal(0)<=r<=Decimal(1):raise RuntimeError('invalid ratio')
    q['num']+=c*r;q['den']+=c;q['positive']+=1;g['num']+=c*r;g['den']+=c;g['rows']+=1
    if name=='DispatchFFNCombine':q['mc2']+=1
 for i,w in enumerate(ws):
  q=work[i]
  if q['mc2']!=1:raise RuntimeError('non-unique MC2 '+rank)
  if w['kind']=='draft':
   if q['num']!=Decimal(w['expected_num']) or q['den']!=Decimal(w['expected_den']) or q['skipped']!=w['expected_skipped']:raise RuntimeError('draft total mismatch '+rank)
  else:
   v=rank_target[rank];v['den']+=q['den'];v['num']+=q['num'];v['skipped']+=q['skipped']
  windows.append({'rank':rank,'kind':w['kind'],'start':w['start'],'end':w['end'],'positive':q['positive'],'skipped':q['skipped'],'den':str(q['den']),'num':str(q['num'])})
for rank,g in rank_target.items():
 p=x['target_expected_by_rank'][rank]
 if g['den']!=Decimal(p['den']) or g['num']!=Decimal(p['num']) or g['skipped']!=p['skipped']:raise RuntimeError('target rank total mismatch '+rank)
result=[]
for (kind,name),v in sorted(operator.items()):
 result.append({'kind':kind,'type':name,'rows':v['rows'],'skipped_no_positive_cycles':v['skipped'],'den':str(v['den']),'num':str(v['num']),'ratio':str(v['num']/v['den']) if v['den'] else None})
print(json.dumps({'scope':'15 ranks; 162 strictly attributed draft windows and 45 target layer0 windows; original v2 main-stream filtering',
 'inputs':x['inputs'],'csv_sha256_by_rank':{k:x['csv_sha256_by_rank'][k] for k in sorted(by)},'windows':windows,'operators':result,
 'limits':['Count duplicate or overlapping source rows exactly as original metric; no independent physical cycle deduplication',
 'ratio is cycle-weighted reported-field proxy, not hardware busy time or end-to-end cost',
 'Type is exporter label, not proven Python-module/tensor identity; excluded metadata and nonpositive/missing cycles are reported',
 'strict162/288 draft sample is selected; target and draft workload/sample counts differ and request pairing is unproven',
 'category weighted shares can explain an arithmetic aggregate, not a causal performance bottleneck'], 'no_device_work':True},ensure_ascii=False,indent=2))
