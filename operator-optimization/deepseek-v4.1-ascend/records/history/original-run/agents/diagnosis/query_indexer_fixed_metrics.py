#!/usr/bin/env python3
"""Select six exact CSV rows from one frozen profile; no raw PMU replay join."""
import argparse,csv,hashlib,json
from decimal import Decimal
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--trace-root',required=True);a=ap.parse_args();x=json.loads(FROZEN_INPUT);p=Path(a.trace_root)/x['rank']/'ASCEND_PROFILER_OUTPUT/kernel_details.csv';digest=hashlib.sha256(p.read_bytes()).hexdigest()
if digest!=x['csv_sha256']:raise RuntimeError('CSV source hash mismatch')
if len(x['targets'])!=6:raise RuntimeError('unexpected target count')
keys=['aic_total_cycles','aic_mac_ratio','aic_scalar_ratio','aic_mte1_ratio','aic_mte2_ratio','aic_fixpipe_ratio','aic_icache_miss_rate','aiv_total_cycles','aiv_vec_ratio','aiv_scalar_ratio','aiv_mte2_ratio','aiv_mte3_ratio','aiv_icache_miss_rate','cube_utilization(%)']
hits={str(i):[] for i in range(6)}
with p.open(newline='') as f:
 reader=csv.DictReader(f)
 if not {'Type','Model ID','Stream ID','Task ID','Accelerator Core','Start Time(us)','Duration(us)'}.issubset(reader.fieldnames or ()):raise RuntimeError('missing identity columns')
 for z in reader:
  if z['Type']!='VllmAscendQuantLightningIndexerV2':continue
  start=int(Decimal(z['Start Time(us)'].strip())*1000);end=start+int(Decimal(z['Duration(us)'].strip())*1000)
  for i,t in enumerate(x['targets']):
   if (start,end,z['Model ID'].strip(),z['Stream ID'].strip())==(t['start'],t['end'],str(t['model']),str(t['stream'])):
    hits[str(i)].append({'task_id':z['Task ID'].strip(),'core':z['Accelerator Core'].strip(),
      'fields':{k:z.get(k) for k in keys},'missing_columns':[k for k in keys if k not in z]})
rows=[]
for i,t in enumerate(x['targets']):
 a=hits[str(i)]
 if len(a)!=1:raise RuntimeError('target not uniquely matched '+str(t['layer'])+'/'+str(t['run'])+' count '+str(len(a)))
 rows.append({'layer':t['layer'],'run':t['run'],'start_ns':t['start'],'end_ns':t['end'],'duration_us':(t['end']-t['start'])/1000,**a[0]})
print(json.dumps({'rank':x['rank'],'csv_sha256':digest,'prior_sha256':x['prior_sha256'],'samples':rows,
 'limits':['Exact CSV event/time/model/stream match, not independent proof of per-replay hardware counter exporter semantics',
 'No raw PMU timestamp-based join; repeated globalTaskId cannot uniquely identify replay',
 'Reported ratios have finite displayed precision, source version and denominator/aggregation not proven',
 'counter proxies are not FLOPs, HBM bytes, pure waiting, or a causal explanation',
 'same rank/layer across runs does not establish same request, cache or dynamic workload'], 'no_device_work':True},indent=2))
