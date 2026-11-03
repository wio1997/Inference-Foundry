#!/usr/bin/env python3
"""Compare seven already-saved candidate PMU/TASK pairs; never access NPU/DB."""
import csv,hashlib,json
from decimal import Decimal,ROUND_DOWN,ROUND_HALF_UP
from pathlib import Path
T=Path(__file__).resolve().parents[2]
paths={'pmu':T/'agents/diagnosis/mc2-pmu-replay-links-ep8.json',
       'csv':T/'mc2-arrival-analysis-v2/rank_target_calls.csv'}
expected={'pmu':'ef5d4b8400d2615de23e57c92b882861d0db78badd40b2cab9c28948e776e966',
          'csv':'19ff5196289b93ee53bcb4a6a27946dbff8c84ebd28988fd40b4fe0bacefbe17'}
for k,p in paths.items():
 if hashlib.sha256(p.read_bytes()).hexdigest()!=expected[k]:raise RuntimeError('input hash mismatch')
j=json.loads(paths['pmu'].read_text(),parse_float=Decimal);groups={}
for row in j['selected_counter_rows']:
 group=groups.setdefault(row['timestampNs'],{})
 if row['counter_name'] in group:raise RuntimeError('duplicate counter name')
 group[row['counter_name']]=Decimal(str(row['value']))
tasks=j['all_TASK_rows_sharing_globalTaskId']
if len(tasks)!=7 or len(groups)!=7:raise RuntimeError('unexpected sample count')
with paths['csv'].open() as f:rows=[r for r in csv.DictReader(f) if '_ep8_' in r['rank'] and r['source_layer']=='0']
seen=set();pairs=[];stats={k:{'total':0,'nearest_matches':0,'truncate_matches':0} for k in ('aic_mac_ratio','aic_mte2_ratio')}
for t in tasks:
 hits=[r for r in rows if int(r['start_ns'])==int(t['startNs']) and int(r['end_ns'])==int(t['endNs']) and int(r['task_id'])==t['taskId'] and int(r['stream_id'])==t['streamId'] and int(r['model_id'])==t['modelId']]
 if len(hits)!=1:raise RuntimeError('TASK to CSV match not unique')
 r=hits[0];end=int(t['endNs']);ds=sorted((abs(ts-end),ts) for ts in groups)
 if len(ds)<2 or ds[0][0]>21 or ds[0][0]==ds[1][0] or ds[0][1] in seen:raise RuntimeError('PMU candidate not unique under chosen timing')
 ts=ds[0][1];seen.add(ts);fields={}
 for key,col in (('aic_mac_ratio','mac_ratio'),('aic_mte2_ratio','mte2_ratio')):
  value=groups[ts][key];observed=Decimal(r[col]);near=value.quantize(Decimal('.001'),rounding=ROUND_HALF_UP);trunc=value.quantize(Decimal('.001'),rounding=ROUND_DOWN)
  stats[key]['total']+=1;stats[key]['nearest_matches']+=int(near==observed);stats[key]['truncate_matches']+=int(trunc==observed)
  if not Decimal(0)<=value<=Decimal(1) or not Decimal(0)<=observed<=Decimal(1):raise RuntimeError('ratio outside [0,1]')
  nearest_lo=max(Decimal(0),observed-Decimal('.0005'));nearest_hi=min(Decimal(1),observed+Decimal('.0005'))
  trunc_lo=observed;trunc_hi=min(Decimal(1),observed+Decimal('.001'))
  fields[key]={'pmu_saved_value':str(value),'csv_value':str(observed),'nearest':str(near),'truncate':str(trunc),
               'nearest_interval':{'lower':str(nearest_lo),'upper':str(nearest_hi),'upper_inclusive':nearest_hi==Decimal(1)},
               'truncate_interval':{'lower':str(trunc_lo),'upper':str(trunc_hi),'upper_inclusive':observed==Decimal(1)},
               'in_nearest_interval':nearest_lo<=value and (value<nearest_hi or nearest_hi==Decimal(1) and value==nearest_hi),
               'in_truncate_interval':trunc_lo<=value and (value<trunc_hi or observed==Decimal(1) and value==trunc_hi)}
 pairs.append({'end_ns':end,'pmu_timestamp_ns':ts,'candidate_delta_ns':ts-end,'fields':fields})
print(json.dumps({'inputs_sha256':expected,'pairs':pairs,'statistics':stats,
 'limitations':['nearest timestamp is a candidate mapping, not a verified replay key',
  'saved decimal presentation of DB floats does not establish physical measurement precision',
  'quantization intervals assume nonnegative ratios in [0,1], nearest half-up or truncation to 0.001; these are not established historical contracts',
  'finite observations do not prove any general rounding rule or cross-version semantics',
  'AIV was not included in the original selected counter query'], 'no_device_work':True},ensure_ascii=False,indent=2))
