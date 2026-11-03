#!/usr/bin/env python3
"""Audit saved selected-flow exclusion counters; no trace reread or device work."""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json,re
from decimal import Decimal,InvalidOperation
root=Path(__file__).resolve().parents[2]
p=root/'diagnostic-20261029/target-layer-all-kernels-16rank.jsonl'
sha=hashlib.sha256(p.read_bytes()).hexdigest()
assert sha=='0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4'
ranks=[]; totals=Counter(); cats=Counter(); groups=defaultdict(lambda:defaultdict(list))
for line in p.open():
 x=json.loads(line)
 if x.get('kind')!='rank':continue
 d=x['data']; assert d['status']=='completed'
 c=d['counts']; local=Counter()
 for name,v in d['operator_coverage'].items():
  n=v.get('missing_name_or_identity',0)
  if n:
   m=re.fullmatch(r'(hcom_allGather|hcom_alltoall|hcom_reduceScatter)__.*',name)
   local[m.group(1) if m else 'OTHER:'+name]+=n
 assert sum(local.values())==c.get('device_x_missing_name_or_task_identity',0)
 ranks.append({'ep':d['ep'],'categories':dict(local),'start_events':c['target_selected_start_events'], 'start_ids':c['target_selected_start_ids'], 'nonunique_start_ids':c.get('flow_start_nonunique',0),'matched_tasks':c['target_tasks_uniquely_matched']})
 cats.update(local);totals.update(c)
 for layer in d['layers']:
  for group in layer['operator_groups']:
   for task in group['tasks']:
    row=task['csv']; key=(layer['scheduler_digest'],layer['layer'],row['Name'],row['Type'],json.dumps(group['physical_shape_dtype_format'],sort_keys=True),row.get('Accelerator Core'))
    groups[key][d['ep']].append(row)
assert len(ranks)==16 and len({x['ep'] for x in ranks})==16
assert cats==Counter({'hcom_allGather':1280,'hcom_alltoall':1280,'hcom_reduceScatter':1280})
summary_path=root/'diagnostic-20261029/all-kernel-groups-summary.json'
summary_raw=summary_path.read_bytes()
assert hashlib.sha256(summary_raw).hexdigest()=='fa2065112662601df9c73333ea532a2bbcac9ed52b2308933ecce1d268cf417e'
summary=json.loads(summary_raw)
metric_core={'aic':Counter(),'aiv':Counter()}
for key, by_ep in groups.items():
 shape=json.loads(key[4])
 if len(by_ep)!=8 or len({len(v) for v in by_ep.values()})!=1 or any(not isinstance(x,str) or x.strip().lower() in ('','n/a','na','null','none','unknown') for x in shape.values()):continue
 for rows in by_ep.values():
  for row in rows:
   for engine in metric_core:
    try:
     val=Decimal(str(row.get(engine+'_total_cycles')))
     status='missing_or_invalid' if not val.is_finite() else ('positive' if val>0 else 'zero' if val==0 else 'negative')
    except (ValueError,InvalidOperation):status='missing_or_invalid'
    metric_core[engine][str(row.get('Accelerator Core'))+'|'+status]+=1
for engine in metric_core:
 for status in ('positive','zero','missing_or_invalid','negative'):
  observed=sum(v for k,v in metric_core[engine].items() if k.endswith('|'+status))
  lookup={'positive':'cycles_positive','zero':'cycles_zero_reported','missing_or_invalid':'cycles_missing_or_invalid','negative':'cycles_negative'}[status]
  assert observed==summary[engine+'_task_metrics_in_physically_complete_groups'].get(lookup,0)

out={'source':str(p.relative_to(root)),'source_sha256':sha,'ranks':sorted(ranks,key=lambda a:a['ep']),'flow_extractor_counters':dict(sorted(totals.items())), 'missing_identity_selected_flow_by_hcom_name':dict(cats),'metric_cycle_status_by_csv_core_type_in_physically_complete_groups':{k:dict(v) for k,v in metric_core.items()},'physical_group_status':summary['physical_group_status'],'aic_group_exclusions':summary['aic_exclusions'],'aiv_group_exclusions':summary['aiv_exclusions'],'aic_task_metrics_physically_complete_groups':summary['aic_task_metrics_in_physically_complete_groups'],'aiv_task_metrics_physically_complete_groups':summary['aiv_task_metrics_in_physically_complete_groups'], 'limits':['Flow counts and task counts are distinct entities; no assumed inverse flow-to-X uniqueness for excluded records.','Selected flow domain does not census all trace or device tasks.','Names are communication-event names, not proof of internal details, wait attribution or missing physical task count.','Physical-group and metric-exclusion counters refer to grouped accepted tasks and not independent device tasks.']}
print(json.dumps(out,ensure_ascii=False,indent=2))
