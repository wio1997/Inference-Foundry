#!/usr/bin/env python3
"""Aggregate previously attributed kernel-type groups, not end-to-end impact."""
import csv,hashlib,json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
root=Path(__file__).resolve().parents[2]
p=root/'captured-layer-attribution/operator_metrics.csv'
totals=defaultdict(lambda:{'calls':0,'sum_us':Decimal(0),'cycles':Decimal(0),'mac_cycles_proxy':Decimal(0)})
with p.open() as f:
 for r in csv.DictReader(f):
  if r['sample_scope']!='INTERIOR_RUNS_1_2_3':continue
  v=totals[r['operator_type']];cy=Decimal(r['aic_reported_cycles'])
  if cy>0 and not r['reported_cycle_weighted_mac_ratio']:raise ValueError('positive cycles without MAC ratio')
  v['calls']+=int(r['calls']);v['sum_us']+=Decimal(r['task_duration_sum_us']);v['cycles']+=cy
  if cy>0:v['mac_cycles_proxy']+=cy*Decimal(r['reported_cycle_weighted_mac_ratio'])
groups=[]
for name,v in totals.items():
 groups.append({'operator_type':name,'calls':v['calls'],'summed_task_duration_us':float(v['sum_us']),
  'mean_task_duration_us':float(v['sum_us']/v['calls']),
  'aic_mac_reported_cycle_weighted_proxy':float(v['mac_cycles_proxy']/v['cycles']) if v['cycles'] else None,
  'aic_cycles':str(v['cycles'])})
groups.sort(key=lambda x:x['summed_task_duration_us'],reverse=True)
out={'input':str(p.relative_to(root)),'input_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
 'scope':'Interior captured target, 16 ranks x 3 runs x 40 layers; attributed task-type rows; shape and layer may differ',
 'groups':groups,'limitations':['Times sum overlapped cross-rank and stream intervals; no end-to-end percentage or observed speedup.',
 'MAC ratio is AIC-cycle-weighted proxy, not full NPU utilization, MFU or a wait fraction. Vector-only operators lack this metric.',
 'Operator types can include different shapes and layers; no unified clock or causal relationship is asserted.',
 'Profiling window and unprofiled benchmark are different observations.']}
path=Path(__file__).with_name('non-mc2-operator-aggregate.json');path.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(groups[:12],indent=2))
