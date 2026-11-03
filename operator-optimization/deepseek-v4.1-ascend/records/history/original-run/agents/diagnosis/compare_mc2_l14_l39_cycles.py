#!/usr/bin/env python3
"""Read-only arithmetic on previously captured per-rank MC2 task records."""
from decimal import Decimal as D
from hashlib import sha256
from pathlib import Path
import json
R=Path(__file__).resolve().parents[2]
SRC=R/'diagnostic-20261029/target-layer-mc2-16rank.json'
OUT=R/'agents/diagnosis/mc2-l14-l39-cycles-comparison.json'
DIGEST='1b826883b782ba77814d5bd40fcd3e46c759e380069db2e066ff0e9cb4bce89d'
raw=SRC.read_bytes(); doc=json.loads(raw); pairs=[]
keys=('Input Shapes','Output Shapes','Input Data Types','Output Data Types','Input Formats','Output Formats','Accelerator Core','Block Num','Mix Block Num')
for rank in doc['ranks']:
 if rank['ep'] not in range(8,16):continue
 found={}
 for s in rank['layers']:
  if s['scheduler_digest']==DIGEST and s['layer'] in (14,39):
   assert s['layer'] not in found and s['matched_task_count']==1 and len(s['tasks'])==1
   found[s['layer']]=s
 assert set(found)=={14,39}
 a,b=[found[i]['tasks'][0]['csv'] for i in (14,39)]
 assert all(a[k]==b[k] for k in keys)
 def read(x):
  c=D(x['aic_total_cycles']);ratio=D(x['aic_mac_ratio']);t=D(x['Duration(us)'])
  assert c>0 and ratio>0 and t>0
  return c,ratio,t
 ac,ar,at=read(a);bc,br,bt=read(b)
 # Sensitivity example only: rounding to nearest 0.001 is NOT established.
 eps=D('0.0005'); assert br>eps and ar>=eps
 low=ac*(ar-eps)/(bc*(br+eps)); high=ac*(ar+eps)/(bc*(br-eps))
 pairs.append({'ep':rank['ep'],'scope_l14':found[14]['layer_scope_id'], 'scope_l39':found[39]['layer_scope_id'],
  'flow_l14':found[14]['tasks'][0]['flow_id'],'flow_l39':found[39]['tasks'][0]['flow_id'],
  'l14':{'duration_us':str(at),'aic_total_cycles':str(ac),'displayed_mac_ratio':str(ar),'cycles_times_displayed_ratio':str(ac*ar)},
  'l39':{'duration_us':str(bt),'aic_total_cycles':str(bc),'displayed_mac_ratio':str(br),'cycles_times_displayed_ratio':str(bc*br)},
  'l14_over_l39':{'duration':str(at/bt),'aic_total_cycles':str(ac/bc),'displayed_mac_ratio':str(ar/br),
      'cycles_times_displayed_ratio':str((ac*ar)/(bc*br)),
      'conditional_if_nearest_0p001_ratio_rounding_and_exact_cycles_only':{'lower':str(low),'upper':str(high)}}})
assert len(pairs)==8 and {x['ep'] for x in pairs}==set(range(8,16))
out={'input_sha256':sha256(raw).hexdigest(),'script_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),
 'input':str(SRC.relative_to(R)),'digest':DIGEST,'matching':list(keys),
 'warnings':['same recorded shape is not same effective work, expert assignment, mask, bytes or kernel internal state',
             'cycle times displayed ratio is arithmetic proxy, not operations or verified physical busy cycles',
             'rounded-ratio sensitivity assumes nearest 0.001; actual export rounding and cycle semantics not established',
             'individual rank tasks must not be summed as wall-clock and profiled timing is perturbed'],
 'pairs':sorted(pairs,key=lambda x:x['ep'])}
OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
for p in out['pairs']:
 q=p['l14_over_l39']; f=lambda s: f'{float(s):.2f}'
 print(p['ep'],f(q['duration']),f(q['aic_total_cycles']),f(q['displayed_mac_ratio']),f(q['cycles_times_displayed_ratio']),f(q['conditional_if_nearest_0p001_ratio_rounding_and_exact_cycles_only']['lower']),f(q['conditional_if_nearest_0p001_ratio_rounding_and_exact_cycles_only']['upper']))
