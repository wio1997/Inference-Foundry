#!/usr/bin/env python3
"""Recompute local-time lead intervals by DP and sample ordinal; no cross-device subtraction."""
import collections,csv,hashlib,json,statistics
from pathlib import Path
T=Path(__file__).resolve().parents[2];p=T/'mc2-preceding-analysis/rank_layer0_timeline.csv';raw=p.read_bytes()
if hashlib.sha256(raw).hexdigest()!='d3e0d9cbeddcde97df34dcf36b386cc8201b71c82dde8ab4ed42dd713fde7dfb':raise RuntimeError('input mismatch')
groups=collections.defaultdict(list)
with p.open(newline='') as f:
 for row in csv.DictReader(f):
  if row['interior_runs_1_2_3']=='True':groups[(row['dp'],int(row['captured_run_in_model']))].append(row)
if set(groups)!={(dp,run) for dp in ('dp0','dp1') for run in (1,2,3)} or any(len(rows)!=8 or len({r['rank'] for r in rows})!=8 for rows in groups.values()):raise RuntimeError('group dimensions changed')
fields=('layer_to_mc2_us','attention_duration_us','ffn_pre_to_mc2_us','mc2_duration_us')
out=[]
for (dp,run),rows in sorted(groups.items()):
 out.append({'dp':dp,'run':run,'n':len(rows),'medians_us':{k:statistics.median(float(r[k]) for r in rows) for k in fields}})
print(json.dumps({'scope':'three local lead intervals and MC2 durations, each already computed on its own rank; DP per-run medians','input_sha256':hashlib.sha256(raw).hexdigest(),'groups':out,'limits':['Sample ordinal is not a proven cross-rank request or collective ID.', 'No cross-rank absolute timestamp subtraction, device frequency error not calibrated.', 'Comparable local interval durations do not exclude different arrival times or preparation outside layer0.', 'Attention/FFN durations are local attributed intervals, not proof of identical tensor values, actual token counts or dependencies.']},indent=2))
