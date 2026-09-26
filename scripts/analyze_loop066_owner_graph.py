#!/usr/bin/env python3
"""Paired all-rank private Graph A96/B16/A96 replay analysis."""
import argparse
import json
import statistics
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--graph-dir',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
rows=[]
for rank in range(8):
 d=json.loads((a.graph_dir/f'rank{rank}.json').read_text())
 if not d['pass'] or d['stage']!='complete':raise SystemExit(f'rank{rank} gate failed')
 pairs=[]
 for triplet in range(3,13):
  x={v['arm']:v for v in d['samples'] if v['triplet']==triplet}
  if set(x)!={'A','B','A2'}:raise SystemExit(f'rank{rank} triplet{triplet} malformed')
  aa,b,a2=(x[k]['replay_ms'] for k in ('A','B','A2'))
  ctrl=(aa+a2)/2
  pairs.append({'triplet':triplet,'A_ms':aa,'B_ms':b,'A2_ms':a2,
                'full_control_mean_ms':ctrl,'B_minus_full_ms':b-ctrl,
                'A_A2_abs_drift_ms':abs(aa-a2),'B_faster_than_both':b<min(aa,a2)})
 rows.append({'rank':rank,'captured_kv_shapes':d['captured_kv_shapes'],
              'median_full_control_ms':statistics.median(v['full_control_mean_ms'] for v in pairs),
              'median_B_ms':statistics.median(v['B_ms'] for v in pairs),
              'median_paired_B_minus_full_ms':statistics.median(v['B_minus_full_ms'] for v in pairs),
              'median_A_A2_abs_drift_ms':statistics.median(v['A_A2_abs_drift_ms'] for v in pairs),
              'B_faster_than_both_count':sum(v['B_faster_than_both'] for v in pairs),
              'pairs':pairs})
all_pairs=[v for r in rows for v in r['pairs']]
aggregate={'ranks':8,'pairs':len(all_pairs),
 'pooled_paired_B_minus_full_median_ms':statistics.median(v['B_minus_full_ms'] for v in all_pairs),
 'rank_median_B_minus_full_median_ms':statistics.median(r['median_paired_B_minus_full_ms'] for r in rows),
 'rank_medians_B_faster_count':sum(r['median_paired_B_minus_full_ms']<0 for r in rows),
 'B_faster_than_both_pairs':sum(v['B_faster_than_both'] for v in all_pairs),
 'pooled_A_A2_abs_drift_median_ms':statistics.median(v['A_A2_abs_drift_ms'] for v in all_pairs)}
out={'status':'valid_private_graph_replay','scope':'same-prestate one-layer private Graph, not full Target/Product',
     'aggregate':aggregate,'ranks':rows}
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(aggregate,indent=2))
