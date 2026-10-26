#!/usr/bin/env python3
"""Compare verification-free B with contemporary A0 using whole-client and latest-rank windows."""
import json,sys
from pathlib import Path
base=Path(sys.argv[1]);candidate=Path(sys.argv[2]);out=Path(sys.argv[3])
a=json.loads((base/'screen_summary.json').read_text());b=json.loads((candidate/'screen_summary.json').read_text())
if not (a['pass'] and b['pass']):raise SystemExit('invalid screen')
cohorts={c:{'A0_ms_per_cycle':a['formal_latest_rank_cohort_ms_per_cycle'][c],
            'B_ms_per_cycle':b['formal_latest_rank_cohort_ms_per_cycle'][c],
            'delta_ms_per_cycle':b['formal_latest_rank_cohort_ms_per_cycle'][c]-a['formal_latest_rank_cohort_ms_per_cycle'][c]}
         for c in map(str,range(5,9))}
r={'scope':'one warmed 48-request pass per mode; diagnostic screening, not repeated formal KEEP',
   'A0':str(base),'B':str(candidate),'cohorts':cohorts,
   'delta_latest_rank_runtime_s':b['formal_latest_rank_runtime_s']-a['formal_latest_rank_runtime_s'],
   'delta_cycles':b['formal_cycles']-a['formal_cycles'],
   'delta_normalized_ms_per_cycle':b['formal_latest_rank_ms_per_cycle']-a['formal_latest_rank_ms_per_cycle'],
   'delta_client_wall_s':b['formal_client_wall_s']-a['formal_client_wall_s'],
   'delta_tps':b['formal_tps_screen_only']-a['formal_tps_screen_only'],
   'delta_client_minus_runtime_s':(b['formal_client_wall_s']-b['formal_latest_rank_runtime_s'])-(a['formal_client_wall_s']-a['formal_latest_rank_runtime_s']),
   'interpretation':'Cohort paths are separate stochastic trajectories; normalized runtime, cycles and residual must be read together. No ceiling or product KEEP from one pass.'}
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
