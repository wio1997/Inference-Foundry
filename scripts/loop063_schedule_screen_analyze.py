#!/usr/bin/env python3
"""Validate one frozen 48-request scheduling screen without promoting TPS."""
import json, statistics, sys
from pathlib import Path
root=Path(sys.argv[1])
b=json.loads((root/'bench48.json').read_text())
w=json.loads((root/'warmup48.json').read_text())
rows=[json.loads(p.read_text()) for p in (root/'runtime').glob('rank*_cohort*.json')]
formal=[r for r in rows if 5<=r['cohort']<=8]
latest={c:max((r for r in formal if r['cohort']==c),key=lambda r:r['wall_seconds']) for c in range(5,9)} if len(formal)==32 else {}
valid=(len(rows)==64 and len(formal)==32 and all(r['pass'] and r['target_graph_mode']=='FULL' for r in rows)
       and all(v['summary']['success']==48 and v['summary']['fail']==0 and all(q['output_tokens']==1024 and q['error'] is None for q in v['requests']) for v in (w,b))
       and len(latest)==4)
cycle_count=sum(r['cycles'] for r in latest.values())
runtime_wall=sum(r['wall_seconds'] for r in latest.values())
result={'pass':valid,'rank_rows':len(rows),'formal_rank_rows':len(formal),'formal_cohorts':list(latest),
        'formal_cycles':cycle_count,'formal_latest_rank_runtime_s':runtime_wall,
        'formal_latest_rank_ms_per_cycle':1000*runtime_wall/cycle_count if cycle_count else None,
        'formal_latest_rank_cohort_ms_per_cycle':{str(c):1000*r['wall_seconds']/r['cycles'] for c,r in latest.items()},
        'formal_tps_screen_only':b['summary']['output_tps'],
        'formal_client_wall_s':b['summary'].get('duration_s'),
        'scope':'single warmup+48 screen, not repeated formal E2E decision'}
(root/'screen_summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
if not valid:raise SystemExit(1)
