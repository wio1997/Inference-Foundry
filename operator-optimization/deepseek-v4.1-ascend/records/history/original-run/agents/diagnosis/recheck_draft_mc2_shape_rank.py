#!/usr/bin/env python3
"""Offline join of two existing frozen summaries; no device or remote access."""
import collections, hashlib, json, re, statistics
from pathlib import Path
root = Path(__file__).resolve().parents[2]
paths = [root / 'agents/diagnosis/draft-mc2-window-time.json', root / 'agents/diagnosis/16rank-draft-alltoall-distribution.json', root / 'eager-draft-mainstream-attribution-v5/draft_call_metrics.json']
raw = [p.read_bytes() for p in paths]
rows = json.loads(raw[0])['records']; brackets = json.loads(raw[1])['brackets']; v5 = json.loads(raw[2])
ix = collections.defaultdict(list)
for b in brackets: ix[(b['rank'],b['bracket_start_ns'],b['main_stream'])].append(b)
vkeys = collections.Counter((v['rank'],v['model_id'],v['main_stream_id'],v['mc2_task_id'],v['start_ns'],v['end_ns']) for v in v5)
assert len(rows)==len(brackets)==len(v5)==288
assert len({(x['rank'],x['model'],x['stream'],x['task_id']) for x in rows})==288
assert all(vkeys[(x['rank'],x['model'],x['stream'],x['task_id'],x['window_start_ns'],x['window_end_ns'])]==1 for x in rows)
classes = collections.defaultdict(list); rank_m = collections.defaultdict(list)
for x in rows:
    matches=ix[(x['rank'],x['window_start_ns'],x['stream'])]; assert len(matches)==1
    b=matches[0]; m=int(re.match(r'"?(\d+),',b['mc2_input_shape']).group(1)); us=float(x['mc2_duration_ns'])/1000
    classes[(m,x['group'])].append(us); rank_m[(m,x['rank'])].append(us)
report={'sources_sha256':{str(p.relative_to(root)):hashlib.sha256(d).hexdigest() for p,d in zip(paths,raw)},'v5_full_key_unique_matches':288,'bracket_rank_start_stream_unique_matches':288,'groups':[], 'rank_m_counts':{}}
for (m,g),a in sorted(classes.items()):
    report['groups'].append({'physical_input_M':m,'attribution':g,'n':len(a),'min_us':min(a),'median_us':statistics.median(a),'max_us':max(a),'count_ge_1000_us':sum(x>=1000 for x in a)})
for m in sorted({k[0] for k in rank_m}):
    a=[v for (mm,_),v in rank_m.items() if mm==m]
    report['rank_m_counts'][str(m)]={'ranks':len(a),'ranks_with_both_lt_300_and_ge_1000_us':sum(min(v)<300 and max(v)>=1000 for v in a)}
print(json.dumps(report,indent=2,ensure_ascii=False))
