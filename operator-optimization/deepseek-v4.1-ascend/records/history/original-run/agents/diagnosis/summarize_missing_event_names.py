#!/usr/bin/env python3
"""Summarize the extractor's saved operator_coverage, without rereading profiler data."""
from pathlib import Path
import collections, hashlib, json

root=Path(__file__).resolve().parents[2]
source=root/'diagnostic-20261029/target-layer-all-kernels-16rank.jsonl'
names=collections.Counter(); ranks=[]; denominator=collections.Counter()
with source.open('rb') as f:
 for line in f:
  item=json.loads(line)
  if item.get('kind')!='rank':continue
  d=item['data']; cc=d['counts']; oc=d['operator_coverage']
  if d['status']!='completed':raise ValueError('incomplete rank')
  local=collections.Counter({name:counts.get('csv_missing',0) for name,counts in oc.items() if counts.get('csv_missing',0)})
  if sum(local.values())!=cc.get('csv_missing',0):raise ValueError('coverage mismatch')
  names.update(local);denominator.update(cc)
  ranks.append({'ep':d['ep'],'csv_missing':sum(local.values()),'by_name':dict(sorted(local.items()))})
if len(ranks)!=16 or len({r['ep'] for r in ranks})!=16:raise ValueError('rank coverage')
sha=hashlib.sha256(source.read_bytes()).hexdigest()
out={'source':str(source.relative_to(root)),'source_sha256':sha,'rank_count':len(ranks),'selected_start_events':denominator['target_selected_start_events'],'csv_missing':sum(names.values()),'csv_missing_by_unique_trace_x_name':dict(sorted(names.items())),'rank_breakdown':sorted(ranks,key=lambda r:r['ep']), 'scope':'Only selected target flow start with globally unique start, one finish, one X at exact finish track/time and required task identity, but no CSV exact key; not complete synchronization census.'}
print(json.dumps(out,indent=2))
