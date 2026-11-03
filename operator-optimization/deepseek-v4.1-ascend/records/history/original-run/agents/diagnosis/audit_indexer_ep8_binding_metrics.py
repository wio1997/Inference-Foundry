#!/usr/bin/env python3
"""Check fixed historical metadata/Indexer association IDs and aggregate metric labels only."""
import pathlib,sqlite3,json,re,hashlib
base=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime')
db=base/'diagnostic-traces/dp1_pp0_tp0_dcp0_ep8_rank0_727_20261028020238580_ascend_pt/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_8.db';h=hashlib.sha256()
with db.open('rb') as f:
 for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
assert h.hexdigest()=='67d14f7f386658f8c8c5d3c9059dff9b0f1decb200eba621a3432fcf1e23de58'
c=sqlite3.connect(db.as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row;c.execute('PRAGMA automatic_index=OFF')
ids=[78002,78017,158316,158331,164003,164018,184330]
links={str(i):[dict(r) for r in c.execute('SELECT id,connectionId FROM CONNECTION_IDS WHERE connectionId=?',(i,))] for i in ids}
results={}
for name in ['measure1-metrics-before.txt','measure1-metrics-after.txt']:
 p=base/'profile-client-results'/name; raw=p.read_bytes();entries={}
 for line in raw.decode(errors='replace').splitlines():
  m=re.match(r'^(vllm:(?:iteration_tokens_total|num_requests_running|request_prompt_tokens|prompt_tokens_total)(?:_(?:bucket|sum|count))?)(?:\{([^}]*)\})?\s',line)
  if not m:continue
  key=m.group(1);labels=re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=',m.group(2) or '')
  entry=entries.setdefault(key,{'rows':0,'label_keys':set()});entry['rows']+=1;entry['label_keys'].update(labels)
 results[name]={'sha256':hashlib.sha256(raw).hexdigest(),'metric_fields':{k:{'rows':v['rows'],'label_keys':sorted(v['label_keys'])} for k,v in entries.items()}}
print(json.dumps({'db_sha256':h.hexdigest(),'association_rows':links,'metrics':results,'limits':'CONNECTION_IDS describes framework/task associations, not tensor identity; selected Prometheus snapshots expose label names only and no per-replay association is established; only two snapshots and selected metrics are inspected.'},indent=2))
