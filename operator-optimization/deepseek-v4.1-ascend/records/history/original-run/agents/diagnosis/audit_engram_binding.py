#!/usr/bin/env python3
"""Read only: bind retained wrapper records to client window and observer statuses.
Run inside the stage container. Emits aggregates only, no record contents or process IDs.
"""
import collections
import hashlib
import json
from pathlib import Path

RUN = '001d14496b2146bba25dae0f27c780b5'
ROOT = Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029')
base = ROOT / ('engram-diagnostic-' + RUN) / 'runtime'
progress = json.loads((base / 'engram-client-results/progress.json').read_text())
window = progress['window_id']
path = base / 'metadata/engram-metadata.jsonl'
before = path.stat()
raw = path.read_bytes()
after = path.stat()
assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
events = [json.loads(line) for line in raw.splitlines()]
statuses = [json.loads(p.read_text()) for p in (base / 'metadata').glob('observer-*.json')]
status_by_process = collections.defaultdict(list)
for s in statuses:
    status_by_process[s.get('process_instance')].append(s)
records = [e for e in events if e.get('event') in ('embed_begin','embed_end')]
processes = {e.get('process_instance') for e in records}
pids = collections.defaultdict(set)
for e in records:
    pids[e.get('process_instance')].add(e.get('pid'))
matching = {p for p in processes if len(status_by_process[p]) == 1 and status_by_process[p][0].get('run_id') == RUN and status_by_process[p][0].get('window_id') == window and pids[p] == {status_by_process[p][0].get('pid')}}
begins = collections.defaultdict(list)
ends = collections.defaultdict(list)
for e in records:
    (begins if e['event']=='embed_begin' else ends)[(e.get('process_instance'),e.get('call'))].append(e)
valid = [v[0] for k,v in begins.items() if len(v)==1 and k[0] in matching and v[0].get('window_id')==window and isinstance(v[0].get('tensor'),dict)]
pairs = [k for k,v in begins.items() if len(v)==1 and len(ends.get(k,[]))==1 and k[0] in matching and v[0].get('window_id')==window and ends[k][0].get('window_id')==window and ends[k][0].get('outcome')=='returned']
cap = [i for i,e in enumerate(events) if e.get('event')=='trace_cap']
stats = collections.Counter()
for s in statuses:
    for k,v in (s.get('stats') or {}).items():
        if isinstance(v,int): stats[k]+=v
sigs = collections.Counter((tuple(e['tensor'].get('shape',[])),tuple(e['tensor'].get('stride',[])),e['tensor'].get('storage_offset'),e['tensor'].get('is_contiguous'),e['tensor'].get('dtype'),e['tensor'].get('device_type')) for e in valid)
print(json.dumps({'run_id':RUN,'progress_run_matches':progress.get('run_id')==RUN,'progress_complete':progress.get('state')=='COMPLETE','progress_not_benchmark':progress.get('not_a_benchmark_score') is True,'window_id':window,'log_sha256':hashlib.sha256(raw).hexdigest(),'log_bytes':len(raw),'record_count':len(records),'record_window_mismatch_count':sum(e.get('window_id')!=window for e in records),'observer_status_count':len(statuses),'observer_process_unique_count':len(status_by_process),'observer_run_mismatch_count':sum(s.get('run_id')!=RUN for s in statuses),'observer_window_mismatch_count':sum(s.get('window_id')!=window for s in statuses),'observer_closed_window_values':dict(collections.Counter(str(s.get('closed_window')) for s in statuses)),'observer_status_incomplete_count':sum(s.get('state_incomplete') is True for s in statuses),'record_process_count':len(processes),'record_process_matching_unique_status_and_pid_count':len(matching),'record_process_unmatched_count':len(processes-matching),'status_process_without_record_count':len(set(status_by_process)-processes),'record_pid_inconsistent_process_count':sum(len(v)!=1 for v in pids.values()),'valid_metadata_unique_begin_count':len(valid),'unique_returned_pair_count':len(pairs),'duplicate_begin_keys':sum(len(v)>1 for v in begins.values()),'duplicate_end_keys':sum(len(v)>1 for v in ends.values()),'begin_only_keys':sum(k not in ends for k in begins),'end_only_keys':sum(k not in begins for k in ends),'trace_cap_count':len(cap),'trace_cap_is_last_record':bool(cap) and cap[-1]==len(events)-1,'status_stats_totals':dict(stats),'signatures':[{'shape':a,'stride':b,'offset':c,'contiguous':d,'dtype':e,'device':f,'count':n} for (a,b,c,d,e,f),n in sigs.most_common()]},indent=2))
