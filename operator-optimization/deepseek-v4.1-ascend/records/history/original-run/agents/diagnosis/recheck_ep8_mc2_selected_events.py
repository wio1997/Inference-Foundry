#!/usr/bin/env python3
"""Compare previously extracted EP8 DB events and MC2 tasks, no device access."""
import hashlib,json
from pathlib import Path
p=Path(__file__).resolve().parent
task_root=p.parent.parent
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
f=[p/'engram-producer-events-exact.json',p/'mc2-pmu-replay-links-ep8.json',p/'16rank-mc2-other-stream.json']
e,db,csv=[json.loads(x.read_text()) for x in f]
assert e['db_sha256']==db['database_sha256']
out=[]
for run in (1,2,3):
 targets=[x for x in csv['samples'] if '_ep8_' in x['rank'] and x['run']==run]
 assert len(targets)==1
 t=targets[0]; a,b=t['mc2_start_ns'],t['mc2_end_ns']
 matches=[x for x in db['all_TASK_rows_sharing_globalTaskId'] if x['startNs']==a and x['endNs']==b and x['deviceId']==8 and x['modelId']==49 and x['streamId']==89]
 assert len(matches)==1
 bykey={}
 for g in e['gap_windows']:
  if g['run']!=run: continue
  assert not g['event_task_rows']['truncated']
  for v in g['event_task_rows']['rows']:
   if v['task_type_name']=='EVENT WAIT' and v['deviceId']==8 and v['modelId']==49 and v['streamId'] in (87,88):
    bykey[(v['startNs'],v['endNs'],v['streamId'],v['taskId'])]=v
 hits=[]
 for r in sorted(bykey.values(),key=lambda x:(x['streamId'],x['startNs'])):
  n=max(0,min(b,r['endNs'])-max(a,r['startNs']))
  if n:
   hits.append(dict(stream=r['streamId'],task_id=r['taskId'],global_task_id=r['globalTaskId'],start_ns=r['startNs'],end_ns=r['endNs'],overlap_ns=n,fraction=n/(b-a),begins_before_ns=a-r['startNs'],ends_after_ns=r['endNs']-b))
 assert len(hits)==2 and {x['stream'] for x in hits}=={87,88} and all(x['overlap_ns']==b-a for x in hits)
 out.append(dict(run=run,mc2_start_ns=a,mc2_end_ns=b,mc2_duration_ns=b-a,db_task=matches[0],selected_events=hits))
print(json.dumps({'scope':'EP8 only, 3 chosen layer-0 MC2 tasks and event rows previously selected by overlap with layer 0/13 gaps; not an exhaustive event survey','inputs':{str(x.relative_to(task_root)):sha(x) for x in f},'same_source_db_sha256':e['db_sha256'],'samples':out,'limits':['EVENT WAIT is a database label; event object/producer/consumer and runtime semantics are not established.','Temporal overlap and shared model or connection do not prove causal dependency, communication hiding or resource contention.','Selection misses events not overlapping the two chosen gap windows and excludes other ranks, streams and tasks.','Comparison is same-device DB timestamps; does not require cross-device alignment.','Historical CSV did not cover EVENT WAIT as kernel rows; no contradiction with absence of matching kernel rows.']},ensure_ascii=False,indent=2))
