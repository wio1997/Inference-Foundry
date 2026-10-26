"""Read-only historical Host interval partition; no inferred device service cost."""
import json,hashlib
from pathlib import Path
out=Path('evidence/20260929_loop081_bound/run669')
rows=[]; sources={}
for run in (602,606):
 for c in range(5,9):
  for rank in range(8):
   p=Path(f'evidence/20260928_loop081_bound/run{run}/live/b/product/rank{rank}_cohort{c}.json')
   sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
   d=json.loads(p.read_text()); marks=d['marks']; calls=d['calls']
   built=next(m for m in marks if m['kind']=='runtime_built_host')
   ids=set(built['req_ids'])
   start_index=next(i for i,m in enumerate(marks) if m['kind']=='execute_entry' and any(x['req_id'] in ids for x in m['new']))
   entries=[(i+1,m) for i,m in enumerate(marks) if i>=start_index and m['kind']=='execute_entry']
   end=next(m['t_ns'] for m in marks if m['kind']=='runtime_built_host')
   begin=entries[0][1]['t_ns']; parts=[]
   for n,(ordinal,m) in enumerate(entries):
    stop=entries[n+1][1]['t_ns'] if n+1<len(entries) else end
    cs=[x for x in calls if x['execute_mark_count']==ordinal]
    target=next((x for x in cs if x['kind']=='target_forward'),None)
    prop=next((x for x in cs if x['kind']=='propose_and_optional_copy'),None)
    row={'ordinal':ordinal,'scheduled_tokens':m['total_scheduled_tokens'],'new_count':len(m['new']),'cached_count':len(m['cached']),'interval_ms':(stop-m['t_ns'])/1e6}
    if target and prop:
     assert m['t_ns']<=target['host_start_ns']<=target['host_end_ns']<=prop['host_start_ns']<=prop['host_end_ns']<=stop
     row.update(pre_target_ms=(target['host_start_ns']-m['t_ns'])/1e6,target_host_ms=(target['host_end_ns']-target['host_start_ns'])/1e6,target_to_propose_ms=(prop['host_start_ns']-target['host_end_ns'])/1e6,propose_host_ms=(prop['host_end_ns']-prop['host_start_ns'])/1e6,post_propose_to_next_ms=(stop-prop['host_end_ns'])/1e6,padded_tokens=target['num_tokens_padded'],target_current_stream_ms=target['current_stream_elapsed_ms'],propose_current_stream_ms=prop['current_stream_elapsed_ms'])
    else:
     assert not cs
     row['handoff_execute_ms']=row['interval_ms']
    parts.append(row)
   keys=['pre_target_ms','target_host_ms','target_to_propose_ms','propose_host_ms','post_propose_to_next_ms','handoff_execute_ms']
   totals={k:sum(x.get(k,0) for x in parts) for k in keys}
   assert abs(sum(totals.values())-(end-begin)/1e6)<1e-6
   rows.append({'run':run,'cohort':c,'rank':rank,'total_ms':(end-begin)/1e6,'partition_ms':totals,'executes':parts})
result={'scope':'Observer-perturbed Host wall partition only. Current-stream durations overlap Host and cannot be added or called compute/HCCL cost. Post-proposal interval includes scheduler, transport, output and potential device dependency; not scheduler-only wait.','sources':sources,'rows':rows}
(out/'historical_partition.json').write_text(json.dumps(result,indent=2)+'\n')
for r in rows:
 if r['rank']==0:print(r['run'],r['cohort'],{k:round(v,3) for k,v in r['partition_ms'].items()})
print('PASS 64 rank-cohort interval partitions')
