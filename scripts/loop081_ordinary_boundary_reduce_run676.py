"""Exact-ID ordinary preparation composition, never additive removable wall."""
import json,sys,statistics,hashlib
from pathlib import Path
root=Path(sys.argv[1] if len(sys.argv)>1 else 'evidence/20260929_loop081_bound/run676/live')
read=lambda p:json.loads(p.read_text())
assert read(root/'boundary_admission.json')['pass']
cleanup=dict(x.split('=',1) for x in (root/'cleanup_status.txt').read_text().splitlines());assert set(cleanup.values())=={'0'}
for kind in ('source','scripts'):
 assert (root/f'{kind}_before.sha256').read_bytes()==(root/f'{kind}_after.sha256').read_bytes()
 for line in (root/f'{kind}_after.sha256').read_text().splitlines():
  sha,path=line.split(None,1);assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,path
core=sorted([json.loads(x) for x in (root/'boundary/core.jsonl').read_text().splitlines()],key=lambda x:x['t_ns'])
ready={r['req_id']:r for r in core if r['kind']=='queue_put_end'}
decoded={r['req_id']:r for r in core if r['kind']=='socket_add_decoded'}
prep_begin={r['req_id']:r for r in core if r['kind']=='preprocess_begin'}
prep_end={r['req_id']:r for r in core if r['kind']=='preprocess_end'}
client=read(root/'bench48_1.json');clients={r['response_id']:r for r in client['requests']}
submits={r['submit_id']:r for r in core if r['kind']=='schedule_end_submit'}
settled={r['submit_id']:r for r in core if r['kind']=='scheduler_settled'}
drains=[r['t_ns'] for r in core if r['kind']=='input_drain_complete']
rank_events={rank:sum((read(root/f'boundary/rank{rank}_cohort{c}.json')['events'] for c in range(5,9)),[]) for rank in range(8)}
result=[]
for c in range(5,9):
 runtime=read(root/f'runtime/rank0_cohort{c}.json');ids=set(runtime['req_ids'])
 cc=[clients[decoded[r]['external_req_id']] for r in ids]
 perrank=[]
 for rank,events in rank_events.items():
  entries={r['submit_id']:r for r in events if r['kind']=='execute_entry'}
  prompt={x['req_id']:x['prompt_len'] for r in entries.values() for x in r['new']}
  selected=[]
  for sid,e in entries.items():
   reqs=e['new']+e['cached'];own={r['req_id'] for r in reqs}&ids
   if not own:continue
   target=[r for r in events if r['kind']=='target_begin' and r['submit_id']==sid]
   seed=[r for r in events if r['kind']=='seed_begin' and r['submit_id']==sid]
   if not target and not seed:continue # Actual handoff has no ordinary forward.
   row={'submit_id':sid,'req_ids':[r['req_id'] for r in reqs],'own_request_count':len(own),'other_request_count':len({r['req_id'] for r in reqs}-ids),'prompt_bearing_ids':[r['req_id'] for r in reqs if r['req_id'] in ids and r['computed']<prompt[r['req_id']]],'target_calls':len(target),'seed_calls':len(seed),'selected_modes':[r['selected_mode'] for r in target],'scheduled':{r['req_id']:r['scheduled'] for r in reqs},'attention':[r['attention'] for r in target]}
   for kind,begin,end in [('target','target_begin','target_end'),('seed','seed_begin','seed_end')]:
    bb=[r for r in events if r['kind']==begin and r['submit_id']==sid];ee=[r for r in events if r['kind']==end and r['submit_id']==sid]
    assert len(bb)==len(ee)
    row[kind+'_host_ms']=sum((b['t_ns']-a['t_ns'])/1e6 for a,b in zip(bb,ee))
   selected.append(row)
  packet=read(root/f'boundary/rank{rank}_cohort{c}.json')['events']
  stamps={k:next(x['t_ns'] for x in packet if x['kind']==k) for k in ('handoff_entry','runtime_built','runtime_begin','runtime_end')}
  handoff=next(x for x in packet if x['kind']=='handoff_entry')
  perrank.append({'rank':rank,'ordinary_executes':selected,'prompt_bearing_execute_count':sum(bool(x['prompt_bearing_ids']) for x in selected),'ordinary_target_calls':sum(x['target_calls'] for x in selected),'ordinary_seed_calls':sum(x['seed_calls'] for x in selected),'target_host_ms':sum(x['target_host_ms'] for x in selected),'seed_host_ms':sum(x['seed_host_ms'] for x in selected),'handoff_submit_id':handoff['submit_id'],'stamps_ns':stamps})
 first=min(r['submit_id'] for r in core if r['kind']=='schedule_end_submit' and set(r['scheduled'])&ids)
 first_time=submits[first]['t_ns'];previous_drains=[t for t in drains if t<=first_time];assert previous_drains
 drain=previous_drains[-1]
 rr=perrank[0];st=rr['stamps_ns'];settle=settled[rr['handoff_submit_id']]['t_ns']
 result.append({'cohort':c,'request_ids':sorted(ids),'client_start_s':min(r['start'] for r in cc),'client_end_s':max(r['end'] for r in cc),'queue_ready_spread_ms':(max(ready[r]['t_ns'] for r in ids)-min(ready[r]['t_ns'] for r in ids))/1e6,'preprocess_host_sum_ms':sum(prep_end[r]['t_ns']-prep_begin[r]['t_ns'] for r in ids)/1e6,'ready_by_first_drain_count':sum(ready[r]['t_ns']<=drain for r in ids),'first_submit_scheduled_own_count':len(set(submits[first]['scheduled'])&ids),'all_ready_to_handoff_ms':(st['handoff_entry']-max(ready[r]['t_ns'] for r in ids))/1e6,'handoff_to_runtime_ms':(st['runtime_begin']-st['handoff_entry'])/1e6,'runtime_end_to_settled_ms':(settle-st['runtime_end'])/1e6,'settled_to_last_client_ms':(max(r['end'] for r in cc)-settle/1e9)*1000,'ranks':perrank})
output={'scope':'cache-ON diagnostic composition; Host scopes contain waits and overlap across ranks, not removable wall','client':client['summary'],'cohorts':result,'cleanup':cleanup,'limits':['No device-readiness measurement or new device synchronization','Selected graph mode is not independent native branch proof','Own/other request overlap is explicit; do not assign mixed-owner wall entirely to this cohort','Observer and flush work precede client completion; no formal TPS promotion']}
(root.parent/'composition.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps({'scope':output['scope'],'cohorts':[{'cohort':r['cohort'],'ready_by_first_drain_count':r['ready_by_first_drain_count'],'first_scheduled':r['first_submit_scheduled_own_count'],'prompt_bearing_calls_rank0':r['ranks'][0]['prompt_bearing_execute_count'],'target_calls_rank0':r['ranks'][0]['ordinary_target_calls'],'seed_calls_rank0':r['ranks'][0]['ordinary_seed_calls'],'queue_ready_spread_ms':r['queue_ready_spread_ms'],'all_ready_to_handoff_ms':r['all_ready_to_handoff_ms']} for r in result]},indent=2))
