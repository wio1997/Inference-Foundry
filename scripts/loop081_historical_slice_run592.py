#!/usr/bin/env python3
"""Reduce the existing Run246 Level1 production trace, without running an NPU.

This is a historical instrumented Current slice. Run589 native labels provide a
cross-run hypothesis, not a same-generation identity or a Scheduling bound.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import statistics
from decimal import Decimal
from pathlib import Path

ROOT=Path('/data/wio/Inference_Foundry')
OLD=ROOT/'evidence/20260926_loop060_resource/run246'
OUT=ROOT/'evidence/20260928_loop081_bound/run592'
TASKS={
 'partial':(1,40,'MatMulV2'),
 'record_p':(1,41,'EVENT_RECORD'),
 'wait_p':(0,12,'EVENT_WAIT'),
 'rs':(0,13,'AivKernel'),
 'record_r':(0,15,'EVENT_RECORD'),
 'wait_r':(1,42,'EVENT_WAIT'),
 'copy':(1,43,'MEMCPY_ASYNC'),
 'hcpost_candidate':(1,44,'HcPost'),
}

def need(ok,why):
    if not ok:raise ValueError(why)

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def dec(x):return Decimal(str(x).strip())
def event_key(e):
    a=e.get('args',{})
    return a.get('Model Id'),a.get('Physic Stream Id'),a.get('Task Id')

def select_events(events):
    by_role={}
    for role,(stream,task,fragment) in TASKS.items():
        selected=[e for e in events if event_key(e)==(45,stream,task)]
        need(len(selected)==2,f'{role} not exactly twice in Model45')
        need(all(fragment in e.get('name','') and e.get('ph')=='X' for e in selected),
             f'{role} name/type mismatch')
        by_role[role]=sorted(selected,key=lambda e:dec(e['ts']))
    return by_role

def reduce_events(by_role):
    rows=[]
    for i in range(2):
        e={k:v[i] for k,v in by_role.items()}
        t={k:(dec(v['ts']),dec(v['ts'])+dec(v['dur'])) for k,v in e.items()}
        p,rs,cp,h=t['partial'],t['rs'],t['copy'],t['hcpost_candidate']
        need(p[0]<p[1]<=rs[0]<rs[1]<=cp[0]<cp[1]<=h[0],
             f'producer→RS→copy→HcPost chronology invalid occurrence{i}')
        # The event tasks are a cross-stream order check, not communication time.
        need(t['record_p'][0]>=p[0] and t['record_p'][0]<=rs[1],
             'producer event outside interval')
        need(t['record_r'][0]>=rs[0] and t['record_r'][0]<=cp[1],
             'return event outside interval')
        need(t['wait_p'][0]<=rs[0] and t['wait_r'][0]<=cp[0],
             'cross-stream wait begins after consumer')
        rows.append(dict(occurrence=i,
          task_times_us={k:{'start':str(a),'end':str(b),'duration':str(b-a)}
                         for k,(a,b) in t.items()},
          intervals_us={
            'partial_start_to_copy_end':str(cp[1]-p[0]),
            'partial_end_to_copy_end':str(cp[1]-p[1]),
            'partial_end_to_rs_start':str(rs[0]-p[1]),
            'rs_end_to_copy_start':str(cp[0]-rs[1]),
            'copy_end_to_hcpost_start':str(h[0]-cp[1])}))
    need(dec(rows[0]['task_times_us']['hcpost_candidate']['end'])<
         dec(rows[1]['task_times_us']['partial']['start']),
         'occurrences overlap or swapped')
    return rows

def crosscheck_task_time(path, rows):
    with path.open(newline='') as f:raw=list(csv.DictReader(f))
    candidates=[r for r in raw if r['stream_id']=='1' and r['task_id']=='43'
                and r['kernel_type']=='MEMCPY_ASYNC']
    need(len(candidates)==2,'task_time copy not exactly twice')
    candidates.sort(key=lambda r:dec(r['task_start(us)']))
    for i,(raw_row,measured) in enumerate(zip(candidates,rows)):
        copy_event=measured['task_times_us']['copy']
        need(abs(dec(raw_row['task_start(us)'])-dec(copy_event['start']))<=Decimal('.002'),
             f'task_time copy start mismatch {i}')
        need(abs(dec(raw_row['task_stop(us)'])-dec(copy_event['end']))<=Decimal('.002'),
             f'task_time copy stop mismatch {i}')

def negative_gates(by_role):
    cases=[]
    for name,change in (
        ('missing_copy',lambda x:x['copy'].pop()),
        ('wrong_model',lambda x:x['copy'][0]['args'].__setitem__('Model Id',46)),
        ('wrong_stream',lambda x:x['copy'][0]['args'].__setitem__('Physic Stream Id',99)),
        ('copy_before_rs',lambda x:x['copy'][0].__setitem__('ts',x['rs'][0]['ts'])),
    ):
        trial=copy.deepcopy(by_role);change(trial)
        try:
            if name in ('wrong_model','wrong_stream'):
                need(event_key(trial['copy'][0])==(45,1,43),'copy identity')
            elif name=='missing_copy':
                need(len(trial['copy'])==2,'copy count')
            else:reduce_events(trial)
        except (ValueError,KeyError,IndexError):
            cases.append(name)
        else:raise AssertionError(f'negative mutation admitted: {name}')
    return cases

def main():
    need(not OUT.exists(),'Run592 output already exists')
    OUT.mkdir(parents=True)
    hashes={}
    rows=[]; negative=None
    for rank in range(8):
        dirs=sorted((OLD/'profile').glob(f'rank{rank}_*_ascend_pt'))
        need(len(dirs)==5,f'rank{rank} expected five historical profile dirs')
        selected=dirs[-1]
        base=selected/'ASCEND_PROFILER_OUTPUT'
        trace=base/'trace_view.json'; task_time=base/'task_time.csv'
        window=OLD/'profile'/f'rank{rank}_window.json'
        runtime=OLD/'runtime'/f'rank{rank}_cohort5.json'
        for p in (trace,task_time,window,runtime):
            need(p.is_file(),f'missing {p}')
            hashes[str(p.relative_to(ROOT))]=sha(p)
        w=json.loads(window.read_text())
        r=json.loads(runtime.read_text())
        need(w['rank']==rank and w['first_cycle']==64 and w['cycle_count']==2,
             'profile window rank/cycle mismatch')
        need(r['rank']==rank and r['cohort']==5 and r['pass'] and
             r['generated_output_counts']==[1024]*12 and r['cycles']>66,
             'historical Runtime diagnostic contract')
        events=json.loads(trace.read_text())
        selected_events=select_events(events)
        rank_rows=reduce_events(selected_events)
        crosscheck_task_time(task_time,rank_rows)
        if negative is None:negative=negative_gates(selected_events)
        for rr in rank_rows:
            start=dec(rr['task_times_us']['partial']['start'])*1000
            end=dec(rr['task_times_us']['hcpost_candidate']['end'])*1000
            need(dec(w['started_ns'])<=start<end<=dec(w['stopped_ns']),
                 'native slice outside recorded profile window')
            rr.update(rank=rank,profile_dir=str(selected.relative_to(ROOT)),
                      historical_cohort_assignment='fifth of five chronologically sorted profiles; no per-profile cohort ID')
            rows.append(rr)
    need(len(rows)==16 and len(negative)==4,'row/negative count')
    def stats(k):
        vals=[float(dec(r['intervals_us'][k])) for r in rows]
        return {'min':min(vals),'median':statistics.median(vals),
                'max':max(vals),'per_rank_max':[max(float(dec(r['intervals_us'][k]))
                    for r in rows if r['rank']==rank) for rank in range(8)]}
    metrics={k:stats(k) for k in rows[0]['intervals_us']}
    data=dict(status='historical_instrumented_current_slice_only',
      source_run='Run246 Level1 MemoryAccess production FULL Graph, last profile/window per rank',
      conditional_semantic_mapping='Run589 same-looking Model45 task chain is a cross-run hypothesis only',
      rank_count=8,occurrences_per_rank=2,negative_mutations_rejected=negative,
      metrics_us=metrics,rows=rows,source_sha256=hashes,
      current_formal_tps=571.681,strict_resource_bound_s=None,
      strict_scheduling_bound_s=None,strict_product_bound_tps=None,
      fixed_W0_equivalence_to_Run99=False,unmarked_current_interval=None,
      time_source='Run246 CANN Level1 trace_view and task_time; not Graph debug_dump synthetic ts/dur')
    (OUT/'summary.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps({'status':data['status'],'rows':len(rows),'metrics_us':metrics,
                      'negatives':negative}))

if __name__=='__main__':main()
