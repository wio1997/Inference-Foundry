#!/usr/bin/env python3
"""Offline all8 phase/native coverage audit of existing Run246 profiler windows.

Midpoint containment is descriptive overlap, never a Host→device ownership or
critical-path edge. This script performs no NPU work and emits no time bound.
"""
from __future__ import annotations

import collections
import hashlib
import json
import statistics
from decimal import Decimal
from pathlib import Path

ROOT=Path('/data/wio/Inference_Foundry')
OLD=ROOT/'evidence/20260926_loop060_resource/run246'
OUT=ROOT/'evidence/20260928_loop081_bound/run593'
PHASES=('prepare_target','derived_target_metadata','target','acceptance',
        'state_advance','proposer','draft_commit')
NESTED=('dspark_host_mirror_commit','dspark_host_mirror_launch',
        'dspark_refresh_common','dspark_context_slots','dspark_prepare_inputs',
        'dspark_pack_hidden','dspark_model')

def need(ok,msg):
    if not ok:raise ValueError(msg)

def dec(v):return Decimal(str(v).strip())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def interval(x):
    a=dec(x['ts']);return a,a+dec(x['dur'])
def native(x):
    return x.get('ph')=='X' and x.get('args',{}).get('Model Id') in (45,4294967295)

def validate_scopes(events):
    scopes={}
    for name in PHASES:
        a=sorted((x for x in events if x.get('name')=='extreme::'+name),
                 key=lambda x:dec(x['ts']))
        need(len(a)==2,f'{name} scope count differs from two')
        need(all(x.get('cat')=='cpu_op' and x.get('ph')=='X' for x in a),
             f'{name} CPU scope type mismatch')
        scopes[name]=a
    outer=[x for x in events if x.get('name')=='extreme::cycle']
    need(len(outer)==1,'outer cycle scope count differs from one')
    for i in range(2):
        ordered=[interval(scopes[name][i]) for name in PHASES]
        need(all(a<b for a,b in ordered),'nonpositive phase duration')
        need(all(ordered[j][1]<=ordered[j+1][0] for j in range(len(ordered)-1)),
             f'phase scope overlap/order {i}')
        if i==0:
            first,last=interval(outer[0])
            need(first<=ordered[0][0] and ordered[-1][1]<=last,
                 'first cycle not enclosed by outer scope')
    return scopes,outer

def reduce_one(rank,selected,events):
    scopes,outer=validate_scopes(events)
    tasks=[x for x in events if native(x)]
    by_model=collections.Counter(x['args']['Model Id'] for x in tasks)
    need(by_model[45]==2*5412 and by_model[4294967295]>0,
         'native model/task inventory mismatch')
    phase_rows=[]; assignment=collections.Counter()
    for i in range(2):
        for name in PHASES:
            scope=scopes[name][i];start,end=interval(scope)
            contained=[];fully=[]
            for x in tasks:
                a,b=interval(x);mid=(a+b)/2
                if start<=mid<end:
                    contained.append(x)
                    if start<=a and b<=end:fully.append(x)
            if name=='target':
                need(sum(x['args']['Model Id']==45 for x in contained)==5412,
                     f'rank{rank} target{i} lacks complete Model45 Graph')
            counts=collections.Counter(str(x['args']['Model Id']) for x in contained)
            types=collections.Counter(x['args'].get('Task Type','unknown') for x in contained)
            phase_rows.append(dict(occurrence=i,phase=name,
                host_start_us=str(start),host_end_us=str(end),
                host_scope_duration_us=str(end-start),
                native_midpoint_count=len(contained),
                native_fully_contained_count=len(fully),
                native_midpoint_by_model=dict(counts),
                native_midpoint_by_task_type=dict(types),
                assignment_rule='native event midpoint inside CPU scope; temporal overlap only'))
    for x in tasks:
        a,b=interval(x);mid=(a+b)/2
        matches=[]
        for i in range(2):
            for name in PHASES:
                start,end=interval(scopes[name][i])
                if start<=mid<end:
                    matches.append((i,name))
        need(len(matches)<=1,'top-level phase scopes overlap at native midpoint')
        assignment['assigned' if matches else 'outside']+=1
    intersections=0;distinct_crossing=0
    boundaries=[v for i in range(2) for name in PHASES
                for v in interval(scopes[name][i])]
    for x in tasks:
        a,b=interval(x)
        count=sum(a<v<b for v in boundaries)
        intersections+=count
        distinct_crossing+=bool(count)
    nested={name:len([x for x in events if x.get('name')=='extreme::'+name])
            for name in NESTED}
    dspark_layer={str(layer):len([x for x in events
        if x.get('name')==f'extreme::dspark_layer::{layer}'])
        for layer in (43,44,45)}
    need(all(v==2 for v in nested.values()),'DSpark nested scope count drift')
    need(all(v==10 for v in dspark_layer.values()),'DSpark layer scope multiplicity drift')
    return dict(rank=rank,selected_profile=str(selected.relative_to(ROOT)),
        native_count=len(tasks),native_by_model={str(k):v for k,v in by_model.items()},
        midpoint_assignment=dict(assignment),
        native_phase_boundary_intersections=intersections,
        native_distinct_tasks_intersecting_phase_boundary=distinct_crossing,
        complete_outer_cycle_scope_count=len(outer),phase_scope_sets=2,
        nested_scope_counts=nested,dspark_layer_scope_counts=dspark_layer,
        phase_rows=phase_rows)

def negative_tests(events):
    # Keep mutations on the narrow CPU scope list; no full trace copy needed.
    scopes,outer=validate_scopes(events)
    tests=[]
    for label,mutate in (
        ('missing_phase',lambda a:a.remove(scopes['target'][0])),
        ('duplicate_outer',lambda a:a.append(outer[0])),
    ):
        a=[x for x in events if x.get('name','').startswith('extreme::')]
        mutate(a)
        try:validate_scopes(a)
        except ValueError:tests.append(label)
        else:raise AssertionError(f'negative admitted: {label}')
    return tests

def main():
    need(not (OUT/'summary.json').exists(),'Run593 output already exists')
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[];hashes={};negatives=None
    for rank in range(8):
        dirs=sorted((OLD/'profile').glob(f'rank{rank}_*_ascend_pt'))
        need(len(dirs)==5,f'rank{rank} expected five profile sessions')
        selected=dirs[-1]
        trace=selected/'ASCEND_PROFILER_OUTPUT'/'trace_view.json'
        info=selected/f'profiler_info_{rank}.json'
        window=OLD/'profile'/f'rank{rank}_window.json'
        for p in (trace,info,window):hashes[str(p.relative_to(ROOT))]=sha(p)
        meta=json.loads(info.read_text());w=json.loads(window.read_text())
        need(meta['rank_id']==rank and w['rank']==rank and
             w['first_cycle']==64 and w['cycle_count']==2,
             'profile rank/window mismatch')
        need(dec(w['started_ns'])<=dec(meta['end_info']['collectionTimeEnd'])<=dec(w['stopped_ns']),
             'last profile collection end outside final window')
        events=json.loads(trace.read_text())
        if negatives is None:negatives=negative_tests(events)
        rows.append(reduce_one(rank,selected,events))
    need(len(rows)==8 and negatives==['missing_phase','duplicate_outer'],
         'rank/negative count')
    def rollup(name,model):
        vals=[int(x['native_midpoint_by_model'].get(model,0)) for r in rows
              for x in r['phase_rows'] if x['phase']==name]
        return dict(min=min(vals),median=statistics.median(vals),max=max(vals))
    data=dict(status='historical_instrumented_phase_coverage_partial',
        contract='Run246 last-profile native/CPU temporal coverage, no fixed-W0 or direct request join',
        rank_count=8,phase_scope_sets_per_rank=2,
        native_model45_tasks_per_rank=10824,
        host_scope_duration_is_submission_plus_profiler_not_device_service=True,
        phase_ownership_from_midpoint=False,
        dspark_layer_scope_counts_are_overlapping_repeated_scopes_not_replay_count=True,
        stage_midpoint_counts={name:{'graph45':rollup(name,'45'),
                                     'non_graph':rollup(name,'4294967295')}
                               for name in PHASES},
        rows=rows,source_sha256=hashes,negative_mutations_rejected=negatives,
        formal_current_tps=571.681,resource_bound_s=None,
        scheduling_bound_s=None,product_bound_tps=None)
    (OUT/'summary.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps({'status':data['status'],'rank_count':8,
                      'stage_midpoint_counts':data['stage_midpoint_counts'],
                      'outside_native':[r['midpoint_assignment'].get('outside',0) for r in rows]}))

if __name__=='__main__':main()
