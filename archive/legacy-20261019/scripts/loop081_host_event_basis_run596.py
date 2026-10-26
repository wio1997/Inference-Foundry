#!/usr/bin/env python3
"""Split historical Host-event extraction from compact cycle-basis replay.

Run566 A0 full traces are the oracle. Replay consumes only initial state,
accepted token/count history, next-draft history and explicit Host park events.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from loop081_basis_regression_run594 import accepted_prefix, need

ROOT=Path('/data/wio/Inference_Foundry')
OLD=ROOT/'evidence/20260928_loop080_bound/run566/a0'
OUT=ROOT/'evidence/20260928_loop081_bound/run596'

def extract_host_events(full_trace):
    events=[]
    for slot in range(12):
        positions=[r['num_computed_before'][slot] for r in full_trace]
        drops=[i for i in range(1,len(positions)) if positions[i]<positions[i-1]]
        need(len(drops)<=1,'multiple historical rewinds')
        if drops:
            cycle=drops[0]
            events.append(dict(slot=slot,cycle=cycle,anchor=positions[cycle]))
    return sorted(events,key=lambda x:(x['cycle'],x['slot']))

def compact_basis(full_trace):
    first=full_trace[0]
    return dict(initial_position=first['num_computed_before'][:],
                initial_last=first['last_token_before'][:],
                initial_draft=copy.deepcopy(first['draft_before']),
                accepted=[r['accepted'] for r in full_trace],
                counts=[r['counts'] for r in full_trace],
                next_draft=[r['next_draft'] for r in full_trace])

def replay(basis,events,staged,oracle=None):
    cycles=len(basis['counts'])
    need(cycles>0 and len(staged)==12,'basis/staged shape')
    need(all(len(basis[k])==cycles for k in ('accepted','next_draft')),
         'basis cycle count')
    event_map={}
    for e in events:
        slot,cycle,anchor=e['slot'],e['cycle'],e['anchor']
        need(isinstance(slot,int) and 0<=slot<12 and
             isinstance(cycle,int) and 1<=cycle<cycles and
             isinstance(anchor,int),'Host event schema')
        need((cycle,slot) not in event_map,'duplicate Host event')
        event_map[(cycle,slot)]=anchor
    position=basis['initial_position'][:]
    origin=position[:]
    last=basis['initial_last'][:]
    draft=copy.deepcopy(basis['initial_draft'])
    parked=[False]*12
    active_cycles=[0]*12
    emitted=[0]*12
    need(all(len(x)==12 for x in (position,last,draft)),'entry shape')
    for cycle in range(cycles):
        if cycle:
            for slot in range(12):
                if (cycle,slot) in event_map:
                    need(not parked[slot] and emitted[slot]>=1024,
                         'Host event before completion or after park')
                    anchor=event_map[cycle,slot]
                    need(anchor==origin[slot]+960 and position[slot]>anchor,
                         'Host park anchor')
                    position[slot]=anchor
                    parked[slot]=True
        ids=[[last[s]]+draft[s] for s in range(12)]
        positions=[[position[s]+i for i in range(8)] for s in range(12)]
        if oracle is not None:
            row=oracle[cycle]
            need(position==row['num_computed_before'],f'oracle position c{cycle}')
            need(last==row['last_token_before'],f'oracle last c{cycle}')
            need(draft==row['draft_before'],f'oracle draft c{cycle}')
            need(ids==row['target_input_ids'],f'oracle Target ids c{cycle}')
            need(positions==row['target_positions'],f'oracle Target positions c{cycle}')
        need(len(basis['counts'][cycle])==12 and
             len(basis['accepted'][cycle])==12 and
             len(basis['next_draft'][cycle])==12,'cycle slot shape')
        for slot in range(12):
            accepted=basis['accepted'][cycle][slot]
            count=accepted_prefix(accepted)
            need(count==basis['counts'][cycle][slot],
                 f'raw count c{cycle}s{slot}')
            next_draft=basis['next_draft'][cycle][slot]
            need(len(next_draft)==7 and all(x>=0 for x in next_draft),
                 f'next draft c{cycle}s{slot}')
            if not parked[slot]:
                active_cycles[slot]+=1
                emitted[slot]+=count
                position[slot]+=count
                last[slot]=accepted[count-1]
            draft[slot]=next_draft[:]
    need(emitted==staged,'staged output mismatch')
    active=8*sum(active_cycles)
    return dict(cycles=cycles,active_rows=active,
                physical_rows=cycles*96,parked_rows=cycles*96-active,
                park_events=len(events))

def negative_tests(basis,events,staged,oracle):
    replay(basis,events,staged,oracle)
    need(events,'negative fixture needs park event')
    tests=[]
    variants=(
      ('missing',lambda x:x.pop(0),'oracle position'),
      ('duplicate',lambda x:x.append(copy.deepcopy(x[0])),'duplicate Host event'),
      ('wrong_anchor',lambda x:x[0].__setitem__('anchor',x[0]['anchor']+1),'Host park anchor'),
      ('shifted',lambda x:x[0].__setitem__('cycle',x[0]['cycle']+1),'oracle position'),
    )
    for name,mutation,expected in variants:
        e=copy.deepcopy(events);mutation(e)
        try:replay(basis,e,staged,oracle)
        except ValueError as error:
            need(expected in str(error),f'{name} wrong rejection: {error}')
            tests.append(name)
        else:raise AssertionError('Host event mutation admitted: '+name)
    return tests

def main():
    need(not (OUT/'summary.json').exists(),'output already exists')
    rows=[];hashes={};negative=None;cohort_refs={};basis_hashes={}
    for rank in range(8):
        for cohort in range(5,9):
            tp=OLD/'trace'/f'trace_rank{rank}_cohort{cohort}.json'
            rp=OLD/'runtime'/f'rank{rank}_cohort{cohort}.json'
            full=json.loads(tp.read_text())
            runtime=json.loads(rp.read_text())
            need(runtime['rank']==rank and runtime['cohort']==cohort and
                 runtime['pass'] and runtime['cycles']==len(full) and
                 runtime['generated_output_counts']==[1024]*12 and
                 len(runtime['req_ids'])==12 and
                 len(set(runtime['req_ids']))==12,
                 'Runtime/source cohort join')
            basis=compact_basis(full)
            events=extract_host_events(full)
            identity=hashlib.sha256(json.dumps(
                dict(req_ids=runtime['req_ids'],basis=basis,host_events=events),
                sort_keys=True,separators=(',',':')).encode()).hexdigest()
            if cohort in cohort_refs:
                need((runtime['req_ids'],identity)==cohort_refs[cohort],
                     'all8 request/basis/event mismatch')
            else:cohort_refs[cohort]=(runtime['req_ids'],identity)
            basis_hashes[f'rank{rank}_cohort{cohort}']=identity
            outcome=replay(basis,events,runtime['staged_output_counts'],full)
            if negative is None:
                negative=negative_tests(basis,events,runtime['staged_output_counts'],full)
            rows.append(dict(rank=rank,cohort=cohort,**outcome))
            for p in (tp,rp):hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    need(len(rows)==32 and negative==['missing','duplicate','wrong_anchor','shifted'],
         'coverage/negative tests')
    for cohort in range(5,9):
        need(len({(r['cycles'],r['active_rows'],r['parked_rows'],r['park_events'])
                  for r in rows if r['cohort']==cohort})==1,'all8 mismatch')
    rank0=[r for r in rows if r['rank']==0]
    total={k:sum(r[k] for r in rank0)
           for k in ('cycles','physical_rows','active_rows','parked_rows')}
    need(total==dict(cycles=1206,physical_rows=115776,
                     active_rows=98496,parked_rows=17280),'Run570 census mismatch')
    result=dict(status='explicit_host_event_compact_basis_regression_pass',
        scope='Run566 A0; Host events extracted from old full trace before pure replay',
        total_rank0=total,rows=rows,source_sha256=hashes,
        semantic_basis_event_sha256=basis_hashes,all8_basis_event_consensus=True,
        negative_host_event_mutations_rejected=negative,
        future_collector_must_record_host_events=True,
        full_draft_and_kv_ledger_proven=False,formal_run99_same_state=False,
        strict_resource_bound_s=None,strict_scheduling_bound_s=None,
        strict_product_bound_tps=None,formal_current_tps=571.681)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'total_rank0':total,
                      'host_event_negative_tests':negative},sort_keys=True))

if __name__=='__main__':main()
