#!/usr/bin/env python3
"""Offline regression for a minimal fixed-DSpark7 cycle basis on Run566 A0.

Observed parking transitions are derived from the old full trace and stand in
for explicit Host parking events in a future compact collector. They are not
independent evidence that the Host event can be inferred from counts alone.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path('/data/wio/Inference_Foundry')
OLD = ROOT/'evidence/20260928_loop080_bound/run566/a0'
OUT = ROOT/'evidence/20260928_loop081_bound/run594'
WIDTH = 8
BATCH = 12
PARK_OFFSET = 960

def need(condition, message):
    if not condition:
        raise ValueError(message)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def accepted_prefix(row):
    need(len(row) == WIDTH, 'acceptance width')
    count = next((i for i,v in enumerate(row) if v < 0), WIDTH)
    need(1 <= count <= WIDTH and all(v >= 0 for v in row[:count]) and
         all(v == -1 for v in row[count:]), 'acceptance prefix')
    return count

def reconstruct(trace, staged):
    need(trace and len(staged)==BATCH, 'trace/cohort shape')
    initial = trace[0]
    position = initial['num_computed_before'][:]
    last = initial['last_token_before'][:]
    draft = [x[:] for x in initial['draft_before']]
    origin = position[:]
    parked = [False]*BATCH
    parks = [[] for _ in range(BATCH)]
    active_cycles=[0]*BATCH
    active_samples=[0]*BATCH
    physical=0
    for cycle,row in enumerate(trace):
        need(len(row['counts'])==BATCH and len(row['accepted'])==BATCH and
             len(row['next_draft'])==BATCH, 'cycle width')
        need(position==row['num_computed_before'], f'position c{cycle}')
        need(last==row['last_token_before'], f'last token c{cycle}')
        need(draft==row['draft_before'], f'draft c{cycle}')
        expected_ids=[[last[s]]+draft[s] for s in range(BATCH)]
        expected_positions=[[position[s]+j for j in range(WIDTH)]
                            for s in range(BATCH)]
        need(expected_ids==row['target_input_ids'],f'target ids c{cycle}')
        need(expected_positions==row['target_positions'],f'target positions c{cycle}')
        for slot in range(BATCH):
            raw = accepted_prefix(row['accepted'][slot])
            need(raw==row['counts'][slot],f'raw acceptance c{cycle}s{slot}')
            need(len(row['next_draft'][slot])==WIDTH-1 and
                 all(v>=0 for v in row['next_draft'][slot]),
                 f'next draft c{cycle}s{slot}')
            if not parked[slot]:
                active_cycles[slot]+=1
                active_samples[slot]+=raw
                position[slot]+=raw
                last[slot]=row['accepted'][slot][raw-1]
            draft[slot]=row['next_draft'][slot][:]
            physical+=WIDTH
        # The historical full trace is used ONLY to extract Host park events.
        # A new compact collector must record these events explicitly.
        if cycle+1<len(trace):
            nxt=trace[cycle+1]['num_computed_before']
            for slot in range(BATCH):
                if not parked[slot] and nxt[slot] != position[slot]:
                    anchor=origin[slot]+PARK_OFFSET
                    need(nxt[slot]==anchor and position[slot]>anchor and
                         active_samples[slot]>=1024,
                         f'unsupported Host park c{cycle}s{slot}')
                    parks[slot].append(cycle+1)
                    parked[slot]=True
                    position[slot]=anchor
    need(all(len(x)<=1 for x in parks),'multiple parks')
    need(active_samples==staged, 'active sampled/staged mismatch')
    active=WIDTH*sum(active_cycles)
    return dict(cycles=len(trace),physical_target_rows=physical,
                active_target_rows=active,parked_target_rows=physical-active,
                park_cycles=parks,active_sampled=active_samples,
                target_ids_positions_reconstructed=True,
                last_and_draft_state_reconstructed=True)

def negative_tests(real_trace, staged):
    # Each mutation starts from a full valid cohort. A truncated prefix would
    # fail the final staged-total gate even without a mutation.
    tests=[]
    mutations=(
      ('wrong_target_token','target ids',lambda a:a[1]['target_input_ids'][0].__setitem__(0,-99)),
      ('wrong_target_position','target positions',lambda a:a[1]['target_positions'][0].__setitem__(0,-99)),
      ('wrong_draft_lineage','draft',lambda a:a[1]['draft_before'][0].__setitem__(0,-99)),
      ('wrong_raw_count','raw acceptance',lambda a:a[0]['counts'].__setitem__(0,0)),
      ('wrong_staged','active sampled/staged',None),
    )
    import copy
    reconstruct(real_trace,staged)
    for label,expected,mutation in mutations:
        a=copy.deepcopy(real_trace)
        try:
            if mutation is None:reconstruct(a,[v+1 for v in staged])
            else:
                mutation(a);reconstruct(a,staged)
        except ValueError as error:
            need(expected in str(error),f'{label} rejected for wrong reason: {error}')
            tests.append(label)
        else:raise AssertionError('negative admitted: '+label)
    return tests

def main():
    need(not (OUT/'summary.json').exists(),'Run594 output already exists')
    rows=[];hashes={};negatives=None
    for rank in range(8):
        for cohort in range(5,9):
            trace_path=OLD/'trace'/f'trace_rank{rank}_cohort{cohort}.json'
            runtime_path=OLD/'runtime'/f'rank{rank}_cohort{cohort}.json'
            need(trace_path.exists() and runtime_path.exists(), 'missing source')
            trace=json.loads(trace_path.read_text())
            runtime=json.loads(runtime_path.read_text())
            hashes[str(trace_path.relative_to(ROOT))]=digest(trace_path)
            hashes[str(runtime_path.relative_to(ROOT))]=digest(runtime_path)
            need(runtime['rank']==rank and runtime['cohort']==cohort and
                 runtime['cycles']==len(trace) and runtime['pass'] and
                 runtime['generated_output_counts']==[1024]*BATCH,
                 'runtime cohort join')
            if negatives is None:
                negatives=negative_tests(trace,runtime['staged_output_counts'])
            row=reconstruct(trace,runtime['staged_output_counts'])
            rows.append(dict(rank=rank,cohort=cohort,**row))
    need(len(rows)==32 and negatives==[
        'wrong_target_token','wrong_target_position','wrong_draft_lineage',
        'wrong_raw_count','wrong_staged'], 'coverage/negative tests')
    for cohort in range(5,9):
        same=[r for r in rows if r['cohort']==cohort]
        need(len(same)==8 and len({(r['cycles'],r['active_target_rows'],
                                    r['parked_target_rows'],
                                    json.dumps(r['park_cycles'])) for r in same})==1,
             'all8 cohort consensus')
    representative=[r for r in rows if r['rank']==0]
    sums={key:sum(r[key] for r in representative)
          for key in ('cycles','physical_target_rows','active_target_rows',
                      'parked_target_rows')}
    need(sums==dict(cycles=1206,physical_target_rows=115776,
                    active_target_rows=98496,parked_target_rows=17280),
         'Run570 census regression')
    result=dict(status='offline_minimal_basis_regression_pass',
        scope='Run566 A0 only; Host parking events extracted from old full trace',
        rows=rows,rank0_sums=sums,all8_consensus=True,
        source_sha256=hashes,negative_mutations_rejected=negatives,
        future_collector_requires_explicit_host_park_events=True,
        fixed_W0_formal_Run99_join=False,resource_bound_s=None,
        scheduling_bound_s=None,product_bound_tps=None,
        current_formal_tps=571.681)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'rank0_sums':sums,
                      'source_files':len(hashes)},sort_keys=True))

if __name__=='__main__':main()
