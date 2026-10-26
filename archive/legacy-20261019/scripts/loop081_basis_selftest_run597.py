#!/usr/bin/env python3
"""CPU-only Run597 ledger checker regression against old Run566 A0 full trace."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path

from scripts import loop081_basis_capture_run597 as capture

ROOT=Path('/data/wio/Inference_Foundry')
OUT=ROOT/'evidence/20260928_loop081_bound/run597_preflight/selftest.json'
TRACE=ROOT/'evidence/20260928_loop080_bound/run566/a0/trace/trace_rank0_cohort5.json'
RUNTIME=ROOT/'evidence/20260928_loop080_bound/run566/a0/runtime/rank0_cohort5.json'

def build():
    full=json.loads(TRACE.read_text());runtime=json.loads(RUNTIME.read_text())
    events=[]
    for slot in range(12):
        positions=[r['num_computed_before'][slot] for r in full]
        drops=[i for i in range(1,len(full)) if positions[i]<positions[i-1]]
        assert len(drops)<=1
        if drops:
            c=drops[0]
            events.append(dict(slot=slot,next_cycle=c,anchor=positions[c]))
    park={x['slot']:x['next_cycle'] for x in events}
    counts=[[0 if c>=park.get(s,len(full)) else raw
             for s,raw in enumerate(row['counts'])]
            for c,row in enumerate(full)]
    chosen=sorted(({c for c in capture.SELECTED if c<len(full)} |
                   {min(x['next_cycle'] for x in events)}))
    target=[];dspark=[];model=[]
    for c in chosen:
        row=full[c];ids=[v for slot in row['target_input_ids'] for v in slot]
        pos=[v for slot in row['target_positions'] for v in slot]
        target.append(dict(cycle=c,input_ids=ids,positions=pos,
                           seq_lens=[p+8 for p in row['num_computed_before']]))
        indices=[0,1,8,9,16,17]
        raw=row['counts']
        dspark.append(dict(cycle=c,token_indices=list(range(96)),
                           query_start_loc=list(range(0,97,8)),
                           seq_lens=[p+8 for p in row['num_computed_before']],
                           num_rejected=[8-n for n in raw],
                           sample_indices=[8*s+n-1 for s,n in enumerate(raw)],
                           target_token_ids=[ids[i] for i in indices],
                           target_positions=[pos[i] for i in indices]))
        # Actual DirectDSparkHandoff passes all96 rows to prepare_inputs_padded.
        # The old A0 trace has no full DSpark model-input witness. Synthesize
        # this part from the pinned DSpark source formula for checker negatives.
        dspark[-1]['target_token_ids']=ids[:]
        dspark[-1]['target_positions']=pos[:]
        Q=7;parallel=999
        next_last=[row['accepted'][s][raw[s]-1] if counts[c][s]>0
                   else row['last_token_before'][s] for s in range(12)]
        query_ids=[next_last[s] if q==0 else parallel
                   for s in range(12) for q in range(Q)]
        query_pos=[row['num_computed_before'][s]+raw[s]+q
                   for s in range(12) for q in range(Q)]
        model.append(dict(cycle=c,num_query_total=12*Q,num_context=96,
            num_query_per_req=Q,sample_from_anchor=True,
            parallel_drafting_token_id=parallel,
            query_input_ids=query_ids,query_positions=query_pos,
            context_positions=pos[:],
            sample_indices=[s*Q+q for s in range(12) for q in range(7)],
            query_start_loc=list(range(0,12*Q+1,Q)),
            seq_lens=[row['num_computed_before'][s]+raw[s]+Q for s in range(12)],
            causal=False,group_query_slots={'0':[0]*(12*Q)},
            group_context_slots={'0':[0]*96}))
    record=dict(cycles=len(full),
        initial=dict(position=full[0]['num_computed_before'],
                     last_token=full[0]['last_token_before'],
                     draft=full[0]['draft_before']),
        token_history=[x['accepted'] for x in full],count_history=counts,
        draft_history=[x['next_draft'] for x in full],
        host_park_events=events,
        branch_history=[dict(cycle=c,scheduled_target=False,schedule_mode='off')
                        for c in range(len(full))],
        target_witnesses=target,dspark_witnesses=dspark,
        draft_model_witnesses=model,
        staged_output_counts=runtime['staged_output_counts'],
        generated_output_counts=runtime['generated_output_counts'])
    return record

def main():
    if OUT.exists():raise ValueError('selftest output exists')
    row=build()
    positive=capture._check_basis(row)
    negatives=[]
    mutations=(
        ('missing_park',lambda x:x['host_park_events'].pop(0)),
        ('wrong_anchor',lambda x:x['host_park_events'][0].__setitem__(
            'anchor',x['host_park_events'][0]['anchor']+1)),
        ('wrong_target',lambda x:x['target_witnesses'][0]['input_ids'].__setitem__(0,-99)),
        ('missing_expected_checkpoint',lambda x:x['target_witnesses'].pop(0)),
        ('wrong_dspark',lambda x:x['dspark_witnesses'][0]['target_token_ids'].__setitem__(0,-99)),
        ('wrong_sample_index',lambda x:x['dspark_witnesses'][0]['sample_indices'].__setitem__(0,-99)),
        ('wrong_rejected',lambda x:x['dspark_witnesses'][0]['num_rejected'].__setitem__(0,-99)),
        ('wrong_query_position',lambda x:x['draft_model_witnesses'][0]['query_positions'].__setitem__(0,-99)),
        ('wrong_context_position',lambda x:x['draft_model_witnesses'][0]['context_positions'].__setitem__(0,-99)),
        ('wrong_draft',lambda x:x['draft_history'][0][0].__setitem__(0,-99)),
        ('wrong_masked_count',lambda x:x['count_history'][0].__setitem__(0,0)),
    )
    for name,mutation in mutations:
        candidate=copy.deepcopy(row);mutation(candidate)
        try:capture._check_basis(candidate)
        except (AssertionError,ValueError):negatives.append(name)
        else:raise AssertionError('mutation admitted: '+name)
    terminal=copy.deepcopy(row)
    already={e['slot'] for e in terminal['host_park_events']}
    terminal_slot=next(s for s in range(12) if s not in already)
    terminal['host_park_events'].append(dict(slot=terminal_slot,
        next_cycle=terminal['cycles'],
        anchor=terminal['initial']['position'][terminal_slot]+960))
    assert capture._check_basis(terminal)['active_target8_rows']==24368
    wrong_terminal=copy.deepcopy(terminal)
    wrong_terminal['host_park_events'][-1]['anchor']+=1
    try:capture._check_basis(wrong_terminal)
    except AssertionError:negatives.append('wrong_terminal_anchor')
    else:raise AssertionError('wrong terminal Host anchor admitted')
    early_terminal=copy.deepcopy(terminal)
    early_terminal['host_park_events'][-1]['next_cycle']=1
    try:capture._check_basis(early_terminal)
    except AssertionError:negatives.append('early_terminal_park')
    else:raise AssertionError('early Host park admitted')
    assert len(negatives)==13 and positive['active_target8_rows']==24368
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(dict(status='offline_collector_checker_preflight_pass',
        collector_sha256=hashlib.sha256(Path(capture.__file__).read_bytes()).hexdigest(),
        trace_sha256=hashlib.sha256(TRACE.read_bytes()).hexdigest(),
        runtime_sha256=hashlib.sha256(RUNTIME.read_bytes()).hexdigest(),
        target_checkpoint_cycles=positive['target_checkpoint_cycles'],
        active_target8_rows=positive['active_target8_rows'],
        negative_mutations_rejected=negatives,
        scope='old Run566 A0 CPU trace; DSpark query metadata in fixture synthetic',
        live_ready=False),indent=2)+'\n')
    print('Run597 CPU selftest PASS')

if __name__=='__main__':main()
