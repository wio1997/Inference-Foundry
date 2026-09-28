#!/usr/bin/env python3
"""Bounded diagnostic fixed-algorithm lineage collector for Run597.

All tensor-to-CPU materialization and JSON publication happen after the
ordinary cohort drain. Hot-loop additions are one preallocated draft copy,
Host park scalars already computed by serving, and sparse device clones.
This is a diagnostic W0, not an unperturbed formal TPS result.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch

SELECTED={0,1,64,128,192,256,300}

def bind_cohort(runtime, cohort, req_ids, run_ts):
    enabled=(5<=cohort<=8)
    runtime._run597_context=dict(enabled=enabled,cohort=int(cohort),
        req_ids=list(req_ids),run_ts=run_ts)
    state=runtime.state
    state._run597_enabled=enabled
    state._run597_next_postpark=False
    state._run597_capture_this_cycle=False
    state._run597_target_witnesses=[]
    state._run597_dspark_witnesses=[]
    state._run597_draft_model_witnesses=[]
    runtime._run597_branch_history=[]
    runtime.proposer.proposer._run597_state=None

def branch(runtime, use_scheduled):
    if not getattr(runtime.state,'_run597_enabled',False):return
    runtime._run597_branch_history.append(dict(
        cycle=int(runtime.state.cycle_index),
        scheduled_target=bool(use_scheduled),
        schedule_mode=str(runtime._schedule_mode)))

def target_prepared(runtime):
    state=runtime.state
    if not getattr(state,'_run597_enabled',False):return
    cycle=int(state.cycle_index)
    selected=(cycle in SELECTED or state._run597_next_postpark)
    state._run597_capture_this_cycle=selected
    state._run597_next_postpark=False
    if not selected:return
    state._run597_target_witnesses.append(dict(
        cycle=cycle,input_ids=state.target_input_ids[:96].clone(),
        positions=state.target_positions[:96].clone(),
        seq_lens=state.target_seq_lens[:12].clone()))

def dspark_prepared(proposer,state,common,token_indices,sample_indices,
                    num_rejected,target_token_ids,target_positions):
    proposer._run597_state=(state if getattr(state,'_run597_capture_this_cycle',False)
                            else None)
    if not getattr(state,'_run597_capture_this_cycle',False):return
    state._run597_dspark_witnesses.append(dict(
        cycle=int(state.cycle_index),
        token_indices=token_indices.clone(),
        sample_indices=sample_indices.clone(),
        num_rejected=num_rejected.clone(),
        query_start_loc=common.query_start_loc.clone(),
        seq_lens=common.seq_lens.clone(),
        target_token_ids=target_token_ids.clone(),
        target_positions=target_positions.clone()))

def draft_model_inputs(proposer,num_query_total,sample_indices,common):
    state=getattr(proposer,'_run597_state',None)
    proposer._run597_state=None
    if state is None:return
    gids=sorted(proposer._per_group_query_slot_mapping_buffers)
    state._run597_draft_model_witnesses.append(dict(
        cycle=int(state.cycle_index),
        num_query_total=int(num_query_total),
        num_context=int(proposer._dflash_num_context),
        num_query_per_req=int(proposer.num_query_per_req),
        sample_from_anchor=bool(proposer.sample_from_anchor),
        parallel_drafting_token_id=int(proposer.parallel_drafting_token_id),
        query_input_ids=proposer.input_ids[:num_query_total].clone(),
        query_positions=proposer.positions[:num_query_total].clone(),
        context_positions=proposer._context_positions_buffer[
            :proposer._dflash_num_context].clone(),
        sample_indices=sample_indices.clone(),
        query_start_loc=common.query_start_loc.clone(),
        seq_lens=common.seq_lens.clone(),
        causal=bool(common.causal),
        group_query_slots={str(gid):proposer._per_group_query_slot_mapping_buffers[
            gid][:num_query_total].clone() for gid in gids},
        group_context_slots={str(gid):proposer._per_group_context_slot_mapping_buffers[
            gid][:proposer._dflash_num_context].clone() for gid in gids}))

def _cpu_value(value):
    if isinstance(value,torch.Tensor):return value.cpu().tolist()
    if isinstance(value,dict):return {k:_cpu_value(v) for k,v in value.items()}
    if isinstance(value,list):return [_cpu_value(v) for v in value]
    return value

def _prefix(row):
    assert len(row)==8
    n=next((i for i,x in enumerate(row) if x<0),8)
    assert 1<=n<=8 and all(x>=0 for x in row[:n]) and all(x==-1 for x in row[n:])
    return n

def _check_basis(record):
    cycles=record['cycles'];tokens=record['token_history']
    counts=record['count_history'];drafts=record['draft_history']
    assert len(tokens)==len(counts)==len(drafts)==cycles
    initial=record['initial']
    origin=initial['position'][:]
    position=origin[:]
    last=initial['last_token'][:]
    draft=[x[:] for x in initial['draft']]
    events={};event_slots=set()
    for e in record['host_park_events']:
        key=(e['next_cycle'],e['slot'])
        assert key not in events
        assert e['slot'] not in event_slots
        assert 1<=e['next_cycle']<=cycles and 0<=e['slot']<12
        assert e['anchor']==origin[e['slot']]+960
        events[key]=e
        event_slots.add(e['slot'])
    parked=[False]*12
    emitted=[0]*12
    target={r['cycle']:r for r in record['target_witnesses']}
    dspark={r['cycle']:r for r in record['dspark_witnesses']}
    model={r['cycle']:r for r in record['draft_model_witnesses']}
    assert target and set(target)==set(dspark)==set(model)
    assert len(target)==len(record['target_witnesses'])
    assert len(dspark)==len(record['dspark_witnesses'])
    assert len(model)==len(record['draft_model_witnesses'])
    assert {c for c in SELECTED if c<cycles}<=set(target)
    postpark=[e['next_cycle'] for e in record['host_park_events']
              if e['next_cycle']<cycles]
    if postpark:assert min(postpark) in target
    assert len(record['branch_history'])==cycles
    for cycle in range(cycles):
        for slot in range(12):
            event=events.get((cycle,slot))
            if event:
                assert not parked[slot] and emitted[slot]>=1024
                assert position[slot]>event['anchor']
                position[slot]=event['anchor'];parked[slot]=True
        ids=[[last[s]]+draft[s] for s in range(12)]
        positions=[[position[s]+j for j in range(8)] for s in range(12)]
        if cycle in target:
            row=target[cycle]
            assert row['input_ids']==[v for x in ids for v in x]
            assert row['positions']==[v for x in positions for v in x]
            assert row['seq_lens']==[position[s]+8 for s in range(12)]
            d=dspark[cycle]
            flat_ids=row['input_ids'];flat_pos=row['positions']
            indices=d['token_indices']
            assert indices==list(range(96))
            assert d['query_start_loc']==list(range(0,97,8))
            assert d['seq_lens']==row['seq_lens']
            raw=[_prefix(tokens[cycle][s]) for s in range(12)]
            assert d['num_rejected']==[8-n for n in raw]
            assert d['sample_indices']==[8*s+n-1 for s,n in enumerate(raw)]
            assert d['target_token_ids']==[flat_ids[i] for i in indices]
            assert d['target_positions']==[flat_pos[i] for i in indices]
            m=model[cycle]
            Q=m['num_query_per_req']
            assert Q in (7,8) and m['num_query_total']==12*Q
            assert m['num_context']==96
            assert m['sample_from_anchor']==(Q==7)
            assert m['context_positions']==flat_pos
            assert m['query_start_loc']==list(range(0,12*Q+1,Q))
            assert m['seq_lens']==[position[s]+raw[s]+Q for s in range(12)]
            expected_query_pos=[position[s]+raw[s]+q
                                for s in range(12) for q in range(Q)]
            assert m['query_positions']==expected_query_pos
            next_last=[tokens[cycle][s][raw[s]-1] if not parked[s] else last[s]
                       for s in range(12)]
            expected_query_ids=[next_last[s] if q==0 else m['parallel_drafting_token_id']
                                for s in range(12) for q in range(Q)]
            assert m['query_input_ids']==expected_query_ids
            expected_samples=[s*Q+q for s in range(12)
                              for q in (range(7) if Q==7 else range(1,8))]
            assert m['sample_indices']==expected_samples
            assert (m['group_query_slots'] and
                    all(len(x)==12*Q for x in m['group_query_slots'].values()))
            assert set(m['group_query_slots'])==set(m['group_context_slots'])
            assert all(len(x)==96 for x in m['group_context_slots'].values())
        assert record['branch_history'][cycle]['cycle']==cycle
        for slot in range(12):
            raw=_prefix(tokens[cycle][slot])
            masked=counts[cycle][slot]
            assert masked==(0 if parked[slot] else raw)
            assert len(drafts[cycle][slot])==7
            if not parked[slot]:
                emitted[slot]+=masked
                position[slot]+=masked
                last[slot]=tokens[cycle][slot][masked-1]
            draft[slot]=drafts[cycle][slot][:]
    for slot in range(12):
        terminal=events.get((cycles,slot))
        if terminal:
            assert not parked[slot] and emitted[slot]>=1024
            assert position[slot]>terminal['anchor']
    assert emitted==record['staged_output_counts']
    assert record['generated_output_counts']==[1024]*12
    return dict(target_checkpoint_cycles=sorted(target),
                dspark_checkpoint_cycles=sorted(dspark),
                active_target8_rows=8*sum(
                    sum(counts[c][s]>0 for c in range(cycles)) for s in range(12)),
                current_physical_target_rows=cycles*96)

def save(serving,cycles,initial_last,initial_draft,
         tokens_cpu,counts_cpu,drafts_cpu,generated,staged):
    runtime=serving.runtime
    context=getattr(runtime,'_run597_context',None)
    if not context or not context['enabled']:return
    root=os.getenv('EXTREME_RUN597_DIR')
    if not root:raise RuntimeError('Run597 measured collector directory missing')
    rank=int(torch.distributed.get_rank())
    record=dict(rank=rank,cohort=context['cohort'],req_ids=context['req_ids'],
        run_ts=context['run_ts'],cycles=cycles,
        initial=dict(position=list(serving._initial_positions),
                     last_token=_cpu_value(initial_last),
                     draft=_cpu_value(initial_draft)),
        token_history=_cpu_value(tokens_cpu),count_history=_cpu_value(counts_cpu),
        draft_history=_cpu_value(drafts_cpu),
        host_park_events=list(serving._run597_host_events),
        branch_history=list(runtime._run597_branch_history),
        target_witnesses=_cpu_value(runtime.state._run597_target_witnesses),
        dspark_witnesses=_cpu_value(runtime.state._run597_dspark_witnesses),
        draft_model_witnesses=_cpu_value(runtime.state._run597_draft_model_witnesses),
        generated_output_counts=list(generated),
        staged_output_counts=list(staged),
        boundary=dict(initial_kv_generation='unknown',
                      ordinary_prefill_seed_work='not_captured',
                      sparse_checkpoint_only=True,
                      capture_is_diagnostic_and_perturbing=True))
    record['basis_check']=_check_basis(record)
    path=Path(root);path.mkdir(parents=True,exist_ok=True)
    target=path/f'rank{rank}_cohort{context["cohort"]}.json'
    if target.exists():raise RuntimeError('Run597 duplicate cohort record')
    temp=path/f'.rank{rank}_cohort{context["cohort"]}.tmp'
    with temp.open('x') as file:
        json.dump(record,file,separators=(',',':'))
        file.write('\n');file.flush();os.fsync(file.fileno())
    os.replace(temp,target)
