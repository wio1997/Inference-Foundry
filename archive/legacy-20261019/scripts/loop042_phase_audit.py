#!/usr/bin/env python3
"""Compare cross-rank absolute timestamps in Run106/107 with Run98 durations."""
import json,statistics
from pathlib import Path
root=Path(__file__).resolve().parents[1]
r118=json.loads((root/'evidence/20260925_loop039_comm/run118/unsynced_arrival.json').read_text())
rows=r118['rows']
by=[]
for cycle in (0,1):
    x=sorted((v for v in rows if v['cycle']==cycle),key=lambda v:v['rank'])
    assert len(x)==8
    earliest=min(x,key=lambda v:v['prepare_target_start_us'])
    latest=max(x,key=lambda v:v['prepare_target_start_us'])
    first_early=min(x,key=lambda v:v['first_collective_start_us'])
    first_late=max(x,key=lambda v:v['first_collective_start_us'])
    end_skew=(max(v['first_collective_end_us'] for v in x)-min(v['first_collective_end_us'] for v in x))/1000
    by.append({'cycle':cycle,'prepare_earliest_rank':earliest['rank'],'prepare_latest_rank':latest['rank'],
               'prepare_skew_ms':(latest['prepare_target_start_us']-earliest['prepare_target_start_us'])/1000,
               'first_collective_earliest_rank':first_early['rank'],'first_collective_latest_rank':first_late['rank'],
               'first_collective_start_skew_ms':(first_late['first_collective_start_us']-first_early['first_collective_start_us'])/1000,
               'first_collective_end_skew_ms':end_skew,
               'latest_prepare_to_first_collective_ms':(first_late['first_collective_start_us']-latest['prepare_target_start_us'])/1000})
steady=[]
for rank in range(8):
    d=json.loads((root/f'evidence/20260924_loop036_metadata/run98/dag/rank{rank}.json').read_text())
    for cycle in range(64,256):
        stage=d['runtime_stage_ms'][cycle]
        steady.append({'rank':rank,'cycle':cycle,'target_ms':stage['target'],'proposer_ms':stage['proposer'],'prepare_ms':stage['prepare_target']})
spreads=[]
for cycle in range(64,256):
    group=[x for x in steady if x['cycle']==cycle]
    spreads.append({'cycle':cycle,'target_duration_spread_ms':max(x['target_ms'] for x in group)-min(x['target_ms'] for x in group),
                    'proposer_duration_spread_ms':max(x['proposer_ms'] for x in group)-min(x['proposer_ms'] for x in group)})
out={'run':'run136','trace_source':'Run106 unsynchronized profiler, two cycles only','trace_cycles':by,
     'steady_source':'Run98 unprofiled DAG per-rank device event durations, cycles64-255',
     'steady_target_duration_spread_median_ms':statistics.median(x['target_duration_spread_ms'] for x in spreads),
     'steady_proposer_duration_spread_median_ms':statistics.median(x['proposer_duration_spread_ms'] for x in spreads),
     'critical_limit':'Run98 saves per-rank durations but no shared absolute host or NPU timestamps. Rank duration spreads cannot reveal absolute phase offset or prove whether late arrival persists in unprofiled steady cycles. Run106 has only two profiled cycles and cannot establish steady product rank order.',
     'decision':'Collect lightweight per-rank epoch/monotonic time at prepare, target, proposer boundaries for legal 8-rank steady c12 cohort; retain no forced device synchronization. Then decide whether late rank can be advanced.',
     'steady_spreads':spreads}
p=root/'evidence/20260925_loop042_phase/run136/phase_audit.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:out[k] for k in ('trace_cycles','steady_target_duration_spread_median_ms','steady_proposer_duration_spread_median_ms','critical_limit')},indent=2))
