#!/usr/bin/env python3
"""Analyse formal fence-only run without treating it as a refill result."""
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path('/data/wio/Inference_Foundry')
OUT = ROOT / 'evidence/20260926_loop074_refill/run343'
LOG = ROOT / 'logs/serve_dsv4f-w4a8_8npu_dp1tp8_mlen1M_nomooncake_LOOP036-STATIC-E2E-20260926-2153.log'
lines = LOG.read_text(errors='replace').splitlines()
acq_re = re.compile(r'EXTREME_RUN343_FENCE acquire seq=(\d+) queued_before_submit=(\d+) scheduled=(\d+)')
rel_re = re.compile(r'EXTREME_RUN343_FENCE release seq=(\d+) queued_after_consume=(\d+) bulk=(\w+)')
events = []
for n, line in enumerate(lines, 1):
    ma, mr = acq_re.search(line), rel_re.search(line)
    if ma:
        events.append({'type':'acquire','line':n,'seq':int(ma[1]),'queued':int(ma[2]),'scheduled':int(ma[3])})
    if mr:
        events.append({'type':'release','line':n,'seq':int(mr[1]),'queued':int(mr[2]),'bulk':mr[3]=='True'})
assert len(events) == 32
assert [x['type'] for x in events] == ['acquire','release'] * 16
for i in range(0,32,2):
    a,b = events[i:i+2]
    assert a['seq']==b['seq']==i//2+1 and a['scheduled']==96 and b['bulk']
    assert a['queued']==1 and b['queued']==0

runtime = [json.loads(p.read_text()) for p in sorted((OUT/'runtime').glob('rank*_cohort*.json'))]
assert len(runtime)==128 and all(x['pass'] for x in runtime)
cohorts=defaultdict(list)
for x in runtime: cohorts[x['cohort']].append(x)
assert sorted(cohorts)==list(range(1,17))
for c, rows in cohorts.items():
    assert sorted(x['rank'] for x in rows)==list(range(8))
    assert len({tuple(x['req_ids']) for x in rows})==1
    assert len({x['cycles'] for x in rows})==1
    assert all(x['generated_output_counts']==[1024]*12 for x in rows)
formal=json.loads((OUT/'summary.json').read_text())
assert formal['pass'] and len(formal['runs'])==3
tps=[r['output_tps'] for r in formal['runs']]
baseline=json.loads((ROOT/'evidence/20260924_loop036_metadata/run99/summary.json').read_text())
baseline_tps=[r['output_tps'] for r in baseline['runs']]
repeat_accounting=[]
for repeat in range(3):
    cs=list(range(5+4*repeat,9+4*repeat))
    cycles=sum(cohorts[c][0]['cycles'] for c in cs)
    runtime_wall=sum(max(x['wall_seconds'] for x in cohorts[c]) for c in cs)
    client_wall=formal['runs'][repeat]['duration_s']
    repeat_accounting.append({'repeat':repeat+1,'cycles':cycles,
        'sum_cohort_max_rank_runtime_wall_s':runtime_wall,
        'client_wall_s':client_wall,
        'client_minus_runtime_envelope_s':client_wall-runtime_wall,
        'scope':'cross-clock descriptive residual, not identified removable Product time'})
analysis={
 'status':'valid_original_bulk_fence_only_formal',
 'candidate':'EngineCore conservative total96 schedule fence; --async-scheduling retained; original FixedCohortServing bulk unchanged',
 'fence':{'acquires':16,'releases':16,'all_bulk':True,'queued_predecessor_at_acquire':1,'queue_after_candidate_consume':0,'events':events},
 'runtime':{'rank_cohort_files':len(runtime),'cohorts':len(cohorts),'all_pass':True,'all_full_graph':all(x['target_graph_mode']=='FULL' for x in runtime),
            'formal_cohort_cycles':[next(iter({x['cycles'] for x in cohorts[c]})) for c in range(5,17)]},
 'formal':{'warmup_success':json.loads((OUT/'warmup.json').read_text())['summary']['success'],
           'repeat_tps':tps,'median_tps':statistics.median(tps),
           'run99_repeat_tps':baseline_tps,'run99_median_tps':statistics.median(baseline_tps),
           'median_difference_tps':statistics.median(tps)-statistics.median(baseline_tps),
           'median_difference_percent':100*(statistics.median(tps)/statistics.median(baseline_tps)-1),
           'repeat_accounting':repeat_accounting,
           'same_session_matched_control':False},
 'service':{'source_restored_sha256':json.loads((OUT/'patch_restore.json').read_text())['restored_sha256'],
            'stop_hbm_below_6gb':True,
            'post_benchmark_async_llm_error':'EngineDeadError logged after summary.json timestamp during service stop; no benchmark failure'},
 'limits':['This demonstrates a legal pre-submission fence with original bulk outputs; it does not publish early, admit real new requests during old Runtime or test refill.',
           'Run99 is a prior formal session, not a same-session A/B; 0.70% median difference lies within broad observed run variation and proves neither gain nor zero cost.',
           'Source control-flow plus 16 acquire/release rows protects against a successor submitted through this EngineCore path; worker entry/exit sequence IDs were not instrumented.'],
}
(OUT/'analysis.json').write_text(json.dumps(analysis,indent=2)+'\n')
print(json.dumps({'status':analysis['status'],'median_tps':analysis['formal']['median_tps'],
                  'median_difference_percent':analysis['formal']['median_difference_percent'],
                  'fence_count':analysis['fence']['acquires']},indent=2))
