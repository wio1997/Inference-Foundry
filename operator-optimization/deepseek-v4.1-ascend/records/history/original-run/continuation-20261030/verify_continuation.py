#!/usr/bin/env python3
"""Check cross-source integrity, interval arithmetic, case validity and report links.

This does not prove causal optimization or independently recollect NPU data.
"""
import hashlib
import json
from pathlib import Path
import re
import statistics

ROOT=Path(__file__).resolve().parent
read=lambda n:json.loads((ROOT/n).read_text())
graphs=read('graph-executions.json')
old=json.loads((ROOT.parent/'diagnostic-20261029/official-analysis/operators.json').read_text())
old_hash={x['source_dir']:x['sha256']['kernel_details.csv'] for x in old['rows']}
assert graphs['source_count']==len(graphs['rows'])==32
assert {(x['window'],x['rank']) for x in graphs['rows']}=={(w,r) for w in (1,2) for r in range(16)}
assert all(x['kernel_sha256']==old_hash[x['source_dir']] for x in graphs['rows'])
main={}
for x in graphs['rows']:
    gg=[g for g in x['model_streams'] if all(e['main_shape_compatible_40_moe'] for e in g['executions'])]
    assert len(gg)==1
    es=gg[0]['executions']
    assert len(es)==(5 if x['window']==1 else 12)
    assert len({e['signature_sha256'] for e in es})==1
    for e in es:
        assert e['dominant_signature'] and e['record_count']==e['unique_task_id_count']==1318
        assert e['moe_count']==len(e['moe_tasks'])==40
        assert e['record_span_ns']==e['end_epoch_ns']-e['start_epoch_ns']
        assert 0<e['record_union_ns']<=e['record_span_ns']
    main[(x['window'],x['rank'])]=es

gaps=read('iteration-gaps.json')
assert gaps['source_count']==len(gaps['rows'])==16
for x in gaps['rows']:
    es=main[(2,x['rank'])]
    assert len(x['gaps'])==11
    for i,g in enumerate(x['gaps']):
        assert g['ordinal']==i
        assert g['start_epoch_ns']==es[i]['end_epoch_ns']
        assert g['end_epoch_ns']==es[i+1]['start_epoch_ns']
        assert g['span_ns']==g['end_epoch_ns']-g['start_epoch_ns']>0
        assert 0<g['all_record_union_ns']<=g['span_ns']
        assert all(0<v<=g['span_ns'] for v in g['type_union_ns'].values())

host=read('prepare-input-analysis.json')
assert {x['rank'] for x in host['rows']}=={0,8}
for x in host['rows']:
    trace=next(g['trace'] for g in gaps['rows'] if g['rank']==x['rank'])
    assert x['trace_sha256']==trace['sha256']
    assert len(x['prepare_input_scopes'])==12
    for s in x['prepare_input_scopes']:
        assert s['containment_violations']==0
        assert s['duration_ns']==s['e_ns']-s['s_ns']
        assert s['root_exclusive_recorded_scope_ns']+s['descendant_union_ns']==s['duration_ns']
        assert sum(v['count'] for v in s['by_name'].values())==s['descendant_scope_count']
        assert sum(v['exclusive_sum_ns'] for v in s['by_name'].values())+s['root_exclusive_recorded_scope_ns']==s['duration_ns']
        assert all(v['exclusive_sum_ns']>=0 for v in s['by_name'].values())
for path,source in host['sources'].items():
    assert hashlib.sha256(source['text'].encode()).hexdigest()==source['sha256']
    assert hashlib.sha256((ROOT/'frozen-sources'/path).read_bytes()).hexdigest()==source['sha256']

summary=read('continuation-evidence.json')
span=statistics.median(statistics.median(e['record_span_ns'] for e in main[(2,r)]) for r in range(16))
gap=statistics.median(statistics.median(g['span_ns'] for g in x['gaps']) for x in gaps['rows'])
assert span==summary['graph_windows']['2']['rank_median_span_ns']==37441457
assert gap==summary['rank_median_gap_ns']==35740166

worker=read('worker-validation.json')
expected={'prefix_cache_queries_total':262144,'prefix_cache_hits_total':235904,'prompt_tokens_total':262144,'generation_tokens_total':8192,'request_success_total':8,'num_preemptions_total':0}
assert all(worker['counters']['total'][k]==v for k,v in expected.items())
requests=worker['request_timings'];assert len(requests)==8
assert all(x['usage']['prompt_tokens']==32768 and x['usage']['completion_tokens']==1024 for x in requests)
elapsed=(max(x['end_ns'] for x in requests)-min(x['start_ns'] for x in requests))/1e9
assert elapsed==15.404247 and worker['proc']['complete']
assert worker['stack_files_present']==[]
assert worker['samplers']['proc']['exit_code']==0
assert worker['samplers']['pyspy-EP0']['exit_code']==worker['samplers']['pyspy-EP8']['exit_code']==-15
assert 'health_idle' in worker['final_health'] and 'owner' in worker['final_health']

docs=[ROOT/'CONTINUATION-ANALYSIS.md',ROOT/'LITERATURE-SUPPLEMENT.md',ROOT.parent/'publication-draft/RESEARCH-REPORT.md']
links=[]
for p in docs:
    for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
        if target.startswith(('https://','http://','#')):continue
        file=target.split('#')[0]
        assert (p.parent/file).exists(),(p,target)
        links.append(str((p.parent/file).resolve()))
print(json.dumps({'status':'PASS','csv_hashes_cross_checked':32,'repeated_main_segments':272,'later_gaps':176,'worker_prepare_scopes':24,'local_links_checked':len(links),'worker_stack_status':'FAILED_NOT_USED','scope':'integrity/arithmetic/links; no causal speedup proof'},indent=2))
