#!/usr/bin/env python3
"""Reconcile saved task intervals on one rank/run target-span clock; CPU only."""
import csv, hashlib, json, statistics
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

root = Path(__file__).resolve().parents[2]
paths = {
    'windows': 'captured-layer-attribution/module-call-metrics.json',
    'indexer': 'agents/diagnosis/indexer-layer-window-occupancy.json',
    'mc2': 'mc2-arrival-analysis-v2/rank_target_calls.csv',
    'timeline': 'mc2-preceding-analysis/rank_layer0_timeline.csv',
    'matmul': 'agents/diagnosis/16rank-engram-matmul-metrics.json',
    'prior': 'agents/diagnosis/mainstream-occupancy.json',
}
expected = {
    'windows':'d98aac57fbf07885e5bfe405e9f50c2e27bc22d4d5aba27e80b95e1aecf4de5a',
    'indexer':'ca1d1dabed7cede2a95bc1ac117d24ad68f8fd2368f32c3aa225583cb3e9afc8',
    'mc2':'19ff5196289b93ee53bcb4a6a27946dbff8c84ebd28988fd40b4fe0bacefbe17',
    'timeline':'d3e0d9cbeddcde97df34dcf36b386cc8201b71c82dde8ab4ed42dd713fde7dfb',
    'matmul':'35ff1b007a471903324d96de5eeb67b80a13251a4c4dadd8619eead77c23db1d',
    'prior':'00b2bb3ec02b33a78a2924c27b720f08b6896e68a46a7e976fc3be009d9c412e',
}
for k,p in paths.items():
    assert hashlib.sha256((root/p).read_bytes()).hexdigest()==expected[k], k
read=lambda k:json.loads((root/paths[k]).read_text())
def csvrows(k):
    with (root/paths[k]).open() as f: return list(csv.DictReader(f))
windows=defaultdict(list)
for w in read('windows'):
    if w['run_in_model'] in (1,2,3): windows[w['rank'],w['run_in_model']].append(w)
indexers=defaultdict(list)
for x in read('indexer')['rows']: indexers[x['rank'],x['run']].append(x)
matmuls=defaultdict(list)
for x in read('matmul')['samples']:
    x=x['target']; matmuls[x['rank'],x['run']].append(x)
mc2={(x['rank'],int(x['target_call_ordinal'])):x for x in csvrows('mc2')}
timeline={(x['rank'],int(x['captured_run_in_model'])):int(x['target_forward_ordinal'])
          for x in csvrows('timeline') if x['interior_runs_1_2_3']=='True'}
prior={ (x['rank'],x['run']):x for x in read('prior')['rank_run_rows']}
assert len(windows)==len(indexers)==len(matmuls)==len(timeline)==len(prior)==48
assert set(windows)==set(indexers)==set(matmuls)==set(timeline)==set(prior)
rows=[]
for key in sorted(windows):
    ws=sorted(windows[key],key=lambda x:x['layer']); ix=indexers[key]; mm=matmuls[key]
    assert [w['layer'] for w in ws]==list(range(40)) and len(ix)==8 and len(mm)==2
    assert all(a['window_end_ns']<=b['window_start_ns'] for a,b in zip(ws,ws[1:]))
    stream,model=ws[0]['main_stream_id'],ws[0]['model_id']
    assert model=='49' and all(w['main_stream_id']==stream and w['model_id']==model for w in ws)
    events=[]
    for w in ws:
        x=mc2[key[0],timeline[key]*40+w['layer']]
        assert int(x['source_layer'])==w['layer'] and x['stream_id']==stream and x['model_id']==model
        a,b=int(x['start_ns']),int(x['end_ns']); assert w['window_start_ns']<=a<b<=w['window_end_ns']
        events.append(('mc2',a,b))
    for x in ix:
        w=ws[x['layer']]; assert x['model_id']==model and x['main_stream_id']==stream
        a,b=x['indexer']['start_ns'],x['indexer']['end_ns']
        assert w['window_start_ns']<=a<b<=w['window_end_ns']; events.append(('indexer',a,b))
    assert {x['layer'] for x in ix}=={2,8,14,20,24,28,32,36}
    assert sorted(x['after_layer'] for x in mm)==[0,13]
    for x in mm:
        assert str(x['model'])==model and str(x['stream'])==stream
        a,b=x['start_ns'],x['end_ns']; layer=x['after_layer']
        assert ws[layer]['window_end_ns']<=a<b<=ws[layer+1]['window_start_ns']
        events.append(('engram_shape_matmul',a,b))
    events.sort(key=lambda x:x[1]); assert all(x[2]<=y[1] for x,y in zip(events,events[1:]))
    start,end=ws[0]['window_start_ns'],ws[-1]['window_end_ns']; span=end-start
    sums={name:sum(b-a for typ,a,b in events if typ==name) for name in ('mc2','indexer','engram_shape_matmul')}
    window=sum(x['window_ns'] for x in ws)
    assert int(Decimal(str(prior[key]['window_us']))*1000)==window
    assert int(Decimal(str(prior[key]['mc2_us']))*1000)==sums['mc2']
    assert int(Decimal(str(prior[key]['indexer_us']))*1000)==sums['indexer']
    rows.append({'rank':key[0],'run':key[1],'span_ns':span,'window_total_ns':window,
                 'duration_ns':sums,'selected_total_ns':sum(sums.values()),
                 'uncovered_by_selected_ns':span-sum(sums.values()),
                 'span_fraction':{k:v/span for k,v in sums.items()},
                 'selected_fraction':sum(sums.values())/span})
summaries=[]
for run in (1,2,3):
    group=[x for x in rows if x['run']==run]; assert len(group)==16
    summaries.append({'run':run,'fractions':{name:{'min':min(x['span_fraction'][name] for x in group),
        'median':statistics.median(x['span_fraction'][name] for x in group),
        'max':max(x['span_fraction'][name] for x in group)} for name in sums},
        'selected_fraction_median':statistics.median(x['selected_fraction'] for x in group),
        'mc2_larger_than_both_count':sum(x['duration_ns']['mc2']>max(x['duration_ns']['indexer'],x['duration_ns']['engram_shape_matmul']) for x in group),
        'indexer_larger_than_matmul_count':sum(x['duration_ns']['indexer']>x['duration_ns']['engram_shape_matmul'] for x in group)})
print(json.dumps({'input_sha256':expected,'scope':'48 same-rank same-run model 49 main-stream target spans; 40 MC2, 8 Indexer, 2 Engram-shape candidate matmuls per span',
 'checks':'All 50 selected events independently matched to same model and stream and checked pairwise disjoint; each MC2 and indexer within own layer and matmuls in gaps; prior totals reproduced exactly.',
 'limits':['Only selected types; no all-operator ranking, unknown remainder is not idle.', 'Matmul weight identity is unconfirmed.', 'All fractions use same-rank per-run layer0 start to layer39 end; different ranks are not summed.', 'Other streams or host may overlap; not end-to-end critical path or predicted savings.', 'run ordinal is not a proven matched request; profiled workload may differ from baseline.'],
 'per_run':summaries,'rows':rows,'no_device_work':True},ensure_ascii=False,indent=2))
