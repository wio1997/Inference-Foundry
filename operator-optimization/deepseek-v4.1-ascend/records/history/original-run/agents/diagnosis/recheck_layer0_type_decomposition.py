#!/usr/bin/env python3
"""Verify stage159 against frozen target summaries. No device or server access."""
import csv,hashlib,json
from decimal import Decimal as D
from pathlib import Path
T=Path(__file__).resolve().parents[2]
paths={'stage159':T/'agents/diagnosis/layer0-dp-type-counters.json','old_pooled':T/'agents/diagnosis/target-dp-aiv-comparison.json',
       'old_samples':T/'agents/diagnosis/layer0-dp-existing.json','calls':T/'captured-layer-attribution/module-call-metrics.json',
       'task_rows':T/'mc2-arrival-analysis-v2/rank_target_calls.csv','self':Path(__file__).resolve()}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(paths['stage159'])=='0482575955229199d9b2c25e1cff2590d82a27d8a91f89e0d8f3418e7be37fad'
x=json.loads(paths['stage159'].read_text());o=json.loads(paths['old_pooled'].read_text());y=json.loads(paths['old_samples'].read_text())
cs={(r['rank'],r['run_in_model']):r for r in json.loads(paths['calls'].read_text()) if r['layer']==0 and r['run_in_model'] in (1,2,3)}
samples={(r['rank'],r['run']):r for r in y['samples']}
with paths['task_rows'].open() as f:rr=list(csv.DictReader(f))
assert len(cs)==len(samples)==len(x['mc2_by_window'])==len(x['window_coverage'])==48
assert all(c['rows']==38 for c in x['window_coverage'])
shapes=set()
for m in x['mc2_by_window']:
 k=(m['rank'],m['run']); assert k in cs and k in samples
 c=cs[k];s=samples[k];assert m['stream']==c['main_stream_id'] and abs(s['mc2_duration_us']*1000-m['duration_ns'])<.001
 matched=[r for r in rr if r['rank']==m['rank'] and r['task_id']==m['task_id'] and r['source_layer']=='0' and abs(float(r['duration_us'])*1000-m['duration_ns'])<.001]
 assert len(matched)==1 and matched[0]['physical_input_rows']=='3'
 shapes.add(m['input_shape'])
assert shapes=={'"3,5120;24,5120,4608;24,2304,5120;3,6;110592;122880;3,6;3"'}
out=[]
for dp in ('dp0','dp1'):
 rows=[r for r in x['task_types'] if r['dp']==dp]; mc=next(r for r in rows if r['type']=='DispatchFFNCombine')
 assert mc['rows']==24 and sum(r['rows'] for r in rows)==24*38
 old=next(r for r in o['pooled'] if r['dp']==dp and r['layer']==0)
 for dev,tag in (('aic','mac'),('aiv','vec')):
  den=sum(D(r[dev+'_cycles']) for r in rows); num=sum(D(r[dev+'_'+tag+'_cycle_proxy']) for r in rows)
  assert den==D(old['metrics'][dev+'_'+tag]['denominator']) and num==D(old['metrics'][dev+'_'+tag]['numerator'])
  mden=D(mc[dev+'_cycles']);mnum=D(mc[dev+'_'+tag+'_cycle_proxy']);assert den>mden>0
  out.append({'dp':dp,'engine':dev,'all_denominator':str(den),'mc2_denominator':str(mden),
              'mc2_share':float(mden/den),'mc2_ratio':float(mnum/mden),'other_ratio':float((num-mnum)/(den-mden))})
print(json.dumps({'inputs_sha256':{k:sha(v) for k,v in paths.items()},'scope':'48 frozen layer0 task samples; exact per-rank event matching and grouped proxy math',
 'group_comparison':out,'shape':next(iter(shapes)),'limits':['Attribution and proportions describe reported counters, not causality, actual busy time or end-to-end improvement','same shape does not prove same live token or expert distributions','MIX counters and task duration may overlap'], 'no_device_work':True},ensure_ascii=False,indent=2))
