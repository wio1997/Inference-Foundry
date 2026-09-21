"""Compare same-offset cold prompts and warm mixed passes for QLI CPU maxima."""
import json
import statistics
from pathlib import Path

root=Path('/data/wio/Inference_Foundry/evidence')
cand=root/'20260920_qli_cpu_max'
base=root/'20260920_qli_pair_baseline'
rows=[]
for offset,cp in ((24,cand/'cold4_unverified.json'),(28,cand/'cold4b_unverified.json')):
    bp=base/f'cold/cold4_offset{offset}.json'
    c=json.loads(cp.read_text());b=json.loads(bp.read_text())
    assert c['summary']['success']==b['summary']['success']==4
    for cr,br in zip(c['requests'],b['requests']):
        assert cr['i']==br['i'] and cr['input_tokens']==br['input_tokens'] and cr['error'] is None and br['error'] is None
        rows.append({'offset':offset,'local_i':cr['i'],'input_tokens':cr['input_tokens'],
                     'candidate_ttft_ms':cr['ttft_ms'],'baseline_ttft_ms':br['ttft_ms'],
                     'ttft_delta_ms':cr['ttft_ms']-br['ttft_ms'],
                     'ttft_delta_pct':100*(cr['ttft_ms']/br['ttft_ms']-1),
                     'candidate_tpot_ms':cr['tpot_ms'],'baseline_tpot_ms':br['tpot_ms']})
changes=[r['ttft_delta_ms'] for r in rows]
summary={'cold':{'paired_n':len(rows),'candidate_mean_ttft_ms':statistics.mean(r['candidate_ttft_ms'] for r in rows),
                 'baseline_mean_ttft_ms':statistics.mean(r['baseline_ttft_ms'] for r in rows),
                 'mean_delta_ms':statistics.mean(changes),'median_delta_ms':statistics.median(changes),
                 'delta_range_ms':[min(changes),max(changes)],
                 'all_improved':all(x<0 for x in changes), 'rows':rows},
         'mixed':{}}
for name,path in (('candidate',cand/'perf/analysis.json'),('paired_baseline',base/'perf/analysis.json')):
    x=json.loads(path.read_text())
    summary['mixed'][name]={'output_tps':[r['output_tps'] for r in x['runs']],
                            'median_output_tps':x['median_output_tps'],
                            'median_ttft_ms':x['median_ttft_ms'],
                            'median_tpot_ms':x['median_tpot_ms']}
summary['mixed']['tps_delta_pct']=100*(summary['mixed']['candidate']['median_output_tps']/summary['mixed']['paired_baseline']['median_output_tps']-1)
(base/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary['mixed'],indent=2))
print(json.dumps({k:v for k,v in summary['cold'].items() if k!='rows'},indent=2))
