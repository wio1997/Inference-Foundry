"""Summarize prefill-only QLI candidate against same-prompt original source."""
import json
import statistics
from pathlib import Path

root=Path('/data/wio/Inference_Foundry/evidence')
cand=root/'20260920_qli_prefill_only'
base=root/'20260920_qli_pair_baseline'
rows=[]
for offset in (24,28,40,44):
    cp=cand/f'cold/cold4_offset{offset}.json'
    bp=base/(f'cold/cold4_offset{offset}.json' if offset<40 else f'cold_more/cold4_offset{offset}.json')
    c=json.loads(cp.read_text());b=json.loads(bp.read_text())
    assert c['summary']['success']==b['summary']['success']==4
    for cr,br in zip(c['requests'],b['requests']):
        assert cr['i']==br['i'] and cr['input_tokens']==br['input_tokens']
        assert cr['error'] is None and br['error'] is None
        rows.append({'prompt_index':offset+cr['i'],'input_tokens':cr['input_tokens'],
                     'candidate_ttft_ms':cr['ttft_ms'],'baseline_ttft_ms':br['ttft_ms'],
                     'delta_ms':cr['ttft_ms']-br['ttft_ms'],
                     'delta_pct':100*(cr['ttft_ms']/br['ttft_ms']-1)})
changes=[r['delta_ms'] for r in rows]
old=json.loads((base/'perf/analysis.json').read_text())
new=json.loads((cand/'perf/analysis.json').read_text())
out={'cold':{'n':len(rows),'candidate_mean_ttft_ms':statistics.mean(r['candidate_ttft_ms'] for r in rows),
             'baseline_mean_ttft_ms':statistics.mean(r['baseline_ttft_ms'] for r in rows),
             'mean_delta_ms':statistics.mean(changes),
             'mean_delta_pct':100*(statistics.mean(r['candidate_ttft_ms'] for r in rows)/statistics.mean(r['baseline_ttft_ms'] for r in rows)-1),
             'median_delta_ms':statistics.median(changes),'delta_range_ms':[min(changes),max(changes)],
             'all_improved':all(x<0 for x in changes),'rows':rows},
     'mixed':{'candidate_tps':[r['output_tps'] for r in new['runs']],
              'baseline_tps':[r['output_tps'] for r in old['runs']],
              'candidate_median_tps':new['median_output_tps'],
              'baseline_median_tps':old['median_output_tps'],
              'median_tps_delta_pct':100*(new['median_output_tps']/old['median_output_tps']-1),
              'candidate_median_ttft_ms':new['median_ttft_ms'],
              'baseline_median_ttft_ms':old['median_ttft_ms'],
              'candidate_median_tpot_ms':new['median_tpot_ms'],
              'baseline_median_tpot_ms':old['median_tpot_ms']}}
(cand/'comparison.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out['cold'].items() if k!='rows'},indent=2))
print(json.dumps(out['mixed'],indent=2))
