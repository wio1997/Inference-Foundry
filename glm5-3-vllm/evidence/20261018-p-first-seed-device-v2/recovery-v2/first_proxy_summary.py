from pathlib import Path
import json,math,statistics
r=Path('/Users/wio/work/Inference-Foundry-execution-plan');e=r/'glm5-3-vllm/evidence/20261018-p-first-seed-device-v2/recovery-v2';j=json.loads(next((e/'first-proxy-formal').rglob('metrics.json')).read_text());assert j['completed'] and len(j['rows'])==36 and all(v['exact_pass'] and v['text_matches_control'] for v in j['rows']);rows=[]
for n in (256,257):
 modes={}
 for m in ('control','seed','publish-seed'):
  rs=[v for v in j['rows'] if v['input_tokens']==n and v['mode']==m];assert len(rs)==6
  for v in rs:
   v['second_token_ms']=(v['arrival_ns'][1]-v['sent_ns'])/1e6;v['first_second_gap_ms']=(v['arrival_ns'][1]-v['arrival_ns'][0])/1e6
  modes[m]={key:{'median':statistics.median(v[key] for v in rs),'values':[v[key] for v in rs]} for key in ('TTFT_ms','second_token_ms','last_token_ms','TPOT_ms','first_second_gap_ms')};modes[m]['median_arrival_ms']=[statistics.median((v['arrival_ns'][i]-v['sent_ns'])/1e6 for v in rs) for i in range(64)]
 ratios={m:{key:modes[m][key]['median']/modes['control'][key]['median'] for key in ('TTFT_ms','second_token_ms','last_token_ms','TPOT_ms','first_second_gap_ms')} for m in ('seed','publish-seed')};delta=[b-a for a,b in zip(modes['control']['median_arrival_ms'],modes['publish-seed']['median_arrival_ms'])];rows.append({'N':n,'modes':modes,'ratio_to_control':ratios,'publish_minus_control_arrival_ms':delta,'publish_later_median_positions_one_based':[i+1 for i,x in enumerate(delta) if x>0]})
report={'exact_token_text_usage_pass':36,'samples_each_case_mode':6,'rows':rows,'geomean_ratios':{m:{key:math.sqrt(math.prod(row['ratio_to_control'][m][key] for row in rows)) for key in rows[0]['ratio_to_control'][m]} for m in ('seed','publish-seed')},'scope':'Same experimental CPU frontend with three modes, real wire timestamps, two fixed cold C1 fixtures only. Publishing P first changes the mean-TPOT origin; do not conflate it with steady decode speed or backend phase counters. General strict-bitwise and C4 gates remain separate.'};(e/'first-proxy-performance-summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'geomean_ratios':report['geomean_ratios']}))
for row in rows:print(json.dumps({'N':row['N'],'medians':{m:{key:round(v['median'],3) for key,v in vals.items() if key!='median_arrival_ms'} for m,vals in row['modes'].items()},'publish_later_median_positions':row['publish_later_median_positions_one_based']}))
