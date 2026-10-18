from pathlib import Path
import json,math,statistics
r=Path('/Users/wio/work/Inference-Foundry-execution-plan');e=r/'glm5-3-vllm/evidence/20261018-p-first-seed-device-v2/recovery-v2';j=json.loads(next((e/'perf-c1-formal').rglob('metrics.json')).read_text());assert j['completed'] and j['passed'] and len(j['rows'])==36
rows=[]
for n in (256,257):
 modes={}
 for m in ('control','adopt','seed'):
  rs=[v for v in j['rows'] if v['input_tokens']==n and v['mode']==m];assert len(rs)==6 and all(v['matches_native_control'] for v in rs)
  for v in rs:
   v['TPOT_ms']=(v['last_token_ms']-v['TTFT_ms'])/63;v['D_TTFT_ms']=(v['token_arrival_ns'][0]-v['D_send_ns'])/1e6;v['D_last_ms']=(v['token_arrival_ns'][-1]-v['D_send_ns'])/1e6
  modes[m]={key:{'median':statistics.median(v[key] for v in rs),'mean':statistics.mean(v[key] for v in rs),'values':[v[key] for v in rs]} for key in ('TTFT_ms','last_token_ms','TPOT_ms','D_TTFT_ms','D_last_ms')}
 ratios={m:{key:modes[m][key]['median']/modes['control'][key]['median'] for key in modes[m]} for m in ('adopt','seed')};rows.append({'N':n,'modes':modes,'ratio_to_control':ratios})
report={'exact_rows':36,'exact_pass':True,'samples_each_case_mode':6,'rows':rows,'geomean_ratios':{m:{key:math.sqrt(math.prod(row['ratio_to_control'][m][key] for row in rows)) for key in rows[0]['modes'][m]} for m in ('adopt','seed')},'scope':'Two fixed correctness-selected cold C1 sum fixtures, native PD P/D direct requests. No C4 capacity, production SLO or general strict-bitwise acceptance; P export copy/log costs included.'};(e/'performance-summary-c1.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'geomean_ratios':report['geomean_ratios']}))
for row in rows:print(json.dumps({'N':row['N'],'medians':{m:{k:round(v['median'],3) for k,v in ks.items()} for m,ks in row['modes'].items()}}))
