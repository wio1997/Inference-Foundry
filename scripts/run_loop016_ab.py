#!/usr/bin/env python3
import hashlib
import json
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

root=Path('/data/wio/Inference_Foundry')
out=root/'evidence/20260920_loop016_direct_scatter/service_ab'
out.mkdir(parents=True,exist_ok=True)
flag=out/'candidate.flag'
dataset='/data/wio/vllm_ascend_26/datasets/GSM8K-in32768-num48-DeepSeek-V4-Flash-0731-w4a8.jsonl'
base='http://127.0.0.1:8080'
def stamp():
    return datetime.now(timezone.utc).isoformat()
def run(command, log):
    with (out/log).open('w') as f:
        subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
def bench(label, offset):
    result=out/f'{label}_{offset}.json'
    command=['docker','exec','dsv4ab','python3',str(root/'scripts/bench.py'),
             '--dataset',dataset,'--out',str(result),'--offset',str(offset),
             '--limit','1','--concurrency','1','--max-tokens','128']
    run(command,f'{label}_{offset}.log')
    d=json.loads(result.read_text())
    assert d['summary']['success']==1
    return d['requests'][0]['ttft_ms']
def metrics():
    return requests.get(base+'/metrics',timeout=30).text
for _ in range(240):
    try:
        if requests.get(base+'/health',timeout=2).status_code==200: break
    except requests.RequestException: pass
    time.sleep(10)
else:
    raise RuntimeError('service readiness timeout')
marks={'ready_utc':stamp()}
flag.unlink(missing_ok=True)
run(['docker','exec','dsv4ab','python3',str(root/'scripts/check_functional.py'),
     '--golden',str(root/'evidence/20260920_qli_prefill_only/perf/golden4.json'),
     '--out',str(out/'functional_check.json')],'functional.log')
marks['functional_done_utc']=stamp()
(out/'metrics_before.txt').write_text(metrics())
flag.write_text('candidate enabled\n')
try:
    run(['docker','exec','dsv4ab','python3',str(root/'scripts/golden.py'),
         '--dataset',dataset,'--out',str(out/'candidate_golden4.json')],'candidate_golden4.log')
finally:
    flag.unlink(missing_ok=True)
ref=json.loads((root/'evidence/20260920_qli_prefill_only/perf/golden4.json').read_text())
got=json.loads((out/'candidate_golden4.json').read_text())
checks=[{'index':a['index'],'prompt_match':a['prompt_sha256']==b['prompt_sha256'],
         'output_match':a['output_sha256']==b['output_sha256'],
         'tokens_match':a['completion_tokens']==b['completion_tokens']} for a,b in zip(ref,got)]
(out/'candidate_parity.json').write_text(json.dumps(checks,indent=2)+'\n')
if len(checks)!=4 or not all(all(v for k,v in c.items() if k!='index') for c in checks):
    raise RuntimeError('candidate golden parity failed')
trace_files=list(out.glob('rank_*.txt'))
if len(trace_files)!=8:
    raise RuntimeError(f'expected fast-path trace on 8 ranks, got {len(trace_files)}')
marks['candidate_parity_done_utc']=stamp()
pairs=[]
for i in range(8):
    sequence=['baseline','candidate'] if i%2==0 else ['candidate','baseline']
    times={}
    for variant in sequence:
        offset=(8+i) if variant=='baseline' else (24+i)
        if variant=='candidate': flag.write_text('candidate enabled\n')
        else: flag.unlink(missing_ok=True)
        try: times[variant]=bench(variant,offset)
        finally: flag.unlink(missing_ok=True)
    pairs.append({'pair':i,'baseline_offset':8+i,'candidate_offset':24+i,
                  'baseline_ttft_ms':times['baseline'],'candidate_ttft_ms':times['candidate'],
                  'delta_ms':times['candidate']-times['baseline']})
    (out/'pairs_live.json').write_text(json.dumps(pairs,indent=2)+'\n')
marks['ab_done_utc']=stamp()
(out/'metrics_after.txt').write_text(metrics())
summary={'n_pairs':len(pairs),
         'baseline_mean_ttft_ms':statistics.mean(p['baseline_ttft_ms'] for p in pairs),
         'candidate_mean_ttft_ms':statistics.mean(p['candidate_ttft_ms'] for p in pairs),
         'mean_delta_ms':statistics.mean(p['delta_ms'] for p in pairs),
         'median_delta_ms':statistics.median(p['delta_ms'] for p in pairs),
         'pairs_improved':sum(p['delta_ms']<0 for p in pairs)}
(out/'summary.json').write_text(json.dumps({'summary':summary,'pairs':pairs,'marks':marks},indent=2)+'\n')
print(json.dumps(summary,indent=2))
