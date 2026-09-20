#!/usr/bin/env python3
import json
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
import requests

root=Path('/data/wio/Inference_Foundry')
out=root/'evidence/20260920_loop016_direct_scatter/service_ab2'
out.mkdir(parents=True,exist_ok=True)
flag=root/'evidence/20260920_loop016_direct_scatter/service_ab/candidate.flag'
trace_dir=root/'evidence/20260920_loop016_direct_scatter/service_ab'
dataset='/data/wio/vllm_ascend_26/datasets/GSM8K-in32768-num48-DeepSeek-V4-Flash-0731-w4a8.jsonl'
base='http://127.0.0.1:8080'
def stamp(): return datetime.now(timezone.utc).isoformat()
def run(command,log):
    with (out/log).open('w') as f:
        subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
def bench(offset,limit,name):
    path=out/(name+'.json')
    run(['docker','exec','dsv4ab','python3',str(root/'scripts/bench.py'),
         '--dataset',dataset,'--out',str(path),'--offset',str(offset),
         '--limit',str(limit),'--concurrency','1','--max-tokens','128'],name+'.log')
    d=json.loads(path.read_text())
    assert d['summary']['success']==limit
    return d
def metrics(): return requests.get(base+'/metrics',timeout=30).text
for _ in range(240):
    try:
        if requests.get(base+'/health',timeout=2).status_code==200: break
    except requests.RequestException: pass
    time.sleep(10)
else: raise RuntimeError('service readiness timeout')
marks={'ready_utc':stamp()}
flag.unlink(missing_ok=True)
run(['docker','exec','dsv4ab','python3',str(root/'scripts/check_functional.py'),
     '--golden',str(root/'evidence/20260920_qli_prefill_only/perf/golden4.json'),
     '--out',str(out/'functional_check.json')],'functional.log')
marks['functional_done_utc']=stamp()
(out/'metrics_before.txt').write_text(metrics())
flag.write_text('candidate enabled\n')
try:
    bench(0,1,'candidate_probe0')
    trace_files=list(trace_dir.glob('rank_*.txt'))
    meta_files=list(trace_dir.glob('meta_*.txt'))
    marks['trace_files']=len(trace_files)
    marks['meta_files']=len(meta_files)
    (out/'phase_live.json').write_text(json.dumps(marks,indent=2)+'\n')
    if len(trace_files)!=8:
        raise RuntimeError(f'fast path missed ranks: {len(trace_files)}/8; metadata files={len(meta_files)}')
    candidate=bench(8,8,'candidate8_offset8')
finally:
    flag.unlink(missing_ok=True)
marks['candidate_done_utc']=stamp()
(out/'metrics_after.txt').write_text(metrics())
baseline=json.loads((trace_dir/'baseline8_offset8.json').read_text())
pairs=[]
for i,(a,b) in enumerate(zip(baseline['requests'],candidate['requests'])):
    assert a['input_tokens']==b['input_tokens']==32851
    assert a['output_tokens']==b['output_tokens']==128
    pairs.append({'offset':8+i,'baseline_ttft_ms':a['ttft_ms'],
                  'candidate_ttft_ms':b['ttft_ms'],
                  'delta_ms':b['ttft_ms']-a['ttft_ms']})
summary={'n':len(pairs),
         'baseline_mean_ttft_ms':statistics.mean(x['baseline_ttft_ms'] for x in pairs),
         'candidate_mean_ttft_ms':statistics.mean(x['candidate_ttft_ms'] for x in pairs),
         'mean_delta_ms':statistics.mean(x['delta_ms'] for x in pairs),
         'relative_change_pct':100*(statistics.mean(x['candidate_ttft_ms'] for x in pairs)/statistics.mean(x['baseline_ttft_ms'] for x in pairs)-1),
         'pairs_improved':sum(x['delta_ms']<0 for x in pairs)}
(out/'summary.json').write_text(json.dumps({'summary':summary,'pairs':pairs,'marks':marks},indent=2)+'\n')
print(json.dumps(summary,indent=2))
