#!/usr/bin/env python3
"""Read-only local/SSH reproducibility check of two historical EP8 profiler rows."""
import base64
import shlex
import subprocess

REMOTE = r'''import csv, hashlib, json
from decimal import Decimal
from pathlib import Path
p = Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces/dp1_pp0_tp0_dcp0_ep8_rank0_737_20261029023525648_ascend_pt')
prof = p / 'PROF_000001_20261029023525659_00000737LBADKQBH'
files = {
    'op_summary': (prof/'mindstudio_profiler_output/op_summary_20261029024048.csv', '3088d48149dc0e97b5876f5822776ba9a69a968ddd326fb2d776c9a829bb2f1b'),
    'kernel_details': (p/'ASCEND_PROFILER_OUTPUT/kernel_details.csv', '20cfb97abe03befda3fbf2c1194e48531aac44ce17d4f5fc130ea8ffb5261730'),
    'device_info': (prof/'device_8/info.json.8', '32bb376ca72cbd6505af361d3e9f4d0e87a17a0985ee52f5e76c472a445fc829'),
}
for name,(path,expected) in files.items():
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected: raise ValueError('SHA mismatch: '+name)
info=json.loads(files['device_info'][0].read_text())
assert info['cannVersion']=='9.2.0'
freq=Decimal(info['DeviceInfo'][0]['aic_frequency']); cores=Decimal(info['DeviceInfo'][0]['ai_core_num'])
assert (freq,cores)==(Decimal(1800),Decimal(24))
rows={}
for name in ('op_summary','kernel_details'):
    with files[name][0].open(newline='') as f:
        data=list(csv.DictReader(f))
    for task in ('44217','45363'):
        hit=[r for r in data if r['Task ID']==task and r['Stream ID']=='47' and (r.get('Op Name') or r.get('Name'))=='DispatchFFNCombine']
        assert len(hit)==1,(name,task,len(hit))
        r=hit[0]; dur=r.get('Task Duration(us)') or r['Duration(us)']
        rows.setdefault(task,{})[name]={'duration_us':dur,'aic_total_cycles':r['aic_total_cycles'],'aic_mac_ratio':r['aic_mac_ratio'],'cube_utilization_pct':r['cube_utilization(%)']}
for task,vals in rows.items():
    assert vals['op_summary']==vals['kernel_details'], task
    r=vals['kernel_details']; computed=(Decimal(100)*Decimal(r['aic_total_cycles'])/(freq*cores*Decimal(r['duration_us']))).quantize(Decimal('.001'))
    assert computed==Decimal(r['cube_utilization_pct']),task
    r['recomputed_pct_from_rounded_duration']=str(computed)
print(json.dumps({'source_hashes_verified':{k:v[1] for k,v in files.items()},'cannVersion':info['cannVersion'],'aic_frequency_MHz':str(freq),'ai_core_num':str(cores),'tasks':rows,'limitations':['formula reproduces these displayed values; no proof historical binary execution','CSV duration already rounded','not MAC efficiency','no inference about capture completeness or physical peak']},ensure_ascii=False,indent=2))
'''

encoded = base64.b64encode(REMOTE.encode()).decode()
remote_cmd = shlex.join(['docker', 'exec', 'wio-vllm-v41-repro169-stage-20261028', 'python3', '-c', f'import base64;exec(base64.b64decode({encoded!r}))'])
p = subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh', remote_cmd], text=True, capture_output=True, timeout=60)
if p.returncode:
    raise SystemExit('read-only check failed: '+p.stderr[-1000:])
print(p.stdout, end='')
