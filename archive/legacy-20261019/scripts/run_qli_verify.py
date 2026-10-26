import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
import requests

p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--flag',required=True);a=p.parse_args()
out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
flag=Path(a.flag).resolve();flag.parent.mkdir(parents=True,exist_ok=True)
for _ in range(180):
    try:
        if requests.get('http://127.0.0.1:8080/health',timeout=2).status_code==200:break
    except requests.RequestException:pass
    time.sleep(10)
else:raise RuntimeError('service readiness timeout')
marks={'ready_utc':datetime.now(timezone.utc).isoformat()}
flag.write_text('verify CPU/NPU QLI maxima\n')
try:
    for name,dataset,offset,limit,concurrency in (
        ('warm_first','/data/wio/vllm_ascend_26/datasets/GSM8K-in32768-num48-DeepSeek-V4-Flash-0731-w4a8-repeatRate0.9.jsonl',0,4,4),
        ('warm_repeat','/data/wio/vllm_ascend_26/datasets/GSM8K-in32768-num48-DeepSeek-V4-Flash-0731-w4a8-repeatRate0.9.jsonl',0,4,4),
        ('cold2','/data/wio/vllm_ascend_26/datasets/GSM8K-in32768-num48-DeepSeek-V4-Flash-0731-w4a8.jsonl',22,2,1),
    ):
        cmd=['docker','exec','dsv4ab','python3','/data/wio/Inference_Foundry/scripts/bench.py',
             '--dataset',dataset,'--out',str(out/f'{name}.json'),'--offset',str(offset),
             '--limit',str(limit),'--concurrency',str(concurrency),'--max-tokens','128']
        with (out/f'{name}.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
        result=json.loads((out/f'{name}.json').read_text())['summary']
        assert result['success']==limit and result['fail']==0,result
        marks[name+'_done_utc']=datetime.now(timezone.utc).isoformat()
finally:
    flag.unlink(missing_ok=True)
    marks['verify_flag_removed_utc']=datetime.now(timezone.utc).isoformat()
    (out/'phase_times.json').write_text(json.dumps(marks,indent=2)+'\n')
print(json.dumps(marks,indent=2))
