import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
import requests

p=argparse.ArgumentParser()
p.add_argument('--out', required=True)
a=p.parse_args()
out=Path(a.out).resolve(); out.mkdir(parents=True,exist_ok=True)
base='http://127.0.0.1:8080'
for _ in range(180):
    try:
        if requests.get(base+'/health',timeout=2).status_code==200: break
    except requests.RequestException: pass
    time.sleep(10)
else: raise RuntimeError('service readiness timeout')
marks={'ready_utc':datetime.now(timezone.utc).isoformat()}
for name in ('warmup','measured'):
    cmd=['docker','exec','dsv4ab','python3','/data/wio/Inference_Foundry/scripts/bench.py',
         '--dataset','/data/wio/vllm_ascend_26/datasets/GSM8K-in32768-num48-DeepSeek-V4-Flash-0731-w4a8-repeatRate0.9.jsonl',
         '--out',str(out/f'{name}.json'),'--offset','0','--limit','4','--concurrency','4','--max-tokens','128']
    with (out/f'{name}.log').open('w') as log:
        subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    marks[name+'_done_utc']=datetime.now(timezone.utc).isoformat()
(out/'phase_times.json').write_text(json.dumps(marks,indent=2)+'\n')
print(json.dumps(marks,indent=2))
