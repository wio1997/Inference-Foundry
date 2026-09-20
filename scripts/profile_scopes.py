import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

p = argparse.ArgumentParser()
p.add_argument('--out', required=True)
p.add_argument('--root', default='/data/wio/Inference_Foundry')
a = p.parse_args()
root = Path(a.root)
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
base = 'http://127.0.0.1:8080'
warm = '/data/wio/vllm_ascend_26/datasets/GSM8K-in32768-num48-DeepSeek-V4-Flash-0731-w4a8-repeatRate0.9.jsonl'
cold = '/data/wio/vllm_ascend_26/datasets/GSM8K-in32768-num48-DeepSeek-V4-Flash-0731-w4a8.jsonl'

def now():
    return datetime.now(timezone.utc).isoformat()

def bench(name, dataset, offset, limit, concurrency, max_tokens):
    cmd = ['docker', 'exec', 'dsv4ab', 'python3', str(root / 'scripts/bench.py'),
           '--dataset', dataset, '--out', str(out / f'{name}.json'),
           '--offset', str(offset), '--limit', str(limit), '--concurrency', str(concurrency),
           '--max-tokens', str(max_tokens)]
    with (out / f'{name}.log').open('w') as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)

for _ in range(180):
    try:
        if requests.get(base + '/health', timeout=2).status_code == 200:
            break
    except requests.RequestException:
        pass
    time.sleep(10)
else:
    raise RuntimeError('service readiness timeout')
marks = {'ready_utc': now()}
bench('warm_preprofile', warm, 0, 4, 4, 128)
marks['warm_preprofile_done_utc'] = now()
resp = requests.post(base + '/start_profile', timeout=120)
resp.raise_for_status()
marks['profile_started_utc'] = now()
try:
    marks['warm_start_utc'] = now()
    bench('warm_profile', warm, 0, 4, 4, 128)
    marks['warm_end_utc'] = now()
    marks['cold_start_utc'] = now()
    bench('cold_profile', cold, 20, 2, 1, 128)
    marks['cold_end_utc'] = now()
finally:
    marks['profile_stop_request_utc'] = now()
    try:
        response = requests.post(base + '/stop_profile', timeout=1200)
        marks['profile_stop_http'] = response.status_code
        response.raise_for_status()
    finally:
        marks['profile_stopped_utc'] = now()
        (out / 'phase_times.json').write_text(json.dumps(marks, indent=2) + '\n')
print(json.dumps(marks, indent=2))
