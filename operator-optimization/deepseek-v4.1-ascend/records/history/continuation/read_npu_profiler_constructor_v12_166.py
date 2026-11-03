import json,subprocess,socket,hashlib
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
code="from pathlib import Path;import json,hashlib;p=Path('/vllm-workspace/vllm-ascend/vllm_ascend/profiler/torch_npu_profiler.py');b=p.read_bytes();print(json.dumps(dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),lines=[str(i+1)+': '+s for i,s in enumerate(b.decode().splitlines()) if i<130])))"
r=subprocess.run(['docker','exec','wio-vllm-v41-repro166-stage-20261030','python3','-c',code],capture_output=True,text=True,timeout=30)
assert r.returncode==0,r.stderr
x=json.loads(r.stdout);assert x['sha256']=='5242ee153033afcf12dbfe022dbd807fa248d991ff5c53efbd1adb372d5575f4'
print(json.dumps(x))
