"""Retire exact snapshotted own models only, after idle and identity checks."""
from pathlib import Path
import datetime
import json
import socket
import subprocess
import urllib.request

node = {'B4-33U-Atlas900RCKA3-32':32, 'B5-5U-Atlas900RCKA3-33':33}[socket.gethostname()]
cid = {32:'a04ab7b22797797ca23431b9b7b5f0ef536ae625aad28390cde4b0550f99f14e',
       33:'bfac6522d752e0595df01f537e2f74bbe4c4d00cf9334668283e6885b91e910d'}[node]
assert node == 33, 'Keep existing P/proxy loaded'
root = Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/early-publication-live2-20261019/node'+str(node))
plan = json.loads((root/'plan.json').read_text()); launch = json.loads((root/'launch.json').read_text())
d = json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
assert d['Id'] == launch['container_id'] == cid and d['Name'] == '/'+plan['container']['name']
assert d['Image'] == plan['image']['id'] == 'sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
with urllib.request.urlopen('http://172.16.12.%d:%d/metrics'%(node,29481 if node==32 else 29482),timeout=5) as f:data=f.read().decode()
idle=[z for z in data.splitlines() if z.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{','vllm:kv_cache_usage_perc{'))]
assert len(idle)>=3 and all(float(z.rsplit(' ',1)[1])==0 for z in idle)
if node==33:assert not(root/'early-sample-publication-probe/enable').exists()
out=root/'stop-for-rope-step-reuse-v1.json';assert not out.exists()
rows=[]
items=[('model',cid,d)]
if node==32:
    pl=json.loads((root/'official-proxy-v1/launch.json').read_text())
    assert pl['parent_P_CID']==cid and pl['CID']=='68bbb6dda77541c148c012346130d2ee8c86d937a2a13d90a83b29ddd5f857c8'
    pd=json.loads(subprocess.check_output(['docker','inspect',pl['CID']]))[0]
    assert pd['Image']==d['Image'] and not pd['HostConfig'].get('Devices')
    items.insert(0,('official-proxy',pl['CID'],pd))
for kind,ident,ins in items:
    stop=None
    if ins['State']['Running']:
        q=subprocess.run(['docker','stop','-t','30',ident],capture_output=True,text=True,timeout=45)
        stop={'rc':q.returncode,'stdout':q.stdout,'stderr':q.stderr};assert q.returncode==0
    after=json.loads(subprocess.check_output(['docker','inspect',ident]))[0]
    assert not after['State']['Running']
    rows.append(dict(kind=kind,CID=ident,before=ins['State'],stop=stop,after=after['State']))
q=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=40)
assert q.returncode==0 and q.stdout.count('No running processes found in NPU')==8
report=dict(node=node,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),idle_metrics=idle,
            stopped=rows,empty_physical_cards=8,reason='Exact own model/proxy source/log snapshot SHA verified on client; replace only D for same-step RoPE source reuse with original native publication/retirement. Existing finite holders retained. No other job or access/system pod touched.')
out.write_text(json.dumps(report,indent=2)+'\n')
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(report)+'\nGLM53_RESULT_JSON_END')
