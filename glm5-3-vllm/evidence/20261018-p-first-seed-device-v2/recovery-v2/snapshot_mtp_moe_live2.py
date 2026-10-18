from pathlib import Path
import gzip
import hashlib
import json
import shutil
import socket
import subprocess
import urllib.request

host=socket.gethostname();node={'B4-33U-Atlas900RCKA3-32':32,'B5-5U-Atlas900RCKA3-33':33}[host]
root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/mtp-moe-graph-live2-20261018/node'+str(node))
cid={32:'12234db89fa8430727bba41ac176e8e21a81ecdd82a80eb07b26fc5c5b95c975',33:'084035e84399bd88c3788c80f184590f309fd6730f93cc308f460487c75af213'}[node]
plan=json.loads((root/'plan.json').read_text());launch=json.loads((root/'launch.json').read_text());d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
assert d['Id']==launch['container_id']==cid and d['Name']=='/'+plan['container']['name'] and d['Image']==plan['image']['id']
assert not (root/'mtp-moe-graph-probe/enable').exists()
if node==33:
    p=json.loads((root/'mtp-moe-graph-c4-profile-v1/result.json').read_text());assert p['completed'] and p['profile_stopped'] and p['ranks']==list(range(16))
out=Path(__file__).resolve().parent/'artifacts';out.mkdir(exist_ok=False)
selected=[]
for p in root.rglob('*'):
    if not p.is_file() or p.is_symlink():continue
    rel=p.relative_to(root)
    if rel.parts[0]=='profile' or '-raw-' in rel.parts[0] or p.name=='trace_view.json':continue
    if p.suffix in ['.json','.jsonl','.py','.bin']:selected.append(p)
assert sum(p.stat().st_size for p in selected)<256*1024**2
for p in selected:
    q=out/'model-root'/p.relative_to(root);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
for patch in plan['runtime_patches']:
    p=Path(patch['source']);b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==patch['sha256'];q=out/'overlay-sources'/p.name;q.parent.mkdir(exist_ok=True);q.write_bytes(b)
items=[('model',cid,d)]
if node==32:
    proxy='e4d73792d53736a7608c4256a5d9168a2e1d4639d50b27a7a5702a83fb3c4a17';pd=json.loads(subprocess.check_output(['docker','inspect',proxy]))[0];pl=json.loads((root/'official-proxy-v1/launch.json').read_text());assert pl['CID']==proxy and not pd['HostConfig'].get('Devices') and pd['Image']==d['Image'];items.append(('official-proxy',proxy,pd))
native_metrics={}
for n,port in [(32,29481),(33,29482)]:
 with urllib.request.urlopen('http://172.16.12.%d:%d/metrics'%(n,port),timeout=5) as f:data=f.read().decode()
 native_metrics[str(n)]=[z for z in data.splitlines() if not z.startswith('#') and any(k in z for k in ['spec_decode_num_','num_preemptions_total','num_requests_running{','num_requests_waiting{','kv_cache_usage_perc{','generation_tokens_total','prompt_tokens_total'])]
rows=[]
for name,ident,ins in items:
    with gzip.open(out/(name+'-console.txt.gz'),'wb') as f:
        q=subprocess.run(['docker','logs','--timestamps',ident],stdout=f,stderr=subprocess.STDOUT,timeout=30);assert q.returncode==0
    rows.append({'kind':name,'CID':ident,'state':ins['State']})
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps({'node':node,'snapshots':rows,'native_metrics':native_metrics,'selected_files':len(selected),'selected_bytes':sum(p.stat().st_size for p in selected),'scope':'Exactownedmodels/proxy snapshot, accept naturallyexpiredstates, no stop or restart. All16 GPUprofile stopped first; raw/export retained separately, no otherjobs touched.'})+'\nGLM53_RESULT_JSON_END')
