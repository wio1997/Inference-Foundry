from pathlib import Path
import gzip
import hashlib
import json
import shutil
import socket
import subprocess
import urllib.request

host=socket.gethostname();node={'B4-33U-Atlas900RCKA3-32':32,'B5-5U-Atlas900RCKA3-33':33}[host]
root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/mtp-moe-graph-live1-20261018/node'+str(node))
cid={32:'3a6df4aa1d780554e85afa148d925f0ad90b43f939b45bdab5da67a93a4855b7',33:'9774dd714ad98295848b710f3a08d25cf91203eba5a801e554d10060f67b594d'}[node]
plan=json.loads((root/'plan.json').read_text());launch=json.loads((root/'launch.json').read_text());d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
assert d['Id']==launch['container_id']==cid and d['Name']=='/'+plan['container']['name'] and d['Image']==plan['image']['id']
assert not (root/'mtp-moe-graph-probe/enable').exists()
if node==33:
    p=json.loads((root/'mtp-moe-graph-profile-v1/result.json').read_text());assert p['completed'] and p['profile_stopped'] and p['ranks']==list(range(16))
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
    proxy='78a0d2e96c462407f55efe6c3cd1d37a39e90c5474b88104ed3a2de6d259bf2c';pd=json.loads(subprocess.check_output(['docker','inspect',proxy]))[0];pl=json.loads((root/'official-proxy-v1/launch.json').read_text());assert pl['CID']==proxy and not pd['HostConfig'].get('Devices') and pd['Image']==d['Image'];items.append(('official-proxy',proxy,pd))
rows=[]
for name,ident,ins in items:
    with gzip.open(out/(name+'-console.txt.gz'),'wb') as f:
        q=subprocess.run(['docker','logs','--timestamps',ident],stdout=f,stderr=subprocess.STDOUT,timeout=30);assert q.returncode==0
    rows.append({'kind':name,'CID':ident,'state':ins['State']})
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps({'node':node,'snapshots':rows,'selected_files':len(selected),'selected_bytes':sum(p.stat().st_size for p in selected),'scope':'Exactownedmodels/proxy snapshot, accept naturallyexpiredstates, no stop or restart. All16 GPUprofile stopped first; raw/export retained separately, no otherjobs touched.'})+'\nGLM53_RESULT_JSON_END')
