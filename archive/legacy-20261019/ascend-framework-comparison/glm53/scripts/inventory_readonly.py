#!/usr/bin/env python3
"""One-shot bounded read-only inventory, executed only by local ZCode."""
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs' / '20261017T0210-inventory'
SSH = ['ssh', '-S', '/Users/wio/work/sglang-glm53-pd/ssh-control', '-p', '60011',
       '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'root@61.241.77.34']
NODES = {
    30: ('sglang-node-access-30', 'B4-15U-Atlas900RCKA3-30'),
    31: ('sglang-node-access-31', 'B4-23U-Atlas900RCKA3-31'),
    34: ('sglang-tuning-access-34-20261013', 'B5-15U-Atlas900RCKA3-34'),
    35: ('sglang-tuning-access-35-20261013', 'B5-23U-Atlas900RCKA3-35'),
    36: ('sglang-tuning-access-36-20261013', 'B5-33U-Atlas900RCKA3-36'),
    41: ('sglang-tuning-access-41-20261014', 'B7-5U-Atlas900RCKA3-41'),
    43: ('sglang-container-access-43-20261013', 'B7-23U-Atlas900RCKA3-43'),
    45: ('sglang-container-access-45-20261013', 'B8-5U-Atlas900RCKA3-45'),
}
HOST = r'''
import os, sys, json, hashlib, subprocess, pathlib, socket, datetime
expected, image = sys.argv[1:3]
if socket.gethostname() != expected:
    print(json.dumps({'identity_error':socket.gethostname(),'expected':expected})); sys.exit(21)
def cmd(args, limit=40):
    try:
        r=subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=limit)
        return {'rc':r.returncode,'stdout':r.stdout[:160000],'stderr':r.stderr[:8000]}
    except Exception as e: return {'error':str(e)}
r={'hostname':socket.gethostname(),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
r['load']=cmd(['cat','/proc/loadavg'])
r['memory']=cmd(['free','-b'])
r['numa']=cmd(['lscpu','-e=CPU,NODE,SOCKET'])
r['devices']=sorted(str(p) for p in pathlib.Path('/dev').glob('davinci*'))
r['npu']=cmd(['npu-smi','info'],60)
r['topology']=cmd(['npu-smi','info','-t','topo'],30)
r['processes']=cmd(['ps','-eo','pid,ppid,user,comm,etimes'])
r['listeners']=cmd(['ss','-lntp'])
r['containers']=cmd(['docker','ps','-a','--no-trunc','--format','{{.ID}} {{.Names}} {{.Image}} {{.Status}}'])
r['image']=cmd(['docker','image','inspect',image,'--format','{{.Id}} {{.Os}}/{{.Architecture}} {{json .RepoTags}} {{json .RootFS.Layers}}'])
containers=cmd(['docker','ps','-aq'])
r['container_metadata']=[]
for cid in containers.get('stdout','').splitlines()[:40]:
    r['container_metadata'].append(cmd(['docker','inspect',cid,'--format','{{.Id}} {{.Name}} {{.Image}} {{json .State}} {{json .Mounts}} {{json .Config.Entrypoint}} {{json .Config.Cmd}}']))
ns=pathlib.Path('/data/wio/inference-foundry/glm53')
r['namespace_exists']=ns.exists()
r['namespace_files']=[]
if ns.exists():
    for p in list(ns.rglob('*'))[:100]:
        if p.is_file(): r['namespace_files'].append({'path':str(p),'size':p.stat().st_size})
model=pathlib.Path('/data/wio/models/GLM-5.3-w8a8')
r['model']={'path':str(model),'resolved':str(model.resolve()),'exists':model.exists()}
if model.exists():
    files=sorted(p for p in model.rglob('*') if p.is_file())
    r['model']['files']=[{'name':str(p.relative_to(model)),'bytes':p.stat().st_size} for p in files]
    r['model']['total_bytes']=sum(p.stat().st_size for p in files)
    r['model']['metadata']=[]
    for p in files:
        if p.suffix in ('.json','.jinja','.txt','.md') and p.stat().st_size < 30000000:
            data=p.read_bytes(); entry={'name':str(p.relative_to(model)),'sha256':hashlib.sha256(data).hexdigest()}
            if p.name=='config.json': entry['config']=json.loads(data)
            elif 'quant' in p.name.lower():
                try:
                    q=json.loads(data); entry['top_keys']=list(q)[:80]
                    entry['scalar_fields']={k:v for k,v in q.items() if isinstance(v,(str,int,float,bool)) or v is None}
                except Exception as e: entry['parse_error']=str(e)
            elif p.name.endswith('.index.json'):
                q=json.loads(data); entry['metadata']=q.get('metadata'); entry['tensor_count']=len(q.get('weight_map',{}))
            r['model']['metadata'].append(entry)
r['mount']=cmd(['findmnt','-T',str(model),'-o','TARGET,SOURCE,FSTYPE,OPTIONS'])
r['sdk_dirs']={base:sorted(p.name for p in pathlib.Path(base).glob('*'))[:100] for base in ['/usr/local/Ascend','/data/wio','/opt']}
r['libraries']=cmd(['sh','-c',"ldconfig -p 2>/dev/null | grep -Ei 'hixl|transfer|mooncake|hccl|ascend' || true"])
print(json.dumps(r,ensure_ascii=False))
'''

def invoke(args, data=None, timeout=180):
    p = subprocess.run(SSH + args, input=data, text=True, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=timeout)
    return {'rc':p.returncode,'stdout':p.stdout,'stderr':p.stderr}

def node(n, pod, hostname):
    import shlex
    image = ('sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
             if n in (30,31,43,45) else
             'sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263')
    mapping=invoke(['kubectl get pod -n default '+shlex.quote(pod)+
                    " -o jsonpath='{.status.hostIP} {.spec.nodeName} {.status.phase}'"],timeout=30)
    if mapping['rc'] or mapping['stdout'].split()[0] != '172.16.12.'+str(n):
        return {'node':n,'mapping':mapping,'error':'pod mapping mismatch; host probe skipped'}
    command=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--',
                       'nsenter','-t','1','-m','--','python3','-',hostname,image])
    return {'node':n,'mapping':mapping,'probe':invoke([command],HOST,timeout=240)}

def main():
    RUN.mkdir(parents=True,exist_ok=True)
    lock=RUN/'execution-once.lock'
    fd=os.open(str(lock),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    os.write(fd,str(os.getpid()).encode()); os.close(fd)
    start=datetime.datetime.now(datetime.timezone.utc).isoformat()
    source=invoke(["hostname; ip -4 -o addr show | awk '{print $4}'"],timeout=30)
    (RUN/'source.json').write_text(json.dumps(source,indent=2))
    if source['rc'] or 'A3-5U-Atlas900RCKA3-1' not in source['stdout'] or '172.16.12.1/' not in source['stdout']:
        raise RuntimeError('source identity mismatch; nodes not touched')
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(node,n,*v):n for n,v in NODES.items()}
        for f in concurrent.futures.as_completed(futures):
            n=futures[f]
            try: result=f.result()
            except Exception as e: result={'node':n,'error':repr(e)}
            (RUN/('node-%s.json'%n)).write_text(json.dumps(result,indent=2))
            results.append({'node':n,'error':result.get('error'),'rc':result.get('probe',{}).get('rc')})
            print(json.dumps(results[-1]),flush=True)
    receipt={'project_id':'glm53','start_utc':start,'end_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'launcher_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'nodes':results,
             'remote_mutations':False,'model_requests':0,'npu_compute_requests':0}
    (RUN/'receipt.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt),flush=True)

if __name__=='__main__': main()
