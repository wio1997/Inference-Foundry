"""Bounded metadata/header and live-site inspection; never reads tensor payload."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

PROBE = r'''
import hashlib,json,os,subprocess,time,math,urllib.request
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone

def cmd(argv):
    try:
        p=subprocess.run(argv,capture_output=True,text=True,timeout=20)
        return dict(exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr)
    except Exception as e:return dict(error=type(e).__name__)

root=Path('/data/tiankuan/wio/GLM-5.3-w8a8')
model=dict(path=str(root),payload_bytes_read=0,weight_payload_hashes=0)
before={p.name:(p.stat().st_size,p.stat().st_mtime_ns) for p in root.glob('*.safetensors')}
for name in ('config.json','tokenizer_config.json','generation_config.json','GLM-5.3_best_practice.yaml'):
    p=root/name
    model[name]=dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),text=p.read_text()) if p.exists() else None
index=root/'quant_model_weights.safetensors.index.json'
raw=index.read_bytes(); data=json.loads(raw); weight_map=data['weight_map']
model['index']=dict(path=str(index),sha256=hashlib.sha256(raw).hexdigest(),metadata=data.get('metadata'),tensor_count=len(weight_map))
shards=sorted(set(weight_map.values())); mapped={s:set() for s in shards}
for name,shard in weight_map.items():mapped[shard].add(name)
dtype_bits={'BOOL':8,'U8':8,'I8':8,'I16':16,'U16':16,'I32':32,'U32':32,'I64':64,'U64':64,'F16':16,'BF16':16,'F32':32,'F64':64,'F8_E4M3':8,'F8_E5M2':8,'F8_E4M3FN':8}
summaries=[]; errors=[]; total_payload=0; rotation_payload=0; total_headers=0; tensors=0; dtype_counts=Counter()
for shard in shards:
    p=root/shard
    try:
        if p.parent != root:raise ValueError('non-flat index path')
        stat0=p.stat()
        # Read exactly the 8-byte header length and JSON header; never seek to
        # or read tensor offsets. No safetensors loader/import is used.
        with p.open('rb') as f:
            length_bytes=f.read(8)
            if len(length_bytes)!=8:raise ValueError('short prefix')
            n=int.from_bytes(length_bytes,'little')
            if n>64*1024*1024 or n+8>stat0.st_size:raise ValueError('invalid header size')
            header=f.read(n)
            if len(header)!=n:raise ValueError('short header')
        parsed=json.loads(header); entries={k:v for k,v in parsed.items() if k!='__metadata__'}
        if set(entries)!=mapped[shard]:raise ValueError('index/header tensor membership mismatch')
        spans=[]
        for name,value in entries.items():
            lo,hi=value['data_offsets']; shape=value['shape']; dtype=value['dtype']
            if type(lo)!=int or type(hi)!=int or not 0<=lo<=hi:raise ValueError('invalid offsets')
            if not all(type(x)==int and x>=0 for x in shape):raise ValueError('invalid shape')
            if dtype not in dtype_bits:raise ValueError('unverified dtype '+str(dtype))
            if (hi-lo)*8 != math.prod(shape)*dtype_bits[dtype]:raise ValueError('shape/dtype/offset size mismatch')
            spans.append((lo,hi));dtype_counts[dtype]+=1
        spans.sort(); cursor=0
        for lo,hi in spans:
            if lo!=cursor:raise ValueError('payload overlap/gap')
            cursor=hi
        if 8+n+cursor!=stat0.st_size:raise ValueError('truncated/trailing file')
        stat1=p.stat()
        if (stat0.st_size,stat0.st_mtime_ns)!=(stat1.st_size,stat1.st_mtime_ns):raise ValueError('file changed during header inspection')
        total_payload+=cursor;total_headers+=8+n;tensors+=len(entries)
        if set(entries)=={"rot.weight"}:rotation_payload+=cursor
        summaries.append(dict(file=shard,bytes=stat1.st_size,mtime_ns=stat1.st_mtime_ns,header_sha256=hashlib.sha256(header).hexdigest(),header_bytes=8+n,tensors=len(entries)))
    except Exception as e:errors.append(dict(file=shard,error=str(e)))
after={p.name:(p.stat().st_size,p.stat().st_mtime_ns) for p in root.glob('*.safetensors')}
if before!=after:errors.append(dict(error='directory shard metadata changed during inspection'))
if total_payload-data.get('metadata',{}).get('total_size',-1) not in (0,rotation_payload):errors.append(dict(error='index metadata.total_size differs from tensor payload offsets'))
model.update(structural_errors=errors,structural_complete=not errors,shards=summaries,unique_shards=len(shards),tensors=tensors,dtype_counts=dict(dtype_counts),total_payload_bytes=total_payload,header_bytes_read=total_headers,changed=before!=after,cryptographic_payload_integrity='unknown')

procs=[]
ps=cmd(['ps','-eo','pid,stat,comm'])
for line in ps.get('stdout','').splitlines()[1:]:
    parts=line.split(None,2)
    if len(parts)!=3 or 'Z' in parts[1]:continue
    pid,stat,comm=parts
    if not any(w in comm for w in ('python','VLLM','zcode','asc_dumper')):continue
    p=Path('/proc')/pid
    try:
        args=p.joinpath('cmdline').read_bytes().decode(errors='replace').split('\0')[:-1]
        if not any(w in ' '.join(args) for w in ('glm52-pd','Inference-Foundry/glm5-3','vllm.entrypoints','VLLM::')):continue
        # Never return credential arguments or entire environ.
        if any(w in a.lower() for a in args for w in ('password','api-key','authorization','bearer','token=')):
            args=['REDACTED credential-bearing argv']
        env=p.joinpath('environ').read_bytes().split(b'\0'); py=[x.decode(errors='replace') for x in env if x.startswith(b'PYTHONPATH=')]
        st=p.joinpath('stat').read_text().rsplit(')',1)[1].split()
        procs.append(dict(pid=int(pid),state=stat,comm=comm,start_ticks=st[19],argv=args,PYTHONPATH=py))
    except (OSError,ValueError):pass

listeners=cmd(['ss','-ltnH']);health={};metrics={}
for port in (8000,9081,9900):
    if not any(line.split()[3].endswith(':'+str(port)) for line in listeners.get('stdout','').splitlines() if len(line.split())>3):continue
    for endpoint in ('health','metrics'):
        try:
            with urllib.request.urlopen('http://127.0.0.1:'+str(port)+'/'+endpoint,timeout=3) as response:
                text=response.read(2*1024*1024).decode()
                if endpoint=='health':health[str(port)]=response.status
                else:metrics[str(port)]=[x for x in text.splitlines() if 'requests_running' in x or 'requests_waiting' in x]
        except Exception as e:
            if endpoint=='health':health[str(port)]=type(e).__name__
owner=Path('/data/tiankuan/wio/glm52-pd/controller-owner.json')
site=dict(at=datetime.now(timezone.utc).isoformat(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),processes=procs,health=health,metrics=metrics,listeners=listeners,controller_owner=json.loads(owner.read_text()) if owner.exists() else None,docker=cmd(['docker','inspect','glm52-single','--format','{{json .State}} {{.Config.Image}} {{json .Mounts}}']),npu=cmd(['/usr/local/bin/npu-smi','info']))
repo=Path('/data/tiankuan/wio/Inference-Foundry')
site['git']=cmd(['git','-C',str(repo),'status','--short','--branch'])
site['git_head']=cmd(['git','-C',str(repo),'rev-parse','HEAD'])
site['rules']={name:hashlib.sha256((repo/name).read_bytes()).hexdigest() if (repo/name).exists() else None for name in ('AGENTS.md','glm5-3/AGENTS.md','glm5-3/MISSION.md','glm5-3/PLAN.md')}
site['zcode']=cmd(['/usr/local/bin/zcode','--version'])
site['origins']=cmd(['docker','exec','glm52-single','/usr/local/python3.12.13/bin/python3','-c',"import importlib.util,json;print(json.dumps({x:importlib.util.find_spec(x).origin for x in ('vllm','vllm_ascend')}))"])
print(json.dumps(dict(model=model,site=site)))
'''


def main(job_path):
    job=json.loads(Path(job_path).read_text()); directory=Path(job['result']['path']).parent
    observations={}; evidence=[]; failures=[]
    for host in ('166','167'):
        command=['python3','-'] if host=='166' else ['ssh','-o','BatchMode=yes','root@172.16.10.167','python3','-']
        try:
            p=subprocess.run(command,input=PROBE,text=True,capture_output=True,timeout=140)
            raw=(p.stdout+'\nSTDERR\n'+p.stderr).encode(); target=directory/('observations_'+host+'.raw')
            target.write_bytes(raw)
            evidence.append(dict(id='site'+host,path=str(target),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),locator='JSON model/site plus real stderr'))
            if p.returncode:raise RuntimeError('probe exit '+str(p.returncode))
            observations[host]=json.loads(p.stdout)
        except Exception as error:failures.append(dict(host=host,error=str(error)))
    compact={}
    for host,obs in observations.items():
        m,s=obs['model'],obs['site']
        compact[host]=dict(structural_complete=m['structural_complete'],structural_errors=m['structural_errors'],unique_shards=m['unique_shards'],tensors=m['tensors'],total_payload_bytes=m['total_payload_bytes'],header_bytes_read=m['header_bytes_read'],weight_payload_bytes_read=0,index_sha256=m['index']['sha256'],config_sha256=m['config.json']['sha256'],health=s['health'],metrics=s['metrics'],process_count=len(s['processes']),head=s['git_head'],boot_id=s['boot_id'],cryptographic_payload_integrity='unknown')
    reduced=directory/'reduction.json';reduced.write_text(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(),observations=compact,failures=failures),indent=2)+'\n')
    b=reduced.read_bytes();evidence.append(dict(id='reduction',path=str(reduced),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b),locator='observations/structural_complete and site health'))
    result=dict(schema_version=1,job_id=job['job_id'],status='failed' if failures else 'completed',summary='Bounded structural/header and site inspection; zero tensor payload reads and zero model/service operations.',execution=dict(inner_exit_code=1 if failures else 0,acceptance='failed' if failures else 'passed',processes=[]),findings=[dict(kind='fact',text=json.dumps(compact),scope=dict(hosts=list(compact),device_or_model_tests=0),evidence_ids=['reduction'])],evidence=evidence,unknowns=['User has confirmed both uploads completed; cryptographic payload integrity and5.3 loading/fit/correctness/PD/E2E capacity remain unverified'],decision_request=None,next_check_at=None)
    tmp=directory/'result.tmp';tmp.write_text(json.dumps(result,indent=2)+'\n');os.replace(tmp,directory/'result.json')
    return 1 if failures else 0


if __name__=='__main__':
    sys.exit(main(sys.argv[1]))
