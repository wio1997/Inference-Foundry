"""CPU diagnostic validation and small cache metadata inspection; no model calls."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

CACHE = r'''
import io,pickle,json,hashlib,subprocess
from pathlib import Path
class Restricted(pickle.Unpickler):
 def find_class(self,module,name):raise ValueError('globals disallowed')
 def persistent_load(self,pid):raise ValueError('persistent references disallowed')
root=Path('/data/tiankuan/wio/GLM-5.3-w8a8');cache=root/'.msc';out={}
try:
 b=cache.read_bytes();d=Restricted(io.BytesIO(b)).load()
 def records(x):
  if type(x)==dict:
   yield x
   for v in x.values():yield from records(v)
  elif type(x)in (list,tuple):
   for v in x:yield from records(v)
 refs=set(json.loads((root/'quant_model_weights.safetensors.index.json').read_text())['weight_map'].values())
 rows=[r for r in records(d)if type(r.get('Path'))==str and r['Path'] in refs]
 out=dict(cache_type=type(d).__name__,cache_metadata_sha256=hashlib.sha256(b).hexdigest(),referenced_cache_entries=len(rows),referenced_files=len(refs),cache_keys=sorted(set(k for r in rows for k in r)),payload_hashes_computed=0)
except Exception as e:out=dict(cache_parse_error=type(e).__name__,payload_hashes_computed=0)
code="import importlib.metadata as m,json,hashlib;from pathlib import Path;p=Path('/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_mtp.py');print(json.dumps(dict(versions={x:m.version(x)for x in ('vllm','vllm-ascend','transformers','torch','torch-npu')},ascend_model_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),mooncake_sha256=hashlib.sha256(Path('/vllm-workspace/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py').read_bytes()).hexdigest())))"
p=subprocess.run(['docker','exec','glm52-single','/usr/local/python3.12.13/bin/python3','-c',code],text=True,capture_output=True,timeout=20)
out['source_probe_exit']=p.returncode;out['source_probe']=json.loads(p.stdout)if p.returncode==0 else p.stderr
print(json.dumps(out))
'''


def main(path):
    job=json.loads(Path(path).read_text());dest=Path(job['result']['path']).parent
    research=Path('/data/tiankuan/wio/Inference-Foundry/glm5-3/records/points/GLM-OPT-0002/research/pd_critical_path_20261006')
    evidence=[];files={}
    for name in ('pd_completion_trace.py','reduce_pd_completion_trace.py','test_pd_completion_trace.py','native_aggregator_class.json','native_tracker_class.json'):
        b=(research/name).read_bytes();files[name]=hashlib.sha256(b).hexdigest()
    command=['docker','exec','-w',str(research),'glm52-single','/usr/local/python3.12.13/bin/python3','-m','unittest','-v','test_pd_completion_trace.py']
    p=subprocess.run(command,text=True,capture_output=True,timeout=40)
    raw=(p.stdout+p.stderr).encode();out=dest/'cpu_tests.raw';out.write_bytes(raw)
    evidence.append(dict(id='cpu',path=str(out),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),locator='real8 unittest cases;exact native classes in CPU containers'))
    metadata={}
    for host in ('166','167'):
        cmd=['python3','-'] if host=='166' else ['ssh','-o','BatchMode=yes','root@172.16.10.167','python3','-']
        q=subprocess.run(cmd,input=CACHE,text=True,capture_output=True,timeout=35)
        raw=(q.stdout+'\nSTDERR\n'+q.stderr).encode();out=dest/('cache_source_'+host+'.raw');out.write_bytes(raw)
        evidence.append(dict(id='source'+host,path=str(out),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),locator='restricted primitive metadata and source/package identity;no imports of torch/vllm'))
        metadata[host]=json.loads(q.stdout)if q.returncode==0 else dict(error='probe exit'+str(q.returncode))
    summary=dict(cpu_exit=p.returncode,source_files=files,metadata=metadata,model_imports=0,model_tests=0,service_mutations=0,hook_installations_in_live_processes=0)
    out=dest/'reduction.json';out.write_text(json.dumps(summary,indent=2)+'\n');raw=out.read_bytes();evidence.append(dict(id='summary',path=str(out),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),locator='cpu_exit/source_files/metadata'))
    result=dict(schema_version=1,job_id=job['job_id'],status='completed'if p.returncode==0 else 'failed',summary='CPU observer validation on actual166 service Python; exact native tracker/aggregator class fixtures. Both hosts small cache/source identity checked;no live hooks or NPU/model operations.',execution=dict(inner_exit_code=p.returncode,acceptance='passed'if p.returncode==0 else 'failed',processes=[]),findings=[dict(kind='fact',text=json.dumps(summary),scope=dict(cpu_only=True,device_perf=False),evidence_ids=['summary','cpu'])],evidence=evidence,unknowns=['No hardware PD timing,device readiness,legal missed admission or E2E Gain established'],decision_request=None,next_check_at=None)
    out=dest/'result.tmp';out.write_text(json.dumps(result,indent=2)+'\n');os.replace(out,dest/'result.json')
    return p.returncode


if __name__=='__main__':
    sys.exit(main(sys.argv[1]))
