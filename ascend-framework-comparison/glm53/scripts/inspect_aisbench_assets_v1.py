#!/usr/bin/env python3
"""Read explicit historical client-only assets; no old client execution or mutation."""
import hashlib
import importlib.util
import json
import shlex
from pathlib import Path

P=Path(__file__).resolve();prior=P.parent/'inventory_readonly.py'
assert hashlib.sha256(prior.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
s=importlib.util.spec_from_file_location('base',prior);b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
RUN=b.ROOT/'runs/20261017T0610-aisbench-assets-v1'
HOST=r'''
import pathlib,json,hashlib,socket,subprocess
assert socket.gethostname()==EXPECTED
asset=pathlib.Path('/data/wio/aisbench-offline-43-20261014');assert asset.is_dir()
tag='sglang-tianjin:arm64-cann9.0.0-a3-v0.5.16'
image=subprocess.run(['docker','image','inspect',tag],capture_output=True,text=True,timeout=15)
assert image.returncode==0
image=json.loads(image.stdout)[0]
out={'client_image':{k:image[k] for k in ('Id','Architecture','Os')},'tag_reference_only':tag,'files':[],'metadata':[],'source_scope':'AISBench client only, not SGLang inference image'}
targets=['venv/pyvenv.cfg','benchmark/.git/HEAD','benchmark/.git/refs/heads/master','benchmark/.git/packed-refs','CPU-PATCH-SHA256SUMS']
for name in targets:
 path=asset/name
 if path.is_file():out['metadata'].append({'path':name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'text':path.read_text()[:60000]})
targets=['models/api_models/base_api.py','models/api_models/vllm_custom_api.py','models/output.py',
         'registry.py','cli/main.py','datasets/custom.py','configs/models/vllm_api/vllm_api_general_stream.py',
         'configs/summarizers/perf/stable_stage.py','openicl/icl_inferencer/icl_perf_inferencer.py']
core=asset/'benchmark/ais_bench/benchmark'
targets+=[str(p.relative_to(core)) for p in (core/'calculators').rglob('*.py') if 'perf' in p.name or 'stable' in p.name]
budget=700000
for name in targets:
 path=core/name
 if not path.is_file():continue
 data=path.read_bytes()
 if len(data)>budget:continue
 budget-=len(data);out['files'].append({'path':name,'sha256':hashlib.sha256(data).hexdigest(),'text':data.decode()})
print(json.dumps(out))
'''


def main():
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 pod,hostname=b.NODES[43]
 mapping=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30)
 assert mapping['rc']==0 and mapping['stdout']=='172.16.12.43'
 code=HOST.replace('EXPECTED',repr(hostname))
 command=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])
 result=b.invoke([command],code,timeout=90);(RUN/'source-43.json').write_text(json.dumps(result,indent=2)+'\n')
 (RUN/'receipt.json').write_text(json.dumps({'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'rc':result['rc'],'model_requests':0,'old_asset_mutations':0},indent=2)+'\n')
 print('asset source',result['rc'],flush=True)


if __name__=='__main__':main()
