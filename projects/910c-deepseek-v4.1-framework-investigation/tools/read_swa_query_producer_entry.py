#!/usr/bin/env python3
"""Pin actual CV projection/quantization producers for next larger delivery review."""
from hashlib import sha256
from pathlib import Path
import json,subprocess
from read_prefix_retirement_identity import OUT,FROZEN

REMOTE=r'''
from hashlib import sha256
from pathlib import PurePosixPath
import io,json,subprocess,sys,tarfile
name='wio-vllm-v41-repro171-20261102';c,=json.loads(subprocess.check_output(['docker','inspect',name]))
assert c['Id']=='3be44e7b5c292de6ef7fc51085bfbbad5552acc97fae39c0ae176fe8cfb387eb' and c['Image']=='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
out=[]
for r in json.loads(sys.argv[1]):
 path=r['runtime_path'];assert path.startswith('/vllm-workspace/') and '..' not in PurePosixPath(path).parts
 proc=subprocess.run(['docker','cp',name+':'+path,'-'],capture_output=True,timeout=30);assert proc.returncode==0,proc.stderr.decode()
 with tarfile.open(fileobj=io.BytesIO(proc.stdout)) as archive:
  files=[x for x in archive.getmembers() if x.isfile()];assert len(files)==1
  raw=archive.extractfile(files[0]).read()
 h=sha256(raw).hexdigest();out.append(dict(**r,runtime_sha256=h,matches_local=h==r.get('local_sha256'),content=None if h==r.get('local_sha256') else raw.decode()))
print(json.dumps(dict(container_id=c['Id'],image=c['Image'],container_state=c['State']['Status'],read_method='docker cp individual file to stdout tar, parsed in memory; no extraction or process in container',sources=out,new_NPU_work=0,service_requests=0,new_profile_collection=0,remote_writes=[])))
'''

if __name__=='__main__':
 names=['vllm-ascend/vllm_ascend/ops/cv_linear.py','vllm-ascend/vllm_ascend/ops/linear.py','vllm-ascend/vllm_ascend/quantization/methods/w8a8/w8a8_dynamic.py','vllm-ascend/vllm_ascend/attention/dsa_v1.py','vllm-ascend/vllm_ascend/attention/context_parallel/dsa_cp.py','vllm-ascend/vllm_ascend/attention/context_parallel/dsa_v41_cp.py','vllm-ascend/vllm_ascend/models/deepseek_v41/model.py']
 spec=[]
 for n in names:
  p=FROZEN/n;spec.append(dict(runtime_path='/vllm-workspace/'+n,local_path=str(p),local_sha256=sha256(p.read_bytes()).hexdigest() if p.is_file() else None))
 r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE,json.dumps(spec)],capture_output=True,text=True,timeout=120)
 if r.returncode:raise RuntimeError(r.stderr)
 d=json.loads(r.stdout)
 for row in d['sources']:
  content=row.pop('content',None)
  if content is not None:
   p=OUT/'swa-query-producer-source'/row['runtime_path'].lstrip('/');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(content);assert sha256(p.read_bytes()).hexdigest()==row['runtime_sha256'];row['local_path']=str(p)
 d.update(status='SWA_QUERY_CV_PRODUCER_IDENTITY_ENTRY_ONLY',whole_chain_complete=False,every_replacement_bitwise=True,reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),reviewed_ranges=[],scope='Source identity only at acquisition; observed container exited, not a loaded/runtime service proof. Exact reviewed method/ranges separate; not all source bodies or native kernels covered.')
 (OUT/'swa-query-producer-entry.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(sources=len(d['sources']),matches=sum(r.get('matches_local',False) for r in d['sources']),missing=[r['runtime_path'] for r in d['sources'] if r.get('missing')],new_NPU_work=0,remote_writes=[])))
