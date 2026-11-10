#!/usr/bin/env python3
"""Bounded current-source caller catalog and checkpoint headers, payload reads zero."""
from pathlib import Path
from hashlib import sha256
import json,subprocess

INNER=r'''
from pathlib import Path
from hashlib import sha256
import json,struct
roots=[Path('/vllm-workspace/vllm-ascend/vllm_ascend'),Path('/vllm-workspace/vllm/vllm')];patterns=['close_host_offload','bind_checkpoint','engram_storage_dtype'];hits=[];scanned=0
for root in roots:
 for p in root.rglob('*.py'):
  text=p.read_text();scanned+=1
  selected=[dict(line=i,text=l.strip()) for i,l in enumerate(text.splitlines(),1) if any(k in l for k in patterns)]
  if selected:hits.append(dict(path=str(p),sha256=sha256(p.read_bytes()).hexdigest(),matches=selected))
asset=Path('/data/tiankuan/wio/DeepSeek-V4.1-Flash-1017-w8a8');raw=(asset/'quant_model_weights.safetensors.index.json').read_bytes();index=json.loads(raw)['weight_map'];headers=[]
for layer in [1,14]:
 key=f'layers.{layer}.engram.embed.weight';scale=key.removesuffix('.weight')+'.scale';p=asset/index[key]
 assert index[key]==index[scale]
 with p.open('rb') as f:
  n=struct.unpack('<Q',f.read(8))[0];assert n<8*1024*1024;hraw=f.read(n)
 h=json.loads(hraw);headers.append(dict(layer=layer,path=str(p),header_sha256=sha256(hraw).hexdigest(),weight=h[key],scale=h[scale]))
header_candidates=['/usr/local/Ascend/cann-9.2.0/include/acl/acl_rt.h','/usr/local/Ascend/cann-9.2.0/aarch64-linux/include/acl/acl_rt.h'];sdk=[]
for name in header_candidates:
 p=Path(name)
 if p.is_file():
  data=p.read_bytes();lines=data.decode().splitlines();selected=[dict(line=i,text=l) for i,l in enumerate(lines,1) if any(t in l for t in ['aclrtHostRegisterV2','aclrtHostGetDevicePointer','aclrtHostUnregister','ACL_HOST_REG_MAPPED','ACL_HOST_REG_PINNED'])];sdk.append(dict(path=name,sha256=sha256(data).hexdigest(),anchors=selected))
print(json.dumps(dict(search_roots=list(map(str,roots)),python_files_searched=scanned,patterns=patterns,caller_hits=hits,checkpoint_index_sha256=sha256(raw).hexdigest(),checkpoint_headers=headers,checkpoint_payload_bytes_read=0,CANN_header_catalog=sdk)))
'''
REMOTE=r'''
import json,subprocess,sys
name='wio-vllm-v41-repro171-20261102';item,=json.loads(subprocess.check_output(['docker','inspect',name]));assert item['State']['Running'] and item['Image']=='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
r=subprocess.run(['docker','exec',name,'python3','-B','-I','-S','-c',sys.argv[1]],capture_output=True,text=True,timeout=45);assert r.returncode==0,r.stderr;value=json.loads(r.stdout);value.update(container_id=item['Id'],image=item['Image']);print(json.dumps(value))
'''
if __name__=='__main__':
 r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE,INNER],capture_output=True,text=True,timeout=65)
 if r.returncode:raise RuntimeError(r.stderr)
 value=json.loads(r.stdout);value.update(reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),new_NPU_work=0,new_profile_collection=0,model_loads=0,service_requests=0,remote_writes=[]);Path('audit/execution-search/engram-storage-catalog.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(python_files_searched=value['python_files_searched'],caller_files=len(value['caller_hits']),close_hits=[r for r in value['caller_hits'] if any('close_host_offload' in m['text'] for m in r['matches'])],SDK_headers=[r['path'] for r in value['CANN_header_catalog']],payload_bytes_read=0)))
