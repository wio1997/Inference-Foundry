#!/usr/bin/env python3
"""Read safetensors HEADER JSON only, never tensor payload or framework code."""
from hashlib import sha256
import json
from pathlib import Path
import subprocess

REMOTE=r'''
from pathlib import Path
from hashlib import sha256
import ast,json,subprocess,sys
name='wio-vllm-v41-repro171-20261102';item,=json.loads(subprocess.check_output(['docker','inspect',name]));assert item['State']['Running'] and item['Image']=='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
inner=r"""from pathlib import Path
from hashlib import sha256
import ast,json,struct,math
model=Path('/data/tiankuan/wio/DeepSeek-V4.1-Flash-1017-w8a8');p=model/'model.safetensors.index.json'
if p.is_file():
 raw=p.read_bytes();index=json.loads(raw);index_record=dict(path=str(p),sha256=sha256(raw).hexdigest(),bytes=len(raw))
else:
 raw=b'';index=None;index_record=dict(path=str(p),present=False,scope='Use only direct safetensors HEADER catalog, no arbitrary bin/weight reads')
source=Path('/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/vl_model.py');b=source.read_bytes();m=ast.parse(b);fn=next(n for n in m.body if isinstance(n,ast.FunctionDef) and n.name=='_vision_parameter_name');env={};exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(source),'exec'),env)
groups={}
if index is not None:
 keys=[k for k in index['weight_map'] if env['_vision_parameter_name'](k) is not None]
 for k in keys:groups.setdefault(index['weight_map'][k],[]).append(k)
else:
 groups={p.name:None for p in model.glob('*.safetensors') if p.is_file()}
 assert groups,'No safetensors catalog available; do not infer zero resident vision'
rows=[];headers=[]
for rel,kk in sorted(groups.items()):
 p=model/rel;assert p.resolve().is_relative_to('/data/tiankuan/wio');st=p.stat()
 with p.open('rb') as f:
  nb=f.read(8);assert len(nb)==8;n=struct.unpack('<Q',nb)[0];assert 0<n<32*1024*1024
  rawheader=f.read(n);assert len(rawheader)==n;header=json.loads(rawheader)
 if kk is None:kk=[k for k in header if k!='__metadata__' and env['_vision_parameter_name'](k) is not None]
 for k in kk:
  value=header[k];a,z=value['data_offsets'];rows.append(dict(name=k,parameter=env['_vision_parameter_name'](k),dtype=value['dtype'],shape=value['shape'],numel=math.prod(value['shape']),payload_bytes=z-a))
 assert (st.st_size,st.st_mtime_ns)==(p.stat().st_size,p.stat().st_mtime_ns)
 headers.append(dict(path=str(p),file_bytes=st.st_size,mtime_ns=st.st_mtime_ns,header_bytes=8+n,header_sha256=sha256(nb+rawheader).hexdigest(),selected_tensors=len(kk)))
print(json.dumps(dict(index=index_record,source=dict(path=str(source),sha256=sha256(b).hexdigest(),method='_vision_parameter_name original AST'),selected=rows,headers=headers,total_numel=sum(r['numel'] for r in rows),total_checkpoint_payload_bytes=sum(r['payload_bytes'] for r in rows),tensor_payload_bytes_read=0,scope='Header shapes/dtypes under exact original vision/aligner/image naming rule. Checkpoint tensors, not live allocator residency; source constructor/dtype and non-parameter buffers separate.')))
"""
r=subprocess.run(['docker','exec',name,'python3','-B','-I','-S','-c',inner],capture_output=True,text=True)
if r.returncode:print(r.stderr,file=sys.stderr);sys.exit(r.returncode)
x=json.loads(r.stdout);x.update(container_id=item['Id'],image=item['Image'],model_weights_loads=0,new_NPU_work=0,service_requests=0,new_profile_collection=0,remote_writes=[]);print(json.dumps(x))
'''


def main():
    r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE],capture_output=True,text=True,timeout=100)
    if r.returncode:raise RuntimeError(r.stderr[-2500:])
    x=json.loads(r.stdout);x['reader_sha256']=sha256(Path(__file__).read_bytes()).hexdigest()
    Path('audit/execution-search/startup-vision-header-bank.json').write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(tensors=len(x['selected']),headers=len(x['headers']),elements=x['total_numel'],checkpoint_payload_MiB=x['total_checkpoint_payload_bytes']/1024**2,payload_bytes_read=0,new_NPU_work=0)))


if __name__=='__main__':main()
