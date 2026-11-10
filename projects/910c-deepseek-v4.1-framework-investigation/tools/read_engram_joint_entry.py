#!/usr/bin/env python3
"""Current Engram joint-delivery entry, source and header metadata only."""
from pathlib import Path
from hashlib import sha256
import json,subprocess
from read_qkv_native_catalog import REMOTE
OUT=Path('audit/execution-search')
INNER=r'''
from pathlib import Path
from hashlib import sha256
import json,struct
spec=json.loads(SPEC);sources=[]
for r in spec:
 p=Path(r['runtime_path']);assert p.is_file(),str(p);b=p.read_bytes();h=sha256(b).hexdigest();sources.append(dict(**r,runtime_sha256=h,matches_local=h==r.get('local_sha256'),content=None if h==r.get('local_sha256') else b.decode()))
root=Path('/data/tiankuan/wio/DeepSeek-V4.1-Flash-1017-w8a8');configs=[]
for n in ['config.json','quant_model_description.json']:
 p=root/n
 if not p.is_file():continue
 b=p.read_bytes();d=json.loads(b)
 if n=='config.json':
  d=d.get('text_config',d);sel={k:v for k,v in d.items() if k.startswith('engram') or k in ['hidden_size','hc_mult','rms_norm_eps','rotation_path','quarot_path']}
 else:sel={k:v for k,v in d.items() if '.engram.wkv.' in k or 'rotation' in k.lower() or k=='optional'}
 configs.append(dict(path=str(p),sha256=sha256(b).hexdigest(),selected=sel))
p=root/'quant_model_weights.safetensors.index.json';b=p.read_bytes();idx=json.loads(b)['weight_map'];headers={};tensors=[]
for key in sorted(k for k in idx if '.engram.wkv.' in k):
 p=root/idx[key]
 if str(p) not in headers:
  with p.open('rb') as f:n=struct.unpack('<Q',f.read(8))[0];assert 0<n<16*1024*1024;raw=f.read(n);assert len(raw)==n
  headers[str(p)]=dict(path=str(p),header_bytes=n,header_sha256=sha256(raw).hexdigest(),data=json.loads(raw))
 h=headers[str(p)];tensors.append(dict(name=key,path=str(p),header_sha256=h['header_sha256'],metadata=h['data'][key]))
rot=root/'optional/quarot.safetensors';rotation=dict(path=str(rot),exists=rot.is_file())
if rot.is_file():
 with rot.open('rb') as f:n=struct.unpack('<Q',f.read(8))[0];assert 0<n<16*1024*1024;raw=f.read(n);assert len(raw)==n
 rotation.update(header_sha256=sha256(raw).hexdigest(),header=json.loads(raw))
print(json.dumps(dict(sources=sources,configs=configs,checkpoint_index=dict(path=str(root/'quant_model_weights.safetensors.index.json'),sha256=sha256(b).hexdigest()),headers=[{k:v for k,v in h.items() if k!='data'} for h in headers.values()],WKV_tensors=tensors,rotation=rotation,payload_bytes_read=0)))
'''
def main():
 prior=[]
 for name,key in [('hc-coefficient-consumer-dependencies.json','sources'),('engram-execution-identity.json','source_pins'),('engram-storage-dependencies.json','sources'),('o-projection-native-dependencies.json','sources')]:
  p=OUT/name
  if not p.is_file():continue
  for r in json.loads(p.read_text()).get(key,[]):
   if 'runtime_path' not in r:r=dict(r,runtime_path=r['path'])
   prior.append(r)
 names=['models/deepseek_v41/model.py','models/deepseek_v41/engram/__init__.py','models/deepseek_v41/engram/common.py','models/deepseek_v41/engram/layer.py','models/deepseek_v41/engram/embedding.py','models/deepseek_v41/engram/parallel.py','ops/triton/engram_gate.py','ops/linear.py','utils.py']
 spec=[]
 for name in names:
  runtime='/vllm-workspace/vllm-ascend/vllm_ascend/'+name;r=next((r for r in reversed(prior) if r['runtime_path']==runtime),None);p=Path(r.get('local_path',r.get('path'))) if r else None
  spec.append(dict(runtime_path=runtime,local_path=str(p) if p else None,local_sha256=sha256(p.read_bytes()).hexdigest() if p and p.is_file() else None))
 code='SPEC='+repr(json.dumps(spec))+'\n'+INNER;r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE,code],capture_output=True,text=True,timeout=120)
 if r.returncode:raise RuntimeError(r.stderr)
 d=json.loads(r.stdout)
 for row in d['sources']:
  body=row.pop('content',None)
  if body is not None:
   p=OUT/'engram-joint-entry-source'/row['runtime_path'].lstrip('/');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(body);assert sha256(p.read_bytes()).hexdigest()==row['runtime_sha256'];row['local_path']=str(p)
 d.update(reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),scope='Current source/header entry only. Full reviewed ranges/value/nativebits separate.',new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[]);(OUT/'engram-joint-entry.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(sources=len(d['sources']),matches=sum(r.get('matches_local',False) for r in d['sources']),WKV_headers=len(d['headers']),WKV_tensors=d['WKV_tensors'],rotation=d['rotation'],payload_bytes_read=0),ensure_ascii=False))
if __name__=='__main__':main()
