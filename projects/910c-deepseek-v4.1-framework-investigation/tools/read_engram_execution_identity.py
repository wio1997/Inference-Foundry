#!/usr/bin/env python3
"""Read-only Engram source/metadata qualification; no Torch import or NPU work."""
import hashlib
import json
from pathlib import Path
import subprocess

OUT = Path('audit/execution-search')
BASE = Path('/Users/wio/work/branch-fusion-audit-20261104')
FLOW = json.loads((BASE / 'full-review/flow-index.json').read_text())
SOURCES = []
for row in FLOW['files']:
    name = row['runtime_path']
    if (('/engram/' in name or '/triton/engram_' in name)
        or name.endswith(('/deepseek_v41/model.py', '/ops/linear.py',
                          '/hardware_profile.py', '/vllm_ascend/utils.py', '/sequence_parallel.py',
                          '/common/engram.py', '/worker/model_runner_v1.py'))):
        p = Path(row['local_path'])
        if p.exists():
            SOURCES.append(dict(path=name, local_path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
FROZEN = Path('/Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/whole-scope-20261030/frozen-source-snapshot/vllm-ascend')
for name in ['vllm_ascend/device/hardware_profile.py','vllm_ascend/ascend_config.py',
             'vllm_ascend/models/deepseek_v41/engram/parallel.py']:
    p=FROZEN/name
    path='/vllm-workspace/vllm-ascend/'+name
    if path not in {s['path'] for s in SOURCES}:
        SOURCES.append(dict(path=path,local_path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))

INNER = r'''
import hashlib,json,pathlib,struct,sys
root=pathlib.Path('/data/tiankuan/wio/DeepSeek-V4.1-Flash-1017-w8a8')
sources=json.loads(sys.argv[1]);pins=[]
for s in sources:
 p=pathlib.Path(s['path']);h=hashlib.sha256(p.read_bytes()).hexdigest()
 pins.append(dict(**s,runtime_sha256=h,matches=h==s['sha256']))
def document(name):
 p=root/name;raw=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest()),json.loads(raw)
cfgpin,cfg=document('config.json');cfg=cfg.get('text_config',cfg)
indexpin,index=document('quant_model_weights.safetensors.index.json');index=index['weight_map']
headers={}
def tensor(key):
 p=root/index[key]
 if p.name not in headers:
  with p.open('rb') as f:
   n=struct.unpack('<Q',f.read(8))[0];assert n<8*1024*1024;raw=f.read(n)
  headers[p.name]=(json.loads(raw),hashlib.sha256(raw).hexdigest())
 h,digest=headers[p.name]
 return dict(path=str(p),key=key,header_sha256=digest,tensor=h[key])
tensors=[tensor(k) for layer in [1,14] for k in sorted(index) if k.startswith(f'layers.{layer}.engram.')]
rot=root/'optional/quarot.safetensors';rotation=dict(path=str(rot),exists=rot.is_file())
if rot.is_file():
 with rot.open('rb') as f:
  n=struct.unpack('<Q',f.read(8))[0];assert n<8*1024*1024;raw=f.read(n)
 rotation.update(header_sha256=hashlib.sha256(raw).hexdigest(),header=json.loads(raw))
quant=[]
for n in ['quant_model_description.json','quant_model_description_w8a8.json']:
 p=root/n
 if p.is_file():
  raw=p.read_bytes();d=json.loads(raw)
  entries={k:v for k,v in d.items() if '.engram.wkv' in k or 'rotation' in k.lower() or k in ['model_quant_type','quant_type','optional']}
  quant.append(dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),entries=entries))
nvidia=pathlib.Path('/vllm-workspace/vllm/vllm/models/deepseek_v41/nvidia/engram.py')
raw=nvidia.read_bytes()
print(json.dumps(dict(source_pins=pins,nvidia_parallel_source=dict(path=str(nvidia),sha256=hashlib.sha256(raw).hexdigest(),content=raw.decode()),model_config=cfgpin,config={k:v for k,v in cfg.items() if k.startswith('engram') or k in ['hidden_size','hc_mult','rms_norm_eps']},checkpoint_index=indexpin,tensors=tensors,rotation=rotation,quantization=quant)))
'''
REMOTE = r'''
import hashlib,json,pathlib,shlex,subprocess,sys
p=json.loads(sys.argv[1]);name='wio-vllm-v41-repro171-20261102'
i,=json.loads(subprocess.check_output(['docker','inspect',name]))
assert i['State']['Running'];assert i['Image']=='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
r=subprocess.run(['docker','exec',name,'python3','-I','-S','-c',p['inner'],json.dumps(p['sources'])],capture_output=True,text=True,timeout=50)
assert r.returncode==0,r.stderr
d=json.loads(r.stdout)
serve=pathlib.Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/scripts/serve.sh')
raw=serve.read_bytes();line=next(x for x in raw.decode().splitlines() if x.startswith('exec ') and 'vllm serve' in x)
argv=shlex.split(line);selected={}
for option in ['--additional-config','--engram-config','--speculative-config','--compilation-config','--quantization','--tensor-parallel-size','--data-parallel-size']:
 if option in argv:selected[option]=argv[argv.index(option)+1]
d['saved_service_config']=dict(path=str(serve),sha256=hashlib.sha256(raw).hexdigest(),selected_options=selected)
d.update(container_id=i['Id'],started_at=i['State']['StartedAt'],image=i['Image'],new_NPU_work=False,service_changes=[],remote_writes=[])
print(json.dumps(d))
'''

if __name__ == '__main__':
    p = subprocess.run(['/Users/wio/.local/bin/baai-ssh', '171', '--', 'python3', '-c', REMOTE,
                        json.dumps(dict(inner=INNER, sources=SOURCES))], capture_output=True, text=True, timeout=70)
    assert p.returncode == 0, p.stderr
    d = json.loads(p.stdout)
    nvidia = d['nvidia_parallel_source'];dest=OUT/'engram-runtime-source/nvidia-engram.py'
    dest.parent.mkdir(exist_ok=True);dest.write_text(nvidia.pop('content'))
    nvidia['local_path']=str(dest.resolve())
    d['reader_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT / 'engram-execution-identity.json').write_text(json.dumps(d, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(sources=len(d['source_pins']), mismatches=[r['path'] for r in d['source_pins'] if not r['matches']], nvidia_source=nvidia, service_config=d['saved_service_config'], new_NPU_work=False), ensure_ascii=False))
