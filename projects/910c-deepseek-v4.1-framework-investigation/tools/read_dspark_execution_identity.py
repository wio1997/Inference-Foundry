#!/usr/bin/env python3
"""Read-only DSpark source, dense tensor headers and frozen branch options."""
import hashlib
import json
from pathlib import Path
import subprocess

OUT=Path('audit/execution-search')
BASE=Path('/Users/wio/work/branch-fusion-audit-20261104')
FROZEN=Path('/Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/whole-scope-20261030/frozen-source-snapshot')
names=['/models/deepseek_v41/dspark.py','/models/deepseek_v41/model.py',
       '/spec_decode/dspark_proposer.py','/spec_decode/dflash_proposer.py',
       '/spec_decode/llm_base_proposer.py','/models/qwen3_dspark.py',
       '/layers/logits_processor.py','/layers/vocab_parallel_embedding.py',
       '/ops/vocab_parallel_embedding.py','/sample/rejection_sampler.py',
       '/sample/sampler.py','/worker/model_runner_v1.py',
       '/triton/spec_decode/utils.py','/spec_decode/utils.py']
sources=[]
flow=json.loads((BASE/'full-review/flow-index.json').read_text())
for row in flow['files']:
    if any(row['runtime_path'].endswith(n) for n in names):
        p=Path(row['local_path']);sources.append(dict(path=row['runtime_path'],local_path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
for package,name in [('vllm','vllm/config/speculative.py'),('vllm-ascend','vllm_ascend/ascend_config.py'),
                     ('vllm-ascend','vllm_ascend/utils.py')]:
    p=FROZEN/package/name;path='/vllm-workspace/'+package+'/'+name
    if path not in {s['path'] for s in sources}:
        sources.append(dict(path=path,local_path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
patch=OUT/'dspark-runtime-source/patch-source-manifest.json'
if patch.is_file():
    d=json.loads(patch.read_text());p=Path(d['local_path'])
    assert hashlib.sha256(p.read_bytes()).hexdigest()==d['sha256']
    sources.append(dict(path=d['path'],local_path=str(p),sha256=d['sha256']))

INNER=r'''
import hashlib,json,pathlib,struct,sys
root=pathlib.Path('/data/tiankuan/wio/DeepSeek-V4.1-Flash-1017-w8a8')
sources=json.loads(sys.argv[1]);pins=[]
for s in sources:
 h=hashlib.sha256(pathlib.Path(s['path']).read_bytes()).hexdigest()
 pins.append(dict(**s,runtime_sha256=h,matches=h==s['sha256']))
def doc(name):
 p=root/name;raw=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest()),json.loads(raw)
cp,cfg=doc('config.json');tc=cfg.get('text_config',cfg)
ip,index=doc('quant_model_weights.safetensors.index.json');index=index['weight_map']
qp,quant=doc('quant_model_description.json');headers={};tensors=[]
for key in sorted(index):
 if not key.startswith('mtp.') or not key.endswith('weight'):continue
 if not any(s in key for s in ['main_proj','markov_head','confidence_head','head.weight','.wkv.','.kv_norm.']):continue
 p=root/index[key]
 if p.name not in headers:
  with p.open('rb') as f:
   n=struct.unpack('<Q',f.read(8))[0];assert n<8*1024*1024;raw=f.read(n)
  headers[p.name]=(json.loads(raw),hashlib.sha256(raw).hexdigest())
 h,digest=headers[p.name]
 tensors.append(dict(path=str(p),key=key,header_sha256=digest,tensor=h[key],quant_description=quant.get(key)))
fields=['hidden_size','hc_mult','vocab_size','draft_vocab_size','num_hidden_layers','num_nextn_predict_layers','rms_norm_eps','sample_from_anchor','draft_sample_method','dspark_draft_topk','n_routed_experts','num_experts_per_tok','moe_intermediate_size','sliding_window','num_attention_heads','qk_rope_head_dim','qk_nope_head_dim']
print(json.dumps(dict(source_pins=pins,config_pin=cp,config={k:v for k,v in tc.items() if k.startswith('dspark') or k in fields},composite_dspark_fields={k:v for k,v in cfg.items() if 'dspark' in k or k in ['sample_from_anchor','draft_sample_method']},checkpoint_index=ip,quant_description=qp,tensors=tensors,new_NPU_work=False)))
'''
REMOTE=r'''
import hashlib,json,pathlib,shlex,subprocess,sys
p=json.loads(sys.argv[1]);name='wio-vllm-v41-repro171-20261102'
i,=json.loads(subprocess.check_output(['docker','inspect',name]));assert i['State']['Running']
assert i['Image']=='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
r=subprocess.run(['docker','exec',name,'python3','-I','-S','-c',p['inner'],json.dumps(p['sources'])],capture_output=True,text=True,timeout=45)
assert r.returncode==0,r.stderr;d=json.loads(r.stdout)
serve=pathlib.Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/scripts/serve.sh');raw=serve.read_bytes()
line=next(x for x in raw.decode().splitlines() if x.startswith('exec ') and 'vllm serve' in x);argv=shlex.split(line)
options={}
for k in ['--speculative-config','--additional-config','--tensor-parallel-size','--data-parallel-size','--compilation-config','--async-scheduling']:
 if k in argv:options[k]=True if k=='--async-scheduling' else argv[argv.index(k)+1]
d.update(saved_service_config=dict(path=str(serve),sha256=hashlib.sha256(raw).hexdigest(),options=options),container_id=i['Id'],started_at=i['State']['StartedAt'],image=i['Image'],remote_writes=[],service_requests=0)
print(json.dumps(d))
'''

if __name__=='__main__':
    p=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-c',REMOTE,json.dumps(dict(inner=INNER,sources=sources))],capture_output=True,text=True,timeout=60)
    assert p.returncode==0,p.stderr
    d=json.loads(p.stdout);d['reader_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT/'dspark-execution-identity.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(sources=len(d['source_pins']),mismatches=[s['path'] for s in d['source_pins'] if not s['matches']],config=d['config'],tensors=[{k:t[k] for k in ['key','tensor','quant_description']} for t in d['tensors']],new_NPU_work=False),ensure_ascii=False))
