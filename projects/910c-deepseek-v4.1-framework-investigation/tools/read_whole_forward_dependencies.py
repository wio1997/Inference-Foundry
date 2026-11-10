#!/usr/bin/env python3
"""Current dispatch/MRO dependencies and whitelisted frozen launch/config facts."""
from pathlib import Path
from hashlib import sha256
import json,subprocess
from read_swa_query_producer_entry import REMOTE as SOURCES
ROOT=Path('audit/execution-search')
CONFIG=r'''
from pathlib import Path
from hashlib import sha256
import io,json,shlex,subprocess,tarfile,re
name='wio-vllm-v41-repro171-20261102';c,=json.loads(subprocess.check_output(['docker','inspect',name]));assert c['Id']=='3be44e7b5c292de6ef7fc51085bfbbad5552acc97fae39c0ae176fe8cfb387eb' and c['Image']=='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
path='/data/tiankuan/wio/DeepSeek-V4.1-Flash-1017-w8a8/config.json';p=subprocess.run(['docker','cp',name+':'+path,'-'],capture_output=True,timeout=30);assert p.returncode==0
with tarfile.open(fileobj=io.BytesIO(p.stdout)) as a:
 fs=[v for v in a.getmembers() if v.isfile()];assert len(fs)==1;raw=a.extractfile(fs[0]).read()
config=dict(path=path,sha256=sha256(raw).hexdigest(),content=json.loads(raw))
p=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/scripts/serve.sh');raw=p.read_bytes();tokens=shlex.split(raw.decode().replace('\\\n',' '),comments=True)
allow={'--tensor-parallel-size','--data-parallel-size','--pipeline-parallel-size','--enable-expert-parallel','--enable-sequence-parallelism','--enable-sequence-parallel-moe','--additional-config','--speculative-config','--compilation-config','--quantization','--enable-chunked-prefill','--block-size','--max-model-len','--max-num-seqs','--max-num-batched-tokens','--async-scheduling','--enforce-eager','--enable-lora'}
json_keys={'enable_fused_mc2','enable_dsa_cp','use_sequence_parallel_moe','method','num_speculative_tokens','enforce_eager','cudagraph_mode','cudagraph_capture_sizes','enable_graph_mode','dynamic_spec_config','model','speculative_token_tree','mix_placement','enable_eplb'}
def filtered(value):
 if isinstance(value,dict):return {k:filtered(v) for k,v in value.items() if k in json_keys}
 if isinstance(value,list):return [filtered(v) for v in value]
 if value is None or isinstance(value,(bool,int,float)):return value
 return value if isinstance(value,str) and len(value)<80 and re.fullmatch(r'[A-Za-z0-9_ .-]+',value) else '<non-scalar-or-path-redacted>'
options={}
for i,t in enumerate(tokens):
 k=t.split('=',1)[0]
 if k not in allow:continue
 v=t.split('=',1)[1] if '=' in t else tokens[i+1] if i+1<len(tokens) and not tokens[i+1].startswith('--') else True
 if k in {'--additional-config','--speculative-config','--compilation-config'}:
  try:v=filtered(json.loads(v))
  except (TypeError,ValueError):v={'nonliteral_not_expanded':True}
 else:v=filtered(v)
 options[k]=v
env_keys={'VLLM_USE_V1','VLLM_USE_V2_MODEL_RUNNER','VLLM_MOE_SKIP_PADDING','VLLM_ASCEND_ENABLE_FUSED_MC2','VLLM_ASCEND_ENABLE_DSA_CP','VLLM_ASCEND_ENABLE_FLASHCOMM'};env={}
for k in env_keys:
 m=re.search(r'(?:^|\n)\s*(?:export\s+)?'+k+r'=[\x22\x27]?(0|1|True|False)[\x22\x27]?(?:\s|$)',raw.decode())
 if m:env[k]=m.group(1)
print(json.dumps(dict(container_id=c['Id'],image=c['Image'],container_state=c['State']['Status'],model_config=config,saved_serve=dict(path=str(p),sha256=sha256(raw).hexdigest(),selected_options=options,selected_bool_env=env,absent_whitelisted_options=sorted(allow-set(options))),qualification='Saved script and current model-config bytes only, no shell expansion/process-env/loaded-object/health inference. All other launch/log fields discarded; no credentials emitted.',new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[])))
'''
def main():
    old=json.loads((ROOT/'whole-forward-entry.json').read_text());model=next(s for s in old['sources'] if s['runtime_path'].endswith('/model.py'));frozen=Path(model['local_path']).parents[4];prior={s['runtime_path']:s for s in old['sources']}
    extras=['vllm-ascend/vllm_ascend/__init__.py','vllm-ascend/vllm_ascend/models/deepseek_v41/vl_model.py','vllm-ascend/vllm_ascend/models/deepseek_v41/cache_config.py','vllm-ascend/vllm_ascend/device/hardware_profile.py','vllm-ascend/vllm_ascend/device/device_op.py','vllm-ascend/vllm_ascend/models/common/ops/sequence_parallel.py','vllm-ascend/vllm_ascend/utils.py','vllm-ascend/vllm_ascend/ascend_config.py','vllm-ascend/vllm_ascend/spec_decode/dflash_proposer.py','vllm-ascend/vllm_ascend/spec_decode/eagle_proposer.py','vllm-ascend/vllm_ascend/spec_decode/utils.py','vllm-ascend/vllm_ascend/ops/fused_moe/fused_moe.py','vllm-ascend/vllm_ascend/ops/fused_moe/moe_comm_method.py','vllm-ascend/vllm_ascend/quantization/methods/w8a8/w8a8_dynamic.py','vllm/vllm/model_executor/models/registry.py','vllm/vllm/model_executor/models/interfaces.py','vllm/vllm/model_executor/layers/fused_moe/layer.py','vllm/vllm/model_executor/layers/fused_moe/runner/moe_runner.py','vllm/vllm/v1/spec_decode/eagle.py','vllm/vllm/v1/spec_decode/llm_base_proposer.py']
    for n in extras:
        p=frozen/n;name='/vllm-workspace/'+n;prior[name]=dict(runtime_path=name,local_path=str(p) if p.exists() else None)
    spec=[]
    for n,s in sorted(prior.items()):
        p=Path(s['local_path']) if s.get('local_path') else None;spec.append(dict(runtime_path=n,local_path=str(p) if p else None,local_sha256=sha256(p.read_bytes()).hexdigest() if p and p.exists() else None))
    r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',SOURCES,json.dumps(spec)],capture_output=True,text=True,timeout=115)
    if r.returncode:raise RuntimeError(r.stderr)
    d=json.loads(r.stdout)
    for s in d['sources']:
        body=s.pop('content',None)
        if body is not None:
            p=ROOT/'whole-forward-dependency-source'/s['runtime_path'].lstrip('/');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(body);s['local_path']=str(p)
    d.update(reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),qualification='Current disk identities only; exact reviewed ranges/selected dispatch/branch legality separate. No imports of Torch/model/device runtime.')
    (ROOT/'whole-forward-dependencies.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',CONFIG],capture_output=True,text=True,timeout=70)
    if r.returncode:raise RuntimeError(r.stderr)
    c=json.loads(r.stdout);c['reader_sha256']=d['reader_sha256'];(ROOT/'whole-forward-frozen-config.json').write_text(json.dumps(c,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(sources=len(d['sources']),matches=sum(s['matches_local'] for s in d['sources']),config_sha=c['model_config']['sha256'],serve_sha=c['saved_serve']['sha256'],options=c['saved_serve']['selected_options'],env=c['saved_serve']['selected_bool_env'],new_NPU_work=0)))
if __name__=='__main__':main()
