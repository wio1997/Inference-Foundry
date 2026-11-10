#!/usr/bin/env python3
"""Pin actual embedding/config selection and only relevant saved-launch fields."""
from pathlib import Path
from hashlib import sha256
import json,subprocess
from read_swa_query_producer_entry import REMOTE
ROOT=Path('audit/execution-search')
CONFIG=r'''
from pathlib import Path
from hashlib import sha256
import json,shlex
p=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/scripts/serve.sh');b=p.read_bytes();tokens=shlex.split(b.decode().replace('\\\n',' '),comments=True)
allow={'--limit-mm-per-prompt','--enable-prompt-embeds','--language-model-only','--skip-mm-profiling','--mm-encoder-tp-mode','--runner','--convert','--model-impl','--hf-overrides','--additional-config'}
options={}
for i,t in enumerate(tokens):
 k=t.split('=',1)[0]
 if k not in allow:continue
 v=t.split('=',1)[1] if '=' in t else tokens[i+1] if i+1<len(tokens) and not tokens[i+1].startswith('--') else True
 if k=='--additional-config':
  try:
   d=json.loads(v);fine=d.get('finegrained_tp_config',{});v={'finegrained_tp_config':{n:x for n,x in fine.items() if n in {'embedding_tensor_parallel_size','lmhead_tensor_parallel_size'} and isinstance(x,(int,bool))},'embedding_lmhead_weight_sharding':d.get('enable_pcp_embedding_lmhead_weight_sharding','ABSENT')}
  except (ValueError,TypeError,AttributeError):v={'nonliteral_not_expanded':True}
 elif k in {'--limit-mm-per-prompt','--hf-overrides'}:
  try:
   d=json.loads(v);keys={'image','video','audio'} if k=='--limit-mm-per-prompt' else {'vision_n_layers','is_mm_prefix_lm','model_type','architectures','tie_word_embeddings'}
   v={n:x for n,x in d.items() if n in keys and isinstance(x,(bool,int,float,str,list))}
  except (ValueError,TypeError,AttributeError):v={'nonliteral_not_expanded':True}
 else:
  v=v if isinstance(v,bool) or isinstance(v,str) and len(v)<80 and all(c.isalnum() or c in '_- ' for c in v) else '<nonliteral-redacted>'
 options[k]=v
print(json.dumps(dict(saved_serve_path=str(p),sha256=sha256(b).hexdigest(),selected_options=options,absent_options=sorted(allow-set(options)),qualification='Saved script literals only; no shell expansion/live env/loaded config claim. Secrets and all unrelated options discarded.',remote_writes=[],new_NPU_work=0,service_requests=0)))
'''
def main():
    old=json.loads((ROOT/'whole-forward-dispatch.json').read_text());previous=json.loads((ROOT/'v1-input-dependencies.json').read_text())
    prior={s['runtime_path']:s for s in old['sources']}
    prior.update({s['runtime_path']:s for s in previous['sources']})
    model=next(s for k,s in prior.items() if k.endswith('/models/deepseek_v41/model.py'));frozen=Path(model['local_path']).parents[4]
    names=['vllm/vllm/transformers_utils/model_arch_config_convertor.py','vllm/vllm/multimodal/registry.py','vllm/vllm/config/multimodal.py','vllm/vllm/model_executor/layers/vocab_parallel_embedding.py','vllm-ascend/vllm_ascend/ops/__init__.py']
    suffixes=['/config/model.py','/v1/worker/gpu_model_runner.py','/models/deepseek_v41/model.py','/models/deepseek_v41/vl_model.py','/worker/model_runner_v1.py','/spec_decode/dspark_proposer.py','/vllm_ascend/spec_decode/llm_base_proposer.py']
    for suf in suffixes:names.append(next(k for k in prior if k.endswith(suf)).removeprefix('/vllm-workspace/'))
    spec=[]
    for n in names:
        runtime='/vllm-workspace/'+n;p=Path(prior[runtime]['local_path']) if runtime in prior else frozen/n
        spec.append(dict(runtime_path=runtime,local_path=str(p) if p.exists() else None,local_sha256=sha256(p.read_bytes()).hexdigest() if p.exists() else None))
    r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE,json.dumps(spec)],capture_output=True,text=True,timeout=115)
    if r.returncode:raise RuntimeError(r.stderr)
    d=json.loads(r.stdout)
    for s in d['sources']:
        body=s.pop('content',None)
        if body is not None:
            p=ROOT/'embedding-entry-source'/s['runtime_path'].lstrip('/');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(body);assert sha256(p.read_bytes()).hexdigest()==s['runtime_sha256'];s['local_path']=str(p)
    d.update(reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),whole_chain_complete=False,qualification='Current disk identities; reviewed ranges and chosen dispatch still separate. No framework import/container process or remote write.')
    (ROOT/'embedding-entry-dependencies.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',CONFIG],capture_output=True,text=True,timeout=60)
    if r.returncode:raise RuntimeError(r.stderr)
    cfg=json.loads(r.stdout);assert cfg['sha256']==json.loads((ROOT/'whole-forward-frozen-config.json').read_text())['saved_serve']['sha256'];cfg['reader_sha256']=d['reader_sha256'];(ROOT/'embedding-selected-launch.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(sources=len(d['sources']),matches=sum(s['matches_local'] for s in d['sources']),new_bodies=sum(s['local_sha256'] is None for s in d['sources']),options=cfg['selected_options'],remote_writes=[],new_NPU_work=0)))
if __name__=='__main__':main()
