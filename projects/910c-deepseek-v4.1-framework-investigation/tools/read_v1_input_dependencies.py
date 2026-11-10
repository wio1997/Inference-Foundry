#!/usr/bin/env python3
"""Read frozen V1 input/config/grouping dependencies; no container execution."""
from pathlib import Path
from hashlib import sha256
import json,subprocess
from read_swa_query_producer_entry import REMOTE
ROOT=Path('audit/execution-search')
def main():
    old=json.loads((ROOT/'whole-forward-dispatch.json').read_text())
    model=next(s for s in old['sources'] if s['runtime_path'].endswith('/models/deepseek_v41/model.py'))
    frozen=Path(model['local_path']).parents[4]
    names=['vllm/vllm/v1/worker/gpu_model_runner.py','vllm/vllm/config/model.py',
           'vllm/vllm/config/parallel.py','vllm/vllm/config/cache.py','vllm/vllm/envs.py',
           'vllm/vllm/v1/core/kv_cache_utils.py','vllm/vllm/v1/attention/backends/utils.py',
           'vllm-ascend/vllm_ascend/core/kv_cache_interface.py','vllm-ascend/vllm_ascend/attention/utils.py',
           'vllm-ascend/vllm_ascend/utils.py']
    prior={s['runtime_path']:s for s in old['sources']+[old['prior_exact_SHA_metadata_reference']]}
    for suffix in ['/worker/model_runner_v1.py','/attention/dsa_v41.py','/attention/context_parallel/dsa_v41_cp.py','/models/deepseek_v41/model.py','/models/deepseek_v41/vl_model.py','/worker/device_metadata.py']:
        s=next(v for k,v in prior.items() if k.endswith(suffix));names.append(s['runtime_path'].removeprefix('/vllm-workspace/'))
    spec=[]
    for n in names:
        runtime='/vllm-workspace/'+n;p=Path(prior[runtime]['local_path']) if runtime in prior else frozen/n
        spec.append(dict(runtime_path=runtime,local_path=str(p) if p.exists() else None,local_sha256=sha256(p.read_bytes()).hexdigest() if p.exists() else None))
    r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE,json.dumps(spec)],capture_output=True,text=True,timeout=115)
    if r.returncode: raise RuntimeError(r.stderr)
    d=json.loads(r.stdout)
    for s in d['sources']:
        body=s.pop('content',None)
        if body is not None:
            p=ROOT/'v1-input-runtime-source'/s['runtime_path'].lstrip('/');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(body)
            assert sha256(p.read_bytes()).hexdigest()==s['runtime_sha256'];s['local_path']=str(p)
    d.update(reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),whole_chain_complete=False,
        qualification='Disk identities and source bodies only; no Torch/Transformers/NPU import or loaded-object inspection. Exact caller/branch/read ranges recorded separately.')
    (ROOT/'v1-input-dependencies.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(sources=len(d['sources']),prior_hashes=sum(x['local_sha256'] is not None for x in d['sources']),prior_matches=sum(x['matches_local'] for x in d['sources']),new_bodies=sum(x['local_sha256'] is None for x in d['sources']),remote_writes=[],new_NPU_work=0)))
if __name__=='__main__':main()
