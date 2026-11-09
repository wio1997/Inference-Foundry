#!/usr/bin/env python3
"""Read-only source pins for actual V1 metadata/host coordination, without importing vLLM."""
import hashlib
import json
from pathlib import Path
import subprocess

BASE=Path('/Users/wio/work/branch-fusion-audit-20261104')
FROZEN=Path('/Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/whole-scope-20261030/frozen-source-snapshot')
OUT=Path('audit/execution-search')
NAMES=['vllm-ascend/vllm_ascend/worker/model_runner_v1.py', 'vllm-ascend/vllm_ascend/spec_decode/dspark_proposer.py', 'vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py', 'vllm-ascend/vllm_ascend/attention/context_parallel/dsa_v41_cp.py', 'vllm-ascend/vllm_ascend/attention/context_parallel/dsa_cp.py', 'vllm-ascend/vllm_ascend/attention/dsa_v41.py', 'vllm-ascend/vllm_ascend/attention/dsa_v1.py', 'vllm-ascend/vllm_ascend/worker/device_metadata.py', 'vllm-ascend/vllm_ascend/ascend_forward_context.py', 'vllm-ascend/vllm_ascend/utils.py', 'vllm-ascend/vllm_ascend/ascend_config.py', 'vllm-ascend/vllm_ascend/models/deepseek_v41/dspark.py', 'vllm/vllm/distributed/parallel_state.py', 'vllm-ascend/vllm_ascend/models/deepseek_v41/model.py', 'vllm-ascend/vllm_ascend/models/deepseek_v41/indexer.py', 'vllm-ascend/vllm_ascend/models/deepseek_v41/mixed_quant_attention.py']
REMOTE=r'''
import hashlib,json,pathlib,subprocess,sys
spec=json.loads(sys.argv[1]);name='wio-vllm-v41-repro171-20261102'
i,=json.loads(subprocess.check_output(['docker','inspect',name]));assert i['State']['Running'] and i['Image']=='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
inner="""import hashlib,json,pathlib,sys
out=[]
for row in json.loads(sys.argv[1]):
 raw=pathlib.Path(row['runtime_path']).read_bytes();sha=hashlib.sha256(raw).hexdigest()
 out.append(dict(**row,runtime_sha256=sha,matches_local=sha==row.get('local_sha256'),content=None if sha==row.get('local_sha256') else raw.decode()))
print(json.dumps(out))
"""
r=subprocess.run(['docker','exec',name,'python3','-I','-S','-c',inner,json.dumps(spec)],capture_output=True,text=True,timeout=35);assert r.returncode==0,r.stderr
print(json.dumps(dict(container_id=i['Id'],image=i['Image'],sources=json.loads(r.stdout),new_NPU_work=False,remote_writes=[])))
'''

def main():
    flow=json.loads((BASE/'full-review/flow-index.json').read_bytes())
    indexed={r['runtime_path']:Path(r['local_path']) for r in flow['files']}
    prior=json.loads((OUT/'sampler-execution-identity.json').read_bytes())
    indexed.update({r['runtime_path']:Path(r['local_path']) for r in prior['source_pins']})
    existing=OUT/'metadata-readers-execution-identity.json'
    if existing.is_file():
        indexed.update({r['runtime_path']:Path(r['local_path']) for r in json.loads(existing.read_bytes())['sources']})
    specs=[]
    for name in NAMES:
        runtime='/vllm-workspace/'+name
        choices=[indexed.get(runtime),FROZEN/name,FROZEN.parent/'sampler-source-supplement/frozen-source-snapshot'/name]
        p=next((p for p in choices if p and p.is_file()),None)
        specs.append(dict(runtime_path=runtime,local_path=str(p) if p else None,local_sha256=hashlib.sha256(p.read_bytes()).hexdigest() if p else None))
    r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-c',REMOTE,json.dumps(specs)],capture_output=True,text=True,timeout=55)
    assert r.returncode==0,r.stderr
    d=json.loads(r.stdout);retrieved=[]
    for row in d['sources']:
        content=row.pop('content')
        if content is not None:
            p=OUT/'metadata-readers-runtime-source'/Path(row['runtime_path']).relative_to('/vllm-workspace');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(content)
            row['previous_local_path']=row['local_path'];row['local_path']=str(p)
            assert hashlib.sha256(p.read_bytes()).hexdigest()==row['runtime_sha256']
            retrieved.append(row['runtime_path'])
    d['reader_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT/'metadata-readers-execution-identity.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(source_count=len(d['sources']),matching_previous=sum(r['matches_local'] for r in d['sources']),retrieved=retrieved,new_NPU_work=False,remote_writes=[])))

if __name__=='__main__':main()
