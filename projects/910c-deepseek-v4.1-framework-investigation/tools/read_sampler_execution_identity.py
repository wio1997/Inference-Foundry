#!/usr/bin/env python3
"""Pin the actual V1 sampler path and retrieve only missing source, read-only."""
import hashlib
import json
from pathlib import Path
import subprocess

BASE=Path('/Users/wio/work/branch-fusion-audit-20261104')
FROZEN=Path('/Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/whole-scope-20261030/frozen-source-snapshot')
SUPPLEMENT=FROZEN.parent/'sampler-source-supplement/frozen-source-snapshot'
OUT=Path('audit/execution-search')
NAMES=[
 'vllm-ascend/vllm_ascend/sample/sampler.py',
 'vllm-ascend/vllm_ascend/sample/rejection_sampler.py',
 'vllm-ascend/vllm_ascend/sample/penalties.py',
 'vllm-ascend/vllm_ascend/ops/triton/reject_sample.py',
 'vllm-ascend/vllm_ascend/ops/triton/penalty.py',
 'vllm-ascend/vllm_ascend/ops/triton/bincount.py',
 'vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py',
 'vllm-ascend/vllm_ascend/patch/worker/patch_rejection_sampler.py',
 'vllm-ascend/vllm_ascend/worker/model_runner_v1.py',
 'vllm-ascend/vllm_ascend/worker/utils.py',
 'vllm/vllm/v1/sample/sampler.py',
 'vllm/vllm/v1/sample/rejection_sampler.py',
 'vllm/vllm/v1/sample/metadata.py',
 'vllm/vllm/v1/sample/ops/bad_words.py',
 'vllm/vllm/v1/sample/ops/penalties.py',
 'vllm/vllm/v1/sample/ops/topk_topp_sampler.py',
 'vllm/vllm/v1/sample/ops/topk_topp_triton.py',
 'vllm/vllm/v1/sample/ops/logprobs.py',
 'vllm/vllm/v1/sample/logits_processor/builtin.py',
 'vllm/vllm/v1/sample/logits_processor/state.py',
 'vllm/vllm/v1/sample/logits_processor/__init__.py',
 'vllm/vllm/v1/sample/thinking_budget_state.py',
 'vllm/vllm/v1/worker/gpu_input_batch.py',
 'vllm/vllm/v1/worker/gpu_model_runner.py',
 'vllm/vllm/v1/utils.py',
 'vllm/vllm/v1/core/sched/scheduler.py',
 'vllm/vllm/v1/executor/multiproc_executor.py',
 'vllm-ascend/vllm_ascend/patch/platform/patch_multiproc_executor.py',
]
INNER=r'''
import hashlib,json,pathlib,sys
spec=json.loads(sys.argv[1]);out=[]
for row in spec:
 p=pathlib.Path(row['runtime_path']);raw=p.read_bytes();sha=hashlib.sha256(raw).hexdigest()
 out.append(dict(**row,runtime_sha256=sha,matches_local=sha==row.get('local_sha256'),content=None if sha==row.get('local_sha256') else raw.decode()))
print(json.dumps(out))
'''
REMOTE=r'''
import json,subprocess,sys
name='wio-vllm-v41-repro171-20261102';i,=json.loads(subprocess.check_output(['docker','inspect',name]))
assert i['State']['Running'] and i['Image']=='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
r=subprocess.run(['docker','exec',name,'python3','-I','-S','-c',sys.argv[1],sys.argv[2]],capture_output=True,text=True,timeout=35)
assert r.returncode==0,r.stderr
print(json.dumps(dict(container_id=i['Id'],started_at=i['State']['StartedAt'],image=i['Image'],source_pins=json.loads(r.stdout),new_NPU_work=False,service_requests=0,remote_writes=[])))
'''

def main():
    flow=json.loads((BASE/'full-review/flow-index.json').read_text())
    indexed={r['runtime_path']:Path(r['local_path']) for r in flow['files']}
    spec=[]
    for name in NAMES:
        runtime='/vllm-workspace/'+name
        choices=[indexed.get(runtime),SUPPLEMENT/name,FROZEN/name]
        p=next((p for p in choices if p and p.is_file()),None)
        spec.append(dict(runtime_path=runtime,local_path=str(p) if p else None,local_sha256=hashlib.sha256(p.read_bytes()).hexdigest() if p else None))
    r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-c',REMOTE,INNER,json.dumps(spec)],capture_output=True,text=True,timeout=55)
    assert r.returncode==0,r.stderr
    d=json.loads(r.stdout);retrieved=[]
    for row in d['source_pins']:
        content=row.pop('content')
        if content is not None:
            p=OUT/'sampler-runtime-source'/Path(row['runtime_path']).relative_to('/vllm-workspace')
            p.parent.mkdir(parents=True,exist_ok=True);p.write_text(content)
            row['previous_local_path']=row['local_path'];row['local_path']=str(p)
            row['retrieved_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
            assert row['retrieved_sha256']==row['runtime_sha256']
            retrieved.append(row['runtime_path'])
    d['reader_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT/'sampler-execution-identity.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(source_pins=len(d['source_pins']),matching_original_local=sum(r['matches_local'] for r in d['source_pins']),retrieved=retrieved,new_NPU_work=False,remote_writes=[])))

if __name__=='__main__':main()
