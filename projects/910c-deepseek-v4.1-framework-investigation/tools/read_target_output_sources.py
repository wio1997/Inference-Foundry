#!/usr/bin/env python3
"""Read only the next actual sampling/output callees; never import a framework."""
from pathlib import Path
from hashlib import sha256
import json, subprocess
from read_swa_query_producer_entry import REMOTE

R = Path('audit/execution-search')
NAMES = [
    'vllm/vllm/v1/sample/sampler.py',
    'vllm/vllm/v1/sample/rejection_sampler.py',
    'vllm-ascend/vllm_ascend/sample/sampler.py',
    'vllm-ascend/vllm_ascend/sample/rejection_sampler.py',
    'vllm/vllm/v1/worker/gpu_model_runner.py',
    'vllm-ascend/vllm_ascend/worker/model_runner_v1.py',
    'vllm/vllm/v1/sample/ops/topk_topp_sampler.py',
    'vllm/vllm/v1/sample/logits_processor/builtin.py',
    'vllm/vllm/v1/sample/logits_processor/__init__.py',
    'vllm/vllm/v1/sample/ops/bad_words.py',
    'vllm/vllm/v1/sample/ops/penalties.py',
    'vllm-ascend/vllm_ascend/sample/penalties.py',
    'vllm/vllm/v1/sample/metadata.py',
    'vllm/vllm/v1/worker/gpu_input_batch.py',
    'vllm-ascend/vllm_ascend/worker/npu_input_batch.py',
    'vllm/vllm/v1/engine/core.py',
    'vllm/vllm/v1/core/sched/scheduler.py',
    'vllm/vllm/v1/core/sched/async_scheduler.py',
    'vllm/vllm/v1/outputs.py',
]

def main():
    previous = json.loads((R/'draft-output-dependencies.json').read_text())
    old = {s['runtime_path']: s for s in previous['sources']}
    frozen = Path(old['/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py']['local_path']).parents[3]
    specs = []
    for name in NAMES:
        runtime = '/vllm-workspace/'+name
        p = Path(old[runtime]['local_path']) if runtime in old else frozen/name
        if not p.is_file():
            found = list(R.glob('*/vllm-workspace/'+name))
            if found: p = found[0]
        specs.append(dict(runtime_path=runtime,local_path=str(p) if p.is_file() else None,
                          local_sha256=sha256(p.read_bytes()).hexdigest() if p.is_file() else None))
    result = subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE,json.dumps(specs)],capture_output=True,text=True,timeout=150)
    if result.returncode: raise RuntimeError(result.stderr)
    d = json.loads(result.stdout)
    for s in d['sources']:
        body=s.pop('content',None)
        if body is not None:
            p=R/'target-output-source'/s['runtime_path'].lstrip('/');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(body)
            assert sha256(p.read_bytes()).hexdigest()==s['runtime_sha256'];s['local_path']=str(p)
    d.update(reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),qualification='Disk identity only; reviewed methods, recursive callsites, native boundaries and unknown runtime inputs are recorded separately.',whole_chain_complete=False)
    (R/'target-output-dependencies.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(sources=len(d['sources']),matches=sum(x['matches_local'] for x in d['sources']),new_bodies=sum(x['local_sha256'] is None for x in d['sources']),remote_writes=[])))

if __name__=='__main__': main()
