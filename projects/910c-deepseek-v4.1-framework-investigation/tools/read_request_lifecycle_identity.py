#!/usr/bin/env python3
"""Pin actual request/stop/abort/error consumers without importing vLLM."""
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from read_prefix_retirement_identity import FROZEN, OUT, REMOTE

NAMES = [
    'vllm/vllm/v1/engine/input_processor.py',
    'vllm/vllm/v1/engine/async_llm.py',
    'vllm/vllm/v1/engine/core_client.py',
    'vllm/vllm/v1/engine/output_processor.py',
    'vllm/vllm/v1/engine/core.py',
    'vllm/vllm/v1/engine/parallel_sampling.py',
    'vllm/vllm/v1/request.py',
    'vllm/vllm/sampling_params.py',
    'vllm/vllm/v1/core/sched/scheduler.py',
    'vllm/vllm/v1/core/sched/utils.py',
    'vllm/vllm/v1/core/sched/async_scheduler.py',
    'vllm/vllm/v1/worker/gpu_model_runner.py',
    'vllm/vllm/v1/worker/gpu_input_batch.py',
    'vllm-ascend/vllm_ascend/worker/model_runner_v1.py',
    'vllm-ascend/vllm_ascend/spec_decode/dspark_proposer.py',
    'vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py',
    'vllm/vllm/v1/outputs.py',
    'vllm/vllm/envs.py',
]


def main():
    previous=[]
    for p in OUT.glob('*identity.json'):
        try:
            x=json.loads(p.read_text())
        except (ValueError,OSError):
            continue
        previous.extend(x.get('sources',[]))
    spec=[]
    for name in NAMES:
        rp='/vllm-workspace/'+name
        known=next((r['local_path'] for r in previous if r.get('runtime_path')==rp
                    and r.get('local_path') and Path(r['local_path']).is_file()),None)
        p=Path(known) if known else FROZEN/name
        spec.append(dict(runtime_path=rp,local_path=str(p),
                         local_sha256=sha256(p.read_bytes()).hexdigest() if p.is_file() else None))
    r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-I','-S','-c',REMOTE,json.dumps(spec)],
                     capture_output=True,text=True,timeout=100)
    if r.returncode:raise RuntimeError(r.stderr)
    x=json.loads(r.stdout)
    for q in x['sources']:
        content=q.pop('content',None)
        if content is None:continue
        p=OUT/'request-lifecycle-runtime-source'/q['runtime_path'].removeprefix('/vllm-workspace/')
        p.parent.mkdir(parents=True,exist_ok=True);p.write_text(content)
        assert sha256(p.read_bytes()).hexdigest()==q['runtime_sha256']
        q['local_path']=str(p)
    x['reader_sha256']=sha256(Path(__file__).read_bytes()).hexdigest()
    x['imported_reader_sha256']=sha256(Path('tools/read_prefix_retirement_identity.py').read_bytes()).hexdigest()
    (OUT/'request-lifecycle-execution-identity.json').write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(sources=len(x['sources']),matched=sum(r.get('matches_local',False) for r in x['sources']),
                          missing=[r['runtime_path'] for r in x['sources'] if r.get('missing')],
                          new_NPU_work=0,remote_writes=[])))


if __name__=='__main__':main()
