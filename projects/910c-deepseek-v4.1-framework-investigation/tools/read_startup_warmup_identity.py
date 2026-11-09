#!/usr/bin/env python3
"""Read-only pins of actual startup warmup/workspace/MM producer entry."""
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from read_prefix_retirement_identity import FROZEN, OUT, REMOTE

NAMES=[
 "vllm-ascend/vllm_ascend/model_executor/warmup/kernel_warmup.py",
 "vllm/vllm/v1/worker/workspace.py",
 "vllm/vllm/multimodal/encoder_budget.py",
 "vllm-ascend/vllm_ascend/worker/worker.py",
 "vllm-ascend/vllm_ascend/worker/model_runner_v1.py",
 "vllm/vllm/v1/worker/gpu_model_runner.py",
 "vllm/vllm/multimodal/registry.py"
]


def main():
    old=[]
    for p in OUT.glob('*identity.json'):
        try:old.extend(json.loads(p.read_text()).get('sources',[]))
        except (ValueError,OSError):pass
    spec=[]
    for name in NAMES:
        rp='/vllm-workspace/'+name
        found=next((r['local_path'] for r in old if r.get('runtime_path')==rp and r.get('local_path') and Path(r['local_path']).is_file()),None)
        p=Path(found) if found else FROZEN/name
        spec.append(dict(runtime_path=rp,local_path=str(p),local_sha256=sha256(p.read_bytes()).hexdigest() if p.is_file() else None))
    remote=REMOTE.replace("'python3','-I','-S'","'python3','-B','-I','-S'")
    r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',remote,json.dumps(spec)],capture_output=True,text=True,timeout=100)
    if r.returncode:raise RuntimeError(r.stderr)
    d=json.loads(r.stdout)
    for r in d['sources']:
        content=r.pop('content',None)
        if content is None:continue
        p=OUT/'startup-warmup-runtime-source'/r['runtime_path'].removeprefix('/vllm-workspace/')
        p.parent.mkdir(parents=True,exist_ok=True);p.write_text(content)
        assert sha256(p.read_bytes()).hexdigest()==r['runtime_sha256'];r['local_path']=str(p)
    d['reader_sha256']=sha256(Path(__file__).read_bytes()).hexdigest()
    d['imported_reader_sha256']=sha256(Path('tools/read_prefix_retirement_identity.py').read_bytes()).hexdigest()
    (OUT/'startup-warmup-execution-identity.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(sources=len(d['sources']),matched=sum(r.get('matches_local',False) for r in d['sources']),missing=[r['runtime_path'] for r in d['sources'] if r.get('missing')],new_NPU_work=0,remote_writes=[])))


if __name__=='__main__':main()
