#!/usr/bin/env python3
"""Current disk pins for shared-frame inputs, mutations and target/draft callers."""
from pathlib import Path
from hashlib import sha256
import json,subprocess
from read_swa_query_producer_entry import REMOTE
ROOT=Path('audit/execution-search')
def main():
    prior=[]
    for name in ['common-frame-lifetime-dependencies.json','metadata-readers-execution-identity.json','common-frame-measured-entry.json']:
        prior+=json.loads((ROOT/name).read_text())['sources']
    base='/vllm-workspace/vllm-ascend/vllm_ascend/'
    names=[base+n for n in ['worker/model_runner_v1.py','attention/context_parallel/dsa_v41_cp.py','attention/context_parallel/dsa_cp.py','attention/dsa_v41.py','worker/device_metadata.py','attention/utils.py','spec_decode/dspark_proposer.py','spec_decode/llm_base_proposer.py','models/deepseek_v41/dspark.py']]
    names+=['/vllm-workspace/vllm/vllm/v1/attention/backends/utils.py','/vllm-workspace/vllm/vllm/v1/attention/backend.py','/vllm-workspace/vllm/vllm/distributed/parallel_state.py']
    spec=[]
    for name in names:
        r=next(r for r in reversed(prior) if r['runtime_path']==name);p=Path(r.get('local_path',r.get('path')))
        spec.append(dict(runtime_path=name,local_path=str(p),local_sha256=sha256(p.read_bytes()).hexdigest()))
    r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE,json.dumps(spec)],capture_output=True,text=True,timeout=110)
    if r.returncode:raise RuntimeError(r.stderr)
    d=json.loads(r.stdout)
    for row in d['sources']:
        body=row.pop('content',None)
        if body is not None:
            p=ROOT/'common-frame-key-source'/row['runtime_path'].lstrip('/');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(body);assert sha256(p.read_bytes()).hexdigest()==row['runtime_sha256'];row['local_path']=str(p)
    d.update(reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),scope='Current disk identity; exactbody/key/mutation review separate, no model/framework imports or nativebit admission.')
    (ROOT/'common-frame-key-dependencies.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(sources=len(d['sources']),matches=sum(r.get('matches_local',False) for r in d['sources']),container_state=d['container_state'],new_NPU_work=0)))
if __name__=='__main__':main()
