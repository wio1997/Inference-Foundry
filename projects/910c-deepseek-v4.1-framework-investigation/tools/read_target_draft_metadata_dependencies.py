#!/usr/bin/env python3
"""Pin selected target-phase draft metadata and its indirect consumer boundaries."""
from pathlib import Path
from hashlib import sha256
import json,subprocess
from read_swa_query_producer_entry import REMOTE
ROOT=Path('audit/execution-search')
def main():
    previous=json.loads((ROOT/'input-resource-source-review.json').read_text());prior={s['runtime_path']:s for s in previous['sources']};model=next(s for s in previous['sources'] if s['runtime_path'].endswith('/models/deepseek_v41/model.py'));frozen=Path(model['local_path']).parents[4]
    names=['vllm-ascend/vllm_ascend/worker/model_runner_v1.py','vllm-ascend/vllm_ascend/attention/dsa_v41.py','vllm-ascend/vllm_ascend/attention/context_parallel/dsa_v41_cp.py','vllm-ascend/vllm_ascend/attention/context_parallel/dsa_cp.py','vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py','vllm-ascend/vllm_ascend/models/deepseek_v41/model.py','vllm-ascend/vllm_ascend/models/deepseek_v41/dspark.py','vllm-ascend/vllm_ascend/worker/device_metadata.py','vllm-ascend/vllm_ascend/compilation/acl_graph.py','vllm-ascend/vllm_ascend/compilation/breakable_aclgraph.py','vllm-ascend/vllm_ascend/ascend_forward_context.py','vllm-ascend/vllm_ascend/ops/rope_dsv4.py','vllm/vllm/v1/worker/gpu_model_runner.py'];spec=[]
    for name in names:
        runtime='/vllm-workspace/'+name;p=Path(prior[runtime]['local_path']) if runtime in prior else frozen/name;spec.append(dict(runtime_path=runtime,local_path=str(p) if p.exists() else None,local_sha256=sha256(p.read_bytes()).hexdigest() if p.exists() else None))
    r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE,json.dumps(spec)],capture_output=True,text=True,timeout=140)
    if r.returncode:raise RuntimeError(r.stderr)
    d=json.loads(r.stdout)
    for s in d['sources']:
        body=s.pop('content',None)
        if body is not None:
            p=ROOT/'target-draft-metadata-source'/s['runtime_path'].lstrip('/');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(body);assert sha256(p.read_bytes()).hexdigest()==s['runtime_sha256'];s['local_path']=str(p)
    d.update(reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),whole_chain_complete=False,qualification='Current disk identities only. Source ranges, whole delivery/side effects, original profiles and native bit validation are separate; no framework/container process or remote write.')
    (ROOT/'target-draft-metadata-dependencies.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(sources=len(d['sources']),matches=sum(s['matches_local'] for s in d['sources']),new_bodies=sum(s['local_sha256'] is None for s in d['sources']),new_NPU_work=0,remote_writes=[])))
if __name__=='__main__':main()
