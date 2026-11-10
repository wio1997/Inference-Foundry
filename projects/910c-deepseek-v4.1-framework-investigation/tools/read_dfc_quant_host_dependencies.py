#!/usr/bin/env python3
"""Read current DFC host tiling headers without launching a container process."""
from pathlib import Path
from hashlib import sha256
import json, subprocess
from read_swa_query_producer_entry import REMOTE, FROZEN
OUT=Path('audit/execution-search')
def main():
    root='csrc/mc2/dispatch_ffn_combine/op_kernel/moe_init_routing_quant_v2/'
    specs=[]
    for name in ['moe_init_routing_quant_v2_tiling.h','moe_init_routing_v2_tiling.h','tiling_base.h']:
        rel=root+name;p=FROZEN/'vllm-ascend'/rel
        specs.append(dict(runtime_path='/vllm-workspace/vllm-ascend/'+rel,local_path=str(p),local_sha256=sha256(p.read_bytes()).hexdigest()))
    r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE,json.dumps(specs)],capture_output=True,text=True,timeout=110)
    if r.returncode:raise RuntimeError(r.stderr)
    d=json.loads(r.stdout)
    for row in d['sources']:
        body=row.pop('content',None)
        if body is not None:
            p=OUT/'dfc-input-host-source'/row['runtime_path'].lstrip('/');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(body)
            assert sha256(p.read_bytes()).hexdigest()==row['runtime_sha256'];row['local_path']=str(p)
    d.update(reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),scope='Current disk host source only; isolated local CPU DoTiling replay is not active NPU key or numerical proof.')
    (OUT/'dfc-input-host-dependencies.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(sources=len(d['sources']),matches=sum(r.get('matches_local',False) for r in d['sources']),container_state=d['container_state'],new_NPU_work=0)))
if __name__=='__main__':main()
