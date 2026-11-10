#!/usr/bin/env python3
"""Pin selected DFC route/quant/dispatch entry from stopped container read-only."""
from pathlib import Path
from hashlib import sha256
import json,subprocess
from read_swa_query_producer_entry import REMOTE,FROZEN
OUT=Path('audit/execution-search')
def main():
 old=json.loads((OUT/'dfc-gemm-tile-source-review.json').read_text())['sources'];base='/vllm-workspace/vllm-ascend/';prefix='csrc/mc2/dispatch_ffn_combine/'
 rel=[prefix+n for n in ['op_kernel/dispatch_ffn_combine.h','op_kernel/dispatch_ffn_combine_kernel.hpp','op_kernel/moe_init_routing_quant_v2/moe_init_routing_quant_v2.cpp','op_kernel/moe_init_routing_quant_v2/moe_v2_fullload_dynamic_quant.h','op_kernel/moe_init_routing_quant_v2/moe_v2_gather_dynamic_quant.h','op_host/dispatch_ffn_combine_tiling.cpp']]
 spec=[]
 for n in rel:
  runtime=base+n;r=next((r for r in old if r['runtime_path']==runtime),None);p=Path(r.get('path',r.get('local_path'))) if r else FROZEN/'vllm-ascend'/n
  spec.append(dict(runtime_path=runtime,local_path=str(p),local_sha256=sha256(p.read_bytes()).hexdigest() if p.is_file() else None))
 r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE,json.dumps(spec)],capture_output=True,text=True,timeout=110)
 if r.returncode:raise RuntimeError(r.stderr)
 d=json.loads(r.stdout)
 for row in d['sources']:
  content=row.pop('content',None)
  if content is not None:
   p=OUT/'dfc-input-entry-source'/row['runtime_path'].lstrip('/');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(content);assert sha256(p.read_bytes()).hexdigest()==row['runtime_sha256'];row['local_path']=str(p)
 d.update(reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),scope='CurrentselectedDFC source entry identities only. Bounded bodyranges/loadedkey/nativebit separate.');(OUT/'dfc-input-delivery-entry.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(sources=len(d['sources']),matches=sum(r.get('matches_local',False) for r in d['sources']),read_method=d['read_method'],container_state=d['container_state'],new_NPU_work=0)))
if __name__=='__main__':main()
