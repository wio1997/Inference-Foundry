#!/usr/bin/env python3
"""Read-only native cache-epilogue catalog; never import Torch/TBE/backends."""
from pathlib import Path
from hashlib import sha256
import json,subprocess

INNER=r'''
from pathlib import Path
from hashlib import sha256
import json,re
site=Path('/usr/local/python3.12.13/lib/python3.12/site-packages')
needles=['inplace_partial_rotary_mul','npu_scatter_nd_update_sk','npu_dynamic_quant','npu_rms_norm','AscendRMSNorm']
roots=[Path('/vllm-workspace/vllm-ascend/csrc'),Path('/vllm-workspace/vllm-ascend/vllm_ascend/ops'),site/'torch_npu']
source=[];scanned=0;skipped=[]
for root in roots:
 if not root.is_dir():continue
 for p in root.rglob('*'):
  if not p.is_file() or p.suffix not in ['.py','.h','.cpp','.cc','.hpp','.yaml','.yml']:continue
  if p.stat().st_size>1500000:skipped.append(str(p));continue
  scanned+=1
  text=p.read_text(errors='replace');hits=[dict(line=i,text=l[:500]) for i,l in enumerate(text.splitlines(),1) if any(n in l for n in needles)]
  if hits:source.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha256(p.read_bytes()).hexdigest(),hit_count=len(hits),hits=hits[:24],hits_truncated=len(hits)>24))
name_roots=[Path('/usr/local/Ascend/cann-9.2.0/opp/built-in/op_impl'),Path('/usr/local/Ascend/cann-9.2.0/opp/vendors'),Path('/vllm-workspace/vllm-ascend/csrc')]
inventory=[];total=0
for root in name_roots:
 if not root.is_dir():continue
 for p in root.rglob('*'):
  if not p.is_file() or p.suffix not in ['.py','.cpp','.h','.cc','.hpp','.json']:continue
  if not any(n in str(p).lower() for n in ['rms_norm','rmsnorm','partial_rotary','scatter_nd_update_sk','dynamic_quant']):continue
  total+=1
  if len(inventory)<700:inventory.append(dict(path=str(p),bytes=p.stat().st_size))
print(json.dumps(dict(symbol_source_roots=list(map(str,roots)),source_files_searched=scanned,skipped_large_sources=skipped,symbol_files=source,name_inventory_roots=list(map(str,name_roots)),name_inventory_total=total,name_inventory=inventory,name_inventory_truncated=total>len(inventory),scope='Source symbol hits/file names only; no body or loaded-binary equivalence inferred.')))
'''
REMOTE=r'''
import json,subprocess,sys
name='wio-vllm-v41-repro171-20261102';c,=json.loads(subprocess.check_output(['docker','inspect',name]));assert c['State']['Running'] and c['Image']=='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
r=subprocess.run(['docker','exec',name,'python3','-B','-I','-S','-c',sys.argv[1]],capture_output=True,text=True,timeout=90);assert r.returncode==0,r.stderr;d=json.loads(r.stdout);d.update(container_id=c['Id'],image=c['Image']);print(json.dumps(d))
'''
if __name__=='__main__':
 r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE,INNER],capture_output=True,text=True,timeout=110)
 if r.returncode:raise RuntimeError(r.stderr)
 d=json.loads(r.stdout);d.update(reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),new_NPU_work=0,service_requests=0,model_loads=0,new_profile_collection=0,remote_writes=[]);Path('audit/execution-search/cache-epilogue-catalog.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(source_files=d['source_files_searched'],symbol_files=len(d['symbol_files']),inventory_total=d['name_inventory_total'],inventory_truncated=d['name_inventory_truncated'],symbol_paths=[r['path'] for r in d['symbol_files'] if '/csrc/' in r['path'] or 'register_custom_ops' in r['path']][:25])))
