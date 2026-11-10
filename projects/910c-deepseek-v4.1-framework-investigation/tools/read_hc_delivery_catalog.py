#!/usr/bin/env python3
"""Inventory actual HC OOT/native sources and callers, stdlib only."""
from pathlib import Path
from hashlib import sha256
import json,subprocess
from read_qkv_native_catalog import REMOTE

INNER=r'''
from pathlib import Path
from hashlib import sha256
import json
tokens=['hc_pre','hc_post','rms_norm_cast','HcPre','HcPost','RmsNormCast']
roots=[Path('/vllm-workspace/vllm-ascend/csrc'),Path('/vllm-workspace/vllm-ascend/vllm_ascend/device'),Path('/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41')]
symbols=[];scanned=0;skipped=[]
for root in roots:
 assert root.is_dir(),str(root)
 for p in root.rglob('*'):
  if not p.is_file() or p.suffix not in ['.py','.cpp','.cc','.h','.hpp']:continue
  if p.stat().st_size>1500000:skipped.append(str(p));continue
  scanned+=1;b=p.read_bytes();hits=[dict(line=i,text=l[:1000]) for i,l in enumerate(b.decode(errors='replace').splitlines(),1) if any(t in l for t in tokens)]
  if hits:symbols.append(dict(path=str(p),sha256=sha256(b).hexdigest(),hits=hits))
names=[]
for root in [Path('/vllm-workspace/vllm-ascend/csrc'),Path('/usr/local/Ascend/cann-9.2.0/opp/built-in/op_impl'),Path('/usr/local/Ascend/cann-9.2.0/opp/vendors')]:
 if not root.is_dir():continue
 for p in root.rglob('*'):
  if p.is_file() and p.suffix in ['.py','.cpp','.h','.hpp','.json'] and any(t.lower() in str(p).lower() for t in tokens):names.append(dict(path=str(p),bytes=p.stat().st_size))
print(json.dumps(dict(roots=list(map(str,roots)),source_files_searched=scanned,skipped_large=skipped,symbol_files=symbols,native_names=names,scope='Filename/symbol navigation only. No body/loadedbinary/bit proof.')))
'''
if __name__=='__main__':
 r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE,INNER],capture_output=True,text=True,timeout=120)
 if r.returncode:raise RuntimeError(r.stderr)
 d=json.loads(r.stdout);d.update(reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[])
 Path('audit/execution-search/hc-delivery-catalog.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(dict(symbol_files=len(d['symbol_files']),names=len(d['native_names']),symbols=[s['path'] for s in d['symbol_files']],native_paths=[s['path'] for s in d['native_names'] if '/op_kernel/' in s['path'] or '/hc_pre/' in s['path']]),ensure_ascii=False))
