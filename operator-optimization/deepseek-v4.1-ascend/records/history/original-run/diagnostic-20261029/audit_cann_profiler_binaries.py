#!/usr/bin/env python3
"""Fingerprint bounded installed profiling binaries and search exact static byte strings."""
import base64,json,pathlib,shlex,subprocess
remote=r'''import hashlib,json,pathlib,re
base=pathlib.Path('/usr/local/Ascend/cann-9.2.0');roots=[base/'tools/profiler',base/'lib64',base/'aarch64-linux/lib64'];pats=[b'additional.tensor_info',b'aging.additional',b'unaging.additional',b'tensor_info',b'TensorInfo',b'storage_offset',b'inputStrides',b'input_strides'];out=[];skipped=[];seen=set();budget=400_000_000;consumed=0
for root in roots:
 if not root.is_dir():continue
 for p in sorted(root.rglob('*')):
  if not p.is_file() or p.is_symlink():continue
  if root.name!='profiler' and not re.search(r'(prof|profil|analysis)',p.name,re.I):continue
  try:
   st=p.stat()
   if not (p.suffix in ('.so','.bin') or '.so.' in p.name or st.st_mode&0o111):continue
   if p in seen:continue
   seen.add(p)
   if st.st_size>80_000_000 or consumed+st.st_size>budget:skipped.append({'path':str(p),'bytes':st.st_size});continue
   b=p.read_bytes();consumed+=len(b);matched=[x.decode() for x in pats if x in b]
   out.append({'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'matched_literals':matched})
  except OSError:skipped.append({'path':str(p),'error':'OSError'})
print(json.dumps({'roots':[str(x) for x in roots if x.is_dir()],'bytes_scanned':consumed,'scanned':out,'skipped':skipped,'scope':'selected installed profiler executables/shared libraries only; exact literal presence, no tensor files or execution'}))'''
e=base64.b64encode(remote.encode()).decode();cmd='docker exec wio-vllm-v41-repro169-stage-20261028 nice -n 15 python3 -c '+shlex.quote("import base64;exec(base64.b64decode('"+e+"'))")
r=subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh',cmd],capture_output=True,text=True,timeout=120,check=True);v=json.loads(r.stdout);o=pathlib.Path(__file__).parent/'cann-profiler-binaries-audit.json';o.write_text(json.dumps(v,indent=2)+'\n');print('binaries',len(v['scanned']),'skipped',len(v['skipped']),'with_matches',sum(bool(x['matched_literals']) for x in v['scanned']))
