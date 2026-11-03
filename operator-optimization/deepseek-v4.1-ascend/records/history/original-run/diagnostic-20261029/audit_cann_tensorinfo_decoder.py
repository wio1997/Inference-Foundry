#!/usr/bin/env python3
"""Find CANN profiler Python source references, and fingerprint only raw file bytes."""
import base64,json,pathlib,shlex,subprocess
remote=r'''import hashlib,json,pathlib,re
base=pathlib.Path('/usr/local/Ascend/cann-9.2.0');roots=[base/'tools/profiler',base/'tools/msprof'];hits=[];existing=[]
for root in roots:
 if not root.is_dir():continue
 existing.append(str(root))
 for p in sorted(root.rglob('*.py')):
  if p.is_symlink() or p.stat().st_size>600000:continue
  try:b=p.read_bytes();lines=b.decode().splitlines()
  except (UnicodeDecodeError,OSError):continue
  ix=[i for i,l in enumerate(lines) if re.search(r'additional\.tensor_info|aging\.additional|unaging\.additional|tensor_info\.slice',l,re.I)]
  if ix:hits.append({'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'references':[{'line':i+1,'text':lines[i][:250]} for i in ix[:20]]})
r=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/binary-binding-draft/second-binding-run/offline-ep8-20261029/dp1_pp0_tp0_dcp0_ep8_rank0_746_20261029050352206_ascend_pt');host=list(r.glob('PROF_*/host/data'));assert len(host)==1
files=[]
for p in sorted(host[0].glob('*additional.tensor_info.slice_*')):
 if p.name.endswith('.done') or not p.is_file() or p.is_symlink():continue
 before=p.stat();data=p.read_bytes();after=p.stat();
 files.append({'path':str(p.relative_to(r)),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'stable':(before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'first_512_contains_nul':b'\x00' in data[:512]})
print(json.dumps({'searched_python_roots':existing,'python_decoder_name_matches':hits,'raw_files':files,'scope':'source references and raw file fingerprints only; no raw fields or tensor values'}))'''
e=base64.b64encode(remote.encode()).decode();cmd='docker exec wio-vllm-v41-repro169-stage-20261028 nice -n 15 python3 -c '+shlex.quote("import base64;exec(base64.b64decode('"+e+"'))")
r=subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh',cmd],capture_output=True,text=True,timeout=120,check=True);v=json.loads(r.stdout);o=pathlib.Path(__file__).parent/'cann-tensorinfo-decoder-audit.json';o.write_text(json.dumps(v,indent=2)+'\n');print('roots',len(v['searched_python_roots']),'python matches',len(v['python_decoder_name_matches']),'raw slices',len(v['raw_files']))
