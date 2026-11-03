from pathlib import Path
from datetime import datetime,timezone
import subprocess,json,hashlib,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
STAGE='wio-vllm-v41-repro166-stage-20261030'
expected='bd6ff19d7328271dc0b82f20414129fca1dc9e8eefa472f259eabcfb3e48f46d'
image='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
def cmd(args,timeout=60):return subprocess.run(args,capture_output=True,text=True,timeout=timeout)
r=cmd(['docker','inspect',STAGE]);assert r.returncode==0;d=json.loads(r.stdout)[0]
assert d['Id']==expected and d['Image']==image and d['State']['Running']
roots=['/vllm-workspace/vllm','/vllm-workspace/vllm-ascend','/data/tiankuan/wio/projects/910c-deepseek-v4.1-flash/instrumentation']
probe=cmd(['docker','exec',STAGE,'rg','--version'],10)
pattern='get_mtp_target_hidden_states|_mtp_hidden_buffer'
if probe.returncode==0:
 r=cmd(['docker','exec',STAGE,'rg','--threads','1','-n','--color','never','--glob','*.py','--glob','*.pyi','--glob','*.cpp','--glob','*.h',pattern,*roots])
 assert r.returncode in [0,1],r.stderr
 raw_matches=r.stdout.splitlines();method='rg_exact_symbols'
 paths=sorted({l.split(':',2)[0] for l in raw_matches})
 code='''import sys,json,hashlib,pathlib
paths=json.loads(sys.argv[1]);roots=json.loads(sys.argv[2]);out=[]
for text in paths:
 p=pathlib.Path(text);assert any(str(p).startswith(r+'/') for r in roots)
 a=p.stat();b=p.read_bytes();z=p.stat();assert (a.st_size,a.st_mtime_ns)==(z.st_size,z.st_mtime_ns)
 out.append(dict(path=text,sha256=hashlib.sha256(b).hexdigest(),bytes=len(b)))
print(json.dumps(out))'''
 rr=cmd(['docker','exec',STAGE,'python3','-c',code,json.dumps(paths),json.dumps(roots)])
 assert rr.returncode==0,rr.stderr;files=json.loads(rr.stdout)
else:
 code='''import sys,json,hashlib,pathlib,re
roots=json.loads(sys.argv[1]);pattern=re.compile('get_mtp_target_hidden_states|_mtp_hidden_buffer');matches=[];files=[]
for root in roots:
 p=pathlib.Path(root)
 if not p.exists():continue
 for f in p.rglob('*'):
  if not f.is_file() or f.suffix not in ['.py','.pyi','.cpp','.h']:continue
  if '.git' in f.parts:continue
  a=f.stat();b=f.read_bytes();z=f.stat();assert (a.st_size,a.st_mtime_ns)==(z.st_size,z.st_mtime_ns)
  text=b.decode('utf-8',errors='replace');hits=[str(f)+':'+str(i)+':'+l for i,l in enumerate(text.splitlines(),1) if pattern.search(l)]
  if hits:matches.extend(hits);files.append(dict(path=str(f),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b)))
print(json.dumps(dict(matches=matches,files=files)))'''
 r=cmd(['docker','exec',STAGE,'python3','-c',code,json.dumps(roots)])
 assert r.returncode==0,r.stderr;x=json.loads(r.stdout);raw_matches=x['matches'];files=x['files'];method='stdlib_exact_symbol_fallback_after_rg_unavailable'
assert len(raw_matches)<=1000 and all(len(s)<=2000 for s in raw_matches)
result=dict(checked_utc=datetime.now(timezone.utc).isoformat(),container=STAGE,container_id=expected,image=image,source_roots=roots,
 search_pattern=pattern,method=method,matches=raw_matches,matched_files=files,target_implementation_unchanged=True,no_NPU_import_inference_or_service_mutation=True,
 limits=['Exact symbol references in declared installed source roots, not proof against arbitrary dynamic reflection or external clients.',
  'Source matching the recorded version defines the stated program contract, not every historical loaded byte or tensor alias value.',
  'No source is executed/imported; stdlib Python is only used to search/hash text.'])
out=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/profile-readonly-20261031/installed-fullhc-symbol-consumers-v11.json')
assert not out.exists();payload=json.dumps(result,ensure_ascii=False,indent=2)+'\n';out.write_text(payload)
print(json.dumps(dict(result=result,full_result=str(out),result_sha256=hashlib.sha256(payload.encode()).hexdigest()),ensure_ascii=False))
