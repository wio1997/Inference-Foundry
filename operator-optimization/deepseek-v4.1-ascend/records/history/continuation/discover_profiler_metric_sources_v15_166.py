from pathlib import Path
from datetime import datetime,timezone
import subprocess,json,hashlib,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
code="""from pathlib import Path
import sys,sysconfig,shutil,subprocess,json,hashlib,re
roots=[]
for base in list(sys.path)+[sysconfig.get_paths()['purelib']]:
 if not base:continue
 for suffix in ['torch_npu/profiler/analysis','msprof_analyze','msparser']:
  p=Path(base)/suffix
  if p.is_dir():roots.append(p.resolve())
exe=shutil.which('msprof')
if exe:
 p=Path(exe).resolve().parent.parent
 for suffix in ['profiler_tool/analysis','python','analysis']:
  q=p/suffix
  if q.is_dir():roots.append(q.resolve())
for base in ['/usr/local/Ascend/ascend-toolkit/latest/tools/profiler/profiler_tool/analysis','/usr/local/Ascend/cann/tools/profiler/profiler_tool/analysis']:
 p=Path(base)
 if p.is_dir():roots.append(p.resolve())
roots=list(dict.fromkeys(roots));pattern=r'aic_total_cycles|aic_total_time|aic_mac_ratio|aiv_total_cycles|CONNECTION_IDS'
rg=shutil.which('rg');hits=[];scanned=0
if rg and roots:
 r=subprocess.run([rg,'-l','--glob','*.py',pattern]+[str(p)for p in roots],capture_output=True,text=True,timeout=30)
 assert r.returncode in [0,1],r.stderr
 paths=[Path(line)for line in r.stdout.splitlines()]
else:
 paths=[]
 for root in roots:
  for p in root.rglob('*.py'):
   scanned+=1;assert scanned<4000
   if p.stat().st_size<=1024**2 and re.search(pattern,p.read_text(errors='replace')):paths.append(p)
for p in dict.fromkeys(paths):
 s=p.read_text();ls=s.splitlines();numbers=set()
 for i,line in enumerate(ls):
  if re.search(pattern,line):numbers.update(range(max(0,i-2),min(len(ls),i+3)))
 hits.append(dict(path=str(p),sha256=hashlib.sha256(s.encode()).hexdigest(),matched_context_lines=[str(i+1)+': '+ls[i]for i in sorted(numbers)]))
print(json.dumps(dict(roots=[str(p)for p in roots],msprof_executable=exe,search_method='rg'if rg else'bounded_literal_python_scan',scanned_files_if_fallback=scanned,hits=hits)))
"""
r=subprocess.run(['docker','exec','wio-vllm-v41-repro166-stage-20261030','python3','-c',code],capture_output=True,text=True,timeout=45);assert r.returncode==0,r.stderr;result=json.loads(r.stdout)
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-communications-v15')
OUT=P/'profiler-metric-source-index-v15.json';assert not OUT.exists();b=(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(),result=result,no_target_model_source_import_or_device_work=True),ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(b)
public=dict(result=result,full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest())
if len(json.dumps(public).encode())>30000:public['result']={k:v for k,v in result.items()if k!='hits'};public['result']['hit_files']=[dict(path=x['path'],sha256=x['sha256'])for x in result['hits']];public['display_truncated']=True
print(json.dumps(public,ensure_ascii=False))
