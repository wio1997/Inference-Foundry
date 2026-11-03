from pathlib import Path
from datetime import datetime,timezone
import subprocess,json,hashlib,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-communications-v15')
OUT=P/'PMU-report-conversion-and-CPP-gate-sources-v18.json';assert not OUT.exists()
code="""from pathlib import Path
import ast,json,hashlib
B=Path('/usr/local/Ascend/cann-9.2.0/tools/profiler/profiler_tool/analysis');rows=[];hits=[]
def extract(p,names):
 s=p.read_text();ls=s.splitlines();fs=[]
 for n in ast.walk(ast.parse(s)):
  if isinstance(n,ast.FunctionDef)and n.name in names:fs.append(dict(name=n.name,start=n.lineno,end=n.end_lineno,source=[str(i+1)+': '+ls[i]for i in range(n.lineno-1,n.end_lineno)]))
 return dict(path=str(p),sha256=hashlib.sha256(s.encode()).hexdigest(),functions=fs)
rows.append(extract(B/'viewer/ai_core_op_report.py',{'get_op_summary_data','_format_summary_data'}))
for p in B.rglob('*.py'):
 s=p.read_text();ls=s.splitlines();points=set()
 for i,l in enumerate(ls):
  if 'FREQ_TO_MHz ='in l or('mac_ratio'in l and('lambda'in l or 'CALCULATION_FORMULA'in l)):points.update(range(max(0,i-3),min(len(ls),i+4)))
 if points:hits.append(dict(path=str(p),sha256=hashlib.sha256(s.encode()).hexdigest(),lines=[str(i+1)+': '+ls[i]for i in sorted(points)]))
 if any('class '+c in s for c in ['DeviceParseScene','CannCalculatorScene','ProfilingScene']):
  r=extract(p,{'is_cpp_enable','is_all_export','__init__','set_cpp_enable','set_cpp_mode'});r['class_and_method_signatures']=[dict(name=n.name,start=n.lineno,end=n.end_lineno)for n in ast.walk(ast.parse(s))if isinstance(n,(ast.FunctionDef,ast.ClassDef))];rows.append(r)
print(json.dumps(dict(files=rows,exact_literal_hits=hits)))
"""
r=subprocess.run(['docker','exec','wio-vllm-v41-repro166-stage-20261030','python3','-c',code],capture_output=True,text=True,timeout=35);assert r.returncode==0,r.stderr
result=json.loads(r.stdout);payload=(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(),source=result,no_model_source_import_or_device_work=True,overall_complete=False),ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(payload)
print(json.dumps(dict(result=result,full_result=str(OUT),result_sha256=hashlib.sha256(payload).hexdigest()),ensure_ascii=False))
