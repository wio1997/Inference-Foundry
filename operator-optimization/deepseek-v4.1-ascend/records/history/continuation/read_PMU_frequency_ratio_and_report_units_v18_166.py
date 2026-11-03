from pathlib import Path
from datetime import datetime,timezone
import subprocess,json,hashlib,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-communications-v15')
OUT=P/'PMU-frequency-ratio-and-report-unit-sources-v18.json';assert not OUT.exists()
code="""from pathlib import Path
import ast,json,hashlib
B=Path('/usr/local/Ascend/cann-9.2.0/tools/profiler/profiler_tool/analysis');rows=[]
specs=[('common_func/info_conf_reader.py',{'get_freq','get_root_data'},set()),('mscalculate/factory/calculator_factory.py',{'run'},set()),('mscalculate/calculate_ai_core_data.py',{'_cal_pmu_metrics'},set()),('viewer/ai_core_op_report.py',set(),{475})]
for rel,names,points in specs:
 p=B/rel;s=p.read_text();ls=s.splitlines();tree=ast.parse(s);fs=[]
 for n in ast.walk(tree):
  if isinstance(n,ast.FunctionDef)and(n.name in names or any(n.lineno<=q<=n.end_lineno for q in points)):
   fs.append(dict(name=n.name,start=n.lineno,end=n.end_lineno,source=[str(i+1)+': '+ls[i]for i in range(n.lineno-1,n.end_lineno)]))
 rows.append(dict(path=str(p),sha256=hashlib.sha256(s.encode()).hexdigest(),functions=fs,function_signatures=[dict(name=n.name,start=n.lineno,end=n.end_lineno)for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)]))
constant_hits=[]
for p in (B/'common_func').rglob('*.py'):
 s=p.read_text();ls=s.splitlines();points=set()
 for i,l in enumerate(ls):
  if any(v in l for v in ['CHIP_V2_1_0 =','CHIP_V4_1_0 =','PLATFORM_VERSION =','MHZ_TO_HZ =']):points.update(range(max(0,i-2),min(len(ls),i+3)))
 if points:constant_hits.append(dict(path=str(p),sha256=hashlib.sha256(s.encode()).hexdigest(),lines=[str(i+1)+': '+ls[i]for i in sorted(points)]))
print(json.dumps(dict(files=rows,constant_hits=constant_hits)))
"""
r=subprocess.run(['docker','exec','wio-vllm-v41-repro166-stage-20261030','python3','-c',code],capture_output=True,text=True,timeout=35);assert r.returncode==0,r.stderr
result=json.loads(r.stdout);payload=(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(),source=result,no_model_source_import_or_device_work=True,overall_complete=False),ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(payload)
print(json.dumps(dict(result=result,full_result=str(OUT),result_sha256=hashlib.sha256(payload).hexdigest()),ensure_ascii=False))
