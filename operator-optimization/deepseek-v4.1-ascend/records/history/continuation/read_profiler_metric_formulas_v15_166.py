from pathlib import Path
from datetime import datetime,timezone
import sqlite3,subprocess,json,hashlib,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis')
b=(P/'native-execution-identity-v14/index.json').read_bytes();assert hashlib.sha256(b).hexdigest()=='a3aa47cdf9170785432ecbc18117a489f3fba37d89959f110f4fcbd8890c7dc4';idx=json.loads(b);d=json.loads(Path(idx['databases'][0]['path']).read_text())
db=sqlite3.connect(Path(d['database_path']).as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row;db.execute('PRAGMA query_only=ON')
device_info=dict(NPU_INFO_schema=[dict(r)for r in db.execute('PRAGMA table_info(NPU_INFO)')],NPU_INFO=[dict(r)for r in db.execute('SELECT * FROM NPU_INFO')]);db.close()
code="""from pathlib import Path
import ast,json,hashlib,re
base=Path('/usr/local/Ascend/cann-9.2.0/tools/profiler/profiler_tool/analysis')
specs=[(base/'common_func/data_manager.py',set(),{72,77}), (base/'viewer/runtime_report.py',set(),{275}), (base/'mscalculate/stars/ffts_pmu_calculator.py',{'calculate_total_time'},set()), (base/'mscalculate/stars/chip6_pmu_calculator.py',{'_calculate_total_time','_calculate_total_time_for_mix'},set())]
rows=[]
for p,names,points in specs:
 s=p.read_text();ls=s.splitlines();tree=ast.parse(s);fs=[]
 for n in ast.walk(tree):
  if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))and(n.name in names or any(n.lineno<=i<=n.end_lineno for i in points)):
   fs.append(dict(name=n.name,start=n.lineno,end=n.end_lineno,source=[str(i+1)+': '+ls[i]for i in range(n.lineno-1,n.end_lineno)]))
 rows.append(dict(path=str(p),sha256=hashlib.sha256(s.encode()).hexdigest(),functions=fs,class_and_function_signatures=[dict(name=n.name,line=n.lineno,kind=type(n).__name__)for n in ast.walk(tree)if isinstance(n,(ast.ClassDef,ast.FunctionDef))]))
selector=[]
for p in (base/'mscalculate').rglob('*.py'):
 s=p.read_text()
 if ('FftsPmuCalculator'in s or 'Chip6PmuCalculator'in s)and p.name not in ['ffts_pmu_calculator.py','chip6_pmu_calculator.py']:
  ls=s.splitlines();points=set()
  for i,l in enumerate(ls):
   if 'FftsPmuCalculator'in l or 'Chip6PmuCalculator'in l:points.update(range(max(0,i-6),min(len(ls),i+7)))
  selector.append(dict(path=str(p),sha256=hashlib.sha256(s.encode()).hexdigest(),lines=[str(i+1)+': '+ls[i]for i in sorted(points)]))
print(json.dumps(dict(formula_files=rows,calculator_selector_hits=selector)))
"""
r=subprocess.run(['docker','exec','wio-vllm-v41-repro166-stage-20261030','python3','-c',code],capture_output=True,text=True,timeout=35);assert r.returncode==0,r.stderr;result=json.loads(r.stdout)
OUT=P/'native-communications-v15/profiler-metric-formulas-and-chip-v15.json';assert not OUT.exists();b=(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(),device_info=device_info,source=result,no_model_source_import_or_device_work=True),ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(b)
public=dict(device_info=device_info,result=result,full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest())
if len(json.dumps(public).encode())>31000:
 for f in public['result']['formula_files']:f.pop('class_and_function_signatures',None)
 public['display_signatures_omitted']=True
print(json.dumps(public,ensure_ascii=False))
