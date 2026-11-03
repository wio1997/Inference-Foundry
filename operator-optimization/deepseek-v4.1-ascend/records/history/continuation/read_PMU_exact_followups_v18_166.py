from pathlib import Path
from datetime import datetime,timezone
import subprocess,json,hashlib,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-communications-v15')
OUT=P/'PMU-exact-denominator-and-export-followups-v18.json';assert not OUT.exists()
code="""from pathlib import Path
import ast,json,hashlib
B=Path('/usr/local/Ascend/cann-9.2.0/tools/profiler/profiler_tool/analysis');T=Path('/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/profiler/analysis')
rows=[];files={p:p.read_text()for root in [B,T]for p in root.rglob('*.py')}
def extract(p,names,points=None):
 s=files[p];ls=s.splitlines();fs=[]
 for n in ast.walk(ast.parse(s)):
  if isinstance(n,ast.FunctionDef)and(n.name in names or points and any(n.lineno<=v<=n.end_lineno for v in points)):
   fs.append(dict(name=n.name,start=n.lineno,end=n.end_lineno,source=[str(i+1)+': '+ls[i]for i in range(n.lineno-1,n.end_lineno)]))
 return dict(path=str(p),sha256=hashlib.sha256(s.encode()).hexdigest(),functions=fs)
specs=[('common_func/platform/chip_manager.py',{'load_chip_info'}),('mscalculate/aic/pmu_calculator.py',{'init_params','get_config_params','_get_current_block'}),('mscalculate/stars/ffts_pmu_calculator.py',{'add_block_mix_pmu_to_context_pmu'})]
for rel,names in specs:
 p=B/rel;r=extract(p,names)
 if rel.endswith('chip_manager.py'):r['class_mapping_lines']=[str(i+1)+': '+files[p].splitlines()[i]for i in range(26,61)]
 rows.append(r)
for p,s in files.items():
 if 'class CalculateAiCoreData'in s:rows.append(extract(p,{'compute_ai_core_data','add_pipe_time'}))
hits=[]
for p,s in files.items():
 ls=s.splitlines();points=set();matching=[]
 for i,l in enumerate(ls):
  if 'TASK_PMU_INFO'in l or ('get_parsers('in l and 'def get_parsers'not in l)or('DataCalculatorConfig'in l and p.name!='data_calculator_config.py'):
   points.update(range(max(0,i-5),min(len(ls),i+6)));matching.append(i+1)
 if points:hits.append(dict(path=str(p),sha256=hashlib.sha256(s.encode()).hexdigest(),exact_contexts=[str(i+1)+': '+ls[i]for i in sorted(points)]))
print(json.dumps(dict(files=rows,exact_symbol_hits=hits)))
"""
r=subprocess.run(['docker','exec','wio-vllm-v41-repro166-stage-20261030','python3','-c',code],capture_output=True,text=True,timeout=45);assert r.returncode==0,r.stderr
result=json.loads(r.stdout);payload=(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(),source=result,no_model_source_import_or_device_work=True,overall_complete=False),ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(payload)
print(json.dumps(dict(result=result,full_result=str(OUT),result_sha256=hashlib.sha256(payload).hexdigest()),ensure_ascii=False))
