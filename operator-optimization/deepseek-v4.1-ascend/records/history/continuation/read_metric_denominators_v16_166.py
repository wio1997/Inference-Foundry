from pathlib import Path
from datetime import datetime,timezone
import subprocess,json,hashlib,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-communications-v15')
OUT=P/'metric-denominators-and-source-pins-v16.json';assert not OUT.exists()
code="""from pathlib import Path
import ast,json,hashlib,re
base=Path('/usr/local/Ascend/cann-9.2.0/tools/profiler/profiler_tool/analysis')
rows=[];hits=[]
def extract(p,names=None,class_name=None,points=None):
 s=p.read_text();ls=s.splitlines();tree=ast.parse(s);nodes=[]
 for n in ast.walk(tree):
  if not isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)):continue
  if names and n.name not in names:continue
  if class_name and not any(isinstance(c,ast.ClassDef)and c.name==class_name and c.lineno<=n.lineno<=c.end_lineno for c in ast.walk(tree)):continue
  if points and not any(n.lineno<=x<=n.end_lineno for x in points):continue
  nodes.append(dict(name=n.name,start=n.lineno,end=n.end_lineno,source=[str(i+1)+': '+ls[i]for i in range(n.lineno-1,n.end_lineno)]))
 return dict(path=str(p),sha256=hashlib.sha256(s.encode()).hexdigest(),functions=nodes)
ffts=base/'mscalculate/stars/ffts_pmu_calculator.py'
rows.append(extract(ffts,{'__init__','calc_total_time_by_cycle'},'FftsPmuCalculator'))
for p in sorted(base.rglob('*.py')):
 s=p.read_text();ls=s.splitlines();pts=set();funcnames=set()
 for i,l in enumerate(ls):
  if (('FftsPmuCalculator'in l or 'Chip6PmuCalculator'in l)and p.name not in ['ffts_pmu_calculator.py','chip6_pmu_calculator.py'])or('add_cube_usage('in l and 'def add_cube_usage'not in l)or re.search(r'\\bNS_TO_US\\s*=',l):
   pts.update(range(max(0,i-5),min(len(ls),i+6)))
  if 'def cal_total_time('in l:funcnames.add('cal_total_time')
  if 'def get_data_under_device('in l:funcnames.add('get_data_under_device')
 if pts:hits.append(dict(path=str(p),sha256=hashlib.sha256(s.encode()).hexdigest(),lines=[str(i+1)+': '+ls[i]for i in sorted(pts)]))
 if funcnames:rows.append(extract(p,funcnames))
modelpins=[]
for rel in ['vllm_ascend/worker/model_runner_v1.py','vllm_ascend/models/deepseek_v41/model.py','vllm_ascend/models/deepseek_v41/vl_model.py','vllm_ascend/compilation/acl_graph.py','vllm_ascend/attention/dsa_v1.py','vllm_ascend/attention/dsa_v41.py','vllm_ascend/ops/vocab_parallel_embedding.py']:
 p=Path('/vllm-workspace/vllm-ascend')/rel;modelpins.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
p=Path('/vllm-workspace/vllm/vllm/v1/worker/gpu_model_runner.py');modelpins.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
print(json.dumps(dict(functions=rows,exact_symbol_context_hits=hits,model_source_hashes_only=modelpins)))
"""
r=subprocess.run(['docker','exec','wio-vllm-v41-repro166-stage-20261030','python3','-c',code],capture_output=True,text=True,timeout=50);assert r.returncode==0,r.stderr
result=json.loads(r.stdout);payload=(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(),source=result,no_model_import_or_device_work=True,overall_complete=False),ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(payload)
print(json.dumps(dict(result=result,full_result=str(OUT),result_sha256=hashlib.sha256(payload).hexdigest()),ensure_ascii=False))
