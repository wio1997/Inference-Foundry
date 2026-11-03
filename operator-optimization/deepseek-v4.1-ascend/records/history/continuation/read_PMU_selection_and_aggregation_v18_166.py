from pathlib import Path
from datetime import datetime,timezone
import subprocess,json,hashlib,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-communications-v15')
OUT=P/'PMU-selector-frequency-aggregation-sources-v18.json';assert not OUT.exists()
code="""from pathlib import Path
import ast,json,hashlib
B=Path('/usr/local/Ascend/cann-9.2.0/tools/profiler/profiler_tool/analysis')
files={p:p.read_text()for p in B.rglob('*.py')}
def extract(p,names,cls=None):
 s=files[p];ls=s.splitlines();tree=ast.parse(s);nodes=[]
 classes=[c for c in ast.walk(tree)if isinstance(c,ast.ClassDef)and(cls is None or c.name==cls)]
 for n in ast.walk(tree):
  if isinstance(n,ast.FunctionDef)and n.name in names and(cls is None or any(c.lineno<=n.lineno<=c.end_lineno for c in classes)):
   nodes.append(dict(name=n.name,start=n.lineno,end=n.end_lineno,source=[str(i+1)+': '+ls[i]for i in range(n.lineno-1,n.end_lineno)]))
 return dict(path=str(p),sha256=hashlib.sha256(s.encode()).hexdigest(),functions=nodes,class_headers=[dict(name=c.name,line=c.lineno,bases=[ast.unparse(b)for b in c.bases])for c in classes],function_signatures=[dict(name=n.name,start=n.lineno,end=n.end_lineno)for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)])
ffts=B/'mscalculate/stars/ffts_pmu_calculator.py';assert hashlib.sha256(files[ffts].encode()).hexdigest()=='72b65a386260f79b34ab1894e6ba4787616a62584c988272195d4a9a6909f66d'
ff=extract(ffts,{'_get_total_cycle_and_pmu_data','_is_not_mix_main_core','calculate_pmu_list','calculate_mix_pmu_list','_calculate_total_cycle_and_cycle_list'},'FftsPmuCalculator');rows=[ff]
parent_names={b for c in ff['class_headers']for b in c['bases']}
for p,s in files.items():
 for classname in sorted(parent_names|{'ChipManager'}):
  if 'class '+classname+'('in s or 'class '+classname+':'in s:
   rows.append(extract(p,{'__init__','_get_current_freq','get_current_freq','chip_id','is_chip_v4','set_chip_id','get_chip_id'},classname))
for rel in ['common_func/info_conf_reader.py','framework/config_data_parsers.py']:
 p=B/rel;s=files[p];ls=s.splitlines();tree=ast.parse(s);points=set();matches=[]
 for i,l in enumerate(ls):
  if any(v in l for v in ['platform_version','chip_model','aic_frequency','aiv_frequency','ChipManager().chip_id','ChipManager().is_chip_v4']):points.update(range(max(0,i-4),min(len(ls),i+5)));matches.append(i+1)
 names={n.name for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and n.end_lineno-n.lineno<=100 and any(n.lineno<=i<=n.end_lineno for i in matches)}
 r=extract(p,names);r['exact_symbol_contexts']=[str(i+1)+': '+ls[i]for i in sorted(points)];rows.append(r)
print(json.dumps(dict(files=rows)))
"""
r=subprocess.run(['docker','exec','wio-vllm-v41-repro166-stage-20261030','python3','-c',code],capture_output=True,text=True,timeout=45);assert r.returncode==0,r.stderr
result=json.loads(r.stdout);payload=(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(),source=result,no_model_source_import_or_device_work=True,overall_complete=False),ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(payload)
print(json.dumps(dict(result=result,full_result=str(OUT),result_sha256=hashlib.sha256(payload).hexdigest()),ensure_ascii=False))
