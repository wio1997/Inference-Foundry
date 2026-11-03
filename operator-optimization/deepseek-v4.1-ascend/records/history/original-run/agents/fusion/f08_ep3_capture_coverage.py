#!/usr/bin/env python3
"""Read-only coverage audit of the saved EP3 capture; no model input or NPU access."""
import json, pathlib, sqlite3, hashlib
p=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces/dp0_pp0_tp3_dcp0_ep3_rank3_2128_20261029023525650_ascend_pt')
prof=p/'PROF_000001_20261029023525664_00002128QFKFLBMH'
info=p/'profiler_info_3.json'; sample=prof/'host/sample.json'
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
i=json.loads(info.read_text()); s=json.loads(sample.read_text())
keys=['host_sys','host_cpu_profiling','host_osrt_profiling','hostAllPidCpuProfiling','sys_profiling','taskTrace','hostProfiling','prof_level']
dbs=[]
for f in [p/'ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_3.db',p/'ASCEND_PROFILER_OUTPUT/analysis.db',prof/'analyze/communication_analyzer.db']:
 c=sqlite3.connect(f.as_uri()+'?mode=ro',uri=True)
 dbs.append({'file':str(f.relative_to(p)), 'tables':sorted(r[0] for r in c.execute("select name from sqlite_master where type='table'"))})
warnings=[]
for directory in [p/'logs',prof/'mindstudio_profiler_log']:
 for f in sorted(directory.glob('*.log')):
  for line in f.read_text(errors='replace').splitlines():
   if any(s in line for s in ['Failed to get the step range','Kernel view failed to get the step range','Unable to get freq data','Failed to connect database, please check database of step_trace']):
    warnings.append({'file':str(f.relative_to(p)),'line':line})
out={'versions':{k:i.get(k) for k in ['torch_npu_version','cann_version','rank_id']},'common_config':i['config']['common_config'],'experimental_config':i['config']['experimental_config'],'host_sample_selected':{k:s.get(k) for k in keys},'sha256':{'profiler_info_3.json':digest(info),'host/sample.json':digest(sample)},'databases':dbs,'selected_warnings':warnings,'limits':'No OS runtime or host CPU analysis enabled in the recorded config. Scope is listed config, table names, and selected warnings. No claim that all raw files or every warning were exhaustively decoded; no thread run/wait attribution or data-loss conclusion.'}
print(json.dumps(out,ensure_ascii=False,indent=2))
