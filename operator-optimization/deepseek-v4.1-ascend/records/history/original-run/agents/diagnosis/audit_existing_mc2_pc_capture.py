#!/usr/bin/env python3
"""CPU-only read of existing 16-rank profiler configuration and EP8 DB schema."""
import decimal, hashlib, json, pathlib, shlex, subprocess
root = pathlib.Path(__file__).resolve().parents[2]
data = json.loads((root / 'diagnostic-20261029/target-layer-mc2-16rank.json').read_text())
ranks = [r['rank'] for r in data['ranks']]
if len(ranks) != 16 or len(set(ranks)) != 16:
    raise RuntimeError('expected exactly 16 distinct ranks')
ep8 = next(r['rank'] for r in data['ranks'] if r['ep'] == 8)
source = json.loads((root/'diagnostic-20261029/mc2-extra-pipe-fields-16-tasks.json').read_text())
targets = []
for r in source['rows']:
 if r['scope']['ep'] == 8:
  e = r['scope']
  targets.append({'layer':e['layer'], 'task':int(e['task']), 'stream':int(e['stream']), 'model':int(e['model']), 'start_ns':int(decimal.Decimal(e['start'])*1000)})
if sorted(t['layer'] for t in targets) != [14,39]: raise RuntimeError('expected EP8 layers 14 and 39')
base = '/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces'
code = r'''import pathlib, json, sys, hashlib, sqlite3
base=pathlib.Path(sys.argv[1]); ranks=json.loads(sys.argv[2]); ep8=sys.argv[3]; targets=json.loads(sys.argv[4]); rows=[]
for rank in ranks:
 p=base/rank; files=list(p.glob("PROF_*/device_*/sample.json"))
 if len(files)!=1: raise RuntimeError("sample count "+rank+" "+str(len(files)))
 f=files[0]; j=json.loads(f.read_text()); dat=f.parent/"data"
 kinds=sorted(set(x.name.rsplit(".slice_",1)[0] for x in dat.iterdir() if ".slice_" in x.name and not x.name.endswith(".done")))
 rows.append({"rank":rank,"sample_relative_path":str(f.relative_to(base)),"sample_sha256":hashlib.sha256(f.read_bytes()).hexdigest(),"pcSampling":j.get("pcSampling"),"instrProfiling":j.get("instrProfiling"),"ai_core_profiling_mode":j.get("ai_core_profiling_mode"),"aiv_profiling_mode":j.get("aiv_profiling_mode"),"raw_kind_prefixes":kinds})
db=base/ep8/"ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_8.db"
con=sqlite3.connect("file:"+str(db)+"?mode=ro",uri=True)
schema={}
for name, in con.execute("select name from sqlite_master where type='table' order by name"):
 schema[name]=[x[1] for x in con.execute("pragma table_info("+chr(34)+name+chr(34)+")")]
pmu_targets=[]
for target in targets:
 candidates=con.execute("select globalTaskId,startNs,endNs from TASK where taskId=? and streamId=? and modelId=?", (target['task'],target['stream'],target['model'])).fetchall()
 exact=[x for x in candidates if x[1]==target['start_ns']]
 if len(exact)!=1: raise RuntimeError("exact match count "+str(target))
 gid,start,end=exact[0]
 vals=con.execute("select p.timestampNs,s.value,p.value from TASK_PMU_INFO p left join STRING_IDS s on s.id=p.name where p.globalTaskId=? order by s.value",(gid,)).fetchall()
 pmu_targets.append({"target":target,"coarse_task_candidate_count":len(candidates),"exact_global_task_id":gid,"end_ns":end,"pmu_row_count":len(vals),"pmu_distinct_timestamp_ns":sorted(set(x[0] for x in vals)),"fields":{x[1]:x[2] for x in vals}})
print(json.dumps({"rows":rows,"ep8_db_relative_path":str(db.relative_to(base)),"ep8_db_tables_columns":schema,"ep8_targets":pmu_targets},ensure_ascii=False))
'''
cmd=shlex.join(['docker','exec','wio-vllm-v41-repro169-stage-20261028','python3','-c',code,base,json.dumps(ranks),ep8,json.dumps(targets)])
p=subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh',cmd],check=True,capture_output=True,text=True,timeout=50)
out=json.loads(p.stdout)
out['scope']='existing 16 rank capture; EP8 SQLite schema only; no device operation'
out['input_sha256']=hashlib.sha256((root/'diagnostic-20261029/target-layer-mc2-16rank.json').read_bytes()).hexdigest()
out['target_source_sha256']=hashlib.sha256((root/'diagnostic-20261029/mc2-extra-pipe-fields-16-tasks.json').read_bytes()).hexdigest()
out['script_sha256']=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
t=root/'agents/diagnosis/existing-mc2-pc-capture-audit.json'
t.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
print('rank count',len(out['rows']),'pc_sampling_off',sum(r['pcSampling']=='off' for r in out['rows']),'instr_empty',sum(r['instrProfiling']=='' for r in out['rows']),'EP8 tables',len(out['ep8_db_tables_columns']))
print('saved',t,'sha256',hashlib.sha256(t.read_bytes()).hexdigest())
