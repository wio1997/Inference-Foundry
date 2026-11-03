#!/usr/bin/env python3
"""Audit eight specific saved TensorMove tasks; read-only stage CPU; no new NPU work."""
import base64,hashlib,json,pathlib,shlex,subprocess
root=pathlib.Path(__file__).resolve().parents[2]
inp=root/'diagnostic-20261029/target-layer-all-kernels-16rank.jsonl'
expected='0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4'
assert hashlib.sha256(inp.read_bytes()).hexdigest()==expected
selected=[]
for line in inp.open():
 x=json.loads(line)
 if x.get('kind')!='rank':continue
 d=x['data']
 for layer in d['layers']:
  if layer['layer']!=0 or layer['scheduler_digest']!='1b826883b782ba77814d5bd40fcd3e46c759e380069db2e066ff0e9cb4bce89d':continue
  for g in layer['operator_groups']:
   physical=g['physical_shape_dtype_format']
   if g['operator_name']!='aclnnInplaceCopy_TensorMoveAiCore_TensorMove' or physical['Input Shapes']!='"8184,5120"' or physical['Output Shapes']!='"8184,5120"':continue
   assert physical['Input Data Types']==physical['Output Data Types']=='DT_BF16' and physical['Input Formats']==physical['Output Formats']=='ND' and len(g['tasks'])==1
   selected.append(dict(ep=d['ep'],rank=d['rank'],csv_sha=d['input_sha256']['kernel_csv'],task=g['tasks'][0]))
assert sorted(x['ep'] for x in selected)==list(range(8,16))
code=r'''import base64,csv,decimal,hashlib,json,pathlib,sqlite3,sys
selected=json.loads(base64.b64decode(sys.argv[1]));base=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces')
output=[]
for item in selected:
 ep=item['ep'];p=base/item['rank']/'ASCEND_PROFILER_OUTPUT'; f=p/'kernel_details.csv'; db=p/('ascend_pytorch_profiler_%s.db'%ep)
 assert hashlib.sha256(f.read_bytes()).hexdigest()==item['csv_sha']
 key=item['task']['csv']; start=int(decimal.Decimal(key['Start Time(us)'].strip())*1000)
 with f.open(newline='') as h:
  rows=[(i,r) for i,r in enumerate(csv.DictReader(h),2) if r['Model ID']==key['Model ID'] and r['Stream ID']==key['Stream ID'] and r['Task ID']==key['Task ID'] and r['Start Time(us)'].strip()==key['Start Time(us)'].strip()]
 assert len(rows)==1 and rows[0][0]==item['task']['csv_line']
 line,row=rows[0]
 for k,v in key.items():assert row[k]==v,(ep,k,row[k],v)
 con=sqlite3.connect('file:'+str(db)+'?mode=ro',uri=True)
 candidates=con.execute('select globalTaskId,startNs,endNs from TASK where taskId=? and streamId=? and modelId=?',(int(key['Task ID']),int(key['Stream ID']),int(key['Model ID']))).fetchall()
 exact=[x for x in candidates if x[1]==start];assert len(exact)==1,(ep,len(candidates),len(exact))
 gid,s,e=exact[0];replays=con.execute('select startNs,endNs,modelId,streamId,taskId from TASK where globalTaskId=? order by startNs',(gid,)).fetchall()
 vals=con.execute('select p.timestampNs,s.value,p.value from TASK_PMU_INFO p left join STRING_IDS s on s.id=p.name where p.globalTaskId=? order by p.timestampNs,s.value',(gid,)).fetchall()
 output.append(dict(ep=ep,csv_sha=item['csv_sha'],csv_line=line,csv_row=row,task_identity=dict(globalTaskId=gid,startNs=s,endNs=e,coarse_candidates=len(candidates),exact_candidates=len(exact),gid_task_rows=replays),pmu_rows=vals,trace_flow=item['task']['flow_id']))
print(json.dumps(output))'''
arg=base64.b64encode(json.dumps(selected).encode()).decode()
cmd=shlex.join(['docker','exec','wio-vllm-v41-repro169-stage-20261028','python3','-c',code,arg])
r=subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh',cmd],capture_output=True,text=True,timeout=90,check=True)
rows=json.loads(r.stdout)
out={'input_sha256':expected,'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'selection':'digest 1b826883, layer 0, EP8-15, BF16 [8184,5120] ND TensorMove','rows':rows,'limitations':['Raw database numerals are not proof of hardware measurement resolution, exporter formula, pipeline exclusivity, transferred bytes, bandwidth, wall-clock or causality.','Repeated globalTaskId would require separate verified PMU replay association; time proximity alone is not a validated replay key.','CSV/DB are same historical capture, not independent experiments.']}
path=root/'agents/diagnosis/tensormove-8rank-l0-pmu.json';path.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
for x in rows:
 print('EP',x['ep'],'gid_task_rows',len(x['task_identity']['gid_task_rows']),'pmu',len(x['pmu_rows']),'timestamps',len(set(p[0] for p in x['pmu_rows'])))
print('output_sha256',hashlib.sha256(path.read_bytes()).hexdigest())
