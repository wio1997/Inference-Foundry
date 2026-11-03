#!/usr/bin/env python3
"""Read the historical EP8 L0 TensorMove row and its exact task PMU; CPU only."""
import json,pathlib,shlex,subprocess,hashlib
root=pathlib.Path(__file__).resolve().parents[2]
rank='dp1_pp0_tp0_dcp0_ep8_rank0_737_20261029023525648_ascend_pt'
base='/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces'
code=r'''import csv,decimal,hashlib,json,pathlib,sqlite3,sys
base=pathlib.Path(sys.argv[1])/sys.argv[2]/'ASCEND_PROFILER_OUTPUT'
csvpath=base/'kernel_details.csv'; db=base/'ascend_pytorch_profiler_8.db'
hashcsv=hashlib.sha256(csvpath.read_bytes()).hexdigest()
assert hashcsv=='20cfb97abe03befda3fbf2c1194e48531aac44ce17d4f5fc130ea8ffb5261730'
with csvpath.open(newline='') as f:
 rows=[(i,r) for i,r in enumerate(csv.DictReader(f),2) if r.get('Stream ID')=='47' and r.get('Task ID')=='43441' and r.get('Model ID')=='4294967295' and r.get('Start Time(us)','').strip()=='1793241328465293.999']
assert len(rows)==1, len(rows)
line,row=rows[0]; assert row['Type']=='TensorMove'
con=sqlite3.connect('file:'+str(db)+'?mode=ro',uri=True)
candidates=con.execute('select globalTaskId,startNs,endNs from TASK where taskId=? and streamId=? and modelId=?',(43441,47,4294967295)).fetchall()
exact=[x for x in candidates if x[1]==int(decimal.Decimal(row['Start Time(us)'].strip())*1000)];assert len(exact)==1,(candidates,exact)
gid,start,end=exact[0]; vals=con.execute('select p.timestampNs,s.value,p.value from TASK_PMU_INFO p left join STRING_IDS s on s.id=p.name where p.globalTaskId=? order by s.value',(gid,)).fetchall()
print(json.dumps({'csv_sha256':hashcsv,'csv_line':line,'csv_fields':row,'candidate_count':len(candidates),'exact_globalTaskId':gid,'start_ns':start,'end_ns':end,'pmu_rows':vals}))
'''
cmd=shlex.join(['docker','exec','wio-vllm-v41-repro169-stage-20261028','python3','-c',code,base,rank])
p=subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh',cmd],capture_output=True,text=True,timeout=50,check=True)
j=json.loads(p.stdout);j['source_script_sha256']=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest();j['scope']='EP8, digest 1b826883, layer 0; historical capture; single task; existing installed exporter not proven historical';out=root/'agents/diagnosis/tensormove-ep8-l0-pmu.json';out.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n');print(json.dumps(j,ensure_ascii=False,indent=2))
