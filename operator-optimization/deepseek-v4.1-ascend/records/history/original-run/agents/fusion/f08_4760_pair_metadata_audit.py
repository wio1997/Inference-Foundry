#!/usr/bin/env python3
"""Bounded read-only EP0/EP3 L9 profiler fields; no NPU, no tensor contents."""
import hashlib,json,shlex,subprocess
from pathlib import Path
ssh='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh';stage='wio-vllm-v41-repro169-stage-20261028'
identity='42a5ed03e3a3014d64e47af56a0304a3e9724573bf8df0c98adc47866c587602|sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
code=r'''import pathlib,sqlite3,hashlib,json,ijson
from decimal import Decimal
R=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces')
cases=[(0,5198,13780,'754112','33','a1cf18e420823e625e6f38c42f9f16dd816500bccf9a18346c815aaa9b4f6edc','37c524879574b67d0b90fd5d0bf5232488f4da84a346cf80847b58b692763e5e'),(3,5004,13720,'2179523','35','084b8718888e33271435ec79d7573c95ca1615f3e7484c807ea24590fad234c4','c1eb4df8708a10d2c0be9af84ef8ed2a18f29c19f6297eb1f62aa6743ecaaa85')]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for v in iter(lambda:f.read(1048576),b''):h.update(v)
 return h.hexdigest()
res=[]
for ep,gid,tid,pid,stream,dbhash,trhash in cases:
 ps=list(R.glob(f'*ep{ep}_rank*/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_{ep}.db'));assert len(ps)==1;p=ps[0];tr=p.parent/'trace_view.json';assert sha(p)==dbhash and sha(tr)==trhash
 c=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True);c.execute('pragma query_only=on');c.row_factory=sqlite3.Row
 task=[dict(v) for v in c.execute('select * from TASK where globalTaskId=?',(gid,))];assert len(task)==1 and task[0]['taskId']==tid and task[0]['deviceId']==ep
 rows=[dict(v) for v in c.execute('select * from COMPUTE_TASK_INFO where globalTaskId=?',(gid,))];assert len(rows)==1;comp=rows[0]
 for k in ('name','opType','inputFormats','inputDataTypes','inputShapes','outputFormats','outputDataTypes','outputShapes','attrInfo','opState','hf32Eligible','gridDim','blockDim'):
  v=c.execute('select value from STRING_IDS where id=?',(comp[k],)).fetchone();comp[k]=v[0] if v else None
 assert comp['name']=='aclnnScatterNdUpdateSk_ScatterNdUpdateSkAiCore_ScatterNdUpdateSk'
 pmu=[]
 for v in c.execute('select timestampNs,name,value from TASK_PMU_INFO where globalTaskId=?',(gid,)):
  n=c.execute('select value from STRING_IDS where id=?',(v['name'],)).fetchone();pmu.append({'timestampNs':v['timestampNs'],'name':n[0] if n else None,'value':v['value']})
 assert len(pmu)==28 and len({z['name'] for z in pmu})==28 and len({z['timestampNs'] for z in pmu})==1
 events=[]
 with tr.open('rb') as f:
  for e in ijson.items(f,'item'):
   if not isinstance(e,dict) or e.get('ph')!='X' or str(e.get('pid'))!=pid or str(e.get('tid'))!=stream:continue
   if (e.get('args') or {}).get('Task Id')==tid:
    events.append({'name':e.get('name'),'ts':str(e.get('ts')),'dur':str(e.get('dur')),'args':e.get('args')})
 assert len(events)==1
 assert Decimal(events[0]['ts'])*1000==task[0]['startNs']
 assert Decimal(events[0]['dur'])*1000==task[0]['endNs']-task[0]['startNs']
 res.append({'ep':ep,'db_sha256':dbhash,'trace_sha256':trhash,'task':task[0],'compute_fields':comp,'pmu':pmu,'trace_x':events[0]})
assert {k:v for k,v in res[0]['compute_fields'].items() if k!='globalTaskId'}=={k:v for k,v in res[1]['compute_fields'].items() if k!='globalTaskId'}
assert {x['name'] for x in res[0]['pmu']}=={x['name'] for x in res[1]['pmu']}
print(json.dumps({'cases':res,'limitations':['One TASK, 28 same-timestamp PMU fields and one selected stream/task trace event per rank; no claim about full raw profiler format or runtime memory data.','Compute attrs empty is not proof of absence or runtime stride; name/shape do not bind binary or tiling key.','PMU values are exported fields, not direct validated end-to-end time, HBM bytes or count of valid slots.']},indent=2,default=str))'''
def call(a):
 r=subprocess.run([ssh,shlex.join(a)],capture_output=True,text=True,timeout=180)
 if r.returncode:raise RuntimeError(r.stderr[-1000:])
 return r.stdout
assert call(['docker','inspect','--format','{{.Id}}|{{.Image}}',stage]).strip()==identity
raw=call(['docker','exec',stage,'nice','-n','15','ionice','-c','3','python3','-c',code]);data=json.loads(raw)
assert call(['docker','inspect','--format','{{.Id}}|{{.Image}}',stage]).strip()==identity
p=Path(__file__).with_name('f08-4760-pair-metadata-audit.json');p.write_text(json.dumps(data,indent=2)+'\n');print('output_sha256',hashlib.sha256(p.read_bytes()).hexdigest())
for x in data['cases']:
 print('EP',x['ep'],'trace_args',sorted(x['trace_x']['args']),'PMU_names',[(m['name'],m['value']) for m in x['pmu'] if m['name'] in ('aiv_total_cycles','aiv_vec_ratio','aiv_total_time')])
