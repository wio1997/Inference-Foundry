#!/usr/bin/env python3
"""Read existing per-rank databases for the selected layer-9 F08 pair only."""
import hashlib,json,shlex,subprocess
from pathlib import Path
ssh='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh';stage='wio-vllm-v41-repro169-stage-20261028'
identity='42a5ed03e3a3014d64e47af56a0304a3e9724573bf8df0c98adc47866c587602|sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
code=r'''import pathlib,sqlite3,hashlib,json
R=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces')
ends={0:[1793241333882594678,1793241333882620097,1793241333882639117],3:[1793241333883394355,1793241333884739135,1793241333885955955]}; names=['RmsNorm','InplacePartialRotaryMul','aclnnScatterNdUpdateSk_ScatterNdUpdateSkAiCore_ScatterNdUpdateSk'];pts=['aclnnRmsNorm','aclnnInplacePartialRotaryMul','aclnnScatterNdUpdateSk'];res=[]
for ep,starts in ends.items():
 ps=list(R.glob(f'*ep{ep}_rank*/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_{ep}.db'));assert len(ps)==1;p=ps[0];h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 assert h.hexdigest()=={0:'a1cf18e420823e625e6f38c42f9f16dd816500bccf9a18346c815aaa9b4f6edc',3:'084b8718888e33271435ec79d7573c95ca1615f3e7484c807ea24590fad234c4'}[ep]
 c=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True);c.execute('pragma query_only=on');c.row_factory=sqlite3.Row;rr=[]
 for start,name,ptname in zip(starts,names,pts):
  rows=c.execute('select * from TASK where startNs=? and deviceId=?',(start,ep)).fetchall();assert len(rows)==1;t=dict(rows[0]);info=c.execute('select s.value from COMPUTE_TASK_INFO i left join STRING_IDS s on s.id=i.name where i.globalTaskId=?',(t['globalTaskId'],)).fetchall();assert len(info)==1 and info[0][0]==name
  cann=[dict(x) for x in c.execute('select a.startNs,a.endNs,a.connectionId,a.globalTid,s.value name from CANN_API a left join STRING_IDS s on s.id=a.name where a.connectionId=?',(t['connectionId'],))];assert len(cann)==1 and cann[0]['name']=='launch'
  links=[v[0] for v in c.execute('select id from CONNECTION_IDS where connectionId=?',(t['connectionId'],))]
  pt=[]
  for ident in links:
   pt.extend(dict(x) for x in c.execute('select a.startNs,a.endNs,a.globalTid,a.connectionId,a.callchainId,s.value name from PYTORCH_API a left join STRING_IDS s on s.id=a.name where a.connectionId=?',(ident,)))
  assert len(pt)==1 and pt[0]['name']==ptname
  rr.append({'task':t,'cann_launch':cann[0],'pytorch_api':pt[0]})
 assert len({x['task']['streamId'] for x in rr})==1 and len({x['cann_launch']['globalTid'] for x in rr})==1 and len({x['pytorch_api']['globalTid'] for x in rr})==1
 assert [x['task']['taskId'] for x in rr]==list(range(rr[0]['task']['taskId'],rr[0]['task']['taskId']+3))
 res.append({'ep':ep,'db_sha256':h.hexdigest(),'records':rr,
 'cann_rope_launch_to_scatter_launch_start_ns':rr[2]['cann_launch']['startNs']-rr[1]['cann_launch']['startNs'],
 'pytorch_rope_to_scatter_start_ns':int(rr[2]['pytorch_api']['startNs'])-int(rr[1]['pytorch_api']['startNs'])})
print(json.dumps({'samples':res,'limits':['Only exact recorded device tasks and same-rank connection IDs; no data identity, slot contents, tiling, or binary identity.','Differences are separately within CANN same thread or PyTorch same thread; no host/device or cross-rank timestamp subtraction.']},indent=2))'''
def call(a):
 r=subprocess.run([ssh,shlex.join(a)],capture_output=True,text=True,timeout=90)
 if r.returncode:raise RuntimeError(r.stderr[-1000:])
 return r.stdout
assert call(['docker','inspect','--format','{{.Id}}|{{.Image}}',stage]).strip()==identity
raw=call(['docker','exec',stage,'nice','-n','15','ionice','-c','3','python3','-c',code]);data=json.loads(raw)
assert call(['docker','inspect','--format','{{.Id}}|{{.Image}}',stage]).strip()==identity
p=Path(__file__).with_name('f08-4760-ep0-ep3-db-binding.json');p.write_text(json.dumps(data,indent=2)+'\n');print('sha256',hashlib.sha256(p.read_bytes()).hexdigest())
for d in data['samples']:print('ep',d['ep'],'db_sha256',d['db_sha256'],'cann_start_gap_ns',d['cann_rope_launch_to_scatter_launch_start_ns'],'pt_start_gap_ns',d['pytorch_rope_to_scatter_start_ns'])
