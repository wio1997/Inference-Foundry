from pathlib import Path
import ast,hashlib,json,types
r=Path('/Users/wio/work/Inference-Foundry-glm5-3/glm5-3/records/points/GLM-OPT-0002/runs/GLM-RUN-0277');p=r/'pd_rope_correctness.py';tree=ast.parse(p.read_text());body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='workflow'];assert len(body)==1
scenarios=[None,'launch','baseline_short','candidate_short','complete','baseline_warm','terminal_witness','terminal_guard','recovery_launch','recovery_warm']
results=[]
for failure in scenarios:
 calls=[];records={};in_recovery=False
 def rpc(host,action,*values):
  global in_recovery
  calls.append((host,action,values))
  if action=='recoverycleanup':in_recovery=True;return {}
  if failure=='launch' and action=='launch' and values==('candidate',):raise RuntimeError('candidate launch')
  if failure=='recovery_launch' and action=='launch':raise RuntimeError('launch')
  if failure in ('terminal_guard','recovery_warm') and action=='guardfresh' and not in_recovery and len([c for c in calls if c[1]=='switch'])==2:raise RuntimeError('terminal guard')
  if failure=='terminal_guard' and action=='guardfresh' and not in_recovery and any(c[1]=='switch' and c[2]==(0,) for c in calls[10:]):raise RuntimeError('terminal guard')
  if action=='epoch':return {'epoch':'baseline_recovery' if in_recovery else 'candidate'}
  if failure=='terminal_witness' and action=='witness' and len([c for c in calls if c[1]=='witness'])==4:raise RuntimeError('terminal witness')
  return {}
 requests=[]
 def request(label,complete=False):
  requests.append(label)
  expected={'baseline_short':'correctness_mode0_short','candidate_short':'correctness_mode1_short','complete':'correctness_mode1_complete','baseline_warm':'retained_mode_warm','recovery_warm':'H6_H5_recovery_warm'}
  if label==expected.get(failure):raise RuntimeError(failure)
  return {'accepted':True}
 class MockPath:
  def __init__(self,x):self.x=str(x)
  def __truediv__(self,x):return MockPath(self.x+'/'+str(x))
  def read_text(self):
   if self.x.endswith('controller-owner.json'):return json.dumps({'run_id':'GLM-RUN-0277','status':'running','owner':{}})
   if self.x.endswith('_P.raw'):return json.dumps({'kv_transfer_params':{'remote_request_id':'fake'}})
   raise AssertionError(self.x)
 ns=dict(Path=MockPath,json=json,PLAN={'run_id':'GLM-RUN-0277'},ROOT=MockPath('ROOT'),m=types.SimpleNamespace(live=lambda x:True,write=lambda n,v:records.__setitem__(n,v)),rpc=rpc,ready=lambda:None,request=request)
 exec(compile(ast.Module(body=body,type_ignores=[]),str(p),'exec'),ns)
 error=None
 try:ns['workflow']()
 except BaseException as e:error=repr(e)
 assert len(requests)<=5 and len([c for c in calls if c[1]=='launch' and c[2]==('baseline_recovery',)])<=1
 if failure is None:
  assert records['diagnostic_status.json']['status']=='completed' and error is None
 else:assert records.get('diagnostic_status.json',{}).get('status')!='completed'
 if failure in ('baseline_warm','terminal_witness'):assert any(c[1]=='launch' and c[2]==('baseline_recovery',) for c in calls)
 results.append(dict(failure=failure,requests=requests,error=error,terminal_status=records.get('diagnostic_status.json',{}).get('status'),recovery_launches=len([c for c in calls if c[1]=='launch' and c[2]==('baseline_recovery',)])))
out=r/'workflow_CPU_result.json';assert not out.exists();out.write_text(json.dumps(dict(passed=True,driver_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),scenarios=results,NPU_requests=0),indent=2)+'\n');print(json.dumps(dict(passed=True,scenarios=len(results),max_requests=max(len(x['requests']) for x in results))))
