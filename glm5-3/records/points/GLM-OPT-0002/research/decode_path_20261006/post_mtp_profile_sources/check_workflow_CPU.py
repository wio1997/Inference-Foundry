from pathlib import Path
import ast,json,types
repo=Path('/Users/wio/work/Inference-Foundry-glm5-3');r=repo/'glm5-3/records/points/GLM-OPT-0002/runs/GLM-RUN-0276';source=(r/'pd_current_profile.py').read_text();node=ast.parse(source);fn=next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name=='workflow');code=compile(ast.Module(body=[fn],type_ignores=[]),'actualworkflow','exec')
roles={'166':{'root':{'pid':1},'worker_start_ticks':{'1':'1'}},'167':{'root':{'pid':2},'worker_start_ticks':{'2':'2'}}}
witness={'H11_rows':[{'transition':11,'rows':[{},{}]} for _ in range(16)]}
class FakePath:
 def __init__(self,p):self.p=str(p)
 def __truediv__(self,p):return FakePath(self.p+'/'+p)
 def read_text(self):
  if self.p.endswith('state.json'):v={'status':'completed'}
  elif self.p.endswith('decision.json'):v={'verdict':'INCONCLUSIVE'}
  else:v={'run_id':'GLM-RUN-0276','status':'running','owner':{}}
  return json.dumps(v)
rows=[]
for fail in ('none','start','request_on','stop','request_off'):
 calls=[];written={};x=types.SimpleNamespace();old=types.SimpleNamespace();d=types.SimpleNamespace(x=x,oldclient=old,PLAN={'gold':'actual'},rpc=lambda h,a,*args: {'epoch':'candidate'} if a=='epoch' else witness if a=='witness' else roles[h]);m=types.SimpleNamespace(live=lambda p:True,write=lambda n,v:written.update({n:v}));c=types.SimpleNamespace()
 def control(a):
  calls.append(a)
  if fail==a:raise RuntimeError(fail)
 def request(label,complete):
  calls.append(label);assert complete
  if fail==('request_on' if 'on' in label else 'request_off'):raise RuntimeError(fail)
  return {'workload_signature':{'drafts':11}}
 c.measured_request=request
 env={'Path':FakePath,'PLAN':{'matched_root':'/matched','run_id':'GLM-RUN-0276','transition':11},'ROOT':FakePath('/new'),'json':json,'m':m,'c':c,'d':d,'inventory':lambda:{'files':[] if not any(a=='start' for a in calls) else [{'path':'trace'}]},'profile_control':control}
 exec(code,env);error=None
 try:env['workflow']()
 except Exception as e:error=str(e)
 assert calls.count('start')==calls.count('stop')==1
 assert c.ROOT is env['ROOT'] and x.ROOT is env['ROOT'] and m.ROOT is env['ROOT'];assert x.PLAN['run_id']=='GLM-RUN-0276' and old.PLAN==x.PLAN
 if fail=='none':assert error is None and calls==['start','profile_on_complete','stop','profile_off_complete'] and written['diagnostic_status.json']['status']=='completed'
 else:assert error==fail and 'diagnostic_status.json' not in written
 if fail in ('start','request_on','stop'):assert 'profile_off_complete' not in calls
 rows.append({'failure':fail,'calls':calls,'completed':error is None})
result={'passed':True,'actual_AST_workflow_cases':rows,'NPU_operations':0,'not_device_proof':True}
q=repo/'glm5-3/records/points/GLM-OPT-0002/research/decode_path_20261006/post_mtp_profile_sources';p=q/'workflow_CPU_result.json';assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n');(q/'check_workflow_CPU.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(result))
