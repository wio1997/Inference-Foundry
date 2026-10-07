"""Execute the frozen workflow AST with service/request doubles; no imports of runtime."""
from pathlib import Path
from types import SimpleNamespace
import ast, hashlib, json

root=Path(__file__).resolve().parent
source=root/'pd_local_prepare_correctness.py'
node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='workflow')
scenarios=[None,'boot','offlineguard','artifact','install','launch_candidate','ready_candidate',
           'request_old','request_new','request_complete','request_retained',
           'transfer','witness','observeoff','terminal_guard',
           'recovery_cleanup','recovery_restore','recovery_launch','recovery_ready','recovery_request','recovery_witness','recovery_transfer']
out=[]
for failure in scenarios:
    calls=[];requests=[];records={};state={'recovery':False,'ready':0,'witness':0}
    class FakePath:
        def __init__(self,value):self.value=str(value)
        def __truediv__(self,name):return FakePath(self.value+'/'+str(name))
        def read_text(self):
            if self.value.endswith('controller-owner.json'):
                return json.dumps({'run_id':'GLM-RUN-0282','status':'running','owner':{}})
            if self.value.endswith('_P.raw'):
                return json.dumps({'kv_transfer_params':{'remote_request_id':'fixture'}})
            raise AssertionError(self.value)
    def rpc(host,action,*values):
        calls.append((host,action,list(values)))
        if action=='recoverycleanup':state['recovery']=True
        is_recovery=state['recovery']
        if failure==action and not is_recovery:raise RuntimeError(failure)
        if failure=='transfer' and action=='transfergate' and not is_recovery:raise RuntimeError(failure)
        if action=='launch' and values==('candidate',) and (failure=='launch_candidate' or str(failure).startswith('recovery_')):
            raise RuntimeError('candidate launch failure')
        if failure=='terminal_guard' and action=='guardfresh' and len(requests)==4 and not is_recovery:
            raise RuntimeError(failure)
        mapping={'recovery_cleanup':'recoverycleanup','recovery_restore':'restore','recovery_launch':'launch','recovery_witness':'witness','recovery_transfer':'transfergate'}
        if is_recovery and action==mapping.get(failure):raise RuntimeError(failure)
        return {}
    def ready():
        state['ready']+=1
        if failure=='ready_candidate' and not state['recovery']:raise RuntimeError(failure)
        if failure=='recovery_ready' and state['recovery']:raise RuntimeError(failure)
    def request(label,complete=False):
        requests.append(label)
        mapping={'request_old':'correctness_mode0_short','request_new':'correctness_mode1_short',
                 'request_complete':'correctness_mode1_complete','request_retained':'retained_mode_warm',
                 'recovery_request':'H6_H5_recovery_warm'}
        if label==mapping.get(failure):raise RuntimeError(failure)
        return {'accepted':True}
    ns=dict(Path=FakePath,json=json,PLAN={'run_id':'GLM-RUN-0282','initial_D_offline':True},ROOT=FakePath('ROOT'),
            m=SimpleNamespace(live=lambda _:True,write=lambda name,value:records.__setitem__(name,value)),
            rpc=rpc,ready=ready,request=request)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
    error=None
    try:ns['workflow']()
    except BaseException as e:error=repr(e)
    status=records['diagnostic_status.json']
    recovery_launches=sum(action=='launch' and values==['baseline_recovery'] for _,action,values in calls)
    assert len(requests)<=5 and recovery_launches<=1
    assert not any(action=='retire_old' for _,action,_ in calls)
    if failure in ('boot','offlineguard'):assert recovery_launches==0 and not requests
    if failure is None:
        assert error is None and status['status']=='completed' and len(requests)==4 and not status['recovery_used']
    else:
        assert error is not None and status['status']=='failed'
    if failure in ('request_new','request_complete','request_retained','terminal_guard'):
        assert recovery_launches==1 and status['terminal_verified']
    if str(failure).startswith('recovery_'):
        assert not status['terminal_verified']
        assert records['retained_stack.json']['H6'] is None and records['retained_stack.json']['target_FULL'] is None
    if failure=='request_new':
        assert requests==['correctness_mode0_short','correctness_mode1_short','H6_H5_recovery_warm']
    out.append(dict(failure=failure,request_attempts=requests,recovery_launches=recovery_launches,error=error,status=status))
result=dict(passed=True,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            normal_requests=4,max_failure_requests=max(len(row['request_attempts']) for row in out),
            max_recovery_launches=max(row['recovery_launches'] for row in out),
            scenarios=out,model_requests=0,NPU_requests=0)
dest=root/'workflow_CPU_result.json';assert not dest.exists();dest.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='scenarios'}))
