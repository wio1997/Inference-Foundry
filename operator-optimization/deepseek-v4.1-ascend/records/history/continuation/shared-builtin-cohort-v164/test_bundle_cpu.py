"""New bundle orchestration/factory contracts only; no real Torch/CANN or Docker."""
import ast,contextlib,hashlib,importlib.util,json,os,shlex,sys,tempfile,types
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).parent;N=ROOT.parent;sys.path.insert(0,str(ROOT/'kit'))
import cohort_policy as policy
import controller
import shared_runtime as rt
checks=[]
def check(ok,label):
    assert ok,label
    checks.append(label)
def raises(fn,label):
    try:fn()
    except RuntimeError:checks.append(label)
    else:raise AssertionError(label)

original=N/'profile-entrypoints-cache-v162/client.py'
def semantic(path):
    tree=ast.parse(path.read_text())
    tree.body=[n for n in tree.body if not (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ['INPUT','URL'] for t in n.targets))]
    return ast.dump(tree,include_attributes=False)
check(semantic(original)==semantic(ROOT/'kit/client.py'),'Frozen full client algorithm unchanged;only endpoint/input-root assignments differ')
old=shlex.split((N/'profile-entrypoints-cache-v162/serve.sh').read_text().splitlines()[-1])
new=shlex.split((ROOT/'kit/serve-diagnostic.sh').read_text().splitlines()[-1])
check(new[:-2]==old and new[-2]=='--profiler-config','Full model/TP/DP/EP/spec/graph CLI remains unchanged apart from profiler configuration')
cfg=json.loads(new[-1]);check(cfg['max_iterations']==3 and cfg['ignore_frontend'] and cfg['torch_profiler_with_memory'],'Three-worker-step limit/frontend exclusion/memory flags present')

plan={'approval_slot_id':'prepared-slot','run_id':'cpu-test','limits':{'acquisition_seconds':2700,'client_seconds':900,'directory_stop_threshold_bytes':16*1024**3,'diagnostic_load_seconds':1800,'restore_load_seconds':2400},'diagnostic_port':30076}
pb=json.dumps(plan).encode();pr=json.dumps({'diagnostic_container_id':'d'*64,'execution_pins':[]}).encode()
approval={'status':'AUTHORIZED_ONCE','user_reply':'CPU fixture only','plan_sha256':hashlib.sha256(pb).hexdigest(),
 'prepared_sha256':hashlib.sha256(pr).hexdigest(),'approval_id':'prepared-slot'}
check(policy.approved(pb,pr,approval)==plan,'Exact plan/prepared receipt and explicit approval are jointly required')
raises(lambda:policy.approved(pb,pr,dict(approval,status='PREPARED_NOT_AUTHORIZED')),'Prepared slot alone does not authorize execution')
raises(lambda:policy.approved(pb+b' ',pr,approval),'Any changed plan byte rejects the old approval')
raises(lambda:policy.approved(pb,pr+b' ',approval),'Changed diagnostic preparation rejects approval')
raises(lambda:policy.owns_processes({1,999},{1}),'Foreign NPU PID prevents workload/container transition')
check(not policy.restore_allowed({999}) and policy.restore_allowed(set()),'Restoration waits for an idle NPU pool without killing foreign processes')
zero='\n'.join(f'vllm:num_requests_{x}{{engine="{e}",model_name="dsv41"}} 0' for x in ['running','waiting'] for e in [0,1])
check(len(policy.idle_metrics(zero))==4,'Both DP running/waiting gauges explicitly checked')
raises(lambda:policy.idle_metrics(zero.replace(' 0',' 1',1)),'Busy service rejected instead of interrupting current requests')

# Execute the exact factory class source against constructor-only fake objects.
source=N/'profile-entrypoints-cache-v162/profiler_factory.py';tree=ast.parse(source.read_text())
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='TorchNPUProfilerWrapper')
captured=[];handler=[]
fake=types.SimpleNamespace(profiler=types.SimpleNamespace(
 _ExperimentalConfig=lambda **k:k,ExportType=types.SimpleNamespace(Text='Text'),
 ProfilerLevel=types.SimpleNamespace(Level1='Level1'),AiCMetrics=types.SimpleNamespace(PipeUtilization='PipeUtilization'),
 ProfilerActivity=types.SimpleNamespace(CPU='CPU',NPU='NPU'),
 profile=lambda **k:captured.append(k) or types.SimpleNamespace(),
 tensorboard_trace_handler=lambda *a,**k:handler.append((a,k)) or (lambda x:None)))
mod=types.ModuleType('vllm_ascend.profiler.torch_npu_profiler');mod.__file__=str(source)
mod.__dict__.update(torch_npu=fake,ProfilerConfig=object,WorkerProfiler=object,Any=object,
 suppress=contextlib.suppress,get_ascend_config=lambda:types.SimpleNamespace(msmonitor_use_daemon=False))
exec(compile(ast.Module(body=[cls],type_ignores=[]),str(source),'exec'),mod.__dict__)
events=[];fakeobs=types.SimpleNamespace(sink=types.SimpleNamespace(emit=lambda *a,**k:events.append((a,k))))
with patch.object(rt,'observer',return_value=fakeobs):
    rt.install_factory(mod)
    config=types.SimpleNamespace(profiler='torch',torch_profiler_dir='/own/formal',torch_profiler_with_memory=True,torch_profiler_with_stack=True,max_iterations=3)
    mod.TorchNPUProfilerWrapper._create_profiler(config,'rank0')
check(len(captured)==1 and all(captured[0][x] for x in ['record_shapes','profile_memory','with_stack']),
 'Transformed exact factory forwards the runtime-validated field flags once')
check(captured[0]['activities']==['CPU','NPU'] and handler[0][1]['analyse_flag'] is False,
 'Formal CPU/NPU capture retained;automatic offline analysis disabled')
check(events[-1][1]['max_iterations']==3,'Factory records the actual configured worker-step bound')

def simulation(kind):
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);(root/'control').mkdir();(root/'kit').mkdir();(root/'outputs').mkdir()
        (root/'control/PLAN.json').write_bytes(pb);(root/'control/PREPARED.json').write_bytes(pr)
        (root/'control/APPROVAL.json').write_text(json.dumps(approval))
        state={'main':True,'diag_attempted':False};commands=[]
        def inspect(cid):
            if cid==controller.MAIN:return {'Id':cid,'Image':controller.IMAGE,'State':{'Running':state['main'],'StartedAt':'known-start','Status':'running' if state['main'] else 'exited'},'Config':{'Labels':{'wio.migration':'166-to-171-20261102'}}}
            return {'Id':cid,'Image':controller.IMAGE,'State':{'Running':False,'Status':'created'},'Config':{'Labels':{'wio.shared-cohort':'cpu-test'}}}
        def command(argv,timeout=30):
            commands.append(argv)
            if argv[:2]==['docker','top']:return 'PID\n1\n'
            if argv[:2]==['docker','stop'] and argv[-1]==controller.MAIN:state['main']=False;return controller.MAIN
            if argv[:2]==['docker','start']:
                state['diag_attempted']=True;raise RuntimeError('Intentional diagnostic-start fixture failure')
            if argv[0]=='python3' and argv[-1]=='start':state['main']=True;return 'original-started'
            raise AssertionError(argv)
        def busy():
            if state['main']:return {1},'own-main'
            if state['diag_attempted'] and kind=='foreign_restore':return {999},'foreign'
            return set(),'idle'
        class Child:
            pid=987654;returncode=1 if kind=='baseline_error' else 0
            def poll(self):return self.returncode
        def popen(argv,**kwargs):
            if kind!='baseline_error':
                out=Path(kwargs['env']['WIO_OUTPUT']);out.mkdir();(out/'window-status.json').write_text(json.dumps({'valid':True,'windows':[]}))
            return Child()
        counter=[0]
        def monotonic():counter[0]+=100;return counter[0]
        with patch.object(controller.socket,'gethostname',return_value='bm-jn-zs-zone1-910C-64G-10-171'),patch.object(controller.os,'geteuid',return_value=0),\
             patch.object(controller,'inspect',side_effect=inspect),patch.object(controller,'command',side_effect=command),\
             patch.object(controller,'busy',side_effect=busy),patch.object(controller,'request',side_effect=lambda port,path,*a,**k:(200,zero.encode())),\
             patch.object(controller.subprocess,'Popen',side_effect=popen),patch.object(controller.time,'monotonic',side_effect=monotonic),\
             patch.object(controller.signal,'signal'),patch('builtins.print'):
            controller.execute(root)
        result=json.loads((root/'STATUS.json').read_text())
        return result,commands

state,commands=simulation('baseline_error')
check(state['reference_attempts']==1 and state['profile_attempts']==0 and not any(c[:2]==['docker','stop'] for c in commands),
 'Failed reference neither retries nor stops the original model service')
check(state['restore']=='NOT_REQUIRED_ORIGINAL_SERVICE_NOT_STOPPED','Reference failure preserves the running original service')
state,commands=simulation('diagnostic_error')
check(state['reference_attempts']==1 and state['profile_attempts']==0 and state['restore']=='OWN_ORIGINAL_SERVICE_HEALTH_200',
 'Diagnostic start failure still restores the original service exactly once')
check(sum(c[0]=='python3' and c[-1]=='start' for c in commands)==1,'No repeated model load on a polling timeout')
state,commands=simulation('foreign_restore')
check(state['restore']=='DEFERRED_OCCUPIED_OR_UNCLEARED_NPU' and state['restore_busy_pids']==[999],
 'Foreign NPU after cleanup defers restore and records the actual blocking PID')
check(not any(c[0]=='python3' and c[-1]=='start' for c in commands),'No restore model starts over an occupied foreign device')
with tempfile.TemporaryDirectory() as tmp:
    p=Path(tmp)/'STATUS.json';p.write_text('{}')
    raises(lambda:policy.no_existing_submission(p),'Existing submission,even without terminal observation,never authorizes resubmission')
print(json.dumps({'status':'PASS_NEW_CPU_BUNDLE_CONTRACTS_ONLY','gate_count':len(checks),'gates':checks,
 'real_Torch_CANN_or_Docker_tested':False,'new_model_or_device_work':False,'overall_complete':False},ensure_ascii=False,indent=2))
