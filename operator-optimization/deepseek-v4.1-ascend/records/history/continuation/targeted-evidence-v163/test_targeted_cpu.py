"""CPU adversarial contracts; exact graph Python source with fake device methods.

These are NOT real Torch dispatch/CANN execution or native-coverage tests.
"""
import importlib.util,json,sys,tempfile,types
from pathlib import Path
from contextlib import nullcontext
from evidence_core import EvidenceSink,join_native_candidates
from targeted_runtime import TargetedObserver

SOURCE=Path(__file__).parents[1]/'profile-entrypoints-cache-v163/compilation/updatable_graph.py'
checks=[]
def check(ok,label):
    assert ok,label
    checks.append(label)

native=dict(run_uuid='r',process_incarnation='p1',device_namespace='dev0-bound',
 context_epoch='ctx1',api_witness='exact-record-44',api_witness_status='VERIFIED_EXACT_CONNECTION',
 task_id_status='QUERIED_SUCCESS',stream_id_status='QUERIED_SUCCESS',stream_id=47,task_id=8,
 kernel_name='K',num_blocks=48)
task={**native,'task_row':7}
check(join_native_candidates(native,[task])['status']=='MATCHED_IN_ATTESTED_CONNECTION_DOMAIN','Exact attested connection domain can match one task')
check(join_native_candidates(native,[task,dict(task)])['status']=='UNKNOWN','Repeated same numeric/full key remains ambiguous')
check(join_native_candidates(native,[dict(task,process_incarnation='p2')])['status']=='UNKNOWN','PID/numeric reuse across process incarnations cannot join')
check(join_native_candidates(dict(native,task_id_status='NOT_QUERIED',task_id=0),[task])['status']=='UNKNOWN','Unqueried task0 sentinel never matches')
check(join_native_candidates(dict(native,context_epoch=None),[task])['status']=='UNKNOWN','Missing context epoch stays unknown')
check(join_native_candidates(native,[dict(task,num_blocks=24)])['status']=='UNKNOWN','Contradictory native signature blocks ID-only join')
check(join_native_candidates(dict(native,api_witness_status='UNVERIFIED'),[task])['status']=='UNKNOWN','Caller-invented matching witness string is insufficient')
check(join_native_candidates({k:v for k,v in native.items() if k!='task_id'},[task])['status']=='UNKNOWN','Missing task number remains unknown without raising or defaulting to0')

calls=[]
class Storage:
    _next=100
    def __init__(self,n):Storage._next+=1;self._cdata=Storage._next;self.n=n
    def nbytes(self):return self.n
class Tensor:
    _next=1000
    def __init__(self,n=24,storage=None,offset=0,inference=False):
        Tensor._next+=1;self._cdata=Tensor._next;self.n=n;self.storage=storage or Storage(n)
        self.offset=offset;self.dtype='bool';self.inference=inference
    def untyped_storage(self):return self.storage
    def storage_offset(self):return self.offset
    def data_ptr(self):return 100000+self.storage._cdata*1024+self.offset
    @property
    def _version(self):
        if self.inference:raise RuntimeError('Inference tensor has no version counter')
        return 0
    def numel(self):return self.n
    def element_size(self):return 1
    def item(self):raise AssertionError('Tensor value read forbidden')
    def cpu(self):raise AssertionError('Tensor copy forbidden')
    def __repr__(self):raise AssertionError('Tensor repr forbidden')
class Event:
    def wait(self,s):calls.append('event_wait')
    def reset(self,s):calls.append('event_reset')
    def record(self,s):calls.append('event_record')
class BaseGraph:
    def capture_begin(self,**k):
        calls.append('native_capture_begin')
        if getattr(self,'fail_begin',False):raise ValueError('native begin fixture error')
    def capture_end(self):
        calls.append('native_capture_end')
        if getattr(self,'fail_end',False):raise ValueError('native end fixture error')
    def replay(self):calls.append('native_replay');return 'REPLAY_RESULT'
class FakeMode:
    def __enter__(self):return self
    def __exit__(self,*a):return False

torch=types.ModuleType('torch');torch.Tensor=Tensor;torch.bool='bool'
torch.npu=types.SimpleNamespace(NPUGraph=BaseGraph,current_stream=lambda:'current-stream',
 ExternalEvent=Event,graph_task_group_begin=lambda s:calls.append('group_begin'),
 graph_task_group_end=lambda s:88,graph_task_update_begin=lambda *a:calls.append('update_begin'),
 graph_task_update_end=lambda *a:calls.append('update_end'),stream=lambda s:nullcontext())
tn=types.ModuleType('torch_npu');tn._npu_paged_attention=lambda **kw:None
tn._npu_paged_attention_get_workspace=lambda **kw:Tensor(32)
logger_mod=types.ModuleType('vllm.logger');logger_mod.logger=types.SimpleNamespace(debug_once=lambda *a:None)
utils=types.ModuleType('vllm_ascend.utils');utils.weak_ref_tensors=lambda x:x
sys.modules.update({'torch':torch,'torch_npu':tn,'vllm.logger':logger_mod,'vllm_ascend.utils':utils})
name='vllm_ascend.compilation.updatable_graph'
spec=importlib.util.spec_from_file_location(name,SOURCE);module=importlib.util.module_from_spec(spec)
sys.modules[name]=module;spec.loader.exec_module(module)

with tempfile.TemporaryDirectory() as d:
    sink=EvidenceSink(d,'cpu-contract',max_records=512,max_bytes=1024**2)
    obs=TargetedObserver(sink,torch,FakeMode);obs.install_graph(module)
    g=module.UpdatableGraph();g.capture_begin()
    check(module._ACTIVE_GRAPH.get() is g,'Exact source capture context retained')
    check(obs.graph_identity(g)['capture_epoch']==1,'Own capture epoch records the actual begin boundary')
    allocations=[]
    def factory(n=24):
        allocations.append(n);return Tensor(n)
    a=g.get_capture_resource('a',factory)
    check(len(allocations)==1,'Resource instrumentation never invokes the original factory extra')
    same=g.get_capture_resource('a',factory)
    check(same is a and len(allocations)==1,'Cached return retains exact Tensor object without factory call')
    larger=g.get_capture_resource('a',lambda:factory(32),True)
    check(larger.n==32 and allocations==[24,32],'Existing max-workspace branch invokes one candidate factory')
    new=g.get_capture_resource('new',factory,True)
    check(allocations==[24,32,24,24],'Absent max-workspace branch invokes two factories exactly as source')
    effects=[]
    def operation(**kwargs):effects.append(kwargs)
    class Provider:
        layer_name='L'
        def resolve(self,context):effects.append('resolve');return {'new':context}
    provider=Provider();g.register_task(operation,{'input':a},provider)
    check(len(effects)==1 and len(g.tasks)==1,'Capture registration executes the original operation once')
    resolved=g.resolve_tasks(module.ContextSource(3))
    check(len(effects)==2 and resolved[0].kwargs['new']==3,'Resolver invoked once and task override preserved')
    g.update('update-stream',resolved)
    check(len(effects)==3 and calls.count('update_begin')==calls.count('update_end')==1,'Update task runs once with original begin/end ordering')
    check(calls[-1]=='event_record','Original event record remains after task update')
    g.capture_end();check(module._ACTIVE_GRAPH.get() is None,'Capture source context reset preserved')
    check(obs.graph_identity(g)['capture_boundary_state']=='CAPTURE_END_RETURNED',
        'Completed Python capture end has an explicit separate boundary state')
    check(g.replay()=='REPLAY_RESULT' and calls.count('native_replay')==1,'Replay return and exactly-one call preserved')
    bad=ValueError('original operation failure')
    def raises(**kw):raise bad
    before=len(g.tasks)
    try:g.register_task(raises,{},provider)
    except ValueError as e:check(e is bad and len(g.tasks)==before,'Original exception object preserved; no invented registration')
    else:raise AssertionError('Original error was swallowed')
    owner=Tensor(21);view=Tensor(20,storage=owner.storage,offset=1,inference=True)
    extra=obs.extra(view)
    check(extra['storage_offset']==1 and extra['storage_nbytes']==21 and extra['StorageImpl']==owner.storage._cdata,
        'Supplemental view offset/extent does not confuse Tensor and storage identity')
    check(extra['version_counter'] is None and extra['version_counter_unknown_reason']=='RuntimeError',
        'Inference tensor version counter remains unknown rather than zero')
    mode=obs.dispatch_mode();op_calls=[]
    arg=types.SimpleNamespace(name='self',alias_info=types.SimpleNamespace(is_write=True))
    class Op:
        _schema=types.SimpleNamespace(name='aten::zero_',overload_name='',arguments=[arg])
        def __call__(self,t):op_calls.append(t);return t
    result=mode.__torch_dispatch__(Op(),(Tensor,),(view,),{})
    check(result is view and op_calls==[view],'Selected BOOL metadata observes one original op and preserves alias return')
    class BadMetadataOp:
        @property
        def _schema(self):raise RuntimeError('Metadata unavailable')
        def __call__(self,t):return t
    check(mode.__torch_dispatch__(BadMetadataOp(),(Tensor,),(view,),{}) is view,
        'Metadata failure does not convert original successful operation into an application error')
    check(obs.enable_dispatch is False,'Dispatch mode is off by default; compilation/capture compatibility not assumed')
    bad_begin=module.UpdatableGraph();bad_begin.fail_begin=True
    try:bad_begin.capture_begin()
    except ValueError:pass
    else:raise AssertionError('Expected fixture begin error')
    identity=obs.graph_identity(bad_begin)
    check(identity['capture_epoch'] is None and identity['capture_boundary_state']=='CAPTURE_BEGIN_RAISED',
        'Failed native begin does not fabricate a successful capture epoch')
    bad_end=module.UpdatableGraph();bad_end.capture_begin();bad_end.fail_end=True
    try:bad_end.capture_end()
    except ValueError:pass
    else:raise AssertionError('Expected fixture end error')
    check(module._ACTIVE_GRAPH.get() is None and obs.graph_identity(bad_end)['capture_boundary_state']=='CAPTURE_END_RAISED',
        'Failed native end retains source cleanup and does not mark capture complete')
    obs.uninstall()
    check('replay' not in module.UpdatableGraph.__dict__,'Uninstall restores inherited native replay descriptor')
    sink.close();rows=[json.loads(x) for x in Path(sink.path).read_text().splitlines()]
    check(rows[-1]['kind']=='footer' and not rows[-1]['failures'],'Complete own CPU fixture has footer and no logging drops')
    check(any(r['kind']=='supplementary_metadata_error' for r in rows),'Metadata failure is an explicit unknown record')
    check(all(r.get('allocation_generation')!='PROVED' for r in rows),'No allocator-generation proof fabricated')
    check(any(r.get('status')=='original_call_raised' for r in rows),'Original exception boundary is retained in the evidence')
with tempfile.TemporaryDirectory() as d:
    sink=EvidenceSink(d,'quota-contract',max_records=8,max_bytes=4096)
    for i in range(20):sink.emit('event',i=i)
    sink.close();rows=[json.loads(x) for x in Path(sink.path).read_text().splitlines()]
    check(rows[-1]['failures'].get('record_limit',0)>0 and Path(sink.path).stat().st_size<=4096,
        'Quota exhaustion retained in footer and hard byte limit respected')
print(json.dumps({'status':'PASS_CPU_CONTRACTS_ONLY','gates':checks,'gate_count':len(checks),
 'exact_graph_source':str(SOURCE),'real_Torch_dispatch_tested':False,'real_CANN_tested':False,
 'new_device_work':False,'whole_modules_closed':0,'overall_complete':False},ensure_ascii=False,indent=2))
