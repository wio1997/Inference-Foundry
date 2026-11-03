"""Builtin profiler fields, startup CPU capture, and Python graph/guard scopes.

No native interposer, TorchDispatchMode, Tensor value read or extra graph replay.
"""
import ast,contextlib,functools,hashlib,inspect,json,os,threading,textwrap
from pathlib import Path

FACTORY_SHA='5242ee153033afcf12dbfe022dbd807fa248d991ff5c53efbd1adb372d5575f4'
WORKER_SHA='51cfb3da18046af8c72ccefe940de5f24d9502bfdae75b941e1798c0a165cda4'
_observer=None;_installed=set();_capture_profiles=0;_capture_active=False

def verify_approval():
    control=Path(os.environ['WIO_COHORT_CONTROL'])
    plan=(control/'PLAN.json').read_bytes();approval=json.loads((control/'APPROVAL.json').read_text())
    digest=hashlib.sha256(plan).hexdigest()
    if approval.get('status')!='AUTHORIZED_ONCE' or approval.get('plan_sha256')!=digest:
        raise RuntimeError('This full-model cohort lacks its own exact approval')
    if approval.get('approval_id')!=os.environ.get('WIO_COHORT_APPROVAL_ID'):
        raise RuntimeError('Cohort approval identity mismatch')
    if os.environ.get('WIO_TARGETED_DISPATCH_ENABLE','0')!='0' or os.environ.get('WIO_HOST_TRACE_ENABLE','0')!='0':
        raise RuntimeError('This cohort excludes dispatch/native hooks')
    return json.loads(plan)

def observer():
    global _observer
    if _observer is None:
        plan=verify_approval()
        import torch
        from evidence_core import EvidenceSink
        from targeted_runtime import TargetedObserver
        class ProfileMarkerSink(EvidenceSink):
            @contextlib.contextmanager
            def scope(self,name,**identity):
                marker='wio_shared/'+self.process_uuid+'/'+str(self.next_id())+'/'+name
                with torch.profiler.record_function(marker):
                    with super().scope(name,profile_marker=marker,**identity) as sid:
                        yield sid
        sink=ProfileMarkerSink(str(Path(os.environ['WIO_COHORT_OUTPUT'])/'supplement'),
            plan['run_id'],max_records=65536,max_bytes=64*1024**2)
        # No TorchDispatchMode is instantiated or entered by this cohort.
        _observer=TargetedObserver(sink,torch,None,enable_dispatch=False)
        sink.emit('cohort_observer_created',plan_sha256=hashlib.sha256((Path(os.environ['WIO_COHORT_CONTROL'])/'PLAN.json').read_bytes()).hexdigest(),
            native_hooks=False,TorchDispatchMode=False,
            compiled_or_capture_path_identity_requires_post_run_check=True)
    return _observer

def _after_fork():
    global _observer,_capture_profiles,_capture_active
    if _observer is not None:
        # Mutate inherited wrapper owners to use a fresh child-owned sink. Never
        # enter a parent's lock/file or reuse its Python graph identity registry.
        old=_observer;_observer=None
        try:replacement=observer()
        except BaseException:os._exit(126)
        old.sink=replacement.sink;old.tokens={};old.epochs={};old.capture_states={};old.next_token=0
        _observer=old
    _capture_profiles=0;_capture_active=False

if hasattr(os,'register_at_fork'):os.register_at_fork(after_in_child=_after_fork)

def _source(module,want):
    got=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    if got!=want:raise RuntimeError('Pinned source mismatch: '+module.__name__)

def install_factory(module):
    _source(module,FACTORY_SHA)
    cls=module.TorchNPUProfilerWrapper;original=cls._create_profiler
    tree=ast.parse(textwrap.dedent(inspect.getsource(original)))
    function=tree.body[0];function.decorator_list=[];function.name='_wio_field_factory'
    profiles=[n for n in ast.walk(function) if isinstance(n,ast.Call) and ast.unparse(n.func)=='torch_npu.profiler.profile']
    handlers=[n for n in ast.walk(function) if isinstance(n,ast.Call) and ast.unparse(n.func)=='torch_npu.profiler.tensorboard_trace_handler']
    if len(profiles)!=1 or len(handlers)!=1:raise RuntimeError('Factory shape changed')
    call=profiles[0];keywords={k.arg:k for k in call.keywords}
    if 'record_shapes' in keywords or ast.unparse(keywords['with_stack'].value)!='False':
        raise RuntimeError('Unexpected builtin field defaults')
    keywords['with_stack'].value=ast.Constant(True)
    keywords['profile_memory'].value=ast.Constant(True)
    call.keywords.append(ast.keyword(arg='record_shapes',value=ast.Constant(True)))
    if any(k.arg=='analyse_flag' for k in handlers[0].keywords):raise RuntimeError('Trace-handler contract changed')
    handlers[0].keywords.append(ast.keyword(arg='analyse_flag',value=ast.Constant(False)))
    ast.fix_missing_locations(tree);ns=dict(module.__dict__)
    exec(compile(tree,'temporary-shared-builtin-factory','exec'),ns)
    transformed=ns['_wio_field_factory']
    @functools.wraps(original)
    def factory(config,trace_name):
        if _capture_active:raise RuntimeError('Formal and startup profilers may not overlap')
        import resource
        resource.setrlimit(resource.RLIMIT_FSIZE,(256*1024**2,256*1024**2))
        prof=transformed(config,trace_name)
        observer().sink.emit('formal_profiler_created',trace_name=trace_name,
            forced_flags={'CPU':True,'NPU':True,'record_shapes':True,'profile_memory':True,'with_stack':True},
            offline_auto_analysis=False,max_iterations=config.max_iterations,
            max_iteration_bound_is_worker_steps_not_semantic_tokens=True)
        return prof
    cls._create_profiler=staticmethod(factory)

def _startup_capture(original):
    @functools.wraps(original)
    def call(obj,*a,**k):
        global _capture_profiles,_capture_active
        obs=observer()
        if _capture_profiles>=1:
            obs.sink.emit('startup_CPU_profile_skipped',reason='one capture_model profile per process limit',remaining_scope_unknown=True)
            return original(obj,*a,**k)
        if _capture_active:raise RuntimeError('Nested startup capture profiler')
        _capture_profiles+=1
        # Model shared weight backings have already been created by load_model.
        import resource
        resource.setrlimit(resource.RLIMIT_FSIZE,(256*1024**2,256*1024**2))
        obs.sink.emit('diagnostic_file_limit_applied_after_model_load',bytes=256*1024**2)
        import torch_npu
        prof=torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU],
            record_shapes=True,profile_memory=True,with_stack=True,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                str(Path(os.environ['WIO_COHORT_OUTPUT'])/'startup'),
                worker_name='capture-'+obs.sink.process_uuid,analyse_flag=False))
        _capture_active=True
        obs.sink.emit('startup_CPU_profile_begin',capture_profile=_capture_profiles,
            native_NPU_profiler_activity=False,raw_fields_are_not_native_template_bridge=True)
        try:
            with prof:return original(obj,*a,**k)
        finally:
            _capture_active=False
            obs.sink.emit('startup_CPU_profile_end',capture_profile=_capture_profiles,
                actual_files_and_flags_require_post_run_validation=True)
    return call

def _scheduler_snapshot(scheduler):
    def scalar(value):return value if value is None or type(value) in (int,bool,str,float) else {'unknown_type':type(value).__name__}
    scheduled=getattr(scheduler,'num_scheduled_tokens',None)
    planned=getattr(scheduler,'scheduled_spec_decode_tokens',None)
    out={'total_num_scheduled_tokens':scalar(getattr(scheduler,'total_num_scheduled_tokens',None))}
    if type(scheduled) is dict and len(scheduled)<=64:
        out['scheduled_by_request']={str(k)[:192]:scalar(v) for k,v in scheduled.items()}
    else:out['scheduled_by_request']={'unknown':'not bounded host dict'}
    if type(planned) is dict and len(planned)<=64:
        out['planned_draft_counts']={str(k)[:192]:len(v) if type(v) in (list,tuple) else None for k,v in planned.items()}
    else:out['planned_draft_counts']={'unknown':'not bounded host dict'}
    out['planned_counts_are_not_accepted_GPU_token_values']=True
    return out

def install_module(module,label):
    if module.__name__ in _installed:return
    verify_approval()
    if label=='factory':install_factory(module)
    elif label=='worker':
        _source(module,WORKER_SHA);old=module.NPUWorker.profile
        @functools.wraps(old)
        def profile(obj,*a,**k):
            try:return old(obj,*a,**k)
            finally:
                is_start=k.get('is_start',a[0] if a else True)
                if not is_start and (Path(os.environ['WIO_COHORT_CONTROL'])/'FINALIZE_METADATA').is_file():
                    observer().sink.emit('explicit_metadata_finalize',invoked_by='stop_profile after full client returned',is_not_GPU_retirement_proof=True)
                    observer().sink.close()
        module.NPUWorker.profile=profile
    else:
        obs=observer()
        if label=='graph':obs.install_graph(module)
        elif label=='wrapper':obs.install_wrapper(module)
        elif label=='runner':
            obs.install_runner(module)
            module.NPUModelRunner.capture_model=_startup_capture(module.NPUModelRunner.capture_model)
            old=module.NPUModelRunner.execute_model
            @functools.wraps(old)
            def execute(obj,*a,**k):
                scheduler=k.get('scheduler_output',a[0] if a else None)
                if obs.sink.wants_records():
                    obs.observe(lambda:obs.sink.emit('scheduler_host_snapshot',**_scheduler_snapshot(scheduler)))
                return old(obj,*a,**k)
            module.NPUModelRunner.execute_model=execute
        else:raise RuntimeError('Unknown diagnostic module label')
    _installed.add(module.__name__)
