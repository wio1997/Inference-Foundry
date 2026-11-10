#!/usr/bin/env python3
"""Execute original Python cleanup bodies with explicit non-device stubs."""
from __future__ import annotations
import ast,copy,json,sys
from contextlib import contextmanager,nullcontext
from functools import partial
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace as NS
sys.dont_write_bytecode=True
ROOT=Path('audit/execution-search')
def main():
    dep=json.loads((ROOT/'input-resource-dependencies.json').read_text());flag=json.loads((ROOT/'input-frozen-flags.json').read_text());sources=list({s['runtime_path']:s for s in dep['sources']+flag['sources']}.values());excerpts=[]
    for s in sources:assert sha256(Path(s['local_path']).read_bytes()).hexdigest()==s['runtime_sha256']
    def path(suffix):return Path(next(s['local_path'] for s in sources if s['runtime_path'].endswith(suffix)))
    def node(p,owner,name):
        cls=next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.ClassDef) and n.name==owner);n=next(n for n in cls.body if getattr(n,'name',None)==name)
        excerpts.append(dict(path=str(p),sha256=sha256(p.read_bytes()).hexdigest(),owner=owner,method=name,first=n.lineno,last=n.end_lineno,lines=p.read_text().splitlines()[n.lineno-1:n.end_lineno]));return copy.deepcopy(n)
    def execute(nodes,g):exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])),'<original-cleanup-body>','exec'),g)
    def method(p,owner,name,g):
        n=node(p,owner,name);execute([n],g);return g[name]
    controls=[];log=[]
    class ModelStub:
        def prepare_engram_inputs(self,*a,**kw):
            log.append('prepare_external')
            if self.error=='prepare':raise ValueError('explicit prepare stub error')
            return {'engram_lookups':{'L1':'bank1','L14':'bank14'}}
        def prepare_engram_graph_inputs(self,*a):log.append('bind_capture_inputs');return {'engram_lookups':{}}
        def set_attn_backend(self,*a):log.append('set_backend')
        def retire_engram_lookups(self,**kw):log.append('retire_engram')
        def __call__(self,**kwargs):
            log.append('model_call')
            if self.error=='model':raise ValueError('explicit model stub error')
            return 'output'
    def record(label,threw):controls.append(dict(label=label,threw=threw,events=list(log),qualification='Original Python bodies executed with explicit opaque callbacks/stream/event stubs only; not framework/NPU runtime, native completion or observed race'))
    ctx=NS(cudagraph_runtime_mode='NONE',capturing=False,device_metadata_executor=None)
    g={'torch':NS(npu=NS(is_current_stream_capturing=lambda:False)),'get_forward_context':lambda:ctx,'CUDAGraphMode':NS(FULL='FULL'),'partial':partial}
    forward=method(path('/worker/model_runner_v1.py'),'NPUModelRunner','_model_forward',g)
    for mode,error,capturing in [('NONE',None,False),('FULL',None,False),('NONE','model',False),('FULL','model',False),('NONE','prepare',False),('FULL',None,True)]:
        log.clear();model=ModelStub();model.error=error;ctx.cudagraph_runtime_mode=mode;ctx.capturing=capturing
        executor=NS(submission_in_flight=True,release=lambda:log.append('metadata_release'));ctx.device_metadata_executor=executor
        runner=NS(model=model,attn_backend='backend',enable_enpu=False,_get_engram_device_inputs=lambda:{},_update_full_graph_params_if_needed=lambda *a:log.append('graph_params'))
        threw=False
        try:forward(runner,24,input_ids='ids',positions='positions')
        except ValueError:threw=True
        assert threw==(error is not None)
        assert 'retire_engram' not in log
        assert ('metadata_release' in log)==(error!='prepare')
        assert ('bind_capture_inputs' in log)==capturing
        record('V1_'+mode+'_'+str(error)+'_capture'+str(capturing),threw)
    # V2 has a different explicit finish hook; both retirement and base cleanup
    # remain observable, even when the retire callback raises.
    class Base:
        def finish_execution(self,*,failed):log.append('base_finish')
    n=node(path('/models/deepseek_v41/engram/model_state.py'),'EngramModelState','finish_execution')
    cls=ast.ClassDef(name='EngramModelState',bases=[ast.Name(id='Base',ctx=ast.Load())],keywords=[],body=[n],decorator_list=[]);g={'Base':Base};execute([cls],g)
    for failed in [False,True]:
        for retire_raises in [False,True]:
            log.clear()
            def retire(**kw):
                log.append('retire_engram_reset'+str(kw['reset_events']))
                if retire_raises:raise ValueError('explicit retire stub error')
            obj=g['EngramModelState']();obj.model=NS(retire_engram_lookups=retire);threw=False
            try:obj.finish_execution(failed=failed)
            except ValueError:threw=True
            assert log==['retire_engram_reset'+str(failed),'base_finish'] and threw==retire_raises and obj._engram_inputs is None
            record('V2_finish_failed'+str(failed)+'_retireRaises'+str(retire_raises),threw)
    # Existing captured producer helper has a true try/finally auxiliary join.
    class Stream:
        def __init__(self,name):self.name=name
        def wait_stream(self,other):log.append(self.name+'_wait_'+other.name)
    main=Stream('main');aux=Stream('aux')
    g={'contextmanager':contextmanager,'torch':NS(npu=NS(current_stream=lambda:main,stream=lambda s:nullcontext()))}
    captured=method(path('/models/deepseek_v41/model.py'),'DeepseekV41Model','captured_engram_inputs',g)
    for error in [None,'prepare','consumer']:
        log.clear()
        def prepare(*a,**kw):
            log.append('producer_call')
            if error=='prepare':raise ValueError('explicit producer stub error')
        model=NS(_get_engram_input_buffers=lambda:({'L1':'bank'},'mask'),_engram_max_tokens=100,_engram_capture_stream=aux,_engram_capture_events=('maskevent',{'L1':'lookupevent'}),prepare_engram=prepare)
        threw=False
        try:
            with captured(model,'ids',NS(shape=(24,),device='npu'),None,'qsl','valid'):
                log.append('consumer_call')
                if error=='consumer':raise ValueError('explicit consumer stub error')
        except ValueError:threw=True
        assert log[0]=='aux_wait_main' and log[-1]=='main_wait_aux' and threw==(error is not None)
        record('captured_helper_'+str(error),threw)
    # Retirement helper's reset semantics are tested independently from callers.
    g={'torch':NS(npu=NS(current_stream=lambda:main))};retire=method(path('/models/deepseek_v41/model.py'),'DeepseekV41Model','retire_engram_lookups',g)
    for reset in [False,True]:
        log.clear();ev=NS(reset=lambda stream:log.append('event_reset'))
        model=NS(has_engram=True,_engram_prepare_stream=aux,_engram_graph_events={'bucket':(ev,{'L1':ev,'L14':ev})})
        retire(model,reset_events=reset);assert log==['main_wait_aux']+(['event_reset']*3 if reset else [])
        record('explicit_retire_helper_reset'+str(reset),False)
    # The inherited V1 input-prep context protects reused CPU buffers via its
    # own copy event; it does not imply every later producer/consumer is done.
    g={'contextmanager':contextmanager}
    synchronize=method(path('/v1/worker/gpu_model_runner.py'),'GPUModelRunner','synchronize_input_prep',g)
    for event_present in [False,True]:
        for error in [False,True]:
            log.clear();event=NS(synchronize=lambda:log.append('copy_event_sync'),record=lambda:log.append('copy_event_record'))
            threw=False
            try:
                with synchronize(NS(prepare_inputs_event=event if event_present else None)):
                    log.append('CPU_write')
                    if error:raise ValueError('explicit CPU prep callback error')
            except ValueError:threw=True
            assert log==(['copy_event_sync'] if event_present else [])+['CPU_write']+(['copy_event_record'] if event_present else []) and threw==error
            record('V1_CPU_input_guard_event'+str(event_present)+'_error'+str(error),threw)
    assert len(controls)==19 and 'torch' not in sys.modules
    out=dict(status='PASS_ORIGINAL_PYTHON_RETIREMENT_CONTRACTS_ONLY',controls=controls,count=len(controls),findings=['Selected V1 _model_forward finally releases metadata executor, not Engram producer; explicit callback prepare failure occurs before that protected try.','V2 EngramModelState.finish_execution explicitly retires Engram and always invokes superclass finish. It is not the selected V1 cleanup path.','Captured slotless helper joins producer on producer/consumer exception, but selected V1 SP/slot-cache route is ineligible for that helper.','Existing producer preparation catches its own auxiliary prepare() error and invokes retire helper; a later model callback failure is a separate boundary.'],qualification='Opaque callback failures are source controls, not an observed device race or proof of service recovery policy. No changed primitive, framework integration, live task or NPU evidence.',new_primitive_admissions=0,new_framework_replacement_admissions=0,whole_chain_complete=False,new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[],inputs=[dict(path=str(ROOT/n),sha256=sha256((ROOT/n).read_bytes()).hexdigest()) for n in ['input-resource-dependencies.json','input-frozen-flags.json']],reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'input-retirement-controls.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');(ROOT/'input-retirement-source-excerpts.json').write_text(json.dumps(dict(excerpts=excerpts,scope='Only original bodies exercised by opaque stubs'),ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(controls=len(controls),new_primitive_admissions=0,new_NPU_work=0,whole_chain_complete=False)))
if __name__=='__main__':main()
