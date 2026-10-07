"""Actual runner/wrapper/base AST, CPU queue and external dependency doubles.

Checks control flow and queue-order semantics; neither device correctness nor
real native execution/timing is inferred from these doubles.
"""
from __future__ import annotations
import ast,itertools,json,enum,functools
from pathlib import Path
from types import SimpleNamespace as NS

ROOT=Path(__file__).resolve().parent
class Mode(enum.Enum):
    NONE=0;FULL=1;PIECEWISE=2

def node(path,name):
    matches=[n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name]
    assert len(matches)==1,(path,name,len(matches));return matches[0]

class Queue:
    def __init__(self):self.work=[];self.events=[]
    def enqueue(self,label,func):self.events.append(label);self.work.append(func)
    def drain(self):
        while self.work:self.work.pop(0)()
    def synchronize(self):self.events.append('host_sync');self.drain()

def compile_method(n,namespace):
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),n],type_ignores=[]);ast.fix_missing_locations(module)
    exec(compile(module,'actual_method_AST','exec'),namespace);return namespace[n.name]

def setup(patched,wrapper_source=None):
    q=Queue();state={'context':None,'device_input':0};extra=NS(is_draft_model=False)
    ns={'CUDAGraphMode':Mode,'get_forward_context':lambda:state['context'],'_EXTRA_CTX':extra,'partial':functools.partial,
        'torch':NS(npu=NS(current_stream=lambda:q)),
        'get_offloader':lambda:NS(sync_prev_onload=lambda:q.events.append('offloader_fence')),
        'update_full_graph_params':lambda *args:q.events.append('async_attention_param_update'),
        'weak_ref_workspaces':lambda params:q.events.append('weak_workspace_contract'),
        'get_graph_params':lambda:None,'get_draft_graph_params':lambda:None,'get_draft_graph_prefill_params':lambda:None,
    }
    base_replay=compile_method(node(ROOT/'original/breakable_cudagraph.py','_replay'),ns)
    class Base:
        _replay=base_replay
        def _capture(self,entry,args,kwargs):q.events.append('capture_body');return entry.output
    ns['BreakableCUDAGraphWrapper']=Base
    path=ROOT/('patched' if patched else 'original')
    wrapper_tree=ast.parse((wrapper_source or path/'breakable_aclgraph.py').read_text())
    helpers=[n for n in wrapper_tree.body if isinstance(n,ast.FunctionDef) and n.name.startswith('_h9_') or isinstance(n,ast.Assign) and all(isinstance(t,ast.Name) and t.id.startswith('_H9_') for t in n.targets)]
    if helpers:
        module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*helpers],type_ignores=[]);ast.fix_missing_locations(module);exec(compile(module,'actual_selector_AST','exec'),ns)
    cls=next(n for n in wrapper_tree.body if isinstance(n,ast.ClassDef) and n.name=='BreakableACLGraphWrapper')
    cls.body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('_capture','_replay')]
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),cls],type_ignores=[]);ast.fix_missing_locations(module);exec(compile(module,'actual_wrapper_AST','exec'),ns)
    model_forward=compile_method(node(path/'model_runner_v1.py','_model_forward'),ns)
    updater=compile_method(node(ROOT/'original/model_runner_v1.py','_update_full_graph_params_if_needed'),ns)
    wrapper=ns['BreakableACLGraphWrapper']();wrapper.enable_enpu=False;wrapper.use_eagle=False;wrapper.is_debugging_mode=False
    output={'value':None};capture=NS(replay=lambda:q.enqueue('native_replay_enqueue',lambda:output.update(value=state['device_input'])))
    entry=NS(capture=capture,input_addresses=None,output=output)
    def run_model(**kwargs):
        if state['context'].cudagraph_runtime_mode==Mode.FULL:return wrapper._replay(entry,(),kwargs)
        q.enqueue('eager_enqueue',lambda:output.update(value=state['device_input']));return output
    runner=NS(model=run_model,enable_enpu=False,use_sparse=True,use_compress=False,device_metadata_executor=None,attn_backend=NS(get_name=lambda:'ASCEND_SFA'),model_config=NS(hf_config=NS(model_type='glm_moe_dsa')),update_stream=object(),vllm_config=object(),speculative_config=object())
    runner._model_forward=model_forward.__get__(runner);runner._update_full_graph_params_if_needed=updater.__get__(runner)
    return q,state,extra,wrapper,runner,entry

def case(objects,settings,new_input=47):
    q,state,extra,wrapper,runner,entry=objects
    mode,sparse,compress,backend,executor,model_type,draft,eagle,enpu=settings
    q.events.clear();assert not q.work
    # Reuse the actual per-forward object deliberately to catch stale opt-in.
    if state['context'] is None:state['context']=NS(cudagraph_runtime_mode=mode,capturing=False)
    state['context'].cudagraph_runtime_mode=mode
    runner.use_sparse=sparse;runner.use_compress=compress;runner.attn_backend=NS(get_name=lambda:backend);runner.device_metadata_executor=object() if executor else None
    runner.model_config.hf_config=NS(model_type=model_type);runner.enable_enpu=wrapper.enable_enpu=enpu;wrapper.use_eagle=eagle;extra.is_draft_model=draft
    q.enqueue('input_copy_enqueue',lambda:state.update(device_input=new_input))
    returned=runner._model_forward(2,input_ids=object(),positions=object())
    assert returned is entry.output
    observed=[];q.enqueue('MTP_consumer_enqueue',lambda:observed.append(entry.output['value']))
    q.drain();assert entry.output['value']==new_input and observed==[new_input]
    return list(q.events),getattr(state['context'],'_ascend_sfa_stream_ordered_replay',None)

def main():
    old=setup(False);new=setup(True);counts={'parity':0,'sync_removed':0};examples=[]
    for settings in itertools.product((Mode.NONE,Mode.FULL,Mode.PIECEWISE),(False,True),(False,True),('ASCEND_SFA','ASCEND_MLA'),(False,True),('glm_moe_dsa','other',None),(False,True),(False,True),(False,True)):
        before,_=case(old,settings);after,optin=case(new,settings)
        removed=before.count('host_sync')-after.count('host_sync')
        if removed:
            assert removed==1 and optin is True and settings[0]==Mode.FULL and not settings[6] and not settings[8]
            assert [e for e in before if e!='host_sync']==after
            assert after.index('input_copy_enqueue')<after.index('offloader_fence')<after.index('native_replay_enqueue')<after.index('MTP_consumer_enqueue')
            counts['sync_removed']+=1
            if len(examples)<2:examples.append({'settings':[str(x) for x in settings],'before':before,'after':after})
        else:
            assert before==after,(settings,before,after);counts['parity']+=1
    safe=(Mode.FULL,True,False,'ASCEND_SFA',False,'glm_moe_dsa',False,False,False)
    changes=[(4,True),(1,False),(2,True),(3,'ASCEND_MLA'),(5,'other'),(0,Mode.NONE),(0,Mode.PIECEWISE)]
    for idx,value in changes:
        case(new,safe);unsafe=list(safe);unsafe[idx]=value
        events,flag=case(new,tuple(unsafe));assert flag is False
        baseline,_=case(old,tuple(unsafe));assert events==baseline
    case(new,safe);draft=list(safe);draft[6]=True
    events,flag=case(new,tuple(draft));baseline,_=case(old,tuple(draft))
    assert flag is True and events==baseline and events.count('host_sync')==1
    q,state,extra,wrapper,runner,entry=setup(True);state['context']=NS(cudagraph_runtime_mode=Mode.FULL,capturing=False)
    wrapper._replay(entry,(),{});assert q.events==['host_sync','offloader_fence','native_replay_enqueue'];q.drain()
    for patched in (False,True):
        q,state,extra,wrapper,runner,entry=setup(patched);state['context']=NS(cudagraph_runtime_mode=Mode.FULL,capturing=False)
        result=wrapper._capture(entry,(),{});assert result is entry.output and state['context'].capturing
        assert q.events==['capture_body','weak_workspace_contract','weak_workspace_contract','weak_workspace_contract']
    out={'type':'CPU_ACTUAL_AST_QUEUE_ORDER','cases':sum(counts.values()),'counts':counts,'dynamic_revocation_cases':len(changes),'draft_role_revocation_case':True,'default_missing_property_sync_retained':True,'capture_and_workspace_contract_unchanged':True,'examples':examples,'NPU_initialized':False,'performance_claim':False,'full_API_accepted':False,'limitations':['Native replay and stream operations are CPU queue doubles; real device correctness and gain remain untested.','Native H6 build source calls ExecuteAsync on getCurrentNPUStream; no claim of measured host-barrier duration or largest FULL gap.']}
    (ROOT/'CPU_result.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ('examples','limitations')}))

if __name__=='__main__':main()
