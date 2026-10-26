"""Run494 conditional Target FULL replay frontier. No import-time device effects."""
from __future__ import annotations
import hashlib, inspect, json, os, threading, time, weakref
from contextlib import contextmanager
from pathlib import Path

ENV='EXTREME_RUN494_DIR'
DUMP_ENV='EXTREME_RUN494_DUMP_DIR'
EXPECTED_DEBUG_DUMP_SOURCE='eb11cac181087766e8489cf3a1da0c0fa150933b7e74454dba4a47db6114164e'
LABELS=('T','R0','R1','H','U')
D=None
TLS=threading.local()
CAPTURES={}
CAPTURE_GENERATION=0
EXPECTED_BINDINGS={'graph_call':'91a9f202c5d455dd92a4129113c1830e30073259ebd8c18a3b49775d848fd8c7',
    'current_stream':'608b5984180706f4179c7d6427242ec9163b840b5d4fb56459f92d359c3a84b0',
    'event':'241741b356db5006797dbec119007afb33b73b0e1753bfb06ab27c989c5bfac4',
    'graph_replay':'eb11cac181087766e8489cf3a1da0c0fa150933b7e74454dba4a47db6114164e'}
EXPECTED_CANN={'libascendcl.so':'123d67c313f743e5d6f2e856f40c24d5edac376975cf0a7a1019e079b4171480',
               'libruntime.so':'7bc2b610dff552df2f114c95701156b4aba43ffcba24e06e3c249e1cf57bbc21'}

def loaded_library(name,expected):
    paths={line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
           if line.split() and line.split()[-1].endswith('/'+name)}
    gate(len(paths)==1,'loaded '+name+' identity missing/ambiguous')
    path=next(iter(paths)); h=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    gate(h==expected,'loaded '+name+' hash')
    return dict(path=path,sha256=h)


def gate(ok, why):
    if not ok: raise RuntimeError('Run494 OUT_OF_SCOPE: '+why)


def desc(t):
    s=t.untyped_storage()
    return dict(data=t.data_ptr(),storage=s.data_ptr(),storage_bytes=s.nbytes(),
                offset=t.storage_offset(),shape=list(t.shape),stride=list(t.stride()),
                dtype=str(t.dtype),device=str(t.device))


def tree(x):
    import torch
    if isinstance(x,torch.Tensor): return dict(kind='tensor',value=desc(x))
    if isinstance(x,(list,tuple)):
        return dict(kind='tuple' if isinstance(x,tuple) else 'list',children=[tree(v) for v in x])
    raise RuntimeError('Run494 OUT_OF_SCOPE: unsupported graph output tree')


def identity(t):
    import torch
    stream=torch.npu.current_stream(t.device)
    try:
        sid=dict(stream_id=stream.stream_id,device_index=stream.device_index,
                 device_type=stream.device_type)
    except (AttributeError,TypeError,ValueError) as exc:
        raise RuntimeError('Run494 OUT_OF_SCOPE: immutable stream identity missing') from exc
    gate(str(t.device).startswith('npu:') and str(t.device)[4:].isdigit(),
         'tensor NPU device identity')
    gate(all(type(sid[k]) is int for k in ('stream_id','device_index','device_type')) and
         sid['stream_id']>=0 and sid['device_index']==int(str(t.device)[4:]) and
         sid['device_type']==20, 'stream/tensor device mismatch')
    return (str(t.device),sid),stream


def same_stream(t):
    ident,_=identity(t)
    gate(D is not None and ident==D['stream'],'caller stream switched')
    return ident


def mark(label,t):
    gate(D is not None and D['selected'] and getattr(TLS,'selected',False),'marker outside selected Target')
    gate(label not in D['markers'],'duplicate '+label)
    ident,stream=identity(t)
    gate(ident==D['stream'],'marker stream switched')
    D['events'][label].record(stream)
    D['markers'][label]=dict(event=id(D['events'][label]),device=ident[0],stream=ident[1],
                             tensor=desc(t),host_ns=time.monotonic_ns())


@contextmanager
def target_scope(state):
    if not os.getenv(ENV):
        yield
        return
    # Ordinary Target scope is tracked even on unselected warmup cycles so the
    # first successful FULL capture can be associated with its real producer.
    old=getattr(TLS,'target_scope',False)
    gate(not old,'nested Target scope')
    TLS.target_scope=True
    try: yield
    finally: TLS.target_scope=old


def cache_miss(wrapper,entry,forward_context):
    if not os.getenv(ENV): return
    if D is not None and D['selected'] and getattr(TLS,'selected',False):
        gate(False,'selected Target graph cache miss')


def _binding(x):
    path=inspect.getsourcefile(x)
    gate(path is not None and Path(path).is_file(),'Python binding source unavailable')
    return dict(module=getattr(x,'__module__',None),
                qualname=getattr(x,'__qualname__',None),
                path=path,sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())


def _argument(x):
    import torch
    if type(x) is torch.Tensor:return dict(kind='tensor',value=desc(x))
    if x is None or type(x) in (bool,int,float,str):return dict(kind='scalar',type=type(x).__name__,value=x)
    if type(x) in (tuple,list):return dict(kind=type(x).__name__,children=[_argument(v) for v in x])
    if type(x) is dict:return dict(kind='dict',items={str(k):_argument(v) for k,v in sorted(x.items())})
    raise RuntimeError('Run494 OUT_OF_SCOPE: unsupported graph argument')


def capture_family(wrapper,entry,args,kwargs,forward_context,is_draft_model):
    """Source-pinned ordinary main-model FULL decode96 capture family."""
    import torch
    if wrapper.runtime_mode.name!='FULL' or wrapper.enable_enpu or is_draft_model:
        return False
    if (getattr(forward_context,'is_draft_model',False) or
        forward_context.cudagraph_runtime_mode!=wrapper.runtime_mode):
        return False
    batch=forward_context.batch_descriptor
    if (batch is None or batch!=entry.batch_descriptor or
        type(batch.num_tokens) is not int or batch.num_tokens!=96 or
        type(batch.num_reqs) is not int or batch.num_reqs!=12 or
        batch.uniform is not True or batch.has_lora is not False or
        batch.num_active_loras!=0 or forward_context.num_tokens!=96 or
        forward_context.model_instance is None):
        return False
    if args or not {'input_ids','positions'}.issubset(kwargs):return False
    input_ids,positions=kwargs['input_ids'],kwargs['positions']
    if type(input_ids) is not torch.Tensor or type(positions) is not torch.Tensor:
        return False
    if tuple(input_ids.shape)!=(96,) or tuple(positions.shape)!=(96,):return False
    context_input=getattr(forward_context,'input_ids',None)
    if type(context_input) is not torch.Tensor or desc(context_input)!=desc(input_ids):
        return False
    return True


def capture(wrapper,entry,graph,output,args,kwargs,forward_context,is_draft_model):
    """Post-success startup/Runtime FULL decode96 registration; no device events."""
    global CAPTURE_GENERATION
    if not os.getenv(ENV):return
    if not capture_family(wrapper,entry,args,kwargs,forward_context,is_draft_model):return
    gate(D is None or not D['selected'],'selected cycle hit graph capture')
    try:
        output_tree=tree(output)
        arguments=_argument(dict(args=args,kwargs=kwargs))
    except RuntimeError:
        # A same-shape unrelated wrapper is left entirely uninstrumented.
        # The actual Runtime Target must still find one valid owner at start.
        return
    CAPTURE_GENERATION+=1
    CAPTURES[id(entry)]=dict(ref=weakref.ref(entry),generation=CAPTURE_GENERATION,
                             graph_id=id(graph),output=output_tree,arguments=arguments,
                             capture_scope=dict(target=False,runtime_mode='FULL',
                                 enable_enpu=False,is_draft_model=False,
                                 use_eagle=bool(wrapper.use_eagle),
                                 batch_descriptor=str(entry.batch_descriptor),
                                 forward_model_id=id(forward_context.model_instance),
                                 origin=('runtime_target' if getattr(TLS,'target_scope',False)
                                         else 'startup_full_decode96'),
                                 batch_num_tokens=96,batch_num_reqs=12,uniform=True),
                             selected_observations=0)


def _callable_target(obj):
    """Stable identity for bound methods; no source read or method invocation."""
    fn=getattr(obj,'__func__',obj)
    owner=getattr(obj,'__self__',None)
    return fn,dict(callable_id=id(fn),owner_id=(id(owner) if owner is not None else None))


def _backend_from_update(handoff):
    fn=handoff.graph_update
    cells=dict(zip(fn.__code__.co_freevars,(c.cell_contents for c in fn.__closure__ or ())))
    backend=cells.get('attn_backend')
    gate(backend is not None,'graph update attention backend closure missing')
    return backend


def graph_update_identity(handoff):
    """Cheap immutable identity only; safe for the selected submission path."""
    fn=getattr(handoff,'graph_update',None)
    if fn is None:return dict(branch='none',callable_id=None,
                            configured_update_stream=None,
                            private_stream_relation='unresolved')
    freevars=dict(zip(fn.__code__.co_freevars,(c.cell_contents for c in fn.__closure__ or ())))
    update_stream=freevars.get('update_stream')
    gate(update_stream is not None,'graph update stream closure missing')
    sid=dict(stream_id=update_stream.stream_id,
             device_index=update_stream.device_index,
             device_type=update_stream.device_type)
    gate(all(type(v) is int for v in sid.values()),
         'graph update stream immutable fields')
    return dict(branch='before' if handoff.graph_update_before else 'after',
                callable_id=id(fn),configured_update_stream=sid,
                private_stream_relation='unresolved')


def graph_update_contract(handoff):
    """Setup-only source certificate; never called from the selected path."""
    identity=graph_update_identity(handoff)
    identity['callable']=(_binding(handoff.graph_update)
                          if identity['callable_id'] is not None else None)
    if identity['callable_id'] is not None:
        backend=_backend_from_update(handoff)
        getter_fn,getter_ref=_callable_target(backend.get_impl_cls)
        identity['backend']=dict(object_id=id(backend),source=_binding(
            backend if inspect.isclass(backend) else type(backend)))
        identity['get_impl_cls']=dict(**getter_ref,source=_binding(getter_fn))
    else:
        identity['backend']=None;identity['get_impl_cls']=None
    return identity


def start(serving):
    global D
    gate(D is None,'prior cohort not exported')
    if not os.getenv(ENV): return
    import torch
    from scripts.loop079_logits_join_run472 import source_contract, _lib_from_maps
    from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
    import torch.distributed.distributed_c10d as c10d
    source_pins=dict(c10d=source_contract(),torch_npu=_lib_from_maps(),
                     cann={name:loaded_library(name,digest) for name,digest in EXPECTED_CANN.items()})
    source_pins['c10d_callable_id']=id(c10d.all_gather_into_tensor)
    source_pins['graph_debug_dump']=_binding(torch.npu.NPUGraph.debug_dump)
    source_pins['graph_debug_dump_callable_id']=id(torch.npu.NPUGraph.debug_dump)
    gate(source_pins['graph_debug_dump']['sha256']==EXPECTED_DEBUG_DUMP_SOURCE,
         'installed graph debug_dump Python binding hash')
    source_pins['bindings']={name:_binding(obj) for name,obj in
        [('graph_call',ACLGraphWrapper.__call__),('current_stream',torch.npu.current_stream),
         ('event',torch.npu.Event),('graph_replay',torch.npu.NPUGraph.replay)]}
    gate(all(source_pins['bindings'][name]['sha256']==digest for name,digest in
             EXPECTED_BINDINGS.items()),'installed graph/stream/event Python binding hashes')
    rt=serving.runtime
    gate(not any((rt._profile_dag,rt._diagnose,rt._cycle_profiler,rt._cycle_profile_dir,
                  rt._profile_scopes,rt.kv_slot_audit is not None,
                  rt.target_page_audit is not None,rt._schedule_verify)),
         'competing profiler/audit')
    competing_env={name:os.getenv(name) for name in
        ('EXTREME_COUNT_MARKER_DIR','EXTREME_BOUND_EVENT_DIR',
         'EXTREME_BOUND_ROW_CAPTURE_DIR','EXTREME_BOUND_TOKEN_CAPTURE_DIR',
         'EXTREME_RUN472_DIR','EXTREME_RUN439_DIR',
         'EXTREME_RUNTIME_TARGET_DIAGNOSTIC_REFRESH') if os.getenv(name)}
    gate(not competing_env,'competing capture environment')
    gate(rt._schedule_mode=='off','fixed schedule mode')
    handoff=getattr(rt.target.binding.forward,'__self__',None)
    flags=dict(schedule_mode=rt._schedule_mode,profile_dag=bool(rt._profile_dag),
       diagnose=bool(rt._diagnose),cycle_profiler=bool(rt._cycle_profiler),
       cycle_profile_dir=rt._cycle_profile_dir,profile_scopes=bool(rt._profile_scopes),
       kv_slot_audit=rt.kv_slot_audit is not None,
       target_page_audit=rt.target_page_audit is not None,
       schedule_verify=bool(rt._schedule_verify),competing_capture=bool(competing_env),
       competing_capture_env=competing_env,
       diagnostic_refresh=bool(getattr(handoff,'diagnostic_metadata_refresh',None)))
    gate(not flags['diagnostic_refresh'],'target diagnostic refresh')
    bound_graph_update=graph_update_contract(handoff)
    gate(handoff is not None and getattr(handoff,'model',None) is not None and
         getattr(handoff,'batch_descriptor',None) is not None,'bound Target model missing')
    expected_model_id=id(handoff.model)
    expected_batch=str(handoff.batch_descriptor)
    matching=[cap for cap in CAPTURES.values() if cap['ref']() is not None and
              cap['capture_scope']['forward_model_id']==expected_model_id and
              cap['capture_scope']['batch_descriptor']==expected_batch]
    gate(len(matching)==1,'startup FULL decode96 graph capture owner missing/ambiguous')
    matching[0]['capture_scope']['target']=True
    matching[0]['capture_scope']['bound_runtime_target_model_id']=expected_model_id
    matching[0]['capture_scope']['bound_runtime_target_batch']=expected_batch
    gate(serving.config.batch_size==12 and serving.config.target_token_count==96 and
         serving.config.target_tokens_per_request==8 and len(serving.initial_output_counts)==12 and
         len(serving.remaining)==12 and
         all(a+b==1024 for a,b in zip(serving.initial_output_counts,serving.remaining)),
         'frozen serving shape/output contract')
    e={k:torch.npu.Event(enable_timing=True) for k in LABELS}
    gate(len({id(x) for x in e.values()})==5,'event identity reused')
    stream_id,stream=identity(rt.state.accepted_tokens)
    for x in e.values(): x.record(stream); x.synchronize()  # setup only, before cycle0
    rank=torch.distributed.get_rank(); out=Path(os.environ[ENV]); gate(out.is_dir(),'output directory')
    cohort=1+len(list(out.glob(f'rank{rank}_cohort*.json')))
    gate(1<=cohort<=5,'cohort count')
    D=dict(run_id=os.environ['EXTREME_RUN494_RUN_ID'],rank=rank,pid=os.getpid(),
           cohort=cohort,phase='warmup' if cohort<=4 else 'diagnostic',
           start_cycle=rt.state.cycle_index,selected=False,seen=False,serving=serving,
           stream=stream_id,events=e,markers={},sync=None,replay=None,
           graph_output=None,pre_gather=None,post_gather=None,hidden=[],hidden_active=None,
           native_calls=[],graph_update=[],actual_update=None,_actual_update_objects=None,runtime_flags=flags,
           config=dict(batch_size=serving.config.batch_size,
               target_tokens_per_request=serving.config.target_tokens_per_request,
               target_token_count=serving.config.target_token_count,
               max_output_tokens=serving.initial_output_counts[0]+serving.remaining[0],
               max_cycles=serving.max_cycles),
           expected_model_id=expected_model_id,expected_batch=expected_batch,
           setup=dict(stream=stream_id,events_initialized=5,
                      completion='setup-only event.synchronize',source_pins=source_pins,
                      graph_update_bound=bound_graph_update))


def step(rt):
    if D is None:return
    entry=rt.state.cycle_index-D['start_cycle']; D['selected']=entry==64
    if entry==64:
        gate(not D['seen'] and not any(D['serving']._parked),'cycle64 repeated/parked')
        D['seen']=True;D['cycle_index']=rt.state.cycle_index
        D['active_mask']=desc(rt.state.active_mask)


def target_begin(state):
    if D is None or not D['selected']:return
    gate(not getattr(TLS,'selected',False),'nested Target')
    TLS.selected=True
    D['input']=desc(state.target_input_ids);D['positions']=desc(state.target_positions)
    D['indices']=desc(state.target_logits_indices)
    mark('T',state.target_input_ids)


def sync_begin(wrapper,entry,is_draft_eagle,need_sync):
    if D is None or not D['selected'] or not getattr(TLS,'selected',False):return
    gate(D['sync'] is None and wrapper.runtime_mode.name=='FULL' and
         not wrapper.enable_enpu and not is_draft_eagle and need_sync,'FULL sync branch')
    D['sync']=dict(begin_ns=time.monotonic_ns(),end_ns=None,
                   entry_id=id(entry),stream=same_stream_tensorless(),
                   original_branch=dict(runtime_mode=wrapper.runtime_mode.name,
                       enable_enpu=bool(wrapper.enable_enpu),
                       is_draft_eagle=bool(is_draft_eagle),need_sync=bool(need_sync)))


def same_stream_tensorless():
    import torch
    # Query current stream object and immutable identity without a tensor.
    device=D['stream'][0]
    stream=torch.npu.current_stream(device)
    sid=dict(stream_id=stream.stream_id,device_index=stream.device_index,
             device_type=stream.device_type)
    gate(all(type(sid[k]) is int for k in ('stream_id','device_index','device_type')),
         'immutable stream fields')
    gate((device,sid)==D['stream'],'caller stream switched')
    return sid


def sync_end():
    if D is None or not D['selected'] or not getattr(TLS,'selected',False):return
    gate(D['sync'] is not None and D['sync']['end_ns'] is None,'sync end without begin')
    same_stream_tensorless();D['sync']['end_ns']=time.monotonic_ns()


def replay_pre(wrapper,entry,args,kwargs,is_draft_eagle,need_sync):
    if D is None or not D['selected'] or not getattr(TLS,'selected',False):return
    gate('T' in D['markers'] and D['sync'] is not None and D['sync']['end_ns'] is not None,
         'replay without existing sync')
    gate(D['replay'] is None and entry.aclgraph is not None,'extra/missing graph replay')
    cap=CAPTURES.get(id(entry))
    gate(cap is not None and cap['ref']() is entry and cap['graph_id']==id(entry.aclgraph),
         'capture generation/graph identity')
    gate(cap['arguments']==_argument(dict(args=args,kwargs=kwargs)),
         'graph argument generation/storage')
    inp={k:desc(kwargs[k]) for k in ('input_ids','positions')}
    gate(inp['input_ids']==D['input'] and inp['positions']==D['positions'],
         'graph input generation/storage')
    branch=dict(runtime_mode=wrapper.runtime_mode.name,
                enable_enpu=bool(wrapper.enable_enpu),
                is_draft_eagle=bool(is_draft_eagle),need_sync=bool(need_sync))
    gate(branch==D['sync']['original_branch'] and branch==dict(
         runtime_mode='FULL',enable_enpu=False,is_draft_eagle=False,need_sync=True),
         'selected original replay branch')
    gate(cap['output']==tree(entry.output),'graph output owner changed')
    gate(cap['capture_scope']['target'] and
         str(entry.batch_descriptor)==cap['capture_scope']['batch_descriptor'] and
         cap['capture_scope'].get('bound_runtime_target_model_id')==D['expected_model_id'] and
         cap['capture_scope'].get('bound_runtime_target_batch')==D['expected_batch'] and
         cap['capture_scope']['runtime_mode']=='FULL'
         and not cap['capture_scope']['is_draft_model'] and
         cap['capture_scope']['forward_model_id']==D['expected_model_id'] and
         cap['capture_scope']['batch_descriptor']==D['expected_batch'],
         'capture Target FULL model/descriptor owner')
    D['replay']=dict(entry_id=id(entry),graph_id=cap['graph_id'],
                     capture_generation=cap['generation'],
                     selected_observation_ordinal=cap['selected_observations']+1,
                     original_branch=branch,capture_scope=cap['capture_scope'],
                     batch_descriptor=str(entry.batch_descriptor),
                     output_owner=cap['output'],arguments=cap['arguments'],
                     completion='conditional: caller stream/child join/source contract')
    cap['selected_observations']+=1
    mark('R0',kwargs['input_ids'])


def replay_post(entry):
    if D is None or not D['selected'] or not getattr(TLS,'selected',False):return
    gate(D['replay'] is not None and D['replay']['entry_id']==id(entry),'replay return identity')
    gate(tree(entry.output)==D['replay']['output_owner'],'replay output owner')
    D['graph_output']=tree(entry.output)
    mark('R1',D['serving'].runtime.state.target_input_ids)


def update_begin(handoff,phase):
    if D is None or not D['selected'] or not getattr(TLS,'selected',False):return
    gate(handoff.graph_update is not None and
         phase==('before' if handoff.graph_update_before else 'after'),
         'graph update branch')
    observed=graph_update_identity(handoff)
    bound=D['setup']['graph_update_bound']
    gate(observed['branch']==phase and all(
         observed[k]==bound[k] for k in ('branch','callable_id',
             'configured_update_stream','private_stream_relation')),
         'graph update callable changed from setup')
    D['graph_update'].append(dict(phase=phase,begin_ns=time.monotonic_ns(),end_ns=None,
        callable_id=bound['callable_id'],callable=bound['callable'],
        caller_stream=same_stream_tensorless(),
        configured_update_stream=bound['configured_update_stream'],
        private_stream_relation='unresolved'))


def update_end():
    if D is None or not D['selected'] or not getattr(TLS,'selected',False):return
    gate(D['graph_update'] and D['graph_update'][-1]['end_ns'] is None,'update end')
    D['graph_update'][-1]['end_ns']=time.monotonic_ns()


def actual_update_begin(attn_backend,impl_cls,update_callable,update_stream):
    """Observe the existing resolved FULL update call, selected path only."""
    if D is None or not D['selected'] or not getattr(TLS,'selected',False):return
    gate(D['graph_update'] and D['graph_update'][-1]['end_ns'] is None and
         D['actual_update'] is None,'selected FULL update placement/count')
    handoff=getattr(D['serving'].runtime.target.binding.forward,'__self__',None)
    bound=D['setup']['graph_update_bound']
    gate(handoff is not None and _backend_from_update(handoff) is attn_backend and
         id(attn_backend)==bound['backend']['object_id'],'selected actual backend changed')
    update_fn,update_ref=_callable_target(update_callable)
    gate(callable(update_fn),'selected implementation update callable missing')
    sid=dict(stream_id=update_stream.stream_id,device_index=update_stream.device_index,
             device_type=update_stream.device_type)
    gate(sid==bound['configured_update_stream'],'selected update stream changed')
    D['actual_update']=dict(backend_id=id(attn_backend),impl_id=id(impl_cls),
        update_callable=update_ref,configured_update_stream=sid,
        branch=D['graph_update'][-1]['phase'],return_count=0)
    D['_actual_update_objects']=(attn_backend,impl_cls,update_fn)


def actual_update_end():
    if D is None or not D['selected'] or not getattr(TLS,'selected',False):return
    gate(D['actual_update'] is not None and D['actual_update']['return_count']==0,
         'selected update return missing/duplicate')
    D['actual_update']['return_count']=1


def gather_begin(output,context,handoff):
    if D is None or not D['selected'] or not getattr(TLS,'selected',False):return
    gate(D['graph_output'] is not None and D['pre_gather'] is None,'gather without graph output')
    D['pre_gather']=tree(output)
    gate(D['pre_gather']==D['graph_output'],'graph output→handoff ownership')
    D['graph_update_branch']=('before' if handoff.graph_update_before else 'after') if handoff.graph_update is not None else 'none'
    gate(len(D['graph_update'])==(0 if D['graph_update_branch']=='none' else 1) and
         all(row['phase']==D['graph_update_branch'] and row['end_ns'] is not None
             for row in D['graph_update']), 'graph update call count/branch')
    D['flash_comm']=bool(context.flash_comm_v1_enabled)
    gate(D['flash_comm'], 'hidden/aux gather branch disabled')
    D['pad_size']=int(context.pad_size);D['num_tokens']=int(context.num_tokens)
    D['padded']=int(getattr(context,'padded_length',context.num_tokens+context.pad_size))


def hidden_begin(hidden):
    if D is None or not D['selected'] or not getattr(TLS,'selected',False):return
    gate(D['flash_comm'] and D['hidden_active'] is None,'unexpected hidden gather')
    branch=tuple(hidden.shape)[0]!=D['padded']
    gate(branch,'hidden leaf bypassed native gather')
    row=dict(input=desc(hidden),gather=branch,native_calls_before=len(D['native_calls']),
             stream=same_stream(hidden))
    D['hidden_active']=row


def native_pre(comm,out,inp):
    if D is None or not D['selected'] or D['hidden_active'] is None:return
    import torch,torch.distributed as dist
    from torch.overrides import has_torch_function
    from scripts.loop079_logits_join_run472 import _dispatch_gate
    from vllm.distributed.parallel_state import get_tp_group
    group=get_tp_group(); row=D['hidden_active']
    gate(row['gather'] and group.world_size==8 and list(group.ranks)==list(range(8)) and
         comm.device_group is group.device_group and dist.get_backend(comm.device_group)=='hccl',
         'hidden native TP/HCCL group')
    gate(not torch._dynamo.is_compiling() and not torch.npu.is_current_stream_capturing(),
         'compile/graph capture substitution')
    dispatch_gate=_dispatch_gate()
    gate(type(inp) is torch.Tensor and type(out) is torch.Tensor and
         not has_torch_function((inp,out)) and desc(inp)==row['input'],
         'hidden native input/override')
    in_d,out_d=desc(inp),desc(out)
    gate(len(in_d['shape'])==len(out_d['shape']) and
         out_d['shape']==[8*in_d['shape'][0]]+in_d['shape'][1:] and
         in_d['dtype']==out_d['dtype'] and in_d['device']==out_d['device'],
         'hidden native exact concat shape/dtype/device')
    gate(same_stream(inp)==D['stream'] and same_stream(out)==D['stream'],
         'hidden native stream')
    import torch.distributed.distributed_c10d as c10d
    member_rank=dist.get_rank(comm.device_group)
    gate(dist.all_gather_into_tensor is c10d.all_gather_into_tensor and
         id(c10d.all_gather_into_tensor)==D['setup']['source_pins']['c10d_callable_id'] and
         comm.device_group not in c10d._world.pg_coalesce_state and
         member_rank==D['rank'],'hidden native ordinary dispatch')
    gate(len(D['native_calls'])==row['native_calls_before'],'extra hidden native call')
    D['native_calls'].append(dict(input=desc(inp),output=desc(out),return_ns=None,
                                   group=dict(name=group.unique_name,members=list(group.ranks),
                                              member_rank=member_rank,backend='hccl'),
                                   dispatch_gate=dispatch_gate,
                                   dispatch=dict(ordinary=True,async_op=False,coalescing=False,
                                                 capture=False,compiled=False,modes_clean=True),
                                   torch_function_override=False,entry_ns=time.monotonic_ns()))


def native_post(comm,out,inp):
    if D is None or not D['selected'] or D['hidden_active'] is None:return
    row=D['native_calls'][-1]
    gate(row['return_ns'] is None and desc(out)==row['output'] and same_stream(out)==D['stream'],
         'hidden native return')
    row['return_ns']=time.monotonic_ns()


def hidden_end(hidden):
    if D is None or not D['selected'] or not getattr(TLS,'selected',False):return
    row=D['hidden_active'];gate(row is not None,'hidden end without begin')
    n=len(D['native_calls'])-row['native_calls_before']
    gate(n==(1 if row['gather'] else 0),'hidden native call count')
    row['output']=desc(hidden)
    if row['gather']:
        native=D['native_calls'][-1]
        gate(native['return_ns'] is not None and native['input']==row['input'],
             'native gather lineage missing')
        row['native_output_same_storage']=(native['output']['storage']==row['output']['storage'])
        gate(row['native_output_same_storage'],'native→hidden storage lineage copy')
    gate(same_stream(hidden)==D['stream'],'hidden output stream')
    D['hidden'].append(row);D['hidden_active']=None


def gather_end(output):
    if D is None or not D['selected'] or not getattr(TLS,'selected',False):return
    gate(D['pre_gather'] is not None and D['post_gather'] is None,'gather end')
    D['post_gather']=tree(output)
    expected=(1+len(output[1])) if isinstance(output,tuple) else 1
    gate(len(D['hidden'])==expected,'hidden/aux leaf count')
    leaves=([output[0]]+list(output[1])) if isinstance(output,tuple) else [output]
    gate(all(desc(t)==row['output'] for t,row in zip(leaves,D['hidden'])),
         'hidden/aux return ownership')
    mark('H',D['serving'].runtime.state.target_input_ids)


def target_after(hidden,sample,state):
    if D is None or not D['selected']:return
    gate(getattr(TLS,'selected',False) and D['post_gather'] is not None,'sample without handoff')
    result=D['post_gather']
    first=result['children'][0]['value'] if result['kind']=='tuple' else result['value']
    gate(desc(hidden)==first and desc(state.target_logits_indices)==D['indices'],
         'hidden/indices generation')
    D['sample_hidden']=desc(sample)
    mark('U',sample);TLS.selected=False


def finish_step():
    if D is not None and D['selected']:
        gate(tuple(D['markers'])==LABELS and D['hidden_active'] is None and
             not getattr(TLS,'selected',False),'incomplete selected frontier')
        D['selected']=False


def _post_drain_backend(serving,d):
    """Certify the recorded selected call after ordinary cohort drain."""
    handoff=getattr(serving.runtime.target.binding.forward,'__self__',None)
    gate(handoff is not None and getattr(handoff,'graph_update',None) is not None,
         'bound graph update missing at dump')
    bound=d['setup']['graph_update_bound']
    gate(graph_update_identity(handoff)=={k:bound[k] for k in
         ('branch','callable_id','configured_update_stream','private_stream_relation')},
         'post-drain graph update changed')
    actual=d.get('actual_update');objects=d.get('_actual_update_objects')
    gate(actual is not None and actual['return_count']==1 and
         actual['branch']==bound['branch'] and objects is not None and len(objects)==3,
         'selected update not returned exactly once')
    backend,impl,update_fn=objects
    gate(_backend_from_update(handoff) is backend and
         id(backend)==actual['backend_id']==bound['backend']['object_id'] and
         id(impl)==actual['impl_id'] and
         _callable_target(impl.update_graph_params)[0] is update_fn and
         _callable_target(impl.update_graph_params)[1]==actual['update_callable'],
         'post-drain selected backend/implementation changed')
    getter_fn,getter_ref=_callable_target(backend.get_impl_cls)
    gate(getter_ref=={k:bound['get_impl_cls'][k] for k in ('callable_id','owner_id')},
         'post-drain backend getter identity changed')
    backend_cert=dict(object_id=id(backend),source=_binding(
        backend if inspect.isclass(backend) else type(backend)))
    getter_cert=dict(**getter_ref,source=_binding(getter_fn))
    gate(backend_cert==bound['backend'] and getter_cert==bound['get_impl_cls'],
         'post-drain backend source changed')
    return dict(attn_backend=backend_cert,get_impl_cls=getter_cert,
                impl=dict(object_id=id(impl),source=_binding(impl)),
                update_graph_params=dict(**actual['update_callable'],source=_binding(update_fn)),
                selected_actual_update=actual,
                configured_update_stream=bound['configured_update_stream'],
                private_stream_relation='unresolved')


def graph_task_metadata(nodes):
    """Validate observed native task identities; exporter order is not a dependency order."""
    gate(isinstance(nodes,list) and bool(nodes),'graph dump JSON node list')
    ids=set();models=set();streams={}
    supported={'EVENT_RECORD','EVENT_RESET','EVENT_WAIT','NOTIFY_RECORD','NOTIFY_WAIT',
               'MEMCPY','MEMCPY_ASYNC','MEMSET'}
    for node in nodes:
        gate(isinstance(node,dict) and isinstance(node.get('name'),str) and
             bool(node['name'].strip()) and isinstance(node.get('args'),dict),
             'graph task name/args')
        args=node['args'];model=args.get('Model Id');stream=args.get('Stream Id');task=args.get('Task Id')
        kind=args.get('Task Type')
        gate(all(type(x) is int and x>=0 for x in (model,stream,task)) and
             isinstance(kind,str) and bool(kind.strip()),'native graph task fields')
        key=(model,stream,task);gate(key not in ids,'duplicate native task identity');ids.add(key)
        if kind.startswith('KERNEL_'):
            gate(isinstance(args.get('Kernel Args'),str) and bool(args['Kernel Args'].strip()) and
                 type(args.get('Kernel Args Size')) is int and args['Kernel Args Size']>0 and
                 type(args.get('Numblocks')) is int and args['Numblocks']>0 and
                 type(args.get('Schem Mode')) is int and args['Schem Mode']>=0,
                 'kernel native arguments missing')
        else:gate(kind in supported,'unsupported native task type')
        models.add(model);streams[stream]=streams.get(stream,0)+1
    gate(len(models)==1 and bool(streams),'single native model/nonempty streams')
    return dict(native_model_ids=sorted(models),task_count=len(nodes),
                stream_ids=sorted(str(x) for x in streams),
                stream_task_counts={str(k):v for k,v in sorted(streams.items())})


def post_drain_dump(d,serving):
    """Optional cohort5 graph dump after the existing counts_cpu Host drain."""
    requested=os.getenv(DUMP_ENV)
    if not requested or d['cohort']!=5:return None
    out=Path(requested);gate(out.is_dir(),'graph dump directory absent')
    replay=d['replay'];cap=CAPTURES.get(replay['entry_id'])
    gate(cap is not None and cap['ref']() is not None,'selected capture entry expired')
    entry=cap['ref']()
    gate(id(entry)==replay['entry_id'] and cap['generation']==replay['capture_generation']
         and cap['graph_id']==replay['graph_id'] and
         id(entry.aclgraph)==replay['graph_id'] and
         str(entry.batch_descriptor)==d['expected_batch'] and
         cap['capture_scope']['target'] and
         cap['capture_scope']['forward_model_id']==d['expected_model_id'] and
         cap['capture_scope']['batch_descriptor']==d['expected_batch'] and
         cap['output']==replay['output_owner']==d['graph_output']==tree(entry.output),
         'selected graph entry/generation/owner/output changed before dump')
    handoff=getattr(serving.runtime.target.binding.forward,'__self__',None)
    gate(handoff is not None and id(handoff.model)==d['expected_model_id'] and
         str(handoff.batch_descriptor)==d['expected_batch'],
         'Runtime Target model/batch changed before dump')
    backend=_post_drain_backend(serving,d)
    graph=entry.aclgraph
    debug_bound=getattr(graph,'debug_dump',None)
    gate(callable(debug_bound),'installed graph debug_dump missing')
    debug_fn,_=_callable_target(debug_bound)
    gate(getattr(debug_bound,'__self__',None) is graph and
         id(debug_fn)==d['setup']['source_pins']['graph_debug_dump_callable_id'] and
         _binding(debug_fn)==d['setup']['source_pins']['graph_debug_dump'],
         'bound graph debug_dump implementation changed')
    dump=out/f"rank{d['rank']}_cohort5_acl_graph.json"
    meta_path=out/f"rank{d['rank']}_cohort5_acl_graph.meta.json"
    gate(not dump.exists() and not meta_path.exists(),'graph dump already exists')
    debug_bound(str(dump))
    gate(dump.is_file() and dump.stat().st_size>0,'empty/missing graph dump')
    raw=dump.read_bytes();nodes=json.loads(raw)
    task_meta=graph_task_metadata(nodes)
    meta=dict(run_id=d['run_id'],rank=d['rank'],cohort=5,
              path=str(dump),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),
              node_count=len(nodes),
              stream_ids=task_meta['stream_ids'],native_model_ids=task_meta['native_model_ids'],
              task_count=task_meta['task_count'],stream_task_counts=task_meta['stream_task_counts'],
              entry_id=replay['entry_id'],capture_generation=replay['capture_generation'],
              graph_id=replay['graph_id'],model_id=d['expected_model_id'],
              batch_descriptor=d['expected_batch'],
              output_owner_sha256=hashlib.sha256(json.dumps(d['graph_output'],sort_keys=True).encode()).hexdigest(),
              debug_dump_binding=d['setup']['source_pins']['graph_debug_dump'],
              debug_dump_callable_id=d['setup']['source_pins']['graph_debug_dump_callable_id'],
              debug_dump_owner_id=id(graph),
              graph_update_backend=backend,
              completion='post ordinary counts_cpu drain; graph state after further replays, not cycle64 dynamic parameters')
    with meta_path.open('x') as f:json.dump(meta,f,indent=2)
    return dict(path=str(dump),meta_path=str(meta_path),sha256=meta['sha256'],
                bytes=meta['bytes'],node_count=meta['node_count'])


def export(serving,counts_cpu,cycles):
    global D
    if D is None:return
    d=D;gate(d['seen'] and tuple(d['markers'])==LABELS and cycles>64,'incomplete c64 frontier')
    intervals={a+'_'+b+'_ms':d['events'][a].elapsed_time(d['events'][b])
               for a,b in zip(LABELS,LABELS[1:])}  # post ordinary drain only
    dump_meta=post_drain_dump(d,serving)
    out=Path(os.environ[ENV])/f"rank{d['rank']}_cohort{d['cohort']}.json"
    payload={k:d[k] for k in ('run_id','rank','pid','cohort','phase','start_cycle','cycle_index',
            'stream','markers','sync','replay','graph_output','pre_gather','post_gather',
            'hidden','native_calls','graph_update','actual_update','graph_update_branch','config','flash_comm','pad_size','num_tokens','padded',
            'input','positions','indices','sample_hidden','active_mask','runtime_flags','setup')}
    payload.update(schema=1,cycles=cycles,intervals=intervals,
                   accepted_counts=counts_cpu.tolist(),
                   initial_output_counts=list(serving.initial_output_counts),
                   remaining=list(serving.remaining),
                   request_ids_available=False,request_ids=None,
                   debug_dump=dump_meta,
                   scope='conditional native replay/output join; instrumented local frontier only')
    with out.open('x') as f:json.dump(payload,f)
    D=None
