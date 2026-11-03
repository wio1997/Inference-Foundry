"""Opt-in diagnostic adapters for exact pinned modules; never installed by import.

Torch metadata is supplementary. No .item/.cpu/.numpy/.tolist, device sync,
extra graph execution, runtime getter or pointer dereference is introduced.
"""
import functools,hashlib,inspect,os,weakref
from pathlib import Path
from contextlib import nullcontext
from evidence_core import safe_host_value

PINNED={
 'vllm_ascend.compilation.acl_graph':'8e1c269e331d005f1361acc5cec24402bff7ae39b75e952ea890da7812804955',
 'vllm_ascend.compilation.updatable_graph':'d41d912ce8f4183228fdc182dbdb6a504f07642fddb8872c657e7f4429ac0729',
 'vllm_ascend.worker.model_runner_v1':'568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c',
 'vllm_ascend.profiler.torch_npu_profiler':'5242ee153033afcf12dbfe022dbd807fa248d991ff5c53efbd1adb372d5575f4'}

class TargetedObserver:
    def __init__(self,sink,torch_module,mode_base,enable_dispatch=False):
        self.sink=sink;self.torch=torch_module;self.epochs={};self.capture_states={};self.tokens={};self.next_token=0
        self.originals=[];self.mode_base=mode_base;self.graph_module=None
        self.enable_dispatch=enable_dispatch

    def observe(self,callback):
        try:return callback()
        except Exception as e:
            self.sink.emit('supplementary_metadata_error',error_type=type(e).__name__,
                metadata_is_unknown=True,original_call_result_preserved=True)
            return None

    def token(self,obj):
        # Weak identity for Python graph objects, not native handles or allocator epochs.
        oid=id(obj);old=self.tokens.get(oid)
        if old and old[0]() is obj:return old[1]
        self.next_token+=1;tag=self.next_token
        try:self.tokens[oid]=(weakref.ref(obj),tag)
        except TypeError:return {'unknown':'Python object cannot be weakly tracked','observed_object_id':oid}
        return tag

    def graph_identity(self,obj):
        token=self.token(obj)
        return {'graph_python_token':token,'capture_epoch':self.epochs.get(token) if isinstance(token,int) else None,
            'capture_boundary_state':self.capture_states.get(token) if isinstance(token,int) else 'UNKNOWN',
            'native_modelId':'UNKNOWN','native_context':'UNKNOWN'}

    def extra(self,t):
        if self.sink.exhausted:return {'unknown':'supplementary metadata budget exhausted'}
        result={'TensorImpl':None,'StorageImpl':None,'storage_offset':None,'storage_nbytes':None,
            'view_data_ptr':None,'version_counter':None,
            'allocation_generation':'UNKNOWN','actual_NPU_format':'UNKNOWN',
            'version_is_not_GPU_completion_or_complete_update_epoch':True}
        def field(name,fn):
            try:result[name]=fn()
            except Exception as e:result[name+'_unknown_reason']=type(e).__name__
        field('TensorImpl',lambda:int(t._cdata))
        storage=None
        try:storage=t.untyped_storage()
        except Exception as e:result['storage_unknown_reason']=type(e).__name__
        if storage is not None:
            field('StorageImpl',lambda:int(storage._cdata))
            field('storage_nbytes',lambda:int(storage.nbytes()))
        field('storage_offset',lambda:int(t.storage_offset()))
        field('view_data_ptr',lambda:int(t.data_ptr()))
        field('version_counter',lambda:int(t._version))
        return result

    def encode(self,value):
        return safe_host_value(value,lambda t:isinstance(t,self.torch.Tensor),self.extra)

    def identity(self):
        graph=self.graph_module._ACTIVE_GRAPH.get() if self.graph_module else None
        return self.graph_identity(graph) if graph is not None else {'graph_python_token':None,'capture_epoch':None}

    def _wrap(self,cls,name,build):
        old=getattr(cls,name)
        if getattr(old,'_wio_targeted_evidence',False):raise RuntimeError('Adapter already installed')
        new=functools.wraps(old)(build(old));new._wio_targeted_evidence=True
        # Preserve inherited methods when uninstalling.
        self.originals.append((cls,name,cls.__dict__.get(name),name in cls.__dict__))
        setattr(cls,name,new)

    def uninstall(self):
        for cls,name,old,owned in reversed(self.originals):
            if owned:setattr(cls,name,old)
            else:delattr(cls,name)
        self.originals.clear()

    def verify_source(self,module):
        want=PINNED[module.__name__];path=Path(module.__file__)
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest!=want:raise RuntimeError('Targeted evidence source mismatch: '+module.__name__)
        self.sink.emit('module_source_bound',module=module.__name__,path=str(path),sha256=digest)

    def install_graph(self,module):
        self.verify_source(module);self.graph_module=module;owner=self;cls=module.UpdatableGraph
        def capture_begin(old):
            def call(obj,*a,**k):
                tag=owner.token(obj)
                with owner.sink.scope('capture_begin',**owner.graph_identity(obj)):
                    try:result=old(obj,*a,**k)
                    except BaseException:
                        if isinstance(tag,int):owner.capture_states[tag]='CAPTURE_BEGIN_RAISED'
                        raise
                    if isinstance(tag,int):
                        owner.epochs[tag]=owner.epochs.get(tag,0)+1
                        owner.capture_states[tag]='CAPTURE_BEGIN_RETURNED'
                    owner.sink.emit('capture_begin_return',identity=owner.graph_identity(obj),
                        successful_Python_boundary_is_not_native_task_or_binary_attestation=True)
                    return result
            return call
        self._wrap(cls,'capture_begin',capture_begin)
        for name in ['capture_end','replay']:
            def boundary(old,label=name):
                def call(obj,*a,**k):
                    with owner.sink.scope(label,**owner.graph_identity(obj)):
                        try:result=old(obj,*a,**k)
                        except BaseException:
                            if label=='capture_end':
                                tag=owner.token(obj)
                                if isinstance(tag,int):owner.capture_states[tag]='CAPTURE_END_RAISED'
                            raise
                        if label=='capture_end':
                            tag=owner.token(obj)
                            if isinstance(tag,int):owner.capture_states[tag]='CAPTURE_END_RETURNED'
                            owner.sink.emit('capture_end_return',identity=owner.graph_identity(obj),
                                is_not_all_device_reader_completion=True)
                        return result
                return call
            self._wrap(cls,name,boundary)
        def resource(old):
            def call(obj,key,factory,*a,**k):
                # Never call the factory, do an extra cache lookup, or retain its Tensor.
                result=old(obj,key,factory,*a,**k)
                owner.sink.emit('capture_resource_return',identity=owner.graph_identity(obj),
                    key=owner.encode(key),result=owner.encode(result),
                    use_max_workspace=owner.encode(k.get('use_max_workspace',a[0] if a else False)),
                    return_is_not_lifetime_or_ownership_proof=True)
                return result
            return call
        self._wrap(cls,'get_capture_resource',resource)
        def register(old):
            def call(obj,operation,kwargs,provider,*a,**k):
                before=len(obj.tasks)
                with owner.sink.scope('register_update_task',**owner.graph_identity(obj)):
                    owner.sink.emit('registration_inputs',kwargs=owner.encode(kwargs),
                        operation=getattr(operation,'__name__',type(operation).__name__))
                    result=old(obj,operation,kwargs,provider,*a,**k)
                    delta=len(obj.tasks)-before
                    owner.sink.emit('registration_result',added_task_count=delta,
                        unique_task_metadata=owner.task_metadata(obj.tasks[-1]) if delta==1 else {'unknown':'nonunique task append'})
                    return result
            return call
        self._wrap(cls,'register_task',register)
        def resolve(old):
            def call(obj,*a,**k):
                result=old(obj,*a,**k)
                owner.sink.emit('resolved_update_tasks',identity=owner.graph_identity(obj),
                    tasks=[owner.task_metadata(t) for t in result] if len(result)<=64 else {'unknown':'over64 tasks'},
                    does_not_call_provider_again=True)
                return result
            return call
        self._wrap(cls,'resolve_tasks',resolve)
        def update(old):
            def call(obj,*a,**k):
                with owner.sink.scope('graph_update',**owner.graph_identity(obj)):
                    return old(obj,*a,**k)
            return call
        self._wrap(cls,'update',update)
        def apply(old):
            def call(task,*a,**k):
                with owner.sink.scope('apply_update_task'):
                    owner.sink.emit('task_apply_inputs',task=owner.task_metadata(task))
                    result=old(task,*a,**k)
                    owner.sink.emit('task_apply_return',is_not_task_or_event_GPU_completion=True)
                    return result
            return call
        self._wrap(module.GraphUpdateTask,'apply',apply)

    def task_metadata(self,task):
        return {'operation':getattr(task.operation,'__name__',type(task.operation).__name__),
            'provider_type':type(task.provider).__module__+'.'+type(task.provider).__qualname__,
            'provider_index':task.provider_index,'handle':self.encode(task.handle),
            'event_object':self.token(task.event),'kwargs':self.encode(task.kwargs),
            'native_taskID_and_completion':'UNKNOWN'}

    def dispatch_mode(self):
        owner=self
        class SelectedMetadataMode(self.mode_base):
            def __torch_dispatch__(self,func,types,args=(),kwargs=None):
                kwargs={} if kwargs is None else kwargs
                # Original operator is invoked exactly once. Metadata is emitted after
                # return, so this alone is not a pre-write snapshot or value proof.
                result=func(*args,**kwargs)
                if not owner.sink.wants_records():return result
                def has_small_bool(v,depth=0):
                    if isinstance(v,owner.torch.Tensor):
                        try:return v.dtype==owner.torch.bool and v.numel()<=4096
                        except Exception:return False
                    if depth<3 and type(v) in (tuple,list):return any(has_small_bool(x,depth+1) for x in v[:64])
                    if depth<3 and type(v) is dict:return any(has_small_bool(x,depth+1) for x in list(v.values())[:64])
                    return False
                def report():
                    schema=getattr(func,'_schema',None);name=getattr(schema,'name',None)
                    if name=='aten::constant_pad_nd' or has_small_bool(args) or has_small_bool(kwargs) or has_small_bool(result):
                        bindings=[]
                        if schema is not None:
                            for index,arg in enumerate(schema.arguments):
                                if index<len(args):value=args[index]
                                elif arg.name in kwargs:value=kwargs[arg.name]
                                else:continue
                                alias=getattr(arg,'alias_info',None)
                                bindings.append({'name':arg.name,'value':owner.encode(value),
                                    'schema_declared_write':bool(alias.is_write) if alias is not None else False})
                        owner.sink.emit('selected_operator_return',schema=name,overload=getattr(schema,'overload_name',None),
                            identity=owner.identity(),bindings=bindings,result=owner.encode(result),
                            selection='constant_pad_nd OR input/output BOOL with <=4096 elements',
                            metadata_timing='AFTER_RETURN_NOT_PREWRITE',
                            native_internals_and_all_device_readers_covered=False)
                owner.observe(report)
                return result
        return SelectedMetadataMode()

    def install_wrapper(self,module):
        self.verify_source(module);owner=self
        def boundary(old):
            def call(obj,*a,**k):
                if not owner.sink.wants_records():return old(obj,*a,**k)
                # Read only the already-established forward context and plain fields.
                # Do not call use_updatable_graph or perform a native context query.
                context=module.get_forward_context()
                descriptor=context.batch_descriptor
                data=obj.__dict__;backend=data.get('attn_backend')
                fields={'wrapper_python_token':owner.token(obj),
                    'runtime_mode':getattr(data.get('runtime_mode'),'name',None),
                    'forward_mode':getattr(context.cudagraph_runtime_mode,'name',None),
                    'enable_enpu':data.get('enable_enpu'),'use_eagle':data.get('use_eagle'),
                    'attn_backend_MRO':[c.__module__+'.'+c.__qualname__ for c in backend.__mro__] if isinstance(backend,type) else None}
                with owner.sink.scope('ACLGraphWrapper_call',**fields):
                    result=old(obj,*a,**k)
                    def report():
                        entry=data.get('concrete_aclgraph_entries',{}).get(descriptor)
                        graph=entry.aclgraph if entry is not None else None
                        owner.sink.emit('wrapper_graph_binding',wrapper_python_token=fields['wrapper_python_token'],
                            graph=owner.graph_identity(graph) if graph is not None else None,
                            descriptor={name:owner.encode(getattr(descriptor,name,None))
                                for name in ['num_tokens','num_reqs','uniform','has_lora']},
                            binding_does_not_attest_native_modelId_or_task_template=True)
                    owner.observe(report)
                    return result
            return call
        self._wrap(module.ACLGraphWrapper,'__call__',boundary)

    def install_runner(self,module):
        self.verify_source(module);owner=self
        for name in ['capture_model','execute_model']:
            def boundary(old,label=name):
                def call(obj,*a,**k):
                    if not owner.sink.wants_records():return old(obj,*a,**k)
                    def guards():
                        data=obj.__dict__;backend=data.get('attn_backend')
                        owner.sink.emit('runner_guard_snapshot',boundary=label,
                            use_compress=data.get('use_compress'),use_sparse=data.get('use_sparse'),
                            attn_backend_MRO=[c.__module__+'.'+c.__qualname__ for c in backend.__mro__] if isinstance(backend,type) else None,
                            enabled_dispatch_mode=owner.enable_dispatch,
                            guards_are_current_own_run_only=True)
                    owner.observe(guards)
                    with owner.sink.scope(label),owner.dispatch_mode() if owner.enable_dispatch else nullcontext():
                        return old(obj,*a,**k)
                return call
            self._wrap(module.NPUModelRunner,name,boundary)

def install_explicit(sink,modules):
    """Call only in a separately authorized diagnostic process after Torch is loaded."""
    if os.environ.get('WIO_TARGETED_EVIDENCE_ENABLE')!='1':
        raise RuntimeError('Explicit diagnostic opt-in is required')
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    observer=TargetedObserver(sink,torch,TorchDispatchMode,
        enable_dispatch=os.environ.get('WIO_TARGETED_DISPATCH_ENABLE')=='1')
    try:
        if 'graph' in modules:observer.install_graph(modules['graph'])
        if 'wrapper' in modules:observer.install_wrapper(modules['wrapper'])
        if 'runner' in modules:observer.install_runner(modules['runner'])
    except BaseException:
        observer.uninstall();raise
    return observer
