"""Opt-in metadata and CPU spans for the installed Mamba state-copy dispatch."""
import functools,sys,time

def install(emit,span):
    installed=[]
    for name in ['vllm.v1.worker.mamba_utils','vllm_ascend.patch.worker.patch_mamba_utils']:
        module=sys.modules.get(name)
        if module is None:continue
        for fn in ['collect_mamba_copy_meta','do_mamba_copy_block','_do_mamba_copy_block_npu','_do_mamba_copy_block_torch','do_mamba_copy_block_for_layer']:
            original=getattr(module,fn,None)
            if not callable(original)or getattr(original,'_q27_mamba_copy_observer',False):continue
            def wrap(original,fn,name):
                @functools.wraps(original)
                def observed(*args,**kwargs):
                    buf=args[0]if args else kwargs.get('copy_bufs');n=getattr(buf,'offset',None)
                    n=n if type(n)is int else None
                    def log(event,**extra):
                        try:emit(event,function=name+'.'+fn,mamba_copy_pair_count=n,**extra)
                        except Exception:pass
                    log('mamba_copy_dispatch_enter');start=time.monotonic_ns()
                    try:
                        with span('Q27 vLLM mamba_state_copy function='+name+'.'+fn):result=original(*args,**kwargs)
                    except BaseException as ex:log('mamba_copy_dispatch_exception',error=type(ex).__name__);raise
                    log('mamba_copy_dispatch_return',host_wall_ns=time.monotonic_ns()-start,copy_dispatch_semantics='Python return does not establish device-copy completion or state equality')
                    return result
                observed._q27_mamba_copy_observer=True
                return observed
            setattr(module,fn,wrap(original,fn,name));installed.append(name+'.'+fn)
    if installed:
        try:emit('mamba_copy_observer_installed',functions=installed)
        except Exception:pass
    return installed
