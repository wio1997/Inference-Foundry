"""Opt-in only in the isolated, specifically approved diagnostic container."""
import os
if os.environ.get('WIO_BUILTIN_COHORT_ENABLE')=='1':
    import importlib.abc,importlib.machinery,sys
    from shared_runtime import verify_approval
    try:verify_approval()
    except BaseException:
        os.write(2,b'Owned diagnostic approval check failed before model import\n')
        os._exit(126)
    wanted={
        'vllm_ascend.compilation.updatable_graph':'graph',
        'vllm_ascend.compilation.acl_graph':'wrapper',
        'vllm_ascend.worker.model_runner_v1':'runner',
        'vllm_ascend.worker.worker':'worker',
        'vllm_ascend.profiler.torch_npu_profiler':'factory'}
    class Loader(importlib.abc.Loader):
        def __init__(self,wrapped,label):self.wrapped=wrapped;self.label=label
        def create_module(self,spec):
            f=getattr(self.wrapped,'create_module',None)
            return f(spec) if f else None
        def exec_module(self,module):
            self.wrapped.exec_module(module)
            from shared_runtime import install_module
            try:install_module(module,self.label)
            except BaseException:
                os.write(2,('Owned diagnostic adapter failed: '+module.__name__+'\n').encode())
                os._exit(126)
        def __getattr__(self,key):return getattr(self.wrapped,key)
    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            if fullname not in wanted:return None
            spec=importlib.machinery.PathFinder.find_spec(fullname,path)
            if spec is not None and spec.loader is not None:
                spec.loader=Loader(spec.loader,wanted[fullname])
            return spec
    sys.meta_path.insert(0,Finder())
    for name,label in wanted.items():
        if name in sys.modules:
            from shared_runtime import install_module
            install_module(sys.modules[name],label)
