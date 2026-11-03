"""Diagnostic-only lazy import hook. Enabled only by an isolated PYTHONPATH."""
import importlib.abc
import importlib.machinery
import os
import sys

if os.environ.get('WIO_OBSERVER_ROOT'):
    _wanted='vllm_ascend.ops.fused_moe.moe_comm_method'
    class _ObserverLoader(importlib.abc.Loader):
        def __init__(self,wrapped):self.wrapped=wrapped
        def create_module(self,spec):
            create=getattr(self.wrapped,'create_module',None)
            return create(spec) if create else None
        def exec_module(self,module):
            self.wrapped.exec_module(module)
            import wio_moe_observer
            wio_moe_observer.install(module)
        def __getattr__(self,name):return getattr(self.wrapped,name)
    class _ObserverFinder(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            if fullname!=_wanted:return None
            spec=importlib.machinery.PathFinder.find_spec(fullname,path)
            if spec is not None and spec.loader is not None:spec.loader=_ObserverLoader(spec.loader)
            return spec
    sys.meta_path.insert(0,_ObserverFinder())
    if _wanted in sys.modules:
        import wio_moe_observer
        wio_moe_observer.install(sys.modules[_wanted])
