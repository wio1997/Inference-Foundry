"""Diagnostic-only lazy hook, present only in the isolated task container."""
import importlib.abc
import importlib.machinery
import os
import sys

if os.environ.get('WIO_HOST_PROFILE_ROOT'):
    _wanted = {
        'vllm_ascend.worker.model_runner_v1': 'install',
        'vllm_ascend.profiler.torch_npu_profiler': 'install_profiler',
    }

    class _HostProfileLoader(importlib.abc.Loader):
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def create_module(self, spec):
            create = getattr(self.wrapped, 'create_module', None)
            return create(spec) if create else None

        def exec_module(self, module):
            self.wrapped.exec_module(module)
            import wio_host_profile
            getattr(wio_host_profile, _wanted[module.__name__])(module)

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

    class _HostProfileFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname not in _wanted:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is not None and spec.loader is not None:
                spec.loader = _HostProfileLoader(spec.loader)
            return spec

    sys.meta_path.insert(0, _HostProfileFinder())
    for _name, _method in _wanted.items():
        if _name in sys.modules:
            import wio_host_profile
            getattr(wio_host_profile, _method)(sys.modules[_name])
