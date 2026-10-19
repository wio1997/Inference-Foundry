"""P-only lazy hook for validated buffer-mode storage deduplication."""
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import sys

TARGET = 'sglang.srt.mem_cache.buffer_mode.pipeline'
OPT_SHA = '78ec7134ff11ead6f30504e80aa23a0b5cf92f4c7149ac2f57dab4c756e4e08d'


class Loader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec) if hasattr(self.original, 'create_module') else None

    def exec_module(self, module):
        self.original.exec_module(module)
        import glm53_buffer_dedup_validation_v1 as optimizer
        assert hashlib.sha256(Path(optimizer.__file__).read_bytes()).hexdigest() == OPT_SHA
        installed = optimizer.install(module)
        print('GLM53_BUFFER_DEDUP_INSTALLED=' + json.dumps({'project_id': 'glm53', 'pid': os.getpid(), 'node': os.environ['GLM53_NODE'], 'optimizer_sha256': OPT_SHA, **installed}), flush=True)

    def __getattr__(self, name):
        return getattr(self.original, name)


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        assert spec is not None and spec.loader is not None and not isinstance(spec.loader, Loader)
        spec.loader = Loader(spec.loader)
        return spec


def activate():
    assert os.environ.get('GLM53_BUFFER_DEDUP_VALIDATE') == '1'
    assert os.environ.get('GLM53_NODE') in ('34', '35'), 'Storage dedup validation is P-only'
    assert not any(isinstance(item, Finder) for item in sys.meta_path)
    assert TARGET not in sys.modules
    sys.meta_path.insert(0, Finder())
    print('GLM53_BUFFER_DEDUP_HOOK_READY=' + json.dumps({'project_id': 'glm53', 'pid': os.getpid(), 'node': os.environ['GLM53_NODE'], 'optimizer_sha256': OPT_SHA}), flush=True)
