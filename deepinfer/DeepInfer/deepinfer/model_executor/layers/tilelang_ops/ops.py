"""Kernel dispatcher: interface modules do
``from deepinfer.model_executor.layers.tilelang_ops import ops`` and call
``ops.xxx_tl``; backend selection happens only in this file.

Loads the requested ``xxx_tl`` factory of the selected backend. Importing a
control or metadata module does not load every unrelated device operator.
"""

import importlib
from types import ModuleType
from typing import Any

from deepinfer import PLATFORM
from .kernel_registry import factory_modules

_pkg = importlib.import_module(f"{__package__}.{PLATFORM.tilelang_ops_backend}")
_factory_modules = factory_modules(_pkg.__path__, _pkg.__name__)
__all__ = sorted(_factory_modules) + ['KernelUnavailableError']


class KernelUnavailableError(AttributeError):
    """Selected backend lacks this kernel factory; conftest.py turns it into
    a test skip on non-cuda backends. Subclasses AttributeError to keep the
    module ``__getattr__`` / ``hasattr`` protocol intact."""


def __getattr__(name: str) -> Any:
    providers = _factory_modules.get(name, ())
    found = False
    value = None
    for provider in providers:
        module = importlib.import_module(provider)
        if name not in vars(module) or isinstance(vars(module)[name], ModuleType):
            continue
        candidate = vars(module)[name]
        assert not found or candidate is value, f"backend symbol collision: {name}"
        value = candidate
        found = True
    if not found:
        raise KernelUnavailableError(f"tilelang_ops backend {PLATFORM.tilelang_ops_backend!r} has no kernel {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_factory_modules))
