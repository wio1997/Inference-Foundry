"""Late, non-importing bootstrap for the SGLang hybrid restore probe.

Call from the existing runtime overlay after module imports. This module only
looks in sys.modules; it never imports SGLang classes and therefore does not
create import cycles.
"""
from __future__ import annotations
import sys

import q27_hybrid_sg_probe as probe

REQUIRED = (
    ("sglang.srt.mem_cache.unified_cache.components.mamba", "MambaComponent", "mamba_component_cls"),
    ("sglang.srt.managers.schedule_batch", "ScheduleBatch", "schedule_batch_cls"),
    ("sglang.srt.model_executor.model_runner", "ModelRunner", "model_runner_cls"),
    ("sglang.srt.mem_cache.memory_pool", "HybridReqToTokenPool", "hybrid_pool_cls"),
)
OPTIONAL_CHECKPOINT = (
    "sglang.srt.mem_cache.mamba_checkpoint_pool", "MambaCheckpointPool", "checkpoint_pool_cls"
)


def install_from_modules(emit, span_factory=None):
    """Install when all required exact runtime classes are already imported.

    Returns a small status dict. Missing modules/classes produce ``pending`` and
    have no import side effects. Runtime class import errors are intentionally
    left to the framework's ordinary import path.
    """
    found, missing, kwargs = {}, [], {}
    for module_name, class_name, arg_name in REQUIRED:
        module = sys.modules.get(module_name)
        cls = getattr(module, class_name, None) if module is not None else None
        if not isinstance(cls, type):
            missing.append({"module": module_name, "class": class_name,
                            "reason": "module_not_imported" if module is None else "class_not_available"})
            continue
        found[class_name] = cls
        kwargs[arg_name] = cls
    if missing:
        return {"status": "pending", "missing": missing, "installed": []}

    checkpoint_module_name, checkpoint_class_name, checkpoint_arg = OPTIONAL_CHECKPOINT
    checkpoint_module = sys.modules.get(checkpoint_module_name)
    optional_status = "not_imported"
    if checkpoint_module is not None:
        checkpoint_cls = getattr(checkpoint_module, checkpoint_class_name, None)
        if isinstance(checkpoint_cls, type):
            kwargs[checkpoint_arg] = checkpoint_cls
            optional_status = "installed_if_hook_available"
        else:
            optional_status = "module_imported_class_unavailable"

    try:
        installed = probe.install(**kwargs, emit=emit, span_factory=span_factory)
    except Exception as exc:
        # This is an opt-in observer. Installation trouble must not break startup.
        return {"status": "install_error", "missing": [], "installed": [],
                "exception_type": type(exc).__name__, "optional_checkpoint": optional_status}
    return {"status": "installed", "missing": [], "installed": installed,
            "classes": {name: f"{cls.__module__}.{cls.__name__}" for name, cls in found.items()},
            "optional_checkpoint": optional_status}
