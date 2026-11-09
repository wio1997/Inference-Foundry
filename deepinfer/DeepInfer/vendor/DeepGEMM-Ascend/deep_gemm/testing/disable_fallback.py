import torch

# Keys that mean "there is real code to run" (vs. only the backend fallback).
_REAL_KEYS = (
    "PrivateUse1",
    "CompositeImplicitAutograd",
    "CompositeExplicitAutograd",
    "CompositeExplicitAutogradNonFunctional",
)

# Keep registrations alive for the whole process (GC would undo them).
_libs = {}
_installed = False
_count = 0


def _would_fallback(name: str) -> bool:
    has = torch._C._dispatch_has_kernel_for_dispatch_key
    for key in _REAL_KEYS:
        try:
            if has(name, key):
                return False
        except Exception:
            # If we cannot reason about it, do not touch it.
            return False
    return True


def _make_raiser(op_name: str):
    def _raise(*args, **kwargs):
        info = []
        for a in list(args) + list(kwargs.values()):
            if isinstance(a, torch.Tensor):
                info.append(f"{tuple(a.shape)}:{a.dtype}")
        details = (" inputs=" + ", ".join(info)) if info else ""
        raise RuntimeError(
            f"[npu_no_fallback] operator '{op_name}' has no NPU kernel and "
            f"would fall back to the CPU.{details}"
        )

    return _raise


def disable_cpu_fallback(allow: list[str] | None = None) -> int:
    """Register a raising PrivateUse1 kernel on every would-fallback op."""
    global _installed, _count
    if _installed:
        return _count
    _installed = True

    allow = allow or []

    for full in torch._C._dispatch_get_all_op_names():
        if "::" not in full or full in allow:
            continue
        ns, _, overload = full.partition("::")
        # Only override ops that would otherwise hit the CPU fallback.
        if not _would_fallback(full):
            continue
        lib = _libs.get(ns)
        if lib is None:
            try:
                lib = torch.library.Library(ns, "IMPL")
            except Exception:
                continue
            _libs[ns] = lib
        try:
            lib.impl(overload, _make_raiser(full), "PrivateUse1")
            _count += 1
        except Exception:
            # Some overloads cannot accept a boxed python kernel; skip them.
            pass
    return _count
