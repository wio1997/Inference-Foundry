"""E33-gated sitecustomize.

Python imports this module at startup only because the E33 launcher puts
``tools/e33_vllm_bootstrap`` first on PYTHONPATH.  Nothing here runs unless
``E33_DIAG_ROOT`` is set.

The bootstrap is intentionally a narrow import hook:
  * it intercepts only the eleven target module names below;
  * it never imports vLLM, torch or torch_npu itself;
  * it delegates spec discovery, module creation and execution to the original
    loader, preserves the original spec/loader on the module, and re-raises the
    original import exception unchanged;
  * after a target module executes, it calls the telemetry installer once.

If an existing ``sitecustomize`` is shadowed by this file, the next one on
sys.path is executed to preserve the image's normal startup behavior.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import sys
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_TELEMETRY_PATH = os.path.join(os.path.dirname(_HERE), "e33_vllm_telemetry.py")
_TELEMETRY_MODULE = "e33_vllm_telemetry"
_FINDER = None


def _chain_original_sitecustomize() -> None:
    """Run the next sitecustomize on sys.path, if one exists.

    Prepending our directory shadows a sitecustomize that may live in the
    image's site-packages.  Python normally executes only the first match, so
    execute the next match manually and keep its failure non-fatal, matching
    site.py's non-fatal handling of sitecustomize errors.
    """
    try:
        here = os.path.realpath(_HERE)
        paths = []
        for path in sys.path:
            if not path:
                continue
            try:
                if os.path.realpath(path) == here:
                    continue
            except OSError:
                pass
            paths.append(path)
        spec = importlib.machinery.PathFinder.find_spec("sitecustomize", paths)
        if spec is None or spec.loader is None or spec.origin is None:
            return
        try:
            if os.path.realpath(os.path.dirname(spec.origin)) == here:
                return
        except OSError:
            pass
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except BaseException:
        # A broken pre-existing sitecustomize must not make this diagnostic
        # bootstrap abort interpreter startup.
        pass


def _load_telemetry():
    module = sys.modules.get(_TELEMETRY_MODULE)
    if module is not None:
        return module
    if not os.path.isfile(_TELEMETRY_PATH):
        raise FileNotFoundError(_TELEMETRY_PATH)
    spec = importlib.util.spec_from_file_location(_TELEMETRY_MODULE, _TELEMETRY_PATH)
    if spec is None or spec.loader is None:
        raise ImportError("cannot build spec for %s" % _TELEMETRY_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[_TELEMETRY_MODULE] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(_TELEMETRY_MODULE, None)
        raise
    return module


def _write_bootstrap_error(exc: BaseException) -> None:
    root = os.environ.get("E33_DIAG_ROOT")
    if not root:
        return
    try:
        import json
        directory = os.path.join(root, "jsonl")
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, "bootstrap-error-%d.json" % os.getpid())
        payload = {
            "pid": os.getpid(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        }
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")
    except BaseException:
        pass


class _Loader:
    """Delegate to the real loader, then install telemetry wrappers."""

    def __init__(self, original_loader, original_spec, telemetry, fullname):
        self.original_loader = original_loader
        self.original_spec = original_spec
        self.telemetry = telemetry
        self.fullname = fullname

    def create_module(self, spec):
        create = getattr(self.original_loader, "create_module", None)
        if create is None:
            return None
        return create(spec)

    def exec_module(self, module):
        try:
            self.original_loader.exec_module(module)
        finally:
            # Preserve the original spec and loader on the module even when
            # execution failed; the import exception itself is never altered.
            try:
                module.__spec__ = self.original_spec
                module.__loader__ = self.original_loader
            except BaseException:
                pass
        try:
            self.telemetry.install_for_module(module)
        except BaseException as exc:
            try:
                self.telemetry.note(
                    "vllm.bootstrap.patch_error",
                    "telemetry install failed for %s; module left usable" % self.fullname,
                    module_name=self.fullname,
                    error_type=type(exc).__name__,
                    error=str(exc)[:2000],
                )
            except BaseException:
                pass

    def __getattr__(self, name):
        return getattr(self.original_loader, name)


class _Finder:
    """Intercept only exact target module names."""

    def __init__(self, targets, telemetry):
        self.targets = frozenset(targets)
        self.telemetry = telemetry
        self._skipped_loader = set()

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self.targets:
            return None
        try:
            for finder in tuple(sys.meta_path):
                if finder is self or isinstance(finder, _Finder):
                    continue
                find = getattr(finder, "find_spec", None)
                if not callable(find):
                    continue
                try:
                    spec = find(fullname, path, target)
                except BaseException:
                    continue
                if spec is None or spec.loader is None:
                    continue
                loader = spec.loader
                if not hasattr(loader, "exec_module"):
                    self.telemetry.note(
                        "vllm.bootstrap.unsupported_loader",
                        "target loader has no exec_module; module imported without wrapper",
                        module_name=fullname,
                        loader_type=type(loader).__name__,
                    )
                    # Preserve the normal import path rather than failing it.
                    return spec
                # Copy the spec so the original finder's cached spec is not
                # mutated; restore the original spec/loader on the module after
                # execution.
                import copy
                new_spec = copy.copy(spec)
                new_spec.loader = _Loader(loader, spec, self.telemetry, fullname)
                return new_spec
        except BaseException as exc:
            try:
                self.telemetry.note(
                    "vllm.bootstrap.find_error",
                    "target find_spec failed; falling through to normal import",
                    module_name=fullname,
                    error_type=type(exc).__name__,
                    error=str(exc)[:2000],
                )
            except BaseException:
                pass
        return None


def _install(root: str) -> None:
    global _FINDER
    telemetry = _load_telemetry()
    if _FINDER is not None:
        return
    _FINDER = _Finder(telemetry.TARGET_MODULES, telemetry)
    sys.meta_path.insert(0, _FINDER)
    # If a target was somehow imported before this hook (not expected at
    # startup), patch it now without importing anything new.
    for name in telemetry.TARGET_MODULES:
        module = sys.modules.get(name)
        if module is not None:
            try:
                telemetry.install_for_module(module)
            except BaseException as exc:
                try:
                    telemetry.note(
                        "vllm.bootstrap.patch_error",
                        "telemetry install failed for already-imported module",
                        module_name=name,
                        error_type=type(exc).__name__,
                        error=str(exc)[:2000],
                    )
                except BaseException:
                    pass
    try:
        telemetry.note(
            "vllm.bootstrap",
            "E33 import hook installed",
            diag_root=root,
            targets=list(telemetry.TARGET_MODULES),
            telemetry_file=_TELEMETRY_PATH,
            run_id=os.environ.get("E33_RUN_ID") or "unknown",
            note="target modules are not imported at startup; hook is post-import only",
        )
    except BaseException:
        pass


def _main() -> None:
    root = os.environ.get("E33_DIAG_ROOT")
    if not root:
        # No E33 diagnostics: retain whatever sitecustomize would normally run.
        _chain_original_sitecustomize()
        return
    try:
        _install(root)
    except BaseException as exc:
        _write_bootstrap_error(exc)
    # Even with diagnostics enabled, do not shadow the image's own
    # sitecustomize startup behavior.
    _chain_original_sitecustomize()


_main()
