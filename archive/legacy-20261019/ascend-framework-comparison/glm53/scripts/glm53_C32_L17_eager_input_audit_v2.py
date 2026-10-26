"""GLM53 L17 D-only native eager-input first-use audit v2 (stdlib only, fail closed).

The candidate sets SGLANG_EAGER_INPUT_NO_COPY=1.  This hook pins the installed
``sglang.srt.model_executor.runner.eager_runner`` source before it executes and
wraps only ``EagerRunner.load_batch``.  The original method is always invoked
exactly once per call and its return value is returned unchanged.

The native flag is read through the same object the runner uses:
``sglang.srt.environ.envs.SGLANG_EAGER_INPUT_NO_COPY.get()``.  The bytes of the
installed ``sglang.srt.environ`` module are SHA-pinned and its ``EnvBool``
descriptor is checked before the first audited forward.

For the first forward of each
(pid, is_draft_worker, actual_forward_mode_name, padded_batch_size) key in a
process, the hook records metadata-only evidence to a per-PID JSONL file.  The
batch size is the Python integer ``forward_batch.batch_size`` only; tensor
contents are never inspected.  Decode-like modes (DECODE, TARGET_VERIFY, IDLE)
are bounded by the existing max4 D capture/admission policy; extend/mixed modes
use the existing 64-request admission bound.  A larger batch is never clamped or
reinterpreted as bs4: it fails closed.  A key is accepted only after the native
getter returned True, the original call succeeded, the complete clone validation
passed and the evidence record was written and fsynced.  Any audit failure,
including an exception from the original call, poisons the process audit state:
every later call raises without invoking the original again.  The failure cache
stores only a bounded string.

The hook never reads tensor values, never calls .cpu()/.item()/data_ptr(), and
never retains tensor or batch references in the seen cache.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
import pathlib
import sys
import threading
import time

PROJECT_ID = "glm53"
AUDIT_VERSION = "v2"
SCHEMA = "glm53.L17.eager_input_audit.v2"
SOURCE_MODULE = "sglang.srt.model_executor.runner.eager_runner"
ENVIRON_MODULE = "sglang.srt.environ"
SOURCE_SHA256 = "a04d7344319eccbeacefad3f60e559c716f412e680ed0f712645ef149238a6b5"
ENVIRON_SHA256 = "0b0bb923ab563986fe8b1e068330959d249c859f6b24ae3f884df4812d457d1c"
DEFAULT_AUDIT_ROOT = "/run/glm53/eager-input-audit"
ALLOWED_NODES = ("36", "41")
MAX_DECODE_BATCH_SIZE = 4
MAX_EXTEND_BATCH_SIZE = 64
DECODE_LIKE_MODES = ("DECODE", "TARGET_VERIFY", "IDLE")
ALLOWLISTED_FIELDS = (
    "input_ids",
    "positions",
    "seq_lens",
    "out_cache_loc",
    "req_pool_indices",
    "seq_lens_cpu",
    "spec_info",
)
_IMMUTABLE_TYPES = (bool, int, float, complex, str, bytes, type(None))
_MAX_DATACLASS_DEPTH = 2
_MAX_FAILURE_CHARS = 480

_FIRST_USE_LOCK = threading.Lock()
_WRITE_LOCK = threading.Lock()
_SEEN = set()
_FAILURE = None
_AUDIT_ROOT = None
_ACTIVE_SOURCE_SHA = None
_NODE = None
_FINDER = None
_ENVIRON_SOURCE_PINNED = False
_PINNED_ENVS = None
_PINNED_FLAG = None
_PINNED_GETTER = None


class AuditFailure(RuntimeError):
    """Raised on any first-use audit violation; the process must fail closed."""


def _current_pid():
    return os.getpid()


def _env(name):
    return os.environ.get(name)


def _bounded_error_text(error):
    """Return a bounded type+message string; never retain an exception object."""
    try:
        message = str(error)
    except BaseException:
        message = "<unprintable error>"
    message = message.replace("\r", " ").replace("\n", " ")[:_MAX_FAILURE_CHARS]
    return "%s: %s" % (type(error).__name__, message)


def _poison(error):
    global _FAILURE
    if _FAILURE is None:
        _FAILURE = _bounded_error_text(error)


def _raise_if_failed_locked():
    if _FAILURE is not None:
        raise AuditFailure("sticky first-use audit failure: " + _FAILURE)


def _require_audit_environment():
    if _env("GLM53_EAGER_INPUT_AUDIT") != "1":
        raise AuditFailure("GLM53_EAGER_INPUT_AUDIT is not exactly '1'")
    node = _env("GLM53_NODE")
    if node not in ALLOWED_NODES:
        raise AuditFailure("GLM53_NODE is not one of %r: %r" % (ALLOWED_NODES, node))
    if _env("SGLANG_EAGER_INPUT_NO_COPY") != "1":
        raise AuditFailure("SGLANG_EAGER_INPUT_NO_COPY is not exactly '1'")
    return node


def _resolve_root(root=None):
    candidate = root
    if candidate is None:
        candidate = _env("GLM53_EAGER_INPUT_AUDIT_ROOT")
    if candidate is None:
        candidate = DEFAULT_AUDIT_ROOT
    if not isinstance(candidate, str) or not candidate:
        raise AuditFailure("invalid GLM53_EAGER_INPUT_AUDIT_ROOT: %r" % (candidate,))
    if not os.path.isabs(candidate):
        raise AuditFailure("audit root is not absolute: %r" % (candidate,))
    return os.path.normpath(candidate)


def _sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path):
    return _sha256_bytes(pathlib.Path(path).read_bytes())


# ---------------------------------------------------------------------------
# Native environ pinning
# ---------------------------------------------------------------------------


def _pin_environ_source(require=True):
    """Verify the installed environ bytes and resolve the actual native getter.

    Called once before the first audited forward.  Later first-use keys only
    perform a cheap identity check; accepted keys never touch this path.
    """
    global _ENVIRON_SOURCE_PINNED, _PINNED_ENVS, _PINNED_FLAG, _PINNED_GETTER

    module = sys.modules.get(ENVIRON_MODULE)
    if module is None:
        if require:
            raise AuditFailure("%s is not imported" % (ENVIRON_MODULE,))
        return False

    envs_obj = getattr(module, "envs", None)
    if envs_obj is None:
        raise AuditFailure("%s.envs is missing" % (ENVIRON_MODULE,))
    flag = getattr(envs_obj, "SGLANG_EAGER_INPUT_NO_COPY", None)
    getter = getattr(flag, "get", None)
    if not callable(getter):
        raise AuditFailure(
            "%s.envs.SGLANG_EAGER_INPUT_NO_COPY has no EnvBool.get()"
            % (ENVIRON_MODULE,)
        )

    if _ENVIRON_SOURCE_PINNED:
        if flag is not _PINNED_FLAG or envs_obj is not _PINNED_ENVS:
            raise AuditFailure("native environ envs/SGLANG_EAGER_INPUT_NO_COPY changed")
        return True

    origin = getattr(module, "__file__", None)
    if not origin:
        raise AuditFailure("%s is imported without __file__" % (ENVIRON_MODULE,))
    actual = _sha256_file(origin)
    if actual != ENVIRON_SHA256:
        raise AuditFailure(
            "installed environ source SHA mismatch: %s actual=%s expected=%s"
            % (origin, actual, ENVIRON_SHA256)
        )
    if getattr(flag, "name", None) != "SGLANG_EAGER_INPUT_NO_COPY":
        raise AuditFailure(
            "SGLANG_EAGER_INPUT_NO_COPY descriptor name is not bound: %r"
            % (getattr(flag, "name", None),)
        )
    if type(flag).__name__ != "EnvBool":
        raise AuditFailure(
            "SGLANG_EAGER_INPUT_NO_COPY is not EnvBool: %r"
            % (type(flag).__name__,)
        )

    _ENVIRON_SOURCE_PINNED = True
    _PINNED_ENVS = envs_obj
    _PINNED_FLAG = flag
    _PINNED_GETTER = getter
    return True


def _native_no_copy_get_true():
    _pin_environ_source(require=True)
    value = _PINNED_GETTER()
    if value is not True:
        raise AuditFailure(
            "native envs.SGLANG_EAGER_INPUT_NO_COPY.get() is not True: %r" % (value,)
        )
    return True


# ---------------------------------------------------------------------------
# Source pinning and patching
# ---------------------------------------------------------------------------


class _PinnedSourceLoader(importlib.abc.Loader):
    """Load the exact bytes whose SHA-256 was verified before execution."""

    def __init__(self, origin, expected_sha256, audit_root):
        self.origin = origin
        self.expected_sha256 = expected_sha256
        self.audit_root = audit_root

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        data = pathlib.Path(self.origin).read_bytes()
        actual = _sha256_bytes(data)
        if actual != self.expected_sha256:
            raise AuditFailure(
                "installed source SHA mismatch before exec: %s actual=%s expected=%s"
                % (self.origin, actual, self.expected_sha256)
            )
        try:
            code = compile(data, self.origin, "exec", dont_inherit=True)
        except BaseException as error:
            raise AuditFailure("source compile failed before exec: %r" % (error,)) from error
        try:
            exec(code, module.__dict__)
        except BaseException as error:
            raise AuditFailure("pinned source exec failed: %r" % (error,)) from error
        _patch_module(module, actual, self.audit_root)


class _AuditFinder(importlib.abc.MetaPathFinder):
    """Intercept exactly SOURCE_MODULE and no other import."""

    def __init__(self, expected_sha256, audit_root):
        self.expected_sha256 = expected_sha256
        self.audit_root = audit_root

    def find_spec(self, fullname, path=None, target=None):
        if fullname != SOURCE_MODULE:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.origin is None or not spec.has_location:
            raise AuditFailure(
                "cannot locate installed %s (spec=%r origin=%r)"
                % (SOURCE_MODULE, spec, getattr(spec, "origin", None))
            )
        origin = os.path.realpath(spec.origin)
        if not origin.endswith(".py") or not os.path.isfile(origin):
            raise AuditFailure("installed %s origin is not a .py file: %r" % (SOURCE_MODULE, origin))
        spec.loader = _PinnedSourceLoader(origin, self.expected_sha256, self.audit_root)
        return spec


def _patch_module(module, source_sha256, audit_root):
    global _ACTIVE_SOURCE_SHA, _AUDIT_ROOT
    if source_sha256 != _ACTIVE_SOURCE_SHA and _ACTIVE_SOURCE_SHA is not None:
        raise AuditFailure(
            "patch source SHA mismatch: actual=%s expected=%s"
            % (source_sha256, _ACTIVE_SOURCE_SHA)
        )
    runner = getattr(module, "EagerRunner", None)
    if runner is None:
        raise AuditFailure("%s has no EagerRunner" % (getattr(module, "__name__", module),))
    original = getattr(runner, "load_batch", None)
    if not callable(original):
        raise AuditFailure("EagerRunner.load_batch is missing or not callable")
    if getattr(original, "__glm53_l17_eager_input_audit__", None) == AUDIT_VERSION:
        return original

    # Verify the native environ module before touching the runner class.
    _pin_environ_source(require=False)

    if _ACTIVE_SOURCE_SHA is None:
        _ACTIVE_SOURCE_SHA = source_sha256
    if audit_root is not None:
        _AUDIT_ROOT = audit_root

    @functools.wraps(original)
    def load_batch(self, forward_batch, *args, **kwargs):
        with _FIRST_USE_LOCK:
            _raise_if_failed_locked()
            try:
                key = _first_use_key(self, forward_batch)
            except BaseException as error:
                _poison(error)
                raise
            accepted = key in _SEEN

        if accepted:
            return _call_original(original, self, forward_batch, args, kwargs)

        with _FIRST_USE_LOCK:
            _raise_if_failed_locked()
            if key not in _SEEN:
                return _first_use_locked(
                    original, self, forward_batch, args, kwargs, key
                )

        return _call_original(original, self, forward_batch, args, kwargs)

    load_batch.__glm53_l17_eager_input_audit__ = AUDIT_VERSION
    load_batch.__glm53_l17_source_sha256__ = source_sha256
    load_batch.__glm53_l17_original__ = original
    runner.load_batch = load_batch
    try:
        module.__glm53_l17_eager_input_audit__ = {
            "version": AUDIT_VERSION,
            "source_sha256": source_sha256,
            "environ_source_sha256": ENVIRON_SHA256,
        }
    except BaseException:
        pass
    return load_batch


def _call_original(original, self, forward_batch, args, kwargs):
    try:
        return original(self, forward_batch, *args, **kwargs)
    except BaseException as error:
        with _FIRST_USE_LOCK:
            _poison(error)
        raise


def _first_use_locked(original, self, forward_batch, args, kwargs, key):
    """Run the complete first-use chain while holding _FIRST_USE_LOCK."""
    try:
        _native_no_copy_get_true()
    except AuditFailure as error:
        _poison(error)
        raise
    except BaseException as error:
        _poison(error)
        raise AuditFailure("native environ check failed: " + _bounded_error_text(error)) from None

    try:
        result = original(self, forward_batch, *args, **kwargs)
    except BaseException as error:
        _poison(error)
        raise

    try:
        record = _build_record(key, forward_batch, result)
        _append_record(record, _resolve_root(_AUDIT_ROOT))
        _SEEN.add(key)
    except AuditFailure as error:
        _poison(error)
        raise
    except BaseException as error:
        _poison(error)
        raise AuditFailure("first-use evidence failed: " + _bounded_error_text(error)) from None
    return result


# ---------------------------------------------------------------------------
# First-use evidence
# ---------------------------------------------------------------------------


def _forward_mode_name(forward_batch):
    mode = getattr(forward_batch, "forward_mode", None)
    if mode is None:
        raise AuditFailure("forward_batch.forward_mode is missing")
    name = getattr(mode, "name", None)
    if isinstance(name, str) and name:
        return name
    return str(mode)


def _is_draft_worker(runner):
    model_runner = getattr(runner, "model_runner", None)
    if model_runner is None:
        raise AuditFailure("EagerRunner.model_runner is missing")
    value = getattr(model_runner, "is_draft_worker", None)
    if type(value) is not bool:
        raise AuditFailure(
            "model_runner.is_draft_worker is not bool: %r" % (type(value).__name__,)
        )
    return value


def _batch_size(forward_batch, mode_name):
    value = getattr(forward_batch, "batch_size", None)
    if type(value) is not int:
        raise AuditFailure(
            "forward_batch.batch_size is not a Python int: %r"
            % (type(value).__name__,)
        )
    if value < 1:
        raise AuditFailure("forward_batch.batch_size is not positive: %r" % (value,))
    limit = (
        MAX_DECODE_BATCH_SIZE
        if mode_name in DECODE_LIKE_MODES
        else MAX_EXTEND_BATCH_SIZE
    )
    if value > limit:
        raise AuditFailure(
            "forward_batch.batch_size=%d exceeds the %s mode bound %d; not clamped"
            % (value, mode_name, limit)
        )
    return value


def _first_use_key(runner, forward_batch):
    mode_name = _forward_mode_name(forward_batch)
    return (
        _current_pid(),
        _is_draft_worker(runner),
        mode_name,
        _batch_size(forward_batch, mode_name),
    )


def _field_verdict(old, new):
    if old is new:
        return "identity"
    if type(old) is type(new) and isinstance(old, _IMMUTABLE_TYPES):
        try:
            if old == new:
                return "immutable_equal"
        except BaseException:
            return None
    return None


def _int_sequence(value):
    try:
        items = list(value)
    except BaseException:
        return None
    if not all(type(item) is int for item in items):
        return None
    return items


def _tensor_metadata(value):
    if not (
        hasattr(value, "shape")
        and hasattr(value, "dtype")
        and hasattr(value, "device")
        and callable(getattr(value, "stride", None))
    ):
        return None
    try:
        shape = _int_sequence(value.shape)
        stride = _int_sequence(value.stride())
        dtype = str(value.dtype)
        device = str(value.device)
    except BaseException:
        return None
    if shape is None or stride is None:
        return None
    return {
        "tensor": True,
        "shape": shape,
        "dtype": dtype,
        "device": device,
        "stride": stride,
    }


def _summarize_value(value, depth=0):
    if value is None:
        return None
    tensor = _tensor_metadata(value)
    if tensor is not None:
        return tensor
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = dataclasses.fields(value)
        summary = {
            "dataclass": type(value).__name__,
            "field_count": len(fields),
        }
        if depth < _MAX_DATACLASS_DEPTH:
            summary["fields"] = {
                field.name: _summarize_value(getattr(value, field.name), depth + 1)
                for field in fields
            }
        else:
            summary["fields_deferred"] = True
        return summary
    if isinstance(value, (list, tuple)):
        return {"container": type(value).__name__, "len": len(value)}
    if isinstance(value, _IMMUTABLE_TYPES):
        return {"scalar_type": type(value).__name__}
    return {"type": type(value).__name__}


def _batch_summary(forward_batch):
    batch_size = getattr(forward_batch, "batch_size", None)
    if type(batch_size) is int:
        value = batch_size
    else:
        value = None
    return {"batch_size": value, "batch_size_type": type(batch_size).__name__}


def _validate_clone(forward_batch, result):
    if result is forward_batch:
        raise AuditFailure("native branch returned the same ForwardBatch object")
    if type(result) is not type(forward_batch):
        raise AuditFailure(
            "native branch returned a different type: %r vs %r"
            % (type(result).__name__, type(forward_batch).__name__)
        )
    if not dataclasses.is_dataclass(forward_batch) or not dataclasses.is_dataclass(result):
        raise AuditFailure("ForwardBatch is not a dataclass")
    before_fields = dataclasses.fields(forward_batch)
    after_fields = dataclasses.fields(result)
    before_names = tuple(field.name for field in before_fields)
    after_names = tuple(field.name for field in after_fields)
    if before_names != after_names:
        raise AuditFailure("dataclass field set changed across replace")
    verdicts = {"identity": 0, "immutable_equal": 0}
    for name in before_names:
        verdict = _field_verdict(getattr(forward_batch, name), getattr(result, name))
        if verdict is None:
            raise AuditFailure("dataclass field changed across replace: %s" % (name,))
        verdicts[verdict] += 1
    return before_names, verdicts


def _build_record(key, forward_batch, result):
    pid, is_draft_worker, mode_name, batch_size = key
    before_names, verdicts = _validate_clone(forward_batch, result)
    allowlisted = {}
    for name in ALLOWLISTED_FIELDS:
        if name not in before_names:
            allowlisted[name] = {"absent": True}
        else:
            allowlisted[name] = _summarize_value(getattr(result, name))
    node = _NODE
    if node is None:
        raw_node = _env("GLM53_NODE")
        if raw_node not in ALLOWED_NODES:
            raise AuditFailure("GLM53_NODE is not one of %r: %r" % (ALLOWED_NODES, raw_node))
        node = int(raw_node)
    return {
        "schema": SCHEMA,
        "audit_version": AUDIT_VERSION,
        "project_id": PROJECT_ID,
        "source_sha256": _ACTIVE_SOURCE_SHA,
        "eager_source_sha256": _ACTIVE_SOURCE_SHA,
        "environ_source_sha256": ENVIRON_SHA256,
        "node": node,
        "pid": pid,
        "forward_mode": mode_name,
        "is_draft_worker": is_draft_worker,
        "batch_size": batch_size,
        "batch_size_source": "forward_batch.batch_size (Python int, padded live batch)",
        "first_use_key_fields": [
            "pid",
            "is_draft_worker",
            "forward_mode",
            "batch_size",
        ],
        "native_no_copy_get": True,
        "clone_distinct": True,
        "clone_type": type(result).__name__,
        "dataclass_field_count": len(before_names),
        "field_names_sha256": _sha256_bytes("\n".join(before_names).encode("utf-8")),
        "field_verdicts": verdicts,
        "batch": _batch_summary(forward_batch),
        "allowlisted": allowlisted,
        "timestamp_unix": time.time(),
    }


def _append_record(record, root):
    directory = pathlib.Path(root)
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except BaseException as error:
        raise AuditFailure("cannot create audit root %r: %r" % (root, error)) from error
    path = directory / ("pid-%d.jsonl" % record["pid"])
    line = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    try:
        with _WRITE_LOCK:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())
    except BaseException as error:
        raise AuditFailure(
            "cannot append first-use audit record %r: %r" % (str(path), error)
        ) from error


# ---------------------------------------------------------------------------
# Fork handling
# ---------------------------------------------------------------------------


def _after_fork_in_child():
    global _FIRST_USE_LOCK, _WRITE_LOCK, _SEEN, _FAILURE
    _FIRST_USE_LOCK = threading.Lock()
    _WRITE_LOCK = threading.Lock()
    _SEEN = set()
    _FAILURE = None


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_in_child)


# ---------------------------------------------------------------------------
# Activation
# ---------------------------------------------------------------------------


def activate(root=None):
    """Validate configuration, pin the installed sources, and install the hook."""
    global _AUDIT_ROOT, _ACTIVE_SOURCE_SHA, _NODE, _FINDER
    node = _require_audit_environment()
    resolved_root = _resolve_root(root)
    _AUDIT_ROOT = resolved_root
    _ACTIVE_SOURCE_SHA = SOURCE_SHA256
    _NODE = int(node)

    # Fail closed now if the native environ module is already loaded.
    _pin_environ_source(require=False)

    existing = sys.modules.get(SOURCE_MODULE)
    if existing is not None:
        origin = getattr(existing, "__file__", None)
        if not origin:
            raise AuditFailure("%s is already imported without __file__" % (SOURCE_MODULE,))
        actual = _sha256_file(origin)
        if actual != SOURCE_SHA256:
            raise AuditFailure(
                "already-imported source SHA mismatch: %r actual=%s expected=%s"
                % (origin, actual, SOURCE_SHA256)
            )
        _patch_module(existing, actual, resolved_root)
        return {"mode": "patched_already_imported", "source_sha256": actual}

    for finder in list(sys.meta_path):
        if isinstance(finder, _AuditFinder):
            sys.meta_path.remove(finder)
    _FINDER = _AuditFinder(SOURCE_SHA256, resolved_root)
    sys.meta_path.insert(0, _FINDER)
    return {"mode": "meta_path_installed", "source_sha256": SOURCE_SHA256}


def _reset_for_tests():
    global _AUDIT_ROOT, _ACTIVE_SOURCE_SHA, _NODE, _FINDER
    global _FIRST_USE_LOCK, _WRITE_LOCK, _FAILURE
    global _ENVIRON_SOURCE_PINNED, _PINNED_ENVS, _PINNED_FLAG, _PINNED_GETTER
    _FIRST_USE_LOCK = threading.Lock()
    _WRITE_LOCK = threading.Lock()
    _SEEN.clear()
    _FAILURE = None
    _AUDIT_ROOT = None
    _ACTIVE_SOURCE_SHA = None
    _NODE = None
    _ENVIRON_SOURCE_PINNED = False
    _PINNED_ENVS = None
    _PINNED_FLAG = None
    _PINNED_GETTER = None
    if _FINDER is not None:
        for finder in list(sys.meta_path):
            if finder is _FINDER:
                sys.meta_path.remove(finder)
    _FINDER = None
