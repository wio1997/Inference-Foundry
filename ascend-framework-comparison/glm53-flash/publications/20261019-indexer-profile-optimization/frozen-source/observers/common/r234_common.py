"""Shared, stdlib-only support for the R234 cold-prefill window observers.

The observers run inside the P service process.  They never read NPU tensors:
all window evidence comes from scheduler metadata that is already host-side.
The module is intentionally import-light so that sitecustomize can load it
before torch/vLLM/SGLang are imported.
"""
from __future__ import annotations

import hashlib
import importlib.abc
import json
import os
import pathlib
import sys
import time

SCHEMA_META = "r234/profile-forward/v1"
SCHEMA_STATUS = "r234/profile-observer-status/v1"
SCHEMA_MANIFEST = "r234/observer-manifest/v1"
SCHEMA_SITECUSTOMIZE_PRECHECK = "r234/sitecustomize-precheck/v1"

DEFAULT_WINDOWS = (
    ("early", 8192, 16384),
    ("mid", 32768, 40960),
    ("late", 61440, 73728),
)


def synchronize_profile_boundary():
    """Drain previous device work only at a diagnostic capture boundary.

    No tensor values are read. This is deliberately outside a model forward;
    the measured capture is not a production throughput measurement.
    """
    import torch
    torch.npu.synchronize()


class WindowedProfiler:
    """Proxy a native profiler with one device-drained capture annotation.

    A CPU execute_model annotation can end before its asynchronous kernels.
    This outer annotation closes only after the final device drain. It is the
    sole denominator for window occupancy; individual kernel ownership still
    requires native correlation IDs and is not inferred from CPU timestamps.
    """

    def __init__(self, native, namespace, window):
        import torch
        self.native = native
        self.closed = False
        self.marker = torch.profiler.record_function(
            "r234_capture_window_" + namespace + "_" + window)
        self.marker.__enter__()

    def __getattr__(self, name):
        return getattr(self.native, name)

    def stop(self):
        if self.closed:
            return None
        error = None
        try:
            synchronize_profile_boundary()
        except BaseException as exc:
            error = exc
        finally:
            self.marker.__exit__(None, None, None)
            self.closed = True
        try:
            result = self.native.stop()
        except BaseException:
            if error is None:
                raise
        if error is not None:
            raise error
        return result

    def shutdown(self):
        if not self.closed:
            return self.stop()
        method = getattr(self.native, "shutdown", None)
        return method() if method else None


def env_int(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def parse_windows(value=None):
    if not value:
        return [{"name": n, "lo": lo, "hi": hi} for n, lo, hi in DEFAULT_WINDOWS]
    rows = []
    for item in value.split(","):
        name, lo, hi = item.split(":")
        assert name and int(lo) < int(hi), item
        rows.append({"name": name, "lo": int(lo), "hi": int(hi)})
    assert rows, "at least one window is required"
    return rows


def rank_namespace(dp_rank, rank):
    """A path/session namespace that is unique per logical DP engine AND rank.

    vLLM DP0 and DP1 each expose visible dies 0..7, so both TP leaders have
    local_rank 0 while their global ranks are 0 and 8.  local_rank alone is a
    visible-set offset, never a physical die or a global rank; keying files by
    it aliases the two engines.  The namespace therefore contains both values.
    """
    return "dp%d-rank%d" % (int(dp_rank), int(rank))


def metadata_filename(namespace):
    return "forward-metadata-" + namespace + ".jsonl"


def status_filename(namespace):
    return "observer-status-" + namespace + ".json"


def parse_rank_namespace(text):
    """Parse dp<D>-rank<R>; return None for an unknown/legacy name."""
    import re
    match = re.search(r"dp(\d+)-rank(\d+)", str(text))
    if match:
        return {"dp_rank": int(match.group(1)), "rank": int(match.group(2))}
    return None



def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, obj):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp." + str(os.getpid()))
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def append_jsonl(path, obj):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n"
    # One O_APPEND write per record; the observer is the only writer.
    fd = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    try:
        os.write(fd, line.encode())
    finally:
        os.close(fd)


def now_ns():
    return time.monotonic_ns()


def dir_bytes(path):
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def free_bytes(path):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize


def load_manifest(path):
    path = pathlib.Path(path)
    raw = path.read_bytes()
    data = json.loads(raw)
    assert data.get("schema") == SCHEMA_MANIFEST, "wrong observer manifest schema"
    return data, hashlib.sha256(raw).hexdigest()


def verify_observer_files(manifest_path, root):
    """Verify every staged observer file before a target module is patched."""
    data, manifest_sha = load_manifest(manifest_path)
    root = pathlib.Path(root)
    seen = {}
    for rel, expected in data["files"].items():
        p = root / rel
        assert p.is_file(), "missing observer file " + rel
        actual = sha256_file(p)
        assert actual == expected, "observer SHA mismatch " + rel
        seen[rel] = actual
    return {"manifest_sha256": manifest_sha, "files": seen,
            "entrypoint_sha256": seen.get(data["entrypoint"])}


class DeferredPatchLoader(importlib.abc.Loader):
    def __init__(self, loader, patch):
        self._loader = loader
        self._patch = patch

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module):
        self._loader.exec_module(module)
        self._patch(module)

    def __getattr__(self, name):
        return getattr(self._loader, name)


class DeferredPatchFinder(importlib.abc.MetaPathFinder):
    """Intercept exactly one module and patch its class after native exec."""

    def __init__(self, target, patch, on_error):
        self.target = target
        self.patch = patch
        self.on_error = on_error
        self.seen = False

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.target or self.seen:
            return None
        try:
            for finder in sys.meta_path:
                if finder is self:
                    continue
                spec = finder.find_spec(fullname, path, target)
                if spec is None:
                    continue
                if spec.loader is None:
                    return spec
                self.seen = True
                spec.loader = DeferredPatchLoader(spec.loader, self._patch_guarded)
                return spec
        except BaseException:
            # Fall through to the remaining native finders; a diagnostic
            # observer must not change import resolution.
            return None
        return None

    def _patch_guarded(self, module):
        try:
            self.patch(module)
        except BaseException as exc:  # never break native service behaviour
            try:
                self.on_error(repr(exc))
            except BaseException:
                pass
            # Swallow by design: a diagnostic observer must never turn a
            # service startup error into a native regression.  The status
            # file records installed=false and the controller fails closed.


def install_post_import_patch(target, patch, on_error):
    """Install the finder once; return False when already installed."""
    for finder in sys.meta_path:
        if isinstance(finder, DeferredPatchFinder) and finder.target == target:
            return False
    sys.meta_path.insert(0, DeferredPatchFinder(target, patch, on_error))
    return True


class WindowPolicy:
    """Choose the first forward of each bounded context window.

    A window starts on the first prefill forward whose every active request has
    context_end in [lo, hi].  Two consecutive forwards are captured when both
    that forward and the predicted next one are inside the window; otherwise
    only the current forward is captured and the window is honestly marked
    partial.  The prediction is verified against real metadata after capture.

    ``expected_active`` is the production native batch shape, not the logical
    request count.  The original R201 slow-B logs contain 551 native prefill
    batches with one new sequence row and 31 with two, so the production B
    profile expects one row.  Every observed prefill batch shape is counted
    from real scheduler metadata and disclosed; mixed multi-row batches are
    skipped, never forced or merged into a same-context pair.
    """

    def __init__(self, windows, max_forwards=6, expected_active=None):
        self.windows = list(windows)
        self.index = 0
        self.captured_forwards = 0
        self.max_forwards = max_forwards
        self.expected_active = expected_active
        self.results = {}
        self._active = None
        self._active_records = []
        self._expected = 0
        self.seen_batches = {}
        self.expected_active_skips = 0
        self.mixed_batches_skipped = 0

    @property
    def active(self):
        return self._active is not None

    @staticmethod
    def _active_requests(meta):
        return [r for r in meta.get("requests", []) if r.get("query_tokens", 0) > 0]

    def note_batch(self, meta):
        """Count every real prefill batch shape, captured or skipped."""
        if not isinstance(meta, dict) or meta.get("mode") != "prefill":
            return
        reqs = self._active_requests(meta)
        if not reqs:
            return
        ends = [r.get("context_end") for r in reqs]
        queries = sorted(int(r.get("query_tokens", 0)) for r in reqs)
        key = "rows=%d;contexts_equal=%s;query=%s" % (
            len(reqs), len(set(ends)) <= 1, ",".join(str(q) for q in queries))
        self.seen_batches[key] = self.seen_batches.get(key, 0) + 1
        if self.expected_active is not None and len(reqs) != self.expected_active:
            self.expected_active_skips += 1
        if len(reqs) > 1:
            self.mixed_batches_skipped += 1

    def _advance_missed(self, meta):
        reqs = self._active_requests(meta)
        if not reqs or meta.get("mode") != "prefill":
            return
        while self.index < len(self.windows):
            window = self.windows[self.index]
            ends = [r.get("context_end") for r in reqs]
            if all(type(end) is int and end > window["hi"] for end in ends):
                self.results[window["name"]] = {
                    "name": window["name"], "lo": window["lo"], "hi": window["hi"],
                    "expected_forwards": 2, "captured_forwards": 0,
                    "forward_indices": [], "context_ends": [], "query_tokens": [],
                    "active_requests": [], "valid": False,
                    "reason": "context_passed_window_without_trigger"}
                self.index += 1
                continue
            break

    def _candidate(self, meta):
        self._advance_missed(meta)
        if self.index >= len(self.windows):
            return None
        window = self.windows[self.index]
        reqs = self._active_requests(meta)
        if not reqs or meta.get("mode") != "prefill":
            return None
        if self.expected_active is not None and len(reqs) != self.expected_active:
            return None
        ends = []
        for req in reqs:
            end = req.get("context_end")
            query = req.get("query_tokens")
            if type(end) is not int or type(query) is not int or query <= 0:
                return None
            if not (window["lo"] <= end <= window["hi"]):
                return None
            ends.append(end)
        if self.expected_active and self.expected_active > 1 and len(set(ends)) != 1:
            return None
        # Predict one more equal-sized chunk per active request.
        for req in reqs:
            if req["context_end"] + req["query_tokens"] > window["hi"]:
                return window, 1
        return window, 2

    def compatible(self, meta):
        """Whether an active capture may record this real forward unchanged."""
        if not self.active or not isinstance(meta, dict) or meta.get("mode") != "prefill":
            return False
        reqs = self._active_requests(meta)
        if not reqs:
            return False
        if self.expected_active is not None and len(reqs) != self.expected_active:
            return False
        window = self._active
        if any(type(r.get("context_end")) is not int
               or not (window["lo"] <= r["context_end"] <= window["hi"])
               or type(r.get("query_tokens")) is not int
               or r["query_tokens"] <= 0 for r in reqs):
            return False
        if self._active_records:
            existing = sorted(r.get("rid") for r in
                              self._active_requests(self._active_records[0]))
            current = sorted(r.get("rid") for r in reqs)
            if current != existing:
                return False
        return True


    def begin(self, meta):
        """Return (window, expected_forwards) when this forward starts a capture."""
        if self.active:
            return None
        if self.captured_forwards >= self.max_forwards:
            return None
        candidate = self._candidate(meta)
        if candidate is None:
            return None
        window, expected = candidate
        if self.captured_forwards + expected > self.max_forwards:
            expected = max(0, self.max_forwards - self.captured_forwards)
        if expected <= 0:
            return None
        self._active = window
        self._active_records = []
        self._expected = expected
        return window, expected

    def record(self, meta):
        """Record one captured forward; return True when the capture is done."""
        if not self.active:
            return False
        self._active_records.append(meta)
        self.captured_forwards += 1
        return len(self._active_records) >= self._expected

    def finish(self):
        """Close the active window and return its immutable proof record."""
        window = self._active
        records = self._active_records
        expected = self._expected
        result = validate_capture(window, records, expected, self.expected_active)
        self.results[window["name"]] = result
        self._active = None
        self._active_records = []
        self._expected = 0
        if self.index < len(self.windows) and self.windows[self.index] is window:
            self.index += 1
        return result

    def abort(self, reason):
        if self.active:
            window = self._active
            result = validate_capture(window, self._active_records, self._expected,
                                      self.expected_active)
            result["valid"] = False
            result["reason"] = reason
            self.results[window["name"]] = result
            self._active = None
            self._active_records = []
            self._expected = 0
            if self.index < len(self.windows) and self.windows[self.index] is window:
                self.index += 1

    def status(self):
        return {
            "windows": self.results,
            "captured_forwards": self.captured_forwards,
            "max_forwards": self.max_forwards,
            "expected_active": self.expected_active,
            "observed_prefill_batch_shapes": dict(sorted(self.seen_batches.items())),
            "expected_active_skips": self.expected_active_skips,
            "mixed_batches_skipped": self.mixed_batches_skipped,
            "mixed_batches_disclosure": (
                "Actual multi-row prefill batches are counted from native "
                "scheduler metadata, skipped for the one-row sampling rule, "
                "and never merged into an invented same-context pair."),
            "complete": self.index >= len(self.windows),
        }


def validate_capture(window, records, expected, expected_active=None):
    result = {"name": window["name"], "lo": window["lo"], "hi": window["hi"],
              "expected_forwards": expected, "captured_forwards": len(records),
              "forward_indices": [r.get("forward_index") for r in records],
              "context_ends": [[q.get("context_end") for q in r.get("requests", [])]
                               for r in records],
              "query_tokens": [[q.get("query_tokens") for q in r.get("requests", [])]
                               for r in records],
              "active_requests": [r.get("active_requests") for r in records],
              "valid": False, "reason": None}
    if len(records) < expected:
        result["reason"] = "capture_ended_early"
        return result
    if len(records) != expected:
        result["reason"] = "capture_count_mismatch"
        return result
    rids = None
    last_ends = None
    for record in records:
        reqs = [r for r in record.get("requests", []) if r.get("query_tokens", 0) > 0]
        if not reqs:
            result["reason"] = "no_active_prefill_request"
            return result
        if expected_active is not None and len(reqs) != expected_active:
            result["reason"] = "active_request_count_mismatch"
            return result
        current_rids = sorted(r.get("rid") for r in reqs)
        if rids is None:
            rids = current_rids
        elif current_rids != rids:
            result["reason"] = "request_set_changed"
            return result
        ends = []
        for req in reqs:
            end = req.get("context_end")
            query = req.get("query_tokens")
            if type(end) is not int or type(query) is not int or query <= 0:
                result["reason"] = "missing_host_metadata"
                return result
            if not (window["lo"] <= end <= window["hi"]):
                result["reason"] = "context_outside_window"
                return result
            ends.append(end)
        if len(ends) > 1 and len(set(ends)) != 1:
            result["reason"] = "request_contexts_not_equal"
            return result
        if last_ends is not None:
            for before, after, req in zip(last_ends, ends, reqs):
                if after - before != req["query_tokens"]:
                    result["reason"] = "context_query_discontinuous"
                    return result
        last_ends = ends
    result["valid"] = True
    result["reason"] = "two_real_prefill_forwards_in_window" if expected == 2 else "single_forward_in_window"
    return result


def classify(name, rules):
    low = (name or "").lower()
    for row in rules:
        if any(token in low for token in row["tokens"]):
            return row["category"]
    return "unknown"
