#!/usr/bin/env python3
"""Read-only E33 DeepInfer profile-control checker.

The tool reads the adapter's existing ``GET /profile/status`` endpoint (or a
saved copy of that payload) and reports five distinct stages:

    configured -> armed -> collecting -> exported -> qualified

Stage definitions are deliberately conservative:

* ``configured``: all expected rank entries are present and each worker has a
  ProfilePlan (or, after worker close, the final export-status identity).
* ``armed``: every expected rank was armed at least once.
* ``collecting``: at least one profiler window has already been exported.
  The adapter has no live "window started" event, so between arming and the
  first export only ``armed`` is observable.
* ``exported``: every expected window is reported by ``/profile/status`` with
  positive steps, ``export_ok=true`` and no errors.  This is still only the
  controller's own statement.
* ``qualified``: exported plus 8 unique ranks, a unique rank->device mapping,
  matching expected run/case, locally readable manifests with the expected
  window/step/steady-state fields, and a passing independent
  ``npu_worker_profile.py --check-cohort`` JSON for every window label.

The tool never starts, stops or arms a profile.  It does not invent a remote
endpoint.  Real capture is never claimed by this tool alone.

Examples (local fixtures only):
    python3 tools/e33_profile_control.py \
        --status-json data/fixtures/e33/status-8-unarmed.json \
        --expected-run-id E33-fixture-run \
        --expected-case-id E33-fixture-case

    python3 tools/e33_profile_control.py --base-url http://127.0.0.1:18121 \
        --include-status --expected-run-id E33-DI-L32K-C32-NORMAL \
        --expected-case-id L32K-C32-N64-128OUT-0CACHE \
        --expect-prompt-windows 1 --expect-decode-windows 1 \
        --path-map /work/performance-20261021=/host/mounted/performance-20261021 \
        --trace-check-json evidence/E33-trace-check-prompt.json \
        --trace-check-json evidence/E33-trace-check-decode.json
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

TOOL = "e33_profile_control"
SCHEMA_VERSION = "1.0"
STAGES = ("configured", "armed", "collecting", "exported", "qualified")
STAGE_RANK = {name: index for index, name in enumerate(STAGES)}
DEFAULT_BASE_URL = "http://127.0.0.1:18121"
MAX_MANIFEST_BYTES = 64 * 1024 * 1024

LIMITS = [
    "This tool is read-only: it never starts, stops, arms or configures a profile.",
    "The adapter does not expose a live collecting flag or a window-start event; "
    "between arming and the first exported window only 'armed' is observable.",
    "After worker close, the adapter replaces the plan-bearing status entry with "
    "export_status(); run_id/case_id then require readable manifests or a trace check.",
    "window entries contain out_dir/manifest but no device; the rank->device mapping "
    "comes from the final identity or from each manifest.",
    "A detached docker exec start proves only that the process was launched; it does "
    "not prove that any rank armed or exported a trace.",
    "qualified is a control-plane verdict only. It requires independent "
    "npu_worker_profile.py --check-cohort JSON and does not replace a full trace review.",
    "Any server, container or NPU fact must be verified by the main controller; local "
    "fixtures in this repository are synthetic and can never be qualified.",
]

INTERFACE = {
    "endpoint": "GET /profile/status -> {\"profile_status\": [entry, ...]}",
    "profile_ready_entry": {"rank": "int", "plan": "ProfilePlan.public()", "armed": False},
    "profile_armed_entry": {
        "rank": "int", "plan": "ProfilePlan.public()", "armed": True,
        "request_ids": "list[str]",
    },
    "final_entry": {
        "rank": "int", "identity": "worker identity dict", "armed": "bool",
        "disarmed_reason": "str|null", "windows": "list[window]",
        "errors": "list[dict]", "records": "int",
    },
    "window": {
        "label": "str", "out_dir": "str", "manifest": "str",
        "engine_steps": "[first,last]", "steps": "int",
        "phase_counts": "dict", "export_ok": "bool", "error": "dict|null",
    },
    "identity_device": "identity.logical_device (int); physical_card is a separate string",
    "notes": [
        "ProfilePlan.public() fields include run_id, case_id, mode, steps, "
        "prompt_windows, decode_windows, arm_after_dispatched and arm_lead.",
        "A window is appended only after its profiler session stops and writes its "
        "manifest; /profile/status does not update while a window is running.",
        "The final entry does not contain the plan; window manifests embed "
        "profile_plan and run_id/case_id/device.",
        "The adapter does not expose prompt_windows/decode_windows as top-level "
        "per-rank status fields; count windows[].label instead.",
    ],
}


class ProfileControlError(RuntimeError):
    """Input or interface error that prevents a status evaluation."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _uniq(values):
    seen = set()
    output = []
    for value in values:
        if value is None:
            continue
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def _to_int(value):
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if text.lstrip("-").isdigit():
            return int(text)
    return None


def _fetch_json(url: str, timeout: float) -> dict:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, OSError) as exc:
        raise ProfileControlError("cannot read %s: %r" % (url, exc)) from exc
    if not isinstance(payload, dict):
        raise ProfileControlError("%s returned %s, expected a JSON object"
                                  % (url, type(payload).__name__))
    return payload


def _load_json_file(path: str, label: str):
    if path == "-":
        try:
            return json.load(sys.stdin)
        except ValueError as exc:
            raise ProfileControlError("cannot parse stdin as %s: %r" % (label, exc)) from exc
    source = Path(path)
    try:
        return json.loads(source.read_text())
    except OSError as exc:
        raise ProfileControlError("cannot read %s %s: %r" % (label, path, exc)) from exc
    except ValueError as exc:
        raise ProfileControlError("cannot parse %s %s: %r" % (label, path, exc)) from exc


def _parse_path_maps(values):
    maps = []
    for raw in values or []:
        if "=" not in raw:
            raise ProfileControlError("--path-map must be CONTAINER_PREFIX=HOST_PREFIX, got %r" % raw)
        source, destination = raw.split("=", 1)
        source = source.rstrip("/")
        destination = destination.rstrip("/")
        if not source or not destination:
            raise ProfileControlError("--path-map has an empty prefix: %r" % raw)
        maps.append((source, destination))
    return sorted(maps, key=lambda item: len(item[0]), reverse=True)


def _resolve_path(raw, path_maps):
    """Return (candidate, mapped, exists).  Never raises for a missing export."""
    if not raw:
        return None, False, False
    direct = Path(str(raw))
    if direct.exists():
        return direct, False, True
    for source, destination in path_maps:
        if str(raw) == source or str(raw).startswith(source + "/"):
            relative = str(raw)[len(source):].lstrip("/")
            candidate = Path(destination) / relative
            return candidate, True, candidate.exists()
    return direct, False, False


def _window_kind(label: str) -> str:
    if label.startswith("prompt"):
        return "prompt"
    if label.startswith("decode"):
        return "decode"
    return "other"


def _entry_view(slot: int, raw) -> dict:
    if not isinstance(raw, dict) or not raw:
        return {
            "slot": slot, "rank": None, "present": False, "raw": raw,
            "plan": None, "identity": None, "armed": None,
            "disarmed_reason": None, "windows": [], "errors": [],
            "request_ids": [], "records": None,
        }
    plan = raw.get("plan") if isinstance(raw.get("plan"), dict) else None
    identity = raw.get("identity") if isinstance(raw.get("identity"), dict) else None
    rank = _to_int(raw.get("rank"))
    if rank is None and (plan is not None or identity is not None):
        rank = slot
    windows = raw.get("windows") if isinstance(raw.get("windows"), list) else []
    errors = raw.get("errors") if isinstance(raw.get("errors"), list) else []
    request_ids = raw.get("request_ids") if isinstance(raw.get("request_ids"), list) else []
    return {
        "slot": slot, "rank": rank, "present": True, "raw": raw,
        "plan": plan, "identity": identity,
        "armed": raw.get("armed") if isinstance(raw.get("armed"), bool) else None,
        "disarmed_reason": raw.get("disarmed_reason"),
        "windows": [item for item in windows if isinstance(item, dict)],
        "errors": [item for item in errors if item is not None],
        "request_ids": [str(item) for item in request_ids],
        "records": _to_int(raw.get("records")),
    }


def _plan_view(plan) -> dict:
    if not isinstance(plan, dict):
        return {}
    mode = plan.get("mode")
    prompt_windows = _to_int(plan.get("prompt_windows"))
    decode_windows = _to_int(plan.get("decode_windows"))
    if mode == "prompt":
        decode_windows = 0
    elif mode == "decode":
        prompt_windows = 0
    return {
        "run_id": plan.get("run_id"),
        "case_id": plan.get("case_id"),
        "mode": mode,
        "steps": _to_int(plan.get("steps")),
        "prompt_windows": prompt_windows,
        "decode_windows": decode_windows,
        "arm_after_dispatched": _to_int(plan.get("arm_after_dispatched")),
        "arm_lead": _to_int(plan.get("arm_lead")),
        "request_ids": [str(item) for item in (plan.get("request_ids") or [])
                        if item is not None],
        "out_dir": plan.get("out_dir"),
    }


def _plan_problems(views, source: str):
    problems = []
    if not views:
        return problems

    def values(field):
        return {json.dumps(view.get(field), sort_keys=True) for view in views}

    for field in ("run_id", "case_id", "mode", "steps", "prompt_windows",
                  "decode_windows", "arm_after_dispatched", "arm_lead",
                  "request_ids"):
        if len(values(field)) > 1:
            problems.append("%s_%s_inconsistent" % (source, field))
    return problems


def _read_manifest(window, path_maps):
    raw_path = window.get("manifest")
    if not raw_path and window.get("out_dir"):
        raw_path = str(Path(str(window["out_dir"])) / "profile_manifest.json")
    if not raw_path:
        return {"readable": False, "path": None, "raw_path": None,
                "mapped": False, "reason": "manifest_path_missing", "data": None}
    candidate, mapped, exists = _resolve_path(raw_path, path_maps)
    if not exists:
        return {"readable": False, "path": str(candidate) if candidate else None,
                "raw_path": str(raw_path), "mapped": mapped,
                "reason": "manifest_not_found", "data": None}
    try:
        if candidate.stat().st_size > MAX_MANIFEST_BYTES:
            return {"readable": False, "path": str(candidate), "raw_path": str(raw_path),
                    "mapped": mapped, "reason": "manifest_too_large", "data": None}
        data = json.loads(candidate.read_text())
    except OSError as exc:
        return {"readable": False, "path": str(candidate), "raw_path": str(raw_path),
                "mapped": mapped, "reason": "manifest_unreadable", "error": repr(exc),
                "data": None}
    except ValueError as exc:
        return {"readable": False, "path": str(candidate), "raw_path": str(raw_path),
                "mapped": mapped, "reason": "manifest_invalid_json", "error": repr(exc),
                "data": None}
    if not isinstance(data, dict):
        return {"readable": False, "path": str(candidate), "raw_path": str(raw_path),
                "mapped": mapped, "reason": "manifest_not_object", "data": None}
    return {"readable": True, "path": str(candidate), "raw_path": str(raw_path),
            "mapped": mapped, "reason": None, "data": data}


def _probe_trace_file(path: Path, kind: str):
    """Cheap structural probe; deep validation belongs to npu_worker_profile.py."""
    try:
        size = path.stat().st_size
    except OSError as exc:
        return False, "stat_failed: %r" % (exc,)
    if not path.is_file():
        return False, "not_a_file"
    if size <= 0:
        return False, "empty_file"
    if kind == "csv":
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                header = handle.readline().strip()
                first_row = handle.readline().strip()
        except OSError as exc:
            return False, "read_failed: %r" % (exc,)
        if not header:
            return False, "csv_header_missing"
        if not first_row:
            return False, "csv_has_no_rows"
    elif kind == "trace_view":
        try:
            with path.open("rb") as handle:
                prefix = handle.read(4096)
        except OSError as exc:
            return False, "read_failed: %r" % (exc,)
        stripped = prefix.lstrip()
        if not stripped or stripped[:1] not in (b"{", b"["):
            return False, "not_json_object_or_array"
    return True, "ok"


def _manifest_verification(window, manifest, expected, identity_device, min_steps):
    result = {
        "rank": window.get("rank"),
        "label": window.get("label"),
        "kind": window.get("kind"),
        "manifest_path": manifest.get("path"),
        "raw_manifest_path": manifest.get("raw_path"),
        "manifest_mapped": bool(manifest.get("mapped")),
        "manifest_readable": bool(manifest.get("readable")),
        "verified": False,
        "structural_ok": False,
        "run_case_ok": None,
        "synthetic": None,
        "steps": window.get("steps"),
        "records": None,
        "device": {"manifest": None, "identity": identity_device, "match": None},
        "trace_files": {"kernel_details": [], "trace_view": [], "missing": [], "empty": []},
        "reasons": [],
        "checks": {},
    }
    reasons = result["reasons"]

    def fail(code):
        if code not in reasons:
            reasons.append(code)

    if not manifest.get("readable"):
        fail(manifest.get("reason") or "manifest_unreadable")
        return result

    data = manifest["data"]
    export = data.get("export") if isinstance(data.get("export"), dict) else {}
    steady = data.get("steady_state") if isinstance(data.get("steady_state"), dict) else {}
    records = data.get("profiler_step_records")
    records_list = records if isinstance(records, list) else []
    manifest_steps = _to_int(data.get("steps"))
    manifest_device = _to_int(data.get("device"))
    result["steps"] = manifest_steps if manifest_steps is not None else result["steps"]
    result["records"] = len(records_list)
    result["synthetic"] = data.get("synthetic") if isinstance(data.get("synthetic"), bool) else None
    result["device"] = {
        "manifest": manifest_device,
        "identity": identity_device,
        "match": (manifest_device == identity_device
                  if manifest_device is not None and identity_device is not None else None),
    }

    checks = result["checks"]
    checks["rank"] = {"pass": _to_int(data.get("rank")) == window.get("rank"),
                      "detail": "manifest=%r expected=%r"
                                % (data.get("rank"), window.get("rank"))}
    checks["window_label"] = {"pass": data.get("window_label") == window.get("label"),
                              "detail": "manifest=%r expected=%r"
                                        % (data.get("window_label"), window.get("label"))}
    checks["run_id"] = {"pass": (str(data.get("run_id")) == str(expected.get("run_id"))
                                 if expected.get("run_id") else None),
                        "detail": "manifest=%r expected=%r"
                                  % (data.get("run_id"), expected.get("run_id"))}
    checks["case_id"] = {"pass": (str(data.get("case_id")) == str(expected.get("case_id"))
                                  if expected.get("case_id") else None),
                         "detail": "manifest=%r expected=%r"
                                   % (data.get("case_id"), expected.get("case_id"))}
    checks["steps_nonzero"] = {"pass": manifest_steps is not None and manifest_steps > 0,
                               "detail": "steps=%r" % (manifest_steps,)}
    if expected.get("steps") is not None:
        checks["steps_match"] = {"pass": manifest_steps == expected.get("steps"),
                                 "detail": "manifest=%r expected=%r"
                                           % (manifest_steps, expected.get("steps"))}
    else:
        checks["steps_min"] = {"pass": manifest_steps is not None and manifest_steps >= min_steps,
                               "detail": "manifest=%r min=%d" % (manifest_steps, min_steps)}
    checks["step_records"] = {
        "pass": bool(records_list) and manifest_steps is not None
                and len(records_list) == manifest_steps and len(records_list) >= min_steps,
        "detail": "records=%d steps=%r min=%d"
                  % (len(records_list), manifest_steps, min_steps),
    }
    checks["stopped"] = {
        "pass": data.get("stopped") is True and data.get("stop_complete") is True,
        "detail": "stopped=%r stop_complete=%r"
                  % (data.get("stopped"), data.get("stop_complete")),
    }
    checks["synthetic_false"] = {
        "pass": data.get("synthetic") is False,
        "detail": "synthetic=%r (must be explicitly false)" % (data.get("synthetic"),),
    }
    checks["steady_state"] = {
        "pass": steady.get("present") is True and bool(steady.get("ranges")),
        "detail": "present=%r ranges=%d"
                  % (steady.get("present"), len(steady.get("ranges") or [])),
    }
    checks["export_ok"] = {
        "pass": export.get("ok") is True,
        "detail": "export.ok=%r" % (export.get("ok"),),
    }
    kernel_paths = export.get("kernel_details") if isinstance(export.get("kernel_details"), list) else []
    view_paths = export.get("trace_view") if isinstance(export.get("trace_view"), list) else []
    checks["export_lists"] = {
        "pass": bool(kernel_paths) and bool(view_paths),
        "detail": "kernel_details=%d trace_view=%d"
                  % (len(kernel_paths), len(view_paths)),
    }
    result["trace_files"]["kernel_details"] = [str(item) for item in kernel_paths]
    result["trace_files"]["trace_view"] = [str(item) for item in view_paths]

    file_ok = True
    for raw_path in kernel_paths:
        path, _mapped, exists = _resolve_path(raw_path, expected.get("_path_maps") or [])
        if not exists:
            result["trace_files"]["missing"].append(str(raw_path))
            file_ok = False
            continue
        ok, detail = _probe_trace_file(path, "csv")
        if not ok:
            result["trace_files"]["empty"].append("%s: %s" % (path, detail))
            file_ok = False
    for raw_path in view_paths:
        path, _mapped, exists = _resolve_path(raw_path, expected.get("_path_maps") or [])
        if not exists:
            result["trace_files"]["missing"].append(str(raw_path))
            file_ok = False
            continue
        ok, detail = _probe_trace_file(path, "trace_view")
        if not ok:
            result["trace_files"]["empty"].append("%s: %s" % (path, detail))
            file_ok = False
    checks["trace_files_readable"] = {
        "pass": file_ok and bool(kernel_paths) and bool(view_paths),
        "detail": "missing=%d unusable=%d"
                  % (len(result["trace_files"]["missing"]),
                     len(result["trace_files"]["empty"])),
    }

    for name, check in checks.items():
        if check.get("pass") is False:
            fail("manifest_%s" % name)
    if result["synthetic"] is not True:
        if result["synthetic"] is not False:
            fail("manifest_synthetic_not_false")
    if result["device"]["match"] is False:
        fail("manifest_device_mismatch")
    if manifest_device is None:
        fail("manifest_device_missing")

    run_case_values = [checks["run_id"].get("pass"), checks["case_id"].get("pass")]
    if any(value is False for value in run_case_values):
        result["run_case_ok"] = False
        fail("manifest_run_case_mismatch")
    elif all(value is True for value in run_case_values):
        result["run_case_ok"] = True

    structural_names = (
        "rank", "window_label", "steps_nonzero", "steps_match", "steps_min",
        "step_records", "stopped", "synthetic_false", "steady_state",
        "export_ok", "export_lists", "trace_files_readable",
    )
    structural_checks = [checks[name]["pass"] for name in structural_names
                         if name in checks and checks[name].get("pass") is not None]
    result["structural_ok"] = bool(structural_checks) and all(structural_checks)
    result["verified"] = bool(result["structural_ok"] and result["run_case_ok"] is not False)
    if not result["structural_ok"] and not reasons:
        fail("manifest_structural_failure")
    return result


def _load_trace_checks(paths):
    results = []
    for raw in paths or []:
        try:
            data = _load_json_file(raw, "trace-check JSON")
        except ProfileControlError as exc:
            results.append({"path": raw, "readable": False, "error": str(exc),
                            "data": None})
            continue
        if not isinstance(data, dict):
            results.append({"path": raw, "readable": False,
                            "error": "trace-check JSON is not an object", "data": None})
            continue
        results.append({"path": raw, "readable": True, "error": None, "data": data})
    return results


def _trace_check_summary(inputs, expected_labels, rank_devices, expect_ranks):
    summary = {
        "provided": bool(inputs),
        "pass": False,
        "reason": None,
        "expected_labels": sorted(expected_labels),
        "results": [],
    }
    if not inputs:
        summary["reason"] = "independent_trace_check_missing"
        return summary
    if not expected_labels:
        summary["reason"] = "no_expected_window_labels"
        return summary

    by_label = {}
    unknown_labels = set()
    for item in inputs:
        view = {"path": item["path"], "readable": item["readable"],
                "label": None, "pass": False, "reasons": []}
        summary["results"].append(view)
        if not item["readable"]:
            view["reasons"].append("trace_check_unreadable")
            continue
        data = item["data"]
        label = data.get("window")
        if label is None and len(expected_labels) == 1:
            label = next(iter(expected_labels))
        view["label"] = label
        if not isinstance(label, str) or not label:
            view["reasons"].append("trace_check_window_ambiguous")
            continue
        if label not in expected_labels:
            unknown_labels.add(label)
            view["reasons"].append("trace_check_unknown_window")
        if data.get("qualified") is not True:
            view["reasons"].append("trace_check_not_qualified")
        if data.get("ok") is not True:
            view["reasons"].append("trace_check_not_ok")
        if _to_int(data.get("expect_ranks")) != expect_ranks:
            view["reasons"].append("trace_check_rank_count")
        ranks = data.get("ranks") if isinstance(data.get("ranks"), list) else []
        seen_ranks = set()
        for rank_item in ranks:
            if not isinstance(rank_item, dict):
                view["reasons"].append("trace_check_rank_entry_invalid")
                continue
            rank = _to_int(rank_item.get("rank"))
            if rank is None:
                view["reasons"].append("trace_check_rank_missing")
                continue
            seen_ranks.add(rank)
            if rank_item.get("qualified") is not True:
                view["reasons"].append("trace_check_rank_not_qualified:%d" % rank)
            device = _to_int(rank_item.get("device"))
            expected_device = rank_devices.get(rank)
            if device is None:
                view["reasons"].append("trace_check_device_missing:%d" % rank)
            elif expected_device is not None and device != expected_device:
                view["reasons"].append("trace_check_device_mismatch:%d" % rank)
            kernels = _to_int(rank_item.get("kernels_in_steady_window"))
            if kernels is None or kernels <= 0:
                view["reasons"].append("trace_check_no_kernels_in_window:%d" % rank)
        if seen_ranks != set(range(expect_ranks)):
            view["reasons"].append("trace_check_rank_coverage")
        view["reasons"] = _uniq(view["reasons"])
        view["pass"] = not view["reasons"]
        by_label.setdefault(label, []).append(view)

    missing = []
    failed = []
    for label in sorted(expected_labels):
        views = by_label.get(label) or []
        if not views:
            missing.append(label)
            continue
        if any(not view["pass"] for view in views):
            failed.append(label)
    if unknown_labels:
        summary["reason"] = "unknown_window_labels"
    elif missing:
        summary["reason"] = "missing_labels:" + ",".join(missing)
    elif failed:
        summary["reason"] = "failed_labels:" + ",".join(failed)
    elif any(not view["pass"] for view in summary["results"]):
        summary["reason"] = "trace_check_failed"
    else:
        summary["pass"] = True
        summary["reason"] = "all_window_labels_passed"
    return summary


def _stage(pass_value, reasons):
    return {"pass": bool(pass_value), "reasons": _uniq(reasons)}


def evaluate(args):
    if args.expect_ranks <= 0:
        raise ProfileControlError("--expect-ranks must be positive")
    if args.min_steps < 1:
        raise ProfileControlError("--min-steps must be >= 1")
    path_maps = _parse_path_maps(args.path_map)

    if args.status_json:
        payload = _load_json_file(args.status_json, "profile-status JSON")
        source = {"kind": "file", "path": args.status_json}
    else:
        url = args.base_url.rstrip("/") + "/profile/status"
        payload = _fetch_json(url, args.timeout)
        source = {"kind": "http", "url": url}

    if isinstance(payload, list):
        entries_raw = payload
        payload_meta = {}
    elif isinstance(payload, dict):
        entries_raw = payload.get("profile_status")
        payload_meta = payload
    else:
        raise ProfileControlError("profile status payload must be an object or list")
    if not isinstance(entries_raw, list):
        raise ProfileControlError("payload has no profile_status list")
    fixture_meta = payload_meta.get("fixture") if isinstance(payload_meta, dict) else None
    synthetic_fixture = bool(isinstance(fixture_meta, dict) and fixture_meta.get("synthetic"))

    entries = [_entry_view(slot, raw) for slot, raw in enumerate(entries_raw)]
    present_entries = [entry for entry in entries if entry["present"]]
    by_rank = {}
    duplicate_ranks = []
    for entry in present_entries:
        rank = entry["rank"]
        if rank is None:
            continue
        if rank in by_rank:
            duplicate_ranks.append(rank)
        else:
            by_rank[rank] = entry

    expected_ranks = list(range(args.expect_ranks))
    present_ranks = sorted(by_rank)
    missing_ranks = [rank for rank in expected_ranks if rank not in by_rank]
    unexpected_ranks = [rank for rank in present_ranks if rank not in expected_ranks]

    plans = {rank: entry["plan"] for rank, entry in by_rank.items()
             if isinstance(entry["plan"], dict)}
    plan_views = {rank: _plan_view(plan) for rank, plan in plans.items()}
    plan_consistency = _plan_problems(list(plan_views.values()), "plan")

    windows = []
    window_duplicates = []
    for rank in sorted(by_rank):
        seen_labels = set()
        for raw_window in by_rank[rank]["windows"]:
            label = str(raw_window.get("label") or "")
            if not label:
                continue
            if label in seen_labels:
                window_duplicates.append({"rank": rank, "label": label})
            seen_labels.add(label)
            windows.append({
                "rank": rank,
                "label": label,
                "kind": _window_kind(label),
                "steps": _to_int(raw_window.get("steps")),
                "export_ok": raw_window.get("export_ok") is True,
                "error": raw_window.get("error"),
                "out_dir": raw_window.get("out_dir"),
                "manifest": raw_window.get("manifest"),
                "engine_steps": raw_window.get("engine_steps")
                               if isinstance(raw_window.get("engine_steps"), list) else [],
            })

    manifest_inputs = []
    for window in windows:
        manifest = _read_manifest(window, path_maps)
        manifest_inputs.append((window, manifest))

    # Expected values: CLI > plans > manifests.  A plan is authoritative while
    # it is present; after worker close only manifests still carry run/case.
    manifest_plan_views = []
    manifest_run_ids = []
    manifest_case_ids = []
    for _window, manifest in manifest_inputs:
        if not manifest.get("readable"):
            continue
        data = manifest["data"]
        manifest_run_ids.append(data.get("run_id"))
        manifest_case_ids.append(data.get("case_id"))
        plan = data.get("profile_plan")
        if isinstance(plan, dict):
            manifest_plan_views.append(_plan_view(plan))
    manifest_plan_consistency = _plan_problems(manifest_plan_views, "manifest_plan")

    expected = {
        "run_id": args.expected_run_id,
        "case_id": args.expected_case_id,
        "ranks": args.expect_ranks,
        "mode": args.expect_mode,
        "prompt_windows": args.expect_prompt_windows,
        "decode_windows": args.expect_decode_windows,
        "steps": args.expect_steps,
        "arm_after_dispatched": None,
        "arm_lead": None,
    }
    sources = {name: ("cli" if value is not None else None)
               for name, value in expected.items() if name != "ranks"}

    def fill_from(views, source_name):
        if not views:
            return
        first = views[0]
        for field in ("run_id", "case_id", "mode", "prompt_windows", "decode_windows",
                      "steps", "arm_after_dispatched", "arm_lead"):
            if expected.get(field) is None and first.get(field) is not None:
                expected[field] = first[field]
                sources[field] = source_name

    fill_from(list(plan_views.values()), "plan")
    fill_from(manifest_plan_views, "manifest")

    if expected["mode"] == "prompt" and expected["decode_windows"] is None:
        expected["decode_windows"] = 0
        sources["decode_windows"] = sources.get("mode") or "mode"
    elif expected["mode"] == "decode" and expected["prompt_windows"] is None:
        expected["prompt_windows"] = 0
        sources["prompt_windows"] = sources.get("mode") or "mode"
    # An explicit CLI mode is authoritative over a plan whose mode differs:
    # a prompt-only expectation cannot also require decode windows unless the
    # operator passed --expect-decode-windows explicitly.
    if args.expect_mode == "prompt" and args.expect_decode_windows is None:
        expected["decode_windows"] = 0
        sources["decode_windows"] = "cli_mode"
    elif args.expect_mode == "decode" and args.expect_prompt_windows is None:
        expected["prompt_windows"] = 0
        sources["prompt_windows"] = "cli_mode"

    expected_known = (expected["prompt_windows"] is not None
                      and expected["decode_windows"] is not None)
    expected_total = ((expected["prompt_windows"] or 0) + (expected["decode_windows"] or 0)
                      if expected_known else None)
    expected["_path_maps"] = path_maps

    # The E32 failure mode was a plan with no trigger at all.  Keep it visible
    # even though "configured" itself only means the ProfilePlan is installed.
    all_plan_views = list(plan_views.values()) + manifest_plan_views
    plan_trigger = {
        "arm_after_dispatched": sorted({
            view.get("arm_after_dispatched") for view in all_plan_views
            if view.get("arm_after_dispatched") is not None}),
        "arm_lead": sorted({
            view.get("arm_lead") for view in all_plan_views
            if view.get("arm_lead") is not None}),
        "request_ids_counts": sorted({
            len(view.get("request_ids") or []) for view in all_plan_views}),
    }
    warnings = []
    if all_plan_views:
        has_trigger = (any(value > 0 for value in plan_trigger["arm_after_dispatched"])
                       or any(count > 0 for count in plan_trigger["request_ids_counts"]))
        if not has_trigger:
            warnings.append("plan_has_no_arming_trigger")

    # Window inventory.
    windows_by_rank = {}
    for window in windows:
        windows_by_rank.setdefault(window["rank"], []).append(window)
    prompt_total = sum(1 for window in windows if window["kind"] == "prompt")
    decode_total = sum(1 for window in windows if window["kind"] == "decode")
    other_total = sum(1 for window in windows if window["kind"] == "other")
    ranks_with_windows = sorted(windows_by_rank)
    expected_labels = sorted({window["label"] for window in windows})

    # Rank/device table.
    rank_rows = []
    rank_devices = {}
    device_reasons = []
    manifest_device_by_rank = {}
    for window, manifest in manifest_inputs:
        if not manifest.get("readable"):
            continue
        device = _to_int(manifest["data"].get("device"))
        if device is not None:
            manifest_device_by_rank.setdefault(window["rank"], set()).add(device)
    for rank in expected_ranks:
        entry = by_rank.get(rank)
        identity = entry["identity"] if entry and isinstance(entry["identity"], dict) else {}
        identity_device = _to_int(identity.get("logical_device"))
        identity_rank = _to_int(identity.get("rank"))
        manifest_devices = sorted(manifest_device_by_rank.get(rank, set()))
        manifest_conflict = len(manifest_devices) > 1
        manifest_device = manifest_devices[0] if len(manifest_devices) == 1 else None
        chosen_device = identity_device if identity_device is not None else manifest_device
        device_match = (identity_device == manifest_device
                        if identity_device is not None and manifest_device is not None else None)
        if manifest_conflict:
            device_reasons.append("manifest_device_conflict:%d" % rank)
        if identity_rank is not None and identity_rank != rank:
            device_reasons.append("identity_rank_mismatch:%d" % rank)
        if device_match is False:
            device_reasons.append("identity_manifest_device_mismatch:%d" % rank)
        if chosen_device is None:
            device_reasons.append("device_missing:%d" % rank)
        else:
            rank_devices[rank] = chosen_device
        rank_windows = windows_by_rank.get(rank, [])
        rank_rows.append({
            "rank": rank,
            "present": entry is not None,
            "plan_present": bool(entry and entry["plan"]),
            "identity_present": bool(entry and entry["identity"]),
            "identity_rank": identity_rank,
            "identity_device": identity_device,
            "physical_card": identity.get("physical_card"),
            "manifest_devices": manifest_devices,
            "manifest_device_conflict": manifest_conflict,
            "chosen_device": chosen_device,
            "device_match": device_match,
            "currently_armed": bool(entry and entry["armed"] is True),
            "ever_armed": bool(entry and (
                entry["armed"] is True or entry["windows"]
                or entry["disarmed_reason"] is not None or entry["request_ids"])),
            "disarmed_reason": entry["disarmed_reason"] if entry else None,
            "arm_request_ids": entry["request_ids"] if entry else [],
            "windows": {
                "prompt": sum(1 for window in rank_windows if window["kind"] == "prompt"),
                "decode": sum(1 for window in rank_windows if window["kind"] == "decode"),
                "other": sum(1 for window in rank_windows if window["kind"] == "other"),
                "total": len(rank_windows),
            },
            "window_steps": [window["steps"] for window in rank_windows],
            "errors": entry["errors"] if entry else [],
        })
    device_values = [rank_devices[rank] for rank in expected_ranks if rank in rank_devices]
    duplicate_devices = sorted({value for value in device_values
                                if device_values.count(value) > 1})
    if duplicate_devices:
        device_reasons.append("device_not_unique:%s"
                              % ",".join(str(value) for value in duplicate_devices))
    device_ok = (not missing_ranks and not duplicate_ranks and not unexpected_ranks
                 and not device_reasons
                 and len(rank_devices) == args.expect_ranks)

    # Error inventory.
    rank_errors = [{"rank": row["rank"], "errors": row["errors"]}
                   for row in rank_rows if row["errors"]]
    window_errors = [{"rank": window["rank"], "label": window["label"],
                      "error": window["error"]}
                     for window in windows if window["error"]]

    # Manifest verification.
    verified_windows = []
    structural_ok_all = True
    any_unreadable = False
    for window, manifest in manifest_inputs:
        verification = _manifest_verification(
            window, manifest, expected,
            rank_devices.get(window["rank"]), args.min_steps)
        verified_windows.append(verification)
        if not verification["manifest_readable"]:
            any_unreadable = True
        if not verification["structural_ok"]:
            structural_ok_all = False
    exports_verified_all = bool(verified_windows) and all(
        item["verified"] for item in verified_windows) and structural_ok_all

    # Run/case source comparison.
    run_sources = [view.get("run_id") for view in plan_views.values()]
    case_sources = [view.get("case_id") for view in plan_views.values()]
    run_sources += manifest_run_ids
    case_sources += manifest_case_ids
    if not expected["run_id"]:
        run_case = {"run_id": None, "case_id": None, "pass": None,
                    "detail": "expected run/case not supplied"}
    else:
        run_mismatch = any(str(value) != str(expected["run_id"])
                           for value in run_sources if value is not None)
        case_mismatch = (any(str(value) != str(expected["case_id"])
                             for value in case_sources if value is not None)
                         if expected["case_id"] else False)
        has_run_source = any(value is not None for value in run_sources)
        has_case_source = any(value is not None for value in case_sources)
        run_state = ("mismatch" if run_mismatch
                     else ("pass" if has_run_source else "unverified"))
        case_state = (None if not expected["case_id"]
                      else ("mismatch" if case_mismatch
                            else ("pass" if has_case_source else "unverified")))
        if run_mismatch or case_mismatch:
            run_case_pass = False
        elif run_state == "pass" and case_state in ("pass", None):
            run_case_pass = True
        else:
            run_case_pass = None
        run_case = {
            "run_id": run_state,
            "case_id": case_state,
            "pass": run_case_pass,
            "detail": "run_sources=%r case_sources=%r expected=%r/%r"
                      % (run_sources, case_sources,
                         expected["run_id"], expected["case_id"]),
        }

    # Optional /status facts.
    server = None
    server_error = None
    if args.include_status or args.server_status_json:
        try:
            if args.server_status_json:
                server_payload = _load_json_file(args.server_status_json, "server-status JSON")
            else:
                server_payload = _fetch_json(args.base_url.rstrip("/") + "/status", args.timeout)
            if not isinstance(server_payload, dict):
                raise ProfileControlError("server status is not an object")
            server = {
                "ready": server_payload.get("ready"),
                "ranks_ready": server_payload.get("ranks_ready"),
                "server_idle": server_payload.get("server_idle"),
                "inflight": server_payload.get("inflight"),
                "running": server_payload.get("running"),
                "waiting": server_payload.get("waiting"),
                "fatal": server_payload.get("fatal"),
                "ready_order": server_payload.get("ready_order"),
            }
        except ProfileControlError as exc:
            server_error = str(exc)

    server_fatal = bool(server and server.get("fatal"))
    live_collection_inferred = bool(
        server and not windows
        and any(row["present"] for row in rank_rows)
        and all(row["currently_armed"] for row in rank_rows if row["present"])
        and ((server.get("inflight") or 0) > 0 or server.get("server_idle") is False)
    )

    # Stage 1: configured.
    configured_reasons = []
    if not present_entries:
        configured_reasons.append("no_profile_status")
    if missing_ranks:
        configured_reasons.append("missing_ranks:%s"
                                  % ",".join(str(rank) for rank in missing_ranks))
    if duplicate_ranks:
        configured_reasons.append("duplicate_ranks:%s"
                                  % ",".join(str(rank) for rank in sorted(set(duplicate_ranks))))
    if unexpected_ranks:
        configured_reasons.append("unexpected_ranks:%s"
                                  % ",".join(str(rank) for rank in unexpected_ranks))
    for rank in expected_ranks:
        entry = by_rank.get(rank)
        if entry is None:
            continue
        if not entry["plan"] and not entry["identity"]:
            configured_reasons.append("rank_not_configured:%d" % rank)
    if plans and len(plans) != args.expect_ranks:
        configured_reasons.append("plan_missing_on_ranks")
    configured_reasons.extend(plan_consistency)
    if expected["run_id"]:
        for rank, view in plan_views.items():
            if str(view.get("run_id")) != str(expected["run_id"]):
                configured_reasons.append("plan_run_id_mismatch:%d" % rank)
    if expected["case_id"]:
        for rank, view in plan_views.items():
            if str(view.get("case_id")) != str(expected["case_id"]):
                configured_reasons.append("plan_case_id_mismatch:%d" % rank)
    configured_pass = not configured_reasons

    # Stage 2: armed.
    armed_reasons = []
    if not configured_pass:
        armed_reasons.append("not_configured")
    not_armed = [row["rank"] for row in rank_rows
                 if not row["present"] or not row["ever_armed"]]
    if not_armed:
        armed_reasons.append("not_armed:%s"
                             % ",".join(str(rank) for rank in not_armed))
    armed_pass = configured_pass and not not_armed

    # Stage 3: collecting.
    collecting_reasons = []
    if not armed_pass:
        collecting_reasons.append("not_armed")
    if not windows:
        collecting_reasons.append("no_windows_exported")
    collecting_pass = armed_pass and bool(windows)

    # Stage 4: exported.
    exported_reasons = []
    if not collecting_pass:
        exported_reasons.append("not_collecting")
    if not expected_known:
        exported_reasons.append("expected_windows_unknown")
    if collecting_pass:
        if window_duplicates:
            exported_reasons.append("window_duplicate_label")
        for row in rank_rows:
            wanted_prompt = expected["prompt_windows"]
            wanted_decode = expected["decode_windows"]
            if wanted_prompt is not None and row["windows"]["prompt"] != wanted_prompt:
                exported_reasons.append("window_count_mismatch:prompt:rank%d:%d!=%d"
                                        % (row["rank"], row["windows"]["prompt"], wanted_prompt))
            if wanted_decode is not None and row["windows"]["decode"] != wanted_decode:
                exported_reasons.append("window_count_mismatch:decode:rank%d:%d!=%d"
                                        % (row["rank"], row["windows"]["decode"], wanted_decode))
            if row["windows"]["other"]:
                exported_reasons.append("unexpected_window_label:rank%d" % row["rank"])
        for window in windows:
            steps = window["steps"]
            if steps is None or steps <= 0:
                exported_reasons.append("window_steps_zero:rank%d/%s"
                                        % (window["rank"], window["label"]))
            elif expected.get("steps") is not None and steps != expected["steps"]:
                exported_reasons.append("window_steps_short:rank%d/%s:%d!=%d"
                                        % (window["rank"], window["label"], steps,
                                           expected["steps"]))
            if not window["export_ok"]:
                exported_reasons.append("window_export_failed:rank%d/%s"
                                        % (window["rank"], window["label"]))
            if window["error"]:
                exported_reasons.append("window_error:rank%d/%s"
                                        % (window["rank"], window["label"]))
    if rank_errors:
        exported_reasons.append("rank_error:%s"
                                % ",".join(str(item["rank"]) for item in rank_errors))
    if server_fatal:
        exported_reasons.append("server_fatal")
    if run_case.get("pass") is False:
        exported_reasons.append("run_case_mismatch")
    exported_pass = collecting_pass and not exported_reasons

    # Stage 5: qualified.
    qualified_reasons = []
    if not exported_pass:
        qualified_reasons.append("not_exported")
    if not expected["run_id"]:
        qualified_reasons.append("expected_run_id_missing")
    if not expected["case_id"]:
        qualified_reasons.append("expected_case_id_missing")
    if expected.get("steps") is None:
        qualified_reasons.append("expected_steps_unknown")
    if run_case.get("pass") is None:
        qualified_reasons.append("run_case_unverified")
    elif run_case.get("pass") is False:
        qualified_reasons.append("run_case_mismatch")
    if not device_ok:
        qualified_reasons.extend(device_reasons or ["device_mapping_unverified"])
    if not exports_verified_all:
        if any_unreadable:
            qualified_reasons.append("export_manifests_not_verified")
        else:
            qualified_reasons.append("export_trace_not_verified")
    if not verified_windows:
        qualified_reasons.append("no_windows_to_verify")
    readable_synthetic = [item.get("synthetic") for item in verified_windows
                          if item.get("manifest_readable")]
    if readable_synthetic and any(value is not False for value in readable_synthetic):
        qualified_reasons.append("synthetic_or_unmarked_manifest")
    if rank_errors:
        qualified_reasons.append("rank_error:%s"
                                 % ",".join(str(item["rank"]) for item in rank_errors))
    if window_errors:
        qualified_reasons.append("window_error")
    if server_fatal:
        qualified_reasons.append("server_fatal")
    if (args.include_status or args.server_status_json) and server is None:
        qualified_reasons.append("server_status_unavailable")
    if server is not None and _to_int(server.get("ranks_ready")) != args.expect_ranks:
        qualified_reasons.append("server_ranks_ready_mismatch")
    if synthetic_fixture:
        qualified_reasons.append("synthetic_status_fixture")
    trace_summary = _trace_check_summary(
        _load_trace_checks(args.trace_check_json), set(expected_labels),
        rank_devices, args.expect_ranks)
    if not trace_summary["pass"]:
        qualified_reasons.append("independent_trace_check:" + str(trace_summary["reason"]))
    qualified_pass = (exported_pass and not qualified_reasons)

    stages = {
        "configured": _stage(configured_pass, configured_reasons),
        "armed": _stage(armed_pass, armed_reasons),
        "collecting": _stage(collecting_pass, collecting_reasons),
        "exported": _stage(exported_pass, exported_reasons),
        "qualified": _stage(qualified_pass, qualified_reasons),
    }
    stage = "none"
    blocking = []
    for name in STAGES:
        if stages[name]["pass"]:
            stage = name
        else:
            # Report only the first blocker; the later stages still carry their
            # own full reasons for anyone who needs the complete picture.
            blocking = stages[name]["reasons"]
            break
    if stage == "none":
        blocking = stages["configured"]["reasons"]

    if stage == "none":
        next_actions = [
            "No rank configuration is visible. Check that the adapter was started with "
            "PROFILE_DIR/PROFILE_RUN_ID and that all eight ranks reported profile_ready.",
        ]
    elif stage == "configured":
        next_actions = [
            "Profile is configured but no rank has armed. Check PROFILE_ARM_AFTER=65, "
            "PROFILE_REQUEST_IDS, and the actual dispatch count after the 64-request warmup.",
        ]
    elif stage == "armed":
        next_actions = [
            "Ranks are armed but no window has been exported. Check the trigger mode, "
            "arm lead and immediate/initial prompt-window settings; run the target cohort "
            "and poll again. The adapter cannot show a live window start before export.",
        ]
    elif stage == "collecting":
        next_actions = [
            "At least one window has exported but the planned set is incomplete. Inspect "
            "the per-rank errors and trigger/cadence settings; do not call this Run complete.",
        ]
    elif stage == "exported":
        next_actions = [
            "Collection is complete according to /profile/status. Provide readable export "
            "paths (--path-map) and independent npu_worker_profile.py --check-cohort JSON "
            "for every window label, then rerun for a qualified verdict.",
        ]
    else:
        next_actions = [
            "Control-plane qualification passed. Preserve the raw traces, manifests, "
            "source/config pins and the independent trace-check evidence.",
        ]

    output = {
        "tool": TOOL,
        "schema_version": SCHEMA_VERSION,
        "checked_utc": utc_now(),
        "input": dict(source, synthetic_fixture=synthetic_fixture,
                      expected_ranks=args.expect_ranks,
                      path_map=args.path_map,
                      trace_check_json=args.trace_check_json,
                      fixture=fixture_meta),
        "interface": INTERFACE,
        "expected": {
            "run_id": expected["run_id"],
            "case_id": expected["case_id"],
            "ranks": expected["ranks"],
            "mode": expected["mode"],
            "prompt_windows": expected["prompt_windows"],
            "decode_windows": expected["decode_windows"],
            "total_windows": expected_total,
            "steps": expected["steps"],
            "min_steps": args.min_steps,
            "arm_after_dispatched": expected.get("arm_after_dispatched"),
            "arm_lead": expected.get("arm_lead"),
            "sources": sources,
            "problems": _uniq(plan_consistency + manifest_plan_consistency),
        },
        "observed": {
            "ranks_present": present_ranks,
            "missing_ranks": missing_ranks,
            "unexpected_ranks": unexpected_ranks,
            "duplicate_ranks": sorted(set(duplicate_ranks)),
            "currently_armed_ranks": [row["rank"] for row in rank_rows
                                      if row["currently_armed"]],
            "ever_armed_ranks": [row["rank"] for row in rank_rows if row["ever_armed"]],
            "windows_total": len(windows),
            "prompt_windows_total": prompt_total,
            "decode_windows_total": decode_total,
            "other_windows_total": other_total,
            "ranks_with_windows": ranks_with_windows,
            "window_labels": expected_labels,
            "rank_errors": rank_errors,
            "window_errors": window_errors,
            "server": server,
            "server_error": server_error,
            "live_collection_inferred": live_collection_inferred,
            "plan_arming_trigger": plan_trigger,
            "warnings": warnings,
        },
        "ranks": rank_rows,
        "windows": verified_windows,
        "run_case": run_case,
        "device_mapping": {
            "pass": device_ok,
            "rank_devices": [{"rank": rank, "device": rank_devices.get(rank)}
                             for rank in expected_ranks],
            "duplicate_devices": duplicate_devices,
            "reasons": _uniq(device_reasons),
        },
        "exports": {
            "manifests_readable": sum(1 for item in verified_windows
                                      if item["manifest_readable"]),
            "manifests_total": len(verified_windows),
            "verified_total": sum(1 for item in verified_windows if item["verified"]),
            "all_verified": exports_verified_all,
        },
        "trace_check": trace_summary,
        "stages": stages,
        "stage": stage,
        "collection_complete": stages["exported"]["pass"],
        "qualified": stages["qualified"]["pass"],
        "blocking_reasons": _uniq(blocking),
        "next_actions": next_actions,
        "limits": LIMITS,
    }
    return output


def _parser():
    parser = argparse.ArgumentParser(
        description="Read-only E33 DeepInfer /profile/status stage checker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL,
                        help="adapter base URL (default: %(default)s)")
    parser.add_argument("--status-json", default=None,
                        help="read /profile/status payload from a file instead of HTTP ('-' = stdin)")
    parser.add_argument("--include-status", action="store_true",
                        help="also read GET /status for fatal/inflight/server_idle facts")
    parser.add_argument("--server-status-json", default=None,
                        help="read GET /status payload from a file")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--expected-run-id", default=None)
    parser.add_argument("--expected-case-id", default=None)
    parser.add_argument("--expect-ranks", type=int, default=8)
    parser.add_argument("--expect-prompt-windows", type=int, default=None)
    parser.add_argument("--expect-decode-windows", type=int, default=None)
    parser.add_argument("--expect-steps", type=int, default=None)
    parser.add_argument("--expect-mode", choices=("prompt", "decode", "both"),
                        default=None)
    parser.add_argument("--min-steps", type=int, default=1)
    parser.add_argument("--path-map", action="append", default=[],
                        metavar="CONTAINER_PREFIX=HOST_PREFIX",
                        help="map container export paths to a local mount; repeatable")
    parser.add_argument("--trace-check-json", action="append", default=[],
                        help="npu_worker_profile.py --check-cohort JSON for one window "
                             "label; repeat once per label")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--require-qualified", action="store_true",
                        help="exit 1 unless the qualified stage passes")
    parser.add_argument("--require-stage", choices=STAGES, default=None,
                        help="exit 1 unless this stage or a later one passes")
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = evaluate(args)
    except ProfileControlError as exc:
        error = {
            "tool": TOOL,
            "schema_version": SCHEMA_VERSION,
            "checked_utc": utc_now(),
            "ok": False,
            "error": str(exc),
        }
        if args.json_out:
            try:
                Path(args.json_out).write_text(json.dumps(error, indent=2, sort_keys=True) + "\n")
            except OSError:
                pass
        print(json.dumps(error, indent=2, sort_keys=True))
        return 2
    except Exception as exc:  # noqa: BLE001 - CLI must fail closed
        error = {
            "tool": TOOL,
            "schema_version": SCHEMA_VERSION,
            "checked_utc": utc_now(),
            "ok": False,
            "error": "%s: %s" % (type(exc).__name__, exc),
        }
        if args.json_out:
            try:
                Path(args.json_out).write_text(json.dumps(error, indent=2, sort_keys=True) + "\n")
            except OSError:
                pass
        print(json.dumps(error, indent=2, sort_keys=True))
        return 2
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        try:
            Path(args.json_out).write_text(text)
        except OSError as exc:
            print(json.dumps({"tool": TOOL, "ok": False,
                              "error": "cannot write --json-out: %r" % (exc,)},
                             indent=2, sort_keys=True))
            return 2
    print(text, end="")
    if args.require_qualified and not result["qualified"]:
        return 1
    if args.require_stage:
        current_rank = STAGE_RANK.get(result["stage"], -1)
        if current_rank < STAGE_RANK[args.require_stage]:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
