#!/usr/bin/env python3
"""Generate SHA-pinned source-only Run661 readiness query candidate."""
from __future__ import annotations
import ast
import difflib
import hashlib
import json

from loop081_ready_packet_patch_run659 import ROOT, SOURCES, cycle as typed_cycle, target
from loop081_runtime_observer_patch_run637 import PATCH, one

OUT = ROOT / "evidence/20260929_loop081_bound/run661"
HELPERS = (ROOT / "runtime/bound_observer.py", ROOT / "runtime/ready_edge_observer.py",
           ROOT / "runtime/ready_query_observer_run661.py")


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def cycle(src: str) -> str:
    src = typed_cycle(src)
    src = one(src, "from .ready_edge_observer import ReadyEdgeRecorder",
              "from .ready_query_observer_run661 import ReadyQueryRecorder")
    src = one(src, "self._bound_probe = ReadyEdgeRecorder(",
              "self._bound_probe = ReadyQueryRecorder(")
    src = one(src, '            mark("begin")\n            use_scheduled',
              '            mark("begin")\n            if self._bound_probe is not None:\n'
              '                self._bound_probe.poll_prior(_bound_cycle, "cycle_begin")\n'
              '            use_scheduled')
    src = one(src, '            mark("prepare_target")\n            if self._bound_probe is not None:',
              '            mark("prepare_target")\n            if self._bound_probe is not None:\n'
              '                self._bound_probe.poll_prior(_bound_cycle, "prepare_target")\n'
              '            if self._bound_probe is not None:')
    src = one(src, '                        self._bound_probe.mark(_bound_cycle, "target_before")\n'
              '                    if self._cycle_profiler',
              '                        self._bound_probe.mark(_bound_cycle, "target_before")\n'
              '                        self._bound_probe.poll_prior(_bound_cycle, "target_before")\n'
              '                    if self._cycle_profiler')
    return src


PATCH = dict(PATCH, cycle=cycle, target=target)


def main() -> None:
    candidate = OUT / "candidate"
    diff = OUT / "diff"
    candidate.mkdir(parents=True, exist_ok=False)
    diff.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "source_only_not_installed", "helpers": {}, "sources": {}}
    for path in HELPERS:
        raw = path.read_bytes()
        ast.parse(raw.decode())
        manifest["helpers"][str(path)] = sha(raw)
    for key, (path, before_sha) in SOURCES.items():
        before_raw = path.read_bytes()
        if sha(before_raw) != before_sha:
            raise RuntimeError(f"source drift: {key}")
        before = before_raw.decode()
        after = PATCH[key](before)
        ast.parse(after)
        for api in (".synchronize(", ".wait_stream(", ".wait_event("):
            if after.count(api) != before.count(api):
                raise RuntimeError(f"new wait API: {key}/{api}")
        (candidate / f"{key}.py").write_text(after)
        diff_text = "".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                                  fromfile=str(path), tofile=str(path)+".run661"))
        (diff / f"{key}.diff").write_text(diff_text)
        manifest["sources"][key] = {"path": str(path), "before_sha256": before_sha,
                                    "candidate_sha256": sha(after.encode()),
                                    "diff_sha256": sha(diff_text.encode())}
    (OUT / "candidate_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({"status": manifest["status"], "sources": len(manifest["sources"])}))


if __name__ == "__main__":
    main()
