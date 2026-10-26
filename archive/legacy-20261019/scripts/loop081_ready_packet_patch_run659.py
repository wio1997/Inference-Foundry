#!/usr/bin/env python3
"""Build a source-only typed ready-edge candidate on Run637's proven markers."""
from __future__ import annotations

import ast
import difflib
import hashlib
import json
from pathlib import Path

from loop081_runtime_observer_patch_run637 import PATCH, ROOT, SOURCES, one

OUT = ROOT / "evidence/20260929_loop081_bound/run659"
HELPER = ROOT / "runtime/ready_edge_observer.py"


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def cycle(src: str) -> str:
    src = PATCH["cycle"](src)
    src = one(src, "from .bound_observer import BoundEventRecorder",
              "from .ready_edge_observer import ReadyEdgeRecorder")
    src = one(src, "self._bound_probe = BoundEventRecorder(",
              "self._bound_probe = ReadyEdgeRecorder(")
    src = one(src, '        if self._schedule_mode not in ("off", "serial", "overlap"):\n            raise ValueError("invalid next-target metadata scheduling mode")', '        if self._schedule_mode not in ("off", "serial", "overlap"):\n            raise ValueError("invalid next-target metadata scheduling mode")\n        if self._bound_probe is not None and self._schedule_mode != "off":\n            raise RuntimeError("typed ready-edge packet requires scheduling mode off")')
    src = one(src, '''            mark("prepare_target")
            if diag is not None:''', '''            mark("prepare_target")
            if self._bound_probe is not None:
                for _role, _tensor in (
                    ("after_prepare.target_input_ids", self.state.target_input_ids),
                    ("after_prepare.target_positions", self.state.target_positions),
                    ("after_prepare.target_slot_mapping", self.state.target_slot_mapping),
                ):
                    self._bound_probe.observe(_bound_cycle, _role, _tensor,
                                              after_label="prepare_target")
            if diag is not None:''')
    src = one(src, '''                mark("derived_target_metadata")
                if use_scheduled:''', '''                mark("derived_target_metadata")
                if self._bound_probe is not None:
                    for _index, _group in enumerate(self.target_metadata.groups):
                        self._bound_probe.observe(
                            _bound_cycle, f"metadata.group{_index}.sas",
                            _group.sas_metadata, after_label="derived_target_metadata")
                        if _group.qli_metadata is not None:
                            self._bound_probe.observe(
                                _bound_cycle, f"metadata.group{_index}.qli",
                                _group.qli_metadata, after_label="derived_target_metadata")
                if use_scheduled:''')
    src = one(src, '''            mark("state_advance")
            if self._schedule_mode == "overlap":''', '''            mark("state_advance")
            if self._bound_probe is not None:
                self._bound_probe.observe(
                    _bound_cycle, "after_state.last_sampled_tokens",
                    self.state.last_sampled_tokens, after_label="state_advance_after")
                self._bound_probe.observe(
                    _bound_cycle, "after_state.num_computed_tokens",
                    self.state.num_computed_tokens, after_label="state_advance_after")
            if self._schedule_mode == "overlap":''')
    src = one(src, '''            mark("proposer")
            if self._schedule_mode == "serial":''', '''            mark("proposer")
            if self._bound_probe is not None:
                self._bound_probe.observe(
                    _bound_cycle, "after_proposer.next_draft",
                    next_draft, after_label="proposer_after")
            if self._schedule_mode == "serial":''')
    src = one(src, '''            mark("draft_commit")
            if diag is not None:''', '''            mark("draft_commit")
            if self._bound_probe is not None:
                self._bound_probe.observe(
                    _bound_cycle, "after_commit.draft_tokens",
                    self.state.draft_tokens, after_label="draft_commit_after")
            if diag is not None:''')
    return src


def target(src: str) -> str:
    src = PATCH["target"](src)
    return one(src, '''        model_output = self.binding.forward(
            state.target_input_ids,
            state.target_positions,
        )
''', '''        if getattr(self, "_bound_probe", None) is not None:
            self._bound_probe.observe(
                state.cycle_index, "target_forward.target_input_ids",
                state.target_input_ids, after_label="target_before")
        model_output = self.binding.forward(
            state.target_input_ids,
            state.target_positions,
        )
''')


def main() -> None:
    ast.parse(HELPER.read_text())
    manifest = {
        "status": "source_only_NOT_INSTALLED_NOT_LIVE_READY",
        "helper_sha256": digest(HELPER.read_bytes()),
        "source": {},
    }
    candidate_dir = OUT / "candidate"
    diff_dir = OUT / "diff"
    candidate_dir.mkdir(parents=True, exist_ok=False)
    diff_dir.mkdir(parents=True, exist_ok=False)
    funcs = dict(PATCH, cycle=cycle, target=target)
    for key, (path, expected) in SOURCES.items():
        raw = path.read_bytes()
        if digest(raw) != expected:
            raise ValueError(f"source SHA drift: {key}")
        before = raw.decode()
        after = funcs[key](before)
        ast.parse(after)
        for api in (".synchronize(", ".wait_stream(", ".wait_event("):
            if after.count(api) != before.count(api):
                raise ValueError(f"new wait API in {key}: {api}")
        candidate = candidate_dir / f"{key}.py"
        candidate.write_text(after)
        diff = "".join(difflib.unified_diff(
            before.splitlines(True), after.splitlines(True),
            fromfile=str(path), tofile=str(path) + ".run659"))
        diff_path = diff_dir / f"{key}.diff"
        diff_path.write_text(diff)
        manifest["source"][key] = {
            "path": str(path), "before_sha256": expected,
            "candidate_sha256": digest(candidate.read_bytes()),
            "diff_sha256": digest(diff.encode()),
        }
    (OUT / "candidate_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": manifest["status"], "files": list(funcs)}))


if __name__ == "__main__":
    main()
