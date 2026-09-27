#!/usr/bin/env python3
"""Exercise Loop080 stage-two patch/restore on isolated source copies."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

import loop079_formal_ledger_patch as base
import loop080_frontier_patch as frontier


def invoke(action: str, state: Path, record: Path) -> None:
    sys.argv = ["loop080_frontier_patch.py", action,
                "--state-dir", str(state), "--record", str(record),
                "--offline-confirmed"]
    frontier.main()


def main() -> None:
    true_sources = dict(frontier.SOURCES)
    checks = {}
    with tempfile.TemporaryDirectory(prefix="loop080-patch-test-") as name:
        root = Path(name)
        frontier.SOURCES = {key: root / f"{key}.py" for key in true_sources}
        for key, path in true_sources.items():
            frontier.SOURCES[key].write_bytes(path.read_bytes())
        pristine_files, _ = frontier.preview()
        checks["pristine_virtual_preview"] = set(pristine_files) == set(true_sources)
        for key in ("runner", "serving"):
            path = frontier.SOURCES[key]
            stage_one = base.patch(key, path.read_text()).encode()
            base.atomic_replace(path, stage_one)
        first_stage = {key: base.sha(path.read_bytes())
                       for key, path in frontier.SOURCES.items()}
        staged_files, _ = frontier.preview()
        checks["installed_stage_one_preview"] = all(
            staged_files[key]["original"] == first_stage[key]
            for key in true_sources)
        state = root / "state"
        invoke("install", state, root / "install.json")
        checks["installed_exact"] = all(
            base.sha(path.read_bytes()) == staged_files[key]["patched"]
            for key, path in frontier.SOURCES.items())
        invoke("restore", state, root / "restore.json")
        checks["restored_exact"] = all(
            base.sha(path.read_bytes()) == first_stage[key]
            for key, path in frontier.SOURCES.items())
        path = frontier.SOURCES["runner"]
        path.write_bytes(path.read_bytes() + b"\n# unknown mutation\n")
        try:
            frontier.preview()
        except ValueError:
            checks["unknown_source_rejected"] = True
        else:
            checks["unknown_source_rejected"] = False
    frontier.SOURCES = true_sources
    if not all(checks.values()):
        raise AssertionError(checks)
    print(json.dumps(dict(status="PASS", checks=checks,
                          primary_patch_sha256=base.sha(Path(base.__file__).read_bytes()),
                          frontier_patch_sha256=base.sha(Path(frontier.__file__).read_bytes())),
                     sort_keys=True))


if __name__ == "__main__":
    main()
