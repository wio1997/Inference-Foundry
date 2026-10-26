#!/usr/bin/env python3
"""Compile/import and source-only state checks for R247."""
from __future__ import annotations

import importlib
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
R = HERE.parent
sys.path.insert(0, str(R / "controller"))
sys.path.insert(0, str(R))


def test_compile():
    count = 0
    for path in sorted(R.rglob("*.py")):
        if "results" in path.parts or "artifacts" in path.parts:
            continue
        compile(path.read_text(), str(path), "exec")
        count += 1
    assert count >= 15, count


def test_imports():
    sys.path.insert(0, str(R))
    import controller  # noqa: F401
    import dispatch  # noqa: F401
    import launcher  # noqa: F401
    import build_stage  # noqa: F401
    assert controller.UNIT == "r247"
    assert launcher.UNIT == "r247"
    assert dispatch.EXACT_COMMAND == "python3 dispatch.py"


def test_source_only_state():
    assert not (R / "review.json").exists()
    assert not (R / "review-result.json").exists()
    assert not (R / "r246-gate.json").exists()
    assert not (R / "once.lock").exists()
    assert not (R / "dispatch.once.lock").exists()
    assert not (R / "controller.once.lock").exists()
    combined = json.loads((R / "stage/B/combined-manifest.json").read_text())
    assert combined["status"] == "R247_CANDIDATE_DELTA_NOT_ORIGINAL_46"
    assert len(combined["files"]) == 47
    stage = json.loads((R / "stage/manifest.json").read_text())
    assert len(stage["B"]) == 55
    precheck = json.loads((R / "sitecustomize-precheck.json").read_text())
    assert precheck["images"]["B"]["image"] == (
        "sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263")
    assert precheck["images"]["B"]["passed"] is True


def main():
    test_compile()
    test_imports()
    test_source_only_state()
    print(json.dumps({"passed": True, "unit": "r247",
                      "source_only_state": True }))


if __name__ == "__main__":
    sys.exit(main())
