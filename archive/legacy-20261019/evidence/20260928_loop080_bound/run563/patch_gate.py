"""CPU-only atomic install/restore and negative gate for Run563 source patch."""
from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory


ROOT = Path("/data/wio/Inference_Foundry")
sys.path.insert(0, str(ROOT / "scripts"))
import loop080_pause_control_patch_run563 as patch  # noqa: E402


def invoke(*args):
    previous = sys.argv
    sys.argv = [str(patch.__file__), *map(str, args)]
    try:
        with redirect_stdout(io.StringIO()):
            patch.main()
    finally:
        sys.argv = previous


with TemporaryDirectory(prefix="run563-patch-gate-") as temp:
    temp = Path(temp)
    original_runner = patch.RUNNER.read_bytes()
    original_segmented = patch.SEGMENTED.read_bytes()
    runner = temp / "runner.py"
    segmented = temp / "segmented.py"
    runner.write_bytes(original_runner)
    segmented.write_bytes(original_segmented)
    original_sources = patch.SOURCES
    patch.SOURCES = {"runner": runner, "segmented": segmented}
    state = temp / "state"
    try:
        invoke("check", "--record", temp / "check.json")
        check = json.loads((temp / "check.json").read_text())
        assert check["files"]["runner"]["original"] == patch.ORIGINAL["runner"]
        invoke("install", "--state-dir", state, "--record", temp / "install.json",
               "--offline-confirmed")
        assert patch.base.sha(runner.read_bytes()) == check["files"]["runner"]["patched"]
        assert patch.base.sha(segmented.read_bytes()) == patch.CANDIDATE_SHA
        installed_runner = runner.read_bytes()

        manifest_path = state / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"]["runner"]["original"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest))
        try:
            invoke("restore", "--state-dir", state, "--record", temp / "bad.json",
                   "--offline-confirmed")
            raise AssertionError("manifest substitution accepted")
        except ValueError as exc:
            assert "backup changed" in str(exc)
        assert runner.read_bytes() == installed_runner
        manifest["files"]["runner"]["original"] = patch.ORIGINAL["runner"]
        manifest_path.write_text(json.dumps(manifest))

        runner.write_bytes(installed_runner + b"\n# unknown source change\n")
        try:
            invoke("restore", "--state-dir", state, "--record", temp / "drift.json",
                   "--offline-confirmed")
            raise AssertionError("unknown installed source accepted")
        except ValueError as exc:
            assert "unknown source bytes" in str(exc)
        runner.write_bytes(installed_runner)

        invoke("restore", "--state-dir", state, "--record", temp / "restore.json",
               "--offline-confirmed")
        assert runner.read_bytes() == original_runner
        assert segmented.read_bytes() == original_segmented
        assert json.loads((temp / "restore.json").read_text())["restored"] is True
        invoke("restore", "--state-dir", state, "--record", temp / "restore2.json",
               "--offline-confirmed")
        assert runner.read_bytes() == original_runner

        runner.write_bytes(original_runner + b"\n# drift\n")
        try:
            patch.preview()
            raise AssertionError("original source drift accepted")
        except ValueError as exc:
            assert "source SHA drift" in str(exc)
        runner.write_bytes(original_runner)

        helper_pin = patch.HELPER_SHA
        patch.HELPER_SHA = "0" * 64
        try:
            patch.preview()
            raise AssertionError("helper drift accepted")
        except ValueError as exc:
            assert "helper SHA drift" in str(exc)
        finally:
            patch.HELPER_SHA = helper_pin
    finally:
        patch.SOURCES = original_sources

print("Run563 CPU patch gate PASS: preview/install/restore/idempotence/manifest/source/helper negatives")
