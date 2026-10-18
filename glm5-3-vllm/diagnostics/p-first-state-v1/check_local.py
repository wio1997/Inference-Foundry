#!/usr/bin/env python3
"""Exercise only source/review-plan integrity rejection on Mac fixtures."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile

from image_identity import EXPECTED, inspect
from validate_packet import validate


def main():
    root = Path(__file__).resolve().parent
    repo = root.parents[2]
    packet = json.loads((root / "review-plan.json").read_text())
    checks = {}
    for source in root.glob("*.py"):
        compile(source.read_text(), str(source), "exec")
    with tempfile.TemporaryDirectory(prefix="glm53-identity-fixture-") as tmp:
        mirror = Path(tmp)
        for row in packet["source_identity"]:
            target = mirror / row["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(row["local_path"])
        checks["18_source_fixture_matches"] = inspect(mirror, EXPECTED)["source_identity_matches"]
        target = mirror / EXPECTED[0]["path"]
        target.unlink()  # Never write through a link to an archived source.
        target.write_bytes(b"deliberately changed fixture")
        checks["changed_source_rejected"] = not inspect(mirror, EXPECTED)["source_identity_matches"]
        target.unlink()
        checks["missing_source_rejected"] = not inspect(mirror, EXPECTED)["source_identity_matches"]
        checks["empty_source_inventory_rejected"] = not inspect(mirror, [])["source_identity_matches"]
        try:
            inspect(mirror, [{"path": "../unsafe", "sha256": "0" * 64}])
        except ValueError:
            checks["unsafe_source_path_rejected"] = True
    positive = validate(packet, repo)
    for label, mutate in [
        ("empty_cases", lambda p: p.update(cases=[])),
        ("incomplete_cases", lambda p: p.update(cases=p["cases"][:1])),
        ("empty_prompt", lambda p: p["cases"][0].update(prompt_token_ids=[])),
        ("wrong_prompt_sha", lambda p: p["cases"][0].update(prompt_sha256="0" * 64)),
        ("misaligned_position", lambda p: p["cases"][0].update(sample_positions=[0, 127, 128, 256])),
        ("missing_position", lambda p: p["cases"][1].update(sample_positions=[0, 127, 128, 255])),
        ("wrong_config_sha", lambda p: p["base_configs"]["D"].update(sha256="0" * 64)),
        ("premature_ready", lambda p: p.update(execution_ready=True)),
        ("premature_device_result", lambda p: p.update(device_result={})),
        ("wrong_sampling", lambda p: p["sampling"].update(min_tokens_D=0)),
    ]:
        bad = copy.deepcopy(packet)
        mutate(bad)
        try:
            validate(bad, repo)
        except (ValueError, KeyError):
            checks[label + "_rejected"] = True
        else:
            raise AssertionError(label)
    assert all(checks.values()) and positive["execution_ready"] is False
    print(json.dumps({"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Mac syntax, source identity and review input/config rejection only",
        "checks": checks, "packet_result": positive, "device_run": False,
        "this_script_contains_device_capture": False,
        "state_capture_rejection_tests": "This file checks source/review-plan integrity only; adapter record/layout checks are in runtime-patches/state-tail-probe-v1/check_local.py. No device execution."}, indent=2))


if __name__ == "__main__":
    main()
