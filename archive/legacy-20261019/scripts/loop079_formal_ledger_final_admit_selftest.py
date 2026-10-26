#!/usr/bin/env python3
"""CPU-only negative tests for the post-cleanup diagnostic admission."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile


SCRIPT = Path(__file__).with_name("loop079_formal_ledger_final_admit.py")


def fixture(root):
    values = {
        "cleanup_status.txt": "".join(f"{k}=0\n" for k in (
            "run_exit", "stop_exit", "stop_verify_exit", "restore_exit",
            "sha_exit", "sha_compare_exit", "script_sha_exit",
            "script_compare_exit", "pre_admission_exit")),
        "source_before.sha256": "original\n",
        "source_after.sha256": "original\n",
        "scripts_before.sha256": "scripts-original\n",
        "scripts_after.sha256": "scripts-original\n",
        "client_admission.json": json.dumps({"status": "client_two_phase_admitted", "request_count": 96}),
        "server_admission.json": json.dumps({"status": "server_two_phase_admitted"}),
        "warmup_phase_barrier.json": json.dumps({"status": "warmup48_phase_barrier_pass"}),
        "server_post_count.txt": "96\n",
        "install.json": json.dumps({"installed": True, "files": {"a": "x"}}),
        "restore.json": json.dumps({"restored": True, "files": {"a": "x"}, "helper_sha_drift": False}),
        "phase_transitions.jsonl": "{}\n{}\n",
    }
    for name, value in values.items():
        (root / name).write_text(value)


def invoke(root):
    return subprocess.run([sys.executable, str(SCRIPT), "--run-dir", str(root),
                           "--output", str(root / "final.json")], capture_output=True,
                          text=True)


def main():
    cases = [
        ("stop_failed", "cleanup_status.txt", lambda s: s.replace("stop_exit=0", "stop_exit=1")),
        ("idle_not_proved", "cleanup_status.txt", lambda s: s.replace("stop_verify_exit=0", "stop_verify_exit=1")),
        ("restore_failed", "cleanup_status.txt", lambda s: s.replace("restore_exit=0", "restore_exit=1")),
        ("source_changed", "source_after.sha256", lambda s: "different\n"),
        ("script_changed", "scripts_after.sha256", lambda s: "different\n"),
        ("server_reduction_missing", "server_admission.json", lambda s: json.dumps({"status": "missing"})),
        ("post_count_wrong", "server_post_count.txt", lambda s: "95\n"),
        ("helper_drift", "restore.json", lambda s: s.replace("false", "true")),
    ]
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        fixture(root)
        good = invoke(root)
        assert good.returncode == 0, good.stderr
        assert json.loads((root / "final.json").read_text())["status"] == (
            "instrumented_bound_diagnostic_admitted_not_formal_tps")
        with (root / "cleanup_status.txt").open("a") as f:
            f.write("admission_exit=0\nfinal_exit=0\n")
        assert invoke(root).returncode == 0, "post-cleanup independent replay failed"
        for name, file, mutate in cases:
            fixture(root)
            (root / "final.json").unlink(missing_ok=True)
            path = root / file
            path.write_text(mutate(path.read_text()))
            bad = invoke(root)
            assert bad.returncode != 0 and not (root / "final.json").exists(), name
    print(json.dumps({"status": "pass", "positive": 2, "negative": len(cases)}))


if __name__ == "__main__":
    main()
