#!/usr/bin/env python3
"""Preserve small probe artifacts and stop only the fresh receipt's own container."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import socket
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--reason", choices=("complete", "negative", "failed", "deadline"), required=True)
    a = p.parse_args()
    root = a.run_dir.resolve()
    assert str(root).startswith("/data/wio/glm5-3-vllm-execution/20261018-state-tail/")
    assert socket.gethostname() in ("B4-33U-Atlas900RCKA3-32", "B5-5U-Atlas900RCKA3-33")
    plan = json.loads((root / "plan.json").read_text())
    launch = json.loads((root / "launch.json").read_text())
    context = json.loads((root / "context.json").read_text())
    assert plan["target_hostname"] == socket.gethostname() and Path(plan["remote"]["run_dir"]).resolve() == root
    lease = context["client"]["lease"]
    assert lease["uid"] == plan["reservation"]["expected_uid"] and lease["name"] == plan["reservation"]["name"]
    assert plan["env"]["GLM53_STATE_TAIL_PROBE"] == "1"
    cid = launch["container_id"]
    assert re.fullmatch(r"[0-9a-f]{64}", cid)
    before = json.loads(subprocess.check_output(["docker", "inspect", cid], timeout=15))[0]
    assert before["Id"] == cid and before["Name"] == "/" + plan["container"]["name"]
    assert before["Image"] == plan["image"]["id"] == "sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83"
    assert any(m["Destination"] == "/out" and Path(m["Source"]).resolve() == root for m in before["Mounts"])
    artifacts = Path(__file__).resolve().parent / "artifacts"
    artifacts.mkdir(exist_ok=False)
    selected = [root / name for name in ("plan.json", "launch.json", "context.json", "manifest.json", "payload.py", "start_capacity.py")]
    selected += sorted((root / "state-tail-probe").glob("*.jsonl"))
    selected = [path for path in selected if path.is_file()]
    assert selected and all(not path.is_symlink() for path in selected)
    assert sum(path.stat().st_size for path in selected) < 64 * 1024**2
    # Copy closed per-record files before stopping; the final console is saved
    # after stop. No model memory, weights or full runtime profile is dumped.
    for path in selected:
        dest = artifacts / path.relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    stop = subprocess.run(["docker", "stop", "-t", "30", cid], capture_output=True, text=True, timeout=40)
    after = json.loads(subprocess.check_output(["docker", "inspect", cid], timeout=15))[0]
    # Recopy after termination to retain any last closed record or partial
    # failure record, instead of accepting the pre-stop snapshot as final.
    (artifacts / "state-tail-probe").mkdir(exist_ok=True)
    for path in sorted((root / "state-tail-probe").glob("*.jsonl")):
        assert path.is_file() and not path.is_symlink() and path.stat().st_size < 64 * 1024**2
        shutil.copy2(path, artifacts / "state-tail-probe" / path.name)
    with gzip.open(artifacts / "console.txt.gz", "wb") as output:
        logger = subprocess.Popen(["docker", "logs", "--timestamps", cid], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        total = 0
        try:
            for chunk in iter(lambda: logger.stdout.read(1048576), b""):
                total += len(chunk)
                assert total <= 64 * 1024**2, "console bound exceeded"
                output.write(chunk)
            assert logger.wait(timeout=15) == 0
        finally:
            if logger.poll() is None:
                logger.terminate()
    patch = plan["runtime_patches"][0]
    data = Path(patch["source"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == patch["sha256"]
    (artifacts / "worker-overlay.py").write_bytes(data)
    report = {"container_id": cid, "reason": a.reason, "stop_returncode": stop.returncode,
              "before_state": before["State"], "after_state": after["State"],
              "release_holder_after_backup_and_idle_check": {"namespace": plan["reservation"]["namespace"],
                  "name": lease["name"], "uid_precondition": lease["uid"]},
              "holder_deleted_by_script": False, "scope": "Own exact CID only; no other container or reservation changed"}
    (artifacts / "finish.json").write_text(json.dumps(report, indent=2) + "\n")
    print("GLM53_RESULT_JSON_BEGIN\n" + json.dumps(report, indent=2) + "\nGLM53_RESULT_JSON_END")
    assert stop.returncode == 0 and not after["State"]["Running"]


if __name__ == "__main__":
    main()
