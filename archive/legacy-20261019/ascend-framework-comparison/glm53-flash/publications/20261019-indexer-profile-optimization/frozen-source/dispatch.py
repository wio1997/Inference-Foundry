#!/usr/bin/env python3
"""The single R247 one-shot entrance: ``python3 dispatch.py``.

It refuses unless the independent R247 review, its matched authorization and
the actual R246 PASS/idle8 signoff all exist, every immutable source byte
matches ``freeze-input.json``, and no once lock exists.  It starts
``launcher.py`` detached exactly once; the gateway then re-checks the R246
receipt and live idle8 before creating any r247 controller root.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess
import sys

R = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(R / "controller"))
sys.path.insert(0, str(R))

import launcher  # noqa: E402

EXACT_COMMAND = "python3 dispatch.py"


def check_ready(root=R):
    if (root / "dispatch.once.lock").exists() or (root / "once.lock").exists():
        raise SystemExit("DISPATCH_ALREADY_STARTED")
    review, auth, freeze, immutable, gate = launcher.load_review_and_authorization(root)
    authorization = auth.get("authorization") or {}
    if authorization.get("exact_command") != EXACT_COMMAND:
        raise SystemExit("AUTHORIZATION_EXACT_COMMAND_MISMATCH")
    return {"review": review, "authorization": auth, "freeze": freeze,
            "immutable": immutable, "gate": gate,
            "review_sha256": hashlib.sha256((root / "review.json").read_bytes()).hexdigest(),
            "freeze_input_sha256": hashlib.sha256(
                (root / "freeze-input.json").read_bytes()).hexdigest(),
            "r246_gate_sha256": hashlib.sha256(
                (root / "r246-gate.json").read_bytes()).hexdigest()}


def main(root=R, popen=subprocess.Popen):
    ready = check_ready(root)
    os.close(os.open(root / "dispatch.once.lock",
                     os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
    with (root / "runner.log").open("wb") as log:
        process = popen([sys.executable, str(root / "launcher.py")],
                        cwd=root, stdin=subprocess.DEVNULL, stdout=log,
                        stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    record = {"project_id": "glm53-flash", "unit": "r247",
              "pid": process.pid, "launcher": "launcher.py",
              "detached_session": True,
              "review_sha256": ready["review_sha256"],
              "freeze_input_sha256": ready["freeze_input_sha256"],
              "r246_gate_sha256": ready["r246_gate_sha256"],
              "immutable_files": len(ready["immutable"]),
              "exact_command": EXACT_COMMAND}
    (root / "dispatch.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
