#!/usr/bin/env python3
"""Produce the complete R247 freeze-input hash DAG.  It creates no review."""
from __future__ import annotations

import json
import pathlib
import sys

R = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R / "controller"))

import freeze_policy


def sha(path):
    return freeze_policy.sha256(path)


def main():
    immutable = {rel: path for rel, path in freeze_policy.immutable_files(R)}
    package = {rel: path for rel, path in freeze_policy.all_files(R)
               if rel != "freeze-input.json"}
    files = {rel: {"bytes": path.stat().st_size, "sha256": sha(path)}
             for rel, path in immutable.items()}
    package_files = {rel: {"bytes": path.stat().st_size, "sha256": sha(path)}
                     for rel, path in package.items()}
    precheck = R / "sitecustomize-precheck.json"
    out = {
        "project_id": "glm53-flash",
        "unit": "r247",
        "kind": "FREEZE_INPUT_NOT_A_REVIEW",
        "policy_schema": freeze_policy.SCHEMA,
        "file_count": len(files),
        "files": files,
        "files_sha256": {rel: row["sha256"] for rel, row in files.items()},
        "mutable_files": sorted(set(package) - set(files)),
        "package_file_count": len(package_files),
        "package_files": package_files,
        "package_files_sha256": {rel: row["sha256"] for rel, row in package_files.items()},
        "sitecustomize_precheck": {
            "path": "sitecustomize-precheck.json",
            "present": precheck.is_file(),
            "sha256": sha(precheck) if precheck.is_file() else None,
            "reused_from": "runs/20261019-r238-paired-cold-prefill-profile/sitecustomize-precheck.json",
            "consumed_image": "B",
        },
        "r246_gate": {
            "path": "r246-gate.json",
            "present": (R / "r246-gate.json").is_file(),
            "required_before_dispatch": True,
            "note": "Actual R246 PASS/idle8 signoff is an external authorization, not a source hash.",
        },
        "note": ("Main must independently review this exact immutable set and write "
                 "review.json plus a matched review-result.json authorization; this "
                 "file grants no authorization. Mutable notes/DRAFT/PENDING are kept "
                 "in package_files_sha256 but are never execution inputs."),
    }
    target = R / "freeze-input.json"
    target.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"file_count": len(files),
                      "package_file_count": len(package_files),
                      "sitecustomize_precheck_present": out["sitecustomize_precheck"]["present"],
                      "sha256": sha(target)}))


if __name__ == "__main__":
    main()
