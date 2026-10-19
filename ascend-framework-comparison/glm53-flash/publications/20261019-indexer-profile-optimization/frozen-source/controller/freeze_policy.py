"""Immutable execution inputs vs mutable review/status files for R247.

The source package keeps every mutable draft, status, review and runtime
output so the hash DAG remains complete.  Only :func:`immutable_files` is
eligible for the independent review and the one-shot execution closure; a
mutable note can never silently become an execution input.  The actual R246
PASS/idle8 signoff is an external authorization and is deliberately mutable.
"""
from __future__ import annotations

import hashlib
import pathlib

SCHEMA = "r247/freeze-policy/v1"

MUTABLE_FILES = {
    # Review communication and status documents change before/after a run.
    "DRAFT.json",
    "PENDING.json",
    "main-review-notes.txt",
    "SOURCE_REVIEW_NOTES.md",
    # Independent review/authorization and the manifest itself.
    "review.json",
    "review-result.json",
    "freeze-input.json",
    # The actual R246 PASS/idle8 signoff arrives after source review.
    "r246-gate.json",
    # One-shot/run outputs.
    "dispatch.json",
    "dispatch.once.lock",
    "once.lock",
    "controller.once.lock",
    "submission.once.lock",
    "submission.json",
    "controller.log",
    "runner.log",
    "remote-submit.json",
    "collection-receipt.json",
    "acceptance.json",
    ".DS_Store",
}

MUTABLE_PREFIXES = (
    "results/",
    "artifacts/",
    "collected/",
    "evidence/",
    "logs/",
)


def is_mutable(rel):
    rel = str(rel).replace("\\", "/")
    return rel in MUTABLE_FILES or any(rel.startswith(p) for p in MUTABLE_PREFIXES)


def all_files(root):
    root = pathlib.Path(root)
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(root))
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        yield rel, path


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def immutable_files(root):
    for rel, path in all_files(root):
        if not is_mutable(rel):
            yield rel, path


def package_files(root):
    return dict((rel, path) for rel, path in all_files(root))


def immutable_sha256(root):
    return {rel: sha256(path) for rel, path in immutable_files(root)}


def package_sha256(root):
    return {rel: sha256(path) for rel, path in package_files(root).items()}
