"""Preserve a native ``sitecustomize`` before installing the R234 observer.

Our observer tree is placed first on ``PYTHONPATH``, so Python imports our
``sitecustomize`` instead of an image-provided one.  This module locates the
next ``sitecustomize.py`` on the remaining import path and executes it in the
same interpreter before any observer patch.  If chaining fails, the caller must
fail closed: do not install the observer and record the exact error.  A blind
shadow or a silent fallback is not permitted.

The helper is import-light (stdlib only), reads no model tensor and changes no
model source.  It is used both at runtime and by the read-only CPU precheck.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import pathlib
import runpy
import sys

SCHEMA = "r234/sitecustomize-chain/v1"
CHAIN_ENV = "R234_CHAINED_SITECUSTOMIZE"
IN_PROGRESS_ENV = "R234_SITECUSTOMIZE_CHAIN_IN_PROGRESS"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def observer_roots():
    """Directories belonging to the observer injection, never chained."""
    roots = set()
    roots.add(pathlib.Path(__file__).resolve().parents[1])
    value = os.environ.get("R234_OBSERVER_ROOT")
    if value:
        roots.add(pathlib.Path(value).resolve())
    return roots


def _is_under(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def search_paths(root=None):
    """Return the image import paths in the same order Python would use."""
    paths = []
    for entry in sys.path:
        if entry:
            paths.append(pathlib.Path(entry))
    if root is not None:
        paths.insert(0, pathlib.Path(root))
    if not paths:
        paths.append(pathlib.Path("."))
    return paths


def find_native_sitecustomize(paths=None, roots=None):
    """Return the first non-observer ``sitecustomize.py`` on the path."""
    roots = {pathlib.Path(p).resolve() for p in (roots or observer_roots())}
    seen = set()
    for entry in (paths if paths is not None else search_paths()):
        try:
            candidate = (pathlib.Path(entry) / "sitecustomize.py").resolve()
        except OSError:
            continue
        if candidate in seen or not candidate.is_file():
            continue
        seen.add(candidate)
        if any(_is_under(candidate, root) for root in roots):
            continue
        return candidate
    return None


def chain(path, record=None):
    """Execute one native sitecustomize source in this interpreter."""
    path = pathlib.Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(str(path))
    if os.environ.get(IN_PROGRESS_ENV) == "1":
        raise RuntimeError("recursive sitecustomize chaining refused")
    result = {
        "schema": SCHEMA,
        "native_present": True,
        "native_path": str(path),
        "native_sha256": _sha256(path),
        "native_bytes": path.stat().st_size,
        "chained": False,
        "error": None,
    }
    os.environ[IN_PROGRESS_ENV] = "1"
    try:
        runpy.run_path(str(path), run_name="_r234_native_sitecustomize")
        result["chained"] = True
    except BaseException as exc:
        result["error"] = repr(exc)
        raise
    finally:
        os.environ.pop(IN_PROGRESS_ENV, None)
        os.environ[CHAIN_ENV] = str(path)
        if record is not None:
            record(result)
    return result


def preserve(record=None, paths=None, roots=None):
    """Find and chain a native sitecustomize.

    Returns ``native_present=false`` when the image has none (safe injection),
    or ``native_present=true, chained=true`` after a successful chain.  A
    chain failure is raised to the caller after the record is written.
    """
    native = find_native_sitecustomize(paths=paths, roots=roots)
    if native is None:
        result = {
            "schema": SCHEMA,
            "native_present": False,
            "native_path": None,
            "native_sha256": None,
            "native_bytes": None,
            "chained": False,
            "error": None,
        }
        if record is not None:
            record(result)
        return result
    return chain(native, record=record)
