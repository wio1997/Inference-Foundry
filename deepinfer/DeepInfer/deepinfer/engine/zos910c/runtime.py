"""Verify the canonical installed extension without a module overlay."""
from pathlib import Path
import hashlib
import importlib
import json


def source_root():
    return Path(__file__).resolve().parents[3]


def binding_identity():
    root=source_root()
    manifest=json.loads((root/'ascend910c/runtime.json').read_text())
    entry=manifest['extension'];artifact=(root/entry['path']).resolve()
    if not artifact.is_relative_to(root):
        raise RuntimeError('unified extension must belong to this source tree')
    if hashlib.sha256(artifact.read_bytes()).hexdigest()!=entry['sha256']:
        raise RuntimeError('installed unified extension differs from runtime manifest')
    return dict(path=str(artifact),sha256=entry['sha256'])


def load_runtime():
    identity=binding_identity()
    from ..official_yoco_metadata910c import load_bindings
    load_bindings()
    extension=importlib.import_module('deepinfer._C')
    if Path(extension.__file__).resolve()!=Path(identity['path']):
        raise RuntimeError('another unified extension shadows this source tree')
    return identity
