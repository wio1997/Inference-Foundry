#!/usr/bin/env python3
"""SHA-pinned reversible installation of Run655 source-only observer candidate."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path

import loop081_light_bridge_patch_run655 as candidate

ROOT = Path('/data/wio/Inference_Foundry')
PREFLIGHT = ROOT / 'evidence/20260928_loop081_bound/run655/source_candidate.json'
SOURCES = {key: path for key, (path, _) in candidate.SOURCES.items()}


def sha(raw): return hashlib.sha256(raw).hexdigest()


def atomic(path, data):
    fd, name = tempfile.mkstemp(prefix=path.name + '.run656.', suffix='.tmp', dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        if path.exists():
            old = path.stat()
            os.chmod(temp, stat.S_IMODE(old.st_mode))
            if os.geteuid() == 0:
                os.chown(temp, old.st_uid, old.st_gid)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def prepared():
    prior = json.loads(PREFLIGHT.read_text())
    helper = ROOT / 'runtime/light_stage_observer.py'
    helper_raw = helper.read_bytes()
    if sha(helper_raw) != prior['helper_sha256']:
        raise ValueError('dormant helper drift')
    compile(helper_raw, str(helper), 'exec')
    for api in ('.synchronize(', '.wait_stream(', '.wait_event('):
        if api.encode() in helper_raw:
            raise ValueError('observer helper introduces explicit wait API')
    if set(prior['sources']) != set(SOURCES):
        raise ValueError('preflight source set drift')
    original, edited = {}, {}
    manifest = {'source_preflight_sha256': sha(PREFLIGHT.read_bytes()),
                'helper_sha256': sha(helper_raw), 'sources': {}}
    for key, (path, expected) in candidate.SOURCES.items():
        row = prior['sources'][key]
        raw = path.read_bytes()
        if sha(raw) != expected or row['path'] != str(path) or row['live_sha256'] != expected:
            raise ValueError('source/preflight drift: ' + key)
        new = candidate.PATCH[key](raw.decode()).encode()
        compile(new, str(path), 'exec')
        if sha(new) != row['candidate_sha256']:
            raise ValueError('candidate hash drift: ' + key)
        for api in ('.synchronize(', '.wait_stream(', '.wait_event('):
            if new.count(api.encode()) != raw.count(api.encode()):
                raise ValueError('new wait API in source: ' + key)
        original[key], edited[key] = raw, new
        manifest['sources'][key] = {'path': str(path),
                                    'original_sha256': expected,
                                    'patched_sha256': sha(new)}
    return manifest, original, edited


def install(state_dir):
    if state_dir.exists():
        raise ValueError('fresh state directory required')
    manifest, original, edited = prepared()
    state_dir.mkdir(parents=True, mode=0o700)
    for key, raw in original.items(): atomic(state_dir / f'{key}.orig', raw)
    atomic(state_dir / 'manifest.json', (json.dumps(manifest, indent=2) + '\n').encode())
    try:
        for key, path in SOURCES.items():
            if path.read_bytes() != original[key]:
                raise ValueError('install race: ' + key)
            atomic(path, edited[key])
        for key, path in SOURCES.items():
            if sha(path.read_bytes()) != manifest['sources'][key]['patched_sha256']:
                raise ValueError('install verify: ' + key)
    except BaseException:
        for key, path in SOURCES.items():
            if sha(path.read_bytes()) == manifest['sources'][key]['patched_sha256']:
                atomic(path, original[key])
        raise
    return manifest


def restore(state_dir):
    manifest = json.loads((state_dir / 'manifest.json').read_text())
    if set(manifest['sources']) != set(SOURCES):
        raise ValueError('restore source set drift')
    backups = {}
    for key, (path, expected) in candidate.SOURCES.items():
        row = manifest['sources'][key]
        backup = (state_dir / f'{key}.orig').read_bytes()
        if (row['path'] != str(path) or row['original_sha256'] != expected
                or sha(backup) != expected or
                sha(path.read_bytes()) not in (expected, row['patched_sha256'])):
            raise ValueError('restore ownership drift: ' + key)
        backups[key] = backup
    for key, path in SOURCES.items():
        if sha(path.read_bytes()) != manifest['sources'][key]['original_sha256']:
            atomic(path, backups[key])
    for key, path in SOURCES.items():
        if sha(path.read_bytes()) != manifest['sources'][key]['original_sha256']:
            raise ValueError('restore verification: ' + key)
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('check', 'install', 'restore'))
    p.add_argument('--state-dir', type=Path)
    p.add_argument('--record', type=Path, required=True)
    p.add_argument('--offline-confirmed', action='store_true')
    a = p.parse_args()
    if a.action != 'check' and not a.offline_confirmed:
        raise ValueError('offline confirmation required')
    if a.action == 'check': manifest = prepared()[0]
    else:
        if a.state_dir is None: raise ValueError('state dir required')
        manifest = install(a.state_dir) if a.action == 'install' else restore(a.state_dir)
    a.record.parent.mkdir(parents=True, exist_ok=True)
    a.record.write_text(json.dumps({'action': a.action, **manifest}, indent=2) + '\n')
    print(json.dumps({'action': a.action, 'files': len(SOURCES)}))


if __name__ == '__main__': main()
