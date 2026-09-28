#!/usr/bin/env python3
"""SHA-pinned, reversible Run637 observer installation. No service control."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path

import loop081_runtime_observer_patch_run637 as candidate

ROOT = Path('/data/wio/Inference_Foundry')
PREFLIGHT = ROOT / 'evidence/20260928_loop081_bound/run637/observer_patch_preflight.json'
SOURCES = {key: path for key, (path, _) in candidate.SOURCES.items()}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def atomic(path: Path, data: bytes) -> None:
    fd, name = tempfile.mkstemp(prefix=path.name + '.run638.',
                                suffix='.tmp', dir=path.parent)
    tmp = Path(name)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        if path.exists():
            original = path.stat()
            os.chmod(tmp, stat.S_IMODE(original.st_mode))
            if os.geteuid() == 0:
                os.chown(tmp, original.st_uid, original.st_gid)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def prepared():
    prior = json.loads(PREFLIGHT.read_text())
    helper = ROOT / 'runtime/bound_observer.py'
    helper_raw = helper.read_bytes()
    if sha(helper_raw) != prior['observer_module_sha256']:
        raise ValueError('dormant helper drift')
    compile(helper_raw, str(helper), 'exec')
    if set(prior['sources']) != set(SOURCES):
        raise ValueError('preflight source set drift')
    original, edited = {}, {}
    manifest = {'source_preflight_sha256': sha(PREFLIGHT.read_bytes()),
                'observer_module_sha256': sha(helper_raw), 'sources': {}}
    for key, (path, expected) in candidate.SOURCES.items():
        row = prior['sources'][key]
        raw = path.read_bytes()
        if sha(raw) != expected or row['live_path'] != str(path) or row['live_sha256'] != expected:
            raise ValueError('live source or preflight drift: ' + key)
        patched = candidate.PATCH[key](raw.decode()).encode()
        compile(patched, str(path), 'exec')
        if sha(patched) != row['candidate_sha256']:
            raise ValueError('candidate digest drift: ' + key)
        for api in ('.synchronize(', '.wait_stream(', '.wait_event('):
            if patched.count(api.encode()) != raw.count(api.encode()):
                raise ValueError('new hot-path wait API: ' + key)
        original[key], edited[key] = raw, patched
        manifest['sources'][key] = {'path': str(path),
                                    'original_sha256': expected,
                                    'patched_sha256': sha(patched)}
    return manifest, original, edited


def install(state_dir: Path):
    if state_dir.exists():
        raise ValueError('fresh state directory required')
    manifest, original, edited = prepared()
    state_dir.mkdir(parents=True, mode=0o700)
    for key, raw in original.items():
        atomic(state_dir / f'{key}.orig', raw)
    atomic(state_dir / 'manifest.json', (json.dumps(manifest, indent=2) + '\n').encode())
    try:
        for key, path in SOURCES.items():
            if path.read_bytes() != original[key]:
                raise ValueError('install race: ' + key)
            atomic(path, edited[key])
        for key, path in SOURCES.items():
            if sha(path.read_bytes()) != manifest['sources'][key]['patched_sha256']:
                raise ValueError('install verification: ' + key)
    except BaseException:
        for key, path in SOURCES.items():
            if sha(path.read_bytes()) == manifest['sources'][key]['patched_sha256']:
                atomic(path, original[key])
        raise
    return manifest


def restore(state_dir: Path):
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
        current = sha(path.read_bytes())
        if current not in (manifest['sources'][key]['original_sha256'],
                           manifest['sources'][key]['patched_sha256']):
            raise ValueError('restore race: ' + key)
        if current != manifest['sources'][key]['original_sha256']:
            atomic(path, backups[key])
    for key, path in SOURCES.items():
        if sha(path.read_bytes()) != manifest['sources'][key]['original_sha256']:
            raise ValueError('restore verification: ' + key)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('check', 'install', 'restore'))
    parser.add_argument('--state-dir', type=Path)
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--offline-confirmed', action='store_true')
    args = parser.parse_args()
    if args.action != 'check' and not args.offline_confirmed:
        raise ValueError('offline confirmation required')
    if args.action == 'check':
        manifest = prepared()[0]
    else:
        if args.state_dir is None:
            raise ValueError('state directory required')
        manifest = install(args.state_dir) if args.action == 'install' else restore(args.state_dir)
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps({'action': args.action, **manifest}, indent=2) + '\n')
    print(json.dumps({'action': args.action, 'files': len(SOURCES)}))


if __name__ == '__main__':
    main()
