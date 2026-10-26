#!/usr/bin/env python3
"""Run610 reversible Run606 observer plus cohort5-only Runtime profiler window.

The existing fixed-DSpark7 algorithm and production operations are unchanged.
The profiler is diagnostic and its timing is not a formal E2E comparison.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import loop081_dispatch_patch_run606 as base

SOURCES = base.SOURCES
ORIGINAL = base.ORIGINAL


def patch(key: str, src: str) -> str:
    edited = base.patch(key, src)
    if key == 'runner':
        old = '            _extreme_runtime._p606_armed = bool(_p602_enabled)\n'
        new = old + '''            # EXTREME_LOOP081_RUN610_COHORT_PROFILE
            _p610_profile_root = os.getenv("EXTREME_RUN610_PROFILE_ROOT")
            _p610_cohort = len(self._extreme_served_cohorts)
            if _p602_enabled and _p610_profile_root and _p610_cohort == 5:
                _extreme_runtime._cycle_profile_dir = os.path.join(
                    _p610_profile_root, f"cohort{_p610_cohort}")
            else:
                _extreme_runtime._cycle_profile_dir = None
'''
        if edited.count(old) != 1:
            raise ValueError('cohort profile anchor drift')
        edited = edited.replace(old, new, 1)
    return edited


def prepared():
    paths = [p.resolve() for p in SOURCES.values()]
    if len(paths) != len(set(paths)):
        raise ValueError('duplicate source path')
    helper = base.product.base.HELPER.read_bytes()
    compile(helper, str(base.product.base.HELPER), 'exec')
    manifest = {'helper_sha256': base.sha(helper), 'sources': {}}
    old, new = {}, {}
    for key, path in SOURCES.items():
        data = path.read_bytes()
        if base.sha(data) != ORIGINAL[key]:
            raise ValueError(f'{key} original SHA drift: {base.sha(data)}')
        edited = patch(key, data.decode()).encode()
        compile(edited, str(path), 'exec')
        old[key], new[key] = data, edited
        manifest['sources'][key] = {'path': str(path), 'original_sha256': base.sha(data),
                                    'patched_sha256': base.sha(edited)}
    return manifest, old, new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('check', 'install', 'restore'))
    parser.add_argument('--state-dir', type=Path)
    parser.add_argument('--record', required=True, type=Path)
    parser.add_argument('--offline-confirmed', action='store_true')
    args = parser.parse_args()
    if args.action != 'check' and not args.offline_confirmed:
        raise ValueError('offline confirmation required')
    if args.action in ('check', 'install'):
        manifest, old, new = prepared()
        if args.action == 'install':
            if args.state_dir is None or args.state_dir.exists():
                raise ValueError('fresh state directory required')
            args.state_dir.mkdir(parents=True)
            for key, data in old.items():
                base.atomic(args.state_dir / f'{key}.orig', data)
            base.atomic(args.state_dir / 'manifest.json',
                        (json.dumps(manifest, indent=2) + '\n').encode())
            try:
                for key, path in SOURCES.items():
                    if path.read_bytes() != old[key]:
                        raise ValueError('install race ' + key)
                    base.atomic(path, new[key])
            except BaseException:
                for key, path in SOURCES.items():
                    if base.sha(path.read_bytes()) == manifest['sources'][key]['patched_sha256']:
                        base.atomic(path, old[key])
                raise
    else:
        if args.state_dir is None:
            raise ValueError('state dir required')
        manifest = json.loads((args.state_dir / 'manifest.json').read_text())
        if set(manifest['sources']) != set(SOURCES):
            raise ValueError('source set drift')
        for key, path in SOURCES.items():
            row = manifest['sources'][key]
            if row['path'] != str(path) or row['original_sha256'] != ORIGINAL[key]:
                raise ValueError('source manifest drift ' + key)
            backup = (args.state_dir / f'{key}.orig').read_bytes()
            if base.sha(backup) != ORIGINAL[key]:
                raise ValueError('backup drift ' + key)
            if base.sha(path.read_bytes()) not in (row['original_sha256'], row['patched_sha256']):
                raise ValueError('installed source drift ' + key)
        for key, path in SOURCES.items():
            if base.sha(path.read_bytes()) != ORIGINAL[key]:
                base.atomic(path, (args.state_dir / f'{key}.orig').read_bytes())
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps({'action': args.action, **manifest}, indent=2) + '\n')
    print(json.dumps({'action': args.action, 'files': len(SOURCES)}))


if __name__ == '__main__':
    main()
