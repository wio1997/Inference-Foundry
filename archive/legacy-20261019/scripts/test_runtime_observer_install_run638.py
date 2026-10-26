#!/usr/bin/env python3
"""No-service transactional install/restore rehearsal on private source copies."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import loop081_runtime_observer_install_run638 as installer


with tempfile.TemporaryDirectory(prefix='run638_installer_') as directory:
    root = Path(directory)
    original_sources = installer.candidate.SOURCES
    original_paths = installer.SOURCES
    original_preflight = installer.PREFLIGHT
    try:
        manifest = json.loads(original_preflight.read_text())
        temp_sources = {}
        for key, (live, digest) in original_sources.items():
            path = root / (key + '.py')
            path.write_bytes(live.read_bytes())
            temp_sources[key] = (path, digest)
            manifest['sources'][key]['live_path'] = str(path)
        preflight = root / 'preflight.json'
        preflight.write_text(json.dumps(manifest))
        installer.candidate.SOURCES = temp_sources
        installer.SOURCES = {key: path for key, (path, _) in temp_sources.items()}
        installer.PREFLIGHT = preflight
        prepared, before, patched = installer.prepared()
        state = root / 'state'
        installed = installer.install(state)
        assert installed == prepared
        assert all(path.read_bytes() == patched[key]
                   for key, path in installer.SOURCES.items())
        installer.restore(state)
        assert all(path.read_bytes() == before[key]
                   for key, path in installer.SOURCES.items())
        # Simulate an interrupted install. Restore must accept mixed
        # original/patched files, while refusing foreign mutations.
        for key in ('cycle', 'runner'):
            installer.SOURCES[key].write_bytes(patched[key])
        installer.restore(state)
        assert all(path.read_bytes() == before[key]
                   for key, path in installer.SOURCES.items())
        installer.SOURCES['cycle'].write_bytes(b'foreign mutation')
        try:
            installer.restore(state)
        except ValueError as exc:
            assert 'ownership drift' in str(exc)
        else:
            raise AssertionError('foreign mutation accepted')
        assert installer.SOURCES['cycle'].read_bytes() == b'foreign mutation'
        atomic_target = root / 'atomic_failure.py'
        atomic_target.write_bytes(b'original')
        original_replace = installer.os.replace
        def fail_replace(*_args):
            raise OSError('injected replace failure')
        installer.os.replace = fail_replace
        try:
            try:
                installer.atomic(atomic_target, b'candidate')
            except OSError as exc:
                assert 'injected' in str(exc)
            else:
                raise AssertionError('atomic failure accepted')
        finally:
            installer.os.replace = original_replace
        assert atomic_target.read_bytes() == b'original'
        assert not list(root.glob('atomic_failure.py.run638.*.tmp'))
        print(json.dumps({'status': 'pass', 'source_copies': len(before),
                          'cases': ['install', 'restore', 'interrupted_restore',
                                    'foreign_mutation_rejected',
                                    'atomic_failure_cleanup']}))
    finally:
        installer.candidate.SOURCES = original_sources
        installer.SOURCES = original_paths
        installer.PREFLIGHT = original_preflight
