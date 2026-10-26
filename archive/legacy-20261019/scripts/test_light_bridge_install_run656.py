#!/usr/bin/env python3
"""Transactional install/restore rehearsal on private source copies."""
from __future__ import annotations
import json
import tempfile
from pathlib import Path
import loop081_light_bridge_install_run656 as installer

with tempfile.TemporaryDirectory(prefix='run656_installer_') as directory:
    root=Path(directory)
    original_sources=installer.candidate.SOURCES
    original_paths=installer.SOURCES
    original_preflight=installer.PREFLIGHT
    try:
        manifest=json.loads(original_preflight.read_text())
        temp_sources={}
        for key,(live,digest) in original_sources.items():
            path=root/(key+'.py')
            path.write_bytes(live.read_bytes())
            temp_sources[key]=(path,digest)
            manifest['sources'][key]['path']=str(path)
        preflight=root/'preflight.json'
        preflight.write_text(json.dumps(manifest))
        installer.candidate.SOURCES=temp_sources
        installer.SOURCES={key:path for key,(path,_) in temp_sources.items()}
        installer.PREFLIGHT=preflight
        prepared,before,patched=installer.prepared()
        state=root/'state'
        assert installer.install(state)==prepared
        assert all(path.read_bytes()==patched[key] for key,path in installer.SOURCES.items())
        installer.restore(state)
        assert all(path.read_bytes()==before[key] for key,path in installer.SOURCES.items())
        for key in ('cycle','runner'):
            installer.SOURCES[key].write_bytes(patched[key])
        installer.restore(state)
        assert all(path.read_bytes()==before[key] for key,path in installer.SOURCES.items())
        installer.SOURCES['cycle'].write_bytes(b'foreign mutation')
        try:installer.restore(state)
        except ValueError as exc:assert 'ownership drift' in str(exc)
        else:raise AssertionError('foreign mutation accepted')
        target=root/'atomic_failure.py'
        target.write_bytes(b'original')
        original_replace=installer.os.replace
        installer.os.replace=lambda *_args:(_ for _ in ()).throw(OSError('injected'))
        try:
            try:installer.atomic(target,b'candidate')
            except OSError:pass
            else:raise AssertionError('atomic failure accepted')
        finally:installer.os.replace=original_replace
        assert target.read_bytes()==b'original'
        assert not list(root.glob('atomic_failure.py.run656.*.tmp'))
        print(json.dumps({'status':'pass','source_copies':len(before),
                          'cases':['install','restore','interrupted_restore',
                                   'foreign_mutation_rejected','atomic_failure_cleanup']}))
    finally:
        installer.candidate.SOURCES=original_sources
        installer.SOURCES=original_paths
        installer.PREFLIGHT=original_preflight
