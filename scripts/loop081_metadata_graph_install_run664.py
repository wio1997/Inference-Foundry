#!/usr/bin/env python3
"""Reversible exact-SHA installation of the Run664 diagnostic candidate."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from loop081_runtime_observer_install_run638 import atomic, sha

ROOT = Path('/data/wio/Inference_Foundry')
PIN = ROOT / 'evidence/20260929_loop081_bound/run664/candidate_manifest.json'


def prepared():
    pin = json.loads(PIN.read_text())
    for name, expected in pin['helpers'].items():
        raw = Path(name).read_bytes()
        if sha(raw) != expected:
            raise RuntimeError('helper drift: ' + name)
        compile(raw, name, 'exec')
    original, edited = {}, {}
    for key, row in pin['sources'].items():
        path = Path(row['path'])
        raw = path.read_bytes()
        candidate = ROOT / 'evidence/20260929_loop081_bound/run664/candidate' / f'{key}.py'
        new = candidate.read_bytes()
        if sha(raw) != row['before_sha256'] or sha(new) != row['candidate_sha256']:
            raise RuntimeError('source/candidate drift: ' + key)
        compile(new, str(path), 'exec')
        for api in (b'.wait_stream(', b'.wait_event('):
            if new.count(api) != raw.count(api):
                raise RuntimeError('new blocking API: ' + key)
        if key == 'metadata':
            before, after = raw.count(b'.synchronize('), new.count(b'.synchronize(')
            capture = new.split(b'    def capture_graph(', 1)[1].split(
                b'    def update(', 1)[0]
            verify = new.split(b'    def update(', 1)[1].split(
                b'    def _update_eager(', 1)[0]
            if (after != before + 4 or capture.count(b'torch.npu.synchronize()') != 2
                    or verify.count(b'torch.npu.synchronize()') != 2):
                raise RuntimeError('unexpected metadata sync scope')
        elif new.count(b'.synchronize(') != raw.count(b'.synchronize('):
            raise RuntimeError('new blocking API: ' + key)
        if key == 'runner':
            capture_at = new.find(b'_extreme_runtime.target_metadata.capture_graph(')
            timer_at = new.find(b'_extreme_wall_start = time.perf_counter()')
            if capture_at < 0 or timer_at < 0 or capture_at >= timer_at:
                raise RuntimeError('metadata capture must precede Runtime timer')
        original[key], edited[key] = raw, new
    return pin, original, edited


def install(state_dir):
    if state_dir.exists():
        raise RuntimeError('fresh state directory required')
    pin, original, edited = prepared()
    state_dir.mkdir(mode=0o700, parents=True)
    atomic(state_dir / 'manifest.json', PIN.read_bytes())
    for key, raw in original.items():
        atomic(state_dir / f'{key}.orig', raw)
    try:
        for key, row in pin['sources'].items():
            path = Path(row['path'])
            if path.read_bytes() != original[key]:
                raise RuntimeError('install race: ' + key)
            atomic(path, edited[key])
        for key, row in pin['sources'].items():
            if sha(Path(row['path']).read_bytes()) != row['candidate_sha256']:
                raise RuntimeError('install verification: ' + key)
    except BaseException:
        for key, row in pin['sources'].items():
            path = Path(row['path'])
            if sha(path.read_bytes()) == row['candidate_sha256']:
                atomic(path, original[key])
        raise
    return pin


def restore(state_dir):
    pin = json.loads((state_dir / 'manifest.json').read_text())
    if pin != json.loads(PIN.read_text()):
        raise RuntimeError('manifest drift')
    backups = {}
    for key, row in pin['sources'].items():
        path = Path(row['path'])
        backup = (state_dir / f'{key}.orig').read_bytes()
        if sha(backup) != row['before_sha256'] or sha(path.read_bytes()) not in (
                row['before_sha256'], row['candidate_sha256']):
            raise RuntimeError('restore ownership drift: ' + key)
        backups[key] = backup
    for key, row in pin['sources'].items():
        path = Path(row['path'])
        if sha(path.read_bytes()) == row['candidate_sha256']:
            atomic(path, backups[key])
    for key, row in pin['sources'].items():
        if sha(Path(row['path']).read_bytes()) != row['before_sha256']:
            raise RuntimeError('restore verification: ' + key)
    return pin


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('check', 'install', 'restore'))
    parser.add_argument('--state-dir', type=Path)
    parser.add_argument('--record', required=True, type=Path)
    parser.add_argument('--offline-confirmed', action='store_true')
    args = parser.parse_args()
    if args.action != 'check' and not args.offline_confirmed:
        raise RuntimeError('offline confirmation required')
    if args.action == 'check':
        pin = prepared()[0]
    elif args.action == 'install':
        pin = install(args.state_dir)
    else:
        pin = restore(args.state_dir)
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps({'action': args.action, 'pin': pin}, indent=2)+'\n')
    print(json.dumps({'action': args.action, 'files': len(pin['sources'])}))


if __name__ == '__main__':
    main()
