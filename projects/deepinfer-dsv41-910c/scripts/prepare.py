#!/usr/bin/env python3
"""Verify and restore the fixed deployment files without overwriting differences."""
import hashlib
import json
from pathlib import Path
import socket
import tarfile

ROOT = Path('/data/tiankuan/wio/910c-deepinfer-172')
VERSION = 'allopt-mc2-cap4096-v1-20261028'
PACKAGE = Path(__file__).resolve().parents[1]


def restore(archive, destination):
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            name = Path(member.name)
            if name.is_absolute() or '..' in name.parts or not (member.isfile() or member.isdir()):
                raise ValueError('Unsupported archive member: ' + member.name)
            target = destination / name
            # Reject existing symlink ancestors as well as symlink leaf paths.
            for component in (target, *target.parents):
                if component == destination.parent:
                    break
                if component.is_symlink():
                    raise ValueError('Symlink in destination: ' + str(component))
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            data = tar.extractfile(member).read()
            if target.exists():
                if target.read_bytes() != data:
                    raise ValueError('Existing file differs: ' + str(target))
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as output:
                output.write(data)
            target.chmod(member.mode & 0o777)


def copy_tree(source, destination):
    for file in source.rglob('*'):
        if not file.is_file():
            continue
        target = destination / file.relative_to(source)
        data = file.read_bytes()
        if target.exists():
            if target.read_bytes() != data:
                raise ValueError('Existing file differs: ' + str(target))
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as output:
                output.write(data)


def main():
    if socket.gethostname() != 'bm-jn-zs-zone1-910C-64G-10-172':
        raise SystemExit('This package is pinned to server 172.')
    if PACKAGE != ROOT / 'delivery/deepinfer-dsv41-910c':
        raise SystemExit('Place the package at the documented delivery path.')
    archives = json.loads((PACKAGE / 'config/archives.json').read_text())
    for name, info in archives.items():
        file = PACKAGE / 'artifacts' / name
        if file.stat().st_size != info['bytes'] or hashlib.sha256(file.read_bytes()).hexdigest() != info['sha256']:
            raise ValueError('Archive verification failed: ' + name)
    for name in ('tmp', 'home', 'cache', 'build/torch_extensions', 'logs/ascend',
                 'runs', 'tools', 'workload', 'benchmarks', 'serving', 'diagnostics',
                 'src/DeepInfer', 'cache/matplotlib', 'cache/huggingface', 'cache/pip',
                 'cache/ascend', 'cache/ascend-work', 'cache/triton-' + VERSION,
                 'cache/ascend-' + VERSION):
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    source = ROOT / 'src' / ('DeepInfer-' + VERSION)
    restore(PACKAGE / 'artifacts/source.tar.gz', source)
    manifest = json.loads((PACKAGE / 'config/source-SHA256.json').read_text())
    for name, digest in manifest.items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Source verification failed: ' + name)
    records = PACKAGE / 'records'
    restore(PACKAGE / 'artifacts/evidence.tar.gz', records)
    restore(PACKAGE / 'artifacts/support.tar.gz', records)
    copy_tree(records / 'workload/frozen', ROOT / 'workload/frozen')
    copy_tree(records / 'tools/aisbench-adapter', ROOT / 'tools/aisbench-adapter')
    print(json.dumps({'state': 'PREPARED', 'source_files': len(manifest), 'source': str(source)}))


if __name__ == '__main__':
    main()
