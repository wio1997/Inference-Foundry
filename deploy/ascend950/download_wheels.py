#!/usr/bin/env python3
"""Fetch public ARM64 wheels from frozen URLs; verify SHA256 before installation."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import time
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parent


def digest_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    entries = json.loads((ROOT / 'locks/wheels.json').read_text())
    entries += json.loads((ROOT / 'locks/recipe-wheels.json').read_text())
    folder = ROOT / 'wheelhouse'
    folder.mkdir(exist_ok=True)

    def one(item):
        name = unquote(urlsplit(item['url']).path.rsplit('/', 1)[-1])
        target = folder / name
        if target.exists():
            if digest_file(target) == item['sha256']:
                return
        partial = target.with_name(name + '.partial')
        command = ['curl', '-4', '-fLsS', '--connect-timeout', '20', '--retry', '5',
                   '--retry-all-errors', '-C', '-', '-o', str(partial), item['url']]
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as error:
            if error.returncode not in (22, 33, 36) or not partial.exists():
                raise
            partial.rename(partial.with_name(partial.name + '.bad-' + str(time.time_ns())))
            subprocess.run(command, check=True)
        actual = digest_file(partial)
        if actual != item['sha256']:
            partial.rename(partial.with_name(partial.name + '.bad-' + str(time.time_ns())))
            raise RuntimeError('SHA256 mismatch: ' + name)
        partial.replace(target)
        print('VERIFIED', name, flush=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(one, entries))
    print('WHEELS_VERIFIED', len(entries))


if __name__ == '__main__':
    main()
