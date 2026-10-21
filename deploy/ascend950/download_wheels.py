#!/usr/bin/env python3
"""Fetch public ARM64 wheels from frozen URLs; verify SHA256 before installation."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parent


def main():
    entries = json.loads((ROOT / 'locks/wheels.json').read_text())
    entries += json.loads((ROOT / 'locks/recipe-wheels.json').read_text())
    folder = ROOT / 'wheelhouse'
    folder.mkdir(exist_ok=True)

    def one(item):
        name = unquote(urlsplit(item['url']).path.rsplit('/', 1)[-1])
        target = folder / name
        if target.exists():
            with target.open('rb') as source:
                if hashlib.file_digest(source, 'sha256').hexdigest() == item['sha256']:
                    return
        partial = target.with_name(name + '.partial')
        subprocess.run(['curl', '-4', '-fLsS', '--connect-timeout', '20', '--retry', '5',
                        '--retry-all-errors', '-C', '-', '-o', str(partial), item['url']], check=True)
        with partial.open('rb') as source:
            actual = hashlib.file_digest(source, 'sha256').hexdigest()
        if actual != item['sha256']:
            raise RuntimeError('SHA256 mismatch: ' + name)
        partial.replace(target)
        print('VERIFIED', name, flush=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(one, entries))
    print('WHEELS_VERIFIED', len(entries))


if __name__ == '__main__':
    main()
