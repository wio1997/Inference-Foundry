#!/usr/bin/env python3
"""Download/verify the historical 84-file ModelScope snapshot by file revision."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
from urllib.parse import quote, urlencode

ROOT = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(16 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dest', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    manifest = json.loads((ROOT / 'locks/model-manifest.json').read_text())
    args.dest.mkdir(parents=True, exist_ok=True)

    def one(item):
        relative = PurePosixPath(item['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe manifest path')
        target = args.dest / relative
        if target.is_file() and target.stat().st_size == item['bytes'] and sha256(target) == item['sha256']:
            return {'path': item['path'], 'passed': True}
        if args.verify_only:
            raise RuntimeError('Missing or mismatched file: ' + item['path'])
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + '.partial')
        url = 'https://modelscope.cn/api/v1/models/' + quote(manifest['repo'], safe='/') + '/repo?'
        url += urlencode({'Revision': item['revision'], 'FilePath': item['path']})
        # Resume only our partial file. Final existing files are never overwritten.
        subprocess.run(['curl', '-4', '--fail', '--location', '--silent', '--show-error',
                        '--connect-timeout', '20', '--retry', '5', '--retry-all-errors',
                        '--continue-at', '-', '--output', str(partial), url], check=True)
        if partial.stat().st_size != item['bytes'] or sha256(partial) != item['sha256']:
            raise RuntimeError('SHA256/size mismatch; retained partial: ' + item['path'])
        if target.exists():
            raise RuntimeError('Existing final file mismatched; inspect it before replacement: ' + str(target))
        partial.replace(target)
        print('VERIFIED', item['path'], flush=True)
        return {'path': item['path'], 'passed': True}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(one, manifest['files']))
    index = json.loads((args.dest / 'model.safetensors.index.json').read_text())
    shards = set(index['weight_map'].values())
    assert len(shards) == 48 and all((args.dest / name).is_file() for name in shards)
    report = {'repo': manifest['repo'], 'phase': 'complete', 'file_count': len(results),
              'sha256_verified_files': len(results), 'weight_shards': len(shards),
              'total_bytes': manifest['total_bytes'], 'path': str(args.dest.resolve())}
    (args.dest / '.deployment-verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
