"""Verify public artifact bytes and encoded CPU shards; no NPU or network."""
import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--node', help='Optional Node executable for JavaScript syntax checks')
    args = parser.parse_args()
    root = args.archive.resolve()
    manifest = json.loads((root / 'PUBLIC_EXPORT_MANIFEST.json').read_text())
    byte_count = 0
    for entry in manifest['files']:
        path = root / entry['path']
        assert path.resolve().is_relative_to(root), entry['path']
        raw = path.read_bytes()
        assert len(raw) == entry['bytes'] and sha(raw) == entry['sha256'], entry['path']
        byte_count += len(raw)
    html = (root / 'map/index.html').read_text()
    scripts = list(re.finditer(r'<script([^>]*)>(.*?)</script>', html, re.S))
    payloads = {}
    javascript = []
    for script in scripts:
        if 'application/json' in script[1]:
            name = re.search(r'id="([^"]+)"', script[1])[1]
            payloads[name] = json.loads(script[2])
        else:
            javascript.append(script[2])
    assert payloads['combined-data']['new_measurements']['native_sharded_map']['public_export_no_raw'] is True
    cpu = payloads['portable-cpu-shards']
    for href, entry in cpu.items():
        packed = base64.b64decode(entry['gzip_base64'], validate=True)
        assert len(packed) == entry['gzip_bytes'] and sha(packed) == entry['gzip_sha256'], href
        raw = gzip.decompress(packed)
        assert len(raw) == entry['bytes'] and sha(raw) == entry['sha256'], href
        assert (root / 'map' / href).read_bytes() == raw, href
        json.loads(raw)
    waves = payloads['combined-data']['new_measurements']['cpu_lifecycle_map'].get('waves', [])
    if isinstance(waves, dict):
        waves = list(waves.values())
    references = 0
    for wave in waves:
        if wave.get('detail_href') in cpu:
            assert wave['detail_sha256'] == cpu[wave['detail_href']]['sha256']
            references += 1
    syntax = 'not_requested'
    if args.node:
        with tempfile.TemporaryDirectory(prefix='q27-public-js-') as directory:
            for index, source in enumerate(javascript):
                file = Path(directory) / ('script-' + str(index) + '.js')
                file.write_text(source)
                subprocess.run([args.node, '--check', str(file)], check=True, capture_output=True)
        syntax = 'passed'
    print(json.dumps({'status': 'passed', 'manifest_files': len(manifest['files']),
                      'manifest_bytes': byte_count, 'portable_CPU_shards_verified': len(cpu),
                      'CPU_wave_SHA_references_verified': references,
                      'native_raw_detail_disabled': True, 'javascript_syntax': syntax,
                      'browser_QA': 'not_executed_for_public_derivative',
                      'performance_or_correctness_verdict': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
