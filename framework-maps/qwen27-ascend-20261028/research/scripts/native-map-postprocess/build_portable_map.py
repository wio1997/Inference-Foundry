#!/usr/bin/env python3
"""Inline exact CPU lifecycle shards into an exported standalone map HTML."""
import argparse
import base64
import gzip
import hashlib
import json
import re
from pathlib import Path

DATA_RE = re.compile(r'(<script id="combined-data" type="application/json">)(.*?)(</script>)', re.S)
APP_START = "<script>\n'use strict';\nconst DATA=JSON.parse(document.getElementById('combined-data').textContent);"
PORTABLE_JS = "<script id=\"portable-cpu-shards\" type=\"application/json\">{payload}</script>\n<script>\n'use strict'; const PORTABLE_CPU_SHARDS=JSON.parse(document.getElementById('portable-cpu-shards').textContent); const DATA=JSON.parse(document.getElementById('combined-data').textContent);"
FETCH_OLD = "const response=await fetch(w.detail_href);if(!response.ok)throw new Error(`HTTP ${response.status}`);const bytes=await response.arrayBuffer();"
FETCH_NEW = "if(!('DecompressionStream' in window))throw new Error('This browser does not support gzip DecompressionStream');const embedded=PORTABLE_CPU_SHARDS[w.detail_href];if(!embedded)throw new Error('portable CPU shard is missing');const packed=Uint8Array.from(atob(embedded.gzip_base64),c=>c.charCodeAt(0));const bytes=await new Response(new Blob([packed]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def build(source: Path, output: Path):
    source = source.resolve()
    output = output.resolve()
    html = source.read_text(encoding='utf-8')
    match = DATA_RE.search(html)
    if not match:
        raise ValueError('combined-data script not found')
    combined = json.loads(match.group(2))
    waves = combined.get('new_measurements', {}).get('cpu_lifecycle_map', {}).get('waves')
    if not isinstance(waves, list) or len(waves) != 41:
        raise ValueError(f'expected 41 CPU lifecycle waves, got {len(waves) if isinstance(waves,list) else type(waves).__name__}')
    refs = {}
    for wave in waves:
        href, expected = wave.get('detail_href'), wave.get('detail_sha256')
        if not isinstance(href, str) or not isinstance(expected, str) or href in refs:
            raise ValueError(f'invalid/duplicate detail receipt: {href!r}')
        shard_path = (source.parent / href).resolve()
        if source.parent not in shard_path.parents:
            raise ValueError(f'shard escapes preview root: {href}')
        raw = shard_path.read_bytes()
        actual = digest(raw)
        if actual != expected:
            raise ValueError(f'source shard SHA mismatch: {href}: {actual} != {expected}')
        compressed = gzip.compress(raw, compresslevel=9, mtime=0)
        refs[href] = {'sha256': actual, 'bytes': len(raw), 'gzip_sha256': digest(compressed),
                      'gzip_bytes': len(compressed), 'gzip_base64': base64.b64encode(compressed).decode('ascii')}
    if FETCH_OLD not in html:
        raise ValueError('expected lazy fetch expression was not found exactly once')
    html = html.replace(FETCH_OLD, FETCH_NEW, 1)
    if html.count(FETCH_NEW) != 1:
        raise ValueError('lazy fetch replacement count is not one')
    payload = json.dumps(refs, separators=(',', ':'), ensure_ascii=True)
    if html.count(APP_START) != 1:
        raise ValueError(f'expected one application script start, found {html.count(APP_START)}')
    html = html.replace(APP_START, PORTABLE_JS.format(payload=payload), 1)
    if 'const PORTABLE_CPU_SHARDS=' not in html:
        raise ValueError('portable shard index injection failed')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding='utf-8')
    receipt = {'schema_version': 1, 'status': 'portable_preview_not_final',
      'source_html': str(source), 'source_html_sha256': digest(source.read_bytes()),
      'source_combined_map_sha256': digest((source.parent/'combined-map.json').read_bytes()),
      'output_html': str(output), 'output_html_sha256': digest(output.read_bytes()),
      'shard_count': len(refs), 'uncompressed_shard_bytes': sum(x['bytes'] for x in refs.values()),
      'compressed_shard_bytes': sum(x['gzip_bytes'] for x in refs.values()),
      'embedded_base64_bytes': sum(len(x['gzip_base64']) for x in refs.values()),
      'shards': {k: {a:b for a,b in v.items() if a!='gzip_base64'} for k,v in refs.items()},
      'behavior': 'Only the selected CPU detail shard is base64-decoded and gzip-decompressed; SHA-256 is checked against original uncompressed bytes before JSON parsing.'}
    receipt_path = output.with_name('portable-build-receipt.json')
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    return receipt


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    print(json.dumps(build(a.source,a.output),indent=2))

if __name__=='__main__':
    main()
