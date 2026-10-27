"""Create a public derivative archive from the sealed Q27 delivery.

Only archival export; never modifies source artifacts or experiment verdicts.
The caller supplies source/destination paths. No network, keys or NPU access.
"""
import argparse
import base64
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import re

PRIVATE_PATH = re.compile(r'(?<![\w/])/(?:data|root|home|workspace|tmp|mnt|var/lib)(?:/[^\s\"\'<>`\\)\],;]*)?')
PRIVATE_REPO = re.compile(r'(?:https?://github\.com/)?(?:wio1997/)?(?:[A-Za-z0-9_.-]+-perf-lab|[A-Za-z0-9_.-]+-ascend-logs)(?:[^\s\"\'<>`\\)\],;]*)?')
IP = re.compile(r'(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])')
HOST = re.compile(r'\bbms-[A-Za-z0-9_.-]+\b')
CREDENTIAL = re.compile(r'gh[pousr]_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{24,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----')
SCRIPT = re.compile(r'(<script[^>]*\bid="([^"]+)"[^>]*type="application/json"[^>]*>)(.*?)(</script>)', re.S)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def jbytes(value):
    return (json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n').encode()


def numeric_digest(value):
    digest = hashlib.sha256()
    count = 0
    def visit(x):
        nonlocal count
        if isinstance(x, dict):
            for v in x.values():
                visit(v)
        elif isinstance(x, list):
            for v in x:
                visit(v)
        elif isinstance(x, (int, float)) and not isinstance(x, bool):
            digest.update((repr(x) + '\n').encode())
            count += 1
    visit(value)
    return {'count': count, 'sha256': digest.hexdigest()}


def scrub(text):
    if CREDENTIAL.search(text):
        raise ValueError('credential material: export stopped')
    text = PRIVATE_REPO.sub(lambda m: 'PRIVATE_REPOSITORY_' + sha(m[0].encode())[:12], text)
    text = PRIVATE_PATH.sub(lambda m: 'PRIVATE_PATH_' + sha(m[0].encode())[:12], text)
    text = HOST.sub('EXPERIMENT_HOST', text)
    text = IP.sub(lambda m: m[0] if m[0] in ('127.0.0.1', '0.0.0.0') else
                  'HOST_' + sha(m[0].encode())[:12], text)
    return re.sub(r'\[([^\]]+)\]\((?:PRIVATE_PATH_|PRIVATE_REPOSITORY_)[^)]*\)', r'\1（非公开定位，参见公开导出清单）', text)


def clean(value):
    if isinstance(value, str):
        return scrub(value)
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, dict):
        if 'gzip_base64' in value:
            raise ValueError('encoded payload requires explicit decode/sanitize/repack')
        result = {scrub(k): clean(v) for k, v in value.items()}
        if len(result) != len(value):
            raise ValueError('redaction key collision')
        return result
    return value


def replace_refs(value, replacements):
    if isinstance(value, str):
        return replacements.get(value, value)
    if isinstance(value, list):
        return [replace_refs(v, replacements) for v in value]
    if isinstance(value, dict):
        return {k: replace_refs(v, replacements) for k, v in value.items()}
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research-source', type=Path, required=True)
    parser.add_argument('--map-source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=False)
    receipts = []
    checks = []
    cpu_refs = {}
    sources = [(args.research_source, 'research'), (args.map_source, 'map')]

    def write_file(relative, raw, origin, original):
        target = args.destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        receipts.append({'path': relative, 'bytes': len(raw), 'sha256': sha(raw),
                         'source_alias': origin, 'source_sha256': sha(original),
                         'source_bytes': len(original), 'changed': raw != original})

    # Materialize JSON CPU shards first: HTML's compressed copies and its digest
    # references must describe these public bytes rather than private originals.
    cpu_dir = args.map_source / 'cpu-lifecycle-details'
    for path in sorted(cpu_dir.glob('*.json')):
        original = path.read_bytes()
        data = json.loads(original)
        derived = clean(data)
        assert numeric_digest(data) == numeric_digest(derived)
        raw = jbytes(derived)
        cpu_refs[sha(original)] = sha(raw)
        write_file('map/cpu-lifecycle-details/' + path.name, raw, 'map-snapshot', original)

    for source, prefix in sources:
        for path in sorted(source.rglob('*')):
            if not path.is_file():
                continue
            if path.is_symlink() or not path.resolve().is_relative_to(source.resolve()):
                raise ValueError('unexpected source link')
            relative = path.relative_to(source)
            if prefix == 'map' and (relative.parts[0] in ('native-rank-details', 'cpu-lifecycle-details')
                                  or path.name.endswith('-download.json')):
                continue
            if path.suffix not in ('.json', '.md', '.csv', '.py', '.html', '.js', '.txt', '.sh', '.patch', '.diff', '.cpp', '.h', '.hpp', '.c', '.cc', '.cu', '.log'):
                raise ValueError('unexpected artifact type: ' + relative.as_posix())
            original = path.read_bytes()
            text = original.decode()
            if path.suffix == '.json':
                data = json.loads(text)
                derived = clean(data)
                assert numeric_digest(data) == numeric_digest(derived)
                if path.name == 'combined-map.json':
                    checks.append({'artifact': 'map/combined-map.json', 'numeric_values_unchanged': numeric_digest(data)})
                    derived = replace_refs(derived, cpu_refs)
                    derived['metadata']['public_archive_scope'] = '脱敏研究快照；原生rank事件大文件未公开；整体机制验收仍有缺口；非优化采用。'
                    derived['new_measurements']['native_sharded_map']['public_export_no_raw'] = True
                raw = jbytes(derived)
            elif prefix == 'map' and relative.as_posix() == 'index.html':
                def block(match):
                    data = json.loads(match[3])
                    if match[2] == 'portable-cpu-shards':
                        result = {}
                        for href, entry in data.items():
                            source_payload = gzip.decompress(base64.b64decode(entry['gzip_base64']))
                            assert sha(source_payload) == entry['sha256']
                            public_payload = jbytes(clean(json.loads(source_payload)))
                            expected = (args.destination / 'map' / href).read_bytes()
                            assert public_payload == expected
                            packed = gzip.compress(public_payload, mtime=0)
                            result[href] = dict(sha256=sha(public_payload), bytes=len(public_payload),
                                                gzip_sha256=sha(packed), gzip_bytes=len(packed),
                                                gzip_base64=base64.b64encode(packed).decode())
                    else:
                        result = clean(data)
                        assert numeric_digest(data) == numeric_digest(result)
                        checks.append({'artifact': 'map/index.html#' + match[2],
                                       'numeric_values_unchanged': numeric_digest(data)})
                        result = replace_refs(result, cpu_refs)
                        if match[2] == 'combined-data':
                            result['metadata']['public_archive_scope'] = '脱敏研究快照；整体机制验收未完成。'
                            result['new_measurements']['native_sharded_map']['public_export_no_raw'] = True
                    return match[1] + jbytes(result).decode().replace('</', '<\\/') + match[4]
                # Scrub visible UI/JS separately. Base64 is sanitized by decoding
                # above; scanning the encoded text would miss private payloads.
                chunks = []
                cursor = 0
                for match in SCRIPT.finditer(text):
                    chunks.extend([scrub(text[cursor:match.start()]), block(match)])
                    cursor = match.end()
                chunks.append(scrub(text[cursor:]))
                text = ''.join(chunks)
                anchor = ' const source=DATA.new_measurements?.native_sharded_map;'
                assert text.count(anchor) == 1
                text = text.replace(anchor, anchor + '''
 if(source?.public_export_no_raw){
  const c=card('原生 profiling 证据索引（公开摘要）','完整原生事件未随公开归档分发，明细加载已关闭。');
  c.append(el('p','warn',`已归档 ${source.index?.per_rank?.length??'unknown'} 个rank身份。这里只保留索引与摘要，不能替代原始trace或完整关键路径证明。`));
  root.append(c);return true;
 }
''')
                text = text.replace('<body>', '<body><aside style="padding:14px;background:#fff3cd;color:#553c00">公开研究归档：数值保留，定位信息脱敏；未分发原始rank事件。输出一致性、输出完成因果链和全stream关键路径仍待验证；非优化采用结论。<a href="../README.md">归档说明</a></aside>', 1)
                raw = text.encode()
            else:
                raw = scrub(text).encode()
                if path.suffix == '.csv':
                    before = list(csv.reader(io.StringIO(text)))
                    after = list(csv.reader(io.StringIO(raw.decode())))
                    assert len(before) == len(after)
                    for a, b in zip(before, after):
                        assert len(a) == len(b)
                        for x, y in zip(a, b):
                            try:
                                float(x)
                            except ValueError:
                                continue
                            assert x == y, (relative, x, y)
                    checks.append({'artifact': prefix + '/' + relative.as_posix(),
                                   'CSV_rows': len(before), 'numeric_cells_unchanged': True})
            write_file(prefix + '/' + relative.as_posix(), raw, prefix + '-snapshot', original)

    manifest = {'schema_version': 1, 'archive': 'qwen27-ascend-20261028',
                'status': 'public_derivative_research_snapshot_not_whole_task_acceptance',
                'original_checkpoint_generation': 105, 'raw_native_events_distributed': False,
                'private_locations_removed': True,
                'historical_internal_digests': '研究文件内历史SHA属于原始输入；本公开副本实际字节以files清单为准。CPU交互的SHA已绑定公开副本。',
                'files': receipts, 'preservation_checks': checks}
    (args.destination / 'PUBLIC_EXPORT_MANIFEST.json').write_bytes(jbytes(manifest))
    print(json.dumps({'files': len(receipts), 'bytes': sum(f['bytes'] for f in receipts),
                      'checks': len(checks), 'output': str(args.destination)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
