"""PREPARED, NOT EXECUTED: bounded immutable SQL/config read of the original saved capture.

No imports of torch/the model, device calls, process queries, service operations,
new profiling or SSH authentication. Relocated capture roots must retain the exact
recorded bytes; this is not a tool for substituting a fresh193 capture.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import socket
import sqlite3
import time

ORIGINAL = Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72')
CAPS_SHA = '79ba556ef3855b9066392e438371a0a9d3b38ea7511269ce7ff2a6814529a095'
NATIVE_SHA = 'a3aa47cdf9170785432ecbc18117a489f3fba37d89959f110f4fcbd8890c7dc4'
TABLES = ('NPU_INFO', 'HOST_INFO', 'META_DATA', 'SESSION_TIME_INFO', 'RANK_DEVICE_MAP')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--capture-root', type=Path, default=ORIGINAL)
    ap.add_argument('--expected-hostname', default='bm-jn-zs-zone1-910C-64G-10-105')
    args = ap.parse_args()
    assert socket.gethostname() == args.expected_hostname, 'unexpected execution host'
    root = args.capture_root.resolve(strict=True)
    start = time.monotonic()

    def bounded():
        assert time.monotonic() - start < 140, 'finite metadata query time guard'

    def relocated(original_path):
        relative = Path(original_path).relative_to(ORIGINAL)
        result = (root / relative).resolve(strict=True)
        assert result.is_relative_to(root), 'capture path escapes relocated root'
        return result

    def digest(path):
        h = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
                bounded()
                h.update(block)
        return h.hexdigest()

    def pin_json(path, expected):
        assert digest(path) == expected, 'saved identity mismatch: ' + str(path)
        return json.loads(path.read_bytes())

    def cell(value):
        if isinstance(value, bytes):
            return dict(type='BLOB', bytes=len(value), sha256=hashlib.sha256(value).hexdigest(), contents_emitted=False)
        if isinstance(value, str) and len(value) > 2048:
            return dict(type='LONG_TEXT', chars=len(value), prefix=value[:2048], truncated=True)
        return value

    caps_path = root / 'analysis/profile-capabilities-index-v12-v2.json'
    native_path = root / 'analysis/native-execution-identity-v14/index.json'
    caps = pin_json(caps_path, CAPS_SHA)
    native = pin_json(native_path, NATIVE_SHA)
    assert caps['summary']['run_id'] == ORIGINAL.name
    assert caps['summary']['input_sha256'] == '755b823003b4d200a4590424a245cdfb484d9b8c3d0b5225a7a058790b87e8c8'
    assert len(caps['databases']) == len(caps['profiler_infos']) == len(native['databases']) == 32
    flags = []
    for ref in caps['profiler_infos']:
        bounded()
        p = relocated(ref['path'])
        assert p.stat().st_size < 2 * 1024 * 1024, 'unexpected profiler_info size'
        obj = pin_json(p, ref['sha256'])
        config = obj['config']
        flags.append(dict(original_path=ref['path'], actual_path=str(p), sha256=ref['sha256'],
            rank_id=obj.get('rank_id'), torch_npu_version=obj.get('torch_npu_version'),
            cann_version=obj.get('cann_version'),
            common_config=config.get('common_config'), experimental_config=config.get('experimental_config'),
            start_info=obj.get('start_info'), end_info=obj.get('end_info')))
    databases = []
    for di, ref in enumerate(caps['databases']):
        bounded()
        p = relocated(ref['path'])
        before = p.stat()
        assert digest(p) == ref['sha256'], 'original DB SHA mismatch'
        db = sqlite3.connect(p.as_uri() + '?mode=ro&immutable=1', uri=True)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA query_only=ON')
        db.set_progress_handler(lambda: int(time.monotonic() - start > 140), 100000)
        try:
            tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            metadata = []
            for table in TABLES:
                # Every identifier is from a fixed constant list, never arbitrary SQL input.
                if table not in tables:
                    metadata.append(dict(table=table, present=False))
                    continue
                schema = [dict(r) for r in db.execute('PRAGMA table_info("' + table + '")')]
                count = db.execute('SELECT COUNT(*) FROM "' + table + '"').fetchone()[0]
                # META_DATA values can contain unspecified payloads; schema/count only.
                rows = [] if table == 'META_DATA' else [{k: cell(r[k]) for k in r.keys()} for r in db.execute('SELECT * FROM "' + table + '" LIMIT 128')]
                metadata.append(dict(table=table, present=True, schema=schema, row_count=count,
                    rows=rows, rows_complete=(table != 'META_DATA' and count <= 128),
                    value_scope='SCHEMA_COUNT_ONLY' if table == 'META_DATA' else 'BOUNDED_FIRST128_ROWS'))
            after = p.stat()
            assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'DB changed while read'
            databases.append(dict(database_index=di, original_path=ref['path'], actual_path=str(p),
                sha256=ref['sha256'], all_table_names=sorted(tables), metadata=metadata,
                source_stat_unchanged=True))
        finally:
            db.close()
    payload = json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(),
        execution_host=socket.gethostname(), capture_root=str(root), original_capture=str(ORIGINAL),
        capability_index_sha256=CAPS_SHA, native_index_sha256=NATIVE_SHA,
        capture_configurations=flags, databases=databases,
        limits=['Metadata identity/config fields do not prove graph phases,tensor versions,native event handles or physical peaks.',
                'META_DATA values not emitted;other metadata at most128rows with explicit completeness flag.',
                'No source/model/device/service/process or new capture work;no raw files mutated.'],
        new_capture=False, whole_modules_closed=0, overall_complete=False), ensure_ascii=False)
    assert len(payload.encode()) < 2 * 1024 * 1024, 'bounded output guard'
    print(payload)


if __name__ == '__main__':
    main()
