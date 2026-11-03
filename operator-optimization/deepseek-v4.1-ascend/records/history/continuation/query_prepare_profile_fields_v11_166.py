from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, sqlite3, socket, time

assert socket.gethostname() == 'bm-jn-zs-zone1-910C-64G-10-105'
ARCHIVE = Path('/data/tiankuan/wio/migration-166-20261030/source169-project')
REL = Path('analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41')
BASE = ARCHIVE / REL
OUT = Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/profile-readonly-20261031/prepare-profile-field-capabilities-v11.json')
assert not OUT.exists()
data = json.loads((BASE / 'continuation-20261030/prepare-input-analysis.json').read_text())
EXPORT = BASE / 'diagnostic-20261029/official-20261030T084726Z-cb252d1d/analysis/operator-export/copies'
results = []
for spec in data['rows']:
    folder = EXPORT / spec['source_dir'] / 'ASCEND_PROFILER_OUTPUT'
    assert folder.is_dir(), folder
    entries = sorted(folder.iterdir())
    result = dict(rank=spec['rank'], folder=str(folder), entries=[dict(name=p.name, bytes=p.stat().st_size) for p in entries], prepare_roots=spec['prepare_input_scopes'])
    dbs = [p for p in entries if p.suffix == '.db']
    result['databases'] = []
    for p in dbs:
        before = p.stat()
        h = hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda: f.read(4*1024*1024), b''):
                h.update(b)
        db = sqlite3.connect(p.as_uri()+'?mode=ro&immutable=1', uri=True)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA query_only=ON')
        deadline = time.monotonic()+30
        db.set_progress_handler(lambda: int(time.monotonic()>deadline), 100000)
        tables = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        selected = [t for t in tables if any(x in t for x in ['PYTORCH', 'CALLCHAIN', 'MEMCPY', 'STRING_IDS', 'CONNECTION'])]
        schemas = {t: [dict(r) for r in db.execute('PRAGMA table_info("'+t+'")')] for t in selected}
        counts = {t: db.execute('SELECT COUNT(*) FROM "'+t+'"').fetchone()[0] for t in selected}
        examples = {t: [dict(r) for r in db.execute('SELECT * FROM "'+t+'" LIMIT 2')] for t in selected}
        db.close()
        after = p.stat()
        assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        result['databases'].append(dict(path=str(p), sha256=h.hexdigest(), bytes=before.st_size, tables=tables, selected_schemas=schemas, selected_counts=counts, selected_examples=examples, stat_unchanged=True))
    results.append(result)
payload = json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(), results=results, no_target_source_read=True, no_inference_NPU_or_service_mutation=True, overall_complete=False), ensure_ascii=False, separators=(',',':'))+'\n'
OUT.write_text(payload)
public = dict(full_result=str(OUT), result_sha256=hashlib.sha256(payload.encode()).hexdigest(), ranks=[dict(rank=r['rank'], files=r['entries'], databases=[dict(path=d['path'], sha256=d['sha256'], selected_counts=d['selected_counts'], selected_schemas=d['selected_schemas'], selected_examples=d['selected_examples']) for d in r['databases']]) for r in results])
assert len(json.dumps(public).encode()) < 30000
print(json.dumps(public, ensure_ascii=False))
