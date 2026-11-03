from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib, json, socket, sqlite3, time

assert socket.gethostname() == 'bm-jn-zs-zone1-910C-64G-10-105'
P = Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72')
OUT = P/'analysis/M21-all32-global-copy-scalar-sync-and-outside-prepare-callers-v37.json'
assert not OUT.exists()

def pin(path, expected):
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == expected
    return json.loads(data)

index = pin(P/'analysis/profile-capabilities-index-v12-v2.json', '79ba556ef3855b9066392e438371a0a9d3b38ea7511269ce7ff2a6814529a095')
prep = pin(P/'analysis/prepare-field-origins-v12.json', 'c23b75f3fbd3ae3d3d7152f6882524e899bade53001ea2f1befa09ce943f091f')
assert len(index['databases']) == 32
roots = defaultdict(list)
for root in prep['prepare_roots']:
    roots[root['database_index']].append(root)
started = time.monotonic()
records, checks = [], []
chains = {}
names = ['aten::_pin_memory', 'Event::synchronize', 'Stream::synchronize', 'NPUStream::synchronize',
         'aten::clone', 'aten::index_fill', 'aten::index_fill_', 'aten::item', 'aten::_local_scalar_dense']

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024**2), b''):
            h.update(chunk)
    return h.hexdigest()

def scope(frames):
    text = '\n'.join(frames)
    if '_dummy_run' in text or ': dummy_run' in text:
        return 'DUMMY_ANCESTRY'
    for function, label in [('prepare_next_token_ids_padded', 'NEXT_ID_HELPER'),
                            ('_copy_valid_sampled_token_count', 'VALID_COUNT_PUBLICATION'),
                            ('take_draft_token_ids', 'DRAFT_OUTPUT_RETRIEVAL')]:
        if ': '+function in text:
            return label
    if '/spec_decode/utils.py(' in text and any(s in text for s in ['compute_verify_budget', 'allocate_verify_budget', '_compute_dspark_token_probs']):
        return 'DYNAMIC_VERIFY_BUDGET'
    if '/spec_decode/' in text or ': propose_draft_token_ids' in text:
        return 'DRAFT_OR_PROPOSAL_ANCESTRY'
    if ': _prepare_inputs' in text or ': _preprocess' in text:
        return 'INPUT_PREPARATION_ANCESTRY'
    return 'OTHER_OR_NO_PROJECT_ANCESTRY'

for di, spec in enumerate(index['databases']):
    assert time.monotonic()-started < 195, 'finite raw query deadline'
    path = Path(spec['path'])
    before = path.stat()
    assert digest(path) == spec['sha256'], str(path)
    db = sqlite3.connect(path.as_uri()+'?mode=ro&immutable=1', uri=True)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA query_only=ON')
    db.set_progress_handler(lambda: int(time.monotonic()-started > 205), 100000)
    query = '''SELECT p.rowid AS api_rowid,CAST(p.startNs AS INTEGER) AS startNs,
      CAST(p.endNs AS INTEGER) AS endNs,p.globalTid,p.connectionId,p.callchainId,
      s.value AS name,sh.value AS input_shapes,dt.value AS input_dtypes
      FROM PYTORCH_API p JOIN STRING_IDS s ON s.id=p.name
      LEFT JOIN STRING_IDS sh ON sh.id=p.inputShapes LEFT JOIN STRING_IDS dt ON dt.id=p.inputDtypes
      WHERE s.value IN ('''+','.join('?' for _ in names)+''') OR s.value LIKE '%memcpy%'
      ORDER BY CAST(p.startNs AS INTEGER),p.rowid'''
    selected = [dict(row) for row in db.execute(query, names)]
    ids = sorted({row['callchainId'] for row in selected if row['callchainId'] is not None})
    local = defaultdict(list)
    for offset in range(0, len(ids), 500):
        batch = ids[offset:offset+500]
        query = 'SELECT c.id,c.stackDepth,s.value AS frame FROM PYTORCH_CALLCHAINS c JOIN STRING_IDS s ON s.id=c.stack WHERE c.id IN ('+','.join('?' for _ in batch)+') ORDER BY c.id,c.stackDepth'
        for row in db.execute(query, batch):
            local[row['id']].append((row['stackDepth'], row['frame']))
    keys = {}
    for cid, frames in local.items():
        key = hashlib.sha256(json.dumps(frames, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
        keys[cid] = key
        chains.setdefault(key, [dict(stackDepth=depth, frame=frame) for depth, frame in frames])
    copies = defaultdict(list)
    query = '''SELECT p.rowid AS api_rowid,CAST(p.startNs AS INTEGER) AS startNs,
      CAST(p.endNs AS INTEGER) AS endNs,p.globalTid,p.callchainId,
      sh.value AS input_shapes,dt.value AS input_dtypes FROM PYTORCH_API p
      JOIN STRING_IDS s ON s.id=p.name LEFT JOIN STRING_IDS sh ON sh.id=p.inputShapes
      LEFT JOIN STRING_IDS dt ON dt.id=p.inputDtypes WHERE s.value='aten::copy_' '''
    for row in db.execute(query):
        value = dict(row)
        if value['callchainId'] in keys:
            copies[(value['callchainId'], value['globalTid'])].append(value)
    for row in selected:
        key = keys.get(row['callchainId'])
        frames = [f['frame'] for f in chains.get(key, []) if '/vllm_ascend/' in f['frame'] or '/vllm/vllm/' in f['frame']]
        bound = [r['root_key'] for r in roots[di] if r['root']['globalTid'] == row['globalTid'] and r['root']['startNs'] <= row['startNs'] and row['endNs'] <= r['root']['endNs']]
        parents = [r for r in copies.get((row['callchainId'], row['globalTid']), []) if r['startNs'] <= row['startNs'] and row['endNs'] <= r['endNs']] if 'memcpy' in row['name'] else []
        minimum = min((r['endNs']-r['startNs'] for r in parents), default=None)
        inner = [r for r in parents if r['endNs']-r['startNs'] == minimum]
        row.update(database_index=di, chain_key=key, project_frames=frames, prepare_root_keys=bound,
                   scope=scope(frames), innermost_copy_parents=inner, parent_copy_candidate_count=len(parents))
        records.append(row)
    db.close()
    after = path.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    checks.append(dict(database_index=di, path=str(path), sha256=spec['sha256'], bytes=before.st_size, stat_unchanged=True, selected_records=len(selected)))

grouped = defaultdict(list)
for row in records:
    grouped[(row['name'], row['scope'], bool(row['prepare_root_keys']), row['chain_key'])].append(row)
headers = []
for (name, role, inside, key), rows in grouped.items():
    headers.append(dict(name=name, scope=role, inside_prepare=inside, chain_key=key, records=len(rows),
        databases=len({r['database_index'] for r in rows}), project_frames=rows[0]['project_frames'],
        parent_shapes=[dict(input_shapes=k[0], input_dtypes=k[1], records=v) for k,v in Counter((p['input_shapes'],p['input_dtypes']) for r in rows for p in r['innermost_copy_parents']).items()]))
summary = dict(created_utc=datetime.now(timezone.utc).isoformat(), databases=32, selected_host_API_records=len(records), canonical_fullchains=len(chains),
    inside_prepare=sum(bool(r['prepare_root_keys']) for r in records), outside_prepare=sum(not r['prepare_root_keys'] for r in records),
    name_inside_counts=[dict(name=k[0],inside_prepare=k[1],records=v) for k,v in Counter((r['name'],bool(r['prepare_root_keys'])) for r in records).items()],
    scope_counts=[dict(scope=k[0],name=k[1],inside_prepare=k[2],records=v) for k,v in Counter((r['scope'],r['name'],bool(r['prepare_root_keys'])) for r in records).items()],
    whole_module_closed=False, overall_complete=False)
result = dict(summary=summary, databases=checks, records=records, full_canonical_callchains=chains, caller_groups=headers,
    limits=['Raw32DB immutable SQL/fullSHA/stat checked; selected hostAPIs are not allCPUwork/allcopies/runtimeCANNevenTs.',
            'Prepare links use exact same-thread containment; outside remains unpaired to target roots.',
            'Copy-parent relation is same-chain/thread host containment, not pointer/DMA bytes/alias proof.',
            'Nested host walls are not exclusiveCPU/request gain; host/device clocks not subtracted.',
            'No new source/model/torch/NPU/service/health work.'])
payload = (json.dumps(result, ensure_ascii=False, separators=(',', ':'))+'\n').encode()
assert len(payload) < 350*1024**2, 'artifact size guard'
OUT.write_bytes(payload)
print(json.dumps(dict(summary=summary, focused_callers=[h for h in headers if h['scope'] in ['NEXT_ID_HELPER','VALID_COUNT_PUBLICATION','DRAFT_OUTPUT_RETRIEVAL','DYNAMIC_VERIFY_BUDGET']], full_result=str(OUT), result_sha256=hashlib.sha256(payload).hexdigest()), ensure_ascii=False))
