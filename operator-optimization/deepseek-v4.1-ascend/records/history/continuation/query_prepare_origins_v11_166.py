from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib, json, socket, sqlite3, time

assert socket.gethostname() == 'bm-jn-zs-zone1-910C-64G-10-105'
A=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/profile-readonly-20261031')
ARCHIVE=Path('/data/tiankuan/wio/migration-166-20261030/source169-project/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41')
OUT=A/'prepare-copy-pin-event-origins-v11.json'
assert not OUT.exists()
cap_bytes=(A/'prepare-profile-field-capabilities-v11.json').read_bytes()
assert hashlib.sha256(cap_bytes).hexdigest()=='cbefbc19e5737c05d7c8a6691361b31097c214eed29c2d186a251fd95fdf4b6a'
cap=json.loads(cap_bytes)
specs=[]
for r in cap['results']:
    d=next(d for d in r['databases'] if Path(d['path']).name.startswith('ascend_pytorch_profiler_'))
    specs.append(dict(population='OFFICIAL_LATER_WINDOW',rank=r['rank'],path=d['path'],sha256=d['sha256'],expected_roots=r['prepare_roots']))
specs.append(dict(population='EARLIER_DIAGNOSTIC_WITH_STACKS',rank=3,path=str(ARCHIVE/'diagnostic-20261029/runtime/diagnostic-traces/dp0_pp0_tp3_dcp0_ep3_rank3_2128_20261029023525650_ascend_pt/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_3.db'),sha256='084b8718888e33271435ec79d7573c95ca1615f3e7484c807ea24590fad234c4',expected_roots=None))
results=[]
for spec in specs:
    p=Path(spec['path']);before=p.stat();h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    assert h.hexdigest()==spec['sha256']
    db=sqlite3.connect(p.as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row
    db.execute('PRAGMA query_only=ON');deadline=time.monotonic()+75
    db.set_progress_handler(lambda:int(time.monotonic()>deadline),100000)
    tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    roots=[dict(r) for r in db.execute('''SELECT p.rowid AS api_rowid,CAST(p.startNs AS INTEGER) AS startNs,CAST(p.endNs AS INTEGER) AS endNs,p.globalTid,p.connectionId,p.callchainId,s.value AS name
     FROM PYTORCH_API p JOIN STRING_IDS s ON s.id=p.name WHERE s.value='prepare input' ORDER BY CAST(p.startNs AS INTEGER)''')]
    if spec['expected_roots'] is not None:
        assert [(r['startNs'],r['endNs']) for r in roots]==[(r['s_ns'],r['e_ns']) for r in spec['expected_roots']]
    api_columns={r['name'] for r in db.execute('PRAGMA table_info(PYTORCH_API)')}
    shape_expr='sh.value AS input_shapes,dt.value AS input_dtypes' if {'inputShapes','inputDtypes'}<=api_columns else 'NULL AS input_shapes,NULL AS input_dtypes'
    shape_join='LEFT JOIN STRING_IDS sh ON sh.id=p.inputShapes LEFT JOIN STRING_IDS dt ON dt.id=p.inputDtypes' if {'inputShapes','inputDtypes'}<=api_columns else ''
    records=[];root_summaries=[]
    for ordinal,root in enumerate(roots):
        rows=[dict(r) for r in db.execute('''SELECT p.rowid AS api_rowid,CAST(p.startNs AS INTEGER) AS startNs,CAST(p.endNs AS INTEGER) AS endNs,p.globalTid,p.connectionId,p.callchainId,s.value AS name,'''+shape_expr+'''
          FROM PYTORCH_API p JOIN STRING_IDS s ON s.id=p.name '''+shape_join+'''
          WHERE p.globalTid=? AND CAST(p.startNs AS INTEGER)>=? AND CAST(p.endNs AS INTEGER)<=? AND p.rowid!=? ORDER BY CAST(p.startNs AS INTEGER),CAST(p.endNs AS INTEGER) DESC''',(root['globalTid'],root['startNs'],root['endNs'],root['api_rowid']))]
        selected=[]
        for row in rows:
            n=row['name']
            if ('memcpy' in n.lower() or '_pin_memory' in n or n in ['Event::synchronize','aten::copy_','aten::_to_copy','aten::to']):
                row['prepare_ordinal']=ordinal;selected.append(row)
        records.extend(selected)
        root_summaries.append(dict(root_ordinal=ordinal,**root,all_API_descendants=len(rows),all_name_counts=dict(Counter(r['name'] for r in rows)),selected_records=len(selected),selected_name_counts=dict(Counter(r['name'] for r in selected)),selected_with_callchain=sum(r['callchainId'] is not None for r in selected)))
    chains={};chain_ids=sorted({r['callchainId'] for r in records if r['callchainId'] is not None})
    if 'PYTORCH_CALLCHAINS' in tables:
        for i in range(0,len(chain_ids),500):
            ids=chain_ids[i:i+500]
            for r in db.execute('SELECT c.id,c.stackDepth,s.value AS frame FROM PYTORCH_CALLCHAINS c JOIN STRING_IDS s ON s.id=c.stack WHERE c.id IN ('+','.join('?' for _ in ids)+') ORDER BY c.id,c.stackDepth',ids):
                chains.setdefault(str(r['id']),[]).append(dict(stackDepth=r['stackDepth'],frame=r['frame']))
    for r in records:
        fs=chains.get(str(r['callchainId']),[])
        r['project_frames']=[f for f in fs if '/vllm_ascend/' in f['frame'] or '/vllm/vllm/' in f['frame']]
    summary=dict(population=spec['population'],rank=spec['rank'],database=str(p),sha256=h.hexdigest(),roots=len(roots),selected_records=len(records),
      selected_name_counts=dict(Counter(r['name'] for r in records)),non_null_chain_ids=len(chain_ids),callchains_present=len(chains),
      records_with_project_frames=sum(bool(r['project_frames']) for r in records),has_callchains_table='PYTORCH_CALLCHAINS' in tables,
      source_location_counts=dict(Counter(f['frame'] for r in records for f in r['project_frames'])))
    results.append(dict(summary=summary,root_summaries=root_summaries,selected_records=records,full_selected_callchains=chains))
    db.close();after=p.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
payload=json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(),results=results,no_target_source_read=True,no_inference_NPU_or_service_mutation=True,
 limits=['Same-thread interval containment identifies recorded host API descendants, not tensor identity or field ownership.',
 'Official and earlier diagnostic observations are separate populations; callsites may explain a source contract but timing/counts/request identities are never joined across them.',
 'Missing callchain/shape/bytes/values are retained; no source or nearest-time repair.',
 'Nested API times are not summed as CPU work or latency gain.'],whole_modules_closed=0,overall_complete=False),ensure_ascii=False,separators=(',',':'))+'\n'
OUT.write_text(payload)
public=dict(full_result=str(OUT),result_sha256=hashlib.sha256(payload.encode()).hexdigest(),summaries=[r['summary'] for r in results])
assert len(json.dumps(public).encode())<30000
print(json.dumps(public,ensure_ascii=False))
