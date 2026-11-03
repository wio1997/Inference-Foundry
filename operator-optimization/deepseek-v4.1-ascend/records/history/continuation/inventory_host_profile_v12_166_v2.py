from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import json, hashlib, sqlite3, socket, time

assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
ROOT=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030')
RUN='hostprof-20261030T212812Z-360e0c72'
P=ROOT/RUN
finished=json.loads((P/'controller-finished.json').read_text())
assert finished['state']=='CAPTURED_AND_RESTORED',finished
restored=json.loads((P/'restore-result.json').read_text())
assert restored['status']=='PASS_ORIGINAL_BASELINE_RESTORED'
OUT=P/'analysis/profile-capabilities-index-v12-v2.json'
assert not OUT.exists();OUT.parent.mkdir(exist_ok=True)
t0=time.monotonic()
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def flags(value,prefix=''):
    out={}
    if isinstance(value,dict):
        for k,v in value.items():
            p=prefix+'.'+k if prefix else k
            if k in ['record_shapes','with_stack','with_modules','profile_memory','rank_id','cann_version','torch_npu_version']:out[p]=v
            out.update(flags(v,p))
    elif isinstance(value,list):
        for i,v in enumerate(value):out.update(flags(v,prefix+'['+str(i)+']'))
    return out
infos=[]
for p in sorted((P/'runtime/official-traces').rglob('profiler_info*.json')):
    assert p.stat().st_size<2*1024**2
    data=json.loads(p.read_text());infos.append(dict(path=str(p),sha256=sha(p),flags=flags(data)))
metadata=[];event_counts=Counter()
for p in sorted((P/'runtime/host-profile').glob('events-*.jsonl')):
    entries=[]
    with p.open() as f:
        for line in f:
            d=json.loads(line);event_counts[d['event']]+=1;entries.append(d)
    metadata.append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size,entries=entries))
databases=[];schemas={}
assert json.loads((P/'analysis/offline-db-v12/database-index.json').read_text())['original_raw_unchanged']
for p in sorted((P/'analysis/offline-db-v12/profiles').rglob('ascend_pytorch_profiler_*.db')):
    assert time.monotonic()-t0<140,'finite inventory time guard'
    before=p.stat();digest=sha(p)
    db=sqlite3.connect(p.as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row
    db.execute('PRAGMA query_only=ON')
    db.set_progress_handler(lambda:int(time.monotonic()-t0>145),100000)
    tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    selected=sorted(t for t in tables if t in ['PYTORCH_API','PYTORCH_CALLCHAINS','CANN_API','CONNECTION_IDS','TASK','COMPUTE_TASK_INFO'])
    counts={t:db.execute('SELECT COUNT(*) FROM "'+t+'"').fetchone()[0] for t in selected}
    schema={t:[dict(r) for r in db.execute('PRAGMA table_info("'+t+'")')] for t in selected}
    schema_key=hashlib.sha256(json.dumps(schema,sort_keys=True).encode()).hexdigest();schemas[schema_key]=schema
    scopes=[dict(r) for r in db.execute('''SELECT p.rowid AS api_rowid,CAST(p.startNs AS INTEGER) AS startNs,CAST(p.endNs AS INTEGER) AS endNs,p.globalTid,p.connectionId,p.callchainId,s.value AS name
        FROM PYTORCH_API p JOIN STRING_IDS s ON s.id=p.name
        WHERE s.value='prepare input' OR s.value LIKE 'wio_hostdiag::execute_model/%' ORDER BY CAST(p.startNs AS INTEGER)''')]
    copy_counts=[dict(r) for r in db.execute('''SELECT s.value AS name,COUNT(*) AS records,SUM(p.callchainId IS NOT NULL) AS nonnull_callchain_records,SUM(p.inputShapes IS NOT NULL) AS nonnull_input_shape_records
        FROM PYTORCH_API p JOIN STRING_IDS s ON s.id=p.name
        WHERE s.value IN ('acl_memcpy_host_to_device','aten::_pin_memory','Event::synchronize','aten::copy_','aten::_to_copy','aten::to') GROUP BY s.value''')]
    db.close();after=p.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    databases.append(dict(path=str(p),sha256=digest,bytes=before.st_size,schema_key=schema_key,table_counts=counts,
        prepare_roots=sum(s['name']=='prepare input' for s in scopes),same_call_markers=sum(s['name'].startswith('wio_hostdiag::') for s in scopes),
        scopes=scopes,all_window_copy_pin_sync_API_counts=copy_counts,stat_unchanged=True))
clients={phase:json.loads((P/'runtime'/('client-'+phase)/'window-status.json').read_text()) for phase in ['warmup','baseline','profile']}
assert all(c['valid'] for c in clients.values())
assert clients['baseline']['input_sha256']==clients['profile']['input_sha256']
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),run_id=RUN,controller_finished=finished,restore=restored,
    profiler_info_files=len(infos),database_files=len(databases),schema_variants=len(schemas),
    prepare_roots=sum(d['prepare_roots'] for d in databases),same_call_markers=sum(d['same_call_markers'] for d in databases),
    databases_with_callchains=sum('PYTORCH_CALLCHAINS' in d['table_counts'] for d in databases),
    metadata_event_counts=dict(event_counts),metadata_files=len(metadata),all_case_counters_valid=True,
    profile_windows=clients['profile']['windows'],input_sha256=clients['profile']['input_sha256'],whole_modules_closed=0,overall_complete=False,
    limits=['Inventory/callchain and marker counts do not yet prove field or numerical/alias/event legality.',
            'Profiled timing and profiler-off baseline remain separate;metadata wrapper overhead is present in both.',
            'New166population is never joined by timestamps/ordinals to old169captures.'])
full=dict(summary=summary,profiler_infos=infos,schemas=schemas,databases=databases,metadata=metadata,clients=clients)
payload=json.dumps(full,ensure_ascii=False,separators=(',',':'))+'\n';OUT.write_text(payload)
public=dict(summary=summary,profiler_info_flags=[dict(path=i['path'],flags=i['flags']) for i in infos],
    database_summary=[{k:v for k,v in d.items() if k not in ['scopes','all_window_copy_pin_sync_API_counts']} for d in databases],
    schema_examples=schemas,first_copy_API_counts=databases[0]['all_window_copy_pin_sync_API_counts'] if databases else [],full_result=str(OUT),result_sha256=hashlib.sha256(payload.encode()).hexdigest())
if len(json.dumps(public).encode())>30000:
    public.pop('profiler_info_flags');public.pop('schema_examples');public['display_truncated']=True
print(json.dumps(public,ensure_ascii=False))
