from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib, json, socket, sqlite3, time

assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72')
def pinned(name,digest):
    b=(P/'analysis'/name).read_bytes();assert hashlib.sha256(b).hexdigest()==digest;return json.loads(b)
x=pinned('prepare-field-origins-v12.json','c23b75f3fbd3ae3d3d7152f6882524e899bade53001ea2f1befa09ce943f091f')
u=pinned('saved-prepare-consumers-v13.json','6ba784cac70931d7764111832ba40fc8ca3927f80c58d63ab70341594fab62eb')
OUT=P/'analysis/event-resource-schema-and-field-shapes-v13.json';assert not OUT.exists()
schemas={};dbresults=[];chains={};t0=time.monotonic()
for di,spec in enumerate(x['databases']):
    db=sqlite3.connect(Path(spec['path']).as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row;db.execute('PRAGMA query_only=ON')
    db.set_progress_handler(lambda:int(time.monotonic()-t0>145),100000)
    tables=[r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
    relevant=[t for t in tables if any(k in t.upper() for k in ['METRIC','PMU','FREQ','CORE','EVENT','COMMUNICATION','HCCL'])]
    schema={t:[dict(r) for r in db.execute('PRAGMA table_info("'+t+'")')] for t in relevant}
    key=hashlib.sha256(json.dumps(schema,sort_keys=True).encode()).hexdigest()
    if key not in schemas:
        schemas[key]=dict(schemas=schema,first_database_index=di,
            examples={t:[dict(r) for r in db.execute('SELECT * FROM "'+t+'" LIMIT 3')] for t in relevant})
    counts={t:db.execute('SELECT COUNT(*) FROM "'+t+'"').fetchone()[0] for t in relevant}
    events=[dict(r) for r in db.execute('''SELECT p.rowid AS api_rowid,CAST(p.startNs AS INTEGER) AS startNs,CAST(p.endNs AS INTEGER) AS endNs,p.globalTid,p.connectionId,p.callchainId,s.value AS name
      FROM PYTORCH_API p JOIN STRING_IDS s ON s.id=p.name WHERE s.value LIKE 'Event::record%' OR s.value='Event::synchronize' ORDER BY CAST(p.startNs AS INTEGER)''')]
    ids=sorted({e['callchainId'] for e in events if e['callchainId']is not None});frames=defaultdict(list)
    for start in range(0,len(ids),500):
        batch=ids[start:start+500]
        for r in db.execute('SELECT c.id,c.stackDepth,s.value AS frame FROM PYTORCH_CALLCHAINS c JOIN STRING_IDS s ON s.id=c.stack WHERE c.id IN ('+','.join('?'for _ in batch)+') ORDER BY c.id,c.stackDepth',batch):frames[r['id']].append(dict(stackDepth=r['stackDepth'],frame=r['frame']))
    for e in events:
        fs=frames.get(e['callchainId'],[]);keychain=hashlib.sha256(json.dumps(fs,ensure_ascii=False,separators=(',',':')).encode()).hexdigest();chains[keychain]=fs;e['chain_key']=keychain
        e['project_frames']=[f['frame'] for f in fs if '/vllm_ascend/'in f['frame'] or '/vllm/vllm/'in f['frame']]
        e['observed_valid_count_record_site']=e['name'].startswith('Event::record') and any('model_runner_v1.py(1891)'in f for f in e['project_frames'])
    native_events=[dict(r) for r in db.execute('''SELECT s.value AS name,COUNT(*)AS records FROM CANN_API c JOIN STRING_IDS s ON s.id=c.name WHERE lower(s.value)LIKE '%recordevent%' OR lower(s.value)LIKE '%synchronizeevent%' GROUP BY s.value''')]
    db.close();dbresults.append(dict(database_index=di,path=spec['path'],database_sha256=spec['sha256'],resource_schema_key=key,resource_table_counts=counts,host_events=events,native_event_name_counts=native_events))
allrecords=x['selected_API_records'];byroot=defaultdict(list)
for r in allrecords:byroot[r['root_key']].append(r)
pairs=[];event_name_counts=Counter()
for root in x['prepare_roots']:
    rs=byroot[root['root_key']]
    early=[r for r in rs if r['name']=='Event::synchronize'and any('gpu_model_runner.py(4987)'in f['frame']for f in r['project_frames'])]
    late=[r for r in rs if r['name']=='Event::synchronize'and any('model_runner_v1.py(1867)'in f['frame']for f in r['project_frames'])]
    event_name_counts.update({k:v for k,v in root['all_API_name_counts'].items()if'Event::'in k})
    if len(early)==len(late)==1:
        writers=[e for e in dbresults[root['database_index']]['host_events']if e['observed_valid_count_record_site']and early[0]['endNs']<e['startNs']<late[0]['startNs']]
        pairs.append(dict(root_key=root['root_key'],early_api=early[0]['api_rowid'],late_api=late[0]['api_rowid'],early_before_late=early[0]['endNs']<=late[0]['startNs'],intervening_observed_valid_count_writer_rows=[e['api_rowid']for e in writers]))
pad=[]
for group in u['utility_copy_callers']:
    if any('model_runner_v1.py(1105)'in f for f in group['project_frames']):
        for c in group['H2D_parent_correspondences']:
            root=next(r for r in x['prepare_roots']if r['root_key']==c['root_key']);meta=root['metadata']
            pin=[r for r in byroot[c['root_key']]if r['name']=='aten::_pin_memory'and any('dsa_v41_cp.py(131)'in f['frame']for f in r['project_frames'])]
            pad.append(dict(root_key=c['root_key'],scheduled_requests=len(meta['scheduled_tokens']),total_scheduled_tokens=meta['total_scheduled_tokens'],CP_qsl_pin_shapes=sorted({r['input_shapes']for r in pin}),parent_shapes=c['parents']))
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),databases=32,resource_schema_variants=len(schemas),
    resource_tables=sorted({t for d in dbresults for t in d['resource_table_counts']}),host_event_name_counts=dict(Counter(e['name']for d in dbresults for e in d['host_events'])),
    observed_valid_count_record_events=sum(e['observed_valid_count_record_site']for d in dbresults for e in d['host_events']),
    prepare_event_name_counts=dict(event_name_counts),paired_valid_count_sync_roots=len(pairs),all_pairs_early_before_late=all(p['early_before_late']for p in pairs),
    pairs_with_no_intervening_observed_valid_count_record=sum(not p['intervening_observed_valid_count_writer_rows']for p in pairs),
    QSL_padding_publication_records=len(pad),QSL_padding_schedule_shape_hist={str(k):v for k,v in Counter((r['scheduled_requests'],r['total_scheduled_tokens'],tuple(r['CP_qsl_pin_shapes']))for r in pad).items()},
    no_target_source_read=True,no_device_or_service_mutation=True,whole_modules_closed=0,overall_complete=False,
    limits=['Absence ofobservedEvent::record is not generation proof unlessrecorder/source coverage isestablished. Preserve fullnames/counts andnative APIrecords.',
            'Resource schema/sample values are notcounter definitions ormodule utilization;needunit/averaging/core/frequency semantics before normalization.',
            'Matching CPUQSL shape/count doesnotprove unchangedGPUbuffer:sourcehas aninterveningGPUtailfill at1462. Do notdropfullsecondcopy byshape alone.'])
payload=json.dumps(dict(summary=summary,resource_schema_variants=schemas,databases=dbresults,full_canonical_event_chains=chains,valid_count_sync_pairs=pairs,QSL_padding_publications=pad,
    utility_shapes_and_chains=[{k:v for k,v in g.items()if k!='H2D_parent_correspondences'}for g in u['utility_copy_callers']]),ensure_ascii=False,separators=(',',':'))+'\n'
OUT.write_text(payload)
public=dict(summary=summary,full_result=str(OUT),result_sha256=hashlib.sha256(payload.encode()).hexdigest(),schema_examples=schemas)
if len(json.dumps(public).encode())>30000:public.pop('schema_examples');public['display_truncated']=True
print(json.dumps(public,ensure_ascii=False))
