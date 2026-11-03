from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone
import hashlib, json, re, socket, sqlite3, time

assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72')
OUT=P/'analysis/prepare-field-origins-v12.json';assert not OUT.exists()
b=(P/'analysis/profile-capabilities-index-v12-v2.json').read_bytes()
assert hashlib.sha256(b).hexdigest()=='79ba556ef3855b9066392e438371a0a9d3b38ea7511269ce7ff2a6814529a095'
index=json.loads(b);assert len(index['databases'])==32
metadata=defaultdict(list)
for file in index['metadata']:
    for event in file['entries']:
        if event['event']=='execute_enter':metadata[event['label']].append(event)
chains_global={};roots_all=[];records_all=[];bindings=[]
selected_names=['acl_memcpy_host_to_device','aten::_pin_memory','Event::synchronize','aten::copy_','aten::_to_copy','aten::to']
t0=time.monotonic()
def file_sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
for di,spec in enumerate(index['databases']):
    assert time.monotonic()-t0<145,'finite query budget'
    p=Path(spec['path']);before=p.stat();assert file_sha(p)==spec['sha256']
    db=sqlite3.connect(p.as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row;db.execute('PRAGMA query_only=ON')
    db.set_progress_handler(lambda:int(time.monotonic()-t0>155),100000)
    markers=[s for s in spec['scopes'] if s['name'].startswith('wio_hostdiag::execute_model/')]
    roots=[s for s in spec['scopes'] if s['name']=='prepare input']
    for marker in markers:
        events=metadata.get(marker['name'],[])
        binding=dict(database_index=di,marker=marker,metadata_matches=len(events),event=events[0] if len(events)==1 else None)
        if len(events)==1:
            e=events[0];binding['same_pid_tid']=marker['globalTid']==e['pid']*2**32+e['tid']
        bindings.append(binding)
    local_records=[]
    for ri,root in enumerate(roots):
        parents=[m for m in markers if m['globalTid']==root['globalTid'] and m['startNs']<=root['startNs'] and root['endNs']<=m['endNs']]
        event=metadata.get(parents[0]['name'],[]) if len(parents)==1 else []
        entry=event[0] if len(event)==1 else None
        root_key=str(di)+':'+str(root['api_rowid'])
        rows=[dict(r) for r in db.execute('''SELECT p.rowid AS api_rowid,CAST(p.startNs AS INTEGER) AS startNs,CAST(p.endNs AS INTEGER) AS endNs,p.globalTid,p.connectionId,p.callchainId,s.value AS name,
            sh.value AS input_shapes,dt.value AS input_dtypes
            FROM PYTORCH_API p JOIN STRING_IDS s ON s.id=p.name
            LEFT JOIN STRING_IDS sh ON sh.id=p.inputShapes LEFT JOIN STRING_IDS dt ON dt.id=p.inputDtypes
            WHERE p.globalTid=? AND CAST(p.startNs AS INTEGER)>=? AND CAST(p.endNs AS INTEGER)<=? AND p.rowid!=?
            ORDER BY CAST(p.startNs AS INTEGER),CAST(p.endNs AS INTEGER) DESC''',(root['globalTid'],root['startNs'],root['endNs'],root['api_rowid']))]
        chosen=[r for r in rows if r['name'] in selected_names]
        for r in chosen:r.update(database_index=di,root_key=root_key)
        local_records.extend(chosen)
        computed=(entry or {}).get('cached_request_state',{}).get('num_computed_tokens')
        decoded=isinstance(computed,list) and bool(computed) and all(type(v)is int and v>=32768 for v in computed)
        roots_all.append(dict(root_key=root_key,database_index=di,root=root,containing_marker_count=len(parents),metadata_matches=len(event),
          metadata=entry,phase='CACHED_CONTEXT_GE32768' if decoded else 'OTHER_OR_NOT_FULLY_CLASSIFIED',
          all_API_descendants=len(rows),all_API_name_counts=dict(Counter(r['name'] for r in rows)),
          selected_name_counts=dict(Counter(r['name'] for r in chosen)),root_wall_ns=root['endNs']-root['startNs'],wall_is_not_CPU_active=True))
    ids=sorted({r['callchainId'] for r in local_records if r['callchainId'] is not None});local_chains=defaultdict(list)
    for start in range(0,len(ids),500):
        batch=ids[start:start+500]
        query='''SELECT c.id,c.stackDepth,s.value AS frame FROM PYTORCH_CALLCHAINS c JOIN STRING_IDS s ON s.id=c.stack WHERE c.id IN ('''+','.join('?' for _ in batch)+') ORDER BY c.id,c.stackDepth'
        for row in db.execute(query,batch):local_chains[row['id']].append((row['stackDepth'],row['frame']))
    chain_key={}
    for cid,frames in local_chains.items():
        payload=json.dumps(frames,ensure_ascii=False,separators=(',',':'))
        key=hashlib.sha256(payload.encode()).hexdigest();chain_key[cid]=key
        if key not in chains_global:chains_global[key]=[dict(stackDepth=d,frame=f) for d,f in frames]
    for r in local_records:
        r['chain_key']=chain_key.get(r['callchainId'])
        frames=chains_global.get(r['chain_key'],[])
        r['project_frames']=[f for f in frames if '/vllm_ascend/' in f['frame'] or '/vllm/vllm/' in f['frame']]
    records_all.extend(local_records)
    db.close();after=p.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
source_groups=defaultdict(list)
for r in records_all:
    leaf=r['project_frames'][0]['frame'] if r['project_frames'] else 'NO_PROJECT_FRAME'
    source_groups[(r['name'],leaf)].append(r)
groups=[]
for (name,leaf),rows in sorted(source_groups.items()):
    groups.append(dict(name=name,first_project_location=leaf,records=len(rows),roots=len({r['root_key'] for r in rows}),databases=len({r['database_index'] for r in rows}),
      chain_keys=sorted({r['chain_key'] for r in rows if r['chain_key']}),shape_dtype_counts=[dict(input_shapes=k[0],input_dtypes=k[1],records=v) for k,v in Counter((r['input_shapes'],r['input_dtypes']) for r in rows).items()],
      example_record_keys=[dict(database_index=r['database_index'],api_rowid=r['api_rowid'],root_key=r['root_key']) for r in rows[:2]]))
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),database_files=32,prepare_roots=len(roots_all),marker_bindings=len(bindings),
    exact_label_metadata_matches=sum(b['metadata_matches']==1 for b in bindings),exact_label_and_pid_tid_matches=sum(b.get('same_pid_tid',False) for b in bindings),
    prepare_roots_with_unique_same_thread_marker=sum(r['containing_marker_count']==1 and r['metadata_matches']==1 for r in roots_all),
    root_phase_counts=dict(Counter(r['phase'] for r in roots_all)),selected_records=len(records_all),selected_name_counts=dict(Counter(r['name'] for r in records_all)),
    selected_with_full_chains=sum(bool(r['chain_key']) for r in records_all),selected_with_project_frames=sum(bool(r['project_frames']) for r in records_all),
    canonical_full_chains=len(chains_global),source_groups=len(groups),name_source_group_counts=dict(Counter(g['name'] for g in groups)),
    per_root_copy_count_histograms={name:dict(Counter(r['selected_name_counts'].get(name,0) for r in roots_all)) for name in selected_names},
    active_method_counts=dict(Counter((r['metadata']or{}).get('speculative_method','UNKNOWN') for r in roots_all)),
    active_aux_flag_counts=dict(Counter(str((r['metadata']or{}).get('use_aux_hidden_state_outputs')) for r in roots_all)),
    no_target_source_read=True,no_new_device_or_service_work=True,whole_modules_closed=0,overall_complete=False,
    limits=['Same-thread nesting plus exact annotation labels/pid/tid binds recorded host work to a schedule,not client request identities or NPU cross-rank clock causality.',
            'Full callchains are deduplicated only by exact ordered(depth,frame)content. Every API keeps database/row/callchainID andcanonical chain key;no source truncation.',
            'First project frame is an indexing convention;field meaning must be reviewed using the full chain. Shapes/dtypes are logical inputs,not physical bytes/pointers/alias generations.',
            'Nested API walls/counts cannot be added as CPU work,removable time or gain;old169timing populations remain separate.'])
payload=json.dumps(dict(summary=summary,databases=index['databases'],bindings=bindings,prepare_roots=roots_all,selected_API_records=records_all,full_canonical_callchains=chains_global,source_groups=groups),ensure_ascii=False,separators=(',',':'))+'\n'
assert len(payload.encode())<200*1024**2,'derived artifact size guard'
OUT.write_text(payload)
public=dict(summary=summary,full_result=str(OUT),result_sha256=hashlib.sha256(payload.encode()).hexdigest(),source_group_digest=[dict(name=g['name'],first_project_location=g['first_project_location'],records=g['records'],roots=g['roots']) for g in groups])
if len(json.dumps(public).encode())>30000:public.pop('source_group_digest');public['display_truncated']=True
print(json.dumps(public,ensure_ascii=False))
