from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone
import hashlib,json,sqlite3,socket,time

assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72')
def pinned(name,digest):
    b=(P/'analysis'/name).read_bytes();assert hashlib.sha256(b).hexdigest()==digest;return json.loads(b)
x=pinned('prepare-field-origins-v12.json','c23b75f3fbd3ae3d3d7152f6882524e899bade53001ea2f1befa09ce943f091f')
phase=pinned('prepare-phase-and-schedule-cohorts-v12.json','55ab5a2ebb4775f0af61039b2c2ba6e82457f97dcec79fc0033b30c733f77563')
ROOT=P/'analysis/native-execution-identity-v14';ROOT.mkdir(exist_ok=True)
OUT=ROOT/'index.json';assert not OUT.exists()
t0=time.monotonic();results=[]
phase_by_root={r['root_key']:r for r in phase['roots']}
metrics=['aic_total_cycles','aic_mac_ratio','aiv_total_cycles','aiv_vec_ratio']
def emit(path,obj):
    b=(json.dumps(obj,ensure_ascii=False,separators=(',',':'))+'\n').encode();path.write_bytes(b)
    return dict(path=str(path),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
for di,spec in enumerate(x['databases']):
    dest=ROOT/('db-%02d.json'%di)
    if dest.exists():
        b=dest.read_bytes();saved=json.loads(b);assert saved['database_sha256']==spec['sha256']
        results.append(dict(database_index=di,summary=saved['summary'],path=str(dest),bytes=len(b),sha256=hashlib.sha256(b).hexdigest()));continue
    assert time.monotonic()-t0<145,'bounded identity census'
    path=Path(spec['path']);before=path.stat()
    assert hashlib.sha256(path.read_bytes()).hexdigest()==spec['sha256']
    db=sqlite3.connect(path.as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row;db.execute('PRAGMA query_only=ON')
    db.set_progress_handler(lambda:int(time.monotonic()-t0>150),100000)
    strings=dict(db.execute('SELECT id,value FROM STRING_IDS'))
    tables=[r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
    selected=[t for t in tables if any(s in t for s in ['MODEL','GRAPH','MEMORY_COPY','MSTX','COMMUNICATION'])]+['TASK','COMPUTE_TASK_INFO','CANN_API','TASK_PMU_INFO']
    schemas={t:[dict(r) for r in db.execute('PRAGMA table_info("'+t+'")')] for t in dict.fromkeys(selected)}
    tasks=[dict(r) for r in db.execute('SELECT rowid AS task_record_rowid,* FROM TASK')]
    assert len(tasks)<200000
    by_gid=defaultdict(list)
    for t in tasks:
        t['task_type_name']=strings.get(t['taskType']);by_gid[t['globalTaskId']].append(t)
    definitions=[dict(r) for r in db.execute('SELECT rowid AS compute_record_rowid,* FROM COMPUTE_TASK_INFO')]
    defs_by_gid=defaultdict(list)
    for c in definitions:
        for col in ['name','opType','inputShapes','inputDataTypes','outputShapes','outputDataTypes']:c[col+'_text']=strings.get(c.get(col))
        defs_by_gid[c['globalTaskId']].append(c)
    compute=[t for t in tasks if t['globalTaskId'] in defs_by_gid]
    connections=[dict(r) for r in db.execute('SELECT id,connectionId FROM CONNECTION_IDS')]
    dev_to_host=defaultdict(set)
    for c in connections:dev_to_host[c['connectionId']].add(c['id'])
    apis=[dict(r) for r in db.execute('SELECT rowid AS api_rowid,CAST(startNs AS INTEGER) AS startNs,CAST(endNs AS INTEGER) AS endNs,globalTid,connectionId,callchainId,name FROM PYTORCH_API WHERE connectionId IS NOT NULL')]
    by_host=defaultdict(list)
    for a in apis:a['name_text']=strings.get(a['name']);by_host[a['connectionId']].append(a['api_rowid'])
    for t in compute:t['native_linked_API_rowids']=sorted({a for h in dev_to_host.get(t['connectionId'],[]) for a in by_host.get(h,[])})
    graphnames={i for i,s in strings.items() if 'aclmdl' in s or 'Graph' in s or 'Capture' in s}
    cann=[dict(r) for r in db.execute('SELECT rowid AS cann_api_rowid,* FROM CANN_API') if r['name'] in graphnames]
    for a in cann:a['name_text']=strings.get(a['name'])
    execute=[a for a in cann if a['name_text']=='aclmdlRIExecuteAsync']
    controls=[t for t in tasks if t['task_type_name'] in ['MODEL_EXECUTE','MODEL_WAIT_COMPLETE']]
    graph_links=[]
    for a in execute:
        ids={a['connectionId']}|{r['connectionId'] for r in connections if r['id']==a['connectionId']}
        ms=[m for m in controls if m['connectionId'] in ids]
        runs=[m for m in ms if m['task_type_name']=='MODEL_EXECUTE'];waits=[m for m in ms if m['task_type_name']=='MODEL_WAIT_COMPLETE']
        markers=[b for b in x['bindings'] if b['database_index']==di and b.get('same_pid_tid') and b['marker']['globalTid']==a['globalTid'] and b['marker']['startNs']<=a['startNs'] and a['endNs']<=b['marker']['endNs']]
        rootkeys=[r['root_key'] for r in x['prepare_roots'] if r['database_index']==di and any(r['metadata']['label']==b['event']['label'] for b in markers)]
        row=dict(CANN_API=a,device_connection_candidates=sorted(ids),control_task_rowids=[m['task_record_rowid'] for m in ms],same_thread_markers=markers,prepare_root_keys=rootkeys,phase_records=[phase_by_root[k] for k in rootkeys],control_pair_valid=False)
        if len(runs)==len(waits)==1 and runs[0]['modelId']==waits[0]['modelId']:
            m,w=runs[0],waits[0];lo=min(m['startNs'],w['startNs']);hi=max(m['endNs'],w['endNs'])
            ts=[t for t in compute if t['modelId']==m['modelId'] and lo<=t['startNs'] and t['endNs']<=hi]
            row.update(control_pair_valid=True,model_id=m['modelId'],control_pair_start_ns=lo,control_pair_end_ns=hi,compute_task_rowids=[t['task_record_rowid'] for t in ts],op_type_counts=dict(Counter(defs_by_gid[t['globalTaskId']][0]['opType_text'] for t in ts if len(defs_by_gid[t['globalTaskId']])==1)))
        graph_links.append(row)
    pmu=defaultdict(list)
    names={i for i,s in strings.items() if s in metrics}
    for p in db.execute('SELECT rowid AS pmu_record_rowid,* FROM TASK_PMU_INFO'):
        if p['name'] in names:
            q=dict(p);q['metric_name']=strings[p['name']];pmu[(p['globalTaskId'],p['timestampNs'])].append(q)
    pmu_groups=[]
    for (gid,stamp),rows in pmu.items():
        ts=by_gid.get(gid,[]);start=[t['task_record_rowid'] for t in ts if t['startNs']==stamp];end=[t['task_record_rowid'] for t in ts if t['endNs']==stamp]
        contained=[t['task_record_rowid'] for t in ts if t['startNs']<=stamp<=t['endNs']]
        pmu_groups.append(dict(globalTaskId=gid,timestampNs=stamp,metric_rows=rows,exact_TASK_start_rows=start,exact_TASK_end_rows=end,containing_TASK_rows=contained,same_global_id_TASK_rows=[t['task_record_rowid'] for t in ts]))
    annotations=[dict(r) for r in db.execute('SELECT p.type,s.value AS name,COUNT(*) AS records FROM PYTORCH_API p JOIN STRING_IDS s ON s.id=p.name GROUP BY p.type,p.name')]
    graph_assignments=Counter(t for g in graph_links if g['control_pair_valid'] for t in g['compute_task_rowids'])
    summary=dict(database_index=di,TASK_records=len(tasks),COMPUTE_definitions=len(definitions),compute_executions=len(compute),compute_definition_multiplicity=dict(Counter(len(v) for v in defs_by_gid.values())),execution_reuse_hist=dict(Counter(len(by_gid[g]) for g in defs_by_gid)),task_type_counts=dict(Counter(t['task_type_name'] for t in tasks)),op_type_counts=dict(Counter(defs_by_gid[t['globalTaskId']][0]['opType_text'] for t in compute if len(defs_by_gid[t['globalTaskId']])==1)),
      compute_with_native_API_links=sum(bool(t['native_linked_API_rowids']) for t in compute),graph_execute_APIs=len(execute),valid_graph_control_pairs=sum(g['control_pair_valid'] for g in graph_links),graph_API_with_unique_same_call_marker=sum(len(g['same_thread_markers'])==1 for g in graph_links),graph_contained_compute_executions=len(graph_assignments),graph_compute_multiple_containers=sum(n>1 for n in graph_assignments.values()),PMU_groups=len(pmu_groups),PMU_exact_start_candidate_hist=dict(Counter(len(p['exact_TASK_start_rows']) for p in pmu_groups)),PMU_exact_end_candidate_hist=dict(Counter(len(p['exact_TASK_end_rows']) for p in pmu_groups)),PMU_containment_candidate_hist=dict(Counter(len(p['containing_TASK_rows']) for p in pmu_groups)),PMU_metric_count_hist=dict(Counter(len(p['metric_rows']) for p in pmu_groups)))
    db.close();after=path.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    obj=dict(database_index=di,database_path=spec['path'],database_sha256=spec['sha256'],summary=summary,schemas=schemas,TASK_records=tasks,COMPUTE_definitions=definitions,CONNECTION_IDS=connections,linked_PYTORCH_APIs=apis,graph_CANN_APIs=cann,graph_API_control_bindings=graph_links,PMU_execution_identity_candidates=pmu_groups,annotation_name_type_census=annotations)
    ref=emit(dest,obj);results.append(dict(database_index=di,summary=summary,**ref))
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),databases=len(results),TASK_records=sum(r['summary']['TASK_records'] for r in results),compute_executions=sum(r['summary']['compute_executions'] for r in results),graph_execute_APIs=sum(r['summary']['graph_execute_APIs'] for r in results),valid_graph_control_pairs=sum(r['summary']['valid_graph_control_pairs'] for r in results),graph_API_with_unique_same_call_marker=sum(r['summary']['graph_API_with_unique_same_call_marker'] for r in results),graph_contained_compute_executions=sum(r['summary']['graph_contained_compute_executions'] for r in results),graph_compute_multiple_containers=sum(r['summary']['graph_compute_multiple_containers'] for r in results),PMU_groups=sum(r['summary']['PMU_groups'] for r in results),PMU_exact_start_candidate_hist=dict(sum((Counter(r['summary']['PMU_exact_start_candidate_hist']) for r in results),Counter())),PMU_exact_end_candidate_hist=dict(sum((Counter(r['summary']['PMU_exact_end_candidate_hist']) for r in results),Counter())),PMU_containment_candidate_hist=dict(sum((Counter(r['summary']['PMU_containment_candidate_hist']) for r in results),Counter())),no_target_source_read=True,no_device_or_service_work=True,whole_modules_closed=0,overall_complete=False,
    limits=['TASK rowid is execution identity; globalTaskId is retained as shared definition identity. All multiplicities remain explicit.', 'Graph native connection/control pair containment is kept separate from exact same-thread host-marker/sidecar schedule association. No timestamp/ordinal joins across ranks.', 'PMU exact timestamp or containment candidates are diagnostic relationships, not accepted identity/normalization until uniqueness and exporter semantics are reviewed.', 'All computation/communication/task types are inventoried without inferring module ownership from op name alone.'])
ref=emit(OUT,dict(summary=summary,databases=results));print(json.dumps(dict(summary=summary,first_database=results[0]['summary'],full_result=ref['path'],result_sha256=ref['sha256'],result_bytes=ref['bytes']),ensure_ascii=False))
