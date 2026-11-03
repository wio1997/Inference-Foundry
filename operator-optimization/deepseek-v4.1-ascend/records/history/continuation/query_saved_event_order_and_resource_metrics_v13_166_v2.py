from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import hashlib, json, socket, sqlite3, statistics, time

assert socket.gethostname() == 'bm-jn-zs-zone1-910C-64G-10-105'
P = Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72')
def pinned(name, digest):
    b = (P/'analysis'/name).read_bytes()
    assert hashlib.sha256(b).hexdigest() == digest
    return json.loads(b)
x = pinned('prepare-field-origins-v12.json', 'c23b75f3fbd3ae3d3d7152f6882524e899bade53001ea2f1befa09ce943f091f')
e = pinned('event-resource-schema-and-field-shapes-v13.json', 'aab4ca0839c2ab1085a9a1d5a9c554ebf2d60f4074cfb41b0ce353d310e2efc5')
c = pinned('CP-dead-producer-device-tasks-v13-v2.json', '6b4129c1e711a369b6ca4ce9f3bd8d304d57ffe72ca19db7a7782fff3052fdf0')
OUT = P/'analysis/saved-event-order-and-resource-metrics-v13-v2.json'
assert not OUT.exists()
pairs = []
for r in x['prepare_roots']:
    matches = [a for a in x['selected_API_records'] if a['root_key'] == r['root_key'] and a['name'] == 'Event::synchronize' and any('gpu_model_runner.py(4987)' in f['frame'] or 'model_runner_v1.py(1867)' in f['frame'] for f in a['project_frames'])]
    if len(matches) != 2:
        continue
    first, second = sorted(matches, key=lambda a:a['startNs'])
    writers = [a for a in e['databases'][r['database_index']]['host_events'] if a['observed_valid_count_record_site'] and first['endNs'] < a['startNs'] < second['startNs']]
    pairs.append(dict(root_key=r['root_key'], database_index=r['database_index'], first_API=first, second_API=second, nonoverlapping_order=first['endNs'] <= second['startNs'], same_host_thread=first['globalTid'] == second['globalTid'], observed_valid_count_writer_rows=[a['api_rowid'] for a in writers], gap_ns=second['startNs']-first['endNs']))
t0 = time.monotonic()
dbstats = []
for di, spec in enumerate(x['databases']):
    db = sqlite3.connect(Path(spec['path']).as_uri()+'?mode=ro&immutable=1', uri=True)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA query_only=ON')
    db.set_progress_handler(lambda:int(time.monotonic()-t0 > 145), 100000)
    metrics = [dict(r) for r in db.execute('''SELECT s.value AS metric_name,p.name AS metric_name_id,COUNT(*) AS records,COUNT(DISTINCT p.globalTaskId) AS unique_global_task_ids,MIN(p.value) AS minimum,MAX(p.value) AS maximum,SUM(CASE WHEN p.value IS NULL THEN 1 ELSE 0 END) AS null_values,SUM(CASE WHEN p.value != 0 THEN 1 ELSE 0 END) AS nonzero_values FROM TASK_PMU_INFO p LEFT JOIN STRING_IDS s ON s.id=p.name GROUP BY p.name,s.value''')]
    frequency = [dict(r) for r in db.execute('SELECT deviceId,dieId,freq,COUNT(*) AS records,MIN(timestampNs) AS first_timestampNs,MAX(timestampNs) AS last_timestampNs FROM AICORE_FREQ GROUP BY deviceId,dieId,freq')]
    pmu_identity = dict(db.execute('''SELECT COUNT(*) AS records,COUNT(DISTINCT globalTaskId) AS unique_global_task_ids FROM TASK_PMU_INFO''').fetchone())
    pmu_identity['records_with_exact_TASK_global_id'] = db.execute('SELECT COUNT(*) FROM TASK_PMU_INFO p WHERE p.globalTaskId IN (SELECT t.globalTaskId FROM TASK t)').fetchone()[0]
    pmu_identity['duplicate_task_metric_pairs'] = db.execute('SELECT COUNT(*) FROM (SELECT globalTaskId,name,COUNT(*) AS n FROM TASK_PMU_INFO GROUP BY globalTaskId,name HAVING n>1)').fetchone()[0]
    pmu_identity['task_global_id_duplicate_groups'] = db.execute('SELECT COUNT(*) FROM (SELECT globalTaskId,COUNT(*) AS n FROM TASK GROUP BY globalTaskId HAVING n>1)').fetchone()[0]
    compute = [dict(r) for r in db.execute('''SELECT s.value AS op_type,c.blockNum,c.mixBlockNum,COUNT(*) AS compute_info_rows FROM COMPUTE_TASK_INFO c LEFT JOIN STRING_IDS s ON s.id=c.opType GROUP BY s.value,c.blockNum,c.mixBlockNum ORDER BY s.value,c.blockNum,c.mixBlockNum''')]
    dbstats.append(dict(database_index=di,path=spec['path'],database_sha256=spec['sha256'],PMU_metrics=metrics,PMU_identity=pmu_identity,frequency_samples=frequency,compute_op_block_census=compute,resource_table_counts=e['databases'][di]['resource_table_counts']))
    db.close()
def extent(v):
    return dict(minimum=min(v),median=statistics.median(v),maximum=max(v),samples=len(v)) if v else None
metric_counts=Counter()
metric_ranges={}
for d in dbstats:
    for m in d['PMU_metrics']:
        metric_counts[m['metric_name']] += m['records']
        metric_ranges.setdefault(m['metric_name'],[]).append({k:m[k] for k in ['minimum','maximum','nonzero_values','records']})
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),databases=len(dbstats),corrected_valid_count_sync_pairs=len(pairs),
    actual_first_source_counts=dict(Counter(p['first_API']['project_frames'][0]['frame'] for p in pairs)),all_pairs_same_thread=all(p['same_host_thread'] for p in pairs),all_pairs_nonoverlapping=all(p['nonoverlapping_order'] for p in pairs),
    pairs_with_observed_valid_count_writer_between=sum(bool(p['observed_valid_count_writer_rows']) for p in pairs),pair_gap_ns=extent([p['gap_ns'] for p in pairs]),
    PMU_metric_record_counts=dict(metric_counts),frequency_value_counts=dict(Counter(str(f['freq']) for d in dbstats for f in d['frequency_samples'])),
    CP_dead_value_kernel_union_ns_per_root=extent([r['kernel_interval_union_ns'] for r in c['per_root_device_work']]),CP_dead_value_kernel_union_ns_per_execution=extent([t['endNs']-t['startNs'] for t in c['compute_executions'] if t['dead_producer_task']]),
    no_target_source_read=True,no_device_or_service_work=True,whole_modules_closed=0,overall_complete=False,
    corrections=['First resource query stopped at its 145s SQLite progress bound while evaluating a correlated TASK membership query. v2 uses equivalent set membership without changing identities; original interrupted receipt retained.', 'Supersedes reversed early/late ordering and invalid zero-writer conclusion in event-resource-schema-and-field-shapes-v13.json; original result retained.'],
    limits=['No event handle/generation identity asserted by timestamp order or absence of observed writer. Source/DCP recorder coverage still required for wait removal.',
            'PMU names/ranges and sampled frequency raw values are inventory only; no units/averaging/MFU claim.',
            'Profiled kernel duration and interval union are not request latency gains or unprofiled costs.'])
payload=json.dumps(dict(summary=summary,corrected_sync_pairs=pairs,databases=dbstats,PMU_metric_ranges=metric_ranges),ensure_ascii=False,separators=(',',':'))+'\n'
OUT.write_text(payload)
print(json.dumps(dict(summary=summary,first_database_metrics=dbstats[0]['PMU_metrics'],full_result=str(OUT),result_sha256=hashlib.sha256(payload.encode()).hexdigest()),ensure_ascii=False))
