#!/usr/bin/env python3
"""Exact-marker, bounded candidate queue audit on single-rank offline DB."""
import json
from pathlib import Path
import sqlite3

root = Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029')
job = root / 'binary-binding-draft/second-binding-run/offline-ep8-20261029'
status = json.loads((job/'status.json').read_text())
db = job / next(x['path'] for x in status['dbs'] if x['path'].endswith('/ascend_pytorch_profiler_8.db'))
assert not Path(str(db)+'-wal').exists() and not Path(str(db)+'-journal').exists()
c = sqlite3.connect(db.resolve().as_uri()+'?mode=ro&immutable=1',uri=True)
c.execute('PRAGMA query_only=ON')
c.row_factory=sqlite3.Row


def rows(sql, args=(), cap=5000):
    found=[dict(x) for x in c.execute(sql,args).fetchmany(cap+1)]
    if len(found)>cap:
        raise RuntimeError('bounded query truncated')
    return found


sidecar=root/'binding-diagnostic-selective2/runtime/metadata/events-746.jsonl'
events=[json.loads(line) for line in sidecar.read_text().splitlines()]
begins=[e for e in events if e.get('kind')=='call_begin' and e.get('label')=='DeepseekV41DecoderLayer.forward' and e.get('layer_index')==14]
ends={e['call_id'] for e in events if e.get('kind')=='call_end'}
results=[]
for event in begins:
    if event['call_id'] not in ends:
        continue
    marker='wio_diag:'+event['label']+':'+event['call_id']
    marker_rows=rows('''SELECT a.startNs,a.endNs,a.globalTid,a.connectionId,a.type
                        FROM PYTORCH_API a JOIN STRING_IDS s ON s.id=a.name
                        WHERE s.value=?''',(marker,))
    record={'call_id':event['call_id'],'layer_index':14,'marker_rows':marker_rows,'operations':[]}
    if len(marker_rows)!=1:
        record['issue']='marker missing or nonunique'
        results.append(record)
        continue
    m=marker_rows[0]
    ops=rows('''SELECT a.startNs,a.endNs,a.globalTid,a.connectionId,a.type,s.value AS name
                FROM PYTORCH_API a JOIN STRING_IDS s ON s.id=a.name
                WHERE a.type=50001 AND a.globalTid=? AND CAST(a.startNs AS INTEGER)>=? AND CAST(a.endNs AS INTEGER)<=?
                AND LOWER(s.value) LIKE '%dispatch_ffn_combine%' ''',(m['globalTid'],int(m['startNs']),int(m['endNs'])))
    for op in ops:
        oprec={'operation':op,'queues':[]}
        enqueues=rows('''SELECT a.startNs,a.endNs,a.globalTid,a.connectionId,a.type,s.value AS name
                         FROM PYTORCH_API a JOIN STRING_IDS s ON s.id=a.name
                         WHERE a.type=50002 AND a.globalTid=? AND CAST(a.startNs AS INTEGER)>=? AND CAST(a.endNs AS INTEGER)<=?
                         AND s.value LIKE 'Enqueue@%' ''',(op['globalTid'],int(op['startNs']),int(op['endNs'])))
        for enq in enqueues:
            rec={'enqueue':enq,'correlations':[]}
            if enq['connectionId'] is None:
                oprec['queues'].append(rec)
                continue
            corrs=rows('SELECT connectionId FROM CONNECTION_IDS WHERE id=?',(enq['connectionId'],))
            for corr in corrs:
                q=rows('''SELECT a.startNs,a.endNs,a.globalTid,a.connectionId,a.type,s.value AS name
                          FROM CONNECTION_IDS ids JOIN PYTORCH_API a ON a.connectionId=ids.id
                          JOIN STRING_IDS s ON s.id=a.name
                          WHERE ids.connectionId=? AND a.type=50002 AND s.value LIKE 'Dequeue@%' ''',(corr['connectionId'],))
                workers=[]
                for deq in q:
                    # Same queued function name; do not use temporal proximity to choose among matches.
                    if deq['name'].removeprefix('Dequeue@') != enq['name'].removeprefix('Enqueue@'):
                        continue
                    cann=rows('''SELECT a.startNs,a.endNs,a.globalTid,a.connectionId,a.type,s.value AS name
                                 FROM CANN_API a LEFT JOIN STRING_IDS s ON s.id=a.name
                                 WHERE a.globalTid=? AND CAST(a.startNs AS INTEGER)>=? AND CAST(a.endNs AS INTEGER)<=?''',(deq['globalTid'],int(deq['startNs']),int(deq['endNs'])),cap=20000)
                    launch=[]
                    cann_all=[]
                    for api in cann:
                        tasks=rows('SELECT startNs,endNs,deviceId,connectionId,globalTaskId,streamId,taskId,modelId FROM TASK WHERE connectionId=?',(api['connectionId'],),cap=20000) if api['connectionId'] is not None else []
                        for task in tasks:
                            task['compute_info']=rows('''SELECT i.globalTaskId,s.value AS op_type
                                FROM COMPUTE_TASK_INFO i LEFT JOIN STRING_IDS s ON s.id=i.opType
                                WHERE i.globalTaskId=?''',(task['globalTaskId'],))
                            task['exact_task_key_matches']=len(rows('''SELECT globalTaskId FROM TASK
                                WHERE globalTaskId=? AND startNs=? AND endNs=? AND streamId=? AND taskId=? AND modelId=?''',
                                (task['globalTaskId'],task['startNs'],task['endNs'],task['streamId'],task['taskId'],task['modelId'])))
                        cann_all.append({'cann_api':api,'task_rows':tasks})
                        if api['name'] and ('LaunchKernel' in api['name'] or 'dispatch' in api['name'].lower()):
                            launch.append({'cann_api':api,'task_rows':tasks})
                    workers.append({'dequeue':deq,'cann_total':len(cann),'cann_all':cann_all,'selected_launch_and_tasks':launch})
                rec['correlations'].append({'correlation':corr['connectionId'],'matching_dequeues':workers})
            oprec['queues'].append(rec)
        record['operations'].append(oprec)
    results.append(record)
c.close()
output={'source_db_sha256':next(x['sha256'] for x in status['dbs'] if x['path'].endswith('/ascend_pytorch_profiler_8.db')),
        'scope':'EP8 layer14 sidecar call_ids; exact marker; DB internal queue correlation and containment',
        'layer14_complete_sidecar_begins':len(begins),'results':results,
        'limits':['Profiler logged SQLite I/O error: DB integrity does not prove capture/export completeness.',
                  'Connection identifiers can be reused and same-name queue association may be ambiguous; all matching rows retained.',
                  'Within-DB timestamp containment follows installed parser matching logic, not proof of cross-domain clock precision or tensor/request causality.',
                  'No direct match to C hook monotonic timestamps is attempted. Device TASK connections alone do not prove graph-node identity.']}
(job/'ep8-layer14-queue-audit.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps({'layer14_calls':len(results),'markers_unique':sum(len(x['marker_rows'])==1 for x in results),
                  'operations':sum(len(x['operations']) for x in results),
                  'queues':sum(len(o['queues']) for x in results for o in x['operations'])}))
for rec in results:
    for op in rec['operations']:
        for q in op['queues']:
            for group in q['correlations']:
                print(json.dumps({'call_id':rec['call_id'],'marker':rec['marker_rows'][0],
                                  'operation':op['operation'],'enqueue':q['enqueue'],
                                  'correlation':group['correlation'],
                                  'dequeues':[{'event':d['dequeue'],'cann_total':d['cann_total'],
                                               'launch_api_count':len(d['selected_launch_and_tasks']),
                                               'cann_records':[{'api':x['cann_api'],'task_count':len(x['task_rows']),
                                                                'task_rows':x['task_rows'] if x['cann_api']['name']=='launch' else []} for x in d['cann_all']]}
                                              for d in group['matching_dequeues']]}))
