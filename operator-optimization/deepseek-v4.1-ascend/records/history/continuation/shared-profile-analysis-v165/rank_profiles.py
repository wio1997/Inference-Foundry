"""Profile-wide measured task ranking and exact vendor correlation census."""
import collections,concurrent.futures,hashlib,json,re,sqlite3,time
from pathlib import Path
H=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/analysis/builtin-v164-official-db')

def union_ns(intervals):
    total=0;end=None
    for a,b in sorted(intervals):
        if end is None or a>end:total+=max(0,b-a);end=b
        elif b>end:total+=b-end;end=b
    return total

def one(db):
    c=sqlite3.connect('file:'+str(db)+'?mode=ro&immutable=1',uri=True);c.row_factory=sqlite3.Row
    strings=dict(c.execute('select id,value from STRING_IDS'))
    s=lambda i:strings.get(i,str(i))
    tasks=[dict(r) for r in c.execute('select * from TASK')]
    compute={r['globalTaskId']:dict(r) for r in c.execute('select * from COMPUTE_TASK_INFO')}
    conn_to_torch=collections.defaultdict(list)
    for r in c.execute('select p.rowid as rowid,p.*,x.connectionId as nativeConnection from PYTORCH_API p join CONNECTION_IDS x on p.connectionId=x.id where p.type=50001'):
        conn_to_torch[r['nativeConnection']].append(dict(r))
    groups={};regimes={};links=collections.Counter();link_examples=[];compute_intervals=[];types={}
    for t in tasks:
        typename=s(t['taskType']);g=types.setdefault(typename,{'count':0,'sum_duration_ns':0});g['count']+=1;g['sum_duration_ns']+=max(0,t['endNs']-t['startNs'])
        x=compute.get(t['globalTaskId'])
        if x is None:continue
        name=s(x['opType']);dt=max(0,t['endNs']-t['startNs']);compute_intervals.append((t['startNs'],t['endNs']))
        g=groups.setdefault(name,{'count':0,'sum_duration_ns':0,'max_duration_ns':0});g['count']+=1;g['sum_duration_ns']+=dt;g['max_duration_ns']=max(g['max_duration_ns'],dt)
        key=json.dumps([name,s(x['inputDataTypes']),s(x['inputShapes']),s(x['outputShapes']),x['blockNum']],separators=(',',':'))
        g=regimes.setdefault(key,{'count':0,'sum_duration_ns':0});g['count']+=1;g['sum_duration_ns']+=dt
        # Exact vendor connection identity within this database. No nearest-time joins.
        candidates={r['rowid']:r for r in conn_to_torch.get(t['connectionId'],[])}
        links['zero' if not candidates else 'one' if len(candidates)==1 else 'multiple']+=1
        if name in ['DispatchFFNCombine','SparseFlashMla','VllmAscendQuantLightningIndexerV2','ZerosLike','MemSet'] and len(link_examples)<25:
            link_examples.append({'task_global_id':t['globalTaskId'],'connection_id':t['connectionId'],'op':name,
                'torch_candidates':[{'rowid':r['rowid'],'op':s(r['name']),'input_shapes':s(r['inputShapes']),'callchain_id':r['callchainId']} for r in candidates.values()]})
    pmu=[]
    for r in c.execute('select name,count(*) n,min(value) lo,max(value) hi from TASK_PMU_INFO group by name'):
        pmu.append({'name':s(r['name']),'rows':r['n'],'min':r['lo'],'max':r['hi']})
    markers=[]
    for r in c.execute('select p.rowid,p.* from PYTORCH_API p'):
        name=s(r['name'])
        if name.startswith('wio_shared/'):
            markers.append({'rowid':r['rowid'],'name':name,'start_ns':int(r['startNs']),'end_ns':int(r['endNs']),'global_tid':r['globalTid']})
    c.close();h=hashlib.sha256()
    with db.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return {'profile':db.parent.parent.name,'database':str(db),'db_bytes':db.stat().st_size,'db_sha256':h.hexdigest(),
      'task_count':len(tasks),'compute_task_count':len(compute_intervals),'compute_task_sum_ns':sum(b-a for a,b in compute_intervals),
      'compute_active_union_ns':union_ns(compute_intervals),'compute_span_ns':max((b for a,b in compute_intervals),default=0)-min((a for a,b in compute_intervals),default=0),
      'task_types':types,'operators':groups,'regimes':regimes,'exact_compute_task_to_torch_connection_cardinality':dict(links),
      'connection_examples':link_examples,'PMU_fields':pmu,'CPU_scope_markers':markers}

def main():
    status=json.loads((H/'STATUS.json').read_text());assert status['state']=='VENDOR_ANALYSE_RETURNED',status['state']
    dbs=sorted((H/'profiles').glob('*/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler*.db'));assert len(dbs)==32,len(dbs)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(one,dbs))
    aggregate={};links=collections.Counter()
    for row in rows:
        links.update(row['exact_compute_task_to_torch_connection_cardinality'])
        for name,g in row['operators'].items():
            x=aggregate.setdefault(name,{'count':0,'sum_duration_ns_across_ranks_windows':0,'profiles':0})
            x['count']+=g['count'];x['sum_duration_ns_across_ranks_windows']+=g['sum_duration_ns'];x['profiles']+=1
    top=sorted([{'op':name,**g} for name,g in aggregate.items()],key=lambda r:-r['sum_duration_ns_across_ranks_windows'])
    out={'status':'ALL32_VENDOR_DATABASES_RANKED','generated_unix':time.time(),'profiles':rows,'operator_ranking':top,
      'exact_compute_task_to_torch_connection_cardinality':dict(links),
      'measurement_limits':['Instrumented short windows; not a production speedup baseline.',
        'Across-rank/window summed task duration is work exposure, not request critical-path latency.',
        'Connection IDs are local to their database; replay-template mapping requires explicit graph relation.',
        'PMU field presence alone does not establish physical traffic or hardware ceiling.'],
      'whole_modules_closed':0,'overall_complete':False}
    (H/'ALL32-RANKING.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':out['status'],'profiles':len(rows),'compute_tasks':sum(r['compute_task_count'] for r in rows),'top':top[:15],'connections':dict(links)},ensure_ascii=False))

if __name__=='__main__':main()
