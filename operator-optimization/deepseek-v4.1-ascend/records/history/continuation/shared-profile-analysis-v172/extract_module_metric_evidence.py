"""One factorized pass over the existing v164 CPU+PMU cohort; no device work."""
import bisect
import collections
import concurrent.futures
import hashlib
import json
import os
import sqlite3
import time
import traceback
from pathlib import Path

O=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/analysis/joint172')
INPUT=json.loads((O/'INPUT-DATABASES.json').read_text())
state={'state':'RUNNING','pid':os.getpid(),'start_ticks':Path('/proc/'+str(os.getpid())+'/stat').read_text().rsplit(')',1)[1].split()[19],'started_unix':time.time(),'completed':0,'max_workers':2,'CPU_only':True}

def status():
    (O/'STATUS.json').write_text(json.dumps(state,indent=2)+'\n')

def one(entry):
    db=Path(entry['database']);bhash=hashlib.sha256()
    with db.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):bhash.update(b)
    assert bhash.hexdigest()==entry['db_sha256'],str(db)
    c=sqlite3.connect('file:'+str(db)+'?mode=ro&immutable=1',uri=True);c.row_factory=sqlite3.Row
    strings=dict(c.execute('select id,value from STRING_IDS'))
    info={r['globalTaskId']:dict(r) for r in c.execute('select * from COMPUTE_TASK_INFO')}
    tasks=[dict(r) for r in c.execute('select rowid TASK_ROW,* from TASK where globalTaskId in (select globalTaskId from COMPUTE_TASK_INFO)')]
    assert len(tasks)==entry['compute_task_count']
    # Connection IDs in PYTORCH_API refer to official connection sets.
    cpu={r['CPU_ROW']:dict(r) for r in c.execute('select rowid CPU_ROW,* from PYTORCH_API where type=50001 and connectionId is not null')}
    by_set=collections.defaultdict(list)
    for rid,r in cpu.items():by_set[r['connectionId']].append(rid)
    by_native=collections.defaultdict(set)
    for r in c.execute('select id,connectionId from CONNECTION_IDS'):
        by_native[r[1]].update(by_set.get(r[0],()))
    used_cpu=set();links={};link_counts=collections.Counter()
    for t in tasks:
        ids=sorted(by_native.get(t['connectionId'],()))
        links[t['TASK_ROW']]=ids;used_cpu.update(ids)
        link_counts['one' if len(ids)==1 else 'zero' if not ids else 'many']+=1
    used_chains={cpu[r]['callchainId'] for r in used_cpu}
    project_frames={k:v for k,v in strings.items() if v.startswith('/vllm-workspace/') and '.py(' in v}
    chains=collections.defaultdict(list)
    if used_chains and project_frames:
        query='select id,stackDepth,stack from PYTORCH_CALLCHAINS where id in ('+','.join('?' for _ in used_chains)+') and stack in ('+','.join('?' for _ in project_frames)+')'
        for r in c.execute(query,[*used_chains,*project_frames]):chains[r['id']].append((r['stackDepth'],r['stack']))
    for k in chains:chains[k]=[x[1] for x in sorted(chains[k])]
    # Keep all exported AIC/AIV fields,not a favorable metric-only sample.
    metric_ids={k:v for k,v in strings.items() if v.startswith('aic_') or v.startswith('aiv_')}
    metric_names=sorted(set(metric_ids.values()));metric_index={n:i for i,n in enumerate(metric_names)}
    by_id=collections.defaultdict(list)
    for t in tasks:by_id[t['globalTaskId']].append(t)
    for ts in by_id.values():ts.sort(key=lambda x:x['startNs'])
    starts={k:[int(t['startNs']) for t in ts] for k,ts in by_id.items()}
    pmu={};pairing=collections.Counter();duplicate_fields=0;invalid_metric_values=0
    if metric_ids:
        sql='select timestampNs,globalTaskId,name,value from TASK_PMU_INFO where name in ('+','.join('?' for _ in metric_ids)+')'
        for r in c.execute(sql,list(metric_ids)):
            ts=by_id.get(r['globalTaskId'],[])
            if not ts:pairing['same_id_compute_absent']+=1;continue
            stamp=int(r['timestampNs']);at=bisect.bisect_right(starts[r['globalTaskId']],stamp)
            matches=[t for t in ts[:at] if stamp<=int(t['endNs'])]
            if len(matches)!=1:pairing['no_containing_task' if not matches else 'ambiguous_containing_tasks']+=1;continue
            pairing['unique_same_id_containment_fields']+=1
            t=matches[0];name=metric_ids[r['name']];values=pmu.setdefault(t['TASK_ROW'],{})
            if name in values:duplicate_fields+=1;values[name]=None;continue
            value=r['value']
            try:value=float(value)
            except (ValueError,TypeError):invalid_metric_values+=1;value=None
            values[name]=value
    regimes={};regime_rows=[];out=[]
    for t in tasks:
        x=info[t['globalTaskId']]
        fields={k:strings.get(x[k],x[k]) for k in ['name','opType','taskType','inputShapes','outputShapes','inputDataTypes','outputDataTypes','inputFormats','outputFormats']}
        for k in ['blockDim','mixBlockDim']:
            if k in x:fields[k]=x[k]
        rk=json.dumps(fields,sort_keys=True)
        if rk not in regimes:regimes[rk]=len(regime_rows);regime_rows.append(fields)
        rid=t['TASK_ROW'];values=pmu.get(rid)
        out.append([rid,t['globalTaskId'],int(t['startNs']),int(t['endNs']),t['streamId'],t['taskId'],t['modelId'],t['contextId'],t['connectionId'],regimes[rk],links[rid],[values.get(n) for n in metric_names] if values is not None else None])
    compact_cpu={rid:{'name':strings[cpu[rid]['name']],'callchain_id':cpu[rid]['callchainId'],'global_tid':cpu[rid]['globalTid'],'start_ns':int(cpu[rid]['startNs']),'end_ns':int(cpu[rid]['endNs'])} for rid in used_cpu}
    comm=[]
    if c.execute("select count(*) from sqlite_master where type='table' and name='COMMUNICATION_OP'").fetchone()[0]:
        for r in c.execute('select * from COMMUNICATION_OP'):
            comm.append({k:strings.get(v,v) if k in ['opName','groupName','opType','dataType'] else v for k,v in dict(r).items()})
    result={'profile':db.parent.parent.name,'database':str(db),'database_sha256':entry['db_sha256'],'native_columns':['TASK_ROW','globalTaskId','start_ns','end_ns','stream_id','task_id','model_id','context_id','native_connection','regime_index','CPU_candidate_rows','metric_values'],
        'metric_names':metric_names,'native_rows':out,'regimes':regime_rows,'CPU_rows':compact_cpu,'source_frame_strings':project_frames,'callchains':dict(chains),'communication_records':comm,
        'summary':{'native_executions':len(out),'PMU_execution_groups':len(pmu),'PMU_fields':sum(len(x) for x in pmu.values()),'PMU_pairing':dict(pairing),'duplicate_metric_fields':duplicate_fields,'invalid_metric_values':invalid_metric_values,'CPU_connection_cardinality':dict(link_counts),'used_CPU_rows':len(used_cpu),'used_project_callchains':len(chains),'project_source_frames':len(project_frames),'native_regimes':len(regime_rows),'communication_records':len(comm)},
        'scope':'Existingv164CPU+PMUwindows,lightweightCPUrecorder/profilingoverhead;notv168devicehistoryorunprofiledbaseline. No sourcebodyread ordevicework.'}
    data=(json.dumps(result,separators=(',',':'))+'\n').encode()
    assert len(data)<=20*1024*1024,'factorizedprofileartifactbudget'
    p=O/'profiles'/(db.parent.parent.name+'.json');p.write_bytes(data);c.close()
    return {'profile':result['profile'],'artifact':str(p),'artifact_bytes':len(data),'artifact_sha256':hashlib.sha256(data).hexdigest(),'summary':result['summary']}

if __name__=='__main__':
    (O/'profiles').mkdir(exist_ok=True);status()
    try:
        rows=[]
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            for row in pool.map(one,INPUT):
                rows.append(row);state['completed']=len(rows);status()
        assert sum(r['artifact_bytes'] for r in rows)<=512*1024*1024
        summary={'status':'ALL32_FACTORISED_MODULE_METRIC_EVIDENCE_RETURNED','profiles':rows,'native_executions':sum(r['summary']['native_executions'] for r in rows),'PMU_execution_groups':sum(r['summary']['PMU_execution_groups'] for r in rows),'duplicate_metric_fields':sum(r['summary']['duplicate_metric_fields'] for r in rows),'invalid_metric_values':sum(r['summary']['invalid_metric_values'] for r in rows),'CPU_connection_cardinality':dict(sum((collections.Counter(r['summary']['CPU_connection_cardinality']) for r in rows),collections.Counter())),'total_artifact_bytes':sum(r['artifact_bytes'] for r in rows),'scope':rows and 'Same32v164CPU+PMUprofiles withverifiedinputDBhashes;no v166NPU-only orv168history timingjoins.','overall_complete':False}
        f=O/'EXTRACTION-REVIEW.json';f.write_text(json.dumps(summary,indent=2)+'\n');state.update(state='RETURNED',elapsed_seconds=time.time()-state['started_unix'],result=str(f));status();print(json.dumps({k:v for k,v in summary.items() if k!='profiles'}))
    except BaseException:
        state.update(state='ERROR',error=traceback.format_exc());status();raise
