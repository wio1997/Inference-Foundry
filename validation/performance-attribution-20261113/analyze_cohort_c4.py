"""解析固定38请求C4归因作业；只读原始数据，不运行模型。"""
from pathlib import Path
from collections import Counter
import argparse
import ast
import hashlib
import json
import math
import sqlite3
import statistics
import struct


def read_array(blob):
    if blob[:8]!=b'\x93NUMPY\x01\x00':raise ValueError('expected NumPy v1 timestamps')
    length=struct.unpack('<H',blob[8:10])[0]
    header=ast.literal_eval(blob[10:10+length].decode())
    if header['descr']!='<f8' or header['fortran_order'] or len(header['shape'])!=1:
        raise ValueError('unexpected timestamp encoding')
    values=struct.unpack('<'+'d'*math.prod(header['shape']),blob[10+length:])
    if len(values)<3 or not all(map(math.isfinite,values)) or any(a>b for a,b in zip(values,values[1:])):
        raise ValueError('invalid timestamp sequence')
    return values


def validate_clocks(client,server):
    for key in ('boot_id_sha256','timens_offsets_sha256'):
        if not isinstance(client.get(key),str) or len(client[key])!=64 or client[key]!=server.get(key):
            raise ValueError('client/server monotonic clock provenance differs: '+key)
    for clock in (client,server):
        a=clock.get('monotonic_implementation');b=clock.get('perf_counter_implementation')
        if not a or a!=b or 'MONOTONIC' not in a:
            raise ValueError('perf_counter and observer clocks not proven equivalent')


def position(event,start,end):
    a=event['started_monotonic_ns'];b=event['host_return_monotonic_ns']
    if type(a) is not int or type(b) is not int or b<a or end<start:
        raise ValueError('invalid host interval')
    if b<start:return 'before'
    if a>end:return 'after'
    if a>=start and b<=end:return 'inside'
    if a<start and b>end:return 'crosses_both'
    return 'crosses_start' if a<start else 'crosses_end'


def event_ids(event):
    values=event['request_id_sha256']
    if not values or len(values)>8 or len(set(values))!=len(values) or any(
            not isinstance(x,str) or len(x)!=64 or any(c not in '0123456789abcdef' for c in x) for x in values):
        raise ValueError('missing or invalid request identity')
    return set(values)


def formal_request_ids(snapshot,records,start,end):
    """Bind client timing to host request identities without assuming completion order."""
    if len(records)!=38:raise ValueError('expected 38 completed service requests')
    by_id={}
    for row in records:
        rid=row['request_id']
        if not isinstance(rid,str) or not rid:raise ValueError('invalid server request ID')
        key=hashlib.sha256(rid.encode()).hexdigest()
        if key in by_id or not row['driver_owner_released'] or row['disconnected']:
            raise ValueError('duplicate or incomplete service request')
        by_id[key]=row
    first={}
    for event in snapshot['target_events']:
        position(event,start,end)
        for rid in event_ids(event):
            first[rid]=min(first.get(rid,event['started_monotonic_ns']),event['started_monotonic_ns'])
    if set(first)!=set(by_id):raise ValueError('observed and completed request sets differ')
    formal={rid for rid,t in first.items() if start<=t<=end}
    if len(formal)!=32 or any(by_id[r]['input_tokens']!=1024 or len(by_id[r]['generated_token_ids'])!=256 for r in formal):
        raise ValueError('formal client window does not identify exactly 32 service requests')
    return formal,set(by_id)


def summarize_journal(snapshot,start,end,formal_ids,all_ids):
    counts=snapshot['counts'];targets=snapshot['target_events'];graphs=snapshot['graph_events']
    if not snapshot['closed']:raise ValueError('observer has not closed')
    for key in ('metadata_errors','graph_metadata_errors','target_failures','target_events_dropped','graph_events_dropped'):
        if counts.get(key,0):raise ValueError('incomplete/failed observation: '+key)
    if len(targets)!=counts.get('target_invocations') or not targets:
        raise ValueError('missing target journal events')
    if len(graphs)!=sum(counts.get(k+'_graph_wrapper_calls',0) for k in ('target','draft')):
        raise ValueError('missing graph journal events')
    partitions={};inside={};ownership={};observed=set()
    for label,events in (('target',targets),('graph',graphs)):
        groups=Counter(position(q,start,end) for q in events)
        assert sum(groups.values())==len(events)
        partitions[label]=dict(groups)
        selected=[];owners=Counter()
        for q in events:
            ids=event_ids(q);observed.update(ids)
            if not ids<=all_ids:raise ValueError('event names an unknown service request')
            owner=('formal_only' if ids<=formal_ids else 'nonformal_only' if not ids&formal_ids else 'mixed_formal_nonformal')
            if position(q,start,end)=='inside':
                owners[owner]+=1
                if owner=='formal_only':selected.append(q)
        ownership[label]=dict(owners);inside[label]=selected
    if observed!=all_ids:raise ValueError('journal does not cover all completed service requests')
    return {'window_partitions':partitions,
        'inside_window_request_ownership':ownership,
        'target_modes_inside':dict(Counter(q['mode'] for q in inside['target'])),
        'target_kinds_inside':dict(Counter(q['kind'] for q in inside['target'])),
        'target_shapes_inside':dict(Counter(f"{q['kind']}/{q['mode']}/{q['requests']}/{q['scheduled_tokens']}" for q in inside['target'])),
        'graph_routes_inside':dict(Counter(q['scope']+('/replay' if q['actual_replay'] else '/non_replay') for q in inside['graph'])),
        'whole_run_counts':counts,
        'scope':'host calls fully contained in formal window and belonging only to formal requests;boundary and nonformal/mixed calls reported separately;not GPU time'}


def analyze(root,client_clock_path):
    read=lambda p:json.loads(p.read_text())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    state=read(root/'native-run/manager-state.json')
    if state['status']!='passed' or not state.get('claims_released_at') or state['container_final_inspect']['Running']:
        raise ValueError('run not passed and released')
    if len(state['jobs'])!=8 or any(q['exit_code']!=0 for q in state['jobs']):
        raise ValueError('worker lifecycle incomplete')
    if read(root/'source-identity.json')['commit']!='76bb59edf66caef19e81a13bf91b698aba09659b':
        raise ValueError('model source differs from fixed139')
    rank0=root/'native-run/communication/native-tp8/rank0'
    result=read(rank0/'result.json');service=read(rank0/'service-ready.json');cohort=read(rank0/'native-admission-cohort.json')
    if result['completed_requests']!=38 or cohort['enabled']!=service['native_admission_cohort']:
        raise ValueError('fixed38request or admission identity differs')
    enabled=cohort['enabled']
    if service['continuous_batching']==enabled or cohort['decoding_cohort_active']:
        raise ValueError('admission lifecycle incomplete')
    cc=cohort['counts']
    if enabled:
        if not (cc['admitted_requests']==38 and cc['cohorts_started']==cc['cohorts_finished']>0):
            raise ValueError('cohort did not run and close')
    elif cc['admitted_requests'] or cc['cohorts_started'] or cc['cohorts_finished']:
        raise ValueError('cohort-off control actually executed cohort policy')
    bench=root/'aisbench-c4-results';details=list(bench.rglob('*_details.jsonl'))
    if len(details)!=1:raise ValueError('select one complete C4 result')
    rows=[json.loads(x) for x in details[0].read_text().splitlines() if x.strip()]
    if len(rows)!=32 or len({q['data_id'] for q in rows})!=32 or any(not q['success'] or q['input_tokens']!=1024 or q['output_tokens']!=256 for q in rows):
        raise ValueError('formal C4 request set incomplete')
    dbs={p.name:p for p in bench.rglob('*.db')};timestamps=[];inputs={str(details[0].relative_to(root)):sha(details[0])}
    for row in rows:
        db=dbs[Path(row['db_name']).name]
        with sqlite3.connect('file:'+str(db.resolve())+'?mode=ro',uri=True) as connection:
            record=connection.execute('SELECT arr_blob FROM numpy_store WHERE id=?',(row['time_points']['__db_ref__'],)).fetchone()
        if record is None:raise ValueError('missing timestamp blob')
        timestamps.append(read_array(record[0]));inputs[str(db.relative_to(root))]=sha(db)
    start=min(t[0] for t in timestamps);end=max(t[-1] for t in timestamps)
    client=read(client_clock_path);ranks=[];observer_shas=set()
    records_path=rank0/'service-requests.jsonl'
    records=[json.loads(x) for x in records_path.read_text().splitlines() if x.strip()]
    inputs[str(records_path.relative_to(root))]=sha(records_path)
    start_ns,end_ns=round(start*1e9),round(end*1e9)
    rank0_snapshot=read(rank0/'execution-route-observer.json')
    formal_ids,all_ids=formal_request_ids(rank0_snapshot,records,start_ns,end_ns)
    for rank in range(8):
        leaf=root/f'native-run/communication/native-tp8/rank{rank}'
        memory=read(leaf/'native-prefill-state.json');counts=memory['counts']
        if not memory['closed'] or memory['entries'] or any(counts.get(k,0) for k in ('saved_entries','reused_requests','state_copy_kernel_launches')):
            raise ValueError('state-off control did not remain off')
        if memory['state_arena_bytes']!=1073741824 or memory['ordinary_pool_budget_bytes']!=3221225472:
            raise ValueError('cache partition differs')
        if memory['total_cache_budget_bytes']!=4294967296 or memory['physical_pool_payload_bytes']+memory['state_arena_bytes']>4294967296:
            raise ValueError('physical cache budget exceeded')
        p=leaf/'execution-route-observer.json';snapshot=read(p)
        validate_clocks(client,snapshot['clock'])
        if snapshot['cohort_enabled']!=enabled:raise ValueError('observer/control flag mismatch')
        observer_shas.add(snapshot['observer_source_sha256']);inputs[str(p.relative_to(root))]=sha(p)
        ranks.append({'rank':rank,**summarize_journal(snapshot,start_ns,end_ns,formal_ids,all_ids)})
    if len(observer_shas)!=1:raise ValueError('observers differ across ranks')
    source=root/'overlay/execution_route_observer.py'
    if sha(source)!=next(iter(observer_shas)):raise ValueError('imported observer differs from collected source')
    inputs[str(source.relative_to(root))]=sha(source)
    metrics={'successful_requests':32,'input_tokens':32768,'output_tokens':8192,'benchmark_seconds':end-start,
        'output_token_throughput':8192/(end-start),
        'ttft_median_ms':statistics.median((t[1]-t[0])*1000 for t in timestamps),
        'tpot_median_ms':statistics.median((t[-1]-t[1])*1000/255 for t in timestamps)}
    official_path=details[0].with_name(details[0].name.removesuffix('_details.jsonl')+'.json')
    official=read(official_path)
    if official['Success Requests']['total']!=32 or official['Failed Requests']['total']!=0:
        raise ValueError('official success/failure counts differ')
    if abs(float(official['Output Token Throughput']['total'].split()[0])-metrics['output_token_throughput'])>0.000051:
        raise ValueError('official throughput and raw timestamps differ')
    inputs[str(official_path.relative_to(root))]=sha(official_path)
    return {'status':'one_C4_arm_analyzed_not_causal_verdict','cohort_enabled':enabled,
        'formal_window_seconds':[start,end],'clock_provenance_matched':True,
        'observer_source_sha256':next(iter(observer_shas)),'metrics':metrics,'ranks':ranks,'input_sha256':inputs,
        'client_clock_sha256':sha(client_clock_path),'official_throughput_agrees':True,
        'formal_request_id_sha256':sorted(formal_ids),
        'identity_rule':'first target host call inside client window;exactly32 of38 completed service IDs;no completion-order assumption',
        'boundary_rule':'only fully-contained host calls counted as inside;crossing calls separately retained;float timestamp conversion has nanosecond-scale rounding',
        'not_proved':['device kernel time','causal effect without other arm','run-order variation','general model quality'],'goal_complete':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--client-clock',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=analyze(args.run_root,args.client_clock)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'metrics':result['metrics']},ensure_ascii=False))
