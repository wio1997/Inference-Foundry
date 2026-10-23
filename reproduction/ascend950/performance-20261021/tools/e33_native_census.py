#!/usr/bin/env python3
"""Census every exported kernel, API and observed work event; no top-K filter.

Read-only input; output is deliberately not a full-chain completion verdict.
Requires NumPy solely to decode the native AISBench clock database.
"""
import argparse
import collections
import csv
from decimal import Decimal
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import statistics
import time
import numpy as np
from e33_clock_map import ClockMap, audit_di_cpu_boundaries

ROOT=Path(__file__).resolve().parents[1]

def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def csvout(path,rows,fields):
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
def union_ns(intervals):
    total=0;end=None
    for start,stop in sorted(intervals):
        if end is None or start>end:total+=stop-start;end=stop
        elif stop>end:total+=stop-end;end=stop
    return total
def ns_us(value):return int(Decimal(str(value).strip())*1000)
def conn(path):return sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True)
def native_clock_rows(run_id,diag,framework='vllm'):
    if framework=='di':return di_clock_rows(run_id,diag)
    anchors=[];spans=[];spec=[];snapshots=[];seen=set();duplicates=[]
    for p in sorted((diag/'jsonl').glob('*.jsonl')):
        with p.open() as f:
            for line in f:
                x=json.loads(line)
                if x.get('run_id')!=run_id:continue
                key=(x.get('pid'),x.get('event_seq'))
                if key in seen:duplicates.append(key)
                seen.add(key)
                if x.get('event')=='cohort_clock_anchor':anchors.append(x)
                if x.get('event')=='span':spans.append(x)
                if 'make_spec_decoding_stats' in x.get('source',''):spec.append(x)
                if x.get('event') in ('snapshot','snapshot_error'):snapshots.append(x)
    return anchors,spans,spec,snapshots,duplicates
def di_clock_rows(run_id,diag):
    """Normalize diagnostic records, leaving every native input file intact."""
    anchors=[];spans=[];spec=[];snapshots=[];duplicates=[];seen=set()
    for p in sorted(diag.rglob('events-*.jsonl')):
        with p.open() as f:
            for seq,line in enumerate(f):
                x=json.loads(line);assert x['run_id']==run_id
                rank=x['rank'];pid=x['pid'];name=x['event_name'];start=x['perf_counter_ns_start'];end=x['perf_counter_ns_end']
                canonical=dict(event='span',event_seq=seq,pid=pid,dp_rank=rank if rank>=0 else None,source='deepinfer.'+x['source']+'.'+name,
                    wall_ns=x['time_time_ns_start'],wall_end_ns=x['time_time_ns_end'],monotonic_ns=start,monotonic_end_ns=end,duration_ns=end-start,
                    thread_cpu_delta_ns=x['thread_time_ns'],work=x.get('extra',{}),extra={'native_event_name':name,'native_source':x['source']},
                    identity=dict(dp_rank=rank),iteration_id=x.get('iteration_id'),request_id=x.get('request_id'))
                if end>start:spans.append(canonical)
                if name=='spec_counters':
                    key=(rank,x['iteration_id'])
                    if key in seen:duplicates.append(key)
                    seen.add(key);spec.append(canonical)
    for p in sorted(diag.rglob('profile_manifest.json')):
        x=json.loads(p.read_text());assert x['run_id']==run_id and not x['synthetic']
        for boundary in ['start','end']:
            before,after=x['mstx'][boundary+'_call_bracket'];assert before and after
            anchors.append(dict(event='cohort_clock_anchor',pid=x['identity']['pid'],dp_rank=x['rank'],identity=dict(dp_rank=x['rank']),
                wall_ns=before['wall_ns'],wall_end_ns=after['wall_ns'],monotonic_ns=before['perf_counter_ns'],monotonic_end_ns=after['perf_counter_ns'],
                extra=dict(boundary=boundary,mstx_name=x['mstx']['range_label']),source_manifest=str(p)))
    for p in sorted(diag.rglob('layout-*.json')):snapshots.append(dict(path=str(p),work=json.loads(p.read_text())))
    return anchors,spans,spec,snapshots,duplicates
def clients(run_id,offset):
    paths=list((ROOT/'runs/e33-ais'/run_id).rglob('*_details.jsonl'));assert len(paths)==1
    path=paths[0];arrays={};rows=[];digest=hashlib.sha256()
    for db in (path.parent/'db_data').glob('*.db'):
        with conn(db) as c:arrays[db.name]={i:np.load(io.BytesIO(blob),allow_pickle=False) for i,blob in c.execute('SELECT id,arr_blob FROM numpy_store')}
    with path.open('rb') as f:
        for line in f:
            digest.update(line);x=json.loads(line);e=x['extra_details_data']
            assert x['success'] and not x['error_info']
            assert x['input_tokens']==32768 and x['output_tokens']==128
            assert len(e['native_output_token_ids'])==128
            clocks=arrays[x['db_name']][x['time_points']['__db_ref__']]
            assert np.all(np.isfinite(clocks)) and np.all(np.diff(clocks)>=0)
            payload=e['first_sse_payload'];choices=payload['choices']
            rows.append(dict(logical_id=str(x['input']),owner_rank=e['owner_rank'],request_id=e['request_id'],
                start_monotonic_ns=int(round(clocks[0]*1e9)),first_token_monotonic_ns=int(round(clocks[1]*1e9)),end_monotonic_ns=int(round(clocks[-1]*1e9)),
                start_wall_ns=int(round(clocks[0]*1e9))+offset,first_token_wall_ns=int(round(clocks[1]*1e9))+offset,end_wall_ns=int(round(clocks[-1]*1e9))+offset,
                TTFT_ms=float((clocks[1]-clocks[0])*1000),TPOT_ms=float((clocks[-1]-clocks[1])*1000/127),E2EL_ms=float((clocks[-1]-clocks[0])*1000),
                first_payload_json_bytes=len(json.dumps(payload,ensure_ascii=False).encode()),first_payload_prompt_ids=sum(len(c.get('prompt_token_ids') or []) for c in choices),first_payload_text_chars=sum(len(c.get('text') or '') for c in choices)))
    assert len(rows)==64 and collections.Counter(x['owner_rank'] for x in rows)=={i:8 for i in range(8)}
    return rows,dict(path=str(path),sha256=digest.hexdigest(),requests=64,actual_outputs=8192)
def profiles(manifest):
    x=json.loads(manifest.read_text())
    if 'exports' in x:values=x['exports']
    elif 'raw_exports' in x:values=x['raw_exports']
    else:raise ValueError('Require explicit native profile list; do not scan unrelated cohorts')
    paths=[]
    for row in values:
        p=Path(row['path'])
        # Archives can be relocated between host/container roots.
        if str(p).startswith('/work/performance-20261021/'):
            p=ROOT/str(p).removeprefix('/work/performance-20261021/')
        paths.append(p)
    assert len(paths)==8 and len(set(paths))==8
    return paths
def rank_census(path,run_id,start,end,anchors,out,marker_pattern=None,spans=None,framework='vllm'):
    native=path/'ASCEND_PROFILER_OUTPUT';dbs=list(native.glob('ascend_pytorch_profiler_*.db'));assert len(dbs)==1
    c=conn(dbs[0]);c.row_factory=sqlite3.Row
    mapping=[dict(x) for x in c.execute('SELECT * FROM RANK_DEVICE_MAP')];assert len(mapping)==1
    rank=mapping[0]['rankId'];device=mapping[0]['deviceId']
    marker=[dict(x) for x in c.execute("SELECT m.*,s.value as label FROM MSTX_EVENTS m JOIN STRING_IDS s ON m.message=s.id WHERE s.value LIKE ?",(marker_pattern or 'e33_cohort:'+run_id+':%',))];assert len(marker)==1
    marker=marker[0]
    if ':pid' in marker['label']:pid=int(marker['label'].rsplit('pid',1)[1])
    else:
        pids={x['pid'] for x in anchors if x.get('dp_rank')==rank};assert len(pids)==1;pid=pids.pop()
    host=[x for x in anchors if x['pid']==pid];assert len(host)==2
    bracket={x['extra']['boundary']:x for x in host}
    clock=ClockMap.from_profile(path);native_start=clock.native_ns(start);native_end=clock.native_ns(end)
    checks={};marker_audit={}
    for label,native_ns in [('start',marker['startNs']),('end',marker['endNs'])]:
        h=bracket[label];wall=clock.wall_ns(native_ns)
        checks[label+'_calibrated_marker_in_host_bracket']=h['wall_ns']<=wall<=h['wall_end_ns']
        marker_audit[label]=dict(native_ns=native_ns,calibrated_wall_ns=wall,host_bracket_ns=[h['wall_ns'],h['wall_end_ns']],raw_in_bracket=h['wall_ns']<=native_ns<=h['wall_end_ns'],calibrated_margins_ns=[wall-h['wall_ns'],h['wall_end_ns']-wall])
    checks['entire_client_window_inside_marker']=clock.wall_ns(marker['startNs'])<=start<=end<=clock.wall_ns(marker['endNs'])
    independent=audit_di_cpu_boundaries(c,clock,spans or [],rank) if framework=='di' else {}
    if independent:checks['independent_midrun_cpu_enclosures']=all(x['passed'] for x in independent.values())
    kernel=collections.defaultdict(lambda:{'count':0,'sum_duration_ns':0,'max_duration_ns':0});full=collections.defaultdict(lambda:{'count':0,'sum_duration_ns':0,'outside_client_count':0});intervals=[];outside=0;fields=None;device_ids=set()
    with (native/'kernel_details.csv').open() as f:
        reader=csv.DictReader(f);fields=reader.fieldnames
        for row in reader:
            raw_start=ns_us(row['Start Time(us)']);raw_end=raw_start+ns_us(row['Duration(us)'])
            s=clock.wall_ns(raw_start);e=clock.wall_ns(raw_end);duration=e-s
            attrs={k:v for k,v in row.items() if any(tag in k.lower() for tag in ['shape','type','format']) and k not in ['Type']}
            key=(row['Name'],row.get('Type',''),json.dumps(attrs,sort_keys=True))
            whole=full[key];whole['count']+=1;whole['sum_duration_ns']+=duration
            if e<=start or s>=end:outside+=1;whole['outside_client_count']+=1;continue
            device_ids.add(int(row['Device_id']))
            duration=min(e,end)-max(s,start);intervals.append((max(s,start),min(e,end)))
            g=kernel[key];g['count']+=1;g['sum_duration_ns']+=duration;g['max_duration_ns']=max(g['max_duration_ns'],duration)
    checks['one_expected_device_in_kernel_csv']=device_ids=={device}
    kernelrows=[dict(rank=rank,device=device,name=k[0],type=k[1],attributes=k[2],**v) for k,v in sorted(kernel.items())]
    csvout(out/('rank%02d-all-kernels.csv'%rank),kernelrows,['rank','device','name','type','attributes','count','sum_duration_ns','max_duration_ns'])
    fullrows=[dict(rank=rank,device=device,name=k[0],type=k[1],attributes=k[2],**v) for k,v in sorted(full.items())]
    csvout(out/('rank%02d-all-capture-kernels.csv'%rank),fullrows,['rank','device','name','type','attributes','count','sum_duration_ns','outside_client_count'])
    cpugroups=[]
    for table,label in [('PYTORCH_API','pytorch'),('CANN_API','cann')]:
        query=f'''SELECT s.value as name,count(*) as calls,
        sum(min(CAST(a.endNs AS INTEGER),?)-max(CAST(a.startNs AS INTEGER),?)) as inclusive_wall_ns
        FROM {table} a JOIN STRING_IDS s ON a.name=s.id
        WHERE CAST(a.endNs AS INTEGER)>? AND CAST(a.startNs AS INTEGER)<? GROUP BY s.value'''
        for row in c.execute(query,(native_end,native_start,native_start,native_end)):
            d=dict(row);d['inclusive_wall_ns']=clock.duration_ns(d['inclusive_wall_ns']);cpugroups.append(dict(rank=rank,layer=label,**d))
    csvout(out/('rank%02d-all-host-api.csv'%rank),cpugroups,['rank','layer','name','calls','inclusive_wall_ns'])
    fullcpu=[]
    for table,label in [('PYTORCH_API','pytorch'),('CANN_API','cann')]:
        for row in c.execute(f'SELECT s.value as name,count(*) as calls,sum(CAST(a.endNs AS INTEGER)-CAST(a.startNs AS INTEGER)) as inclusive_wall_ns FROM {table} a JOIN STRING_IDS s ON a.name=s.id GROUP BY s.value'):
            d=dict(row);d['inclusive_wall_ns']=clock.duration_ns(d['inclusive_wall_ns']);fullcpu.append(dict(rank=rank,layer=label,**d))
    csvout(out/('rank%02d-all-capture-host-api.csv'%rank),fullcpu,['rank','layer','name','calls','inclusive_wall_ns'])
    commrows=[];comm_intervals=[]
    for row in c.execute('''SELECT s.value as name,o.* FROM COMMUNICATION_OP o LEFT JOIN STRING_IDS s ON o.opName=s.id WHERE o.endNs>? AND o.startNs<? ORDER BY o.startNs''',(native_start,native_end)):
        d=dict(row);d['native_startNs']=d['startNs'];d['native_endNs']=d['endNs'];d['startNs']=clock.wall_ns(d['startNs']);d['endNs']=clock.wall_ns(d['endNs']);commrows.append(d);comm_intervals.append((max(start,d['startNs']),min(end,d['endNs'])))
    dump(out/('rank%02d-communication-ops.json'%rank),commrows)
    infofiles=list(path.glob('profiler_info*.json'));assert len(infofiles)==1
    session=[dict(x) for x in c.execute('SELECT * FROM SESSION_TIME_INFO')]
    stats=dict(rank=rank,device=device,pid=pid,profile_dir=str(path),db_bytes=dbs[0].stat().st_size,
               marker=marker,marker_clock_audit=marker_audit,clock_mapping=clock.as_dict(),independent_cpu_clock_validation=independent,host_anchors=host,checks=checks,session=session,kernel_csv_fields=fields,
               observed_kernel_records=sum(v['count'] for v in kernel.values()),distinct_kernel_names=len({k[0] for k in kernel}),distinct_kernel_signatures=len(kernel),
               full_capture_kernel_records=sum(v['count'] for v in full.values()),full_capture_distinct_kernel_names=len({k[0] for k in full}),
               kernel_duration_sum_ns=sum(v['sum_duration_ns'] for v in kernel.values()),kernel_interval_union_ns=union_ns(intervals),
               client_window_not_covered_by_kernel_csv_ns=(end-start)-union_ns(intervals),outside_client_window_kernel_records=outside,
               communication_ops=len(commrows),communication_op_wall_union_ns=union_ns(comm_intervals),
               host_api_distinct_names=len(cpugroups),host_api_calls=sum(x['calls'] for x in cpugroups),
               drops_counter_available=False,coverage_limit='Kernel union is observed kernel time, not complete NPU activity; host API sums are inclusive and nested, communication ops include waiting. Missing drop counters are not proof of zero loss.')
    c.close();return stats

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True);parser.add_argument('--diag-root',required=True);parser.add_argument('--profile-manifest',required=True);parser.add_argument('--out',required=True);parser.add_argument('--framework',choices=['di','vllm'],default='vllm');parser.add_argument('--census-only',action='store_true',help='Finish all tables even if qualification fails; the failed checks remain explicit in summary.json.');a=parser.parse_args()
    diag=ROOT/a.diag_root;out=ROOT/a.out;assert out.resolve().is_relative_to(ROOT.resolve());out.mkdir(parents=True,exist_ok=False)
    anchors,spans,spec,snapshots,duplicates=native_clock_rows(a.run_id,diag,a.framework)
    assert len(anchors)==16 and not duplicates
    offsets=[x['wall_ns']-x['monotonic_ns'] for x in anchors]+[x['wall_end_ns']-x['monotonic_end_ns'] for x in anchors]
    offset=int(statistics.median(offsets));client,evidence=clients(a.run_id,offset)
    start=min(x['start_wall_ns'] for x in client);end=max(x['end_wall_ns'] for x in client)
    csvout(out/'all-client-clocks.csv',client,list(client[0]))
    dump(out/'clock-anchors.json',dict(anchors=anchors,offset_median_ns=offset,offset_spread_ns=max(offsets)-min(offsets),note='Clock offset is measured; native marker must also fall within each host call bracket. No NTP stability assumption is used.'))
    dump(out/'memory-snapshots.json',snapshots)
    work=collections.Counter();work_by_rank=collections.defaultdict(collections.Counter)
    for x in spec:
        w=x['work'];work['calls']+=1;rank=x['identity']['dp_rank'];work_by_rank[rank]['calls']+=1
        for key in ['num_draft_tokens','num_accepted_tokens','num_invalid_spec_tokens_for_request','num_generated_tokens','num_verified_tokens']:
            value=w.get(key)
            if isinstance(value,int):work[key]+=value;work_by_rank[rank][key]+=value
    hook=collections.defaultdict(lambda:dict(calls=0,inclusive_wall_ns=0,thread_cpu_ns=0))
    for x in spans:
        if x.get('monotonic_end_ns',0)<=start-offset or x.get('monotonic_ns',0)>=end-offset:continue
        key=(x.get('dp_rank'),x['source']);g=hook[key];g['calls']+=1;g['inclusive_wall_ns']+=x.get('duration_ns',0);g['thread_cpu_ns']+=x.get('thread_cpu_delta_ns',0)
    hooks=[dict(rank=k[0],source=k[1],**v) for k,v in sorted(hook.items(),key=lambda kv:(str(kv[0][0]),kv[0][1]))]
    csvout(out/'all-host-hook-spans.csv',hooks,['rank','source','calls','inclusive_wall_ns','thread_cpu_ns'])
    ranks=[]
    for p in profiles(ROOT/a.profile_manifest):
        pattern='e33_cohort:'+('hot' if '-HOT-' in a.run_id else 'normal') if a.framework=='di' else None
        result=rank_census(p,a.run_id,start,end,anchors,out,pattern,spans,a.framework);ranks.append(result)
        dump(out/('rank%02d-summary.json'%result['rank']),result)
        print(json.dumps(dict(rank=result['rank'],kernel_records=result['observed_kernel_records'],checks=result['checks'])),flush=True)
    checks=dict(eight_distinct_devices=len({x['device'] for x in ranks})==8,rank_set={x['rank'] for x in ranks}==set(range(8)),all_clock_and_cohort_checks=all(all(x['checks'].values()) for x in ranks))
    result=dict(run_id=a.run_id,framework=a.framework,created_wall_ns=time.time_ns(),client=evidence,client_window_wall_ns=[start,end],client_duration_s=(end-start)/1e9,output_tps=8192/((end-start)/1e9),
                clock_offset_median_ns=offset,clock_offset_spread_ns=max(offsets)-min(offsets),structural_checks=checks,rank_summaries=ranks,
                speculative_work=dict(work),speculative_by_rank={k:dict(v) for k,v in work_by_rank.items()},speculative_counter_scope='rank_iteration_batch_total' if a.framework=='di' else 'per_request_iteration_arguments',
                qualified_full_chain=False,limits=['This is a census and structural coverage audit, not a completion verdict.','API wall times and kernel sums are not additive across threads/ranks or overlap.','Raw speculative accepted/draft counts can exceed delivered output due to final truncation.','Model-visible allocations require a separate disjoint storage and native-buffer ledger.','No explicit dropped-record counters were available in this database schema.'])
    dump(out/'summary.json',result)
    if not a.census_only:assert all(checks.values())
    print(json.dumps({k:result[k] for k in ['run_id','client_duration_s','output_tps','structural_checks','speculative_work']}),flush=True)
if __name__=='__main__':main()
