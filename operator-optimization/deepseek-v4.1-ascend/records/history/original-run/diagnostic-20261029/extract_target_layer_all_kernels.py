#!/usr/bin/env python3
"""CPU only. Stream one rank at a time; stdout is JSON Lines without prompt content.
Wrapper injects HIER_SAMPLES and HIER_SHA from the fixed local hierarchy file.
"""
import bisect,collections,csv,hashlib,io,json,os,re,stat
from decimal import Decimal,InvalidOperation
from pathlib import Path
import ijson

BASE=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime')
ROOT=BASE/'diagnostic-traces'
RANK_RE=re.compile(r'^dp[01]_pp\d+_tp\d+_dcp\d+_ep(\d+)_rank\d+_(\d+)_\d+_ascend_pt$')
SCOPE_RE=re.compile(r'^wio_diag:DeepseekV41DecoderLayer\.forward:(\d+-\d+)$')
DIGEST_RE=re.compile(r'^[0-9a-f]{64}$')
MAX_TRACE=2_000_000_000;MAX_META=30_000_000;MAX_CSV=50_000_000
MAX_SCOPES=10000;MAX_FLOW_IDS=2_000_000;MAX_DEVICE_EVENTS=500_000
FIELDS=('Name','Type','Model ID','Stream ID','Task ID','Start Time(us)','Duration(us)',
        'Input Shapes','Output Shapes','Input Data Types','Output Data Types','Input Formats','Output Formats',
        'Accelerator Core','Block Num','Mix Block Num','aic_total_cycles','aic_mac_ratio','aiv_total_cycles','aiv_vec_ratio')

def safe_file(p,limit):
    if p.resolve(strict=True)!=p:raise RuntimeError('symlink file or ancestor: '+str(p))
    s=p.stat()
    if not stat.S_ISREG(s.st_mode) or not 0<s.st_size<=limit:raise RuntimeError('file size/type: '+str(p))
    return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)

def file_sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
    return h.hexdigest()

def dec(v):
    try:
        d=Decimal(str(v).strip())
        return d if d.is_finite() else None
    except (InvalidOperation,TypeError,ValueError):return None

def numeric_proxy(cycle,ratio):
    c,r=dec(cycle),dec(ratio)
    if c is None or c<=0 or r is None or r<0 or r>1:return None
    return str(c*r)

def events(path):
    with path.open('rb') as f:
        prefix=f.read(1024).lstrip()
    if prefix.startswith(b'['):selector='item'
    elif prefix.startswith(b'{'):selector='traceEvents.item'
    else:raise RuntimeError('unknown JSON top-level')
    with path.open('rb') as f:yield from ijson.items(f,selector)

def enclosing(index,ts):
    starts,scopes,max_end=index
    i=bisect.bisect_right(starts,ts)-1
    candidates=[]
    while i>=0 and max_end[i]>ts:
        item=scopes[i]
        if item['start']<=ts<item['end']:candidates.append(item)
        i-=1
    if not candidates:return None,'outside_layer'
    inner=[x for x in candidates if all(y['start']<=x['start'] and x['end']<=y['end'] for y in candidates)]
    if len(inner)!=1:return None,'layer_overlap_ambiguous'
    return inner[0],None

def process(rank,ep,pid,samples):
    result={'rank':rank.name,'ep':ep,'pid':pid,'status':'unknown','counts':{},'layers':[]}
    count=collections.Counter(); trace=rank/'ASCEND_PROFILER_OUTPUT/trace_view.json';csvpath=rank/'ASCEND_PROFILER_OUTPUT/kernel_details.csv';meta=BASE/'metadata'/('events-%s.jsonl'%pid)
    files={'trace':trace,'kernel_csv':csvpath,'metadata':meta}
    before={k:safe_file(p,{'trace':MAX_TRACE,'kernel_csv':MAX_CSV,'metadata':MAX_META}[k]) for k,p in files.items()}
    expected_trace=TRACE_EXPECTED.get(rank.name)
    if not isinstance(expected_trace,dict) or before['trace'][2]!=expected_trace['bytes']:raise RuntimeError('trace size or previous source missing')
    # Hierarchy derives conditional_target classification and digest from paired sidecar scopes.
    sid={}
    for item in samples:
        cid=item.get('call_id')
        if not isinstance(cid,str) or cid in sid:raise RuntimeError('duplicate hierarchy call ID')
        sid[cid]=item
    if len(sid)>MAX_SCOPES:raise RuntimeError('hierarchy scope limit')
    # Validate sidecar call/label/layer for targeted scopes, without outputting request content.
    pairs=collections.defaultdict(lambda:collections.Counter()); sidecar_threads=collections.defaultdict(lambda:collections.defaultdict(list))
    with meta.open(encoding='utf-8') as f:
        for line in f:
            if len(line)>1_000_000:raise RuntimeError('metadata line limit')
            e=json.loads(line);cid=e.get('call_id');kind=e.get('kind')
            if cid not in sid or kind not in ('call_begin','call_end'):continue
            name=e.get('label'); ok=(name=='DeepseekV41DecoderLayer.forward' and e.get('pid')==pid)
            if kind=='call_begin':ok=ok and e.get('layer_index')==sid[cid].get('layer_index')
            pairs[cid][kind+('_ok' if ok else '_invalid')]+=1
            if ok:sidecar_threads[cid][kind].append(str(e.get('tid')))
    eligible={}
    for cid,x in sid.items():
        if x.get('role')!='conditional_target':continue
        digest=x.get('parent_digest'); layer=x.get('layer_index')
        if x.get('unknown_reasons') or not isinstance(digest,str) or not DIGEST_RE.fullmatch(digest) or type(layer)!=int or not 0<=layer<40:
            count['sidecar_target_invalid']+=1;continue
        z=pairs[cid]
        if z.get('call_begin_ok')!=1 or z.get('call_end_ok')!=1 or sum(z.values())!=2 or sidecar_threads[cid]['call_begin']!=sidecar_threads[cid]['call_end']:
            count['sidecar_target_unpaired_or_ambiguous']+=1;continue
        eligible[cid]=x
    if len(eligible)!=80:count['expected_80_target_scopes_mismatch']+=1
    # Pass 1: collect all decoder layer scopes, including draft and unknown scopes,
    # so selecting the innermost scope cannot accidentally swallow draft work.
    all_scopes=[];seen=collections.Counter()
    for e in events(trace):
        if not isinstance(e,dict) or e.get('ph')!='X' or str(e.get('pid'))!=str(pid):continue
        name=e.get('name');m=SCOPE_RE.fullmatch(name) if isinstance(name,str) else None
        if not m:continue
        ts,dur=dec(e.get('ts')),dec(e.get('dur'))
        if ts is None or dur is None or dur<=0:count['trace_layer_invalid_time']+=1;continue
        cid=m.group(1);seen[cid]+=1
        all_scopes.append({'id':cid,'pid':str(e.get('pid')),'tid':str(e.get('tid')),'start':ts,'end':ts+dur,'role':sid.get(cid,{}).get('role','unknown')})
        if len(all_scopes)>MAX_SCOPES:raise RuntimeError('trace scope limit')
    scopes_by_thread=collections.defaultdict(list);by_id=collections.defaultdict(list)
    for x in all_scopes:
        scopes_by_thread[(x['pid'],x['tid'])].append(x);by_id[x['id']].append(x)
    scope_indices={}
    for thread,items in scopes_by_thread.items():
        items.sort(key=lambda x:(x['start'],x['end'],x['id']))
        max_end=[]
        for x in items:max_end.append(max(x['end'],max_end[-1]) if max_end else x['end'])
        scope_indices[thread]=([x['start'] for x in items],items,max_end)
    for cid in eligible:
        if seen[cid]==0:count['target_scope_absent_trace']+=1
        elif seen[cid]>1:count['target_scope_duplicate_trace']+=1
        elif by_id[cid][0]['tid']!=sidecar_threads[cid]['call_begin'][0]:count['target_scope_thread_mismatch']+=1
    # Even absent tasks get a layer output with UNKNOWN, never a zero-work claim.
    layerout={}
    for cid,x in eligible.items():
        layerout[cid]={'rank':rank.name,'ep':ep,'worker_pid':pid,'layer_scope_id':cid,'layer':x['layer_index'],
                       'scheduler_digest':x['parent_digest'],'trace_scope_count':seen[cid],
                       'sidecar_thread_matches_trace':seen[cid]==1 and by_id[cid][0]['tid']==sidecar_threads[cid]['call_begin'][0],
                       'matched_task_count':0,'tasks':[],'metrics_status':'unknown_no_unique_match'}
    # Pass 2: count all flow-start IDs, select those whose innermost scope is unique target.
    start_counts=collections.Counter();selected=collections.defaultdict(list)
    for e in events(trace):
        if not isinstance(e,dict) or e.get('name')!='torch_to_npu' or e.get('cat')!='async_npu' or e.get('ph')!='s':continue
        if e.get('id') is None:count['flow_start_missing_id']+=1;continue
        flow=str(e['id']);start_counts[flow]+=1
        if len(start_counts)>MAX_FLOW_IDS:raise RuntimeError('flow ID limit')
        ts=dec(e.get('ts'));thread=(str(e.get('pid')),str(e.get('tid')))
        if ts is None:count['flow_start_invalid_ts']+=1;continue
        index=scope_indices.get(thread)
        if not index:continue
        scope,why=enclosing(index,ts)
        if why:
            count[why]+=1
            continue
        cid=scope['id']
        if scope['role']!='conditional_target':
            count['innermost_not_target']+=1;continue
        if cid not in eligible or seen[cid]!=1 or scope['tid']!=sidecar_threads[cid]['call_begin'][0]:
            count['target_flow_invalid_scope']+=1;continue
        selected[flow].append({'scope':cid,'start_us':str(ts),'host_pid':thread[0],'host_tid':thread[1]})
    count['target_selected_start_events']=sum(map(len,selected.values()))
    count['target_selected_start_ids']=len(selected)
    # Pass 3: find finishes for selected IDs. Never infer a match from time proximity.
    finishes=collections.defaultdict(list)
    for e in events(trace):
        if not isinstance(e,dict) or e.get('name')!='torch_to_npu' or e.get('cat')!='async_npu' or e.get('ph')!='f' or e.get('id') is None:continue
        flow=str(e['id'])
        if flow not in selected:continue
        ts=dec(e.get('ts'))
        if ts is None:
            count['flow_finish_invalid_ts']+=1;finishes[flow].append(None)
        else:finishes[flow].append((str(e.get('pid')),str(e.get('tid')),ts))
    wanted={v[0] for k,v in finishes.items() if len(v)==1 and v[0] is not None and start_counts[k]==1 and len(selected[k])==1}
    # Pass 4: inspect every X on a wanted track/time, even non-kernel Xs; any
    # collision is ambiguous and excluded. Thus no silent operator-type filtering.
    device=collections.defaultdict(list)
    for e in events(trace):
        if not isinstance(e,dict) or e.get('ph')!='X':continue
        ts=dec(e.get('ts'))
        if ts is None:continue
        key=(str(e.get('pid')),str(e.get('tid')),ts)
        if key not in wanted:continue
        dur=dec(e.get('dur'));args=e.get('args'); name=e.get('name')
        event={'duration_us':str(dur) if dur is not None else None,'name':name,
               'model':None,'task':None,'stream':None}
        if isinstance(args,dict):
            for attr,keyname in [('model','Model Id'),('task','Task Id'),('stream','Physic Stream Id')]:
                if args.get(keyname) is not None:event[attr]=str(args[keyname])
        device[key].append(event)
    # Only retain CSV keys that could join an unambiguous X, preserving duplicates.
    candidate_keys=set()
    for track,es in device.items():
        if len(es)!=1:continue
        e=es[0]
        if isinstance(e['name'],str) and all(e[x] is not None for x in ('model','task','stream')):
            candidate_keys.add((track[2],e['model'],e['task'],e['stream']))
    csvrows=collections.defaultdict(list)
    with csvpath.open(newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames is None:raise RuntimeError('missing CSV header')
        missing=[k for k in FIELDS if k not in reader.fieldnames]
        result['csv_missing_columns']=missing
        for line,row in enumerate(reader,2):
            ts=dec(row.get('Start Time(us)'))
            if ts is None:continue
            key=(ts,str(row.get('Model ID')),str(row.get('Task ID')),str(row.get('Stream ID')))
            if key in candidate_keys:csvrows[key].append((line,{k:row.get(k) for k in FIELDS}))
    used=collections.Counter(); operator_coverage=collections.defaultdict(collections.Counter)
    for flow,starts in selected.items():
        if start_counts[flow]!=1 or len(starts)!=1:
            count['flow_start_nonunique']+=1;continue
        fin=finishes.get(flow,[])
        if len(fin)!=1:
            count['flow_finish_missing' if not fin else 'flow_finish_nonunique']+=1;continue
        if fin[0] is None:
            count['flow_finish_invalid_only']+=1;continue
        evs=device.get(fin[0],[])
        if len(evs)!=1:
            count['device_x_missing' if not evs else 'device_x_nonunique']+=1;continue
        e=evs[0]; op=e.get('name') if isinstance(e.get('name'),str) else '<unknown>'
        operator_coverage[op]['unique_device_x_candidates']+=1
        if not isinstance(e['name'],str) or any(e[x] is None for x in ('model','task','stream')):
            count['device_x_missing_name_or_task_identity']+=1;operator_coverage[op]['missing_name_or_identity']+=1;continue
        key=(fin[0][2],e['model'],e['task'],e['stream']);matches=csvrows.get(key,[])
        if len(matches)!=1:
            reason='csv_missing' if not matches else 'csv_nonunique'
            count[reason]+=1;operator_coverage[op][reason]+=1;continue
        line,row=matches[0]
        if row.get('Name')!=e['name']:
            count['csv_trace_name_mismatch']+=1;operator_coverage[op]['csv_trace_name_mismatch']+=1;continue
        c=layerout[starts[0]['scope']];used[key]+=1
        duration=dec(row.get('Duration(us)'));trace_dur=dec(e['duration_us'])
        if duration is None or trace_dur is None or duration<=0 or abs(duration-trace_dur)>Decimal('0.001'):
            count['trace_csv_duration_missing_or_mismatch']+=1;operator_coverage[op]['trace_csv_duration_missing_or_mismatch']+=1;continue
        task={'flow_id':flow,'host_start_us':starts[0]['start_us'],'host_pid':starts[0]['host_pid'],
              'host_tid':starts[0]['host_tid'],'device_track_pid':fin[0][0],'device_track_tid':fin[0][1],
              'device_start_us':str(fin[0][2]),'trace_duration_us':e['duration_us'],'csv_line':line,
              'csv':row,'duration_abs_difference_us':str(abs(duration-trace_dur)),
              'aic_mac_cycle_proxy':numeric_proxy(row.get('aic_total_cycles'),row.get('aic_mac_ratio')),
              'aiv_vec_cycle_proxy':numeric_proxy(row.get('aiv_total_cycles'),row.get('aiv_vec_ratio'))}
        c['tasks'].append(task)
    # A device TASK cannot be attributed to multiple scopes/flows.
    for c in layerout.values():
        kept=[]
        for task in c['tasks']:
            x=task['csv'];key=(dec(x.get('Start Time(us)')),str(x.get('Model ID')),str(x.get('Task ID')),str(x.get('Stream ID')))
            if used[key]!=1:
                count['device_task_reused_by_multiple_flows']+=1;operator_coverage[x['Name']]['device_task_reused_by_multiple_flows']+=1;continue
            kept.append(task);operator_coverage[x['Name']]['matched_unique_tasks']+=1
        groups=collections.defaultdict(list)
        shape_fields=('Input Shapes','Output Shapes','Input Data Types','Output Data Types','Input Formats','Output Formats')
        for task in kept:
            row=task['csv'];groups[(row['Name'],tuple(row.get(k) for k in shape_fields))].append(task)
        out=[]
        for (name,shape),tasks in sorted(groups.items(),key=lambda pair:repr(pair[0])):
            tasks.sort(key=lambda x:(dec(x['device_start_us']),x['flow_id']))
            entry={'operator_name':name,'physical_shape_dtype_format':dict(zip(shape_fields,shape)),
                   'task_count':len(tasks),'tasks':tasks,
                   'recorded_duration_sum_us':str(sum((dec(t['csv']['Duration(us)']) for t in tasks),Decimal(0))),
                   'csv_type_counts':dict(collections.Counter(str(t['csv'].get('Type')) for t in tasks))}
            for unit,cycle_key,ratio_key in [('aic','aic_total_cycles','aic_mac_ratio'),('aiv','aiv_total_cycles','aiv_vec_ratio')]:
                stats=collections.Counter();cycle_sum=Decimal(0);eligible_sum=Decimal(0);proxy_sum=Decimal(0)
                for task in tasks:
                    row=task['csv'];cycle=dec(row.get(cycle_key));ratio=dec(row.get(ratio_key))
                    if cycle is None:stats['cycle_missing_or_invalid']+=1;continue
                    if cycle==0:stats['cycle_zero_reported']+=1;continue
                    if cycle<0:stats['cycle_negative_invalid']+=1;continue
                    stats['cycle_positive']+=1;cycle_sum+=cycle
                    if ratio is None:stats['positive_cycle_ratio_missing_or_invalid']+=1;continue
                    if not 0<=ratio<=1:stats['positive_cycle_ratio_out_of_range']+=1;continue
                    stats['cycle_and_ratio_eligible']+=1;eligible_sum+=cycle;proxy_sum+=cycle*ratio
                entry[unit]={'counts':dict(stats),'positive_cycle_sum':str(cycle_sum) if stats['cycle_positive'] else None,
                    'ratio_eligible_cycle_sum':str(eligible_sum) if stats['cycle_and_ratio_eligible'] else None,
                    'ratio_eligible_cycle_times_ratio_sum':str(proxy_sum) if stats['cycle_and_ratio_eligible'] else None,
                    'ratio_weighted_proxy':str(proxy_sum/eligible_sum) if eligible_sum>0 else None}
            out.append(entry)
        c['tasks']=[] # Tasks are grouped below; avoid redundant copies.
        c['operator_groups']=out;c['matched_task_count']=len(kept)
        c['metrics_status']='recorded_task_sums_not_wallclock' if kept else 'unknown_no_unique_match'
    result['layers']=sorted(layerout.values(),key=lambda x:(x['scheduler_digest'],x['layer'],x['layer_scope_id']))
    count['target_scope_count']=len(eligible);count['target_tasks_uniquely_matched']=sum(x['matched_task_count'] for x in result['layers'])
    count['target_scopes_no_verified_task']=sum(x['matched_task_count']==0 for x in result['layers'])
    result['counts']=dict(count)
    result['operator_coverage']={name:dict(c) for name,c in sorted(operator_coverage.items())}
    result['input_sha256']={k:file_sha(p) for k,p in files.items()}
    if any(safe_file(p,{'trace':MAX_TRACE,'kernel_csv':MAX_CSV,'metadata':MAX_META}[k])!=before[k] for k,p in files.items()):raise RuntimeError('input changed while reading')
    if result['input_sha256']['trace']!=expected_trace['sha256']:raise RuntimeError('trace differs from prior scope extraction')
    result['status']='completed'
    return result

if not isinstance(HIER_SAMPLES,list):raise RuntimeError('missing hierarchy payload')
by_rank=collections.defaultdict(list)
for x in HIER_SAMPLES:
    if not isinstance(x,dict) or type(x.get('ep'))!=int or type(x.get('pid'))!=int:raise RuntimeError('bad hierarchy record')
    by_rank[(x['ep'],x['pid'])].append(x)
print(json.dumps({'kind':'header','scope':'same-thread innermost target layer to unique torch_to_npu flow to unique X to exact kernel CSV',
    'hierarchy_sha256':HIER_SHA,
    'limitations':['profiled prefill only; different digests/ranks/shapes are not equal-work',
        'sidecar target role/digest conditional; track timestamps are exported clocks',
        'flow correlation is not proof of tensor dependency or request critical path',
        'sums may overlap and are not wallclock, device utilization, physical upper bounds, or throughput',
        'ratio-times-cycle is a recorded proxy; zero ratio is not proof of zero hardware activity',
        'missing and zero cycles are separately counted; no match is not zero work',
        'physical input shapes do not establish effective tokens or expert received M; same physical shape does not prove equal work',
        'trace SHA is checked against prior scope summary; CSV and metadata hashes first observed now'],
    'no_device_work':True},ensure_ascii=False),flush=True)
errors=[];seen_ep=set();completed=0
for d in sorted(ROOT.iterdir()):
    match=RANK_RE.fullmatch(d.name)
    if not match:continue
    ep,pid=map(int,match.groups())
    if ep in seen_ep:errors.append({'ep':ep,'reason':'duplicate_ep_rank_directory'});continue
    seen_ep.add(ep)
    try:
        rank_result=process(d,ep,pid,by_rank.get((ep,pid),[]));completed+=1
        print(json.dumps({'kind':'rank','data':rank_result},ensure_ascii=False,separators=(',',':')),flush=True)
    except Exception as exc:
        errors.append({'ep':ep,'reason':type(exc).__name__,'detail':str(exc)[:250]})
expected_ep={x['ep'] for x in HIER_SAMPLES if x.get('role')=='conditional_target'}
print(json.dumps({'kind':'footer','rank_errors':errors,'rank_dirs_seen':len(seen_ep),'completed_ranks':completed,
    'missing_expected_ep':sorted(expected_ep-seen_ep),'unexpected_ep':sorted(seen_ep-expected_ep)},ensure_ascii=False),flush=True)
