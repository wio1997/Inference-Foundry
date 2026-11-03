#!/usr/bin/env python3
"""CPU only. Stream one rank at a time; stdout contains anonymous aggregates.
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
    # Pass 3: capture matching flow finishes and Dispatch device X candidates;
    # matching is on track+exact Decimal timestamp as in the earlier EP8 check.
    finishes=collections.defaultdict(list);device=collections.defaultdict(list);device_count=0
    for e in events(trace):
        if not isinstance(e,dict):continue
        if e.get('name')=='torch_to_npu' and e.get('cat')=='async_npu' and e.get('ph')=='f' and e.get('id') is not None:
            flow=str(e['id'])
            if flow in selected:
                ts=dec(e.get('ts'))
                if ts is None:count['flow_finish_invalid_ts']+=1
                else:finishes[flow].append((str(e.get('pid')),str(e.get('tid')),ts))
        if e.get('ph')=='X' and e.get('name')=='DispatchFFNCombine':
            ts,dur=dec(e.get('ts')),dec(e.get('dur'));args=e.get('args')
            if ts is None or dur is None or not isinstance(args,dict):count['device_dispatch_invalid']+=1;continue
            key=(str(e.get('pid')),str(e.get('tid')),ts)
            device[key].append({'duration_us':str(dur),'model':str(args.get('Model Id')),
                                'task':str(args.get('Task Id')),'stream':str(args.get('Physic Stream Id'))})
            device_count+=1
            if device_count>MAX_DEVICE_EVENTS:raise RuntimeError('device event limit')
    # Capture only dispatch rows; preserve identical-key multiplicity for fail-closed joins.
    csvrows=collections.defaultdict(list)
    with csvpath.open(newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f)
        missing=[k for k in FIELDS if k not in reader.fieldnames]
        result['csv_missing_columns']=missing
        for line,row in enumerate(reader,2):
            if row.get('Name')!='DispatchFFNCombine' or row.get('Type')!='DispatchFFNCombine':continue
            ts=dec(row.get('Start Time(us)'))
            if ts is None:count['csv_dispatch_invalid_start']+=1;continue
            key=(ts,str(row.get('Model ID')),str(row.get('Task ID')),str(row.get('Stream ID')))
            csvrows[key].append((line,row))
    used=collections.Counter()
    for flow,starts in selected.items():
        if start_counts[flow]!=1 or len(starts)!=1:
            count['flow_start_nonunique']+=1;continue
        fin=finishes.get(flow,[])
        if len(fin)!=1:
            count['flow_finish_missing' if not fin else 'flow_finish_nonunique']+=1;continue
        evs=device.get(fin[0],[])
        if len(evs)!=1:
            count['device_dispatch_missing_or_other_type' if not evs else 'device_dispatch_nonunique']+=1;continue
        e=evs[0];key=(fin[0][2],e['model'],e['task'],e['stream']);matches=csvrows.get(key,[])
        if len(matches)!=1:
            count['csv_missing' if not matches else 'csv_nonunique']+=1;continue
        line,row=matches[0];c=layerout[starts[0]['scope']];used[(key,line)]+=1
        duration=dec(row.get('Duration(us)'));trace_dur=dec(e['duration_us'])
        if duration is None or trace_dur is None or duration<=0 or abs(duration-trace_dur)>Decimal('0.001'):
            count['trace_csv_duration_missing_or_mismatch']+=1;continue
        task={'flow_id':flow,'host_start_us':starts[0]['start_us'],'host_pid':starts[0]['host_pid'],
              'host_tid':starts[0]['host_tid'],'device_track_pid':fin[0][0],'device_track_tid':fin[0][1],
              'device_start_us':str(fin[0][2]),'trace_duration_us':e['duration_us'],'csv_line':line,
              'csv':{k:row.get(k) for k in FIELDS},
              'duration_abs_difference_us':str(abs(duration-trace_dur)) if duration is not None and trace_dur is not None else None,
              'aic_mac_cycle_proxy':numeric_proxy(row.get('aic_total_cycles'),row.get('aic_mac_ratio')),
              'aiv_vec_cycle_proxy':numeric_proxy(row.get('aiv_total_cycles'),row.get('aiv_vec_ratio'))}
        c['tasks'].append(task)
    # A device TASK cannot be claimed by multiple scopes/flows.
    for c in layerout.values():
        kept=[]
        for task in c['tasks']:
            x=task['csv'];key=(dec(x.get('Start Time(us)')),str(x.get('Model ID')),str(x.get('Task ID')),str(x.get('Stream ID')))
            if used[(key,task['csv_line'])]!=1:
                count['device_task_reused_by_multiple_flows']+=1;continue
            kept.append(task)
        c['tasks']=sorted(kept,key=lambda x:(dec(x['device_start_us']),x['flow_id']))
        c['matched_task_count']=len(kept)
        if kept:
            c['metrics_status']='sum_of_recorded_tasks_not_wallclock'
            vals={
              'task_duration_us':[dec(x['csv'].get('Duration(us)')) for x in kept],
              'aic_cycles':[dec(x['csv'].get('aic_total_cycles')) for x in kept],
              'aic_mac_cycle_proxy':[dec(x.get('aic_mac_cycle_proxy')) for x in kept],
              'aiv_cycles':[dec(x['csv'].get('aiv_total_cycles')) for x in kept],
              'aiv_vec_cycle_proxy':[dec(x.get('aiv_vec_cycle_proxy')) for x in kept],
            }
            c['recorded_sums']={k:(str(sum(v,Decimal(0))) if all(x is not None and (x>0 if k in ('task_duration_us','aic_cycles','aiv_cycles') else x>=0) for x in v) else None) for k,v in vals.items()}
    result['layers']=sorted(layerout.values(),key=lambda x:(x['scheduler_digest'],x['layer'],x['layer_scope_id']))
    count['target_scope_count']=len(eligible);count['target_tasks_uniquely_matched']=sum(x['matched_task_count'] for x in result['layers'])
    count['target_scopes_no_verified_task']=sum(x['matched_task_count']==0 for x in result['layers'])
    result['counts']=dict(count)
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
ranks=[];errors=[];seen_ep=set()
for d in sorted(ROOT.iterdir()):
    match=RANK_RE.fullmatch(d.name)
    if not match:continue
    ep,pid=map(int,match.groups())
    if ep in seen_ep:errors.append({'ep':ep,'reason':'duplicate_ep_rank_directory'});continue
    seen_ep.add(ep)
    try:
        ranks.append(process(d,ep,pid,by_rank.get((ep,pid),[])))
    except Exception as exc:
        errors.append({'ep':ep,'reason':type(exc).__name__,'detail':str(exc)[:250]})
expected_ep={x['ep'] for x in HIER_SAMPLES if x.get('role')=='conditional_target'}
missing_ep=sorted(expected_ep-seen_ep)
unexpected_ep=sorted(seen_ep-expected_ep)
print(json.dumps({'scope':'same-thread innermost target layer → torch_to_npu flow → device X → kernel CSV; one rank streamed at a time',
                  'hierarchy_sha256':HIER_SHA,'ranks':ranks,'rank_errors':errors,'rank_dirs_seen':len(seen_ep),
                  'missing_expected_ep':missing_ep,'unexpected_ep':unexpected_ep,
                  'limitations':['profiled prefill candidates only; digest/rank groups are not equal-work or identical requests',
                     'sidecar target labels and schedule digests are conditional; nesting uses exported clocks',
                     'flow is correlation and submit scope; no direct tensor data dependency or request-level critical path',
                     'task intervals/counter sums may overlap; never sum into wallclock',
                     'ratios and products are exported proxies with finite precision; absent fields are unknown',
                     'no matching task is not evidence of zero work; no global lowest ranking or hardware upper bound',
                     'trace SHA is compared with prior scope summary; CSV/metadata hashes first observed at this read, provenance before read not proved'],
                  'no_device_work':True},ensure_ascii=False,indent=2))
