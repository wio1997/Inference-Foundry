"""Cross-layer function ledger from FT spans and existing torch-NPU exports.

Runs on the remote CPU only. Never initializes a device or imports a framework.
"""
from pathlib import Path
from collections import defaultdict,Counter
import bisect,concurrent.futures,csv,heapq,json,re,statistics,sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from analyze_scopes import merge,intersection,subtract,duration,span
MACRO={'prepare input','preprocess','forward','post process','sample_token','draft_token','async_state_update'}

def leaf_segments(events):
    edges=[]
    for i,e in enumerate(events):
        a,b=span(e);edges.extend([(a,1,i),(b,-1,i)])
    edges.sort();active=set();heap=[];last=None;pos=0;out=defaultdict(list)
    while pos<len(edges):
        t=edges[pos][0]
        while heap and heap[0][2] not in active:heapq.heappop(heap)
        if last is not None and t>last and heap:out[heap[0][2]].append([last,t])
        while pos<len(edges) and edges[pos][0]==t:
            _,start,i=edges[pos];pos+=1
            if start==1:
                active.add(i);a,b=span(events[i]);heapq.heappush(heap,(b-a,-a,i))
            else:active.discard(i)
        last=t
    return out

def analyze_trace(path):
    raw=json.load(open(path));events=raw['traceEvents'] if isinstance(raw,dict) else raw;ep=int(re.search(r'_ep(\d+)_',str(path))[1])
    proc={e['pid']:e.get('args',{}).get('name') for e in events if e.get('ph')=='M' and e.get('name')=='process_name'}
    py={p for p,n in proc.items() if n=='Python'}
    main=[e for e in events if e.get('ph')=='X' and e.get('dur') and e.get('pid') in py and e.get('tid')==e.get('pid')]
    outer_ft=[e for e in main if e.get('name','').startswith('FT::')]
    assert outer_ft,('no fine function marks',ep)
    log_by=defaultdict(list)
    root=path.parents[3]
    for pid in py:
        log=root/'timing'/('spans-'+str(pid)+'.jsonl')
        if not log.exists():continue
        for line in log.read_text().splitlines():
            r=json.loads(line)
            if r.get('kind')!='adapter' and r.get('mode')=='detailed':log_by[r['name']].append(r)
    for rows in log_by.values():rows.sort(key=lambda r:r['wall_ns'])
    # Exported profiler clock can slew relative to CLOCK_REALTIME. Calibrate
    # with long, infrequent ranges whose body must lie inside the FT range.
    # Solve affine offset inequalities, rather than assume equal timestamps.
    anchor_suffixes=('_prepare_inputs','_build_attention_metadata','_model_forward','sample_tokens','execute_model')
    anchors=[];origin=min(float(e['ts']) for e in outer_ft)
    for e in outer_ft:
        if not e['name'].endswith(anchor_suffixes):continue
        rows=log_by[e['name'][4:]]
        if not rows:continue
        a,b=span(e);r=min(rows,key=lambda r:abs(r['wall_ns']/1000-a));x=r['wall_ns']/1000;y=x+r['duration_ns']/1000
        if abs(x-a)<2000:anchors.append(((a-origin)/1e6,x-a,(b-origin)/1e6,y-b))
    assert len(anchors)>=20,('too few clock anchors',ep,len(anchors))
    fits=[]
    for slope in range(-1000,1001):
        lower=max(end_delta-slope*end_t for start_t,start_delta,end_t,end_delta in anchors)
        upper=min(start_delta-slope*start_t for start_t,start_delta,end_t,end_delta in anchors)
        fits.append((upper-lower,slope,lower,upper))
    width,slope,lower,upper=max(fits)
    # Sub-microsecond quantization and clock-read placement can make the
    # enclosures miss by a few us. Keep that required residual explicit.
    residual=max(0,-width/2)
    assert residual<=10,('clock alignment residual exceeds 10us',ep,residual)
    lower-=residual+1;upper+=residual+1
    width=upper-lower
    offset=(lower+upper)/2
    def mapped(wall):return origin+(wall-origin-offset)/(1+slope/1e6)
    log_starts={n:[mapped(r['wall_ns']/1000) for r in rows] for n,rows in log_by.items()}
    ft=[];marker_boundaries=[];unmatched=0
    for event in outer_ft:
        name=event['name'][4:];a,b=span(event);rows=log_by[name]
        start=bisect.bisect_left(log_starts.get(name,[]),a-10)
        choices=[r for r in rows[start:start+4] if mapped(r['wall_ns']/1000)>=a-10 and
                 mapped(r['wall_ns']/1000+r['duration_ns']/1000)<=b+10]
        if not choices:
            unmatched+=1
            continue
        r=min(choices,key=lambda r:abs(mapped(r['wall_ns']/1000)-a));x=mapped(r['wall_ns']/1000);y=mapped(r['wall_ns']/1000+r['duration_ns']/1000)
        ft.append(dict(event,ts=x,dur=y-x))
        marker_boundaries.extend(subtract([[a,b]],[[x,y]]))
    assert len(ft)>len(outer_ft)*.99,('FT/body clock matching failed',ep,len(ft),len(outer_ft))
    marker_union=merge(marker_boundaries)
    graphs=sorted(float(e['ts']) for e in events if e.get('name')=='MODEL_EXECUTE' and e.get('ph')=='X')
    assert len(graphs)>=2,(ep,graphs)
    window=[[graphs[0],graphs[-1]]]
    activity={k:merge([span(e) for e in events if e.get('name')==k and e.get('ph')=='X' and e.get('dur')]) for k in ['Free','Computing','Communication(Not Overlapped)']}
    free=intersection(activity['Free'],window)
    prepare=intersection(merge([span(e) for e in main if e.get('name')=='prepare input']),window)
    prep_free=intersection(prepare,free)
    cpu=[e for e in main if e.get('cat')=='cpu_op' and not e['name'].startswith('FT::') and e['name'] not in MACRO]
    cpu_union=merge([span(e) for e in cpu])
    no_cpu=subtract(prep_free,cpu_union)
    application_prep_free=subtract(prep_free,marker_union)
    application_no_cpu=subtract(no_cpu,marker_union)
    leaves=leaf_segments(ft);grouped=defaultdict(list);selfs=defaultdict(list);fn_cpu=defaultdict(Counter)
    no_cpu_functions=defaultdict(list);self_functions=defaultdict(list)
    for i,pieces in leaves.items():
        name=ft[i]['name'][4:];selfs[name].extend(pieces)
    # Group by function before interval intersections. Intersecting each tiny
    # call with the whole fragmented idle set is quadratic in call count.
    for name,pieces in selfs.items():
        own=subtract(merge(pieces),marker_union)
        selfs[name]=own
        no_cpu_functions[name]=intersection(application_no_cpu,own)
        self_functions[name]=intersection(application_prep_free,own)
    for e in ft:grouped[e['name'][4:]].append(e)
    rows=[]
    for name,es in grouped.items():
        inc=subtract(merge([span(e) for e in es]),marker_union);own=merge(selfs[name]);calls=[e for e in es if window[0][0]<=float(e['ts'])<window[0][1]]
        rows.append({'name':name,'calls':len(calls),'inclusive_wall_ms':duration(intersection(inc,window)),
          'self_wall_ms':duration(intersection(own,window)),'inclusive_free_ms':duration(intersection(inc,free)),
          'self_free_ms':duration(intersection(own,free)),'prepare_self_free_ms':duration(self_functions[name]),
          'prepare_without_cpu_op_ms':duration(no_cpu_functions[name]),
          'call_median_ms':statistics.median([float(e['dur'])/1000 for e in calls]) if calls else None,
          'call_p90_ms':sorted(float(e['dur'])/1000 for e in calls)[min(len(calls)-1,int(len(calls)*.9))] if calls else None})
    fn_parts={name:duration(merge(v)) for name,v in self_functions.items() if v}
    covered=merge([s for v in self_functions.values() for s in v]);fn_parts['unmarked_prepare_self']=duration(subtract(application_prep_free,covered))
    fn_parts['added_marker_boundary']=duration(intersection(prep_free,marker_union))
    no_parts={name:duration(merge(v)) for name,v in no_cpu_functions.items() if v}
    covered=merge([s for v in no_cpu_functions.values() for s in v]);no_parts['unmarked_prepare_self']=duration(subtract(application_no_cpu,covered))
    no_parts['added_marker_boundary']=duration(intersection(no_cpu,marker_union))
    assert abs(sum(fn_parts.values())-duration(prep_free))<1e-5
    assert abs(sum(no_parts.values())-duration(no_cpu))<1e-5
    # Cross operation type with the innermost marked function. Large unmarked
    # Python/runtime portions remain a named function's self time, not a claim
    # that all time is executing Python instructions.
    cpuby=defaultdict(list)
    for e in cpu:cpuby[e['name']].append(span(e))
    op_rows=[]
    for name,ivs in cpuby.items():
        part=intersection(merge(ivs),application_prep_free)
        if not part:continue
        owners=[]
        for fn,own in selfs.items():
            value=duration(intersection(part,merge(own)))
            if value:owners.append({'function':fn,'ms':value})
        op_rows.append({'operation':name,'prepare_free_inclusive_ms':duration(part),'functions':sorted(owners,key=lambda r:-r['ms'])[:8]})
    return {'ep':ep,'dp':ep//8,'python_pids':sorted(py,key=str),'closed_cycles':len(graphs)-1,'window_start_us':window[0][0],'window_end_us':window[0][1],
      'window_ms':duration(window),'activity_ms':{k:duration(intersection(v,window)) for k,v in activity.items()},
      'prepare_free_ms':duration(prep_free),'prepare_without_cpu_op_ms':duration(no_cpu),'prepare_free_function_partition_ms':fn_parts,
      'added_marker_free_ms':duration(intersection(free,marker_union)),
      'added_marker_prepare_free_ms':duration(intersection(prep_free,marker_union)),
      'body_marker_matched':len(ft),'body_marker_unmatched':unmatched,
      'clock_alignment':{'method':'affine enclosure constraints; midpoint of feasible offset interval','anchors':len(anchors),
                         'required_enclosure_residual_us':residual,'slope_ppm':slope,'offset_us':offset,'feasible_offset_width_us':width,'origin_us':origin},
      'prepare_no_cpu_function_partition_ms':no_parts,'functions':sorted(rows,key=lambda r:-r['prepare_self_free_ms']),
      'cpu_operations':sorted(op_rows,key=lambda r:-r['prepare_free_inclusive_ms']), 'function_marker_count':len(ft)}

def aggregate(ranks):
    n=len(ranks);names=sorted({r['name'] for rank in ranks for r in rank['functions']});rows=[]
    fields=['calls','inclusive_wall_ms','self_wall_ms','inclusive_free_ms','self_free_ms','prepare_self_free_ms','prepare_without_cpu_op_ms',
            'cpu_fully_enclosed_calls','thread_cpu_ms','cpu_observed_wall_ms']
    for name in names:
        samples=[next((r for r in rank['functions'] if r['name']==name),{}) for rank in ranks]
        out={'name':name,**{f:sum(s.get(f,0) for s in samples)/n for f in fields}}
        out['call_median_ms']=statistics.median(s['call_median_ms'] for s in samples if s.get('call_median_ms') is not None)
        out['prepare_free_pct']=out['prepare_self_free_ms']/(sum(r['prepare_free_ms'] for r in ranks)/n)*100
        out['thread_cpu_fraction']=out['thread_cpu_ms']/out['cpu_observed_wall_ms'] if out['cpu_observed_wall_ms'] else None
        rows.append(out)
    return {'ranks':n,'window_ms':statistics.mean(r['window_ms'] for r in ranks),
      'activity_ms':{k:statistics.mean(r['activity_ms'][k] for r in ranks) for k in ranks[0]['activity_ms']},
      'prepare_free_ms':statistics.mean(r['prepare_free_ms'] for r in ranks),
      'prepare_without_cpu_op_ms':statistics.mean(r['prepare_without_cpu_op_ms'] for r in ranks),
      'added_marker_free_ms':statistics.mean(r['added_marker_free_ms'] for r in ranks),
      'added_marker_prepare_free_ms':statistics.mean(r['added_marker_prepare_free_ms'] for r in ranks),
      'functions':sorted(rows,key=lambda r:-r['prepare_self_free_ms'])}

def analyze_logs(root):
    processes=[];catalog={}
    for path in sorted((root/'timing').glob('spans-*.jsonl')):
        samples=[]
        for line in path.read_text().splitlines():
            r=json.loads(line)
            if r.get('kind')=='adapter':
                for f in r['functions']:catalog[f['name']]={**f,'path':r['path'],'sha256':r['sha256']}
            else:samples.append(r)
        if not samples:continue
        lookup={r['id']:r for r in samples};children=defaultdict(list)
        for r in samples:
            if r['parent'] is not None:children[r['parent']].append(r)
        groups=defaultdict(list)
        for r in samples:
            own=r['duration_ns']-sum(c['duration_ns'] for c in children[r['id']])
            owncpu=r['thread_cpu_ns']-sum(c['thread_cpu_ns'] for c in children[r['id']])
            assert own>=0,(path,r)
            r['self_wall_ns']=own;r['self_thread_cpu_ns']=owncpu
            groups[(r['label'],r['name'],r['mode'])].append(r)
        rows=[]
        for (label,name,mode),ss in groups.items():
            out={'label':label,'name':name,'mode':mode,'calls':len(ss)}
            for key in ['duration_ns','thread_cpu_ns','self_wall_ns','self_thread_cpu_ns']:
                out[key.replace('_ns','_ms')]=sum(r[key] for r in ss)/1e6
            out['call_median_ms']=statistics.median(r['duration_ns'] for r in ss)/1e6
            out['call_p90_ms']=sorted(r['duration_ns'] for r in ss)[min(len(ss)-1,int(len(ss)*.9))]/1e6
            rows.append(out)
        processes.append({'pid':samples[0]['pid'],'thread_ids':sorted({r['tid'] for r in samples}),
                          'wall_start_ns':min(r['wall_ns'] for r in samples),'wall_end_ns':max(r['wall_ns']+r['duration_ns'] for r in samples),
                          'rows':sorted(rows,key=lambda r:(r['label'],-r['duration_ms']))})
    return {'processes':processes,'catalog':catalog}

if __name__=='__main__':
    root=Path(sys.argv[1]);paths=sorted((root/'traces').rglob('trace_view.json'));assert len(paths)==16,len(paths)
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:ranks=list(pool.map(analyze_trace,paths))
    logs=analyze_logs(root)
    # epoch-us exported by torch-NPU and time.time_ns share the host clock.
    # CPU time cannot be prorated across a cut: retain fully enclosed calls.
    for rank in ranks:
        pids={int(p) for p in rank['python_pids']}
        clock=rank['clock_alignment']
        def mapped(wall):return clock['origin_us']+(wall-clock['origin_us']-clock['offset_us'])/(1+clock['slope_ppm']/1e6)
        ss=[]
        for pid in pids:
            path=root/'timing'/('spans-'+str(pid)+'.jsonl')
            if not path.exists():continue
            for line in path.read_text().splitlines():
                row=json.loads(line)
                if row.get('kind')=='adapter' or row.get('mode')!='detailed':continue
                a=mapped(row['wall_ns']/1000);b=mapped(row['wall_ns']/1000+row['duration_ns']/1000)
                if a>=rank['window_start_us'] and b<=rank['window_end_us']:ss.append(row)
        by=defaultdict(list)
        for row in ss:by[row['name']].append(row)
        for row in rank['functions']:
            samples=by[row['name']]
            row['cpu_fully_enclosed_calls']=len(samples)
            row['thread_cpu_ms']=sum(s['thread_cpu_ns'] for s in samples)/1e6
            row['cpu_observed_wall_ms']=sum(s['duration_ns'] for s in samples)/1e6
            row['thread_cpu_fraction']=row['thread_cpu_ms']/row['cpu_observed_wall_ms'] if row['cpu_observed_wall_ms'] else None
    out={'schema':1,'kind':'fresh version-pinned function marks; additive self/free accounting; not an optimization benchmark',
         'groups':{'all':aggregate(ranks),'dp0':aggregate([r for r in ranks if r['dp']==0]),'dp1':aggregate([r for r in ranks if r['dp']==1])},'ranks':ranks,'host_logs':logs}
    (root/'function-analysis.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
    print(json.dumps({'result':'PASS','ranks':len(ranks),'logged_processes':len(logs['processes']),'functions':len(logs['catalog']),'all':out['groups']['all']['functions'][:8]},ensure_ascii=False))
