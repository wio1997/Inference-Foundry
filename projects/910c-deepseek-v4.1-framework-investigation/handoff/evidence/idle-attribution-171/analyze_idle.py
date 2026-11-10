"""Bounded CPU-only idle analysis of the 16 existing steady worker traces.

Location of idle time and concurrent host activity are separate ledgers.
No overlap bucket is promoted to a causal saving estimate.
"""
from pathlib import Path
from collections import Counter,defaultdict
import bisect,concurrent.futures,json,statistics,sys

def merge(xs):
    out=[]
    for a,b in sorted(xs):
        if b<=a:continue
        if out and a<=out[-1][1]:out[-1][1]=max(out[-1][1],b)
        else:out.append([a,b])
    return out

def intersect(a,b):
    out=[];i=j=0
    while i<len(a) and j<len(b):
        x=max(a[i][0],b[j][0]);y=min(a[i][1],b[j][1])
        if y>x:out.append([x,y])
        if a[i][1]<b[j][1]:i+=1
        else:j+=1
    return out

def subtract(a,b):
    out=[];j=0
    for x,y in a:
        while j<len(b) and b[j][1]<=x:j+=1
        k=j
        while k<len(b) and b[k][0]<y:
            if b[k][0]>x:out.append([x,min(y,b[k][0])])
            x=max(x,b[k][1])
            if x>=y:break
            k+=1
        if x<y:out.append([x,y])
    return out

def ms(xs):return sum(b-a for a,b in xs)/1000
def interval(e):return [float(e['ts']),float(e['ts'])+float(e['dur'])]
def short(e):return {k:e.get(k) for k in ['name','ts','dur','pid','tid','cat','args']}

def analyze(row):
    raw=json.load(open(row['file']));events=raw['traceEvents'] if isinstance(raw,dict) else raw
    proc={e['pid']:e.get('args',{}).get('name') for e in events if e.get('ph')=='M' and e.get('name')=='process_name'}
    py={p for p,n in proc.items() if n=='Python'};hw={p for p,n in proc.items() if n=='Ascend Hardware'};cann={p for p,n in proc.items() if n=='CANN'}
    xs=defaultdict(list);cpu=[];kernels=[];apis=[];enq={};deq={};flow_s={};flow_f={};model_start={};model_end={}
    cpu_by_name=defaultdict(list);cann_main_by_name=defaultdict(list)
    compute_types={'AI_CORE','AI_VECTOR_CORE','MIX_AIC','MIX_AIV','AI_CPU'}
    for e in events:
        ph=e.get('ph');cat=e.get('cat');name=e.get('name','');args=e.get('args',{})
        if ph=='s':flow_s[(cat,str(e.get('id')))]=e
        if ph=='f' and e.get('pid') in hw:flow_f.setdefault((e['pid'],e['tid'],float(e.get('ts'))),[]).append(e)
        if ph!='X' or not e.get('dur'):continue
        span=interval(e)
        if name=='Free':xs['free'].append(span)
        if cat=='cpu_op':
            cpu.append(e);cpu_by_name[name].append(span)
            if name=='gloo:all_reduce':xs['gloo'].append(span)
            if e.get('tid')==e.get('pid') and e.get('pid') in py:
                if name=='Event::synchronize':key='event_sync'
                elif name.startswith('aten::') or name=='empty_tensor':key='tensor_cpu_ops'
                else:key='other_cpu_ops'
                xs[key].append(span)
        if cat=='enqueue':enq[args.get('correlation_id')]=e
        if cat=='dequeue':deq[args.get('correlation_id')]=e;xs['backend_dequeue'].append(span)
        if e.get('pid') in cann:
            apis.append(e)
            if e.get('tid') in py:
                cann_main_by_name[name].append(span)
                if name=='AscendCL@aclmdlRIExecuteAsync':xs['graph_launch_api'].append(span)
                elif name.startswith('AscendCL@aclrtSynchronize'):xs['runtime_sync_api'].append(span)
        if e.get('pid') in hw:
            if args.get('Task Type') in compute_types:kernels.append(e)
            connection=args.get('connection_id')
            if name=='MODEL_EXECUTE':model_start[connection]=e
            if name=='MODEL_WAIT_COMPLETE':model_end[connection]=e
    for key in list(xs):xs[key]=merge(xs[key])
    free=xs['free'];graph_spans=[];graph_cases=[]
    for connection,start in model_start.items():
        end=model_end.get(connection)
        if not end:continue
        a=interval(start)[0];b=interval(end)[1]
        graph_spans.append([a,b]);graph_cases.append({'connection_id':connection,'start_us':a,'end_us':b,'model_id':start['args'].get('Model Id'),'free_ms':ms(intersect(free,[[a,b]]))})
    graphs=merge(graph_spans);inside=intersect(free,graphs);outside=subtract(free,graphs)
    assert abs(ms(inside)+ms(outside)-ms(free))<1e-6
    activity={}
    priority=['graph_launch_api','runtime_sync_api','event_sync','gloo','tensor_cpu_ops','other_cpu_ops','backend_dequeue']
    for phase,regions in [('all',free),('inside_graph',inside),('outside_graph',outside)]:
        remaining=regions;ledger={}
        for key in priority:
            part=intersect(remaining,xs.get(key,[]));ledger[key]=ms(part);remaining=subtract(remaining,part)
        ledger['uncovered_by_these_host_events']=ms(remaining)
        assert abs(sum(ledger.values())-ms(regions))<1e-5
        activity[phase]={'total_ms':ms(regions),'exclusive_by_priority_ms':ledger,
                         'independent_overlap_ms':{key:ms(intersect(regions,spans)) for key,spans in xs.items() if key!='free'}}
    # For graph-external gaps only, inspect the next compute task's matched flow.
    # API start/end bounds are observations about submission, not proof of cause.
    kernels.sort(key=lambda e:float(e['ts']));starts=[float(e['ts']) for e in kernels]
    api_at={(e['pid'],e['tid'],float(e['ts'])):e for e in apis}
    links=[];submission=Counter();gap_types=Counter();gap_us=Counter()
    for a,b in outside:
        index=bisect.bisect_left(starts,b-1.0)
        next_task=kernels[index] if index<len(kernels) else None
        if next_task is None or abs(float(next_task['ts'])-b)>3.0:
            submission['unmatched_or_next_activity_not_compute_ms']+=(b-a)/1000;continue
        if next_task.get('args',{}).get('Model Id')!=4294967295:
            submission['next_is_captured_graph_task_ms']+=(b-a)/1000;continue
        finishes=flow_f.get((next_task['pid'],next_task['tid'],float(next_task['ts'])),[])
        sources=[flow_s.get((finish.get('cat'),str(finish.get('id')))) for finish in finishes]
        source=next((v for v in sources if v and v.get('pid') in cann),None)
        if source is None:
            submission['next_eager_task_has_no_host_flow_ms']+=(b-a)/1000;continue
        api=api_at.get((source['pid'],source['tid'],float(source['ts'])))
        if api is None:
            submission['host_flow_has_no_api_duration_ms']+=(b-a)/1000;continue
        sa,se=interval(api)
        pre=max(0,min(sa,b)-a)/1000
        during=max(0,min(se,b)-max(sa,a))/1000
        post=max(0,b-max(se,a))/1000
        submission['before_next_submission_api_ms']+=pre;submission['during_next_submission_api_ms']+=during;submission['after_next_submission_api_return_ms']+=post
        gap_types[next_task['name']]+=1;gap_us[next_task['name']]+=b-a
        links.append({'gap_start_us':a,'gap_end_us':b,'gap_ms':(b-a)/1000,
                      'before_submission_ms':pre,'during_submission_ms':during,'after_submission_ms':post,
                      'next_task':short(next_task),'host_api':short(api)})
    top_cpu=sorted([{'name':name,'calls':len(spans),'inclusive_ms':ms(spans),
                     'free_overlap_ms':ms(intersect(free,merge(spans))),
                     'outside_graph_free_overlap_ms':ms(intersect(outside,merge(spans)))} for name,spans in cpu_by_name.items()],key=lambda x:-x['free_overlap_ms'])[:35]
    top_cann=sorted([{'name':name,'calls':len(spans),'inclusive_ms':ms(spans),
                      'free_overlap_ms':ms(intersect(free,merge(spans))),
                      'outside_graph_free_overlap_ms':ms(intersect(outside,merge(spans)))} for name,spans in cann_main_by_name.items()],key=lambda x:-x['free_overlap_ms'])[:25]
    # Paired queue IDs give queue residence and main-thread parent operators.
    queue_stats=defaultdict(list);queue_free=defaultdict(list);queue_examples=[]
    main_cpu=sorted([e for e in cpu if e['pid']==e['tid']],key=lambda e:(float(e['ts']),-float(e['dur'])))
    stack=[];pos=0
    for _,entry in sorted(enq.items(),key=lambda item:float(item[1]['ts'])):
        a,b=interval(entry);exit=deq.get(entry.get('args',{}).get('correlation_id'))
        if not exit:continue
        da,db=interval(exit);residence=max(0,da-b)
        while pos<len(main_cpu) and float(main_cpu[pos]['ts'])<=a:
            op=main_cpu[pos];oa,ob=interval(op)
            while stack and interval(stack[-1])[1]<=oa:stack.pop()
            stack.append(op);pos+=1
        parents=[op for op in stack if interval(op)[0]<=a and interval(op)[1]>=b]
        parent=parents[-1]['name'] if parents else None
        key=parent or entry['name'];queue_stats[key].append(residence)
        if da>b:queue_free[key].append([b,da])
        if residence>1000 and len(queue_examples)<8:queue_examples.append({'enqueue':short(entry),'dequeue':short(exit),'parent_cpu_op':parent,'residence_us':residence})
    queue_rows=sorted([{'parent_cpu_op_or_enqueue':name,'calls':len(values),
                        'queue_residence_median_us':statistics.median(values),
                        'queue_residence_p90_us':sorted(values)[int(.9*(len(values)-1))],
                        'queue_residence_free_overlap_ms':ms(intersect(free,merge(queue_free[name])))} for name,values in queue_stats.items()],key=lambda x:-x['queue_residence_free_overlap_ms'])[:20]
    return {'ep':row['ep'],'dp':row['dp'],'source':row['file'],'device_window_ms':row['device_window_ms'],
            'free_ms':ms(free),'target_graph_ms':ms(graphs),'inside_graph_free_ms':ms(inside),
            'outside_graph_free_ms':ms(outside),'inside_graph_fraction_of_free_pct':ms(inside)/ms(free)*100,
            'graph_calls':len(graph_cases),'graph_cases':graph_cases,'host_activity':activity,
            'outside_graph_next_submission':dict(submission),'next_eager_task_types':gap_types.most_common(12),
            'top_cpu_during_free':top_cpu,'top_main_cann_during_free':top_cann,
            'queue':{'matched':len(set(enq)&set(deq)),'enqueued':len(enq),'dequeued':len(deq),'top_parents':queue_rows,'long_examples':queue_examples},
            'submission_examples':sorted(links,key=lambda e:-e['gap_ms'])[:10],
            'python_thread_coverage':Counter((e.get('tid') for e in events if e.get('cat')=='python_function'))}

if __name__=='__main__':
    root=Path(sys.argv[1]);analysis=json.load(open(root/'balanced-analysis.json'))
    rows=[r for r in analysis['files'] if r['phase']=='steady' and r['dp'] is not None]
    out={'scope':'existing balanced steady traces; no new model run','ranks':[]}
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        for result in pool.map(analyze,rows):
            out['ranks'].append(result);print('idle_done',result['ep'],round(result['inside_graph_free_ms'],2),round(result['outside_graph_free_ms'],2),flush=True)
    (root/'idle-attribution.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
    print('DONE',len(out['ranks']),flush=True)
