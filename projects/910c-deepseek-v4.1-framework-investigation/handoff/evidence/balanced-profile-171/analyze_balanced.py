"""Per-call host/device linkage plus union-based per-rank/phase summary."""
from pathlib import Path
from collections import defaultdict,Counter
import concurrent.futures,datetime,gzip,json,re,statistics,sys

def merge(intervals):
    result=[]
    for start,end in sorted(intervals):
        if not result or start>result[-1][1]:
            result.append([start,end])
        else:
            result[-1][1]=max(result[-1][1],end)
    return result

def duration(intervals):
    return sum(end-start for start,end in intervals)

def intersection(first,second):
    i=j=0
    value=0
    while i<len(first) and j<len(second):
        value+=max(0,min(first[i][1],second[j][1])-max(first[i][0],second[j][0]))
        if first[i][1]<second[j][1]:
            i+=1
        else:
            j+=1
    return value

def read_one(item):
    filename,phase=item
    path=Path(filename)
    with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as stream:
        value=json.load(stream)
    events=value.get('traceEvents',[]) if isinstance(value,dict) else value
    processes={e['pid']:e.get('args',{}).get('name') for e in events if e.get('ph')=='M' and e.get('name')=='process_name'}
    hardware={pid for pid,name in processes.items() if name=='Ascend Hardware'}
    cann={pid for pid,name in processes.items() if name=='CANN'}
    xs=defaultdict(list)
    counts=Counter()
    cpu_cost=Counter()
    native_cost=Counter()
    native_count=Counter()
    cpu_events=[]
    interesting_kernels=[]
    flow_starts={}
    flow_finishes=[]
    api_connections=defaultdict(list)
    names={'gloo:all_reduce':'gloo','Event::synchronize':'event_sync','Free':'free','Computing':'compute','Communication(Not Overlapped)':'communication'}
    start=None
    end=None
    for event in events:
        ph=event.get('ph')
        if ph in ['s','f']:
            flow_key=(event.get('cat'),str(event.get('id')))
            if ph=='s':
                flow_starts[flow_key]=event
            elif event.get('pid') in hardware:
                flow_finishes.append(event)
            continue
        if ph!='X' or 'ts' not in event or not event.get('dur'):
            continue
        a=float(event['ts']);b=a+float(event['dur'])
        start=a if start is None else min(start,a)
        end=b if end is None else max(end,b)
        name=event.get('name','')
        cat=event.get('cat','')
        if name in names:
            key=names[name]
            xs[key].append((a,b));counts[key]+=1
        if cat=='cpu_op':
            cpu_cost[name]+=b-a
            if name.startswith('vllm::'):
                cpu_events.append((a,b,name))
        if event.get('pid') in hardware:
            native_cost[name]+=b-a
            native_count[name]+=1
            if name in ['DispatchFFNCombine','SparseFlashMla','HcPre','HcPost','VllmAscendQuantLightningIndexerV2']:
                xs[name].append((a,b))
                if name in ['DispatchFFNCombine','SparseFlashMla']:
                    interesting_kernels.append(event)
        if event.get('pid') in cann:
            args=event.get('args',{})
            for key,v in args.items():
                if key.lower().replace(' ','').replace('_','')=='connectionid':
                    api_connections[str(v)].append(event)
    unions={key:merge(value) for key,value in xs.items()}
    device=merge(unions.get('free',[])+unions.get('compute',[])+unions.get('communication',[]))
    window=duration(device)
    finishes=defaultdict(list)
    for event in flow_finishes:
        finishes[(event.get('pid'),event.get('tid'),str(event.get('ts')))].append(event)
    links=[]
    for kernel in interesting_kernels:
        if len(links)>=16:
            break
        kstart=float(kernel['ts'])
        link={'operator':kernel['name'],'kernel_start_us':kernel['ts'],'kernel_us':kernel['dur'],'device_pid':kernel['pid'],'stream':kernel['tid'],'task_args':kernel.get('args',{})}
        for finish in finishes.get((kernel['pid'],kernel['tid'],str(kernel['ts'])),[]):
            source=flow_starts.get((finish.get('cat'),str(finish.get('id'))))
            if source:
                link['host_flow_source']={'category':source.get('cat'),'ts':source['ts'],'pid':source['pid'],'tid':source['tid']}
                link['submission_to_kernel_us']=kstart-float(source['ts'])
                break
        connection=kernel.get('args',{}).get('connection_id')
        if connection is not None:
            related=api_connections.get(str(connection),[])
            link['connection_api_candidates']=[{'name':v.get('name'),'ts':v['ts'],'dur':v['dur']} for v in related[:3]]
        if 'host_flow_source' in link or link.get('connection_api_candidates'):
            links.append(link)
    rank_match=re.search(r'dp(\d+).*_ep(\d+)_',str(path))
    longest=[]
    for a,b in sorted(unions.get('free',[]),key=lambda item:item[1]-item[0],reverse=True)[:8]:
        longest.append({'start_us':a,'duration_ms':(b-a)/1000,'overlap_gloo_ms':intersection([[a,b]],unions.get('gloo',[]))/1000})
    result={'phase':phase,'file':str(path),'dp':int(rank_match[1]) if rank_match else None,'ep':int(rank_match[2]) if rank_match else None,'events':len(events),'trace_first_us':start,'trace_last_us':end,'device_window_ms':window/1000,'union_ms':{k:duration(v)/1000 for k,v in unions.items()},'device_window_pct':{k:intersection(v,device)/window*100 if window else None for k,v in unions.items()},'calls':dict(counts),'gloo_and_free_ms':intersection(unions.get('gloo',[]),unions.get('free',[]))/1000,'event_and_free_ms':intersection(unions.get('event_sync',[]),unions.get('free',[]))/1000,'largest_free_gaps':longest,'top_cpu_ops':[{'name':name,'inclusive_ms':v/1000} for name,v in cpu_cost.most_common(20)],'top_native_ops':[{'name':name,'total_ms':v/1000,'calls':native_count[name]} for name,v in native_cost.most_common(20)],'host_device_examples':links}
    if rank_match is None:
        python_cost=Counter()
        for event in events:
            if event.get('ph')=='X' and event.get('cat')=='python_function':
                name=event.get('name','')
                if any(k in name for k in ['encode_batch','DecodeStream','httptools_impl.py']):
                    python_cost[name]+=float(event.get('dur',0))
        result['frontend_functions']={name:v/1000 for name,v in python_cost.items()}
    return result

if __name__=='__main__':
    root=Path(sys.argv[1])
    paths=list((root/'traces').rglob('trace_view.json'))
    by_rank=defaultdict(list)
    for path in paths:
        rank=re.search(r'_ep(\d+)_',str(path))[1]
        by_rank[int(rank)].append(path)
    tasks=[]
    for rank,items in sorted(by_rank.items()):
        for index,path in enumerate(sorted(items,key=lambda p:p.parent.parent.name)):
            tasks.append((str(path),['early','steady'][index] if index<2 else 'extra'))
    front=sorted((root/'traces').glob('*.pt.trace.json.gz'))
    tasks.extend((str(path),['early','steady'][i] if i<2 else 'extra') for i,path in enumerate(front))
    result={'rank_directories':{rank:len(v) for rank,v in by_rank.items()},'worker_trace_count':len(paths),'frontend_trace_count':len(front),'files':[]}
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        for item in pool.map(read_one,tasks):
            result['files'].append(item)
            (root/'balanced-analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
            print('finished',item['phase'],item['ep'],flush=True)
    print('DONE',len(tasks),flush=True)
