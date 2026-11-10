from pathlib import Path
from collections import defaultdict,Counter
import concurrent.futures,json,re,sys
from analyze_balanced import merge,duration,intersection

def analyze(path):
    events=json.load(open(path))
    if isinstance(events,dict):
        events=events['traceEvents']
    groups=defaultdict(list)
    count=Counter()
    cpu=[]
    for event in events:
        if event.get('ph')!='X' or not event.get('dur'):
            continue
        a=float(event['ts']);b=a+float(event['dur'])
        name=event.get('name','')
        cat=event.get('cat','')
        if name=='Free':
            groups['free'].append((a,b))
        if cat=='cpu_op':
            count[name]+=1
            cpu.append((a,b,name))
            if name in ['vllm::dsa_v41_forward','vllm::moe_forward_shared','Event::synchronize','gloo:all_reduce']:
                groups[name].append((a,b))
            if name.startswith('aten::') or name=='empty_tensor':
                groups['tensor_ops'].append((a,b))
            if event.get('pid')==event.get('tid') and name not in ['Event::synchronize','gloo:all_reduce']:
                groups['main_thread_cpu_op'].append((a,b))
        if cat=='dequeue':
            groups['api_dequeue_thread'].append((a,b))
    unions={key:merge(value) for key,value in groups.items()}
    free=unions['free']
    largest=[]
    for a,b in sorted(free,key=lambda p:p[1]-p[0],reverse=True)[:5]:
        calls=[]
        for x,y,name in cpu:
            overlap=max(0,min(y,b)-max(x,a))
            if overlap:
                calls.append({'name':name,'overlap_ms':overlap/1000,'call_ms':(y-x)/1000})
        calls.sort(key=lambda x:-x['overlap_ms'])
        largest.append({'duration_ms':(b-a)/1000,'start_us':a,'overlapping_cpu_calls':calls[:8]})
    return {'ep':int(re.search(r'_ep(\d+)_',str(path))[1]),'source':str(path),'free_ms':duration(free)/1000,'intervals':{k:{'union_ms':duration(v)/1000,'overlap_free_ms':intersection(v,free)/1000,'cpu_calls':count.get(k)} for k,v in unions.items() if k!='free'},'largest_free_gaps':largest,'annotation':'Intervals can overlap; cpu_op wall time includes underlying calls and is not pure Python CPU execution time.'}

if __name__=='__main__':
    root=Path(sys.argv[1]);paths=[]
    for prefix in ['dp0_pp0_tp0','dp1_pp0_tp0']:
        paths.append(sorted((root/'traces').glob(prefix+'*/ASCEND_PROFILER_OUTPUT/trace_view.json'),key=lambda p:p.parent.parent.name)[-1])
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(analyze,paths))
    (root/'steady-host-gaps.json').write_text(json.dumps({'representative_ranks':results},indent=2))
    print('host_gap_complete')
