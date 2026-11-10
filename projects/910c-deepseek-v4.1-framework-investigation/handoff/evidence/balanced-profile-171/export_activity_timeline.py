"""CPU-only export of aligned, binned union activity from existing traces."""
from pathlib import Path
import concurrent.futures,json,sys

N=512
KEYS={'Computing':'compute','Communication(Not Overlapped)':'communication',
      'Free':'free','gloo:all_reduce':'gloo','Event::synchronize':'event',
      'DispatchFFNCombine':'dfc'}

def merge(xs):
    result=[]
    for a,b in sorted(xs):
        if result and a<=result[-1][1]:result[-1][1]=max(result[-1][1],b)
        else:result.append([a,b])
    return result

def read(item):
    row,base,span=item
    events=json.load(open(row['file']))
    if isinstance(events,dict):events=events['traceEvents']
    xs={key:[] for key in KEYS.values()}
    hardware={e['pid'] for e in events if e.get('ph')=='M' and e.get('name')=='process_name' and e.get('args',{}).get('name')=='Ascend Hardware'}
    for e in events:
        key=KEYS.get(e.get('name'))
        if key and e.get('ph')=='X' and e.get('dur'):
            if key=='dfc' and e.get('pid') not in hardware:continue
            a=float(e['ts']);xs[key].append([a,a+float(e['dur'])])
    bins={}
    width=span/N
    for key,intervals in xs.items():
        counts=[0.0]*N
        for a,b in merge(intervals):
            first=max(0,int((a-base)//width));last=min(N-1,int((b-base)//width))
            for j in range(first,last+1):
                counts[j]+=max(0.0,min(b,base+(j+1)*width)-max(a,base+j*width))/width
        bins[key]=[round(min(1,c)*255) for c in counts]
    device=merge(xs['compute']+xs['communication']+xs['free'])
    return {'phase':row['phase'],'ep':row['ep'],'dp':row['dp'],
            'start_ms':(device[0][0]-base)/1000,'end_ms':(device[-1][1]-base)/1000,'bins':bins}

if __name__=='__main__':
    root=Path(sys.argv[1]);analysis=json.load(open(root/'balanced-analysis.json'))
    out={'bins':N,'phases':{},'ranks':[]}
    jobs=[]
    for phase in ['early','steady']:
        rows=[r for r in analysis['files'] if r['phase']==phase and r['dp'] is not None]
        base=min(r['trace_first_us'] for r in rows)
        end=max(r['trace_last_us'] for r in rows)
        out['phases'][phase]={'base_us':base,'span_ms':(end-base)/1000}
        jobs.extend((r,base,end-base) for r in rows)
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        for row in pool.map(read,jobs):
            out['ranks'].append(row)
            print('timeline',row['phase'],row['ep'],flush=True)
    (root/'activity-timeline.json').write_text(json.dumps(out,separators=(',',':')))
    print('DONE',len(out['ranks']),flush=True)
