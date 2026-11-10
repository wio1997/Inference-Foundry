"""Attribute idle intervals to built-in worker scopes in the targeted capture."""
from pathlib import Path
from collections import Counter,defaultdict
import concurrent.futures,json,re,statistics,sys

SCOPE_NAMES=['prepare input','preprocess','forward','post process','sample_token','draft_token','async_state_update']

def merge(xs):
    out=[]
    for a,b in sorted(xs):
        if b<=a:continue
        if out and a<=out[-1][1]:out[-1][1]=max(out[-1][1],b)
        else:out.append([a,b])
    return out

def intersection(a,b):
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

def duration(xs):return sum(b-a for a,b in xs)/1000
def span(e):return [float(e['ts']),float(e['ts'])+float(e['dur'])]

def analyze(path):
    raw=json.load(open(path));events=raw['traceEvents'] if isinstance(raw,dict) else raw
    rank=int(re.search(r'_ep(\d+)_',str(path))[1])
    proc={e['pid']:e.get('args',{}).get('name') for e in events if e.get('ph')=='M' and e.get('name')=='process_name'}
    py={p for p,n in proc.items() if n=='Python'}
    xs=defaultdict(list);scopes=defaultdict(list);other_scopes=Counter();cpu=[];graph_start=[];gloo=[]
    for e in events:
        if e.get('ph')!='X' or not e.get('dur'):continue
        name=e.get('name','');cat=e.get('cat');iv=span(e)
        if name in ['Free','Computing','Communication(Not Overlapped)']:xs[name].append(iv)
        if e.get('pid') in py and e.get('tid')==e.get('pid') and name in SCOPE_NAMES:
            scopes[name].append(e)
        elif cat=='user_annotation':other_scopes[name]+=1
        if cat=='cpu_op':cpu.append(e)
        if name=='MODEL_EXECUTE':graph_start.append(iv[0])
        if name=='gloo:all_reduce':gloo.append(iv)
    assert all(scopes.get(k) for k in ['prepare input','forward','draft_token']),{'ep':rank,'scopes':dict((k,len(v)) for k,v in scopes.items())}
    for k in list(xs):xs[k]=merge(xs[k])
    device=merge(xs['Free']+xs['Computing']+xs['Communication(Not Overlapped)'])
    free=xs['Free'];graph_start.sort();ledgers={}
    windows={'all':device}
    if len(graph_start)>1:windows['closed_cycles']=[[graph_start[0],graph_start[-1]]]
    for window,coverage in windows.items():
        observed=intersection(free,coverage);remaining=observed;ledger={};scope_rows={}
        # The six production labels are siblings. Still subtract prior scopes so
        # the displayed accounting cannot double count if a future version nests.
        for name in SCOPE_NAMES:
            ivs=merge([span(e) for e in scopes[name]])
            part=intersection(remaining,ivs);ledger[name]=duration(part)
            remaining=subtract(remaining,ivs)
            scope_rows[name]={'calls':len(scopes[name]),'inclusive_cpu_ms':duration(ivs),
                              'free_overlap_ms':duration(intersection(observed,ivs)),
                              'exclusive_free_ms':duration(part)}
        ledger['outside_recorded_worker_scopes']=duration(remaining)
        total=duration(observed);assert abs(sum(ledger.values())-total)<1e-5
        ledgers[window]={'device_window_ms':duration(coverage),'free_ms':total,
                         'free_pct_of_device':total/duration(coverage)*100,
                         'exclusive_free_ms':ledger,
                         'exclusive_pct_of_free':{k:v/total*100 for k,v in ledger.items()},
                         'scope_details':scope_rows,'gloo_free_ms':duration(intersection(observed,merge(gloo)))}
    # Concrete per-call scope examples keep CPU/tensor names and simultaneous
    # idle separate from the CPU scope's inclusive wall-clock duration.
    examples=[]
    for name in ['prepare input','draft_token']:
        for e in scopes[name][1:3]:
            a,b=span(e);piece=intersection(free,[[a,b]])
            costs=defaultdict(list);counts=Counter()
            for op in cpu:
                x,y=span(op)
                if x>=a and y<=b:
                    costs[op['name']].append([x,y]);counts[op['name']]+=1
            rows=sorted([{'name':n,'calls':counts[n],'inclusive_ms':duration(ivs),
                          'free_overlap_ms':duration(intersection(piece,merge(ivs)))} for n,ivs in costs.items()],key=lambda o:-o['free_overlap_ms'])[:16]
            examples.append({'scope':name,'start_us':a,'end_us':b,'inclusive_cpu_ms':(b-a)/1000,'free_ms':duration(piece),'top_cpu_ops_during_free':rows})
    return {'ep':rank,'dp':rank//8,'file':str(path),'scope_counts':{k:len(v) for k,v in scopes.items()},'other_scope_names':other_scopes,'ledgers':ledgers,'examples':examples}

if __name__=='__main__':
    root=Path(sys.argv[1]);paths=sorted((root/'traces').rglob('trace_view.json'))
    assert len(paths)==16,('expected exactly 16 worker traces',len(paths))
    out={'kind':'one targeted steady capture with built-in scopes','ranks':[]}
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        for row in pool.map(analyze,paths):
            out['ranks'].append(row);print('scopes',row['ep'],row['scope_counts'],flush=True)
    (root/'scope-analysis.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));print('SCOPE_ANALYSIS_COMPLETE',flush=True)
