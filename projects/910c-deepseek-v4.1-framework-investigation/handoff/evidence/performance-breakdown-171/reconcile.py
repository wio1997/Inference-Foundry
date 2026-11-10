"""Reconcile ranked type durations against CANN Computing, using set algebra."""
from pathlib import Path
from collections import defaultdict,Counter
import concurrent.futures,csv,json,re,sys

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
def ms(xs):return sum(b-a for a,b in xs)/1000
def iv(e):return [float(e['ts']),float(e['ts'])+float(e['dur'])]
def role(r):
    if r.get('Type')!='DispatchFFNCombine':return r.get('Type',r.get('Name','unknown'))
    p=r.get('Input Shapes','').strip('"').split(';');top=p[3].split(',')[-1] if len(p)>3 else ''
    return {'6':'DFC_target_top6','3':'DFC_draft_top3'}.get(top,'DFC_unknown')
def comm(n):return n.startswith('hcom_') or 'AicpuKernel' in n

def analyze(path):
    path=Path(path);d=json.load(path.open());es=d.get('traceEvents',[]) if isinstance(d,dict) else d
    starts=sorted(iv(e)[0] for e in es if e.get('ph')=='X' and e.get('name')=='MODEL_EXECUTE')
    lo,hi=starts[0],starts[-1];coverage=[[lo,hi]]
    computing=intersect(merge([iv(e) for e in es if e.get('ph')=='X' and e.get('name')=='Computing']),coverage)
    by=defaultdict(list);core=defaultdict(set)
    with (path.parent/'kernel_details.csv').open() as stream:
        for r in csv.DictReader(stream):
            n=role(r)
            if comm(n):continue
            a=float(r['Start Time(us)']);b=a+float(r['Duration(us)']);a=max(a,lo);b=min(b,hi)
            if b>a:by[n].append([a,b]);core[n].add(r.get('Accelerator Core',''))
    by={k:merge(v) for k,v in by.items()};all_union=merge([p for xs in by.values() for p in xs])
    # Sweep unique-type activity; a shared region is its own additive bucket.
    edges=[]
    for name,xs in by.items():
        for a,b in xs:edges.extend([(a,1,name),(b,-1,name)])
    edges.sort();active=Counter();last=None;pos=0;segments=defaultdict(list)
    while pos<len(edges):
        t=edges[pos][0]
        if last is not None and t>last:
            names=tuple(sorted(n for n,c in active.items() if c>0))
            if names:segments[names].append([last,t])
        while pos<len(edges) and edges[pos][0]==t:
            _,kind,name=edges[pos];active[name]+=kind;pos+=1
        last=t
    rows=[]
    for names,xs in segments.items():
        inside=intersect(merge(xs),computing)
        if inside:rows.append({'types':list(names),'kind':'single' if len(names)==1 else 'parallel',
                               'computing_ms':ms(inside)})
    covered=ms(intersect(all_union,computing));cann=ms(computing);union=ms(all_union);sum_types=sum(ms(v) for v in by.values())
    assert abs(sum(r['computing_ms'] for r in rows)-covered)<1e-5
    per_type=[]
    for name,xs in by.items():
        inside=ms(intersect(xs,computing));per_type.append({'name':name,'cores':sorted(core[name]),
            'ranked_union_ms':ms(xs),'within_computing_ms':inside,'outside_computing_ms':ms(xs)-inside})
    return {'ep':int(re.search(r'_ep(\d+)_',str(path))[1]),'window_ms':(hi-lo)/1000,
            'sum_type_ms':sum_types,'union_all_type_ms':union,'multi_type_extra_count_ms':sum_types-union,
            'union_outside_computing_ms':union-covered,'computing_not_covered_by_csv_ms':cann-covered,
            'computing_ms':cann,'computing_partition':sorted(rows,key=lambda r:-r['computing_ms']),
            'per_type':sorted(per_type,key=lambda r:-r['ranked_union_ms'])}

if __name__=='__main__':
    root=Path(sys.argv[1]);paths=sorted((root/'traces').rglob('trace_view.json'));assert len(paths)==16
    out={'ranks':[]}
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        for row in pool.map(analyze,paths):out['ranks'].append(row);print('reconciled',row['ep'],flush=True)
    (root/'computing-reconciliation.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));print('RECONCILED',len(out['ranks']),flush=True)
