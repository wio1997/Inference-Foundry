#!/usr/bin/env python3
"""Exact-rational resource-window and path lower bounds for a fixed DAG domain.

Illustration only: workloads, hard capacities, durations, edges and mandatory
execution must be proved independently before using for real inference.
"""
if not __debug__:
    raise RuntimeError('optimized Python disables required validation; use python3 without -O')

from fractions import Fraction as F
from pathlib import Path
import hashlib,json

def bound(nodes, edges, capacities):
    """Nodes contain id, fixed resource, work and duration lower bound."""
    by={n['id']:n for n in nodes}; assert len(by)==len(nodes)
    assert all(F(n['work'])>=0 and F(n['duration'])>=0 for n in nodes)
    assert all(F(c)>0 for c in capacities.values())
    assert all(n['resource'] in capacities for n in nodes)
    pred={i:[] for i in by};succ={i:[] for i in by}
    for u,v in edges:
        assert u in by and v in by and u!=v
        pred[v].append(u);succ[u].append(v)
    order=[];remaining=set(by)
    while remaining:
        ready=sorted(i for i in remaining if all(j not in remaining for j in pred[i]))
        if not ready: raise ValueError('cyclic DAG')
        order.extend(ready); remaining.difference_update(ready)
    earliest={}
    for i in order: earliest[i]=max((earliest[j]+F(by[j]['duration']) for j in pred[i]),default=F(0))
    tail={}
    for i in reversed(order):tail[i]=max((F(by[j]['duration'])+tail[j] for j in succ[i]),default=F(0))
    path=max((earliest[i]+F(by[i]['duration'])+tail[i] for i in by),default=F(0))
    windows=[]
    for res,capacity in sorted(capacities.items()):
        ns=[n for n in nodes if n['resource']==res]
        if not ns:continue
        for a in sorted({earliest[n['id']] for n in ns}|{F(0)}):
            for b in sorted({tail[n['id']] for n in ns}|{F(0)}):
                selected=[n for n in ns if earliest[n['id']]>=a and tail[n['id']]>=b]
                if selected:
                    work=sum((F(n['work']) for n in selected),F(0)); value=a+work/F(capacity)+b
                    windows.append((value,res,a,b,[n['id'] for n in selected],work))
    best=max(windows,key=lambda v:v[0]) if windows else None
    low=max(path,best[0] if best else F(0))
    return {'critical_path':str(path),'resource_window_bound':str(best[0]) if best else '0',
            'lower_bound':str(low),'winning_window':{'resource':best[1],'release':str(best[2]),'tail':str(best[3]),'jobs':best[4],'total_work':str(best[5]),'capacity':str(F(capacities[best[1]]))} if best else None,
            'resource_bounds':{r:str(sum((F(n['work']) for n in nodes if n['resource']==r),F(0))/F(c)) for r,c in capacities.items()}}

def example(read_time,compute_time,write_time):
    nodes=[];edges=[]
    for i in range(4):
        for name,res,d in [('in','io',read_time),('calc','cube',compute_time),('out','io',write_time)]:
            nodes.append({'id':f'{name}{i}','resource':res,'work':str(d),'duration':str(d)})
        edges.extend([(f'in{i}',f'calc{i}'),(f'calc{i}',f'out{i}')])
    return bound(nodes,edges,{'io':1,'cube':1})

def validate_witness(intervals):
    # Only validates the explicitly specified 4-tile unit-capacity example.
    spec={f'{name}{i}':(resource,F(duration)) for i in range(4)
          for name,resource,duration in [('in','io',1),('calc','cube',3),('out','io',1)]}
    assert len(intervals)==len(spec)
    times={}
    for name,start,end,resource in intervals:
        assert name in spec and name not in times
        start,end=F(start),F(end)
        assert start>=0 and end>=start and resource==spec[name][0] and end-start>=spec[name][1]
        times[name]=(start,end,resource)
    assert set(times)==set(spec)
    for i in range(4):
        assert times[f'in{i}'][1]<=times[f'calc{i}'][0] and times[f'calc{i}'][1]<=times[f'out{i}'][0]
    for res in ('io','cube'):
        a=sorted((s,e) for s,e,r in times.values() if r==res)
        assert all(e<=s2 for (_,e),(s2,_) in zip(a,a[1:]))
    return str(max(e for s,e,res in times.values()))

x=example(1,3,1);y=example(8,3,8)
w=[]
for i in range(4):w.extend([(f'in{i}',i,i+1,'io'),(f'calc{i}',1+3*i,4+3*i,'cube'),(f'out{i}',4+3*i,5+3*i,'io')])
witness=validate_witness(w)
invalid=[(name,0,0,resource) for name,start,end,resource in w]
try: validate_witness(invalid)
except AssertionError: pass
else: raise AssertionError('zero-duration false witness accepted')
assert x['critical_path']=='5' and x['resource_bounds']=={'io':'8','cube':'12'} and x['lower_bound']=='14' and witness=='14'
assert y['lower_bound']=='64'  # same port has 32+32 mandatory work; not 8+4*3+8
p=Path(__file__);result={'model':'fixed-DAG, mandatory disjoint workloads with hard resource capacities','source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'example_pipeline':x,'example_pipeline_witness_time':witness,'example_shared_io_bottleneck':y,'first_read_plus_compute_plus_last_write_loose_bound':'28','not_169_observation':True}
out=p.with_name('resource_window_bound.v2.json');out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n');print(json.dumps(result,indent=2,ensure_ascii=False))
