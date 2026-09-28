#!/usr/bin/env python3
"""Run99 conditional prefix-active Target-row envelope from saved aggregates.

This is an exact integer relaxation for the declared fixed Target-8/current
serving class, not necessary fresh model work or a time/traffic bound.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

ROOT=Path('/data/wio/Inference_Foundry')
BASE=ROOT/'evidence/20260924_loop036_metadata/run99/runtime'
A0=ROOT/'evidence/20260928_loop080_bound/run566/a0/runtime'
OUT=ROOT/'evidence/20260928_loop081_bound/run595'

def need(ok,msg):
    if not ok:raise ValueError(msg)

def window_data(runtime):
    lengths=[];totals=[];expected=0
    for name,mean in runtime['acceptance_window_means'].items():
        match=re.fullmatch(r'(\d+)-(\d+)',name)
        need(match is not None,'window name')
        start,end=map(int,match.groups())
        need(start==expected and start<=end,'noncontiguous window')
        length=end-start+1
        raw=mean*length*12
        total=round(raw)
        need(abs(raw-total)<1e-6,'window rounded sum not integer')
        lengths.append(length);totals.append(total);expected=end+1
    need(expected==runtime['cycles'],'window cycle coverage')
    staged=runtime['staged_output_counts']
    # Host progress mirror trails the device and may stage more than one
    # width-8 window beyond the exact 1024 output cap.
    need(len(staged)==12 and all(1024<=x<=1039 for x in staged) and
         runtime['generated_output_counts']==[1024]*12,
         'staged count contract')
    need(sum(totals)==sum(staged),'window/staged sum mismatch')
    return lengths,totals,staged

def solve(lengths,totals,staged,maximize):
    W=len(lengths);N=12*W*3
    def idx(slot,w,field):return (slot*W+w)*3+field
    A,Y,Z=0,1,2
    lower=np.zeros(N);upper=np.zeros(N);integral=np.ones(N)
    objective=np.zeros(N)
    for s in range(12):
        for w,L in enumerate(lengths):
            upper[idx(s,w,A)]=L
            upper[idx(s,w,Y)]=8*L
            upper[idx(s,w,Z)]=1
            objective[idx(s,w,A)]=-1 if maximize else 1
        lower[idx(s,0,Z)]=1
    rr=[];cc=[];vv=[];low=[];high=[]
    def constraint(terms,lo=-np.inf,hi=np.inf):
        r=len(low)
        for key,coef in terms:
            rr.append(r);cc.append(key);vv.append(coef)
        low.append(lo);high.append(hi)
    for s in range(12):
        for w,L in enumerate(lengths):
            a,y,z=idx(s,w,A),idx(s,w,Y),idx(s,w,Z)
            # A positive iff active at window start. If active at next window
            # start, the entire current window was active.
            constraint([(a,1),(z,-1)],lo=0)
            constraint([(a,1),(z,-L)],hi=0)
            constraint([(y,1),(a,-1)],lo=0)
            constraint([(y,1),(a,-8)],hi=0)
            if w+1<W:
                nextz=idx(s,w+1,Z)
                constraint([(a,1),(nextz,-L)],lo=0)
                constraint([(z,1),(nextz,-1)],lo=0)
        constraint([(idx(s,w,Y),1) for w in range(W)],
                   lo=staged[s],hi=staged[s])
    for w,total in enumerate(totals):
        constraint([(idx(s,w,Y),1) for s in range(12)],lo=total,hi=total)
    matrix=coo_matrix((vv,(rr,cc)),shape=(len(low),N)).tocsr()
    result=milp(objective,integrality=integral,bounds=Bounds(lower,upper),
        constraints=LinearConstraint(matrix,np.array(low),np.array(high)),
        options={'time_limit':90,'mip_rel_gap':0})
    need(result.status==0 and result.x is not None,
         f'MILP not proved optimal: {result.message}')
    active=round(sum(result.x[idx(s,w,A)] for s in range(12) for w in range(W)))
    need(abs(active+result.fun if maximize else active-result.fun)<1e-5,
         'objective mismatch')
    return active,dict(mip_gap=result.mip_gap,node_count=result.mip_node_count,
                       status=result.message)

def main():
    need(not (OUT/'summary.json').exists(),'output already exists')
    rows=[];source={}
    for label,root,ids in (
        ('formal_run99',BASE,range(5,17)),
        ('diagnostic_run566_a0',A0,range(5,9))):
        for cohort in ids:
            path=root/f'rank0_cohort{cohort}.json'
            runtime=json.loads(path.read_text())
            need(runtime['rank']==0 and runtime['cohort']==cohort and runtime['pass'],
                 'runtime identity')
            for rank in range(8):
                peer_path=root/f'rank{rank}_cohort{cohort}.json'
                peer=json.loads(peer_path.read_text())
                need(peer['rank']==rank and peer['cohort']==cohort and peer['pass'] and
                     all(peer[key]==runtime[key] for key in
                         ('req_ids','cycles','generated_output_counts',
                          'staged_output_counts','acceptance_window_means')),
                     f'all8 Runtime cohort disagreement: {label} {cohort} rank{rank}')
                source[str(peer_path.relative_to(ROOT))]=hashlib.sha256(
                    peer_path.read_bytes()).hexdigest()
            lengths,totals,staged=window_data(runtime)
            minimum,min_meta=solve(lengths,totals,staged,False)
            maximum,max_meta=solve(lengths,totals,staged,True)
            need(minimum<=maximum,'inverted envelope')
            rows.append(dict(source=label,cohort=cohort,cycles=runtime['cycles'],
                window_lengths=lengths,window_sampled_totals=totals,
                staged_total=sum(staged),
                conditional_active_target8_rows_min=8*minimum,
                conditional_active_target8_rows_max=8*maximum,
                min_solver=min_meta,max_solver=max_meta))
            print(label,cohort,8*minimum,8*maximum,flush=True)
    a0=[r for r in rows if r['source']=='diagnostic_run566_a0']
    exact=[24368,23824,26064,24240]
    need(all(r['conditional_active_target8_rows_min']<=actual<=
             r['conditional_active_target8_rows_max']
             for r,actual in zip(a0,exact)),'A0 exact trace outside relaxed envelope')
    formal={}
    for repeat in (1,2,3):
        group=[r for r in rows if r['source']=='formal_run99' and
               4*repeat+1<=r['cohort']<=4*repeat+4]
        need(len(group)==4,'repeat cohort set')
        formal[str(repeat)]=dict(
          cycle_total=sum(r['cycles'] for r in group),
          active_target8_min=sum(r['conditional_active_target8_rows_min'] for r in group),
          active_target8_max=sum(r['conditional_active_target8_rows_max'] for r in group))
    result=dict(status='conditional_prefix_active_integer_envelope',
        model='12 initially active slots; each slot positive 1..8 per active cycle, one prefix then parked zero; exact per-slot staged total and exact aggregate window totals',
        class_limit='Current Target-8 geometry only; not unique fresh work, compulsory FLOPs/traffic, Hardware/Scheduling/Product Bound',
        rows=rows,formal_run99_repeats=formal,source_sha256=source,
        a0_exact_trace_in_envelope=True,formal_current_tps=571.681,
        resource_bound_s=None,scheduling_bound_s=None,product_bound_tps=None)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(formal,sort_keys=True))

if __name__=='__main__':main()
