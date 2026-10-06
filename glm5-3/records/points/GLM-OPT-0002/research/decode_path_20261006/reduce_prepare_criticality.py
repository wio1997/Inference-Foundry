"""Existing Run249 only: source-contained MC2 preparation and local exposure.

CPU scope walls include profiler and scheduling. This is not an OFF saving
estimate; intersections do not reconstruct cross-rank propagated effects.
"""
import bisect
import gzip
import hashlib
import json
from pathlib import Path
from reduce_gap_owners import segments


def merged(xs):
    out=[]
    for a,b in sorted(xs):
        if b<=a:continue
        if out and a<=out[-1][1]:out[-1]=(out[-1][0],max(b,out[-1][1]))
        else:out.append((a,b))
    return out


def duration(xs):return sum(b-a for a,b in merged(xs))/1e6


def intersection(xs,ys):
    xs=merged(xs);ys=merged(ys);out=[];i=j=0
    while i<len(xs) and j<len(ys):
        a,b=xs[i];c,d=ys[j]
        if min(b,d)>max(a,c):out.append((max(a,c),min(b,d)))
        if b<d:i+=1
        else:j+=1
    return duration(out)


def reduce(data,queue):
    cpu=sorted((int(x[0]),int(x[1]),x[2]) for x in data['cpu'])
    starts=[x[0] for x in cpu]
    moes=[(a,b) for a,b,n in cpu if n=='vllm::moe_forward_shared']
    assert len(moes)==380
    selected={'prepare_splits':[], 'finalize_splits':[], 'prepare_pads':[]}
    examples=[]
    for a,b in moes:
        children=[x for x in cpu[bisect.bisect_left(starts,a):bisect.bisect_right(starts,b)] if x[1]<=b]
        dispatch=[x for x in children if x[2]=='npu::npu_moe_distribute_dispatch_v2']
        assert len(dispatch)==1,dispatch
        boundary=dispatch[0][0]
        prep=[(s,e) for s,e,n in children if n=='aten::tensor_split' and e<=boundary]
        final=[(s,e) for s,e,n in children if n=='aten::tensor_split' and s>=boundary]
        pads=[(s,e) for s,e,n in children if n=='aten::pad' and e<=boundary]
        assert len(prep)==3 and len(final)==1 and len(pads)==2
        selected['prepare_splits']+=prep;selected['finalize_splits']+=final;selected['prepare_pads']+=pads
        if len(examples)<2:examples.append(dict(MoE_scope_ns=[a,b],first_dispatch_scope=dispatch[0],prepare_split_ns=prep,prepare_pad_ns=pads,finalize_split_ns=final))
    _,_,gaps=segments(data,queue)
    result=dict(rank=data['rank'],db_sha256=data['db_sha256'],MoE_calls=len(moes),
        scopes={key:dict(count=len(xs),union_ms=duration(xs),local_pre_enqueue_gap_intersection_ms=intersection(xs,gaps)) for key,xs in selected.items()},
        prepare_split_and_pad_union_ms=duration(selected['prepare_splits']+selected['prepare_pads']),
        prepare_split_and_pad_local_gap_intersection_ms=intersection(selected['prepare_splits']+selected['prepare_pads'],gaps),
        examples=examples,
        limits=['Exactly three prepare splits and two pads occur before each MC2 dispatch; one split after dispatch is finalize, excluded from the prepare candidate.',
        'The candidate replaces three 16-view splits with three local narrows and global padding with local allocation/padding. New work remains; original scope time is not fully removed.',
        'Profile-ON scope walls and local exposure cannot be extrapolated to OFF265ms/token or to model/E2E gain. Native tensor storage format and allocator lifetime still need correctness validation before deployment.'])
    return result


if __name__=='__main__':
    here=Path(__file__).resolve().parent;point=here.parents[1];rows=[]
    for rank in (13,15):
        paths=[point/f'jobs/GLM53-DECODE-TRACE-20261006-B/reduced/db_rank{rank}.json.gz',point/f'jobs/GLM53-DECODE-QUEUES-20261006/queues_rank{rank}.json.gz']
        row=reduce(*(json.loads(gzip.decompress(p.read_bytes())) for p in paths))
        row['input_sha256']={str(p.relative_to(point)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        rows.append(row);print(json.dumps({k:v for k,v in row.items() if k not in ('examples','input_sha256','limits')}))
    (here/'local_prepare/trace_criticality.json').write_text(json.dumps(rows,indent=2)+'\n')
