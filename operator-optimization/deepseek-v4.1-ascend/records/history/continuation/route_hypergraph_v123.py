"""Stdlib reader and explicit static-placement reference models, never target code."""
from collections import Counter
from itertools import combinations
import gzip,hashlib,struct
from finalize_origin_coverage_v9 import N,O,read


def records():
    meta=read(O/'route-hyperedges-source-index-v123.json')
    compressed=(N/'route-hypergraph-source-v123/active-routes.bin.gz').read_bytes()
    assert len(compressed)==meta['compressed_bytes'] and hashlib.sha256(compressed).hexdigest()==meta['compressed_sha256']
    data=gzip.decompress(compressed)
    assert len(data)==meta['packed_bytes'] and hashlib.sha256(data).hexdigest()==meta['packed_sha256']
    assert data[:8]==b'WIOEP123'
    offset=8
    for m in meta['record_index']:
        assert offset==m['offset']
        index,K,E,M,V=struct.unpack_from('<IHHII',data,offset);offset+=16
        assert index==m['source_index'] and K in [3,6] and E in [8,24]
        rs=struct.Struct('<I'+'H'*K);rows=[]
        for _ in range(V):
            values=rs.unpack_from(data,offset);offset+=rs.size
            assert 0<=values[0]<M and len(set(values[1:]))==K and all(0<=e<16*E for e in values[1:])
            rows.append(values)
        assert offset==m['offset']+m['bytes'] and len({r[0]for r in rows})==V
        yield dict(m,K=K,E_local=E,M=M,rows=rows)
    assert offset==len(data)


def fit_training(rows,E):
    G=16*E;freq=[0]*G;pairs=[[0]*G for _ in range(G)];sender_freq=[[0]*G for _ in range(16)]
    for rec in rows:
        s=rec['sender_EP_slot']
        for row in rec['rows']:
            experts=row[1:]
            for e in experts:
                freq[e]+=1;sender_freq[s][e]+=1
            for e,f in combinations(experts,2):
                pairs[e][f]+=1;pairs[f][e]+=1
    return freq,pairs,sender_freq


def balanced(freq,E,within_half=False):
    G=len(freq);owner=[-1]*G;load=[0]*16;slots=[0]*16
    for e in sorted(range(G),key=lambda e:(-freq[e],e)):
        allowed=range((e//E//8)*8,(e//E//8+1)*8) if within_half else range(16)
        rank=min((r for r in allowed if slots[r]<E),key=lambda r:(load[r],slots[r],r))
        owner[e]=rank;load[rank]+=freq[e];slots[rank]+=1
    assert slots==[E]*16
    return owner


def colocated(freq,pairs,sender_freq,E,within_half=False):
    """Capacity-constrained greedy pair-affinity scenario, not a global optimizer."""
    G=len(freq);clusters=[]
    domains=[(range(G),range(16))] if not within_half else [(range(0,G//2),range(8)),(range(G//2,G),range(8,16))]
    owner=[-1]*G
    for experts,labels in domains:
        remaining=set(experts);partitions=[]
        while remaining:
            anchor=max(remaining,key=lambda e:(freq[e],-e));remaining.remove(anchor)
            group=[anchor];affinity=[pairs[e][anchor]for e in range(G)]
            while len(group)<E:
                e=max(remaining,key=lambda e:(affinity[e],freq[e],-e));remaining.remove(e);group.append(e)
                for f in remaining:
                    affinity[f]+=pairs[f][e]
            partitions.append(sorted(group))
        labels=list(labels);assigned_groups=set();assigned_labels=set()
        scored=[(-sum(sender_freq[r][e]for e in group),g,r)for g,group in enumerate(partitions)for r in labels]
        for _,g,r in sorted(scored):
            if g in assigned_groups or r in assigned_labels:
                continue
            for e in partitions[g]:
                owner[e]=r
            assigned_groups.add(g);assigned_labels.add(r)
        assert len(assigned_groups)==len(labels)
        clusters.extend(partitions)
    assert Counter(owner)=={r:E for r in range(16)}
    return owner


def score(cohort,owner):
    E=cohort[0]['E_local'];K=cohort[0]['K']
    assert len(owner)==16*E and Counter(owner)=={r:E for r in range(16)}
    A=[[0]*16 for _ in range(16)];D=[[0]*16 for _ in range(16)];pair_local=0;valid=0;expert_freq=[0]*len(owner)
    for rec in cohort:
        s=rec['sender_EP_slot']
        for row in rec['rows']:
            ds=[owner[e]for e in row[1:]];valid+=1
            for e in row[1:]:
                expert_freq[e]+=1
            for d in ds:
                A[s][d]+=1
            for d in set(ds):
                D[s][d]+=1
            hs=Counter(ds);pair_local+=sum(n*(n-1)//2 for n in hs.values())
    ar=sum(A[s][d]for s in range(16)for d in range(16)if s!=d)
    dr=sum(D[s][d]for s in range(16)for d in range(16)if s!=d)
    loads=[sum(A[s][d]for s in range(16))for d in range(16)]
    da=sum(map(sum,D));aa=valid*K
    assert aa==sum(map(sum,A)) and ar>=dr
    return dict(active_recorded_rows=valid,assignments_all=aa,token_destinations_all=da,
        A_remote=ar,D_remote=dr,structural_repeated_remote_references=ar-dr,
        cross_TP8_assignment_references=sum(A[s][d]for s in range(16)for d in range(16)if s//8!=d//8),
        cross_TP8_token_destination_references=sum(D[s][d]for s in range(16)for d in range(16)if s//8!=d//8),
        receiver_assignment_rows=loads,max_receiver_assignment_rows=max(loads),
        unreplicated_receiver_row_count_lower_reference=max(max(expert_freq),-(-aa//16)),
        within_destination_expert_pair_score=pair_local,
        old_format_dispatch_and_return_logical_bytes=ar*(5152+10240),
        selected_packet_dispatch_and_return_logical_bytes=dr*5216+15360+ar*10240,
        selected_packet_vs_same_mapping_old_format_logical_delta=ar*5152-dr*5216-15360)
