"""Pure Python value models for proposal lengths versus accepted-state metadata."""
from itertools import combinations,combinations_with_replacement,product


def publish_prefixes(tokens,keep,cap):
    if keep is None:return [list(row)for row in tokens]
    # Preserve the selected source's clamp and zip, including length mismatch.
    return [list(row[:max(0,min(int(k),cap))])for row,k in zip(tokens,keep)]


def prefix_layout(lengths):
    assert all(isinstance(n,int)and 0<=n<2**31-1 for n in lengths)
    qsl=[0];owner=[]
    for i,n in enumerate(lengths):
        q=n+1;qsl.append(qsl[-1]+q);owner.extend([i]*q)
    assert qsl[-1]<2**31
    return qsl,owner


def next_count(row,backup,V,discard=False):
    y=[-1]*len(row)if discard else list(row)
    assert y
    c=sum(x!=-1 and x<V for x in y)
    return (y[max(c-1,0)]if c else backup),c


def topk_counts(rows,min_k,extra,chosen):
    B=len(rows);D=len(rows[0]);flat=[x for row in rows for x in row[min_k:]]
    assert len(chosen)==extra and len(set(chosen))==extra and all(0<=i<len(flat)for i in chosen)
    selected=set(chosen);unselected=set(range(len(flat)))-selected
    assert not selected or not unselected or min(flat[i]for i in selected)>=max(flat[i]for i in unselected)
    counts=[min_k]*B
    for j in chosen:counts[j//(D-min_k)]+=1
    return counts


def enumerate_layouts():
    records=[]
    for B in range(5):
        for D in range(6):
            for lengths in product(range(D+1),repeat=B):
                qsl,owner=prefix_layout(lengths)
                explicit=[i for i,n in enumerate(lengths)for _ in range(n+1)]
                assert owner==explicit and all(qsl[i+1]-qsl[i]==lengths[i]+1 for i in range(B))
                records.append(dict(B=B,D=D,lengths=list(lengths),Q=qsl[-1],qsl=qsl,owner=owner,
                    new_ABI_logical_content_bytes=4*B+4*(B+1)+4*qsl[-1],
                    actual_baseline_HBM_or_latency_saving=False))
    return records


def enumerate_ties():
    tested=valid=nonprefix=0;examples=[]
    # Positive nonincreasing survival values, representable as multiples of1/4.
    # They are mathematical TopK outcomes, not an attested native tie policy.
    for B in [1,2]:
        for D in range(2,6):
            possible=[tuple(reversed(r))for r in combinations_with_replacement([1,2,4],D)]
            for rows in product(possible,repeat=B):
                flat=[x for r in rows for x in r[1:]];n=len(flat)
                for budget in range(1,D+1):
                    L=min(B*(budget-1),n)
                    for chosen in combinations(range(n),L):
                        tested+=1;inside=set(chosen);outside=set(range(n))-inside
                        if inside and outside and min(flat[i]for i in inside)<max(flat[i]for i in outside):continue
                        counts=topk_counts(rows,1,L,chosen);valid+=1
                        requested={(i,j)for i in range(B)for j in range(counts[i])}
                        selected={(i,0)for i in range(B)}|{(x//(D-1),1+x%(D-1))for x in chosen}
                        assert sum(counts)==B+L
                        if requested!=selected:
                            nonprefix+=1
                            if len(examples)<4:examples.append(dict(B=B,D=D,rows=rows,budget=budget,chosen=list(chosen),keep=counts,selected_positions=sorted(selected),published_prefix_positions=sorted(requested)))
    return dict(candidate_subsets=tested,valid_TopK_subsets=valid,nonprefix_selected_position_cases=nonprefix,examples=examples,native_tie_policy_or_runtime_values_attested=False)
