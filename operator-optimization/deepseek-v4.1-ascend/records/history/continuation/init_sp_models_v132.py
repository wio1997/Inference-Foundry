"""Finite copy/ownership and logical-content models, never native capacity bounds."""
from itertools import product


def copy_cases():
    out=[];bits=[0,0x8000,0x7f80,0xff80,0x7fc1,0x3f80]
    for N in range(25):
        E=[tuple(bits[(i+j)%len(bits)]for j in range(3))for i in range(N)]
        for P in range(1,9):
            n=(N+P-1)//P;A=n*P
            masks=[None,[True]*N,[bool(i%2)for i in range(N)]]
            if N<=4:masks += [list(x)for x in product([False,True],repeat=N)]
            for rank in range(P):
                begin=rank*n;end=begin+n
                old=([tuple(E[i]for _ in range(4))for i in range(N)]+[tuple((0,0,0)for _ in range(4))for _ in range(A-N)])[begin:end]
                local=(E+[(0,0,0)]*(A-N))[begin:end];new=[tuple(e for _ in range(4))for e in local];assert old==new
                for mask in masks:
                    prefix=[False]*N if mask is None else mask
                    padded_mask=(prefix+[True]*(A-N))[begin:end]
                    direct_mask=[prefix[i]if i<N else True for i in range(begin,end)]
                    assert padded_mask==direct_mask
                    out.append(dict(N=N,P=P,rank=rank,A=A,mask=mask,local_HC_bits=new,local_mask=direct_mask))
    return out


def route_cases():
    out=[]
    for N in range(1,25):
        for P in range(1,9):
            n=(N+P-1)//P
            for shift in range(P):
                query_rows=[list(range(((p+shift)%P)*n,min(((p+shift)%P+1)*n,N)))for p in range(P)]
                missing=[(i//n,p,i)for p,rows in enumerate(query_rows)for i in rows if i//n!=p]
                # Each projected value carries the original global row ID.
                X=[3*i+1 for i in range(N)];qa=lambda x:5*x+7
                routed={(p,i):qa(X[i])for owner,p,i in missing}
                restored=[[qa(X[i])if i//n==p else routed[p,i]for i in rows]for p,rows in enumerate(query_rows)]
                expected=[[qa(X[i])for i in rows]for rows in query_rows];assert restored==expected
                out.append(dict(N=N,P=P,A=n*P,shift=shift,query_rows=query_rows,missing_owner_recipient_rows=[list(x)for x in missing],
                    raw_endpoint_value_bytes=2*5120*len(missing),QA_endpoint_value_bytes=2*1280*len(missing),ordered_QA_values=restored))
    return out


def content(N,P=8,H=5120,D=512,Q=1280):
    n=(N+P-1)//P;A=n*P
    return dict(N=N,P=P,A=A,n=n,old_embedding_AR_input_and_output_content_each=2*N*H,
        new_embedding_RS_padded_input_content=2*A*H,new_embedding_RS_output_content=2*n*H,
        old_first_SP_sender_content=2*n*H,new_KV_only_sender_content=2*n*D,new_KV_QA_sender_content=2*n*(D+Q),
        current_global_KV_dense_rows_across_P=P*N,new_uniform_owner_KV_dense_rows=A,
        current_disjoint_cover_QA_dense_rows=N,new_uniform_owner_QA_dense_rows=A,
        old_per_card_HC_Tile_and_Pad_value_IO=2*N*H+2*N*4*H+2*N*4*H+2*A*4*H,
        new_local_HC_initializer_HC_only_value_IO_max=2*n*H+2*n*4*H,
        current_QA_and_KV_weight_content_each_card=2*H*(Q+D),physical_wire_HBM_latency_or_unconditional_necessary_work=False)


def cache_epoch_cases():
    cases=[]
    for epoch in [0,1]:
        for token in range(4):
            # A pure row-function stand-in; production native rounding is not simulated.
            row=epoch*101+token+1;kv=row*row+3;qa=5*row+7
            cases.append(dict(epoch=epoch,token=token,KV=kv,QA=qa))
    old=next(c for c in cases if c['epoch']==0 and c['token']==2);new=next(c for c in cases if c['epoch']==1 and c['token']==2)
    assert old['KV']!=new['KV']
    return dict(cases=cases,stale_token_only_key=dict(old=old,new=new),
        padding_is_not_token_zero=dict(padding_raw_embedding=0,token_zero_embedding=1),
        pre_RoPE_cache_cannot_cache_final_positioned_value=dict(pre_RoPE=[1,0],position_A_rotated=[1,0],position_B_rotated=[0,1]))
