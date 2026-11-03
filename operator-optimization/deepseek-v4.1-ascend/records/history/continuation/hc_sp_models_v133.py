"""Abstract fork schedules and value-content budgets, not measured service."""
from itertools import product


def schedule(C,Y,J,Gh,Ar,Gk,Ak):
    assert len(C)==len(Y)==len(J) and all(v>=0 for v in C+Y+J+[Gh,Ar,Gk,Ak])
    return dict(baseline=max(max(c,y)for c,y in zip(C,Y))+Gh+Ar,
        split_only=max(max(C),max(Y)+Gh+Ar),
        projection_only=max(max(c,y)+j for c,y,j in zip(C,Y,J))+Gk+Ak,
        combined=max(max(C),max(y+j for y,j in zip(Y,J))+Gk+Ak))


def schedules():
    out=[]
    for P in [1,2,3]:
        for values in product([1,4],repeat=3*P):
            C,Y,J=list(values[:P]),list(values[P:2*P]),list(values[2*P:])
            for Gh,Ar,Gk,Ak in product([0,2],repeat=4):
                out.append(dict(C=C,Y=Y,J=J,Gh=Gh,Ar=Ar,Gk=Gk,Ak=Ak,**schedule(C,Y,J,Gh,Ar,Gk,Ak)))
    for C,Y,J,Gh,Ar,Gk,Ak in [([10],[1],[1],4,6,1,2),([10,1],[1,9],[10,0],5,5,0,0),
        ([10]+[1]*7,[1]*8,[1]*8,2,3,1,2),([1]*8,[1,9]+[1]*6,[2]*8,2,3,1,2),([1]*8,[1]*8,[10]+[1]*7,1,1,3,3)]:
        out.append(dict(C=C,Y=Y,J=J,Gh=Gh,Ar=Ar,Gk=Gk,Ak=Ak,**schedule(C,Y,J,Gh,Ar,Gk,Ak)))
    return out


def values(M,H=5120,h=4,D=512,Q=1280):
    return dict(M=M,raw_X_content=2*M*h*H,old_premix_content=4*M*h,
        Y_content=2*M*H,Z_content=2*M*H,KV_content=2*M*D,QA_content=2*M*Q,
        current_pre_post_comb_content=4*M*(h+h+h*h),
        Y_private_write_read_budget=4*M*H,Z_private_write_read_budget=4*M*H,
        both_private_interface_budget=8*M*H,
        retained_coefficient_cast_workspace_value_IO_reference=8*M*h*H,
        physical_HBM_allocation_or_necessary_bytes_established=False)
