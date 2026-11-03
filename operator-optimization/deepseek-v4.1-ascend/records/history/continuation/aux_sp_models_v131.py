"""Declared packet/row-function models, not native timings or wire volumes."""
from itertools import product
from fractions import Fraction
import struct


def packets(N,M,P=8,H=5120,D=512,u_max=None):
    assert N>0 and N%P==0 and M>0
    n=N//P;feature=2*M*(H//P);aux=3*2*n*H;kv=3*2*n*D
    base=aux+3*2*n*H+feature;separate=aux+kv+feature
    out=dict(N_physical=N,M_selected=M,P=P,sender_rows=n,
        baseline_sender_content_bytes=base,
        independent_F32_projected_sender_content_bytes=separate,
        early_aux_KV_copack_sender_content_bytes=separate,
        early_copack_extra_payload_saving_beyond_F32=0,
        baseline_collective_op_reference=7,independent_F32_collective_op_reference=7,early_copack_op_reference=4,
        retained_main_feature_sender_content_bytes=feature,
        compatibility_pack_unpack_value_bytes_three_checkpoints=3*4*(P+1)*n*(H+D),
        owner_main_context_carrier_sender_content_bytes=kv+2*n*H,
        owner_main_KV_carrier_sender_content_bytes=kv+3*2*n*D,
        owner_main_context_and_KV_sender_content_bytes=kv+2*n*H+3*2*n*D,
        owner_main_projection_work_ratio=Fraction(N,M),
        early_feature_shard_packet_sender_content_bytes=aux+kv+2*N*(H//P),
        early_feature_shard_extra_feature_bytes=2*(N-M)*(H//P),
        full_main_weight_bytes=2*(3*H)*H,current_main_weight_shard_bytes=2*(3*H)*(H//P),
        extra_main_weight_per_card_bytes=2*(3*H)*H*(P-1)//P,
        feature_shard_FP32_accumulator_bytes=4*N*(H//P),
        three_full_aux_snapshot_content_bytes=3*2*N*H,
        owner_FP32_accumulator_bytes=4*n*H,three_local_aux_snapshot_content_bytes=3*2*n*H,
        physical_latency_or_wire_or_MFU_established=False)
    if u_max is not None:
        assert 0<=u_max<=n
        out.update(late_aux_owner_max_unique_rows=u_max,
            late_selected_sender_content_bytes=kv+3*2*H*u_max+feature,
            late_selected_op_reference=5,
            late_exact_selector_cannot_gate_current_attention_packet=True)
    out['owner_main_projection_work_ratio']=[out['owner_main_projection_work_ratio'].numerator,out['owner_main_projection_work_ratio'].denominator]
    return out


def canonical(I,N):
    assert N>0 and all(-N<=i<N for i in I)
    return [i%N for i in I]


def row_plan(I,N,P):
    assert N%P==0;n=N//P;norm=canonical(I,N);per=[set()for _ in range(P)]
    for i in norm:per[i//n].add(i%n)
    return dict(canonical_indices=norm,unique_per_owner=[sorted(s)for s in per],
        unique_rows=sum(map(len,per)),u_max=max(map(len,per),default=0),
        fixed_count_rows_over_all_senders=P*max(map(len,per),default=0))


def nonlinear_row_f(aux):
    # A deterministic nonlinear row function, used only to test selection/order
    # identities. It does not simulate GEMM/RMSNorm/BF16 arithmetic.
    x=[v for r in aux for v in r];a=sum((j+1)*v for j,v in enumerate(x))
    return (a*a,sum(v*v for v in x))


def row_selection_cases():
    records=[]
    for P in range(1,5):
        for n in range(1,4):
            N=P*n;aux=[[(101*k+3*i,101*k-2*i)for i in range(N)]for k in range(3)]
            domain=sorted(set([0,N//2,N-1,-1,-N]))
            full=[nonlinear_row_f([aux[k][i]for k in range(3)])for i in range(N)]
            for M in range(5):
                for I in product(domain,repeat=M):
                    norm=canonical(I,N);direct=[nonlinear_row_f([aux[k][i]for k in range(3)])for i in norm]
                    gathered=[full[i]for i in norm];assert direct==gathered
                    plan=row_plan(I,N,P)
                    # Transport each unique owner-local value once,then restore
                    # the complete original order/multiplicity.
                    bank={(p,j):full[p*n+j]for p,ids in enumerate(plan['unique_per_owner'])for j in ids}
                    rebuilt=[bank[(i//n,i%n)]for i in norm];assert rebuilt==direct
                    records.append(dict(P=P,N=N,I=list(I),M=M,plan=plan,ordered_row_function_identity=True))
    return records


def cycle(edges):
    adj={}
    for a,b in edges:adj.setdefault(a,set()).add(b);adj.setdefault(b,set())
    active=set();done=set()
    def visit(a):
        if a in active:return True
        if a in done:return False
        active.add(a)
        if any(visit(b)for b in adj[a]):return True
        active.remove(a);done.add(a);return False
    return any(visit(a)for a in adj)


def bf16(x):
    u=struct.unpack('<I',struct.pack('<f',x))[0]
    u=(u+0x7fff+((u>>16)&1))&0xffff0000
    return struct.unpack('<f',struct.pack('<I',u))[0]
