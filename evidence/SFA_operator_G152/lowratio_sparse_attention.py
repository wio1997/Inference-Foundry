"""Vector-only Ascend sparse attention candidate; same supplied C1/C2 and sink semantics.

Each program owns one query head. This removes the prior four output-feature
score recomputations and Cube/vector synchronization, while retaining BF16 KV
and FP32 online softmax. Actual latency and complete model integration are unverified.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _sparse_shared_kv(Q,ORI,CMP,OI,CI,SINK,OUT,
                      Q_ROW:tl.constexpr,ORI_ROW:tl.constexpr,CMP_ROW:tl.constexpr,
                      OI_ROW:tl.constexpr,CI_ROW:tl.constexpr,OUT_ROW:tl.constexpr,
                      DIM:tl.constexpr,SWA:tl.constexpr,TOPK:tl.constexpr,
                      SCALE:tl.constexpr,BD:tl.constexpr,BN:tl.constexpr):
    row=tl.program_id(0).to(tl.int64);head=tl.program_id(1).to(tl.int64)
    d=tl.arange(0,BD);n=tl.arange(0,BN)
    q=tl.load(Q+row*Q_ROW+head*DIM+d,d<DIM,other=0).to(tl.float32)
    m=tl.load(SINK+head).to(tl.float32)
    denominator=tl.where(m==float('-inf'),0.,1.)
    acc=tl.full((BD,),0,tl.float32)
    for block in range(tl.cdiv(SWA+TOPK,BN)):
        col=block*BN+n
        oi=tl.load(OI+row*OI_ROW+col,col<SWA,other=-1)
        ci=tl.load(CI+row*CI_ROW+col-SWA,(col>=SWA)&(col<SWA+TOPK),other=-1)
        ko=tl.load(ORI+oi.to(tl.int64)[:,None]*ORI_ROW+d[None,:],(oi[:,None]>=0)&(col[:,None]<SWA)&(d[None,:]<DIM),other=0)
        kc=tl.load(CMP+ci.to(tl.int64)[:,None]*CMP_ROW+d[None,:],(ci[:,None]>=0)&(col[:,None]>=SWA)&(col[:,None]<SWA+TOPK)&(d[None,:]<DIM),other=0)
        k=tl.where((col<SWA)[:,None],ko,kc).to(tl.float32)
        valid=tl.where(col<SWA,oi>=0,(col<SWA+TOPK)&(ci>=0))
        score=tl.sum(k*q[None,:],1)*SCALE
        score=tl.where(valid,score,float('-inf'))
        next_m=tl.maximum(m,tl.max(score,0))
        alpha=tl.exp(tl.where(next_m==float('-inf'),0.,m-next_m))
        shifted=tl.where(next_m==float('-inf'),float('-inf'),score-next_m)
        p=tl.exp(shifted)
        acc=acc*alpha+tl.sum(p[:,None]*k,0)
        denominator=denominator*alpha+tl.sum(p,0)
        m=next_m
    out=acc/tl.where(denominator>0,denominator,1.)
    tl.store(OUT+row*OUT_ROW+head*DIM+d,out,d<DIM)


def sparse_shared_kv(q,original_kv,compressed_kv,original_indices,compressed_indices,sink,scale=None):
    assert q.dtype==original_kv.dtype==compressed_kv.dtype==torch.bfloat16
    assert q.ndim==3 and q.shape[-1]==512 and q.shape[1] in (8,16,32,64)
    assert original_kv.ndim==compressed_kv.ndim==2
    assert original_kv.shape[1]==compressed_kv.shape[1]==512
    assert original_indices.shape==(q.shape[0],128) and compressed_indices.shape==(q.shape[0],512)
    assert original_indices.dtype==compressed_indices.dtype==torch.int32
    assert sink.dtype==torch.float32 and sink.shape==(q.shape[1],)
    assert all(x.is_contiguous() and x.device==q.device for x in [q,original_kv,compressed_kv,original_indices,compressed_indices,sink])
    result=torch.empty_like(q)
    if q.shape[0]:
        _sparse_shared_kv[(q.shape[0],q.shape[1])](q,original_kv,compressed_kv,original_indices,compressed_indices,sink,result,
            q.stride(0),original_kv.stride(0),compressed_kv.stride(0),original_indices.stride(0),compressed_indices.stride(0),result.stride(0),
            512,128,512,float(scale or 512**-.5),512,8)
    return result
