"""Ascend Triton sparse shared-KV attention, preserving supplied latent and index semantics.

The candidate lists must already contain exactly the causal SWA and selected compressed
physical rows, with -1 for padding. A finite attention sink enters the denominator once.
This module does not generate candidates, compress latents, or change KV precision.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _sparse_shared_kv(Q,ORI,CMP,OI,CI,SINK,OUT,
                      Q_ROW:tl.constexpr,ORI_ROW:tl.constexpr,CMP_ROW:tl.constexpr,
                      OI_ROW:tl.constexpr,CI_ROW:tl.constexpr,OUT_ROW:tl.constexpr,
                      HEADS:tl.constexpr,DIM:tl.constexpr,SWA:tl.constexpr,TOPK:tl.constexpr,
                      SCALE:tl.constexpr,BH:tl.constexpr,BD:tl.constexpr,BN:tl.constexpr):
    row=tl.program_id(0)
    h=tl.arange(0,BH);d=tl.arange(0,BD);n=tl.arange(0,BN)
    q=tl.load(Q+row*Q_ROW+h[:,None]*DIM+d[None,:],(h[:,None]<HEADS)&(d[None,:]<DIM),other=0)
    m=tl.load(SINK+h,h<HEADS,other=float('-inf')).to(tl.float32)
    denominator=tl.where(m==float('-inf'),0.,1.)
    acc=tl.full((BH,BD),0,tl.float32)
    for block in range(tl.cdiv(SWA+TOPK,BN)):
        col=block*BN+n
        oi=tl.load(OI+row*OI_ROW+col,col<SWA,other=-1)
        ci=tl.load(CI+row*CI_ROW+col-SWA,(col>=SWA)&(col<SWA+TOPK),other=-1)
        ko=tl.load(ORI+oi[None,:]*ORI_ROW+d[:,None],(oi[None,:]>=0)&(col[None,:]<SWA)&(d[:,None]<DIM),other=0)
        kc=tl.load(CMP+ci[None,:]*CMP_ROW+d[:,None],(ci[None,:]>=0)&(col[None,:]>=SWA)&(col[None,:]<SWA+TOPK)&(d[:,None]<DIM),other=0)
        k=tl.where((col<SWA)[None,:],ko,kc)
        valid=tl.where(col<SWA,oi>=0,(col<SWA+TOPK)&(ci>=0))
        score=tl.dot(q,k).to(tl.float32)*SCALE
        score=tl.where(valid[None,:]&(h[:,None]<HEADS),score,float('-inf'))
        next_m=tl.maximum(m,tl.max(score,1))
        delta=tl.where(next_m==float('-inf'),0.,m-next_m)
        alpha=tl.exp(delta)
        shifted=tl.where(next_m[:,None]==float('-inf'),float('-inf'),score-next_m[:,None])
        p=tl.exp(shifted)
        acc=acc*alpha[:,None]+tl.dot(p.to(k.dtype),tl.trans(k))
        denominator=denominator*alpha+tl.sum(p,1)
        m=next_m
    out=acc/tl.where(denominator>0,denominator,1.)[:,None]
    tl.store(OUT+row*OUT_ROW+h[:,None]*DIM+d[None,:],out,(h[:,None]<HEADS)&(d[None,:]<DIM))


def sparse_shared_kv(q,original_kv,compressed_kv,original_indices,compressed_indices,sink,scale=None):
    """q [T,H,512], pools [physical_rows,512], index matrices [T,128]/[T,512]."""
    assert q.dtype==original_kv.dtype==compressed_kv.dtype==torch.bfloat16
    assert q.ndim==3 and q.shape[-1]==512 and q.shape[1] in (8,16,32,64)
    assert original_kv.ndim==compressed_kv.ndim==2
    assert original_kv.shape[1]==compressed_kv.shape[1]==q.shape[-1]
    assert original_indices.shape==(q.shape[0],128) and compressed_indices.shape==(q.shape[0],512)
    assert original_indices.dtype==compressed_indices.dtype==torch.int32
    assert sink.dtype==torch.float32 and sink.shape==(q.shape[1],)
    assert all(x.is_contiguous() for x in [q,original_kv,compressed_kv,original_indices,compressed_indices,sink])
    assert all(x.device==q.device for x in [original_kv,compressed_kv,original_indices,compressed_indices,sink])
    result=torch.empty_like(q)
    if q.shape[0]:
        _sparse_shared_kv[(q.shape[0],)](q,original_kv,compressed_kv,original_indices,compressed_indices,sink,result,
            q.stride(0),original_kv.stride(0),compressed_kv.stride(0),original_indices.stride(0),compressed_indices.stride(0),result.stride(0),
            q.shape[1],q.shape[2],128,512,float(scale or q.shape[-1]**-.5),max(16,triton.next_power_of_2(q.shape[1])),512,64)
    return result
