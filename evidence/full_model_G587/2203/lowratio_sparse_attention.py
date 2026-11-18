"""Sparse shared-KV attention with scalar bank bases and bounded gather offsets.

Prepare pointer tables outside graph capture. Pools and tables stay alive across
replays. Per-bank addresses follow the already validated Engram UVA pointer ABI;
selected rows remain BF16 and the attention sink is counted once.
"""
import torch
import triton
import triton.language as tl

BANK_ROWS=1<<20


def prepare_pool(pool):
    assert pool.dtype==torch.bfloat16 and pool.ndim==2 and pool.shape[1]==512 and pool.is_contiguous()
    identity=(pool.data_ptr(),tuple(pool.shape),tuple(pool.stride()),str(pool.device))
    current=getattr(pool,'_ds41_attention_banks',None)
    if current is not None:
        assert current[0]==identity
        return
    addresses=[pool.data_ptr()+row*pool.stride(0)*pool.element_size() for row in range(0,pool.shape[0],BANK_ROWS)]
    pointers=torch.tensor(addresses,dtype=torch.int64,device=pool.device)
    pool._ds41_attention_banks=(identity,pointers)


@triton.jit
def _sparse_shared_kv(Q,ORI_PTRS,CMP_PTRS,OI,CI,SINK,OUT,
                      Q_ROW:tl.constexpr,ORI_ROW:tl.constexpr,CMP_ROW:tl.constexpr,
                      OI_ROW:tl.constexpr,CI_ROW:tl.constexpr,OUT_ROW:tl.constexpr,
                      OBANKS:tl.constexpr,CBANKS:tl.constexpr,BANK:tl.constexpr,
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
        ko=tl.full((BN,BD),0,tl.float32)
        kc=tl.full((BN,BD),0,tl.float32)
        for bank in range(OBANKS):
            in_bank=(col<SWA)&(oi>=bank*BANK)&(oi<(bank+1)*BANK)
            if tl.sum(in_bank.to(tl.int32),0)>0:
                base=tl.load(ORI_PTRS+bank).to(tl.pointer_type(tl.bfloat16))
                local=tl.where(in_bank,oi-bank*BANK,0)
                part=tl.load(base+local[:,None]*ORI_ROW+d[None,:],in_bank[:,None]&(d[None,:]<DIM),other=0)
                ko+=part.to(tl.float32)
        for bank in range(CBANKS):
            in_bank=(col>=SWA)&(col<SWA+TOPK)&(ci>=bank*BANK)&(ci<(bank+1)*BANK)
            if tl.sum(in_bank.to(tl.int32),0)>0:
                base=tl.load(CMP_PTRS+bank).to(tl.pointer_type(tl.bfloat16))
                local=tl.where(in_bank,ci-bank*BANK,0)
                part=tl.load(base+local[:,None]*CMP_ROW+d[None,:],in_bank[:,None]&(d[None,:]<DIM),other=0)
                kc+=part.to(tl.float32)
        k=tl.where((col<SWA)[:,None],ko,kc)
        valid=tl.where(col<SWA,oi>=0,(col<SWA+TOPK)&(ci>=0))
        score=tl.sum(k*q[None,:],1)*SCALE
        score=tl.where(valid,score,float('-inf'))
        next_m=tl.maximum(m,tl.max(score,0))
        alpha=tl.exp(tl.where(next_m==float('-inf'),0.,m-next_m))
        p=tl.exp(tl.where(next_m==float('-inf'),float('-inf'),score-next_m))
        acc=acc*alpha+tl.sum(p[:,None]*k,0)
        denominator=denominator*alpha+tl.sum(p,0);m=next_m
    result=acc/tl.where(denominator>0,denominator,1.)
    tl.store(OUT+row*OUT_ROW+head*DIM+d,result,d<DIM)


def sparse_shared_kv(q,original_kv,compressed_kv,original_indices,compressed_indices,sink,scale=None):
    assert q.dtype==original_kv.dtype==compressed_kv.dtype==torch.bfloat16
    assert q.ndim==3 and q.shape[-1]==512 and q.shape[1] in (8,16,32,64)
    assert original_indices.shape==(q.shape[0],128) and compressed_indices.shape==(q.shape[0],512)
    assert original_indices.dtype==compressed_indices.dtype==torch.int32
    assert sink.dtype==torch.float32 and sink.shape==(q.shape[1],)
    assert all(x.is_contiguous() and x.device==q.device for x in [q,original_kv,compressed_kv,original_indices,compressed_indices,sink])
    tables=[]
    for pool in [original_kv,compressed_kv]:
        cached=getattr(pool,'_ds41_attention_banks',None)
        if cached is None:raise RuntimeError('Call prepare_pool on persistent pool views before graph capture')
        assert cached[0]==(pool.data_ptr(),tuple(pool.shape),tuple(pool.stride()),str(pool.device))
        tables.append(cached[1])
    result=torch.empty_like(q)
    if q.shape[0]:
        _sparse_shared_kv[(q.shape[0],q.shape[1])](q,*tables,original_indices,compressed_indices,sink,result,
            q.stride(0),original_kv.stride(0),compressed_kv.stride(0),original_indices.stride(0),compressed_indices.stride(0),result.stride(0),
            tables[0].numel(),tables[1].numel(),BANK_ROWS,512,128,512,float(scale or 512**-.5),512,8)
    return result
