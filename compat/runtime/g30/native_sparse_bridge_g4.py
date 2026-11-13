"""Keep native target/draft SparseMLA on one explicitly bound donor ABI.

CPU mirrors are native scheduling hints; the actual donor metadata call uses
its tensor inputs. Main forward/DSpark graph math remains native.
"""
from functools import cache

# Provider-owned ABI: actual_sparse_mla.h vllm_ascend::SMLA_METADATA_SIZE.
# Paired with the same task-specific donor bindings used by native_sparse_ops.
_SPARSE_METADATA_SIZE = 1024


def native_sparse_metadata_size():
    return _SPARSE_METADATA_SIZE



@cache
def native_sparse_ops():
    from a3_donor_bindings_g4 import make_bindings
    b=make_bindings()
    def metadata(*args,**kwargs):
        # These two are SDK wrapper hints, absent from the donor op schema.
        cu_cpu=kwargs.pop('cu_seqlens_q_cpu',None)
        seq_cpu=kwargs.pop('seqused_ori_kv_cpu',None)
        if cu_cpu is not None and 'max_seqlen_q' not in kwargs:
            kwargs['max_seqlen_q']=int((cu_cpu[1:]-cu_cpu[:-1]).max().item()) if len(cu_cpu)>1 else 0
        if seq_cpu is not None and 'max_seqlen_ori_kv' not in kwargs:
            kwargs['max_seqlen_ori_kv']=int(seq_cpu.max().item()) if seq_cpu.numel() else 0
        if not kwargs.get('has_cmp_kv',False):
            # Canonical actual-donor SWA profile: no compression plane.
            kwargs['cmp_ratio']=0
            kwargs.setdefault('max_seqlen_cmp_kv',0)
        value=b.sparse_metadata(*args,**kwargs)
        if tuple(value.shape)!=(_SPARSE_METADATA_SIZE,):
            raise RuntimeError('Bound SparseMLA provider returned an unexpected metadata ABI shape')
        return value
    def attention(q,**kwargs):
        if kwargs.get('cmp_kv') is None:kwargs['cmp_ratio']=0
        return b.sparse(q,**kwargs)
    return metadata,attention
