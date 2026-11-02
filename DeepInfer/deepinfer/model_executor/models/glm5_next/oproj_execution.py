# SPDX-License-Identifier: Apache-2.0
"""Explicit CPU O-projection call contract; supplied stubs, no INT8 compute.

This preparatory seam passes private local NK codes/full output scales and BF16
rank inputs to a caller-supplied backend, then a caller-supplied sum reducer.
No native import/factory, quantization, GEMM, distributed collective, device copy
or default backend. Callback correctness/side effects are the caller's concern;
validation and copying do not attest numerical equivalence or roll back effects.
"""
from dataclasses import dataclass
import hashlib
import torch
from .oproj_payload import OProjColumnManifest,PreparedOProjColumns
from .sfa_inputs import NoPESFAInputPlan,SFAInputIdentity


def _positive(value,name):
    if type(value) is not int or value<1:raise ValueError('positive '+name+' required')


def _sha(value):
    return isinstance(value,str) and len(value)==64 and all(v in '0123456789abcdef' for v in value)


def _digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


@dataclass(frozen=True)
class RankOProjSpec:
    identity: SFAInputIdentity
    input_shape: tuple[int,int]
    output_shape: tuple[int,int]
    input_columns: tuple[int,int]
    world_size: int
    rank: int
    quantization_policy: str
    bias_policy: str
    descriptor_sha256: str
    weight_sha256: str
    runtime_scale_sha256: str
    source_scale_sha256: str
    source_offset_sha256: str


def _tensor(value,shape,dtype,name):
    if not isinstance(value,torch.Tensor) or value.device.type!='cpu' or value.dtype!=dtype or tuple(value.shape)!=shape:
        raise ValueError(name+' must be CPU '+str(dtype)+' '+str(shape))
    if dtype!=torch.int8 and not bool(torch.isfinite(value).all()):raise ValueError('nonfinite '+name)


@torch.inference_mode()
def execute_rank_oproj_cpu(payload,head_values,*,plan,expected_identity,expected_runtime_scale_sha256,quantization_policy,bias_policy,
                           rank_apply,reduce_sum,max_input_bytes=2*1024**2,
                           max_payload_bytes=8*1024**2,max_output_bytes=1024**2):
    """Exercise the explicit rank-apply then sum interface with CPU callbacks.

    Only rank_local_dynamic_int8 / absent-bias is described here. No prepared
    quantized-input shortcut or automatic policy selection. BF16 per-rank output
    must be full [T,4096] BEFORE reduce_sum; the reducer must return that shape.
    Both callbacks receive spec=RankOProjSpec. rank_apply additionally receives
    (private_input_bf16,private_weight_nk,private_scale). reduce_sum receives an
    isolated copy of the validated local output. Returned output is also owned.
    The caller supplies the expected BF16 scale digest from payload preparation.
    This rejects later scale mutation without rereading checkpoint payloads.
    Empty plans validate identity/metadata/input but skip payload hashing/copies
    and both callbacks. Exceptions publish no output; callback effects persist.
    Byte limits bound logical tensors, not process/HBM peak or callback memory.
    """
    if type(plan) is not NoPESFAInputPlan or type(expected_identity) is not SFAInputIdentity or expected_identity!=plan.identity:
        raise ValueError('stale/foreign O-projection plan identity')
    if not _sha(expected_runtime_scale_sha256):raise ValueError('explicit expected runtime scale digest required')
    if quantization_policy!='rank_local_dynamic_int8' or bias_policy!='absent':raise ValueError('explicit rank-local dynamic INT8 and absent-bias policies required')
    if not callable(rank_apply) or not callable(reduce_sum):raise ValueError('explicit rank backend and sum reducer required')
    for value,name in ((max_input_bytes,'input bound'),(max_payload_bytes,'payload bound'),(max_output_bytes,'output bound')):_positive(value,name)
    if type(payload) is not PreparedOProjColumns or type(payload.manifest) is not OProjColumnManifest:raise ValueError('prepared O-projection columns required')
    m=payload.manifest;h=plan.heads;k=h.local_heads*256;n=4096;t=plan.candidate.tokens
    if (m.layer_id,m.heads,m.contract)!=(plan.layer_id,h,plan.contract) or h.local_heads>4:raise ValueError('O-projection layer/partition/source differs')
    if (m.full_shape,m.first_column,m.stop_column,m.prefix)!=((n,64*256),h.head_ids[0]*256,(h.head_ids[-1]+1)*256,f'model.language_model.layers.{plan.layer_id}.self_attn.o_proj'):
        raise ValueError('O-projection column geometry differs')
    if (m.weight_bytes,m.checkpoint_bytes,m.runtime_bytes)!=(n*k,n*k+n*8,n*k+n*2):raise ValueError('O-projection byte manifest differs')
    if m.runtime_bytes>max_payload_bytes or t*n*2>max_output_bytes or t*k*2>max_input_bytes:raise ValueError('O-projection byte budget exceeded')
    if any(not _sha(v) for v in (m.descriptor_sha256,payload.weight_sha256,payload.source_scale_sha256,payload.source_offset_sha256)):raise ValueError('explicit payload provenance digests required')
    # Use the existing plan-bound copy and finite-input guard before any callback.
    x=payload.prepare_input_cpu(head_values,plan=plan,max_input_bytes=max_input_bytes)
    _tensor(payload.weight_nk,(n,k),torch.int8,'rank weight')
    _tensor(payload.weight_scale,(n,),torch.bfloat16,'full output scale')
    if not bool((payload.weight_scale>0).all()):raise ValueError('positive output scale required')
    if plan.skip_operator:return torch.empty((0,n),dtype=torch.bfloat16)
    weight=payload.weight_nk.contiguous().clone();scale=payload.weight_scale.contiguous().clone()
    if _digest(weight)!=payload.weight_sha256:raise ValueError('rank weight digest changed')
    if _digest(scale)!=expected_runtime_scale_sha256:raise ValueError('runtime scale digest changed')
    spec=RankOProjSpec(plan.identity,(t,k),(t,n),(m.first_column,m.stop_column),h.world_size,h.rank,
        quantization_policy,bias_policy,m.descriptor_sha256,payload.weight_sha256,_digest(scale),payload.source_scale_sha256,payload.source_offset_sha256)
    local=rank_apply(x,weight,scale,spec=spec)
    _tensor(local,(t,n),torch.bfloat16,'rank output')
    reduced=reduce_sum(local.contiguous().clone(),spec=spec)
    _tensor(reduced,(t,n),torch.bfloat16,'sum output')
    return reduced.contiguous().clone()
