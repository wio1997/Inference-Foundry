# SPDX-License-Identifier: Apache-2.0
"""Selected GLM expert bank adapter, with explicit global/compact namespaces.

A packed route batch is globally expert-sorted. A bank is stored in strictly
ascending global expert order; each compact cumulative count is therefore the
corresponding global cumulative count, provided every active expert is covered.
Inactive experts may be omitted or retained with zero rows. Missing active
experts are errors: this is not EP ownership, remote dispatch or MC2 support.

The native factory is prepared, not a claim of hardware validation. It keeps
ModelSlim INT8 weights/rounded BF16 scales and rejects CPU execution before
payload reads/importing native operators. CPU checks can inspect manifests
and exercise the adapter through independent grouped-FFN test doubles.
"""
from dataclasses import dataclass

import torch
from torch import nn

from .moe import PackedRoutes,pack_routes,dispatch_hidden,combine_packed


def _bank_ids(values,num_experts,device='cpu'):
    ids=torch.as_tensor(values,device=device)
    if ids.ndim!=1 or (ids.numel() and ids.dtype not in (torch.int32,torch.int64)):
        raise ValueError('bank IDs must be one-dimensional integers')
    ids=ids.long().clone()
    if bool(((ids<0)|(ids>=num_experts)).any()) or bool((ids[1:]<=ids[:-1]).any()):
        raise ValueError('bank global IDs must be unique, ascending and in range')
    return ids


@dataclass(frozen=True)
class CompactExpertPlan:
    routes: PackedRoutes
    global_expert_ids: torch.Tensor  # [bank_experts], INT64, in weight order
    local_expert_ids: torch.Tensor  # [packed_rows], INT64, compact namespace
    group_ends: torch.Tensor  # [bank_experts], INT64, group_list_type=0


def compact_expert_plan(routes,bank_expert_ids=None):
    if type(routes.num_experts)!=int or routes.num_experts<1 or type(routes.num_tokens)!=int or routes.num_tokens<0:
        raise ValueError('invalid global route dimensions')
    rows=routes.expert_ids.numel();device=routes.expert_ids.device
    tensors=(routes.token_indices,routes.expert_ids,routes.group_ends)
    if any(t.dtype!=torch.int64 or t.ndim!=1 or t.device!=device for t in tensors):
        raise ValueError('packed route indices/counts must be same-device INT64 vectors')
    if routes.token_indices.shape!=(rows,) or routes.weights.shape!=(rows,) or routes.weights.dtype!=torch.float32 or routes.weights.device!=device:
        raise ValueError('invalid packed route rows/weights')
    if routes.group_ends.shape!=(routes.num_experts,):
        raise ValueError('global group ends must cover the full expert namespace')
    if bool(((routes.expert_ids<0)|(routes.expert_ids>=routes.num_experts)).any()) or bool(((routes.token_indices<0)|(routes.token_indices>=routes.num_tokens)).any()):
        raise ValueError('packed route index out of range')
    if bool((routes.expert_ids[1:]<routes.expert_ids[:-1]).any()):
        raise ValueError('packed rows must be globally expert-sorted')
    counts=torch.bincount(routes.expert_ids,minlength=routes.num_experts)
    if not torch.equal(routes.group_ends,counts.cumsum(0)):
        raise ValueError('global cumulative counts disagree with packed expert rows')
    if not bool(torch.isfinite(routes.weights).all()) or bool((routes.weights<0).any()):
        raise ValueError('invalid packed route weights')
    ids=counts.nonzero().flatten() if bank_expert_ids is None else _bank_ids(bank_expert_ids,routes.num_experts,device)
    global_to_local=torch.full((routes.num_experts,),-1,dtype=torch.int64,device=device)
    global_to_local[ids]=torch.arange(ids.numel(),device=device,dtype=torch.int64)
    local=global_to_local[routes.expert_ids]
    if bool((local<0).any()):
        raise ValueError('compact bank is missing an active global expert')
    return CompactExpertPlan(routes,ids,local,routes.group_ends[ids])


def execute_compact(hidden_states,plan,forward_grouped,*,shared_output=None):
    """Call the existing expert bank ABI; apply route weights outside FFN once."""
    packed=dispatch_hidden(hidden_states,plan.routes)
    if not packed.shape[0]:
        return combine_packed(packed,plan.routes,shared_output=shared_output)
    output=forward_grouped(packed,plan.group_ends,group_list_type=0)
    if output.shape!=packed.shape or output.dtype!=packed.dtype or output.device!=packed.device:
        raise ValueError('grouped expert output must preserve packed row/hidden/dtype/device')
    return combine_packed(output,plan.routes,shared_output=shared_output)


class SelectedExpertBank(nn.Module):
    def __init__(self,global_expert_ids,bank,*,num_experts,shared=None):
        super().__init__()
        if type(num_experts)!=int or num_experts<1:
            raise ValueError('invalid expert namespace size')
        ids=_bank_ids(global_expert_ids,num_experts)
        if bool(ids.numel())!=(bank is not None):
            raise ValueError('empty bank IDs require no bank module; nonempty IDs require a bank')
        if bank is not None and getattr(bank,'count',ids.numel())!=ids.numel():
            raise ValueError('bank weight count does not match its global IDs')
        self.register_buffer('global_expert_ids',ids)
        self.bank,self.shared,self.num_experts=bank,shared,num_experts

    @torch.inference_mode()
    def forward(self,hidden_states,routes):
        packed=pack_routes(routes,num_experts=self.num_experts)
        compact=compact_expert_plan(packed,self.global_expert_ids)
        if hidden_states.ndim!=2 or hidden_states.shape[0]!=packed.num_tokens:
            raise ValueError('hidden token count does not match routes')
        # Empty requests must not instantiate/run an empty native expert bank.
        if hidden_states.shape[0]==0:
            return execute_compact(hidden_states,compact,None)
        shared=None if self.shared is None else self.shared(hidden_states)
        return execute_compact(hidden_states,compact,self.bank.forward_grouped,shared_output=shared)


@dataclass(frozen=True)
class ExpertSpec:
    global_id: int | None  # None denotes the one shared expert
    prefixes: tuple[str,str,str]  # gate/up/down, matching INT8 expert constructor
    shapes: tuple[tuple[int,int],tuple[int,int],tuple[int,int]]


@dataclass(frozen=True)
class ExpertManifest:
    num_experts: int
    hidden_size: int
    swiglu_limit: float | None
    routed: tuple[ExpertSpec,...]
    shared: ExpertSpec | None


def selected_expert_manifest(reader,layer_id,global_expert_ids,*,include_shared=True):
    """Validate metadata only; never read full INT8 matrices or dequantize."""
    from deepinfer.hugging_face.glm5_next_config import Glm5NextConfig
    c=Glm5NextConfig.from_checkpoint(reader.root);text=c.hf_config['text_config']
    if type(layer_id)!=int or not 0<=layer_id<c.num_hidden_layers or c.layers[layer_id].mlp!='sparse':
        raise ValueError('requested layer is not a GLM MoE backbone layer')
    if text['hidden_act']!='silu' or text['n_shared_experts']!=1:
        raise ValueError('selected GLM bank requires SiLU and one shared expert')
    count=text['n_routed_experts'];ids=_bank_ids(global_expert_ids,count).tolist()
    hidden,intermediate=c.hidden_size,text['moe_intermediate_size']
    shapes=((intermediate,hidden),(intermediate,hidden),(hidden,intermediate))
    root=f'model.language_model.layers.{layer_id}.mlp'
    def spec(global_id):
        stem=root+'.shared_experts' if global_id is None else f'{root}.experts.{global_id}'
        prefixes=tuple(stem+'.'+name for name in ('gate_proj','up_proj','down_proj'))
        for prefix,shape in zip(prefixes,shapes,strict=True):
            name=prefix+'.weight';d=reader.descriptor(name)[2]
            if reader.quant.get(name)!='W8A8_DYNAMIC' or d['dtype']!='I8' or tuple(d['shape'])!=shape:
                raise ValueError('selected experts require explicit INT8 W8A8_DYNAMIC geometry')
            for suffix in ('weight_scale','weight_offset'):
                d=reader.descriptor(prefix+'.'+suffix)[2]
                if d['dtype']!='F32' or tuple(d['shape'])!=(shape[0],1):
                    raise ValueError('selected expert scale/offset metadata must be F32 per output channel')
        return ExpertSpec(global_id,prefixes,shapes)
    return ExpertManifest(count,hidden,text.get('swiglu_limit'),tuple(spec(i) for i in ids),spec(None) if include_shared else None)


def load_selected_expert_bank(reader,layer_id,global_expert_ids,device,*,max_experts=8,include_shared=True,nz=True,
                              max_checkpoint_bytes=512*1024*1024,max_resident_bytes=512*1024*1024,shared_down_partitions=1,fused_guard=False,fused_reference=False):
    """Prepared native factory; caller must hold later authorization/resources.

    max_experts is a hard preparation/test allocation budget, never a truncation
    rule. INT8 payloads and BF16 rounded scales go directly to native operators.
    This is a bounded component factory, not a complete serving/expert-offload
    policy. All native execution remains separately unverified.
    """
    if torch.device(device).type!='npu':
        raise ValueError('native selected bank requires authorized NPU; no CPU dequantization fallback')
    if type(max_experts)!=int or max_experts<1:
        raise ValueError('max_experts must be a positive explicit allocation bound')
    manifest=selected_expert_manifest(reader,layer_id,global_expert_ids,include_shared=include_shared)
    if len(manifest.routed)>max_experts:
        raise ValueError('selected bank exceeds max_experts; no implicit truncation')
    # Enforce logical weight/sidecar budgets before reads or native imports.
    # This does not bound NZ padding, transient duplicate banks or peak HBM.
    from deepinfer.hugging_face.glm5_next_native import plan_expert_residency
    plan_expert_residency(manifest,max_checkpoint_bytes=max_checkpoint_bytes,
                          max_resident_bytes=max_resident_bytes)
    from deepinfer.npu_ops.w8a8 import AscendW8A8Expert,AscendW8A8ExpertBank
    if type(shared_down_partitions) is not int or shared_down_partitions < 1 or manifest.shared is not None and manifest.shared.shapes[2][1] % shared_down_partitions:
        raise ValueError('shared down partitions must evenly divide intermediate width')
    def load(spec):
        pairs=[reader.linear_payload(prefix) for prefix in spec.prefixes]
        if any(w.dtype!=torch.int8 or s is None or s.dtype!=torch.bfloat16 for w,s in pairs):
            raise ValueError('selected bank requires resident INT8 weights and BF16 scales')
        values=[value.to(device) for pair in pairs for value in pair]
        kwargs = {} if spec.global_id is not None or shared_down_partitions == 1 else {'down_partitions': shared_down_partitions}
        return AscendW8A8Expert(*values,limit=manifest.swiglu_limit,nz=nz,**kwargs)
    experts=[load(spec) for spec in manifest.routed]
    bank=AscendW8A8ExpertBank(experts) if experts else None
    if bank is not None: bank.fused_guard=fused_guard; bank.fused_reference=fused_reference
    shared=load(manifest.shared) if manifest.shared is not None else None
    wrapper=SelectedExpertBank([spec.global_id for spec in manifest.routed],bank,num_experts=manifest.num_experts,shared=shared)
    return wrapper.to(device).eval()


class CheckpointSelectedExperts(nn.Module):
    """Load exactly the routed token's global experts and shared expert on demand.

    Bounded correctness-oriented residency for a single token, not an expert
    parallel strategy, production offload scheduler, or performance claim.
    The bank is temporary and all selected weights retain ModelSlim INT8.
    """
    def __init__(self,reader,layer_id,device,*,num_experts,max_checkpoint_bytes,max_resident_bytes,shared_down_partitions=1,fused_guard=False,fused_reference=False):
        super().__init__()
        if type(num_experts) is not int or num_experts<1:raise ValueError('global expert count required')
        self.reader,self.layer_id,self.device,self.num_experts=reader,layer_id,device,num_experts
        self.max_checkpoint_bytes,self.max_resident_bytes=max_checkpoint_bytes,max_resident_bytes
        self.shared_down_partitions=shared_down_partitions
        self.fused_guard=fused_guard
        self.fused_reference=fused_reference

    @torch.inference_mode()
    def forward(self,hidden_states,routes):
        if hidden_states.ndim!=2 or hidden_states.shape[0]!=1:raise ValueError('selected expert residency supports one token per call')
        ids=sorted(set(int(i) for i in routes.expert_ids.detach().cpu().flatten().tolist()))
        if not ids or len(ids)>8:raise ValueError('up to eight selected routed experts required')
        kwargs = {} if self.shared_down_partitions == 1 else {'shared_down_partitions': self.shared_down_partitions}
        if self.fused_guard: kwargs['fused_guard']=True
        if self.fused_reference: kwargs['fused_reference']=True
        bank=load_selected_expert_bank(self.reader,self.layer_id,ids,self.device,max_experts=8,include_shared=True,
            max_checkpoint_bytes=self.max_checkpoint_bytes,max_resident_bytes=self.max_resident_bytes,**kwargs)
        value=bank(hidden_states,routes)
        import torch_npu
        torch_npu.npu.synchronize()
        return value
