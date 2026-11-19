"""Review candidate: rectangular topk1 eager KPool metadata only.

No scheduler, allocation, KDA or graph replacement. An integrating owner must
supply a validated exclusive request lane and no-padding contract explicitly.
"""
from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class EagerLaneContract:
    topk: int
    verify_width: int
    exclusive_request_lane: bool
    padded_requests: bool
    graph: bool = False
    def validate(self,capacity):
        if self.topk!=1 or self.verify_width<1 or self.graph or self.padded_requests or not self.exclusive_request_lane:
            raise NotImplementedError('Only exclusive, unpadded, rectangular topk1 eager lanes are covered')
        if capacity < 4+self.verify_width:
            raise ValueError('Require native conservative R+W ring capacity')


def build_eager_inputs(forward_batch,positions,raw_slots,reqids,capacity,*,contract=None):
    mode=forward_batch.forward_mode;seq=forward_batch.seq_lens.to(torch.int32)
    speculative=(mode.is_target_verify() or mode.is_draft_extend_v2() or
                 (mode.is_decode_or_idle() and forward_batch.spec_info is not None))
    if speculative:
        if contract is None:raise RuntimeError('MTP integration must bind its proven lane contract')
        contract.validate(capacity)
    n=seq.numel()
    if reqids.numel()!=n:raise ValueError('Request rows disagree')
    if mode.is_target_verify():
        width=int(forward_batch.spec_info.draft_token_num)
        if width!=contract.verify_width or positions.numel()!=n*width:raise ValueError('Target verify must be rectangular W')
        qlens=torch.full_like(seq,width);ends=seq+width
        expected=(seq[:,None]+torch.arange(width,device=seq.device)).reshape(-1)
        masked_positions=positions
    elif mode.is_draft_extend_v2():
        width=contract.verify_width
        if positions.numel()!=n*width:raise ValueError('Draft extend must retain the native padded window')
        accepted=forward_batch.spec_info.num_accept_tokens
        if accepted is None or accepted.numel()!=n:raise ValueError('Native accepted count including bonus is required')
        if not bool(((accepted>=1)&(accepted<=width)).all().item()):raise ValueError('Padded/invalid accepted rows are outside this candidate')
        qlens=torch.full_like(seq,width);ends=seq
        expected=(seq[:,None]-width+torch.arange(width,device=seq.device)).reshape(-1)
        valid=(torch.arange(width,device=seq.device)[None,:]<accepted[:,None]).reshape(-1)
        masked_positions=torch.where(valid,positions,torch.full_like(positions,-1))
    elif mode.is_decode_or_idle():
        if positions.numel()!=n:raise ValueError('Decode expects one row/request')
        qlens=torch.ones_like(seq);ends=positions.to(torch.int32)+1
        expected=positions;masked_positions=positions
    else:
        qlens=forward_batch.extend_seq_lens.to(torch.int32);ends=seq
        expected=positions;masked_positions=positions
    # Diagnostic checks intentionally synchronize: graph and perf acceptance separate.
    if speculative and (not torch.equal(positions.to(expected.dtype),expected) or torch.unique(reqids).numel()!=n):
        raise ValueError('Positions/duplicate request owners violate the lane contract')
    slots=torch.where(masked_positions>=0,raw_slots,torch.full_like(raw_slots,-1))
    return masked_positions,slots,ends,qlens
