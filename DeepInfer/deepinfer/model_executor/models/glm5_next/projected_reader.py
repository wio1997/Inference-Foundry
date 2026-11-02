# SPDX-License-Identifier: Apache-2.0
"""Explicit projected CPU reader for private text candidate frames.

No implicit attention/INT8/TP backend, model factory or native ownership. Plans
are bound by the composite request to one attempt, layer and candidate table.
"""
from dataclasses import dataclass
import torch
from .sfa_inputs import SFAHeadPartition,NoPESFAContract,NoPESFAInputPlan
from .rank_dsa import CPUProjectedDSAResult


@dataclass(frozen=True)
class CPUProjectedDSAReader:
    layer_id: int
    heads: SFAHeadPartition
    contract: NoPESFAContract
    operation: object
    query_payload: object = None
    query_backend: object = None

    def __post_init__(self):
        if type(self.layer_id) is not int or self.layer_id not in range(3,44,4):raise ValueError('explicit GLM DSA reader layer required')
        if type(self.heads) is not SFAHeadPartition or type(self.contract) is not NoPESFAContract or self.heads.total_heads!=self.contract.total_heads or self.heads.local_heads>4:raise ValueError('matching bounded rank/source contract required')
        if not callable(self.operation):raise ValueError('explicit projected reader operation required')
        if (self.query_payload is None)!=(self.query_backend is None):raise ValueError('rank query payload/backend must be supplied together')
        if self.query_payload is not None:
            from .query_payload import PreparedRankQB
            if type(self.query_payload) is not PreparedRankQB or not callable(self.query_backend):raise ValueError('explicit prepared rank query/backend required')
            m=self.query_payload.manifest
            if (m.layer_id,m.heads,m.contract)!=self.specification:raise ValueError('rank query payload/reader differs')

    @property
    def specification(self):return self.layer_id,self.heads,self.contract

    def validate_config(self,config,layer_id):
        if layer_id!=self.layer_id or config.hidden_size!=4096 or config.mla_dimensions['q_lora_rank']!=1536 or config.mla_dimensions['v_head_dim']!=256 or config.layers[layer_id].attention!='deepseek_sparse_attention':raise ValueError('projected reader layer/config geometry differs')
        if NoPESFAContract.from_config(config,reference_source_sha256=self.contract.reference_source_sha256)!=self.contract:raise ValueError('projected reader source/config differs')

    def bind_candidate(self,candidate,*,request_id,revision,attempt_owner):
        """The composite owner supplies a fresh nonce for every step attempt."""
        return NoPESFAInputPlan(request_id,self.layer_id,revision,candidate,self.heads,self.contract,_owner=attempt_owner)

    @torch.inference_mode()
    def __call__(self,step,view,*,plan,expected_identity):
        if type(plan) is not NoPESFAInputPlan or (plan.layer_id,plan.heads,plan.contract)!=self.specification:raise ValueError('projected reader layer/partition/source differs')
        view.require_sfa_plan_cpu(plan,expected_identity=expected_identity)
        value=self.operation(step,view,plan=plan,expected_identity=expected_identity)
        if type(value) is not CPUProjectedDSAResult or value.identity!=expected_identity:raise ValueError('projected reader result identity/type differs')
        hidden=value.hidden
        if not isinstance(hidden,torch.Tensor) or hidden.device.type!='cpu' or hidden.dtype!=torch.bfloat16 or hidden.shape!=(plan.candidate.tokens,4096) or not bool(torch.isfinite(hidden).all()):raise ValueError('projected reader requires finite CPU BF16 [T,4096]')
        return CPUProjectedDSAResult(value.identity,hidden.contiguous().clone())
