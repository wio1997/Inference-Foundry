# SPDX-License-Identifier: Apache-2.0
"""Opt-in bounded diagnostic distributed seams for an assembled text runner.

Only dense gate/up rows and routed experts are sharded. Attention, router,
embedding/head, down projections and residual/state are replicated. No model
registry or tensor-parallel guard is bypassed. On installation error the
original modules are restored; this helper does not own request/process groups
or a caller-supplied reusable cache session, which the caller must close.
"""
from .decoder import GlmDenseMLP
from .expert_bank import CheckpointSelectedExperts
from .hybrid_decoder import GlmMoEMLP
from .rank_dense import install_rank_dense
from .rank_linear import RankW8A8OutputLinear
from .rank_expert_adapter import install_rank_experts

def install_rank_text_seams(runner,reader,*,rank,world_size,cache_bytes,
                            max_assembled_bytes,max_intermediate_rows,max_packed_rows,group=None,cache_session=None):
    if type(world_size) is not int or world_size not in (1,2,4,8,16) or type(rank) is not int or not 0<=rank<world_size:
        raise ValueError('explicit supported rank/world required')
    if world_size>1:
        import hashlib,json,torch,torch.distributed as dist
        if not dist.is_initialized() or dist.get_world_size(group)!=world_size or dist.get_rank(group)!=rank:
            raise ValueError('explicit initialized matching process group required')
        failure=None;signature=[0,0,0,0];count=0
        try:
            layers=runner.stack.layers
            if len(layers)!=len(runner.stack.config.layers):raise ValueError('complete native backbone required')
            from .state_archive import _layout_fingerprint
            from .cache import DSACacheLayout
            layout=runner.stack.layout
            expected_layout=DSACacheLayout.from_config(runner.stack.config,logical_block_size=layout.logical_block_size,lookahead=layout.lookahead)
            layout_hash=_layout_fingerprint(layout)
            if layout_hash!=_layout_fingerprint(expected_layout):raise ValueError('cache layout disagrees with config')
            shape=[]
            for layer,spec in zip(layers,runner.stack.config.layers,strict=True):
                if layer.attention_kind!=spec.attention:raise ValueError('attention layer disagrees with config')
                if spec.attention=='deepseek_sparse_attention':
                    if _layout_fingerprint(layer.attention.layout)!=layout_hash or _layout_fingerprint(layer.attention.indexer.layout)!=layout_hash:
                        raise ValueError('DSA attention layout disagrees with stack')
                if type(layer.layer_id) is not int or layer.layer_id<0 or layer.layer_id!=spec.layer_id:raise ValueError('invalid layer id')
                if isinstance(layer.mlp,GlmDenseMLP):
                    if spec.mlp!='dense':raise ValueError('dense layer disagrees with config')
                    width=layer.mlp.intermediate_size
                    if type(layer.mlp) is not GlmDenseMLP or type(width) is not int or width<1:
                        raise ValueError('checkpoint dense decoder and positive width required')
                    if type(max_intermediate_rows) is not int or max_intermediate_rows<width:
                        raise ValueError('explicit intermediate bound required')
                    if isinstance(layer.mlp.gate_proj,RankW8A8OutputLinear) or isinstance(layer.mlp.up_proj,RankW8A8OutputLinear):
                        raise ValueError('rank projections already installed')
                    shape.append(['dense',layer.layer_id,width,spec.attention])
                elif isinstance(layer.mlp,GlmMoEMLP) and isinstance(layer.mlp.experts,CheckpointSelectedExperts):
                    if spec.mlp!='sparse':raise ValueError('sparse layer disagrees with config')
                    shape.append(['sparse',layer.layer_id,spec.attention])
                else:raise ValueError('unsupported or already installed layer')
            # Replicated ranks must agree on dimensions and runtime model settings too.
            digest=hashlib.sha256(json.dumps({'layers':shape,'config':runner.stack.config.to_dict(),'layout_sha256':layout_hash},sort_keys=True,separators=(',',':')).encode()).digest()
            signature=[int.from_bytes(digest[i:i+8],'big',signed=True) for i in range(0,32,8)]
            count=len(shape)
        except Exception as exc:failure=exc
        metadata=torch.tensor([int(failure is not None),count,*signature],dtype=torch.int64,device='npu')
        gathered=[torch.empty_like(metadata) for _ in range(world_size)]
        dist.all_gather(gathered,metadata,group=group)
        if any(int(item[0].item()) for item in gathered) or not all(torch.equal(gathered[0],item) for item in gathered[1:]):
            raise RuntimeError('rank text installation structure differs or is invalid') from failure
    else:
        layers=runner.stack.layers
        if len(layers)!=len(runner.stack.config.layers):raise ValueError('complete native backbone required')
    saved=[];dense=[];sparse=[]
    try:
        for layer in layers:
            if isinstance(layer.mlp,GlmDenseMLP):
                saved.append(('dense',layer,layer.mlp.gate_proj,layer.mlp.up_proj))
                install_rank_dense(layer,reader,'npu',rank=rank,world_size=world_size,
                                   max_intermediate_rows=max_intermediate_rows,group=group)
                dense.append(layer.layer_id)
            elif isinstance(layer.mlp,GlmMoEMLP) and isinstance(layer.mlp.experts,CheckpointSelectedExperts):
                saved.append(('sparse',layer,layer.mlp.experts,None))
                # A sparse adapter can reject rank-local reader/session state.
                # Agree before any peer can enter the next dense collective.
                if world_size>1:
                    import torch,torch.distributed as dist
                    if not dist.is_initialized() or dist.get_world_size(group)!=world_size or dist.get_rank(group)!=rank:
                        raise ValueError('explicit initialized matching process group required')
                    status=torch.zeros((1,),dtype=torch.int32,device='npu')
                failure=None
                try:
                    install_rank_experts(layer,reader,rank=rank,world_size=world_size,cache_bytes=cache_bytes,
                        max_assembled_bytes=max_assembled_bytes,max_packed_rows=max_packed_rows,group=group,cache_session=cache_session)
                    if world_size>1:
                        import torch_npu
                        torch_npu.npu.synchronize()
                except Exception as exc:failure=exc
                if world_size>1:
                    status.fill_(int(failure is not None));dist.all_reduce(status,op=dist.ReduceOp.MAX,group=group)
                    if int(status.item()):raise RuntimeError('one or more ranks failed sparse installation') from failure
                elif failure is not None:raise failure
                sparse.append(layer.layer_id)
            else:raise ValueError('unsupported or already installed layer')
    except BaseException:
        for kind,layer,old,up in reversed(saved):
            if kind=='dense':layer.mlp.gate_proj,layer.mlp.up_proj=old,up
            else:layer.mlp.experts=old
        raise
    return {'dense_layers':dense,'sparse_layers':sparse,'rank':rank,'world_size':world_size}
