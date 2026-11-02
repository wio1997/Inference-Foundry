# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-backed native text assembly; prepared and hardware UNVERIFIED.

Reuse existing GLM loaders and topology. This unsharded diagnostic factory has
no default DSA binding and is deliberately absent from the serving registry.
Caller must provide authorized native DSA operations and explicit expert byte
budgets. It is not a TP/EP residency strategy and cannot fit the full checkpoint
on a single small device merely because topology assembly is supported.
"""
import torch
from .cache import DSACacheLayout
from .decoder import GlmDenseMLP
from .expert_bank import selected_expert_manifest,load_selected_expert_bank,CheckpointSelectedExperts
from .hybrid_decoder import GlmHybridDecoder,GlmMoEMLP,PreparedDSAAttention,PreparedDecoderStack
from .kda import Glm5NextKDA
from .moe import GlmMoERouter
from .projections import GlmMLAProjections,GlmIndexerProjections
from .rank_dsa import CPUProjectedDSAOperation
from .text_boundary import GlmTextBoundary
from .text_request import PreparedTextRunner
from deepinfer.hugging_face.glm5_next_config import Glm5NextConfig
from deepinfer.hugging_face.glm5_next_plan import GlmBackbonePlan,BackboneOptions
from deepinfer.hugging_face.glm5_next_native import plan_expert_residency


def _native_device(device):
    if torch.device(device).type!='npu':raise ValueError('native model assembly requires authorized NPU; no CPU fallback')


def _dsa_binding(operation):
    if not callable(operation) or type(operation) is CPUProjectedDSAOperation:
        raise ValueError('explicit native full-head DSA operation required; CPU projected callbacks are unsupported')


def load_native_decoder(reader,layer_id,device,*,dsa_operation=None,
                        max_expert_checkpoint_bytes,max_expert_resident_bytes,
                        logical_block_size=128,config=None,tokenwise=True,expert_policy='all',dense_down_partitions=1,shared_down_partitions=1,fused_guard=False,fused_reference=False,hc_backend='torch',kda_backend='ascendc'):
    """Actual component loaders for one KDA/DSA and dense/all-expert layer.

    DSA operation must implement (DSAStep,DSAState)->(BF16[T,64,256],
    next DSAState), including cache update/selection/attention. No implicit
    source-only SFA wrapper or private CPU page route is installed here.
    Full global expert coverage prevents a selected bank missing routed IDs.
    """
    _native_device(device)
    if hc_backend not in ('torch','ascend_installed'):raise ValueError('unknown mHC backend')
    if kda_backend not in ('ascendc','ascendc_installed_decode'):raise ValueError('unknown native KDA backend')
    if kda_backend=='ascendc_installed_decode' and tokenwise is not True:raise ValueError('installed decode KDA requires tokenwise execution')
    if hc_backend == 'ascend_installed':
        for op in ('npu_hc_pre_v2','npu_hc_post'):
            if not hasattr(torch.ops._C_ascend,op):raise ValueError('installed mHC operators must be loaded explicitly')
    if type(tokenwise) is not bool:raise ValueError('explicit boolean tokenwise mode required')
    if expert_policy not in ('all','selected_per_token') or expert_policy=='selected_per_token' and not tokenwise:raise ValueError('selected expert residency requires tokenwise execution')
    c=Glm5NextConfig.from_checkpoint(reader.root) if config is None else config
    if type(layer_id) is not int or not 0<=layer_id<len(c.layers):raise ValueError('backbone layer ID required')
    spec=c.layers[layer_id]
    if spec.attention=='deepseek_sparse_attention':_dsa_binding(dsa_operation)
    elif dsa_operation is not None:raise ValueError('KDA layer does not accept a DSA operation')
    # Check all selected expert metadata/budgets before attention allocations.
    if spec.mlp=='sparse' and expert_policy=='all':
        ids=tuple(range(c.hf_config['text_config']['n_routed_experts']))
        manifest=selected_expert_manifest(reader,layer_id,ids,include_shared=True)
        plan_expert_residency(manifest,max_checkpoint_bytes=max_expert_checkpoint_bytes,max_resident_bytes=max_expert_resident_bytes)
    if spec.attention=='linear_attention':
        attention=Glm5NextKDA.from_checkpoint(reader,layer_id,device,backend=kda_backend)
    else:
        mla=GlmMLAProjections.from_checkpoint(reader,layer_id,device,logical_block_size=logical_block_size)
        indexer=GlmIndexerProjections.from_checkpoint(reader,c,layer_id,device,logical_block_size=logical_block_size)
        from .native_dsa import NativeShortDSA
        if isinstance(dsa_operation,NativeShortDSA):dsa_operation.bind(mla,indexer)
        attention=PreparedDSAAttention(mla,indexer,dsa_operation)
    if spec.mlp=='dense':
        mlp=GlmDenseMLP.from_checkpoint(reader,layer_id,device) if dense_down_partitions==1 else GlmDenseMLP.from_checkpoint(reader,layer_id,device,down_partitions=dense_down_partitions)
    else:
        total=c.hf_config['text_config']['n_routed_experts']
        if expert_policy=='all':
            kwargs = {} if shared_down_partitions == 1 else {'shared_down_partitions': shared_down_partitions}
            if fused_guard: kwargs['fused_guard']=True
            if fused_reference: kwargs['fused_reference']=True
            experts=load_selected_expert_bank(reader,layer_id,ids,device,max_experts=len(ids),include_shared=True,
                max_checkpoint_bytes=max_expert_checkpoint_bytes,max_resident_bytes=max_expert_resident_bytes,**kwargs)
        else:
            kwargs = {} if shared_down_partitions == 1 else {'shared_down_partitions': shared_down_partitions}
            if fused_guard: kwargs['fused_guard']=True
            if fused_reference: kwargs['fused_reference']=True
            experts=CheckpointSelectedExperts(reader,layer_id,device,num_experts=total,
                max_checkpoint_bytes=max_expert_checkpoint_bytes,max_resident_bytes=max_expert_resident_bytes,**kwargs)
        mlp=GlmMoEMLP(GlmMoERouter.from_checkpoint(reader,layer_id,device),experts,hidden_size=c.hidden_size,num_experts=total)
    layer=GlmHybridDecoder.from_checkpoint_components(reader,layer_id,attention=attention,mlp=mlp,device=device)
    layer.attn_hc.backend=hc_backend
    layer.ffn_hc.backend=hc_backend
    layer.native_tokenwise=tokenwise
    return layer


def load_native_text_runner(reader,device,*,dsa_operations,
                            max_expert_checkpoint_bytes,max_expert_resident_bytes,
                            vocab_start,vocab_stop,max_head_rows=32,max_tokens=8,
                            max_context_tokens=128,logical_block_size=128,expert_policy='all',dense_down_partitions=1,shared_down_partitions=1,fused_guard=False,fused_reference=False,hc_backend='torch',kda_backend='ascendc'):
    """Wire actual checkpoint factories through existing plan/stack/text runner.

    Missing DSA operations fail before any tensor payload read. Expert budgets
    cover the SUM across all sparse layers, before any native factory executes;
    per-layer loaders retain their own limits. Other weights, NZ padding,
    transient copies, activations and cache are excluded from these byte limits.
    A selected vocabulary shard is diagnostic, not full-vocabulary generation.
    Native numerical/ABI correctness and capacity remain unverified.
    """
    _native_device(device)
    if expert_policy not in ('all','selected_per_token'):raise ValueError('explicit expert residency policy required')
    c=Glm5NextConfig.from_checkpoint(reader.root)
    plan=GlmBackbonePlan.from_config(c,BackboneOptions(logical_block_size=logical_block_size))
    required={s.layer_id for s in c.layers if s.attention=='deepseek_sparse_attention'}
    if not isinstance(dsa_operations,dict) or set(dsa_operations)!=required:raise ValueError('native DSA operations must cover every configured DSA layer exactly')
    for operation in dsa_operations.values():_dsa_binding(operation)
    for value in (max_expert_checkpoint_bytes,max_expert_resident_bytes,max_head_rows,max_tokens,max_context_tokens):
        if type(value) is not int or value<1:raise ValueError('positive explicit native model bounds required')
    if type(vocab_start) is not int or type(vocab_stop) is not int or not 0<=vocab_start<vocab_stop<=c.vocab_size or vocab_stop-vocab_start>max_head_rows:raise ValueError('bounded diagnostic vocabulary shard required')
    checkpoint=resident=0
    for spec in c.layers:
        if spec.mlp=='sparse' and expert_policy=='all':
            manifest=selected_expert_manifest(reader,spec.layer_id,tuple(range(c.hf_config['text_config']['n_routed_experts'])),include_shared=True)
            budget=plan_expert_residency(manifest,max_checkpoint_bytes=max_expert_checkpoint_bytes,max_resident_bytes=max_expert_resident_bytes)
            checkpoint+=budget.checkpoint_payload_bytes;resident+=budget.resident_logical_bytes
    if checkpoint>max_expert_checkpoint_bytes or resident>max_expert_resident_bytes:raise ValueError('total native expert payload/residency budget exceeded')
    # The topology plan already verifies exact config/kinds/component retention.
    # Its factories now invoke real loaders; no fake layer assembly here.
    prepared={}
    def attention(spec):
        kwargs = {} if shared_down_partitions == 1 else {'shared_down_partitions': shared_down_partitions}
        if fused_guard: kwargs['fused_guard']=True
        if fused_reference: kwargs['fused_reference']=True
        layer=load_native_decoder(reader,spec.layer_id,device,dsa_operation=dsa_operations.get(spec.layer_id),
            max_expert_checkpoint_bytes=max_expert_checkpoint_bytes,max_expert_resident_bytes=max_expert_resident_bytes,
            logical_block_size=logical_block_size,config=c,expert_policy=expert_policy,dense_down_partitions=dense_down_partitions,hc_backend=hc_backend,kda_backend=kda_backend,**kwargs)
        prepared[spec.layer_id]=layer
        return layer.attention
    def ffn(spec):return prepared[spec.layer_id].mlp
    def decoder(spec,attention,ffn):return prepared.pop(spec.layer_id)
    layers=plan.assemble(c,attention_factories={k:attention for k in {s.attention for s in c.layers}},
        ffn_factories={k:ffn for k in {s.mlp for s in c.layers}},decoder_factory=decoder)
    layout=DSACacheLayout.from_config(c,logical_block_size=logical_block_size)
    stack=PreparedDecoderStack(c,layout,layers)
    boundary=GlmTextBoundary.from_checkpoint(reader,device,max_unique_tokens=max_tokens,max_tokens=max_tokens)
    head=boundary.head_shard(vocab_start,vocab_stop,max_rows=max_head_rows)
    return PreparedTextRunner(stack,boundary,head,max_context_tokens=max_context_tokens,max_requests=1,native_tokenwise=True).eval()
