"""Separate covered support wrappers from exclusive PMU ownership decisions."""
import json
from pathlib import Path
H=Path(__file__).resolve().parent;N=H.parent
x=json.loads((H/'profile-source-coverage-frontier.json').read_text())
generic_linear=['M07','M08','M09','M11','M13','M14','M16','M18','M20']
rows=[];unresolved=[]
for r in x['unassigned']:
    s=r['source']['enclosing_qualified_symbol'];p=r['source']['path'];mods=[];answer=None
    if '/distributed/' in p:
        mods=['M23'];answer='Collective/group wrapper;retain group,owner,rank,shape/dtype,reduction and readiness contract. This frame is not a second communication payload or an exclusive arithmetic owner.'
    elif s in ['AscendSpecDecodeBaseProposer._propose','AscendDSparkProposer.dummy_run','AscendDSparkProposer._clear_dummy_slot_mappings']:
        mods=['M18','M19','M20','M21','M22','M24'];answer='Proposer/dummy container;actual preparation,decoder,head and state publishers keep their own contracts. Empty or dummy scope does not imply all children/state/remote work are dead.'
    elif s in ['AscendDSAV41Impl.forward','AscendDSAV41CPImpl._prepare_inputs_and_caches','dsa_v41_forward','AscendDSAV41Impl._update_caches']:
        mods=['M07','M08','M09','M10','M11','M21','M24'];answer='Attention container/cache publication boundary;expanded SWA/C1/C2 templates and state ports retain all branches,not an extra primitive or proof of no side consumers.'
    elif s in ['_apply_dsv4_rope','ComplexExpRotaryEmbedding.forward','AscendDSAV41Impl._apply_rotary','BaseDeviceAdaptor.apply_partial_rotary_inplace']:
        mods=['M07','M10'];answer='Selected rotation/partial-rotation wrapper;preserve FP32/BF16 rounds,rotated subset,inverse convention,shape/layout and alias/publication conditions.'
    elif s=='scatter_cache_sk':
        mods=['M07','M08','M09','M24'];answer='Cache scatter helper;preserve selected slot/row bounds,old readers and new generation. Do not replace a whole-bank effect by a shape-only prefix assumption.'
    elif s in ['DSparkDeepseekV41ForCausalLM.markov_embed','DSparkDeepseekV41ForCausalLM.markov_bias']:
        mods=['M20'];answer='Ordered Markov embedding/bias boundary;keep seed/previous prediction order,selected IDs,weight generation and corrected score readers.'
    elif s=='DeepseekV41MoE.forward' or '/fused_moe/router/' in p:
        mods=['M12','M13','M23','M24'];answer='Router/MoE composition;keep logical/physical expert map,record/load state,quantizer distinctions and routed/shared merge. Existing F20-F23 contracts cover the boundary.'
    elif s=='BaseDeviceAdaptor.index_fill':
        mods=['M04','M17','M21'];answer='In-place indexed fill;preserve index range/value/dtype,output mutation/alias and all score/metadata readers. A small recorded shape is not a universal dead-write certificate.'
    elif 'FusedMoE' in s or s=='_moe_forward_shared':
        mods=['M12','M13'];answer='Selected quantized MoE adapter;do not count the adapter as another GEMM. Preserve routed/shared input scales,rounding,expert/row maps and native composite boundary.'
    elif 'compute_logits' in s:
        mods=['M16'];answer='Vocabulary-head delegation;same-shape target/draft parameters are not assumed equal. Preserve all requested/actual score consumers and owner/gather protocol.'
    elif 'embed_input_ids' in s or s=='UnquantizedEmbeddingMethod.embedding':
        mods=['M05','M20'];answer='Selected embedding delegation;use selected rows/IDs and original output rounding/ownership. Allocated weight table is not per-call traffic.'
    elif 'Linear' in s or 'linear' in p or s in ['unquantized_gemm','default_unquantized_gemm','UnquantizedEmbeddingMethod.apply']:
        mods=generic_linear;answer='Generic selected projection/quantizer adapter. Functional owner is determined by the containing invocation,not this shared frame. Preserve bias/return tuple,quantizer,layout and all output contracts.'
    elif s=='AscendRMSNorm.forward_oot':
        mods=['M06','M07','M08','M09','M15','M16','M18','M19'];answer='Generic norm override;keep native epsilon,gamma/bias,dtype/rounding,residual/fanout. Backend-name equality is not a changed-kernel bitwise proof.'
    elif '/compilation/' in p or s=='set_ascend_forward_context' or s=='NPUWorker.compile_or_warm_up_model':
        mods=['M00','M22','M23','M24'];answer='Compilation/capture/forward-context control;keep selected mode,dimensions,streams,constructor/config identities and exception cleanup. No extra device arithmetic is inferred from wrapper nesting.'
    elif s in ['AscendDeepseekV41LLMForCausalLM.forward','AscendDeepseekV41ForCausalLM.forward']:
        mods=['M05','M15','M16','M19'];answer='Model/wrapper entry;fixed request domain is text-only,while loaded vision parameters remain recorded. Expanded target/draft graph supplies compute;wrapper scope itself does not prove a new workload or zero cost.'
    elif s in ['CustomOp.forward','BasevLLMParameter.__torch_function__']:
        mods=['M00']+generic_linear;answer='Dispatch/parameter protocol wrapper;actual selected implementation or nested public op supplies semantics. No extra work or independence is credited from this generic frame.'
    elif '/executor/' in p or '/worker/worker' in p:
        mods=['M03','M04','M17','M21','M22','M24','M25'];answer='Worker/RPC transport boundary;preserve method/output-rank,errors,ordering,publication and lifetime. Device leaf work is accounted to its functional child;CPU/IPC cost remains unknown,not zero.'
    elif '/tracing/' in p:
        mods=['M00','M25'];answer='Tracing wrapper;preserve configured observers and context/error flow. It is not a tensor producer or evidence that all observers are absent.'
    if not mods:unresolved.append(r);continue
    rows.append({'frame':r['frame'],'source':r['source'],'operator_owner_scope':mods,
                 'operator_effect_answer':answer,'exclusive_PMU_owner_not_changed':True,
                 'authoritative_contract':str(N/'profile-first-20261031/module-phase-branch-matrix-v176.json')})
assert len(rows)==100 and not unresolved,(len(rows),[r['source'] for r in unresolved])
out={'status':'ALL100_GENERIC_SUPPORT_FRAMES_HAVE_OPERATOR_SCOPE_ANSWERS',
     'retained_source_index_frames':len(x['rows']),'actual_used_v164_frames':492,
     'retained_extra_index_frames_not_promoted_to_current_execution':13,
     'wrapper_rows':rows,'unresolved':unresolved,
     'does_not_change_exclusive_PMU_attribution_or_close_all_source_guards':True,
     'no_source_bodies_reread':True,'overall_complete':False}
(H/'source-wrapper-operator-scope-reconciliation.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':'PASS','wrapper_frames':len(rows),'unresolved':len(unresolved)}))
