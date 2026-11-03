"""Model HC-to-MoE epochs, typed rounding and overlap/epilogue alternatives."""
from pathlib import Path
from collections import Counter
from itertools import product
import hashlib,json,struct
from finalize_origin_coverage_v9 import N,O,W,read,write


def pin(p):
    p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def bf16(x):
    u=struct.unpack('>I',struct.pack('>f',x))[0]
    return struct.unpack('>f',struct.pack('>I',((u+0x7fff+((u>>16)&1))>>16)<<16))[0]


def schedule(p,fork=False,overlap=False,late_fused=False,price=False):
    z=max(p['attention_out'],p['attention_coeff'])+p['post1']
    pre4=p['attention_coeff']+p['pre4_delay']
    y=max(z,pre4)+p['collapse']
    c=z+p['ffn_coeff']+(p['overhead']if price else 0)
    u=(y if fork else max(y,c))+p['norm']
    r=u+p['router_topk_mc2']+(p['overhead']if price else 0)
    gate=(u if overlap else r)+p['shared_body']+(p['shared_penalty']if price else 0)
    if late_fused:
        return max(gate,r,c)+p['down']+p['merge']+p['post2']+(p['overhead']if price else 0)
    sd=gate+p['down']
    merged=max(sd,r)+p['merge']
    return max(merged,c)+p['post2']


def main():
    assert not(O/'REVIEW-CHECKPOINT-v141.json').exists()
    prof=read(O/'hc-moe-native-frontiers-v141.json');root=W/'frozen-source-snapshot/vllm-ascend/vllm_ascend';parts=[]
    for rel,a,b in [('models/deepseek_v41/model.py',250,329),('models/deepseek_v41/model.py',844,877),('models/deepseek_v41/model.py',887,920),
                    ('ops/fused_moe/fused_moe.py',213,245),('ops/fused_moe/fused_moe.py',390,420),('ops/fused_moe/router/fused_topk_router.py',194,227)]:
        p=root/rel;parts.append(dict(**pin(p),first_line=a,last_line=b,source=p.read_text().splitlines()[a-1:b]))
    facts=[
      'MoE does not consume pre-attention HC collapsed Y. Actual epoch is attention HCPost(Z1) -> FFN HCPre collapse(Yf,current attention pre4) -> RmsNormCast(U_BF16,U_FLOAT) -> router/MC2/shared. Same H5120 is not a value-equality witness.',
      'RmsNormCast FLOAT plane is widened rounded BF16 normalized value per reused v63 contract, not unrounded FP32 norm. Preserve BF16 normalization, FLOAT widening, router HF32/accumulation, TopK nonlinearity/biased rank/unbiased gathered weights and native tie/NaN/scale policies.',
      'FFN HCPre coefficient branch reads full Z1; early collapse publication need not wait its finish under a new native fork. Current attention pre4 remains prerequisite for FFN collapsed Y, and FFN coefficient/pre4 are needed by FFN HCPost/next layer respectively. Raw Z1 is still FFN coefficient input and FFNPost residual.',
      'Source final _maybe_reduce_final_output after merge is a separate guard: native stage adjacency alone does not close its active/inactive state. Down/Merge/HCPost full composition needs either no such reduction or a new reduction handoff; if active, keep it and price its publication before HCPost. Source tensor-return MoE path already preprocesses and may scale routed output before ForeachAdd. R denotes exactly that source input; never multiply a routed factor again. Tuple-return path is a separate unobserved branch.',
      'MC2 is already opaque DispatchFFNCombine: dispatch/quant/GMM1/activation/requant/GMM2/return/weighted unpermute are internal. Do not sum independent imagined services or claim deleting launches which are not there. Receiver valid processed assignments differ from owner-local M*k and physical capacity.',
      'Original shared branch uses INT32 Up, clamp10 DequantSwigluQuant with full intermediate-row amax, then BF16 Down. Routed MC2 internal FP16 boundary differs. Shared/MC2 quant reuse needs same actual row, full-H quantizer, scale/round/mask/record format and a new prequant ABI, not just matching INT8.',
      'Down->BF16 S; source ForeachAdd BF16(S+R)->BF16 T; HCPost then BF16 4H output are distinct rounding points. Epilogue composition may keep those casts in registers but cannot algebraically distribute through them. All coefficient, residual and incoming pre4 values stay live to their last readers.',
      'Full shared overlap can finish Down before routed R or FFN coefficients. Eliminating Down/Merge temporary interfaces by delaying Down or retaining on-chip state competes with that overlap; separate caps cannot be added. Occupancy, row quant reductions, pointer/capture and finite cross-rank progress remain gates.',
      'FFNPost->next HC collapse composition may avoid a chosen private store/read only after preserving BF16 Post round, next pre4 and all full4H coefficient/residual/Engram/aux/tail observers. No permission to erase full4H based on collapsed Y alone. Observed no intervening Compute does not prove all-reader/alias closure.',
      'Model schedules are sufficient chosen whole-frontier references, not required whole-operator barriers or optimal/physical bounds. Tile-specific owner/receiver progress, resource sharing and actual flags/ranges are unbound.'
    ]
    write(O/'hc-moe-source-contract-v141.json',dict(parts=parts,facts=facts,prior_answers=['v56 MC2 fused internals, alias and finite progress','v57 shared numerical/fusion/overlap alternatives','v63 router/RmsNormCast rounding','v78 SP+SEDP work/layout','v98 MC2 capacity16EP and receiver demand','v133 HC currentcoeff/Post/pre4 fork'],new_device_work=False,all_runtime_or_new_native_gates_closed=False,overall_complete=False))
    pairs=[['F01',x]for x in ['F20','F21','F22','F23']]
    contracts=[
      'HC attention/FFN epochs and rounded RmsNormCast bound router/TopK; early FFN collapse uses current attention pre4, cannot substitute pre-attention Y. FLOAT plane fusion requires rounding/HF32/fullnorm contracts.',
      'HC-early FFN input can feed opaque MC2 earlier while coefficients progress, but FFNPost still joins coefficients, routed progress and full residual; receiver hist/capacity/alias/epoch and prequant ABI govern split/reassembly.',
      'HC-early FFN input/shared overlap and typed shared fusion compose only with full-I row quant reduction, INT32/FP16 distinctions, MC2 competing resources and one actual input quantizer contract.',
      'HCPost/MoE merge/Down and next-layer collapse composition preserves BF16 Down->Merge->Post rounds, pre4 and full4H readers. Delayed fused Down trades branch overlap for interfaces; Engram/aux mutations/readers are guards.'
    ]
    ex=[['HG93-HC-ATTENTION-FORK-DEADLINE',x]for x in ['F20','F21','F22','F23']]+[['HG73-MOE','F01'],['HG84-HC-FFN-FORK','F01']]
    write(O/'hc-moe-composition-v141.json',dict(clauses=[dict(id='REL141-0'+str(i+1),a=a,b=b,contract=c)for i,((a,b),c)in enumerate(zip(pairs,contracts))],new_pairs=pairs,new_pair_clauses=4,group_extensions=ex,all_runtime_gates_proved=False,whole_pair_or_all_subsets_closed=False,speedup_additive=False,overall_complete=False))
    observed=[];geoms=Counter();mergebound=0;ordered=0
    for r in prof['records']:
        s=r['stages'];m=int(s['FFN_NORM']['definition']['outputShapes_text'].strip('"').split(',')[0]);pp=int(s['TOPK']['definition']['outputShapes_text'].strip('"').split(',')[0]);geoms[m,pp]+=1
        ordered+=r['current_shared_after_MC2_same_stream']is True;mergebound+=s['MERGE']['TASK']is not None
        observed.append(dict(key=[r['database_index'],r['forward_API_rowid'],r['source_block_ordinal']],M=m,P=pp,source_current_shared_after_MC2_same_stream=r['current_shared_after_MC2_same_stream'],merge_evidence='NATIVE_SOURCE_CID_TASK'if s['MERGE']['TASK']else'SOURCE_API_ONLY',
            shared_start_minus_MC2_end_ns=s['SHARED_QUANT']['TASK']['startNs']-s['MC2']['TASK']['endNs']if s['SHARED_QUANT']['TASK']and s['MC2']['TASK']else None,MC2_internal_stage_times_or_actual_counts=None))
    write(O/'hc-moe-observed-frontiers-v141.json',dict(frontiers=observed,geometry=[dict(M=m,P=pp,n=n)for(m,pp),n in sorted(geoms.items())],merge_TASK_bound=mergebound,shared_after_MC2_same_stream=ordered,
        API_only_roles=prof['API_only_frontiers'],interlayer_bridges=936,Engram_guards=prof['bridge_Engram_guards'],aux_guards=prof['bridge_aux_guards'],no_intervening_compute=prof['recorded_no_intervening_compute_TASK_bridges'],
        no_intervening_compute_is_not_reader_or_alias_closure=True,actual_routing_counts_or_native_fusion_gain=False,new_device_work=False,overall_complete=False))
    cases=[];classes=Counter();trade=Counter()
    for vals in product([2,13],[3,17],[1,4],[2,15],[5,24],[4,28],[3,12],[0,9],[1,6],[0,15],[0,6,30]):
        p=dict(zip(['attention_coeff','attention_out','collapse','ffn_coeff','router_topk_mc2','shared_body','down','pre4_delay','post2','shared_penalty','overhead'],vals),post1=2,norm=2,merge=1)
        ans=dict(baseline=schedule(p),FFN_early_only=schedule(p,True),shared_overlap_only=schedule(p,False,True),joint_ideal=schedule(p,True,True),joint_repriced=schedule(p,True,True,False,True),late_fused=schedule(p,True,True,True),late_fused_repriced=schedule(p,True,True,True,True))
        classes['FASTER'if ans['joint_repriced']<ans['baseline']else'SLOWER'if ans['joint_repriced']>ans['baseline']else'TIED']+=1
        trade['FUSED_LATER'if ans['late_fused']>ans['joint_ideal']else'FUSED_EQUAL'if ans['late_fused']==ans['joint_ideal']else'FUSED_EARLIER']+=1
        cases.append(dict(parameters=p,**ans))
    write(O/'hc-moe-readiness-DAGs-v141.json',dict(cases=cases,count=len(cases),joint_repriced_classes=dict(classes),fixed_service_fused_tradeoff=dict(trade),endpoint='FFN HCPost output;next HC/full4H reader costs omitted and remain gates',
        assumptions='Arbitrary positive hypothetical services, adequate resources/ready metadata/no native claim;full frontier is a sufficient chosen schedule;no intervening final reduction assumed for this reference only,active final reduction must stay or use a separately proved fused handoff. Repriced overhead applies FFNcoeff/route and fused endpoint, shared penalty only shared body. Fused Down waits R and coefficients, preserves rounds and original Down/merge/Post services.',not_physical_lower_or_upper_bound=True,actual_speedup=None,new_device_work=False,overall_complete=False))
    witnesses=dict(
      merge_round=dict(S=1.0,R=1/256,coefficient=1.5,rounded_merge=bf16(1+1/256),required_post=bf16(1.5*bf16(1+1/256)),illegal_distributed_post=bf16(1.5+1.5/256)),
      post_round=dict(pre_round_post=[1+1/256,1/256,0.,0.],rounded_post=[bf16(x)for x in [1+1/256,1/256,0.,0.]],premix=[1.,1.,1.,1.],required_collapse=bf16(sum(bf16(x)for x in [1+1/256,1/256,0.,0.])),illegal_skip_Post_round=bf16(1+2/256)),
      normcast_round=dict(unrounded_normalized=1+1/256,correct_FLOAT_plane=bf16(1+1/256),illegal_FLOAT_plane=1+1/256),
      lost_full4H=dict(X=[1,2,3,4],Xprime=[4,3,2,1],premix=[1,1,1,1],collapsed=10,constructed_coefficient_row=[1,0,0,0],coeff_X=1,coeff_Xprime=4,frozen_loaded_weight_witness=False),
      receiver_vs_owner=dict(local_owned_M=0,peer_assignments_received=7,local_topK=6,incorrect_receiver_work=0,correct_processed_assignments=7,skip_MC2_collective_legal=False),
      BF16_witness_domain='Finite round-to-nearest-even mathematical counterexamples;not loaded native/new fused parity or exact HC coefficient attainability proof',new_device_work=False,overall_complete=False)
    write(O/'hc-moe-rounding-and-owner-witnesses-v141.json',witnesses)
    work=[]
    for(m,pp),n in sorted(geoms.items()):
        H,I,E=5120,2304,384
        work.append(dict(M=m,P=pp,n=n,H=H,I=I,E=E,router_classical_ops=2*m*H*E,shared_classical_ops=6*m*H*I,
            routed_classical_ops='70778880 * A_processed_receiver;actual A and clipping not bound for these eager calls',
            full_FLOAT_plane_store_read=8*m*H,chosen_BF16_router_reread=2*m*H,chosen_FLOAT_elision_net_reference=6*m*H,
            Down_BF16_private_interface=4*m*H,Merge_BF16_private_interface=4*m*H,joint_Down_Merge_private_interface_union=8*m*H,
            shared_DQ_private_quant_scale_interface=2*m*(H+4),full4H_Post_plane_content=8*m*H,full4H_output_deletion_credit=0,
            padded_BF16_input_content_delta=2*(pp-m)*H,recorded_P_not_general_ceil16=True,
            source_dispatch_record_stride=H+32,semantic_quantized_row_plus_scale=H+4,
            physical_HBM_bytes_or_extra_allocation_or_saved_time=None))
    write(O/'hc-moe-work-and-lifetimes-v141.json',dict(geometry=work,
      byte_scope='Logical value interfaces in declared versions only. Private-interface credits require all matching readers internal and same content,including final reduction inactive or legally internalized;gross FLOAT deletion vs BF16 reread is an alternative. Do not add overlapping interface cuts or infer HBM/alloc/time.',
      resources='Q_classic/P_upper only within declared necessary dense arithmetic domain and matching physical scope. Whole fused L>=max(valid matrix,vector,link,memory,dependency bounds) with nonduplicated same-resource work;actual Arecv, valid hardware upper, PMU, critical path unknown.',
      lifetimes=['attention raw4H->attentioncoeff and attentionPost','attentionpre4->FFNcollapse','Z1full4H->FFNcoeff and FFNPostresidual','FFNpre4->nextlayerHCcollapse','U_BF16->MC2 plus shared quant;FLOAT->router unless replaced with typed same-value contract','R->source merge/possible final reduction;SBF16/mergedBF16 rounds remain even inside epilogue','FFNPost4H->nextcoeff/residual plus guarded Engram/aux/tail;lastreader AND queued writers before reuse'],
      all_actual_readers_or_quantizers_or_native_fusion_gates_closed=False,new_device_work=False,overall_complete=False))
    names=['hc-sp-eager-native-frontiers-v134.json','hc-moe-native-frontiers-v141.json','hc-moe-merge-correlation-audit-v141.json','router-vector-numerical-contract-v63.json','shared-operator-numerical-and-library-contract-v57.json','shared-expert-fusion-variants-and-interactions-v57.json','shared-MC2-observed-order-contract-v20.json','shared-layout-work-communication-models-v78.json','MC2-kernel-boundary-and-tiling-contract-v56.json','MC2-attribute-to-ceiling-contract-v98.json','MC2-recorded-attributes-and-executions-v98.json','candidate-relation-clauses-v140.json','hc-rope-latent-postcheck-result-v140p.json']
    write(O/'hc-moe-inputs-v141.json',dict(inputs=[pin(O/n)for n in names],questions=[
      'Can HC attention-branch early Y be used directly by router/MC2/shared,or which distinct FFN epoch and pre4 gates are required?',
      'Which native input/rounding/row-mask/receiver contracts constrain HC fork plus router/shared/opaque MC2 composition?',
      'Can full shared overlap and Down/Merge/HCPost interface removal both receive independent credits?',
      'Which next-layer full4H readers or mutations guard FFNPost->HC collapse composition?'],new_device_work=False,overall_complete=False))
    print(json.dumps(dict(status='MODELED',frontiers=960,source_bound_TASKs=prof['source_bound_TASKs'],API_only=len(prof['API_only_frontiers']),DAGs=len(cases),classes=classes,tradeoff=trade,new_pairs=4,overall_complete=False)))


if __name__=='__main__':main()
