"""Receiver capacity, source workspace requests and shared live-memory constraints;offline."""
from pathlib import Path
from collections import Counter
import hashlib,json,math
from finalize_origin_coverage_v9 import N,O,R,W,read,write
from mirror_saved_profile_core_v53 import CACHE
from analyze_route_quant_v95 import BASE

MIB=1024**2

def requested_workspace(M,k,E,RANKS,C,full_load_scratch_elided=False):
    H=5120
    fixed=((M+255)//256)*256*k*4+2*RANKS*RANKS*E*4+RANKS*4*16+(E+RANKS)*4*16
    coc=fixed+C*(3*H+8)
    init=16*MIB+32*M*k+40*8
    return dict(receiver_capacity=C,coc_fixed_bytes=fixed,capacity_coefficient_bytes=3*H+8,coc_workspace_bytes=coc,
      original_init_workspace_bytes=init,init_workspace_charged_bytes=0 if full_load_scratch_elided else init,
      system_workspace_bytes=16*MIB,
      requested_workspace_bytes=16*MIB+max(coc,0 if full_load_scratch_elided else init),
      discarded_generic_init_scratch=full_load_scratch_elided,
      native_allocation_live_or_reserved_bytes_attested=False)

def main():
    assert not(O/'REVIEW-CHECKPOINT-v97.json').exists()
    idx=read(O/'MC2-operator-inputs-v56.json')['operator_source']
    host=Path(next(r['local_path']for r in idx if r['path'].endswith('op_host/dispatch_ffn_combine_tiling.cpp')))
    dispatch=Path(next(r['local_path']for r in idx if r['path'].endswith('dispatch_ffn_combine_kernel.hpp')))
    comm=W/'frozen-source-snapshot/vllm-ascend/vllm_ascend/ops/fused_moe/moe_comm_method.py'
    config=W/'frozen-source-snapshot/vllm-ascend/vllm_ascend/ascend_config.py'
    files={host:[(46,50),(92,112),(258,286),(296,315)],dispatch:[(445,463),(557,575),(1103,1140)],
      comm:[(418,445),(580,616)],config:[(596,604),(951,955)],
      BASE/'moe_init_routing_v2_tiling.h':[(240,248)],
      BASE/'moe_init_routing_quant_v2_tiling.h':[(212,224)],
      BASE/'moe_init_routing_quant_v2.cpp':[(32,65)],BASE/'moe_v2_fullload_dynamic_quant.h':[(313,354)]}
    parts=[]
    for p,ss in files.items():
        ls=p.read_text().splitlines();parts.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),segments=[dict(first_line=a,last_line=b,source=ls[a-1:b])for a,b in ss]))
    assert 'mega_moe_max_tokens: int = 65536'in config.read_text()
    full=(BASE/'moe_v2_fullload_dynamic_quant.h').read_text()
    occurrences=[dict(line=i,text=line.strip())for i,line in enumerate(full.splitlines(),1)if 'workspace'in line]
    assert len(occurrences)==2 and all('GM_ADDR workspace'in r['text']for r in occurrences)
    write(O/'receiver-budget-operator-source-contract-v97.json',dict(parts=parts,generic_full_load_workspace_mentions=occurrences,
      facts=[
        'Customdispatch_ffn_combine forwards config mega_moe_max_tokens unchanged to max_output_size. Adjacent CANNMegaMoe safe/clamped sizing is a different use_mega_moe branch.',
        'Frozen config default is65536;actual current config and loaded max_output_size are not attested by default or old169records.',
        'Host workspace is16MiB+max(coc(C),initRoutingWorkspace),not a sum of both workspaces. coc(C) includes twoFLOATscales+C*5120*FP16+C*5120*INT8;C1/C2 andA/activation aliases remain counted once.',
        'For these5120 cases genericinit request=16MiB+32*M*k+320. Full-load21000 kernel ignores its genericworkspace argument;entry requires nonnull. Explicit inverse-map prefix,peermem/quant output/count buffers remain required.',
        'Eliding genericinit reservation applies only to selected full-load source path with a valid reused nonnull pointer and preserved all explicit args. Large gathered paths and16MiBSYSTEMreservation are not removed.',
        'Original receiver clipping is part of the program semantics. A reduced capacity must preserve min(A,C_old);increasing C to avoid clipping is a different behavior,not an equivalent optimization.'
      ],source_default_not_current_native_value=True,new_target_source_files=0,no_model_Torch_NPU_or_native_work=True,overall_complete=False))
    old_hash=R/'agents/diagnosis/mc2-candidate-capacity-hash.json';old_cfg=R/'diagnostic-20261029/mega-moe-max-tokens-config-audit.json'
    hist=[]
    for p in [old_hash,old_cfg]:hist.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),contents=read(p),scope='Historical169only;notcurrent166config/argumentorloadedbinaryattestation.'))
    current=[]
    for db in range(32):
        p=CACHE/f'analysis/native-execution-identity-v14/db-{db:02d}.json';d=read(p)
        defs={r['globalTaskId']:r for r in d['COMPUTE_definitions']if r['opType_text']=='DispatchFFNCombine'}
        ts=[r for r in d['TASK_records']if r['globalTaskId']in defs]
        assert all(not any(k in r for k in ['max_output_size','maxOutputSize','mega_moe_max_tokens'])for r in defs.values())
        current.append(dict(database_index=db,cache_path=str(p),cache_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),recorded_original_database_sha256=d['database_sha256'],MC2_TASKs=len(ts),MC2_compute_definitions=len(defs),raw_attrInfo_reference_ids=sorted({r['attrInfo']for r in defs.values()}),
          decoded_current_capacity_in_this_export_projection=False,absence_does_not_prove_raw_database_has_no_arguments=True))
    write(O/'receiver-capacity-evidence-status-v97.json',dict(historical=hist,current_export_projection=current,current_MC2_TASKs=sum(r['MC2_TASKs']for r in current),
      actual_current_C=None,actual_same_call_all_rank_n_k_or_receiver_A=None,
      minimum_missing_evidence=['Current customMC2 max_output_size binding with call/run/rank identity.',
        'Same-call allsource row/topK bounds or same-call receiver assignment counts.',
        'Loaded kernel/path and maximum actual writes/allocator/capture alias lifetimes for smaller buffers.'],
      no_native_capacity_value_or_nonclipping_claim=True,overall_complete=False))
    ref=read(O/'route-quant-tiling-and-work-models-v95.json')['cases'];models=[]
    for r in ref:
        M,k=r['M'],r['topK'];E=24 if k==6 else 8;RANKS=16;C0=65536
        B=RANKS*M*min(k,E)
        C_logic=min(C0,max(1,B));C_guard=min(C0,math.ceil((max(1,B)+127)/128)*128)
        full=r['expected_source_key']==21000
        variants=[]
        for ctag,C in [('original_default_reference',C0),('logical_bound_reference',C_logic),('bound_plus_128row_guard_reference',C_guard)]:
            variants.append(dict(id=ctag,**requested_workspace(M,k,E,RANKS,C)))
            if full:variants.append(dict(id=ctag+'_elide_unused_generic_init',**requested_workspace(M,k,E,RANKS,C,True)))
        models.append(dict(M=M,k=k,local_experts=E,EP_reference=RANKS,assumed_same_call_all_sources_n_le_M=True,
          receiver_assignment_geometry_upper=B,original_default_capacity_reference=C0,
          geometry_alone_proves_no_clip_at_default_reference=B<=C0,
          C_logic=C_logic,C_guard_reference=C_guard,full_load_source_case=full,saved_executions=r['saved_executions'],variants=variants,
          capacity_transform='C_new=min(C_old,max(1,B)) or any C_new in[min(C_old,B),C_old]. For allA<=B,min(A,C_new)=min(A,C_old).',
          guard_is_not_native_write_extent_proof=True,physical_workspace_shrink_or_current_C_proved=False))
    write(O/'receiver-capacity-and-workspace-models-v97.json',dict(cases=models,
      geometry_bound='A_receiver <=sum_sources n_s*k_s. With distinct chosen physicalexperts pertoken,A_receiver <=sum_sources n_s*min(k_s,E_receiver). Bounds are sufficient geometry limits,not a model-checkpoint reachability claim.',
      reference_ranks_not_same_call_join=True,capacity_reduction_preserves_original_clipping_not_changes_it=True,
      no_GEMM_FLOP_or_weight_traffic_reduction=True,no_workspace_request_to_HBM_traffic_conversion=True,no_layer_or_iteration_request_sum_as_peak=True,overall_complete=False))
    target=next(r for r in models if(r['M'],r['k'])==(3,6));table={r['id']:r for r in target['variants']}
    old=table['original_default_reference']['requested_workspace_bytes'];cap=table['bound_plus_128row_guard_reference']['requested_workspace_bytes'];scratch=table['original_default_reference_elide_unused_generic_init']['requested_workspace_bytes'];both=table['bound_plus_128row_guard_reference_elide_unused_generic_init']['requested_workspace_bytes']
    large=next(r for r in models if r['M']==816);prefill=next(r for r in large['variants']if r['id']=='original_default_reference')['requested_workspace_bytes']
    wkv=read(O/'Engram-WKV-storage-and-work-models-v84.json');ring=read(O/'ring-storage-capacity-models-v79.json')
    D_cache=64*(wkv['result_bytes_per_row']+wkv['full_ID_key_bytes_per_row']);D_ring=ring['ring32_bytes_per_owner_all_three_layers'];extra=D_cache+D_ring
    B_mem=prefill+MIB
    joint=dict(source_default_reference_only=True,
      max_workspace_synergy=dict(old=old,capacity_only=cap,unused_init_only=scratch,both=both,
        old_minus_capacity=old-cap,old_minus_init=old-scratch,old_minus_both=old-both,
        extra_saving_from_joint=cap-both,actual_allocator_peak_saving=None),
      persistent_bank_value_reference=dict(WKV_entries=64,WKV_result_and_ID_value_bytes_per_entry=wkv['result_bytes_per_row']+wkv['full_ID_key_bytes_per_row'],WKV_value_bytes=D_cache,
        C2_private_one_owner_all3layers_payload=D_ring,total=extra,metadata_alignment_concurrency_and_invalid_slots_excluded=True,
        genuinely_incremental_banks_with_no_old_pool_retirement_credit_assumed=True),
      synthetic_liveness_counterexample=dict(budget_remainder_before_named_banks=B_mem,
        old_decode_workspace=old,new_decode_workspace=both,unchanged_prefill_workspace=prefill,
        old_phase_peak=max(old,prefill),new_phase_peak=max(both,prefill),peak_credit=0,
        added_persistent_banks=extra,new_peak_with_banks=max(both,prefill)+extra,
        feasible_only_decode=both+extra<=B_mem,feasible_across_both_phases=max(both,prefill)+extra<=B_mem,
        actual_device_memory_snapshot=False),
      common_budget_cut='max_t[W_live(t)+D_WKV_live(t)+D_C2_live(t)+other_live(t)]<=B_device_or_allocator. B_remainder must be beforethese named allocations;do notsubtractworkspace twicefromanexistingKVbudget.',
      pool_capacity_if_joint_rebudgeted='N_globalIDs<=floor((B_rebudget_before_pool_workspace_newbanks-max_t[W_live+D_WKV+D_C2+protocol])/540928) in priorv79normaltupledomain. This rebudget remainder excludes the oldpool,unlike a live headroom number after an alreadyallocatedpool;retire/rebudget/publicationproof required.',
      observed_memory_credit=None,whole_model_feasibility_or_gain_proved=False,overall_complete=False)
    write(O/'receiver-workspace-shared-live-budget-v97.json',joint)
    clauses=[]
    for i,b in enumerate(['F25','F34'],1):
        clauses.append(dict(id=f'REL97-{i:02d}',a='F21',b=b,relation='RECEIVER_WORKSPACE_AND_PERSISTENT_BANK_SHARE_LIVE_MEMORY_BUDGET',
          operator_claim='One current-model/device allocation budget couples selected MC2capacity/phase scratch with pureWKVcache or privateC2banks. Per-decode requested-space reduction is not automatically peak headroom;reuse keys/generations/aliases and preserved operator work remain.',
          evidence=['receiver-budget-operator-source-contract-v97.json','receiver-capacity-and-workspace-models-v97.json','receiver-workspace-shared-live-budget-v97.json','tail-shared-memory-budget-v85.json'],
          applicable_domain='Co-resident named buffers onthe same physicaldevice/runtime pool and overlapping lifetimes;not across unrelated ranks/captures. Source capacity/scratch and state/cache representations are conditional.',
          conditional_compatibility=True,all_runtime_gates_proved=False,speedup_additive=False,critical_path_gain_us=None))
    write(O/'receiver-budget-composition-clauses-v97.json',dict(clauses=clauses,
      extend_group='HG85-SHARED-MEMORY-BUDGET',new_member='F21',new_groups=0,new_maincandidate=False,overall_complete=False))
    write(O/'receiver-budget-variants-v97.json',dict(variants=[
      dict(id='RB97-SEMANTIC-CAP',decision='SCOPED_CAP_EQUIVALENCE_WITH_GLOBAL_BOUND',contract='Derive same-call all-rank geometryB,then chooseC_new<=C_old but>=min(C_old,B). Preserves originalclipping;actualC andglobalboundmustbind. Cguardsizingnot a proof ofnativewriteextents.'),
      dict(id='RB97-FULL-LOAD-SCRATCH',decision='CONDITIONAL_SOURCE_PATH_RESERVATION_ELISION',contract='Forfull-load21000 retainnon-nullgenericpointer andall explicit map/peer/countargs;unusedgenericinitreservationcanbe omittedin anewworkspacecontract. SYSTEM16MiB andlargegatheredpathsremain.'),
      dict(id='RB97-COUNT-FIRST-BUCKET',decision='CONDITIONAL_SPLIT_OR_NEW_BANK_ABI',contract='ExactreceiverA_r isavailable onlyaftercounts/exchange. Size/bucketworkspace thenrequires exposingthatdependency ornewpreallocatedbankselection/captureABI;do not usefuturecountsin theunchangedprelaunchallocatorforfree.'),
      dict(id='RB97-PERSISTENT-REINVEST',decision='CONDITIONAL_COMMON_PEAK_BUDGET',contract='FundWKVcache/C2privatebanks onlyfromactualco-residentpeakheadroom afterprefill/decode/liveness/allocator/captureproof. LocalspacecreditdoesnotdeleteGEMMworkorcountasHBMtime.')],
      implementation=False,native_test=False,overall_complete=False))
    print(json.dumps(dict(current_MC2_TASKs=sum(r['MC2_TASKs']for r in current),source_cases=len(models),workspace_variants=sum(len(r['variants'])for r in models),
      M3_default_workspace=old,M3_guard_capacity_only=cap,M3_guard_plus_unused_init=both,source_reference_not_actual_allocation=True,new_relations=len(clauses),overall_complete=False)))

if __name__=='__main__':main()
